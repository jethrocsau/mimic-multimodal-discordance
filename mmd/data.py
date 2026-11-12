"""The phase0 tables, the slot tables (which record fills each modality's first / last slot), the raw files
behind each record, and the radiology notes.

Tables come from the frozen parquet snapshot in `{root}/<cohort_version>/data/` (written by `mmd data stage`
from BigQuery, or copied there from Drive). Raw files (CXR JPG, ECG WFDB) are resolved as
`{raw}/<physionet_project>/<version>/<relative_path>`, which matches the Drive `unpacked/` layout, a GCS mirror
of it, or a PhysioNet bucket. Note text is read at run time and never written to the store.

Slot rules per modality (the notebooks' rules, so existing runs keep their records):
    x  CXR image   latest frontal baseline film (lateral only when no frontal one); optional pins
    e  ECG         earliest baseline record
    n  note        earliest baseline radiology report (optionally chest-radiograph reports only)
    v  vitals      the window: "<stay_id>:baseline" (first) / "<stay_id>:follow_up" (last)
    l  labs        a lab entry (every panel lab at one charttime): earliest baseline / latest follow-up entry,
                   preferring entries with at least MIN_PANEL_LABS labs

Every modality is loaded raw, in one form per kind: a PIL image (x), a Bundle time series (e, v: physical units,
native rate, NaN where not measured), a LabPanel table (l). Models never get pipeline-processed data: an adapter
asks prepare() for the rate / length / gap filling / scaling it needs (Adapter.series_spec), and presents the
result as it likes (tokens, text, an image), so every model reads the same stored data.
"""
import hashlib
import os
import re
import tempfile
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import fsio
from .modsets import MOD_NAMES, NOTE, ModalitySet

# Every x_* table already holds the union of all levels (download/bq_export.py)
TABLES = ["cohort", "x_stays", "stay_summary", "analysis_dataset",
          "x_vitals", "x_labs", "x_notes", "x_ecg", "x_cxr", "x_file_manifest"]

# Vitals: x_vitals.label -> (channel, unit, plausible range). Values outside the range are charting errors and
# count as not measured.
VITALS = {
    'Heart Rate': ('heart_rate', 'bpm', (1, 300)),
    'Temperature Fahrenheit': ('temperature_f', 'degF', (77, 113)),
    'Non Invasive Blood Pressure systolic': ('nibp_systolic', 'mmHg', (20, 300)),
    'Non Invasive Blood Pressure diastolic': ('nibp_diastolic', 'mmHg', (5, 250)),
    'Non Invasive Blood Pressure mean': ('nibp_mean', 'mmHg', (10, 300)),
}
# Labs: the 22-lab panel of modality_eda/labs/labs_eda.ipynb (hosp.labevents itemid -> short name, verified there)
LAB_PANEL = {
    50983: 'sodium', 50971: 'potassium', 50902: 'chloride', 50882: 'bicarbonate', 50868: 'anion_gap',
    51006: 'bun', 50912: 'creatinine', 50931: 'glucose', 50893: 'calcium', 50960: 'magnesium',
    50970: 'phosphate', 51222: 'hemoglobin', 51301: 'wbc', 51265: 'platelets',
    51003: 'troponin_t', 50963: 'ntprobnp', 50862: 'albumin', 50885: 'bilirubin',
    50861: 'alt', 50878: 'ast', 51237: 'inr', 50813: 'lactate',
}
MIN_PANEL_LABS = 5          # a lab entry with at least this many panel labs wins its slot over a sparser draw


def stage(cfg, tables=TABLES, verbose=True):
    """BigQuery phase0 -> parquet snapshot for this cohort version (overwrites only that version's files)."""
    from google.cloud import bigquery
    client = bigquery.Client(project=cfg.project, location=cfg.location)
    for t in tables:
        df = client.query(f'SELECT * FROM {cfg.table(t)}').to_dataframe()
        fsio.write_parquet(df, f'{cfg.data_dir()}/{t}.parquet')
        if verbose:
            print(f'{t}: {len(df):,} rows -> {cfg.data_dir()}/{t}.parquet')


class Tables:
    def __init__(self, cfg):
        self.cfg = cfg

    @lru_cache(maxsize=None)
    def get(self, name) -> pd.DataFrame:
        return fsio.read_parquet(f'{self.cfg.data_dir()}/{name}.parquet')

    def file_manifest(self) -> pd.DataFrame:
        m = self.get('x_file_manifest')
        m = m[m.relative_path.notna()].copy()
        m['local_path'] = (self.cfg.raw.rstrip('/') + '/' + m.physionet_project + '/' + m.physionet_version
                           + '/' + m.relative_path)
        return m

    def _read(self, name, columns, filters) -> pd.DataFrame:
        """Some columns and rows of a staged table (x_vitals / x_labs are too large to read whole)."""
        with fsio.open_(f'{self.cfg.data_dir()}/{name}.parquet') as f:
            return pd.read_parquet(f, columns=columns, filters=filters)

    @lru_cache(maxsize=None)
    def vitals(self) -> pd.DataFrame:
        """x_vitals rows of the VITALS channels whose value is plausible: stay_id, phase, hours_from_icu_admit,
        channel (index into VITALS), valuenum."""
        v = self._read('x_vitals', ['stay_id', 'phase', 'hours_from_icu_admit', 'label', 'valuenum'],
                       [('label', 'in', list(VITALS))])
        lo = v.label.map({k: r[2][0] for k, r in VITALS.items()})
        hi = v.label.map({k: r[2][1] for k, r in VITALS.items()})
        v = v[v.valuenum.between(lo, hi)].copy()
        v['channel'] = v.label.map({k: i for i, k in enumerate(VITALS)}).astype(int)
        v['stay_id'] = v.stay_id.astype('int64')
        return v.drop(columns='label').reset_index(drop=True)

    @lru_cache(maxsize=None)
    def labs(self) -> pd.DataFrame:
        """x_labs rows of the LAB_PANEL labs, one per (stay, charttime, lab) (the lowest labevent_id), with the
        lab-entry key `entry` = "<stay_id>:<charttime ISO>" (one entry = every panel lab drawn at that time)."""
        l = self._read('x_labs', ['stay_id', 'phase', 'labevent_id', 'charttime', 'hours_from_icu_admit', 'itemid',
                                  'value', 'valuenum', 'valueuom', 'ref_range_lower', 'ref_range_upper', 'flag'],
                       [('itemid', 'in', list(LAB_PANEL))])
        l['stay_id'] = l.stay_id.astype('int64')
        l['entry'] = l.stay_id.astype(str) + ':' + pd.to_datetime(l.charttime).dt.strftime('%Y-%m-%dT%H:%M:%S')
        l = l.sort_values('labevent_id', kind='mergesort').drop_duplicates(['entry', 'itemid'])
        return l.reset_index(drop=True)


def available(paths, root) -> set:
    """Which of `paths` exist, with one listing of the root instead of one request per file."""
    paths = list(dict.fromkeys(paths))
    if not fsio.is_remote(root):
        import os
        return {p for p in paths if os.path.exists(p)}
    fs = fsio.fs_of(root)
    proto = root.split('://', 1)[0]
    prefixes = {p.rsplit('/', 1)[0] for p in paths}
    tops = {'/'.join(p.split('/')[:6]) for p in prefixes}         # gs://bucket/physionet/<project>/<version>
    found = set()
    for t in tops:
        found |= {f'{proto}://{x}' for x in fs.find(t)}
    return {p for p in paths if p in found}


# ── slot tables: which record fills each modality's first / last slot ─────────
def pick_slots(rows: pd.DataFrame, record_col: str, first: str = 'earliest',
               prefer: str = None) -> pd.DataFrame:
    """One record per stay and slot. rows: stay_id, phase, hours_from_icu_admit, record_col
    (+ an optional boolean `prefer` column, e.g. frontal view, which wins inside a window).

    first = the baseline-window record, earliest or latest (`first`);
    last  = the latest record in the follow-up window [w_end, f_end).
    """
    r = rows.dropna(subset=[record_col]).copy()
    r['_p'] = r[prefer].astype(bool) if prefer else True
    # ties (two films of one study, two notes at one charttime) break on the record id, so every
    # session resolves a slot to the same record whatever order the table was read in
    by = ['stay_id', '_p', 'hours_from_icu_admit', record_col]
    b = r[r.phase == 'baseline'].sort_values(by, ascending=[True, False, first == 'earliest', True],
                                             kind='mergesort')
    f = r[r.phase == 'follow_up'].sort_values(by, ascending=[True, False, False, True], kind='mergesort')
    out = pd.DataFrame({'first': b.drop_duplicates('stay_id').set_index('stay_id')[record_col],
                        'last': f.drop_duplicates('stay_id').set_index('stay_id')[record_col]})
    out.index = out.index.astype('int64')
    return out


def apply_pins(slot_table: pd.DataFrame, pins: pd.DataFrame, record_col: str, valid=None) -> int:
    """Pin slots to the records the stored runs actually used (slot_pins_x.csv from the S3 repair)."""
    if valid is not None:
        pins = pins[pins[record_col].isin(valid)]
    for r in pins.itertuples():
        slot_table.loc[r.stay_id, r.slot] = getattr(r, record_col)
    return len(pins)


def cxr_rows(tables: Tables, verbose=True):
    """One row per (stay, dicom_id) with a JPG that exists; and dicom_id -> path."""
    m = tables.file_manifest()
    rows = (m[m.file_role == 'cxr_image_jpg'].rename(columns={'record_id': 'dicom_id'})
            [['stay_id', 'phase', 'hours_from_icu_admit', 'dicom_id', 'study_id', 'view_position', 'local_path']]
            .drop_duplicates(['stay_id', 'dicom_id']))
    rows['frontal'] = rows.view_position.isin(['AP', 'PA'])
    paths = rows.drop_duplicates('dicom_id')
    ok = available(paths.local_path, tables.cfg.raw)
    on = paths.local_path.isin(ok)
    if verbose:
        print(f"{len(paths):,} CXR images in the manifest, {on.sum():,} found under {tables.cfg.raw} ({on.mean():.1%})")
    rows = rows[rows.dicom_id.isin(paths.dicom_id[on])]
    return rows, dict(zip(rows.dicom_id, rows.local_path))


def ecg_rows(tables: Tables):
    """One row per (stay, ECG record); record = the header path without `.hea`, as in the OpenTSLM notebook
    (no file-existence filter there, so none here: the cohort stays the notebook's)."""
    m = tables.file_manifest()
    return (m[m.file_role == 'ecg_header'].assign(record=lambda d: d.local_path.str[:-4])
            [['stay_id', 'phase', 'hours_from_icu_admit', 'record']])


def note_rows(tables: Tables, chest_only=False):
    notes = tables.get('x_notes')
    if chest_only:
        notes = notes[notes.is_chest_radiograph_report]
    return notes[['stay_id', 'phase', 'hours_from_icu_admit', 'note_id']]


def slots_x(tables: Tables, pins: pd.DataFrame = None, verbose=True, **_):
    rows, image_path = cxr_rows(tables, verbose)
    t = pick_slots(rows, 'dicom_id', first='latest', prefer='frontal')
    if pins is not None:
        n = apply_pins(t, pins, 'dicom_id', valid=set(image_path))
        if verbose:
            print(f'{n:,} image slots pinned')
    return t


def slots_e(tables: Tables, verbose=True, **_):
    rows = ecg_rows(tables)
    if verbose:
        print(f"{rows.record.nunique():,} ECG records in the manifest")
    return pick_slots(rows, 'record', first='earliest')


def slots_n(tables: Tables, note_chest_only=False, **_):
    return pick_slots(note_rows(tables, note_chest_only), 'note_id', first='earliest')


def slots_v(tables: Tables, verbose=True, **_):
    """One record per window: "<stay_id>:baseline" (the modality window) and "<stay_id>:follow_up"."""
    v = tables.vitals()
    rows = v.groupby(['stay_id', 'phase'], as_index=False).hours_from_icu_admit.min()
    rows['record'] = rows.stay_id.astype(str) + ':' + rows.phase
    if verbose:
        print(f"{rows.stay_id.nunique():,} stays with vitals ({(rows.phase == 'follow_up').sum():,} with follow-up)")
    return pick_slots(rows, 'record', first='earliest')


def slots_l(tables: Tables, verbose=True, **_):
    """One record per lab entry (every panel lab at one charttime): the earliest baseline entry and the latest
    follow-up one, an entry with at least MIN_PANEL_LABS labs winning over a sparser draw inside the window."""
    rows = (tables.labs().groupby('entry', as_index=False)
            .agg(stay_id=('stay_id', 'first'), phase=('phase', 'first'),
                 hours_from_icu_admit=('hours_from_icu_admit', 'first'), n_labs=('itemid', 'size')))
    rows['full'] = rows.n_labs >= MIN_PANEL_LABS
    if verbose:
        print(f"{len(rows):,} lab entries over {rows.stay_id.nunique():,} stays "
              f"({rows.full.mean():.0%} with >= {MIN_PANEL_LABS} panel labs)")
    return pick_slots(rows, 'entry', first='earliest', prefer='full')


SLOT_BUILDERS = {'x': slots_x, 'e': slots_e, 'n': slots_n, 'v': slots_v, 'l': slots_l}


def slots_for(modset: ModalitySet, tables: Tables, note_chest_only=False, pins: pd.DataFrame = None,
              verbose=True) -> Dict[str, pd.DataFrame]:
    """The slot table of every modality in the set: {mod: DataFrame(index stay_id; first, last)}."""
    missing = [m for m in modset.mods if m not in SLOT_BUILDERS]
    if missing:
        raise NotImplementedError(f'{modset.name}: no slot table for {[MOD_NAMES[m] for m in missing]} yet '
                                  f'(add a builder to data.SLOT_BUILDERS)')
    if pins is not None and 'x' not in modset.mods:
        raise ValueError('--pins pins CXR slots; this set has no CXR')
    return {m: SLOT_BUILDERS[m](tables, note_chest_only=note_chest_only, pins=pins, verbose=verbose)
            for m in modset.mods}


def signal_loader(tables: Tables, mod: str):
    """(load, path) for one signal modality: slot value -> signal object, and -> file path."""
    if mod == 'x':
        _, image_path = cxr_rows(tables)
        images = ImageLoader(image_path)
        return images.load, images.path
    if mod == 'e':
        return EcgLoader().load, (lambda record: record)
    if mod == 'v':
        return VitalsLoader(tables).load, (lambda key: key)
    if mod == 'l':
        return LabsLoader(tables).load, (lambda key: key)
    raise NotImplementedError(f'no loader for {MOD_NAMES.get(mod, mod)} yet')


def signal_loaders(tables: Tables, mods) -> Dict[str, tuple]:
    """{mod: (load, path)} for every signal (non-note) modality of a set."""
    return {m: signal_loader(tables, m) for m in mods if m != NOTE}


class NotesLoader:
    """Report text by note_id, trimmed to the sections that describe the image.

    MIMIC radiology reports open with EXAMINATION / INDICATION / HISTORY, which often state the clinical
    question and can carry prognosis. Keeping only FINDINGS and IMPRESSION removes the most obvious route for
    the outcome to leak in through the referrer's wording.
    """
    KEEP = ("FINDINGS", "IMPRESSION", "CONCLUSION")
    HEAD = re.compile(r"^\s*([A-Z][A-Z /&'\-]{2,40}):", re.M)

    def __init__(self, notes: pd.DataFrame, max_chars: int = 1200, keep_sections: bool = True):
        self.max_chars, self.keep_sections = max_chars, keep_sections
        n = notes.drop_duplicates("note_id")
        self.raw = dict(zip(n.note_id, n.text.astype(str)))

    def sections(self, text: str) -> Dict[str, str]:
        heads = list(self.HEAD.finditer(text))
        out = {}
        for i, h in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
            out[h.group(1).strip()] = text[h.end():end].strip()
        return out

    def text(self, note_id) -> str:
        raw = self.raw.get(note_id)
        if raw is None:
            raise KeyError(f"note {note_id} not found")
        if self.keep_sections:
            sec = self.sections(raw)
            kept = [f"{k}: {v}" for k, v in sec.items() if any(k.startswith(w) for w in self.KEEP)]
            raw = "\n".join(kept) if kept else raw     # unparseable: fall back to the whole note
        raw = " ".join(raw.split())
        return raw[:self.max_chars]


class LeakageAudit:
    """How often a note gives the outcome away. Run it before trusting any result that uses the last note."""
    PATTERNS = {
        "terminal care": r"\b(hospice|comfort (care|measures)|palliat\w*|withdraw\w* of care|"
                         r"end[- ]of[- ]life|DNR|do not resuscitate|goals of care)\b",
        "arrest / resus": r"\b(cardiac arrest|CPR|resuscitat\w*|code blue|ROSC|post[- ]arrest)\b",
        "life support": r"\b(intubat\w*|endotracheal|ETT\b|mechanical ventilat\w*|ECMO|"
                        r"tracheostomy|vasopressor\w*)\b",
        "deceased": r"\b(expired|deceased|post[- ]?mortem|death|died)\b",
    }

    def __init__(self, loader: NotesLoader):
        self.loader = loader
        self.rx = {k: re.compile(v, re.I) for k, v in self.PATTERNS.items()}

    def scan(self, note_ids, labels=None) -> pd.DataFrame:
        rows = []
        for i, nid in enumerate(note_ids):
            try:
                t = self.loader.text(nid)
            except KeyError:
                continue
            r = {k: bool(rx.search(t)) for k, rx in self.rx.items()}
            r["note_id"] = nid
            if labels is not None:
                r["died_1y"] = bool(labels[i])
            rows.append(r)
        return pd.DataFrame(rows)

    def report(self, note_ids, labels=None) -> pd.DataFrame:
        d = self.scan(note_ids, labels)
        if d.empty:
            return d
        cols = list(self.PATTERNS)
        out = pd.DataFrame({"mentions %": (100 * d[cols].mean()).round(1)})
        if labels is not None:
            out["% of these who died"] = [round(100 * d.loc[d[c], "died_1y"].mean(), 1) if d[c].any() else np.nan
                                          for c in cols]
            out["base rate % died"] = round(100 * d.died_1y.mean(), 1)
        return out


class ImageLoader:
    """dicom_id -> PIL image, from a local path or gs://. Recent images are kept in memory: a film is reused by
    many records (the recipient's own arms, and as a donor), so most reads after the first are cache hits."""

    def __init__(self, paths: dict, cache: int = 64):
        self.paths = paths
        self.load = lru_cache(maxsize=cache)(self._load)

    def path(self, dicom_id):
        p = self.paths.get(dicom_id)
        if p is None:
            raise KeyError(f'no image file for {dicom_id}')
        return p

    def _load(self, dicom_id):
        from PIL import Image
        with fsio.open_(self.path(dicom_id)) as f:
            return Image.open(f).convert('RGB')


@dataclass
class Bundle:
    """A multichannel time series: the one form every series modality (ECG, vitals) is loaded in.

    Loaders return it raw: physical units at the native rate, NaN where nothing was measured. A model never gets
    it processed by the pipeline; its adapter asks for the rate, length, gap filling and scaling it needs with
    prepare() (Adapter.series_spec), so every model reads the same stored data."""
    values: np.ndarray                # (K, L): raw units, or z-scored after prepare(znorm=True); NaN = not measured
    names: List[str]
    units: Optional[List[str]] = None
    fs: Optional[float] = None        # samples per second (ECG 500, hourly vitals 1/3600)
    t: Optional[np.ndarray] = None    # (L,) hours from ICU admission of each sample (windowed series), else None
    mean: Optional[np.ndarray] = None  # (K,) raw mean / std per channel, set by prepare(znorm=True)
    std: Optional[np.ndarray] = None
    info: Dict = field(default_factory=dict)


def fill_nan(sig) -> int:
    """Interpolate NaN gaps in place, per column of sig (L, K); an all-NaN column becomes 0. Returns the NaN count."""
    n_nan = int(np.isnan(sig).sum())
    for j in range(sig.shape[1]):
        x, bad = sig[:, j], np.isnan(sig[:, j])
        if bad.all():
            x[:] = 0.0
        elif bad.any():
            x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad])
    return n_nan


def zscore(sig):
    """Per-column z-score of sig (L, K) in float64 -> (norm (K, L) float32, mean (K,), std (K,)). NaN-aware when
    gaps were not filled (an all-NaN channel keeps NaN mean / std)."""
    if np.isnan(sig).any():
        with np.errstate(invalid='ignore', divide='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mean, std = np.nanmean(sig, 0), np.nanstd(sig, 0)
    else:
        mean, std = sig.mean(0), sig.std(0)
    norm = (sig - mean) / np.where(std > 1e-6, std, 1.0)
    return norm.T.astype(np.float32), mean, std


def _resample(sig, t, fs, hz, how):
    """sig (L, K) at fs -> at hz. 'stride' keeps every round(fs/hz)-th sample, 'mean' averages each bin of that
    many samples (NaN-aware), 'linear' interpolates onto the new grid (any ratio; NaN next to a gap stays NaN)."""
    if how in ('stride', 'mean'):
        step = int(round(fs / hz))
        if step < 1 or not np.isclose(fs / step, hz):
            raise ValueError(f"how='{how}' needs fs / hz to be a whole number (fs={fs}, hz={hz}); use how='linear'")
        if how == 'stride':
            return sig[::step], (t[::step] if t is not None else None)
        n = -(-len(sig) // step)
        pad = np.full((n * step - len(sig), sig.shape[1]), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)          # all-NaN bins stay NaN
            out = np.nanmean(np.vstack([sig, pad]).reshape(n, step, -1), 1)
        return out, (t[::step] if t is not None else None)
    if how == 'linear':
        old = np.arange(len(sig)) / fs
        new = np.arange(int(np.floor(old[-1] * hz + 1e-9)) + 1) / hz
        out = np.stack([np.interp(new, old, sig[:, j]) for j in range(sig.shape[1])], 1)
        return out, (t[0] + new / 3600.0 if t is not None else None)
    raise ValueError(f"unknown resampling {how!r}: 'stride' | 'mean' | 'linear'")


def prepare(b: Bundle, hz: float = None, length: int = None, how: str = 'stride', fill: bool = False,
            znorm: bool = False) -> Bundle:
    """A stored Bundle as one model takes it; the stored one is never changed. In order:
        hz      resample to this rate (`how`: 'stride' | 'mean' | 'linear'; see _resample)
        length  cut to this many samples, or edge-pad up to it
        fill    interpolate NaN gaps (info['n_nan'] = how many there were)
        znorm   z-score each channel; values become float32, mean / std keep the raw scale
    prepare(ecg, hz=100, length=1000, fill=True, znorm=True) is exactly what the ECG loader did before it returned
    raw data (OpenTSLM's input), down to the last bit."""
    sig, t, fs = b.values.T, b.t, b.fs                        # (L, K): the layout the operations run in
    if hz is not None and fs is None:
        raise ValueError("prepare(hz=...) needs the bundle's own rate (fs)")
    if hz is not None and not np.isclose(hz, fs):
        sig, t = _resample(sig, t, fs, hz, how)
        fs = float(hz)
    if length is not None:
        sig, t = sig[:length], (t[:length] if t is not None else None)
        if len(sig) < length:
            k = length - len(sig)
            sig = np.pad(sig, ((0, k), (0, 0)), mode='edge')
            if t is not None:
                t = np.concatenate([t, t[-1] + np.arange(1, k + 1) / (fs * 3600.0)])
    sig = np.array(sig, dtype=np.float64)                     # own copy: the stored bundle stays as loaded
    info = dict(b.info)
    if fill:
        info['n_nan'] = fill_nan(sig)
    mean = std = None
    if znorm:
        values, mean, std = zscore(sig)
    else:
        values = sig.T
    return Bundle(values, list(b.names), b.units, fs, t, mean, std, info)


class EcgLoader:
    """ECG record (header path without extension, local or gs://) -> raw Bundle: the 12 leads matched by name, in
    physical units (mV) at the recording's own rate (500 Hz), NaN gaps kept. The model's own rate, length and
    scaling come from prepare() (OpenTSLM: 100 Hz, 1000 samples, gaps filled, z-scored, as it trained on PTB-XL).
    Recent records are kept in memory (one ECG serves many records). wfdb reads local files only, so gs:// records
    are copied (.hea + .dat) to a local cache first."""
    LEADS = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
    cache_dir = os.path.join(tempfile.gettempdir(), 'mmd_ecg')

    def __init__(self, leads=LEADS, cache: int = 64):
        self.leads = tuple(leads)
        self.load = lru_cache(maxsize=cache)(self._load)

    def local(self, record):
        if not fsio.is_remote(record):
            return record
        d = os.path.join(self.cache_dir, hashlib.md5(record.encode()).hexdigest()[:16])
        dst = os.path.join(d, os.path.basename(record))
        if not os.path.exists(dst + '.hea'):
            os.makedirs(d, exist_ok=True)
            for ext in ('.dat', '.hea'):              # header last: its presence marks a complete copy
                with fsio.open_(record + ext) as f, open(dst + ext + '.part', 'wb') as g:
                    g.write(f.read())
                os.replace(dst + ext + '.part', dst + ext)
        return dst

    def _load(self, record) -> Bundle:
        import wfdb
        rec = wfdb.rdrecord(self.local(record))
        pos = {n.lower(): i for i, n in enumerate(rec.sig_name)}
        idx = [pos[l.lower()] for l in self.leads]
        sig = rec.p_signal[:, idx].astype(np.float64)         # (L, K), kept in this layout (values is its .T view)
        sig.setflags(write=False)                             # cached and shared: prepare() works on a copy
        return Bundle(sig.T, list(self.leads), [rec.units[i] for i in idx], float(rec.fs))


class VitalsLoader:
    """"<stay_id>:<phase>" -> raw Bundle of the VITALS channels on an hourly grid (fs = 1/3600), NaN in every hour
    with nothing charted (no filling: missingness is the adapter's to handle, through prepare() or its own view).
    Each hour holds the mean of its charted values. Both windows are the same length, so first and last swap:
        baseline   the modality window [w_start, w_end), hours from ICU admission
        follow_up  the window of that length ending at the stay's last charted follow-up vital (its hours before
                   w_end stay NaN: only follow-up records fill it)"""
    names = [c for c, _, _ in VITALS.values()]
    units = [u for _, u, _ in VITALS.values()]

    def __init__(self, tables: Tables):
        v = tables.vitals()
        self.h = v.hours_from_icu_admit.to_numpy(float)
        self.ch = v.channel.to_numpy(int)
        self.val = v.valuenum.to_numpy(float)
        self.rows = v.groupby(['stay_id', 'phase']).indices
        st = tables.get('x_stays')
        hours = lambda c: (pd.to_datetime(st[c]) - pd.to_datetime(st.intime)).dt.total_seconds().to_numpy() / 3600
        self.window = dict(zip(st.stay_id.astype('int64'), zip(hours('w_start'), hours('w_end'))))

    def load(self, key) -> Bundle:
        sid, phase = key.rsplit(':', 1)
        sid = int(sid)
        start, end = self.window[sid]
        L = int(round(end - start))
        pos = self.rows.get((sid, phase), np.array([], dtype=int))
        h, ch, val = self.h[pos], self.ch[pos], self.val[pos]
        if phase == 'follow_up':
            start = (np.floor(h.max()) + 1 if len(h) else end + L) - L
        b = np.floor(h - start).astype(int)
        ok = (b >= 0) & (b < L)
        total, n = np.zeros((len(VITALS), L)), np.zeros((len(VITALS), L))
        np.add.at(total, (ch[ok], b[ok]), val[ok])
        np.add.at(n, (ch[ok], b[ok]), 1)
        with np.errstate(invalid='ignore'):
            values = total / n                                # 0 / 0 = NaN: nothing charted that hour
        return Bundle(values, list(self.names), list(self.units), 1 / 3600, start + np.arange(L, dtype=float),
                      info=dict(stay_id=sid, phase=phase, n_charted=int(ok.sum())))


@dataclass
class LabPanel:
    """One lab entry: every LAB_PANEL lab drawn at one charttime, as a table in LAB_PANEL order. Labs not drawn
    then are NaN rows (missingness is the adapter's to handle)."""
    table: pd.DataFrame               # index lab; value (numeric), text (as charted), unit, ref_low, ref_high, flag
    hours: float                      # hours from ICU admission
    info: Dict = field(default_factory=dict)


class LabsLoader:
    """"<stay_id>:<charttime ISO>" -> LabPanel."""
    COLUMNS = {'valuenum': 'value', 'value': 'text', 'valueuom': 'unit', 'ref_range_lower': 'ref_low',
               'ref_range_upper': 'ref_high', 'flag': 'flag'}

    def __init__(self, tables: Tables):
        self.labs = tables.labs()
        self.rows = self.labs.groupby('entry').indices
        self.order = list(LAB_PANEL.values())

    def load(self, entry) -> LabPanel:
        r = self.labs.iloc[self.rows[entry]]
        t = (r.assign(lab=r.itemid.map(LAB_PANEL)).set_index('lab')[list(self.COLUMNS)]
             .rename(columns=self.COLUMNS).reindex(self.order))
        t.index.name = 'lab'
        return LabPanel(t, float(r.hours_from_icu_admit.iloc[0]),
                        dict(entry=entry, phase=r.phase.iloc[0], n_labs=int(len(r))))
