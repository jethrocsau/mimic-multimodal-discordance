"""The phase0 tables, the slot tables (which record fills each modality's first / last slot), the raw files
behind each record, and the radiology notes.

Tables come from the frozen parquet snapshot in `{root}/data/<cohort_version>/` (written by `mmd data stage`
from BigQuery, or copied there from Drive). Raw files (CXR JPG, ECG WFDB) are resolved as
`{raw}/<physionet_project>/<version>/<relative_path>`, which matches the Drive `unpacked/` layout, a GCS mirror
of it, or a PhysioNet bucket. Note text is read at run time and never written to the store.

Slot rules per modality (the notebooks' rules, so existing runs keep their records):
    x  CXR image   latest frontal baseline film (lateral only when no frontal one); optional pins
    e  ECG         earliest baseline record
    n  note        earliest baseline radiology report (optionally chest-radiograph reports only)
    v, l           not built yet: vitals and labs have no record-level slot or loader
"""
import hashlib
import os
import re
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List

import numpy as np
import pandas as pd

from . import fsio
from .modsets import MOD_NAMES, NOTE, ModalitySet

# Every x_* table already holds the union of all levels (download/bq_export.py)
TABLES = ["cohort", "x_stays", "stay_summary", "analysis_dataset",
          "x_vitals", "x_labs", "x_notes", "x_ecg", "x_cxr", "x_file_manifest"]


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


SLOT_BUILDERS = {'x': slots_x, 'e': slots_e, 'n': slots_n}


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
    """dicom_id -> PIL image, from a local path or gs://."""

    def __init__(self, paths: dict):
        self.paths = paths

    def path(self, dicom_id):
        p = self.paths.get(dicom_id)
        if p is None:
            raise KeyError(f'no image file for {dicom_id}')
        return p

    def load(self, dicom_id):
        from PIL import Image
        with fsio.open_(self.path(dicom_id)) as f:
            return Image.open(f).convert('RGB')


@dataclass
class Bundle:
    """A loaded multichannel series: z-scored rows plus the raw mean / std that OpenTSLM quotes in its label text."""
    norm: np.ndarray                  # (K, L) float32, each row z-scored
    mean: np.ndarray                  # (K,)
    std: np.ndarray                   # (K,)
    names: List[str]
    info: Dict = field(default_factory=dict)


class EcgLoader:
    """ECG record (header path without extension, local or gs://) -> Bundle. Port of the OpenTSLM notebook's
    EcgLoader / SignalLoader, unchanged: physical units, every 5th sample (500 -> 100 Hz, as OpenTSLM trained on
    PTB-XL), leads matched by name, NaN gaps interpolated, length fixed, per-lead z-score in float64.
    wfdb reads local files only, so gs:// records are copied (.hea + .dat) to a local cache first."""
    LEADS = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
    cache_dir = os.path.join(tempfile.gettempdir(), 'mmd_ecg')

    def __init__(self, leads=LEADS, length=1000, hz=100):
        self.leads, self.length, self.hz = tuple(leads), length, hz

    @staticmethod
    def znorm(sig):                               # sig: (L, K)
        mean, std = sig.mean(0), sig.std(0)
        norm = (sig - mean) / np.where(std > 1e-6, std, 1.0)
        return norm.T.astype(np.float32), mean, std

    @staticmethod
    def fill_nan(sig):
        n_nan = int(np.isnan(sig).sum())
        for j in range(sig.shape[1]):
            x, bad = sig[:, j], np.isnan(sig[:, j])
            if bad.all():
                x[:] = 0.0
            elif bad.any():
                x[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), x[~bad])
        return n_nan

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

    def load(self, record) -> Bundle:
        import wfdb
        N = self.length
        rec = wfdb.rdrecord(self.local(record))
        pos = {n.lower(): i for i, n in enumerate(rec.sig_name)}
        sig = rec.p_signal[:, [pos[l.lower()] for l in self.leads]].astype(np.float64)
        sig = sig[::int(round(rec.fs / self.hz))][:N]
        if len(sig) < N:
            sig = np.pad(sig, ((0, N - len(sig)), (0, 0)), mode='edge')
        n_nan = self.fill_nan(sig)
        norm, mean, std = self.znorm(sig)
        return Bundle(norm, mean, std, list(self.leads), dict(n_nan=n_nan))
