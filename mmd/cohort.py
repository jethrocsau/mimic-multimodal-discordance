"""Record grammar, swap cohorts, donor plans and the frozen cohort manifest.

Ported from the LLaVA-Med / OpenTSLM notebooks (Swap cohort, Donor plan cells) with the logic unchanged, so
records, folds and donors are identical to the existing runs. A record id names a whole configuration:

    <stay>|<mod>:<slot>[|<mod>:<slot>]...[|~<opt>=<value>]...
    mod  ::= e (ECG) | x (CXR) | v (vitals) | l (labs) | n (notes)
    slot ::= first | last | none | mean | d<donor_stay>:first

ECG, CXR and notes slots resolve to one record (ECG path, dicom_id, note_id). Vitals and labs have no single
record: their slot tables hold a window key instead (the stay's baseline / follow-up window), and the loader
builds the series or table from it. The grammar is the same for all five.

    12345|x:first|n:first          reference: the stay's own baseline records
    12345|x:last|n:first           temporal swap of the image
    12345|x:d9876:first|n:first    donor swap: stay 9876's baseline image, own note
    12345|x:mean|n:first           image replaced by the cohort-mean encoder output
    12345|x:first|n:first|~order=flip|~para=1 prompt factors (other slot order, paraphrase 1)
    12345|e:first|v:d9876:first|l:first|n:none   ECG, stay 9876's baseline vitals, own labs, no note

Modality keys appear in the fixed order e, x, v, l, n and options sorted by name, so a configuration has exactly
one spelling and the store can resume on it (records of the existing e/x/n runs keep their spelling).
"""
import hashlib
import os
import warnings
from datetime import datetime, timezone
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import fsio

MOD_ORDER = ('e', 'x', 'v', 'l', 'n')
MOD_NAMES = {'e': 'ECG', 'x': 'CXR image', 'v': 'vitals', 'l': 'labs', 'n': 'note'}
MOD_BIT = {'v': 1, 'l': 2, 'e': 4, 'n': 8, 'x': 16}  # analysis_dataset bitmask (rep_mask)
AGE_GROUPS = ['18-64', '65-74', '75-84', '>=85']       # the brief's / report_swap_strata's bins
PAIRS = {'cxr_notes': ('x', 'n'), 'ecg_notes': ('e', 'n')}


def age_group(age):
    a = np.asarray(age, float)
    return np.select([a < 65, a < 75, a < 85], AGE_GROUPS[:3], AGE_GROUPS[3])


def rec(stay_id, slots: Dict[str, str], opts: Dict[str, str] = None) -> str:
    bad = set(slots) - set(MOD_ORDER)
    if bad:
        raise ValueError(f'unknown modality key(s) {sorted(bad)}: use {MOD_ORDER}')
    r = f"{int(stay_id)}|" + "|".join(f"{m}:{slots[m]}" for m in MOD_ORDER if m in slots)
    if opts:
        r += "".join(f"|~{k}={opts[k]}" for k in sorted(opts))
    return r


def parse(record: str):
    """'12345|e:d9876:first|n:none' -> (12345, {'e': 'd9876:first', 'n': 'none'}). Options are ignored."""
    head, *parts = record.split('|')
    return int(head), {p.split(':', 1)[0]: p.split(':', 1)[1] for p in parts if not p.startswith('~')}


def parse_opts(record: str) -> Dict[str, str]:
    return dict(p[1:].split('=', 1) for p in record.split('|')[1:] if p.startswith('~'))


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


class SwapCohort:
    """Recipients for one tier (cohort level): stays with a baseline record for every modality slot.

    `slots` maps a modality key to pick_slots() output (index stay_id; first, last). A stay's records
    do not depend on the tier, so donors resolve through the same tables. Folds are ONE fixed 5-fold
    partition per tier, stratified on age group x sex x outcome: donors are drawn inside the
    recipient's test fold, so a fold-fitted probe never sees its donors.
    """

    def __init__(self, ad: pd.DataFrame, level: str, slots: Dict[str, pd.DataFrame],
                 n_splits: int = 5, fold_seed: int = 0):
        self.level, self.slots, self.mods = level, slots, [m for m in MOD_ORDER if m in slots]
        if ad is None:                                       # filled by from_tables()
            return
        d = ad[ad.level_id == level][['stay_id', 'subject_id', 'died_1y', 'age', 'sex', 'rep_mask']].copy()
        d['stay_id'] = d.stay_id.astype('int64')
        d['y'] = d.died_1y.astype(int)
        d['age'] = d.age.astype(float)
        d['sex_num'] = d.sex.map({'M': 1, 'F': 0, 'm': 1, 'f': 0, 1: 1, 0: 0}).astype(float)
        d['age_group'] = age_group(d.age)
        d['cell'] = d.age_group + '|' + d.sex.astype(str)
        for m in self.mods:
            t = slots[m]
            d[f'{m}_first'] = d.stay_id.map(t['first'])
            d[f'{m}_last'] = d.stay_id.map(t['last'])
            d[f'{m}_has_last'] = d[f'{m}_last'].notna() & ((d.rep_mask.astype(int) & MOD_BIT[m]) > 0)
        self.all = d.reset_index(drop=True)
        keep = np.all([d[f'{m}_first'].notna() for m in self.mods], axis=0)
        t = d[keep].reset_index(drop=True)
        if len(t) < n_splits:
            got = {m: f"{len(slots[m]):,} stays with a {m} record, {d[f'{m}_first'].notna().sum():,} of them "
                      f"baseline in {level}" for m in self.mods}
            raise ValueError(f'{level}: {len(t)} recipients out of {len(d):,} stays in the tier. '
                             f'Slot coverage: {got}. An empty slot table usually means its source frame is '
                             f'empty (e.g. no CXR file found under the raw-file root).')
        key = t.cell + '|' + t.y.astype(str)
        t['fold'] = -1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)         # rare age x sex x outcome cells
            for k, (_, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=fold_seed)
                                        .split(t, key)):
                t.loc[te, 'fold'] = k
        self.table = t

    @classmethod
    def from_tables(cls, level, slots, table: pd.DataFrame, all_: pd.DataFrame):
        c = cls(None, level, slots)
        c.table, c.all = table.reset_index(drop=True), all_.reset_index(drop=True)
        return c

    def underlying(self, mod: str, stay: int, slot: str):
        """The actual record behind a slot: an ECG path, dicom_id or note_id; 'mean'; or None."""
        if slot == 'none':
            return None
        if slot == 'mean':
            return 'mean'
        if slot.startswith('d'):
            stay, slot = int(slot[1:].split(':')[0]), slot.split(':')[1]
        v = self.slots[mod][slot].get(stay)
        return None if (v is None or (isinstance(v, float) and np.isnan(v))) else v

    def summary(self) -> pd.DataFrame:
        t = self.table
        rows = [('stays in tier', len(self.all), self.all.y.mean()),
                (f'baseline {"+".join(self.mods)} (recipients)', len(t), t.y.mean())]
        rows += [(f'{m}: also a last follow-up record', int(t[f'{m}_has_last'].sum()),
                  t.y[t[f'{m}_has_last']].mean()) for m in self.mods]
        out = pd.DataFrame(rows, columns=['subset', 'stays', 'died_1y'])
        out.insert(0, 'level_id', self.level)
        out['deaths'] = (out.stays * out.died_1y).round().astype(int)
        out['died_1y'] = (100 * out.died_1y).round(1)
        return out


# ── Donor plan and the configuration list ────────────────────────────────────
#   temporal        the slot becomes the stay's own `last` record (only where rep_mask has the bit)
#   matched         a donor's baseline record: same tier, same age group x sex, SAME outcome
#   mean            image/ECG: the cohort-mean encoder output (injected by a hook); notes: removed
#   counterfactual  a donor's baseline record: same tier, same age group x sex, OPPOSITE outcome
# Donors come from the recipient's own test fold and are never the recipient. The plan is drawn once
# and saved, so a rerun asks for exactly the same records and the store resumes instead of redoing.
FAMILIES4 = ['temporal', 'matched', 'mean', 'counterfactual']
DONOR_FAMS = ['matched', 'counterfactual']


class DonorPlan:
    def __init__(self, cohort: SwapCohort, n_draws: int = 20, seed: int = 0, path: str = None,
                 plan: pd.DataFrame = None, verbose=True):
        self.cohort, self.n_draws = cohort, n_draws
        if plan is not None:
            self.plan = plan
            return
        if path and fsio.exists(path):
            p = fsio.read_csv(path)
            same = (set(p.recipient) == set(cohort.table.stay_id)) and p.draw.max() + 1 == n_draws
            if same:
                self.plan = p
                if verbose:
                    print(f"{cohort.level}: donor plan reused from {os.path.basename(path)}")
                return
            if verbose:
                print(f"{cohort.level}: saved donor plan does not match this cohort, drawing a new one")
        self.plan = self._draw(seed)

    def _draw(self, seed):
        t = self.cohort.table
        rng = np.random.default_rng(seed)
        sid, fold, cell, y = (t[c].to_numpy() for c in ('stay_id', 'fold', 'cell', 'y'))
        rows = []
        for i in range(len(t)):
            same = (fold == fold[i]) & (cell == cell[i]) & (sid != sid[i])
            for fam, ok in (('matched', same & (y == y[i])), ('counterfactual', same & (y != y[i]))):
                cand = sid[ok]
                if len(cand):
                    for d, donor in enumerate(rng.choice(cand, self.n_draws, replace=True)):
                        rows.append((self.cohort.level, sid[i], fam, d, int(donor), len(cand)))
        return pd.DataFrame(rows, columns=['level_id', 'recipient', 'family', 'draw', 'donor', 'pool'])

    def coverage(self) -> pd.DataFrame:
        t = self.cohort.table
        g = self.plan.drop_duplicates(['recipient', 'family']).merge(
            t[['stay_id', 'y']], left_on='recipient', right_on='stay_id')
        out = g.groupby('family').agg(recipients=('recipient', 'size'), died=('y', 'sum'),
                                      median_pool=('pool', 'median'))
        out['of_recipients'] = len(t)
        return out.reset_index().assign(level_id=self.cohort.level)

    def balance(self) -> pd.DataFrame:
        """Standardised mean differences, recipients vs their donors (brief S2: < 0.1)."""
        t = self.cohort.table.set_index('stay_id')
        rows = []
        for fam, g in self.plan.groupby('family'):
            r, d = t.loc[g.recipient], t.loc[g.donor]
            for c in ('age', 'sex_num'):
                sd = np.sqrt((r[c].var() + d[c].var()) / 2)
                rows.append(dict(level_id=self.cohort.level, family=fam, covariate=c,
                                 smd=float((r[c].mean() - d[c].mean()) / sd) if sd > 0 else 0.0))
        return pd.DataFrame(rows)


def swap_configs(cohort: SwapCohort, plan: DonorPlan, draws: int = None,
                 note_mean_is_none: bool = True) -> pd.DataFrame:
    """Every configuration a tier needs: one row per (config, recipient, draw) with its record id.

    Controls (OpenTSLM §9): ref = all first; A = first modality only; B = notes only;
    F = both last. Then the four families for each modality slot.
    """
    t = cohort.table
    mods = cohort.mods
    draws = draws or plan.n_draws
    base = {m: 'first' for m in mods}
    rows = []

    def add(config, family, slot, stay, slots, draw=0, donor=None):
        rows.append((cohort.level, config, family, slot, int(stay), draw, donor, rec(stay, slots)))

    for s in t.itertuples():
        add('ref', 'reference', '', s.stay_id, base)
        if len(mods) == 2:
            a, b = mods
            add('A', 'control', b, s.stay_id, {a: 'first', b: 'none'})       # slot = the one removed
            add('B', 'control', a, s.stay_id, {a: 'none', b: 'first'})
            if all(getattr(s, f'{m}_has_last') for m in mods):
                add('F', 'temporal', 'both', s.stay_id, {m: 'last' for m in mods})
        for m in mods:
            if getattr(s, f'{m}_has_last'):
                add(f'temporal_{m}', 'temporal', m, s.stay_id, {**base, m: 'last'})
            add(f'mean_{m}', 'mean', m, s.stay_id,
                {**base, m: 'none' if (m == 'n' and note_mean_is_none) else 'mean'})
    p = plan.plan[plan.plan.draw < draws]
    for r in p.itertuples():
        for m in mods:
            add(f'{r.family}_{m}', r.family, m, r.recipient, {**base, m: f'd{r.donor}:first'},
                draw=r.draw, donor=r.donor)
    return pd.DataFrame(rows, columns=['level_id', 'config', 'family', 'slot', 'stay_id', 'draw',
                                       'donor', 'record'])


def factor_configs(configs: pd.DataFrame, levels=('L1',), configs_in=None,
                   orders=('flip',), paras=(1, 2)) -> pd.DataFrame:
    """Prompt-factor copies of some configurations: slot order ('flip' = the other order
    from the model's default) and paraphrases.

    Default: reference + counterfactual records of L1 only, so the factors add ~30% rather than 6x.
    The copies carry `~order` / `~para` options in the record id and `@order` / `@para` in the config name;
    the standard analysis ignores them (config names containing '@').
    """
    configs_in = configs_in or ['ref'] + [c for c in configs.config.unique() if c.startswith('counterfactual_')]
    base = configs[configs.level_id.isin(levels) & configs.config.isin(configs_in)]
    out = []
    for kind, values in (('order', orders), ('para', paras)):
        for v in values:
            c = base.copy()
            c['config'] = c.config + f'@{kind}={v}'
            c['record'] = [rec(*parse(r), opts={kind: str(v)}) for r in c.record]
            out.append(c)
    return pd.concat(out, ignore_index=True) if out else base.iloc[:0]


# ── The frozen cohort manifest ───────────────────────────────────────────────
class Manifest:
    """Everything a pair's runs share, frozen per cohort version: slot tables, tier tables with folds,
    donor plans, and the configuration list (all draws; input-level passes use draw < n_draws_input).

    Every model reads the same manifest, so the swaps are identical across the zoo. `hash` covers the
    records, folds and donors; perturb and analyse assert it.
    """

    def __init__(self, pair, cohorts: Dict[str, SwapCohort], plans: Dict[str, DonorPlan],
                 configs: pd.DataFrame, meta: dict):
        self.pair, self.cohorts, self.plans, self.configs, self.meta = pair, cohorts, plans, configs, meta

    @property
    def mods(self):
        return PAIRS[self.pair]

    @property
    def levels(self):
        return list(self.cohorts)

    @property
    def resolver(self) -> SwapCohort:
        """Slot tables are per stay, so any tier resolves any record."""
        return self.cohorts[self.levels[-1]]

    def input_configs(self) -> pd.DataFrame:
        return self.configs[self.configs.draw < self.meta['n_draws_input']].reset_index(drop=True)

    @staticmethod
    def digest(configs: pd.DataFrame, cohorts: Dict[str, SwapCohort]) -> str:
        h = hashlib.sha256()
        for r in sorted(set(configs.record)):
            h.update(r.encode()); h.update(b'\n')
        for lv in sorted(cohorts):
            t = cohorts[lv].table
            h.update(t[['stay_id', 'fold']].sort_values('stay_id').to_csv(index=False).encode())
        return h.hexdigest()[:16]

    @property
    def hash(self):
        return self.meta['hash']

    def save(self, path):
        for m, t in self.cohorts[self.levels[0]].slots.items():
            fsio.write_parquet(t.rename_axis('stay_id').reset_index(), f'{path}/slots_{m}.parquet')
        for lv, c in self.cohorts.items():
            fsio.write_parquet(c.table, f'{path}/cohort_{lv}.parquet')
            fsio.write_parquet(c.all, f'{path}/stays_{lv}.parquet')
            fsio.write_csv(self.plans[lv].plan, f'{path}/donor_plan_{self.pair}_{lv}.csv')
        fsio.write_parquet(self.configs, f'{path}/configs.parquet')
        fsio.write_json(self.meta, f'{path}/manifest.json')
        return path

    @classmethod
    def load(cls, path, verbose=True):
        meta = fsio.read_json(f'{path}/manifest.json')
        pair = meta['pair']
        slots = {}
        for m in PAIRS[pair]:
            t = fsio.read_parquet(f'{path}/slots_{m}.parquet').set_index('stay_id')
            t.index = t.index.astype('int64')
            slots[m] = t
        cohorts, plans = {}, {}
        for lv in meta['levels']:
            c = SwapCohort.from_tables(lv, slots, fsio.read_parquet(f'{path}/cohort_{lv}.parquet'),
                                       fsio.read_parquet(f'{path}/stays_{lv}.parquet'))
            cohorts[lv] = c
            plans[lv] = DonorPlan(c, meta['n_draws'], plan=fsio.read_csv(f'{path}/donor_plan_{pair}_{lv}.csv'))
        configs = fsio.read_parquet(f'{path}/configs.parquet')
        m = cls(pair, cohorts, plans, configs, meta)
        got = cls.digest(configs, cohorts)
        if got != meta['hash']:
            raise ValueError(f'manifest {path}: content hash {got} != recorded {meta["hash"]} (edited by hand?)')
        if verbose:
            print(f'cohort manifest {pair} {meta["cohort_version"]} ({meta["hash"]}): '
                  f'{len(configs):,} configuration rows, levels {meta["levels"]}')
        return m


def build_manifest(cfg, pair: str, ad: pd.DataFrame, slots: Dict[str, pd.DataFrame],
                   donor_plan_paths: Dict[str, str] = None, factors: bool = True, verbose=True) -> Manifest:
    """Tiers, folds, donors (reused from saved plans when they match) and every configuration."""
    cohorts = {lv: SwapCohort(ad, lv, slots, n_splits=cfg.n_splits) for lv in cfg.levels}
    plans = {lv: DonorPlan(c, cfg.n_draws, path=(donor_plan_paths or {}).get(lv), verbose=verbose)
             for lv, c in cohorts.items()}
    configs = pd.concat([swap_configs(c, plans[lv], draws=cfg.n_draws) for lv, c in cohorts.items()],
                        ignore_index=True)
    if factors:
        inp = configs[configs.draw < cfg.n_draws_input]
        configs = pd.concat([configs, factor_configs(inp, levels=cfg.levels[:1])], ignore_index=True)
    reused = {lv: (donor_plan_paths or {}).get(lv) for lv in cohorts}
    meta = dict(pair=pair, mods=list(PAIRS[pair]), cohort_version=cfg.cohort_version, levels=list(cfg.levels),
                n_splits=cfg.n_splits, n_draws=cfg.n_draws, n_draws_input=cfg.n_draws_input,
                donor_plans_from=reused, created=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                hash=Manifest.digest(configs, cohorts))
    return Manifest(pair, cohorts, plans, configs, meta)


def run_records(m: Manifest, levels: Sequence[str] = None, families: Sequence[str] = None,
                factors: bool = True) -> list:
    """The records a model pass needs: input-level configs (draw < n_draws_input) for some levels/families."""
    c = m.input_configs()
    if levels:
        c = c[c.level_id.isin(levels)]
    if families and 'all' not in families:
        keep = set(families) | {'reference', 'control'}
        c = c[c.family.isin(keep)]
    if not factors:
        c = c[~c.config.str.contains('@')]
    return sorted(set(c.record))
