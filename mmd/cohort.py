"""Who is perturbed, with which donors: tier cohorts, donor plans and the frozen cohort manifest.

    TierCohort   one tier's recipients (stays with a baseline record for every modality of the set) and ONE
                 fixed 5-fold partition, stratified on age group x sex x outcome
    DonorPlan    per recipient, n_draws donors from its own test fold and age x sex cell: `matched` (same outcome)
                 and `counterfactual` (opposite outcome)
    Manifest     everything a modality set's runs share, frozen per cohort version: slot tables, tiers + folds,
                 donor plans and the configuration list (built by perturb.build_manifest). `hash` covers the
                 records and folds; the runner and the analysis assert it, so every model sees identical inputs.

Ported from the LLaVA-Med / OpenTSLM notebooks (Swap cohort, Donor plan cells) with the logic unchanged, so
records, folds and donors are identical to the existing runs.
"""
import hashlib
import os
import warnings
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import fsio
from .modsets import MOD_BIT, ModalitySet, canonical

AGE_GROUPS = ['18-64', '65-74', '75-84', '>=85']       # the brief's / report_swap_strata's bins


def age_group(age):
    a = np.asarray(age, float)
    return np.select([a < 65, a < 75, a < 85], AGE_GROUPS[:3], AGE_GROUPS[3])


class TierCohort:
    """Recipients for one tier (cohort level): stays with a baseline record for every modality slot.

    `slots` maps a modality key to its slot table (index stay_id; columns first, last). A stay's records do not
    depend on the tier, so donors resolve through the same tables. Folds are ONE fixed partition per tier,
    stratified on age group x sex x outcome: donors are drawn inside the recipient's test fold, so a fold-fitted
    probe never sees its donors.
    """

    def __init__(self, ad: pd.DataFrame, level: str, slots: Dict[str, pd.DataFrame],
                 n_splits: int = 5, fold_seed: int = 0):
        self.level, self.slots = level, slots
        self.mods = list(canonical(slots))
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

    def resolve(self, mod: str, stay: int, slot: str):
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


class DonorPlan:
    """Donors are drawn once and saved, so a rerun asks for exactly the same records and the store resumes."""

    def __init__(self, cohort: TierCohort, n_draws: int = 20, seed: int = 0, path: str = None,
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


CONFIG_COLUMNS = ['level_id', 'config', 'family', 'slot', 'order', 'core', 'factor', 'stay_id', 'draw', 'donor',
                  'record']


class Manifest:
    """Everything a modality set's runs share, frozen per cohort version (see the module docstring)."""

    def __init__(self, modset: ModalitySet, cohorts: Dict[str, TierCohort], plans: Dict[str, DonorPlan],
                 configs: pd.DataFrame, meta: dict):
        self.modset, self.cohorts, self.plans, self.configs, self.meta = modset, cohorts, plans, configs, meta

    @property
    def name(self) -> str:
        return self.modset.name

    @property
    def mods(self):
        return self.modset.mods

    @property
    def levels(self):
        return list(self.cohorts)

    @property
    def hash(self):
        return self.meta['hash']

    @property
    def n_draws_model(self) -> int:
        return int(self.meta['n_draws_model'])

    def resolve(self, mod: str, stay: int, slot: str):
        """Slot -> the actual record (dicom_id, ECG path, note_id), 'mean' or None. Slot tables are per stay,
        so any tier resolves any record."""
        return self.cohorts[self.levels[-1]].resolve(mod, stay, slot)

    def model_configs(self, factors: bool = True) -> pd.DataFrame:
        """The configurations that get a model pass (draw < n_draws_model); prompt-factor copies optional."""
        c = self.configs[self.configs.draw < self.n_draws_model]
        if not factors:
            c = c[c.factor == '']
        return c.reset_index(drop=True)

    def analysis_configs(self, model_level: bool = True) -> pd.DataFrame:
        """The configurations the standard analysis scores: no prompt factors; model-pass draws only, or every
        draw (feature-level lookups are free, so they use all of them)."""
        c = self.model_configs() if model_level else self.configs
        return c[c.factor == ''].reset_index(drop=True)

    @staticmethod
    def digest(configs: pd.DataFrame, cohorts: Dict[str, TierCohort]) -> str:
        h = hashlib.sha256()
        for r in sorted(set(configs.record)):
            h.update(r.encode()); h.update(b'\n')
        for lv in sorted(cohorts):
            t = cohorts[lv].table
            h.update(t[['stay_id', 'fold']].sort_values('stay_id').to_csv(index=False).encode())
        return h.hexdigest()[:16]

    def save(self, path):
        for m, t in self.cohorts[self.levels[0]].slots.items():
            fsio.write_parquet(t.rename_axis('stay_id').reset_index(), f'{path}/slots_{m}.parquet')
        for lv, c in self.cohorts.items():
            fsio.write_parquet(c.table, f'{path}/cohort_{lv}.parquet')
            fsio.write_parquet(c.all, f'{path}/stays_{lv}.parquet')
            fsio.write_csv(self.plans[lv].plan, f'{path}/donor_plan_{self.name}_{lv}.csv')
        fsio.write_parquet(self.configs, f'{path}/configs.parquet')
        fsio.write_json(self.meta, f'{path}/manifest.json')
        return path

    @classmethod
    def load(cls, path, verbose=True):
        meta = fsio.read_json(f'{path}/manifest.json')
        modset = ModalitySet(meta['modset'], tuple(meta['mods']))
        slots = {}
        for m in modset.mods:
            t = fsio.read_parquet(f'{path}/slots_{m}.parquet').set_index('stay_id')
            t.index = t.index.astype('int64')
            slots[m] = t
        cohorts, plans = {}, {}
        for lv in meta['levels']:
            c = TierCohort.from_tables(lv, slots, fsio.read_parquet(f'{path}/cohort_{lv}.parquet'),
                                       fsio.read_parquet(f'{path}/stays_{lv}.parquet'))
            cohorts[lv] = c
            plans[lv] = DonorPlan(c, meta['n_draws'],
                                  plan=fsio.read_csv(f'{path}/donor_plan_{modset.name}_{lv}.csv'))
        configs = fsio.read_parquet(f'{path}/configs.parquet')
        m = cls(modset, cohorts, plans, configs, meta)
        got = cls.digest(configs, cohorts)
        if got != meta['hash']:
            raise ValueError(f'manifest {path}: content hash {got} != recorded {meta["hash"]} (edited by hand?)')
        if verbose:
            print(f'cohort manifest {modset.name} {meta["cohort_version"]} ({meta["hash"]}): '
                  f'{len(configs):,} configuration rows, levels {meta["levels"]}')
        return m
