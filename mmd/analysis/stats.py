"""Statistics per family and slot, decedents vs survivors (ported unchanged from the notebooks).

  delta_died / delta_survived   mean Δp by the RECIPIENT's outcome (for counterfactual these are the
                                two directions: survivor record into a decedent's stay, and back)
  gap                           delta_died - delta_survived, stratified bootstrap 95% CI over stays
  label-specific                per stay, Δcounterfactual - Δmatched (same slot)
  d_auroc                       AUROC(config) - AUROC(ref), paired bootstrap
Δ is already averaged over draws per stay, so nothing counts a repeat as a new observation.
"""
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

from ..cohort import SwapCohort
from .probe import Swap4Probe

N_BOOT = 1000


def _wil(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 10 or np.allclose(x, 0):
        return np.nan
    try:
        return float(wilcoxon(x).pvalue)
    except ValueError:
        return np.nan


def _boot(fn, y, *cols, n_boot=None, seed=0):
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == k) for k in (0, 1)]
    if min(len(g) for g in groups) < 2:
        return np.nan, np.nan
    vals = []
    for _ in range(n_boot or N_BOOT):
        i = np.concatenate([rng.choice(g, len(g)) for g in groups])
        vals.append(fn(y[i], *[c[i] for c in cols]))
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def _rows_of(d: pd.DataFrame):
    """(family, slot, alt column, base column, shift column) for every comparison in a probe frame."""
    cfgs = [c[2:] for c in d.columns if c.startswith('p_') and c != 'p_ref']
    out = []
    for c in cfgs:
        if c in ('A', 'B'):
            continue
        fam, slot = ('temporal', 'both') if c == 'F' else c.rsplit('_', 1)
        out.append((fam, slot, f'p_{c}', 'p_ref', f'shift_{c}'))
    for slot in sorted({c.rsplit('_', 1)[1] for c in cfgs if '_' in c}):
        if f'p_counterfactual_{slot}' in d and f'p_matched_{slot}' in d:
            out.append(('label-specific', slot, f'p_counterfactual_{slot}', f'p_matched_{slot}',
                        f'shift_counterfactual_{slot}'))
    return out


def summarise4(frames: Sequence[pd.DataFrame], model: str = '') -> pd.DataFrame:
    gap = lambda y, x: x[y == 1].mean() - x[y == 0].mean()
    dauc = lambda y, a, b: roc_auc_score(y, a) - roc_auc_score(y, b)
    rows = []
    for d in frames:
        meta = d[['level_id', 'readout', 'readout_family', 'clf']].iloc[0].to_dict()
        for fam, slot, alt, base, sh in _rows_of(d):
            k = (d[alt].notna() & d[base].notna()).to_numpy()
            y = d.y.to_numpy()[k]
            if k.sum() < 10 or len(np.unique(y)) < 2:
                continue
            a, b = d[alt].to_numpy()[k], d[base].to_numpy()[k]
            x = a - b
            glo, ghi = _boot(gap, y, x)
            alo, ahi = _boot(dauc, y, a, b)
            rows.append(dict(model=model, **meta, family=fam, slot=slot, n=int(k.sum()),
                             died=int(y.sum()), prevalence=float(y.mean()),
                             delta_died=x[y == 1].mean(), delta_survived=x[y == 0].mean(),
                             gap=gap(y, x), gap_lo=glo, gap_hi=ghi,
                             p_died=_wil(x[y == 1]), p_survived=_wil(x[y == 0]),
                             auroc_ref=roc_auc_score(y, b), auroc_alt=roc_auc_score(y, a),
                             d_auroc=dauc(y, a, b), d_auroc_lo=alo, d_auroc_hi=ahi,
                             rep_shift_median=float(np.nanmedian(d[sh].to_numpy()[k]))))
    return pd.DataFrame(rows)


def selftest4(probe: Swap4Probe, cohort: SwapCohort, configs, readout, n_perm=10) -> pd.DataFrame:
    """Labels permuted before the probe is fitted and before decedents/survivors are split, one
    permutation per seed. Every gap must be ~0: passes when |mean| <= 3 SE across permutations."""
    per = {}
    for s in range(n_perm):
        d = probe.run(cohort, configs, readout, shuffle_seed=1000 + s)
        for fam, slot, alt, base, _ in _rows_of(d):
            k = (d[alt].notna() & d[base].notna()).to_numpy()
            y = d.y.to_numpy()[k]
            if k.sum() >= 10 and len(np.unique(y)) == 2:
                x = (d[alt] - d[base]).to_numpy()[k]
                per.setdefault((fam, slot), []).append(x[y == 1].mean() - x[y == 0].mean())
    rows = []
    for (fam, slot), v in per.items():
        v = np.asarray(v)
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        rows.append(dict(level_id=cohort.level, readout=readout.name, clf=probe.clf_name,
                         family=fam, slot=slot, perms=len(v), gap_mean=v.mean(), gap_se=se,
                         verdict='too few' if len(v) < 5 else
                                 ('pass' if abs(v.mean()) <= 3 * se else 'FAIL')))     # <=: an all-zero family passes
    return pd.DataFrame(rows)


def discordance4(frames: Sequence[pd.DataFrame], bins: int = 5) -> pd.DataFrame:
    """Temporal Δ against how far the record moved (representation shift of the temporal swap)."""
    out = []
    for d in frames:
        for c in [c for c in d.columns if c.startswith('shift_temporal_')]:
            slot = c.rsplit('_', 1)[1]
            g = d[['y', f'd_temporal_{slot}', c]].dropna()
            if g[c].nunique() < bins:
                continue
            g = g.assign(bin=pd.qcut(g[c], bins, labels=False, duplicates='drop'),
                         delta=g[f'd_temporal_{slot}'])
            b = g.groupby('bin').agg(n=('y', 'size'), discordance=(c, 'median'), delta=('delta', 'mean'))
            b['delta_died'] = g[g.y == 1].groupby('bin').delta.mean()
            b['delta_survived'] = g[g.y == 0].groupby('bin').delta.mean()
            out.append(b.reset_index().assign(level_id=d.level_id.iloc[0], readout=d.readout.iloc[0],
                                              clf=d.clf.iloc[0], slot=slot))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
