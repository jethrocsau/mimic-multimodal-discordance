"""Statistical helpers shared by the effect tables (one copy each)."""
import numpy as np
from scipy.stats import rankdata, wilcoxon as _wilcoxon


def wilcoxon(x) -> float:
    """Wilcoxon signed-rank p-value of x against 0; NaN when there are fewer than 10 values or all are 0."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 10 or np.allclose(x, 0):
        return np.nan
    try:
        return float(_wilcoxon(x).pvalue)
    except ValueError:
        return np.nan


def stratified_boot(fn, y, *cols, n_boot=1000, seed=0):
    """95% CI of fn(y, *cols) over bootstrap resamples that keep the number of decedents and survivors."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == k) for k in (0, 1)]
    if min(len(g) for g in groups) < 2:
        return np.nan, np.nan
    vals = []
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(g, len(g)) for g in groups])
        vals.append(fn(y[i], *[c[i] for c in cols]))
    return tuple(np.nanpercentile(vals, [2.5, 97.5]))


def auc_rows(scores, y):
    """AUROC per row of scores (B, n) against y (B, n), via ranks (ties averaged)."""
    r = rankdata(scores, axis=1)
    n1 = y.sum(1); n0 = y.shape[1] - n1
    return ((r * y).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
