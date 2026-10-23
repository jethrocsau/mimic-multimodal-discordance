"""Swap analysis against the reference, AUROC, donor checks, temporal and paired dominance.

Ported from the notebooks' section 8 / 10 cells, with their module-level settings (SIG, SIG_NAME, NOTE_BLIND,
SWAP_COHORTS, SWAP_CONFIGS, ANSWER_READOUTS, ...) gathered into one `Ctx`. The arithmetic is unchanged.

P(Yes) is already the model's own risk score, so it is read RAW: p_ref is P(Yes) on the unswapped record,
p_<config> the P(Yes) of the swapped pass (mean over the stay's donor draws). Vector readouts go through the
probe. SCORE_FRAMES = probe frames for vectors + raw P(Yes) frames.
"""
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import roc_auc_score

RAW = 'no probe (raw P(Yes))'
DONOR_ROWS = {'matched', 'counterfactual', 'label-specific'}
FAM_ROWS = ['temporal', 'matched', 'mean', 'counterfactual', 'label-specific']
AUC_ROWS = ['temporal', 'matched', 'mean', 'counterfactual']
GROUPS = [('died', 1), ('survived', 0)]
N_BOOT_AUC = 500
N_BOOT_PAIRED = 2000


@dataclass
class Ctx:
    model_name: str
    sig: str                                   # signal slot: 'x' (CXR) or 'e' (ECG)
    sig_name: str                              # 'CXR image' / 'ECG'
    cohorts: Dict                              # level -> SwapCohort
    configs: pd.DataFrame                      # input-level configs (draw < n_draws_input)
    answer_readouts: List = field(default_factory=list)
    note_blind: set = field(default_factory=set)
    out_dir: str = ''
    fig_dir: str = ''
    file_tag: str = 'cxr_notes'

    @property
    def slot_style(self):
        return {self.sig: (f'{self.sig_name} swapped', '#2a78d6', 'o'), 'n': ('note swapped', '#eb6834', 's')}


def raw_answer_frames(ctx: Ctx):
    out = []
    for ro in ctx.answer_readouts:
        for lv, c in ctx.cohorts.items():
            t, cfg = c.table, ctx.configs[ctx.configs.level_id == lv]
            p = cfg.assign(p=ro.X_of(list(cfg.record))[:, 0]).dropna(subset=['p'])
            w = p.groupby(['stay_id', 'config']).p.mean().unstack().add_prefix('p_')
            f = pd.DataFrame({'level_id': lv, 'stay_id': t.stay_id.to_numpy(), 'y': t.y.to_numpy()})
            f = f.join(w, on='stay_id')
            for n in [k[2:] for k in w.columns if k != 'p_ref']:
                f[f'd_{n}'] = f[f'p_{n}'] - f.p_ref
                f[f'shift_{n}'] = np.nan                     # one number: no representation shift
            out.append(f.assign(readout=ro.name, readout_family=ro.family, clf=RAW))
    return out


def score_frames(ctx: Ctx, swap_frames):
    return ([d for d in swap_frames if not d.readout.iloc[0].startswith('answer P(Yes)')]
            + raw_answer_frames(ctx))


def clean_summary(ctx: Ctx, summary: pd.DataFrame) -> pd.DataFrame:
    """d_auroc blanked on donor rows (donors are chosen BY outcome) and rows a readout cannot react to dropped."""
    s = summary.copy()
    donor = s.family.isin(DONOR_ROWS)
    s.loc[donor, ['auroc_alt', 'd_auroc', 'd_auroc_lo', 'd_auroc_hi']] = np.nan
    return s[~(s.readout.isin(ctx.note_blind) & s.slot.isin(['n', 'both']))]


def reference_shifts(ctx: Ctx, frames):
    """Per (tier, readout, clf, family, slot, outcome group): mean reference and swapped risk, and the
    paired Δ with its 95% CI (1.96 SE over patients)."""
    rows = []
    for d in frames:
        ro = d.readout.iloc[0]
        for fam in FAM_ROWS:
            for s in ctx.slot_style:
                if s == 'n' and ro in ctx.note_blind:
                    continue
                if fam == 'label-specific':
                    a, b = f'p_counterfactual_{s}', f'p_matched_{s}'
                else:
                    a, b = f'p_{fam}_{s}', 'p_ref'
                if a not in d or b not in d:
                    continue
                for grp, yv in GROUPS:
                    g = d[(d.y == yv) & d[a].notna() & d[b].notna()]
                    if len(g) < 10:
                        continue
                    delta = (g[a] - g[b]).to_numpy()
                    rows.append(dict(level_id=d.level_id.iloc[0], readout=ro, clf=d.clf.iloc[0],
                                     family=fam, slot=s, group=grp, n=len(g), ref=g[b].mean(),
                                     swapped=g[a].mean(), delta=delta.mean(),
                                     delta_ci=1.96 * delta.std(ddof=1) / np.sqrt(len(g))))
    return pd.DataFrame(rows)


def _auc_rows(scores, y):
    """AUROC per row of scores (B, n) against y (B, n), via ranks (ties averaged)."""
    r = rankdata(scores, axis=1)
    n1 = y.sum(1); n0 = y.shape[1] - n1
    return ((r * y).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def auroc_shifts(ctx: Ctx, frames, n_boot=N_BOOT_AUC, seed=0):
    rng, rows = np.random.default_rng(seed), []
    for d in frames:
        ro = d.readout.iloc[0]
        slots = [s for s in ctx.slot_style if not (s == 'n' and ro in ctx.note_blind)]
        for fam in AUC_ROWS:
            cols = [f'p_{fam}_{s}' for s in slots if f'p_{fam}_{s}' in d]
            if not cols or 'p_ref' not in d:
                continue
            g = d.dropna(subset=['p_ref'] + cols)
            y = g.y.to_numpy().astype(int)
            if len(g) < 20 or y.min() == y.max():
                continue
            sc = np.stack([g.p_ref.to_numpy()] + [g[c].to_numpy() for c in cols])     # (1+S, n)
            idx = rng.integers(0, len(g), (n_boot, len(g)))
            yb = y[idx]
            ok = (yb.sum(1) > 0) & (yb.sum(1) < len(g))
            auc = _auc_rows(sc, np.broadcast_to(y, sc.shape))
            boot = np.stack([_auc_rows(s[idx][ok], yb[ok]) for s in sc])               # (1+S, B)
            for k, c in enumerate(cols, start=1):
                dlt = boot[k] - boot[0]
                rows.append(dict(level_id=d.level_id.iloc[0], readout=ro, clf=d.clf.iloc[0], family=fam,
                                 slot=c.rsplit('_', 1)[1], n=len(g), died=int(y.sum()),
                                 auroc_ref=auc[0], auroc_swap=auc[k],
                                 swap_lo=np.percentile(boot[k], 2.5), swap_hi=np.percentile(boot[k], 97.5),
                                 d_auroc=auc[k] - auc[0], d_lo=np.percentile(dlt, 2.5),
                                 d_hi=np.percentile(dlt, 97.5)))
    return pd.DataFrame(rows)


def auroc_by_draw(ctx: Ctx):
    """Donor swaps: ΔAUROC from ONE donor vs from the MEAN of the donors (raw P(Yes)). Averaging donors pulls
    every score toward its outcome group's mean, so 'mean of draws' inflates matched AUROC by construction."""
    donor_cfgs = [f'{f}_{s}' for f in ('matched', 'counterfactual') for s in (ctx.sig, 'n')]
    rows = []
    for ro in ctx.answer_readouts:
        for lv, c in ctx.cohorts.items():
            cfg = ctx.configs[(ctx.configs.level_id == lv) & ctx.configs.config.isin(['ref'] + donor_cfgs)]
            cfg = cfg.assign(p=ro.X_of(list(cfg.record))[:, 0]).dropna(subset=['p'])
            y = c.table.set_index('stay_id').y
            ref = cfg[cfg.config == 'ref'].groupby('stay_id').p.mean()
            for conf in donor_cfgs:
                sub = cfg[cfg.config == conf]
                per = [(f'draw {int(k)}', sub[sub.draw == k].groupby('stay_id').p.mean())
                       for k in sorted(sub.draw.unique())]
                for name, s in per + [('mean of draws', sub.groupby('stay_id').p.mean())]:
                    ids = s.index.intersection(ref.index)
                    if len(ids) < 20 or y[ids].nunique() < 2:
                        continue
                    fam, slot = conf.rsplit('_', 1)
                    rows.append(dict(readout=ro.name, level_id=lv, family=fam, slot=ctx.slot_style[slot][0],
                                     scored=name, n=len(ids), auroc_ref=roc_auc_score(y[ids], ref[ids]),
                                     auroc_swap=roc_auc_score(y[ids], s[ids])))
    out = pd.DataFrame(rows)
    if len(out):
        out['d_auroc'] = out.auroc_swap - out.auroc_ref
    return out


def _wil(x):
    x = np.asarray(x, float)
    return np.nan if len(x) < 10 or np.allclose(x, 0) else float(wilcoxon(x).pvalue)


def temporal_readouts(ctx: Ctx, swap_frames):
    return [r for r in dict.fromkeys(d.readout.iloc[0] for d in swap_frames)
            if r not in ctx.note_blind and 'P(Yes)' not in r]


def temporal_pairs(ctx: Ctx, frames, clf='Logistic regression'):
    """Temporal swap, signal vs note on the SAME patients (both slots have a follow-up record)."""
    keep = temporal_readouts(ctx, frames)
    sig = ctx.sig
    need = [f'd_temporal_{sig}', 'd_temporal_n', f'shift_temporal_{sig}', 'shift_temporal_n']
    out = []
    for d in frames:
        if d.clf.iloc[0] != clf or d.readout.iloc[0] not in keep:
            continue
        if not all(c in d for c in need):
            continue
        g = d[['level_id', 'readout', 'stay_id', 'y'] + need].dropna()
        s = np.where(g.y == 1, 1.0, -1.0)
        out.append(pd.DataFrame(dict(level_id=g.level_id, readout=g.readout, stay_id=g.stay_id, y=g.y,
                                     sim_sig=1 - g[f'shift_temporal_{sig}'], sim_n=1 - g.shift_temporal_n,
                                     pull_sig=s * g[f'd_temporal_{sig}'], pull_n=s * g.d_temporal_n)))
    p = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if len(p):
        p['pull_diff'] = p.pull_n - p.pull_sig
        p['sim_mean'] = (p.sim_sig + p.sim_n) / 2
    return p


def binned(x, y, bins=5):
    g = pd.DataFrame(dict(x=x, y=y)).dropna()
    g['b'] = pd.qcut(g.x, bins, labels=False, duplicates='drop')
    b = g.groupby('b').agg(x=('x', 'median'), m=('y', 'mean'), sd=('y', 'std'), n=('y', 'size'))
    b['ci'] = 1.96 * b.sd / np.sqrt(b.n)
    return b


def temporal_summary(tp):
    rows = []
    for (lv, ro), g in tp.groupby(['level_id', 'readout']):
        rows.append(dict(level_id=lv, readout=ro, n_paired=len(g), died=int(g.y.sum()),
                         sim_signal=g.sim_sig.mean(), sim_note=g.sim_n.mean(), p_sim=_wil(g.sim_n - g.sim_sig),
                         pull_signal=g.pull_sig.mean(), pull_note=g.pull_n.mean(),
                         pull_diff=g.pull_diff.mean(),
                         pull_diff_ci=1.96 * g.pull_diff.std() / np.sqrt(len(g)), p_pull=_wil(g.pull_diff)))
    return pd.DataFrame(rows)


def _paired_cols(ctx: Ctx, d, fam):
    """(measure, signal values, note values) for one family of one probe frame."""
    sig = ctx.sig
    true_dir = np.where(d.y.to_numpy() == 1, 1.0, -1.0)
    if fam == 'label-specific':
        cx = (d[f'd_counterfactual_{sig}'] - d[f'd_matched_{sig}']).to_numpy()
        cn = (d.d_counterfactual_n - d.d_matched_n).to_numpy()
        return [('|risk change|', np.abs(cx), np.abs(cn)), ('outcome pull', -true_dir * cx, -true_dir * cn)]
    dx, dn = d[f'd_{fam}_{sig}'].to_numpy(), d[f'd_{fam}_n'].to_numpy()
    out = [('representation', d[f'shift_{fam}_{sig}'].to_numpy(), d[f'shift_{fam}_n'].to_numpy()),
           ('|risk change|', np.abs(dx), np.abs(dn))]
    if fam == 'temporal':
        out.append(('outcome pull', true_dir * dx, true_dir * dn))
    elif fam == 'counterfactual':
        out.append(('outcome pull', -true_dir * dx, -true_dir * dn))
    return out


def paired_dominance(ctx: Ctx, frames, seed=0):
    """Within a family, signal swap vs note swap on the same recipients and donors, same probe:
    dominance = (note − signal) / (|note| + |signal|) in [−1, 1], paired bootstrap CI."""
    rng = np.random.default_rng(seed)
    dom = lambda a, b: (b - a) / np.maximum(np.abs(a) + np.abs(b), 1e-12)
    sig = ctx.sig
    rows = []
    for d in frames:
        if d.readout.iloc[0] in ctx.note_blind:
            continue
        for fam in FAM_ROWS:
            need = ([f'd_counterfactual_{sig}', 'd_counterfactual_n', f'd_matched_{sig}', 'd_matched_n']
                    if fam == 'label-specific'
                    else [f'd_{fam}_{sig}', f'd_{fam}_n', f'shift_{fam}_{sig}', f'shift_{fam}_n'])
            if not all(c in d for c in need):
                continue
            for measure, x, n in _paired_cols(ctx, d, fam):
                k = np.isfinite(x) & np.isfinite(n)
                if k.sum() < 20:
                    continue
                x, n = x[k], n[k]
                idx = rng.integers(0, len(x), (N_BOOT_PAIRED, len(x)))
                bx, bn = x[idx].mean(1), n[idx].mean(1)
                diff = n - x
                rows.append(dict(level_id=d.level_id.iloc[0], readout=d.readout.iloc[0],
                                 readout_family=d.readout_family.iloc[0], clf=d.clf.iloc[0], family=fam,
                                 measure=measure, n_paired=int(k.sum()), signal=x.mean(), note=n.mean(),
                                 diff=diff.mean(), diff_lo=np.percentile(bn - bx, 2.5),
                                 diff_hi=np.percentile(bn - bx, 97.5), p_wilcoxon=_wil(diff),
                                 note_wins=(diff > 0).sum() / max((diff != 0).sum(), 1),
                                 dominance=dom(x.mean(), n.mean()),
                                 dominance_lo=np.percentile(dom(bx, bn), 2.5),
                                 dominance_hi=np.percentile(dom(bx, bn), 97.5)))
    return pd.DataFrame(rows)
