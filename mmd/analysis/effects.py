"""Perturbation effects: every table the analysis writes, computed from the probe's long score table.

`scores` has one row per (tier, readout, classifier, stay, configuration) (probe.LONG). A "frame" is one
(level_id, readout, readout_family, clf) group of it. Raw P(Yes) frames (clf = RAW) read the model's own score;
vector readouts go through the probe.

    summarise            Δ per family and slot by the recipient's outcome: delta_died / delta_survived, their gap
                         (stratified bootstrap CI), d_auroc; plus label-specific = Δcounterfactual − Δmatched
    label_shuffle_test   labels permuted before the probe is fitted: every gap must be ~0
    reference_shifts     mean reference and perturbed risk per outcome group, paired Δ with its CI
    auroc_shifts         AUROC at the reference and after each perturbation, same stays within a row
    auroc_by_draw        donor swaps scored from ONE donor vs the MEAN of the donors (raw P(Yes))
    temporal_pairs       within-patient swaps per modality on the same stays: similarity and outcome pull
    reliance             each modality's SHARE of the total effect across the set's k modalities, per family and
                         measure, with a paired bootstrap CI over stays (the headline of the k-way analysis)
    pairwise_dominance   every modality pair a, b: (b − a) / (|a| + |b|); for a pair of signal + note this is the
                         notebooks' signal-vs-note dominance, number for number
    informativeness      AUROC of each "only modality m" input and of all of them, on the same stays
    information_shares   (AUROC_m − .5)+ / Σ_m (AUROC_m − .5)+, the information each modality carries on its own

For a two-modality set the note's reliance share equals the notebooks' (dominance + 1) / 2, since
((n − x) / (n + x) + 1) / 2 = n / (n + x).
"""
import itertools
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..modsets import MOD_ORDER, NOTE, ModalitySet, slot_label, slot_mods
from ..perturb import DONOR_FAMILIES, FAMILIES, REFERENCE
from .probe import RAW, Probe
from .stats import auc_rows, stratified_boot, wilcoxon

FRAME = ['level_id', 'readout', 'readout_family', 'clf']
LABEL_SPECIFIC = 'label-specific'
EFFECT_FAMILIES = FAMILIES + [LABEL_SPECIFIC]
GROUPS = [('died', 1), ('survived', 0)]
MEASURES = ['representation', '|risk change|', 'outcome pull']
N_BOOT_AUC = 500
N_BOOT_PAIRED = 2000
PALETTE = [('#2a78d6', 'o'), ('#eb6834', 's'), ('#1baf7a', 'D'), ('#8a5cc2', '^'), ('#b8860b', 'v'),
           ('#52514e', 'P'), ('#d6336c', 'X'), ('#0f8b8d', 'h')]


@dataclass
class AnalysisContext:
    model_name: str
    modset: ModalitySet
    cohorts: Dict                              # level -> TierCohort
    configs: pd.DataFrame                      # model-level configurations, no prompt factors
    raw_embeddings: List = field(default_factory=list)   # yes_probability, read raw
    note_blind: set = field(default_factory=set)
    out_dir: str = ''
    fig_dir: str = ''

    @property
    def mods(self):
        return self.modset.mods

    @property
    def slots(self) -> list:
        """Perturbed slots present, singles first, in modality order."""
        s = set(self.configs.slot[self.configs.family != REFERENCE])
        return sorted(s, key=lambda x: (len(slot_mods(x)), [MOD_ORDER.index(m) for m in slot_mods(x)]))

    def blind(self, readout, slot) -> bool:
        """A readout that cannot see the note does not react to a perturbation that involves it."""
        return readout in self.note_blind and NOTE in slot_mods(slot)

    def style(self, slot):
        """(label, colour, marker) of a slot, the same in every chart of this context."""
        i = self.slots.index(slot) if slot in self.slots else len(self.slots)
        colour, marker = PALETTE[i % len(PALETTE)]
        return f'{slot_label(slot)} perturbed', colour, marker


# ── shared helpers ───────────────────────────────────────────────────────────
def frames(scores: pd.DataFrame):
    return scores.groupby(FRAME, sort=False)


def wide(g: pd.DataFrame, value='p') -> pd.DataFrame:
    """One frame as stay x config."""
    return g.pivot_table(index='stay_id', columns='config', values=value, aggfunc='first', dropna=False)


def labels(g: pd.DataFrame, index) -> np.ndarray:
    return g.drop_duplicates('stay_id').set_index('stay_id').y.reindex(index).to_numpy()


def comparisons(g: pd.DataFrame):
    """(family, slot, order, alt config, base config) for every comparison in a frame: each perturbation against
    the reference, and label-specific = counterfactual against matched of the same slot."""
    meta = g.drop_duplicates('config').set_index('config')
    cfgs = [c for c in meta.index if c != 'ref']
    out = [(meta.family[c], meta.slot[c], int(meta.order[c]), c, 'ref') for c in cfgs]
    for c in cfgs:
        if meta.family[c] == 'counterfactual' and f'matched_{meta.slot[c]}' in meta.index:
            out.append((LABEL_SPECIFIC, meta.slot[c], int(meta.order[c]), c, f'matched_{meta.slot[c]}'))
    return out


# ── summary and the label-shuffle test ───────────────────────────────────────
def summarise(scores: pd.DataFrame, model: str = '') -> pd.DataFrame:
    gap = lambda y, x: x[y == 1].mean() - x[y == 0].mean()
    dauc = lambda y, a, b: roc_auc_score(y, a) - roc_auc_score(y, b)
    rows = []
    for key, g in frames(scores):
        P, S = wide(g), wide(g, 'shift')
        y_all = labels(g, P.index)
        for fam, slot, order, alt, base in comparisons(g):
            k = (P[alt].notna() & P[base].notna()).to_numpy()
            y = y_all[k]
            if k.sum() < 10 or len(np.unique(y)) < 2:
                continue
            a, b = P[alt].to_numpy()[k], P[base].to_numpy()[k]
            x = a - b
            glo, ghi = stratified_boot(gap, y, x)
            alo, ahi = stratified_boot(dauc, y, a, b)
            rows.append(dict(model=model, **dict(zip(FRAME, key)), family=fam, slot=slot, order=order,
                             n=int(k.sum()), died=int(y.sum()), prevalence=float(y.mean()),
                             delta_died=x[y == 1].mean(), delta_survived=x[y == 0].mean(),
                             gap=gap(y, x), gap_lo=glo, gap_hi=ghi,
                             p_died=wilcoxon(x[y == 1]), p_survived=wilcoxon(x[y == 0]),
                             auroc_ref=roc_auc_score(y, b), auroc_alt=roc_auc_score(y, a),
                             d_auroc=dauc(y, a, b), d_auroc_lo=alo, d_auroc_hi=ahi,
                             rep_shift_median=float(np.nanmedian(S[alt].to_numpy()[k]))))
    return pd.DataFrame(rows)


def label_shuffle_test(probe: Probe, cohort, configs, emb, n_perm=10) -> pd.DataFrame:
    """Labels permuted before the probe is fitted and before decedents / survivors are split, one permutation per
    seed. Every gap must be ~0: passes when |mean| <= 3 SE across permutations."""
    per = {}
    for s in range(n_perm):
        g = probe.score(cohort, configs, emb, shuffle_seed=1000 + s)
        P = wide(g)
        y_all = labels(g, P.index)
        for fam, slot, _, alt, base in comparisons(g):
            k = (P[alt].notna() & P[base].notna()).to_numpy()
            y = y_all[k]
            if k.sum() >= 10 and len(np.unique(y)) == 2:
                x = (P[alt] - P[base]).to_numpy()[k]
                per.setdefault((fam, slot), []).append(x[y == 1].mean() - x[y == 0].mean())
    rows = []
    for (fam, slot), v in per.items():
        v = np.asarray(v)
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        rows.append(dict(level_id=cohort.level, readout=emb.name, clf=probe.name, family=fam, slot=slot,
                         perms=len(v), gap_mean=v.mean(), gap_se=se,
                         verdict='too few' if len(v) < 5 else
                                 ('pass' if abs(v.mean()) <= 3 * se else 'FAIL')))     # <=: an all-zero family passes
    return pd.DataFrame(rows)


def clean_summary(ctx: AnalysisContext, summary: pd.DataFrame) -> pd.DataFrame:
    """d_auroc blanked on donor rows (donors are chosen BY outcome) and rows a readout cannot react to dropped."""
    s = summary.copy()
    donor = s.family.isin(DONOR_FAMILIES + [LABEL_SPECIFIC])
    s.loc[donor, ['auroc_alt', 'd_auroc', 'd_auroc_lo', 'd_auroc_hi']] = np.nan
    return s[[not ctx.blind(r, sl) for r, sl in zip(s.readout, s.slot)]] if len(s) else s


# ── shifts and AUROC ─────────────────────────────────────────────────────────
def reference_shifts(ctx: AnalysisContext, scores: pd.DataFrame) -> pd.DataFrame:
    """Per (tier, readout, clf, family, slot, outcome group): mean reference and perturbed risk, and the paired Δ
    with its 95% CI (1.96 SE over patients)."""
    rows = []
    for key, g in frames(scores):
        ro = key[1]
        P = wide(g)
        y = labels(g, P.index)
        for fam, slot, order, alt, base in comparisons(g):
            if ctx.blind(ro, slot):
                continue
            for grp, yv in GROUPS:
                k = (y == yv) & P[alt].notna().to_numpy() & P[base].notna().to_numpy()
                if k.sum() < 10:
                    continue
                a, b = P[alt].to_numpy()[k], P[base].to_numpy()[k]
                delta = a - b
                rows.append(dict(level_id=key[0], readout=ro, clf=key[3], family=fam, slot=slot, order=order,
                                 group=grp, n=int(k.sum()), ref=b.mean(), swapped=a.mean(), delta=delta.mean(),
                                 delta_ci=1.96 * delta.std(ddof=1) / np.sqrt(k.sum())))
    return pd.DataFrame(rows)


def auroc_shifts(ctx: AnalysisContext, scores: pd.DataFrame, n_boot=N_BOOT_AUC, seed=0) -> pd.DataFrame:
    """AUROC at the reference and after each perturbation. Within a (family, order) row every slot is scored on
    the same stays (those with all of them), with a shared bootstrap."""
    rng, rows = np.random.default_rng(seed), []
    for key, g in frames(scores):
        ro = key[1]
        P = wide(g)
        y_all = labels(g, P.index)
        meta = g[g.family != REFERENCE].drop_duplicates('config')
        for fam in FAMILIES:
            for order in sorted(meta.order[meta.family == fam].unique()):
                m = meta[(meta.family == fam) & (meta.order == order)]
                cols = [c for c, sl in zip(m.config, m.slot) if not ctx.blind(ro, sl)]
                if not cols or 'ref' not in P:
                    continue
                k = P[['ref'] + cols].notna().all(1).to_numpy()
                y = y_all[k].astype(int)
                if k.sum() < 20 or y.min() == y.max():
                    continue
                sc = np.stack([P.ref.to_numpy()[k]] + [P[c].to_numpy()[k] for c in cols])        # (1+S, n)
                idx = rng.integers(0, k.sum(), (n_boot, k.sum()))
                yb = y[idx]
                ok = (yb.sum(1) > 0) & (yb.sum(1) < k.sum())
                auc = auc_rows(sc, np.broadcast_to(y, sc.shape))
                boot = np.stack([auc_rows(s[idx][ok], yb[ok]) for s in sc])                  # (1+S, B)
                for j, c in enumerate(cols, start=1):
                    dlt = boot[j] - boot[0]
                    rows.append(dict(level_id=key[0], readout=ro, clf=key[3], family=fam,
                                     slot=m.slot[m.config == c].iloc[0], order=int(order), n=int(k.sum()),
                                     died=int(y.sum()), auroc_ref=auc[0], auroc_swap=auc[j],
                                     swap_lo=np.percentile(boot[j], 2.5), swap_hi=np.percentile(boot[j], 97.5),
                                     d_auroc=auc[j] - auc[0], d_lo=np.percentile(dlt, 2.5),
                                     d_hi=np.percentile(dlt, 97.5)))
    return pd.DataFrame(rows)


def auroc_by_draw(ctx: AnalysisContext) -> pd.DataFrame:
    """Donor swaps: ΔAUROC from ONE donor vs from the MEAN of the donors (raw P(Yes)). Averaging donors pulls
    every score toward its outcome group's mean, so 'mean of draws' inflates matched AUROC by construction."""
    rows = []
    for emb in ctx.raw_embeddings:
        for lv, c in ctx.cohorts.items():
            cfg = ctx.configs[(ctx.configs.level_id == lv)
                              & (ctx.configs.family.isin(DONOR_FAMILIES) | (ctx.configs.config == 'ref'))]
            cfg = cfg.assign(p=emb.matrix(list(cfg.record))[:, 0]).dropna(subset=['p'])
            y = c.table.set_index('stay_id').y
            ref = cfg[cfg.config == 'ref'].groupby('stay_id').p.mean()
            for conf, sub in cfg[cfg.config != 'ref'].groupby('config', sort=False):
                per = [(f'draw {int(k)}', sub[sub.draw == k].groupby('stay_id').p.mean())
                       for k in sorted(sub.draw.unique())]
                for name, s in per + [('mean of draws', sub.groupby('stay_id').p.mean())]:
                    ids = s.index.intersection(ref.index)
                    if len(ids) < 20 or y[ids].nunique() < 2:
                        continue
                    rows.append(dict(readout=emb.name, level_id=lv, family=sub.family.iloc[0], slot=sub.slot.iloc[0],
                                     scored=name, n=len(ids), auroc_ref=roc_auc_score(y[ids], ref[ids]),
                                     auroc_swap=roc_auc_score(y[ids], s[ids])))
    out = pd.DataFrame(rows)
    if len(out):
        out['d_auroc'] = out.auroc_swap - out.auroc_ref
    return out


# ── within-patient (temporal) swaps per modality ─────────────────────────────
def temporal_pairs(ctx: AnalysisContext, scores: pd.DataFrame, clf='Logistic regression') -> pd.DataFrame:
    """Within-patient swap of each single modality, on the SAME stays (those with a follow-up record for every
    modality): sim = 1 − representation shift, pull = Δ risk toward the true outcome. Vector readouts only."""
    raw = {e.name for e in ctx.raw_embeddings}
    out = []
    for key, g in frames(scores):
        ro = key[1]
        if key[3] != clf or ro in raw or ro in ctx.note_blind:
            continue
        cols = [f'temporal_{m}' for m in ctx.mods]
        D, S = wide(g, 'delta'), wide(g, 'shift')
        if not all(c in D for c in cols):
            continue
        k = D[cols].notna().all(1).to_numpy() & S[cols].notna().all(1).to_numpy()
        y = labels(g, D.index)[k]
        sign = np.where(y == 1, 1.0, -1.0)
        for m, c in zip(ctx.mods, cols):
            out.append(pd.DataFrame(dict(level_id=key[0], readout=ro, stay_id=D.index[k], y=y, mod=m,
                                         sim=1 - S[c].to_numpy()[k], pull=sign * D[c].to_numpy()[k])))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def temporal_summary(tp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lv, ro, m), g in tp.groupby(['level_id', 'readout', 'mod']):
        rows.append(dict(level_id=lv, readout=ro, mod=m, n_paired=len(g), died=int(g.y.sum()), sim=g.sim.mean(),
                         pull=g.pull.mean(), pull_ci=1.96 * g.pull.std() / np.sqrt(len(g)), p_pull=wilcoxon(g.pull)))
    return pd.DataFrame(rows)


def binned(x, y, bins=5):
    g = pd.DataFrame(dict(x=x, y=y)).dropna()
    g['b'] = pd.qcut(g.x, bins, labels=False, duplicates='drop')
    b = g.groupby('b').agg(x=('x', 'median'), m=('y', 'mean'), sd=('y', 'std'), n=('y', 'size'))
    b['ci'] = 1.96 * b.sd / np.sqrt(b.n)
    return b


# ── reliance (k-way) and pairwise dominance ──────────────────────────────────
def _measures(ctx: AnalysisContext, g: pd.DataFrame, fam: str):
    """(measure, stays x k matrix, column order = ctx.mods) for one family's single-modality perturbations."""
    D, S = wide(g, 'delta'), wide(g, 'shift')
    true_dir = np.where(labels(g, D.index) == 1, 1.0, -1.0)[:, None]
    col = lambda f, m: f'{f}_{m}'
    if fam == LABEL_SPECIFIC:
        if not all(col(f, m) in D for f in ('counterfactual', 'matched') for m in ctx.mods):
            return []
        cx = np.stack([(D[col('counterfactual', m)] - D[col('matched', m)]).to_numpy() for m in ctx.mods], 1)
        return [('|risk change|', np.abs(cx)), ('outcome pull', -true_dir * cx)]
    if not all(col(fam, m) in D for m in ctx.mods):
        return []
    d = np.stack([D[col(fam, m)].to_numpy() for m in ctx.mods], 1)
    out = [('representation', np.stack([S[col(fam, m)].to_numpy() for m in ctx.mods], 1)), ('|risk change|', np.abs(d))]
    if fam == 'temporal':
        out.append(('outcome pull', true_dir * d))
    elif fam == 'counterfactual':
        out.append(('outcome pull', -true_dir * d))
    return out


def reliance(ctx: AnalysisContext, scores: pd.DataFrame, seed=0):
    """(reliance, pairwise_dominance). Within a family, every single-modality perturbation on the same recipients
    (and donors), same probe. reliance: each modality's share of the summed mean effect (nonnegative measures;
    signed outcome pull gets its mean only), paired bootstrap CI. pairwise: for every pair a, b in modality order,
    dominance = (b − a) / (|a| + |b|) in [−1, 1]."""
    rng = np.random.default_rng(seed)
    dom = lambda a, b: (b - a) / np.maximum(np.abs(a) + np.abs(b), 1e-12)
    mods = list(ctx.mods)
    rel, pw = [], []
    for key, g in frames(scores):
        if key[1] in ctx.note_blind:
            continue
        base = dict(zip(FRAME, key))
        for fam in EFFECT_FAMILIES:
            for measure, V in _measures(ctx, g, fam):
                k = np.isfinite(V).all(1)
                if k.sum() < 20:
                    continue
                V = V[k]
                idx = rng.integers(0, len(V), (N_BOOT_PAIRED, len(V)))
                B = V[idx].mean(1)                                            # (N_BOOT, k)
                mean = V.mean(0)
                share_ok = measure != 'outcome pull'
                share = mean / mean.sum() if share_ok and mean.sum() > 0 else np.full(len(mods), np.nan)
                Bs = B / np.maximum(B.sum(1, keepdims=True), 1e-12) if share_ok else np.full_like(B, np.nan)
                for j, m in enumerate(mods):
                    rel.append(dict(**base, family=fam, measure=measure, mod=m, n_paired=int(k.sum()),
                                    mean=mean[j], mean_lo=np.percentile(B[:, j], 2.5),
                                    mean_hi=np.percentile(B[:, j], 97.5), share=share[j],
                                    share_lo=np.nanpercentile(Bs[:, j], 2.5) if share_ok else np.nan,
                                    share_hi=np.nanpercentile(Bs[:, j], 97.5) if share_ok else np.nan))
                for i, j in itertools.combinations(range(len(mods)), 2):
                    a, b, ba, bb = V[:, i], V[:, j], B[:, i], B[:, j]
                    diff = b - a
                    pw.append(dict(**base, family=fam, measure=measure, mod_a=mods[i], mod_b=mods[j],
                                   n_paired=int(k.sum()), value_a=a.mean(), value_b=b.mean(), diff=diff.mean(),
                                   diff_lo=np.percentile(bb - ba, 2.5), diff_hi=np.percentile(bb - ba, 97.5),
                                   p_wilcoxon=wilcoxon(diff), b_wins=(diff > 0).sum() / max((diff != 0).sum(), 1),
                                   dominance=dom(a.mean(), b.mean()), dominance_lo=np.percentile(dom(ba, bb), 2.5),
                                   dominance_hi=np.percentile(dom(ba, bb), 97.5)))
    return pd.DataFrame(rel), pd.DataFrame(pw)


# ── informativeness: what each modality carries on its own ───────────────────
def arms(configs: pd.DataFrame, mods) -> Dict[str, str]:
    """arm -> config: 'all' = the reference; m = the core ablation that keeps only modality m."""
    out = {'all': 'ref'}
    core = configs[(configs.family == 'ablate') & configs.core].drop_duplicates('config')
    for c, slot in zip(core.config, core.slot):
        kept = [m for m in mods if m not in slot_mods(slot)]
        if len(kept) == 1:
            out[kept[0]] = c
    return out


def informativeness(cohorts: Dict, configs: pd.DataFrame, mods, embeddings, clfs):
    """(arm AUROC, per-stay scores). AUROC of each "only modality m" input and of all modalities, on the same
    stays. P(Yes): the model's own score, no probe. Vector readouts: a probe trained AND tested on that arm, with
    the tier's fixed folds. The per-stay table lets the zoo recompute AUROC on any subset of stays."""
    per = []
    arm_cfg = arms(configs, mods)
    for lv, c in cohorts.items():
        t = c.table
        cfg = configs[configs.level_id == lv]
        y = t.y.to_numpy()
        for emb in embeddings:
            Xs = {}
            for arm, conf in arm_cfg.items():
                rec = cfg[cfg.config == conf].drop_duplicates('stay_id').set_index('stay_id').record.reindex(t.stay_id)
                Xs[arm] = emb.matrix(list(rec.fillna('')))
            ok = np.all([~np.isnan(X).any(1) for X in Xs.values()], axis=0)     # same stays for every arm
            if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
                continue
            for arm, X in Xs.items():
                base = dict(level_id=lv, readout=emb.name, key=emb.key, arm=arm)
                if emb.raw:
                    per.append(pd.DataFrame(dict(**base, clf=RAW, stay_id=t.stay_id[ok], y=y[ok], p=X[ok, 0])))
                    continue
                for cn, clf in clfs.items():
                    p = np.full(len(t), np.nan)
                    for k in sorted(t.fold.unique()):
                        tr, te = ok & (t.fold != k).to_numpy(), ok & (t.fold == k).to_numpy()
                        if len(np.unique(y[tr])) < 2 or not te.any():
                            continue
                        pr = Probe(clf, cn).pipeline(emb.pca, tr.sum(), X.shape[1]).fit(X[tr], y[tr])
                        p[te] = pr.predict_proba(X[te])[:, 1]
                    k = ~np.isnan(p)
                    per.append(pd.DataFrame(dict(**base, clf=cn, stay_id=t.stay_id[k], y=y[k], p=p[k])))
    per = pd.concat(per, ignore_index=True) if per else pd.DataFrame(
        columns=['level_id', 'readout', 'key', 'arm', 'clf', 'stay_id', 'y', 'p'])
    return arm_auroc(per), per


def arm_auroc(per: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lv, ro, key, clf, arm), g in per.groupby(['level_id', 'readout', 'key', 'clf', 'arm'], sort=False):
        if g.y.nunique() == 2:
            rows.append(dict(level_id=lv, readout=ro, key=key, clf=clf, arm=arm, n=len(g),
                             auroc=roc_auc_score(g.y, g.p)))
    return pd.DataFrame(rows)


def information_shares(info: pd.DataFrame, mods) -> pd.DataFrame:
    """Per (tier, readout, clf, modality): AUROC alone, AUROC with everything, and the information share."""
    rows = []
    for (lv, ro, key, clf), g in info.groupby(['level_id', 'readout', 'key', 'clf'], sort=False):
        a = g.set_index('arm').auroc
        if not set(mods) <= set(a.index):
            continue
        gain = {m: max(a[m] - 0.5, 0.0) for m in mods}
        total = sum(gain.values())
        for m in mods:
            rows.append(dict(level_id=lv, readout=ro, key=key, clf=clf, mod=m, n=int(g.n.max()),
                             auroc_only=a[m], auroc_all=a.get('all'),
                             info_share=gain[m] / total if total > 0 else np.nan))
    return pd.DataFrame(rows)
