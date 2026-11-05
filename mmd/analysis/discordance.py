"""Discordance: how much each modality can mislead a model, against how much a fair combination of the model's own
single-modality evidence would be misled.

Four arms, scored on the SAME stays (paired), so differences between arms are never differences between patients:

    arm              the input                                        in the manifest
    reference        every modality the stay's own                     config 'ref'
    without m        modality m removed                                ablate, slot m
    matched m        m from a same-outcome donor                       matched, slot m        (one AUROC per draw)
    counterfactual m m from an opposite-outcome donor (conflicting)    counterfactual, slot m (one AUROC per draw)

plus, for the benchmark, each stay's (and each donor's) "only m" input (the core ablation keeping only m).

Two tracks, each reduced to the same metrics:

    model track       the model's score for each arm's input
    benchmark track   late fusion of the model's OWN single-modality evidence: one calibrated probe per modality,
                      trained on the "only m" inputs, gives log-odds lo_m; the benchmark score is their average.
                      Arms: reference = mean_m lo_m(own); without m = mean over the other modalities;
                      matched / counterfactual m = the DONOR's lo_m averaged with the stay's own other modalities.
                      Each single-modality probe is Platt-calibrated (inner cross-validation), so the log-odds
                      being averaged are on one scale whatever the classifier.
                      It is what the model would do under each perturbation if it simply added up what it reads
                      from each modality alone ("expected").

A frame is one Readout (what is scored, with which classifier):

    answer            the model's own answer, yes_log_odds (else logit P(Yes)). The model track uses it as is; the
                      benchmark's per-modality "probe" is a one-variable logistic calibration of the only-m answer.
    probes            every vector readout x every classifier mmd uses (probe.default_classifiers: logistic
                      regression, and XGBoost where it is installed). The model track is a probe fitted on the
                      reference inputs and applied unchanged to every arm (as probe.Probe); the benchmark's
                      per-modality probe is the same classifier fitted on the only-m inputs.

Every probe is cross-fitted on the manifest's folds: fitted on the other folds, it scores the held-out fold's stays
and their donors (donors come from the recipient's own fold, so no probe has seen a stay it scores). Donor arms are
scored PER DRAW and their AUROC is the mean over draws of one AUROC per draw: averaging the scores over donors first
would inflate the matched AUROC and deflate the counterfactual one (see effects.auroc_by_draw).

Per modality m, on each track (capture = AUROC(reference) − 0.5):

    necessity                 (AUROC(reference) − AUROC(without m)) / capture        lost when m is absent
    discordance sensitivity   (AUROC(reference) − AUROC(counterfactual m)) / capture lost when m conflicts
    substitution effect       AUROC(reference) − AUROC(matched m)                    negative control, ≈ 0
    discordance gap           discordance sensitivity − necessity                    descriptive
    need share / conflict share   necessity / Σ, discordance sensitivity / Σ         (negatives clipped to 0)

and between the tracks (expected_* = the benchmark's value):

    excess necessity                  necessity − expected necessity
    excess discordance sensitivity    discordance sensitivity − expected discordance sensitivity   (headline:
                                      > 0 m misleads the model more than its own evidence from m justifies)
    modality bias score               conflict share − expected conflict share (Σ = 0)

Per frame: total modality bias = ½ Σ |modality bias score|, the dominant modality, `low_capture` (capture <
MIN_CAPTURE: every ratio divides by capture) and `benchmark_valid` (the benchmark's reference AUROC reaches the best
single modality's: a fair combination should do at least as well as its best part; expected and excess values of an
invalid benchmark are reported but should not be read).

Demographic baseline: per frame, the AUROC of age + sex alone (DEMOGRAPHICS), cross-fitted on the same folds and
the same paired stays with the frame's classifier (logistic regression for the answer): what any model should beat.
`auroc_over_demographics` = AUROC(reference) − demographic AUROC. Donors share the recipient's age group and sex, so
the perturbations leave the demographic baseline unchanged.

The analysis only READS the store (ModelData): extra records there (other strategies, prompt variants, a later dose
run) are ignored. Every row carries `dose` (100 = the whole modality swapped); a later dose-response run adds rows.

    arm_configs -> ModelData -> shared_cohort -> analyse -> frame -> (model_scores, benchmark_scores) -> metrics
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import fsio
from ..modsets import NOTE, slot_mods
from ..perturb import REFERENCE

SCORE_KEYS = ('yes_log_odds', 'yes_probability')
ARMS = ('reference', 'without', 'matched', 'counterfactual')
DONOR_ARMS = ('matched', 'counterfactual')
DEMOGRAPHICS = ('age', 'sex_num')
FULL_DOSE = 100
MIN_STAYS = 20
MIN_CAPTURE = 0.05
VALID_TOL = 0.005
EPS = 1e-6


# ── arms: which stored records each arm reads ───────────────────────────────
def arm_configs(configs: pd.DataFrame, mods, levels: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """One row per (level, stay, arm, mod, draw): its record, and for donor arms the donor and the donor's own
    "only m" record (`donor_only`). Arm 'only' holds each stay's own "only m" record (the benchmark's input).
    configs: Manifest.analysis_configs() (model-pass draws, no prompt factors)."""
    mods = list(mods)
    c = configs if levels is None else configs[configs.level_id.isin(set(levels))]
    single = (c.order == 1) & c.slot.isin(set(mods))
    keep_only = (c.family == 'ablate') & (c.order == len(mods) - 1)
    only = c[keep_only].assign(mod=[next(m for m in mods if m not in slot_mods(s)) for s in c.slot[keep_only]])
    parts = [c[c.family == REFERENCE].assign(arm='reference', mod='', draw=0),
             c[single & (c.family == 'ablate')].assign(arm='without', mod=lambda d: d.slot, draw=0),
             only.assign(arm='only', draw=0)]
    parts += [c[single & (c.family == arm)].assign(arm=arm, mod=lambda d: d.slot) for arm in DONOR_ARMS]
    out = pd.concat(parts, ignore_index=True)
    out = out[['level_id', 'stay_id', 'arm', 'mod', 'draw', 'donor', 'record']].drop_duplicates(
        ['level_id', 'stay_id', 'arm', 'mod', 'draw'])
    own_only = out[out.arm == 'only'].set_index(['level_id', 'stay_id', 'mod']).record
    key = list(zip(out.level_id, out.donor.fillna(-1).astype(int), out['mod']))
    out['donor_only'] = [own_only.get(k) if a in DONOR_ARMS else None for k, a in zip(key, out.arm)]
    return out.assign(draw=out.draw.astype(int), dose=FULL_DOSE).reset_index(drop=True)


def needed_records(arms: pd.DataFrame) -> set:
    return set(arms.record) | set(arms.donor_only.dropna())


def complete(arms: pd.DataFrame, have: Callable[[pd.Series], np.ndarray]) -> pd.DataFrame:
    """Per level, the stays with EVERY arm, modality and draw available, their donors' "only m" records included
    (the paired stays). have(records) -> bool array."""
    ok = have(arms.record)
    donor = arms.arm.isin(DONOR_ARMS).to_numpy()
    ok[donor] &= have(arms.donor_only[donor])
    cell = arms.arm + '|' + arms['mod'] + '|' + arms.draw.astype(str)
    out = []
    for lv, g in arms.assign(cell=cell, ok=ok).groupby('level_id', sort=False):
        have_all = g[g.ok].groupby('stay_id').cell.nunique()
        out.append(g[g.stay_id.isin(set(have_all.index[have_all == g.cell.nunique()]))])
    t = pd.concat(out, ignore_index=True) if out else arms.iloc[:0].assign(cell='', ok=True)
    return t.drop(columns=['cell', 'ok'])


def restrict(arms: pd.DataFrame, stays: Dict[str, set]) -> pd.DataFrame:
    """The arms of the given stays (level -> stay ids) only."""
    keep = {(lv, s) for lv, v in stays.items() for s in v}
    return arms[[k in keep for k in zip(arms.level_id, arms.stay_id)]].reset_index(drop=True)


# ── scores: what the store holds ─────────────────────────────────────────────
def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def load_scores(store, records, chunks=None, workers=16) -> Optional[dict]:
    """{'record', 'yes_log_odds', 'yes_probability'} for `records`, NaN where a chunk lacks a key. Read chunk by
    chunk: EmbeddingStore.load would misalign a key that only some chunks hold (runs from before yes_log_odds)."""
    want = set(records)
    names = {k: store.stored_names(k) for k in SCORE_KEYS}
    fetch = [n for v in names.values() for n in v] + ['record']

    def read(p):
        z = fsio.load_npz(p, keys=fetch)
        rec = z['record']
        sel = np.array([r in want for r in rec])
        if not sel.any():
            return None
        out = dict(record=rec[sel])
        for k, alts in names.items():
            got = next((z[a] for a in alts if a in z), None)
            out[k] = (np.asarray(got, float).reshape(len(rec), -1)[sel, 0] if got is not None
                      else np.full(sel.sum(), np.nan))
        return out
    paths = list(chunks) if chunks is not None else store.paths()
    with ThreadPoolExecutor(workers) as ex:
        parts = [p for p in ex.map(read, paths) if p is not None]
    if not parts:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in ('record',) + SCORE_KEYS}


def score_of(run: dict) -> tuple:
    """(score per record, share of records scored by yes_log_odds). yes_log_odds where stored, else the logit of
    yes_probability = P(Yes) / (P(Yes) + P(No)): the same quantity, except where P(Yes) rounds to 1.0."""
    n = len(run['record'])
    lo = np.asarray(run.get('yes_log_odds', np.full(n, np.nan)), float).ravel()
    p = np.asarray(run.get('yes_probability', np.full(n, np.nan)), float).ravel()
    has = np.isfinite(lo)
    return np.where(has, lo, _logit(p)), float(has.mean()) if n else 0.0


class Matrix:
    """record -> feature row. From the answer scores (one column) or a readout's stored arrays."""

    def __init__(self, records, values):
        values = np.asarray(values, np.float32)
        self.values = values.reshape(len(values), -1)
        self.index = pd.Series(np.arange(len(records)), index=pd.Index(records)).groupby(level=0).last()
        self.finite = np.isfinite(self.values).all(1)

    def have(self, records) -> np.ndarray:
        i = self.index.reindex(pd.Index(records)).to_numpy()
        ok = ~np.isnan(i)
        ok[ok] = self.finite[i[ok].astype(int)]
        return ok

    def __call__(self, records) -> np.ndarray:
        return self.values[self.index.reindex(pd.Index(records)).to_numpy().astype(int)]


@dataclass(frozen=True)
class Tier:
    """One tier's stays: outcome, fold and demographics, each indexed by stay_id."""
    labels: pd.Series
    folds: pd.Series
    demo: pd.DataFrame

    @classmethod
    def of(cls, cohort) -> 'Tier':
        t = cohort.table.set_index('stay_id')
        return cls(t.y, t.fold, t[list(DEMOGRAPHICS)])


@dataclass
class ModelData:
    """One model's stored data for one modality set: the arms, and read-only, lazy (once) access to the store."""
    name: str
    adapter: type
    modset: object                          # modsets.ModalitySet
    manifest: object                        # cohort.Manifest
    store: object                           # store.EmbeddingStore (only read)
    arms: pd.DataFrame

    @property
    def levels(self) -> List[str]:
        return sorted(set(self.arms.level_id))

    @property
    def tiers(self) -> Dict[str, Tier]:
        return {lv: Tier.of(self.manifest.cohorts[lv]) for lv in self.levels}

    @cached_property
    def records(self) -> List[str]:
        return sorted(needed_records(self.arms))

    @cached_property
    def chunks(self):
        return self.store.chunks_for(self.store.load_index(), self.records)

    @cached_property
    def _answer(self) -> tuple:
        run = load_scores(self.store, self.records, self.chunks)
        if run is None:
            return None, np.nan
        score, share = score_of(run)
        return Matrix(run['record'], score), share

    @property
    def answer(self) -> Optional[Matrix]:
        """The model's own answer (log-odds) for every record the arms read."""
        return self._answer[0]

    @property
    def log_odds_share(self) -> float:
        return self._answer[1]

    def complete_stays(self) -> Dict[str, set]:
        """Per level, the stays with every arm, draw and donor input stored (judged on the answer)."""
        if self.answer is None:
            return {}
        return {lv: set(g.stay_id) for lv, g in complete(self.arms, self.answer.have).groupby('level_id')}

    def readout(self, spec, records) -> Optional[Matrix]:
        """A vector readout for `records`; None when it is not stored for every record, or has no value once the
        signal is removed (a signal-only readout: checked on a few records before the full read)."""
        ms = self.modset
        if NOTE in ms.mods and ms.signals:
            probe = self.arms[(self.arms.arm == 'only') & (self.arms['mod'] == NOTE)].record.head(20)
            got = self.store.load(keys=[spec.key], records=set(probe),
                                  chunks=self.store.chunks_for(self.store.load_index(), probe))
            if got is not None and spec.key in got and not np.isfinite(got[spec.key]).any():
                print(f'  {spec.name}: no value once the signal is removed (a signal-only readout), skipped')
                return None
        run = self.store.load(keys=[spec.key], records=set(records), chunks=self.chunks)
        if run is None or spec.key not in run or len(run[spec.key]) != len(run['record']):
            print(f'  {spec.name}: not stored for every record of these arms, skipped')
            return None
        return Matrix(run['record'], run[spec.key].reshape(len(run['record']), -1))


# ── readouts: what a frame scores, and with which classifier ─────────────────
def _calibration(n, d):
    return LogisticRegression(C=1e6, max_iter=1000)


def _demographic_lr(n, d):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def calibrated(make, cv=3):
    """A builder whose estimator is Platt-calibrated (sigmoid) by an inner cross-validation on its training data:
    the benchmark averages single-modality log-odds, which is only fair when each is calibrated (boosted trees
    often are not)."""
    from sklearn.calibration import CalibratedClassifierCV
    return lambda n, d: CalibratedClassifierCV(make(n, d), method='sigmoid', cv=cv)


@dataclass(frozen=True)
class Readout:
    """One frame: a readout and the classifier it is scored with. make(n_train, n_features) builds the probe; None
    = the model's own answer, read as is."""
    name: str
    family: str
    clf: str
    make: Optional[Callable] = None
    make_demo: Optional[Callable] = None

    @property
    def is_answer(self) -> bool:
        return self.make is None

    @property
    def unimodal(self) -> Callable:
        """The benchmark's single-modality probe builder."""
        return _calibration if self.is_answer else calibrated(self.make)

    @property
    def demographic(self) -> Callable:
        return self.make_demo or _demographic_lr

    @property
    def info(self) -> dict:
        return dict(readout=self.name, readout_family=self.family, clf=self.clf)

    @classmethod
    def probe(cls, spec, clf_name, clf, pca) -> 'Readout':
        from .probe import Probe
        return cls(spec.name, spec.family, clf_name,
                   make=lambda nt, nf: Probe(clf, clf_name).pipeline(pca, nt, nf),
                   make_demo=lambda nt, nf: Probe(clf, clf_name).pipeline(None, nt, nf))


ANSWER = Readout('answer', 'answer', 'no probe (answer log-odds)')


# ── the two tracks ───────────────────────────────────────────────────────────
def crossfit(fit_rec, fit_y, fit_fold, pred_rec, pred_fold, X: Matrix, make) -> np.ndarray:
    """Log-odds for pred_rec from a model fitted on the other folds' fit_rec (fold k is scored by the fit without
    fold k). make(n_train, n_features) -> an unfitted sklearn estimator with predict_proba."""
    fit_rec, pred_rec = np.asarray(fit_rec), np.asarray(pred_rec)
    out = np.full(len(pred_rec), np.nan)
    for k in np.unique(pred_fold):
        tr, te = fit_fold != k, pred_fold == k
        if len(np.unique(fit_y[tr])) < 2 or not te.any():
            continue
        Xtr = X(fit_rec[tr])
        est = make(len(Xtr), Xtr.shape[1]).fit(Xtr, fit_y[tr])
        out[te] = _logit(est.predict_proba(X(pred_rec[te]))[:, 1])
    return out


@dataclass
class LevelScores:
    """Both tracks' scores for one level's arms, and what came out of computing them."""
    scores: pd.DataFrame                    # one row per (track, arm, mod, draw, stay)
    best_unimodal_auroc: float
    demographic_auroc: float = np.nan


def model_scores(a: pd.DataFrame, X: Matrix, tier: Tier, readout: Readout) -> np.ndarray:
    """The model track: the answer as is, or a probe fitted on the reference inputs and applied to every arm."""
    if readout.is_answer:
        return X(a.record)[:, 0].astype(float)
    y, f = tier.labels.reindex(a.stay_id).to_numpy(), tier.folds.reindex(a.stay_id).to_numpy()
    ref = a[a.arm == 'reference']
    return crossfit(ref.record, y[ref.index], f[ref.index], a.record, f, X, readout.make)


def benchmark_scores(arms: pd.DataFrame, a: pd.DataFrame, X: Matrix, tier: Tier, readout: Readout) -> tuple:
    """(benchmark score per row of `a`, single-modality log-odds per stay {mod: Series}). Each modality's probe is
    fitted on the "only m" inputs and scores the stay's own and its donors' only-m inputs; the benchmark averages
    the log-odds, with the donor's in place of the stay's for the swapped modality."""
    mods = sorted(set(arms['mod']) - {''})
    k, f_row = len(mods), tier.folds.reindex(a.stay_id).to_numpy()
    own = arms[arms.arm == 'only'].set_index(['mod', 'stay_id']).record
    lo_own, lo_donor = {}, np.full(len(a), np.nan)
    for m in mods:
        o = own.loc[m]
        dm = a.index[a.arm.isin(DONOR_ARMS) & (a['mod'] == m)]
        fo = tier.folds.reindex(o.index).to_numpy()
        lo = crossfit(o.to_numpy(), tier.labels.reindex(o.index).to_numpy(), fo,
                      np.concatenate([o.to_numpy(), a.donor_only[dm].to_numpy()]),
                      np.concatenate([fo, f_row[dm]]), X, readout.unimodal)
        lo_own[m] = pd.Series(lo[:len(o)], index=o.index)
        lo_donor[dm] = lo[len(o):]
    own_mat = pd.DataFrame(lo_own).reindex(a.stay_id).to_numpy()            # rows of a x mods
    total = own_mat.sum(1)
    bench = np.where(a.arm == 'reference', total / k, np.nan)
    for j, m in enumerate(mods):
        w = ((a.arm == 'without') & (a['mod'] == m)).to_numpy()
        bench[w] = (total[w] - own_mat[w, j]) / (k - 1)
        d = (a.arm.isin(DONOR_ARMS) & (a['mod'] == m)).to_numpy()
        bench[d] = (total[d] - own_mat[d, j] + lo_donor[d]) / k
    return bench, lo_own


def level_scores(arms: pd.DataFrame, X: Matrix, tier: Tier, readout: Readout) -> LevelScores:
    """Both tracks for one level's complete arms, the best single modality's AUROC, and the demographic AUROC."""
    a = arms[arms.arm.isin(ARMS)].reset_index(drop=True)
    model = model_scores(a, X, tier, readout)
    bench, lo_own = benchmark_scores(arms, a, X, tier, readout)
    base = a[['level_id', 'stay_id', 'arm', 'mod', 'draw', 'dose']].assign(y=tier.labels.reindex(a.stay_id).to_numpy())
    uni = [roc_auc_score(tier.labels.reindex(s.index), s) for s in lo_own.values()
           if tier.labels.reindex(s.index).nunique() == 2 and s.notna().all()]
    scores = pd.concat([base.assign(track='model', score=model), base.assign(track='benchmark', score=bench)],
                       ignore_index=True)
    return LevelScores(scores, max(uni) if uni else np.nan,
                       demographic_auroc(tier, arms.stay_id, readout.demographic))


def demographic_auroc(tier: Tier, stays, make=_demographic_lr) -> float:
    """AUROC of the demographic columns alone on `stays`, cross-fitted on the folds."""
    stays = np.asarray(sorted(set(stays)))
    X = Matrix(stays, tier.demo.reindex(stays)[list(DEMOGRAPHICS)].to_numpy(float))
    stays = stays[X.have(stays)]
    y, f = tier.labels.reindex(stays).to_numpy(), tier.folds.reindex(stays).to_numpy()
    if len(stays) < MIN_STAYS or len(np.unique(y)) < 2:
        return np.nan
    lo = crossfit(stays, y, f, stays, f, X, make)
    good = np.isfinite(lo)
    return float(roc_auc_score(y[good], lo[good])) if len(np.unique(y[good])) == 2 else np.nan


# ── metrics ──────────────────────────────────────────────────────────────────
def arm_aurocs(scores: pd.DataFrame) -> pd.DataFrame:
    """One row per (level, dose, track, arm, mod): AUROC (donor arms: mean over draws of one AUROC per draw), the
    per-draw range, stays and deaths."""
    rows = []
    for (lv, dose, tr, arm, mod), g in scores.groupby(['level_id', 'dose', 'track', 'arm', 'mod'], sort=False):
        g = g.dropna(subset=['score'])
        per = [roc_auc_score(d.y, d.score) for _, d in g.groupby('draw') if d.y.nunique() == 2]
        if not per:
            continue
        rows.append(dict(level_id=lv, dose=int(dose), track=tr, arm=arm, mod=mod, auroc=float(np.mean(per)),
                         auroc_min=float(np.min(per)), auroc_max=float(np.max(per)), draws=len(per),
                         n=int(g.stay_id.nunique()), died=int(g.drop_duplicates('stay_id').y.sum())))
    return pd.DataFrame(rows)


def _share(v: pd.Series) -> pd.Series:
    v = v.clip(lower=0)
    return v / v.sum() if v.sum() > 0 else v * np.nan


def track_metrics(au: pd.DataFrame, mods) -> pd.DataFrame:
    """Per (level, dose, mod) metrics of ONE track's arm AUROCs."""
    mods = list(mods)
    out = []
    ref_all = au[au.arm == 'reference'].drop_duplicates('level_id').set_index('level_id')
    for (lv, dose), g in au[au.arm != 'reference'].groupby(['level_id', 'dose'], sort=False):
        if lv not in ref_all.index:
            continue
        ref = ref_all.loc[lv]
        a_ref, cap = ref.auroc, ref.auroc - 0.5
        w = g.pivot_table(index='mod', columns='arm', values='auroc').reindex(mods)
        if w.isna().any().any() or not set(ARMS[1:]) <= set(w.columns):
            continue
        d = pd.DataFrame(index=pd.Index(mods, name='mod'))
        d['auroc_reference'] = a_ref
        d['auroc_without'] = w['without']
        d['auroc_matched'] = w['matched']
        d['auroc_counterfactual'] = w['counterfactual']
        d['capture'] = cap
        d['necessity'] = (a_ref - w['without']) / cap
        d['discordance_sensitivity'] = (a_ref - w['counterfactual']) / cap
        d['discordance_sensitivity_vs_matched'] = (w['matched'] - w['counterfactual']) / cap
        d['substitution_effect'] = a_ref - w['matched']
        d['discordance_gap'] = d.discordance_sensitivity - d.necessity
        d['need_share'] = _share(d.necessity) if cap > 0 else np.nan
        d['conflict_share'] = _share(d.discordance_sensitivity) if cap > 0 else np.nan
        d['clipped'] = (d.necessity < 0) | (d.discordance_sensitivity < 0)
        rng = g[g.arm.isin(DONOR_ARMS)].pivot_table(index='mod', columns='arm', values=['auroc_min', 'auroc_max'])
        for arm in DONOR_ARMS:
            d[f'auroc_{arm}_min'] = rng[('auroc_min', arm)].reindex(mods)
            d[f'auroc_{arm}_max'] = rng[('auroc_max', arm)].reindex(mods)
        d['n'], d['died'] = int(ref.n), int(ref.died)
        out.append(d.reset_index().assign(level_id=lv, dose=int(dose)))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


EXPECTED = ['auroc_reference', 'auroc_without', 'auroc_matched', 'auroc_counterfactual', 'capture', 'necessity',
            'discordance_sensitivity', 'need_share', 'conflict_share']


def summarise(lv, dose, d: pd.DataFrame, best_unimodal: float, demographic: float) -> dict:
    """One frame's per-level summary from its per-modality rows `d`."""
    cap, a_ref, bref = d.capture.iloc[0], d.auroc_reference.iloc[0], d.expected_auroc_reference.iloc[0]
    bias = d.set_index('mod').modality_bias_score
    ok = bool(cap > 0 and bias.notna().all())
    return dict(level_id=lv, dose=int(dose), n=int(d.n.iloc[0]), died=int(d.died.iloc[0]),
                auroc_reference=a_ref, capture=cap, low_capture=bool(cap < MIN_CAPTURE),
                benchmark_auroc_reference=bref, best_unimodal_auroc=best_unimodal,
                benchmark_valid=bool(np.isfinite(bref) and np.isfinite(best_unimodal)
                                     and bref >= best_unimodal - VALID_TOL),
                interpretable=ok, total_modality_bias=0.5 * bias.abs().sum() if ok else np.nan,
                dominant_modality=bias.idxmax() if ok and bias.max() > 0 else '',
                max_substitution_effect=float(d.substitution_effect.abs().max()),
                demographic_auroc=demographic, auroc_over_demographics=a_ref - demographic)


def metrics(au: pd.DataFrame, mods, levels: Dict[str, LevelScores]) -> tuple:
    """(per modality, per frame) from both tracks' arm AUROCs of one frame."""
    mt = track_metrics(au[au.track == 'model'], mods)
    if mt.empty:
        return mt, pd.DataFrame()
    bt = track_metrics(au[au.track == 'benchmark'], mods)
    on = ['level_id', 'dose', 'mod']
    if bt.empty:
        bt = mt[on].assign(**{c: np.nan for c in EXPECTED})
    pm = mt.merge(bt[on + EXPECTED].rename(columns={c: f'expected_{c}' for c in EXPECTED}), on=on, how='left')
    pm['excess_necessity'] = pm.necessity - pm.expected_necessity
    pm['excess_discordance_sensitivity'] = pm.discordance_sensitivity - pm.expected_discordance_sensitivity
    pm['modality_bias_score'] = pm.conflict_share - pm.expected_conflict_share
    rows = [summarise(lv, dose, d, levels[lv].best_unimodal_auroc if lv in levels else np.nan,
                      levels[lv].demographic_auroc if lv in levels else np.nan)
            for (lv, dose), d in pm.groupby(['level_id', 'dose'], sort=False)]
    pm['benchmark_valid'] = pm.level_id.map({r['level_id']: r['benchmark_valid'] for r in rows})
    return pm[on + [c for c in pm.columns if c not in on]], pd.DataFrame(rows)


# ── results ──────────────────────────────────────────────────────────────────
@dataclass
class Result:
    """What an analysis produces: per-stay scores, arm AUROCs, per-modality and per-frame metrics."""
    scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    aurocs: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_mod: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_model: pd.DataFrame = field(default_factory=pd.DataFrame)

    TABLES = (('discordance', 'per_mod'), ('discordance_model', 'per_model'),
              ('discordance_aurocs', 'aurocs'), ('discordance_scores', 'scores'))
    CSV = ('per_mod', 'per_model', 'aurocs')                 # the per-stay scores are parquet only

    def __bool__(self):
        return len(self.per_model) > 0

    def tables(self) -> Dict[str, pd.DataFrame]:
        return dict(scores=self.scores, aurocs=self.aurocs, per_mod=self.per_mod, per_model=self.per_model)

    def assign(self, **cols) -> 'Result':
        return Result(**{k: v.assign(**cols) if len(v) else v for k, v in self.tables().items()})

    @classmethod
    def concat(cls, results: Iterable['Result']) -> 'Result':
        results = [r for r in results if r]
        stack = lambda k: (pd.concat([getattr(r, k) for r in results if len(getattr(r, k))], ignore_index=True)
                           if any(len(getattr(r, k)) for r in results) else pd.DataFrame())
        return cls(**{k: stack(k) for k in ('scores', 'aurocs', 'per_mod', 'per_model')})

    def save(self, folder: str):
        """Every non-empty table as parquet (and csv) in `folder`."""
        for name, key in self.TABLES:
            t = getattr(self, key)
            if len(t):
                fsio.write_parquet(t, f'{folder}/{name}.parquet')
                if key in self.CSV:
                    fsio.write_csv(t, f'{folder}/{name}.csv')


def saturation(scores: pd.DataFrame) -> pd.Series:
    """Per level: share of the model's reference scores tied at the top value (P(Yes) rounding to 1.0)."""
    r = scores[(scores.arm == 'reference') & (scores.track == 'model')]
    return r.groupby('level_id').score.apply(lambda s: float((s >= s.max()).mean()))


# ── analysis ─────────────────────────────────────────────────────────────────
def frame(arms: pd.DataFrame, X: Matrix, tiers: Dict[str, Tier], readout: Readout) -> Result:
    """Both tracks and all metrics for one readout, every level of `arms` (restricted to its complete stays)."""
    arms = complete(arms, X.have)
    levels = {lv: level_scores(g, X, tiers[lv], readout) for lv, g in arms.groupby('level_id', sort=False)
              if g.stay_id.nunique() >= MIN_STAYS}
    if not levels:
        return Result()
    scores = pd.concat([s.scores for s in levels.values()], ignore_index=True)
    au = arm_aurocs(scores)
    per_mod, per_model = metrics(au, sorted(set(arms['mod']) - {''}), levels)
    return Result(scores, au, per_mod, per_model).assign(**readout.info)


def shared_cohort(models: List[ModelData]) -> Dict[str, set]:
    """The one cohort every model of a modality set is analysed on: per level, the stays complete for EVERY model."""
    cohort = None
    for d in models:
        got = d.complete_stays()
        if not got:
            return {}
        print(f'  {d.name}: {({lv: len(v) for lv, v in got.items()})} stays with every input')
        cohort = got if cohort is None else {lv: cohort.get(lv, set()) & got.get(lv, set())
                                             for lv in set(cohort) | set(got)}
    return {lv: v for lv, v in (cohort or {}).items() if v}


def analyse(data: ModelData, stays: Dict[str, set] = None, readouts=None, pca_dim: int = 512) -> Result:
    """Every frame of one model: its answer, and every vector readout x classifier (probe.default_classifiers).
    readouts: names or keys to keep (default: all). stays: level -> the stays to use (shared_cohort); default: every
    stay with all inputs."""
    from .probe import RAW_KEY, default_classifiers
    arms, tiers = (data.arms if stays is None else restrict(data.arms, stays)), data.tiers
    want = lambda *names: readouts is None or any(n in readouts for n in names)
    results = []
    if want(ANSWER.name) and data.answer is not None:
        r = frame(arms, data.answer, tiers, ANSWER)
        if r:
            r.per_model = r.per_model.assign(log_odds_share=data.log_odds_share,
                                             saturated=r.per_model.level_id.map(saturation(r.scores)))
            results.append(r)
            print(f'  answer: {r.per_model.n.tolist()} stays per level')
    clfs = default_classifiers()
    for spec in data.adapter.readouts:
        if spec.key == RAW_KEY or not want(spec.name, spec.key):
            continue
        X = data.readout(spec, sorted(needed_records(arms)))
        if X is None:
            continue
        for name, clf in clfs.items():
            r = frame(arms, X, tiers, Readout.probe(spec, name, clf, pca_dim if spec.pca == 'pca' else spec.pca))
            print(f'  {spec.name} · {name}: ' + (f'{r.per_model.n.tolist()} stays per level' if r else
                                                 'too few stays with every arm, skipped'))
            results.append(r)
    res = Result.concat(results).assign(model=data.name, modset=data.modset.name, mods='+'.join(data.modset.mods))
    if res and stays is not None:
        size = {lv: len(v) for lv, v in stays.items()}
        res.per_model = res.per_model.assign(cohort_n=res.per_model.level_id.map(size))
        res.per_model['cohort_complete'] = res.per_model.n == res.per_model.cohort_n
    return res
