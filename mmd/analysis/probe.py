"""Readouts and probe transfer over the four families (ported unchanged from the notebooks).

A Readout turns record ids into a feature matrix (NaN rows where a record has no value):
  StoreReadout   arrays saved by the model passes, keyed by the full configuration id
                 (input-level: every swap was its own forward pass)
  LookupReadout  a unimodal pre-fusion vector looked up by the record actually in the slot
                 (feature-level: free, so it can use all 20 donor draws)
The probe is fitted ONCE per fold on the reference rows of the training folds, then scores every
configuration of the test-fold stays unchanged. Δ = p(config) - p(ref), averaged over draws.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..cohort import SwapCohort, parse


def default_classifiers():
    clfs = {"Logistic regression": LogisticRegression(C=0.1, max_iter=2000)}
    try:
        from xgboost import XGBClassifier
        clfs["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                        n_jobs=4, random_state=0, eval_metric="logloss", tree_method="hist")
    except Exception as e:              # not installed, or no OpenMP runtime (macOS: brew install libomp)
        print(f"xgboost unavailable ({type(e).__name__}) - logistic regression only")
    return clfs


@dataclass
class Readout:
    name: str
    family: str                                 # pre-fusion | projector | cross-attention | post-fusion
    X_of: Callable[[Sequence[str]], np.ndarray]
    pca: Optional[int] = None
    mod: Optional[str] = None                   # set for a unimodal lookup: only that slot's swaps move it
    key: Optional[str] = None                   # the stored key (common name), for the zoo table


def StoreReadout(name, family, run, key, fn=None, pca=None):
    """run: ChunkStore.load(...) output. fn reshapes one array (default: flatten per record)."""
    fn = fn or (lambda a: a.reshape(len(a), -1))
    idx = {r: i for i, r in enumerate(run['record'])} if run is not None else {}
    width = fn(run[key][:1]).shape[1] if run is not None else 1

    def X_of(records):
        X = np.full((len(records), width), np.nan, dtype=np.float32)
        rows = [(i, idx[r]) for i, r in enumerate(records) if r in idx]
        if rows:
            a, b = map(np.array, zip(*rows))
            X[a] = fn(run[key][b])
        return X
    return Readout(name, family, X_of, pca, key=key)


def LookupReadout(name, family, cohort: SwapCohort, mod: str, vectors: Dict, mean_vec=None, pca=None):
    """vectors: underlying record (ECG path / dicom_id) -> 1-D vector."""
    width = len(next(iter(vectors.values())))
    mean_vec = (np.mean(np.stack(list(vectors.values())), 0) if mean_vec is None else mean_vec)

    def X_of(records):
        X = np.full((len(records), width), np.nan, dtype=np.float32)
        for i, r in enumerate(records):
            sid, slots = parse(r)
            u = cohort.underlying(mod, sid, slots.get(mod, 'none'))
            if u == 'mean':
                X[i] = mean_vec
            elif u is not None and u in vectors:
                X[i] = vectors[u]
        return X
    return Readout(name, family, X_of, pca, mod)


def _shift(a, b):
    """Representation shift in the fold's standardised space: cosine distance for vectors; for a
    1-d readout (P(Yes)) cosine is only ever 0 or 2, so the absolute difference instead."""
    if a.shape[1] == 1:
        return np.abs(a[:, 0] - b[:, 0])
    return 1.0 - (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)


class Swap4Probe:
    def __init__(self, clf=None, clf_name='Logistic regression'):
        self.clf = clf if clf is not None else LogisticRegression(C=0.1, max_iter=2000)
        self.clf_name = clf_name

    def _pipe(self, pca, n_train, n_feat):
        steps = [StandardScaler()]
        if pca:
            steps.append(PCA(n_components=min(pca, n_train - 1, n_feat), random_state=0))
        steps.append(clone(self.clf))
        return make_pipeline(*steps)

    def run(self, cohort: SwapCohort, configs: pd.DataFrame, readout: Readout,
            shuffle_seed: int = None) -> pd.DataFrame:
        """One row per recipient: y, p_ref, and for every configuration p_<config>, d_<config>
        (mean over draws), shift_<config> (representation shift) and n_<config> (draws scored)."""
        t = cohort.table
        y = t.y.to_numpy()
        if shuffle_seed is not None:                        # the self-test
            y = np.random.default_rng(shuffle_seed).permutation(y)
        cfg = configs[configs.level_id == cohort.level]
        ref_rec = cfg[cfg.config == 'ref'].set_index('stay_id').record.reindex(t.stay_id)
        X_ref = readout.X_of(list(ref_rec.fillna('')))
        ok = ~np.isnan(X_ref).any(1)
        names = [c for c in cfg.config.unique() if c != 'ref']
        if readout.mod is not None:                          # a unimodal lookup only moves with its slot
            names = [c for c in names if cfg[cfg.config == c].slot.iloc[0] in (readout.mod, 'both')]
        pos = pd.Series(np.arange(len(t)), index=t.stay_id)
        out = pd.DataFrame({'level_id': cohort.level, 'stay_id': t.stay_id, 'y': y, 'fold': t.fold,
                            'p_ref': np.nan})
        acc = {n: (np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))) for n in names}
        for k in sorted(t.fold.unique()):
            tr, te = ok & (t.fold != k).to_numpy(), ok & (t.fold == k).to_numpy()
            if len(np.unique(y[tr])) < 2 or not te.any():
                continue
            model = self._pipe(readout.pca, tr.sum(), X_ref.shape[1]).fit(X_ref[tr], y[tr])
            space = StandardScaler().fit(X_ref[tr])
            out.loc[te, 'p_ref'] = model.predict_proba(X_ref[te])[:, 1]
            Z_ref = space.transform(X_ref)
            te_ids = set(t.stay_id[te])
            for n in names:
                c = cfg[(cfg.config == n) & cfg.stay_id.isin(te_ids)]
                if c.empty:
                    continue
                X = readout.X_of(list(c.record))
                good = ~np.isnan(X).any(1)
                if not good.any():
                    continue
                i = pos.reindex(c.stay_id[good]).to_numpy()
                p = model.predict_proba(X[good])[:, 1]
                s, cs, cnt = acc[n]
                np.add.at(s, i, p)
                np.add.at(cs, i, _shift(Z_ref[i], space.transform(X[good])))
                np.add.at(cnt, i, 1)
        for n, (s, cs, cnt) in acc.items():
            with np.errstate(invalid='ignore', divide='ignore'):
                out[f'p_{n}'] = np.where(cnt > 0, s / cnt, np.nan)
                out[f'shift_{n}'] = np.where(cnt > 0, cs / cnt, np.nan)
            out[f'd_{n}'] = out[f'p_{n}'] - out.p_ref
            out[f'n_{n}'] = cnt
        return out.assign(readout=readout.name, readout_family=readout.family, clf=self.clf_name)


def vectors_by_slot(run, resolver: SwapCohort, mod: str, keys: Sequence[str]) -> Dict[str, dict]:
    """Unimodal vectors by the record in the `mod` slot, from every stored record whose slot holds a real signal
    (the LLaVA notebook's pre-fusion lookup): {key: {underlying record: vector}}. The encoder never sees the text,
    so every stored record with the same signal carries the same vector; the first non-NaN one is kept."""
    out = {k: {} for k in keys}
    for r, i in {r: i for i, r in enumerate(run['record'])}.items():
        sid, sl = parse(r)
        if sl.get(mod) in (None, 'none', 'mean'):
            continue
        d = resolver.underlying(mod, sid, sl[mod])
        if d is not None and d not in out[keys[0]] and not np.isnan(run[keys[0]][i]).any():
            for k in keys:
                out[k][d] = run[k][i]
    return out


def path_key(path: str, raw: str = '') -> str:
    """A raw-file path without its root, so one file matches across roots: `{raw}/<project>/<version>/<file>`
    here, `.../mimic_mmd/unpacked/<project>/<version>/<file>` in the Drive-era notebook runs."""
    raw = raw.rstrip('/')
    if raw and path.startswith(raw + '/'):
        return path[len(raw) + 1:]
    return path.split('/unpacked/', 1)[-1]
