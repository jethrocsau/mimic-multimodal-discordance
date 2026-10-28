"""Embeddings and the probe that turns them into a predicted 1-year risk for every perturbed configuration.

    Embeddings   record ids -> a feature matrix (NaN rows where a record has no value)
        .from_store    arrays saved by the model passes, keyed by the full record id
                       (model-level: every configuration was its own forward pass)
        .from_lookup   a unimodal pre-fusion vector looked up by the record actually in the slot
                       (feature-level: free, so it can use every donor draw)
    Probe        fitted ONCE per fold on the reference embeddings of the training folds, then scores every
                 perturbed configuration of the test-fold stays unchanged. Δ = p(config) - p(ref), averaged over
                 donor draws.
    raw_scores   the same table for yes_probability read as the model's own risk score, no probe.

Both return one LONG table, one row per (stay, configuration): level_id, readout, readout_family, clf, stay_id,
y, fold, config, family, slot, order, p, p_ref, delta, shift, n. The reference has its own rows (config 'ref').
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

from ..modsets import slot_mods
from ..perturb import REFERENCE, resolve
from ..records import parse_record

RAW = 'no probe (raw P(Yes))'
RAW_KEY = 'yes_probability'
LONG = ['level_id', 'readout', 'readout_family', 'clf', 'stay_id', 'y', 'fold', 'config', 'family', 'slot', 'order',
        'p', 'p_ref', 'delta', 'shift', 'n']


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
class Embeddings:
    name: str
    family: str                                 # pre-fusion | projector | cross-attention | post-fusion
    matrix: Callable[[Sequence[str]], np.ndarray]
    pca: Optional[int] = None
    mod: Optional[str] = None                   # set for a unimodal lookup: only its own modality's swaps move it
    key: Optional[str] = None                   # the stored key, for the zoo table

    @property
    def raw(self) -> bool:
        """The model's own P(Yes): read as a risk score directly, as well as probed."""
        return self.key == RAW_KEY

    @classmethod
    def from_store(cls, name, family, run, key, fn=None, pca=None):
        """run: EmbeddingStore.load(...) output. fn reshapes one array (default: flatten per record)."""
        fn = fn or (lambda a: a.reshape(len(a), -1))
        idx = {r: i for i, r in enumerate(run['record'])} if run is not None else {}
        width = fn(run[key][:1]).shape[1] if run is not None else 1

        def matrix(records):
            X = np.full((len(records), width), np.nan, dtype=np.float32)
            rows = [(i, idx[r]) for i, r in enumerate(records) if r in idx]
            if rows:
                a, b = map(np.array, zip(*rows))
                X[a] = fn(run[key][b])
            return X
        return cls(name, family, matrix, pca, key=key)

    @classmethod
    def from_spec(cls, spec, run, pca_dim):
        """An adapter's ReadoutSpec over a loaded run; yes_probability as one column."""
        fn = (lambda a: a.reshape(-1, 1)) if spec.key == RAW_KEY else None
        return cls.from_store(spec.name, spec.family, run, spec.key, fn=fn,
                              pca=pca_dim if spec.pca == 'pca' else spec.pca)

    @classmethod
    def from_lookup(cls, name, family, manifest, mod: str, vectors: Dict, mean_vec=None, pca=None):
        """vectors: underlying record (ECG path / dicom_id) -> 1-D vector; manifest resolves slots to records."""
        width = len(next(iter(vectors.values())))
        mean_vec = (np.mean(np.stack(list(vectors.values())), 0) if mean_vec is None else mean_vec)

        def matrix(records):
            X = np.full((len(records), width), np.nan, dtype=np.float32)
            for i, r in enumerate(records):
                sid, slots = parse_record(r)
                u = manifest.resolve(mod, sid, slots.get(mod, 'none'))
                if u == 'mean':
                    X[i] = mean_vec
                elif u is not None and u in vectors:
                    X[i] = vectors[u]
            return X
        return cls(name, family, matrix, pca, mod)


def _shift(a, b):
    """Representation shift in the fold's standardised space: cosine distance for vectors; for a
    1-d readout (P(Yes)) cosine is only ever 0 or 2, so the absolute difference instead."""
    if a.shape[1] == 1:
        return np.abs(a[:, 0] - b[:, 0])
    return 1.0 - (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)


def select_configs(configs: pd.DataFrame, level=None, strategies=None, mods=None, emb: Embeddings = None):
    """The perturbed configurations to score: one tier, some strategies, slots inside `mods`, and for a unimodal
    lookup only the slots that move it."""
    c = configs[configs.family != REFERENCE]
    if level is not None:
        c = c[c.level_id == level]
    if strategies:
        c = c[c.family.isin({s.family for s in resolve(strategies)})]
    if mods:
        c = c[[set(slot_mods(s)) <= set(mods) for s in c.slot]]
    if emb is not None and emb.mod is not None:
        c = c[[emb.mod in slot_mods(s) for s in c.slot]]
    return c


def _long(level, stay_ids, y, fold, p_ref, swapped: pd.DataFrame, meta: pd.DataFrame, emb, clf):
    """Per-draw scores -> one row per (stay, config), plus the reference rows."""
    t = pd.DataFrame(dict(stay_id=stay_ids, y=y, fold=fold, p_ref=p_ref))
    if len(swapped):
        g = swapped.groupby(['stay_id', 'config']).agg(p=('p', 'mean'), shift=('shift', 'mean'), n=('p', 'size'))
        g = g.reset_index().merge(meta, on='config').merge(t, on='stay_id')
        g['delta'] = g.p - g.p_ref
    else:
        g = pd.DataFrame(columns=LONG)
    ref = t[t.p_ref.notna()].assign(config='ref', family=REFERENCE, slot='', order=0, p=lambda d: d.p_ref,
                                    delta=0.0, shift=0.0, n=1)
    out = pd.concat([ref, g], ignore_index=True).assign(level_id=level, readout=emb.name,
                                                       readout_family=emb.family, clf=clf)
    return out[LONG]


class Probe:
    """A classifier fitted on the reference embeddings, then applied unchanged to every perturbed configuration."""

    def __init__(self, classifier=None, name='Logistic regression'):
        self.classifier = classifier if classifier is not None else LogisticRegression(C=0.1, max_iter=2000)
        self.name = name

    def pipeline(self, pca, n_train, n_feat):
        steps = [StandardScaler()]
        if pca:
            steps.append(PCA(n_components=min(pca, n_train - 1, n_feat), random_state=0))
        steps.append(clone(self.classifier))
        return make_pipeline(*steps)

    def score(self, cohort, configs: pd.DataFrame, emb: Embeddings, strategies=None, mods=None,
              shuffle_seed: int = None) -> pd.DataFrame:
        """The long table (module docstring) for one tier. strategies / mods narrow what is scored (default:
        every configuration in `configs`); shuffle_seed permutes the labels first (the label-shuffle test)."""
        t = cohort.table
        y = t.y.to_numpy()
        if shuffle_seed is not None:
            y = np.random.default_rng(shuffle_seed).permutation(y)
        cfg = configs[configs.level_id == cohort.level]
        ref_rec = cfg[cfg.config == 'ref'].drop_duplicates('stay_id').set_index('stay_id').record.reindex(t.stay_id)
        X_ref = emb.matrix(list(ref_rec.fillna('')))
        ok = ~np.isnan(X_ref).any(1)
        others = select_configs(cfg, None, strategies, mods, emb)
        meta = others.drop_duplicates('config')[['config', 'family', 'slot', 'order']]
        pos = pd.Series(np.arange(len(t)), index=t.stay_id)
        p_ref, parts = np.full(len(t), np.nan), []
        for k in sorted(t.fold.unique()):
            tr, te = ok & (t.fold != k).to_numpy(), ok & (t.fold == k).to_numpy()
            if len(np.unique(y[tr])) < 2 or not te.any():
                continue
            model = self.pipeline(emb.pca, tr.sum(), X_ref.shape[1]).fit(X_ref[tr], y[tr])
            space = StandardScaler().fit(X_ref[tr])
            p_ref[te] = model.predict_proba(X_ref[te])[:, 1]
            c = others[others.stay_id.isin(set(t.stay_id[te]))]
            if c.empty:
                continue
            X = emb.matrix(list(c.record))
            good = ~np.isnan(X).any(1)
            if not good.any():
                continue
            i = pos.reindex(c.stay_id[good]).to_numpy()
            parts.append(pd.DataFrame(dict(stay_id=c.stay_id[good].to_numpy(), config=c.config[good].to_numpy(),
                                           p=model.predict_proba(X[good])[:, 1],
                                           shift=_shift(space.transform(X_ref[i]), space.transform(X[good])))))
        swapped = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['stay_id', 'config', 'p'])
        return _long(cohort.level, t.stay_id.to_numpy(), y, t.fold.to_numpy(), p_ref, swapped, meta, emb, self.name)


def raw_scores(cohort, configs: pd.DataFrame, emb: Embeddings, strategies=None, mods=None) -> pd.DataFrame:
    """The long table for yes_probability read raw: the model's own P(Yes), mean over the stay's donor draws."""
    t = cohort.table
    cfg = configs[configs.level_id == cohort.level]
    ref = cfg[cfg.config == 'ref'].drop_duplicates('stay_id')
    p_ref = pd.Series(emb.matrix(list(ref.record))[:, 0], index=ref.stay_id.to_numpy()).reindex(t.stay_id).to_numpy()
    others = select_configs(cfg, None, strategies, mods, emb)
    swapped = others[['stay_id', 'config']].assign(p=emb.matrix(list(others.record))[:, 0], shift=np.nan).dropna(
        subset=['p'])
    meta = others.drop_duplicates('config')[['config', 'family', 'slot', 'order']]
    out = _long(cohort.level, t.stay_id.to_numpy(), t.y.to_numpy(), t.fold.to_numpy(), p_ref, swapped, meta, emb, RAW)
    out.loc[out.family == REFERENCE, 'shift'] = np.nan
    return out


def vectors_by_slot(run, manifest, mod: str, keys: Sequence[str]) -> Dict[str, dict]:
    """Unimodal vectors by the record in the `mod` slot, from every stored record whose slot holds a real signal
    (the LLaVA notebook's pre-fusion lookup): {key: {underlying record: vector}}. The encoder never sees the text,
    so every stored record with the same signal carries the same vector; the first non-NaN one is kept."""
    out = {k: {} for k in keys}
    for r, i in {r: i for i, r in enumerate(run['record'])}.items():
        sid, sl = parse_record(r)
        if sl.get(mod) in (None, 'none', 'mean'):
            continue
        d = manifest.resolve(mod, sid, sl[mod])
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
