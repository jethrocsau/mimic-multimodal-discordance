"""One model's full swap analysis, the same tables and charts as the notebooks' section 8 / 10.

Every table is written as CSV under the notebooks' file names (so results compare line by line with the
existing runs) and as parquet next to it, to `results/<cohort_version>/<model>/`; charts go to `figures/`.
`informativeness` adds what the zoo table needs: how well each arm separates the outcome on its own.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from .. import fsio
from ..cohort import MOD_NAMES, Manifest
from . import charts as C
from . import swap as S
from .probe import StoreReadout, Swap4Probe, default_classifiers
from .stats import selftest4, summarise4

SIG_NAMES = MOD_NAMES


@dataclass
class Report:
    summary: pd.DataFrame
    clean: pd.DataFrame
    selftest: pd.DataFrame
    ref_shifts: pd.DataFrame
    auroc: pd.DataFrame
    by_draw: pd.DataFrame
    temporal: pd.DataFrame
    paired: pd.DataFrame
    informativeness: pd.DataFrame
    frames: list
    score_frames: list


def standard_configs(manifest: Manifest, input_level=True):
    c = manifest.input_configs() if input_level else manifest.configs
    return c[~c.config.str.contains('@')].reset_index(drop=True)


def build_readouts(adapter_cls, run, pca_dim):
    out = []
    for name, family, key, pca in adapter_cls.readouts:
        if run is None or key not in run:
            print(f'  readout {name}: key {key} not stored, skipped')
            continue
        fn = (lambda a: a.reshape(-1, 1)) if key == 'p_yes' else None
        out.append(StoreReadout(name, family, run, key, fn=fn, pca=pca_dim if pca == 'pca' else pca))
    return out


def informativeness(manifest: Manifest, configs, readouts, clfs):
    """AUROC of each arm on its own: signal only (A), note only (B), both (ref).

    P(Yes): the model's own score, no probe. Vector readouts: a probe trained AND tested on that arm, with the
    tier's fixed folds (so the arms are compared on the same splits). Returns (summary, per-stay scores); the
    per-stay table lets the zoo recompute AUROC on any subset of stays (e.g. the MIMIC-CXR test split)."""
    per = []
    arms = {'signal only': 'A', 'note only': 'B', 'both': 'ref'}
    for lv, c in manifest.cohorts.items():
        t = c.table
        cfg = configs[configs.level_id == lv]
        for ro in readouts:
            Xs = {}
            for arm, conf in arms.items():
                rec = cfg[cfg.config == conf].set_index('stay_id').record.reindex(t.stay_id)
                Xs[arm] = ro.X_of(list(rec.fillna('')))
            ok = np.all([~np.isnan(X).any(1) for X in Xs.values()], axis=0)     # same stays for every arm
            y = t.y.to_numpy()
            if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
                continue
            for arm, X in Xs.items():
                base = dict(level_id=lv, readout=ro.name, key=ro.key, arm=arm)
                if ro.key == 'p_yes':
                    per.append(pd.DataFrame(dict(**base, clf=S.RAW, stay_id=t.stay_id[ok], y=y[ok], p=X[ok, 0])))
                    continue
                for cn, clf in clfs.items():
                    p = np.full(len(t), np.nan)
                    for k in sorted(t.fold.unique()):
                        tr, te = ok & (t.fold != k).to_numpy(), ok & (t.fold == k).to_numpy()
                        if len(np.unique(y[tr])) < 2 or not te.any():
                            continue
                        pr = Swap4Probe(clf, cn)._pipe(ro.pca, tr.sum(), X.shape[1]).fit(X[tr], y[tr])
                        p[te] = pr.predict_proba(X[te])[:, 1]
                    k = ~np.isnan(p)
                    per.append(pd.DataFrame(dict(**base, clf=cn, stay_id=t.stay_id[k], y=y[k], p=p[k])))
    per = pd.concat(per, ignore_index=True) if per else pd.DataFrame(
        columns=['level_id', 'readout', 'key', 'arm', 'clf', 'stay_id', 'y', 'p'])
    return info_auroc(per), per


def info_auroc(per: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lv, ro, key, clf, arm), g in per.groupby(['level_id', 'readout', 'key', 'clf', 'arm'], sort=False):
        if g.y.nunique() == 2:
            rows.append(dict(level_id=lv, readout=ro, key=key, clf=clf, arm=arm, n=len(g),
                             auroc=roc_auc_score(g.y, g.p)))
    return pd.DataFrame(rows)


def run_report(adapter_cls, manifest: Manifest, run, cfg, out_dir, fig_dir, clfs=None, extra=(),
               extra_configs=None, selftest=True, charts=True) -> Report:
    """extra: model-specific readouts (e.g. LLaVA's pre-fusion LookupReadouts), scored on extra_configs
    (all donor draws) when given."""
    clfs = clfs or default_classifiers()
    configs = standard_configs(manifest)
    feat_configs = extra_configs if extra_configs is not None else standard_configs(manifest, input_level=False)
    readouts = build_readouts(adapter_cls, run, cfg.pca_dim)
    tag = manifest.pair
    ctx = S.Ctx(model_name=adapter_cls.name, sig=manifest.mods[0], sig_name=SIG_NAMES[manifest.mods[0]],
                cohorts=manifest.cohorts, configs=configs,
                answer_readouts=[r for r in readouts if r.name.startswith('answer P(Yes)')],
                note_blind=set(adapter_cls.note_blind), out_dir=out_dir, fig_dir=fig_dir, file_tag=tag)
    def save(df, name):                 # every table carries `model`, so tables of all models stack (bq link)
        if 'model' not in df:
            df = df.assign(model=adapter_cls.name)
        fsio.write_csv(df, f'{out_dir}/{name}.csv')
        fsio.write_parquet(df, f'{out_dir}/{name}.parquet')

    st = pd.DataFrame()
    vec = [r for r in readouts if r.key != 'p_yes']
    if selftest and vec:
        lr = 'Logistic regression'
        st = selftest4(Swap4Probe(clfs[lr], lr), manifest.cohorts[manifest.levels[0]], configs, vec[0], n_perm=10)
        save(st, f'swap4_selftest_{tag}')
        print('self-test:', st.verdict.value_counts().to_dict())

    jobs = [(lv, ro, cf, nm, clf) for lv in manifest.levels
            for ro, cf in [(r, configs) for r in readouts] + [(r, feat_configs) for r in extra]
            for nm, clf in clfs.items()]
    frames = [Swap4Probe(clf, nm).run(manifest.cohorts[lv], cf, ro) for lv, ro, cf, nm, clf in
              tqdm(jobs, desc='probe transfer')]
    summary = summarise4(frames, model=adapter_cls.name)
    save(summary, f'swap4_summary_{tag}')
    save(pd.concat(frames, ignore_index=True), f'swap4_per_stay_{tag}')
    clean = S.clean_summary(ctx, summary)
    save(clean, f'swap4_summary_{tag}_clean')

    sf = S.score_frames(ctx, frames)
    save(pd.concat(sf, ignore_index=True), f'swap4_score_frames_{tag}')
    rs = S.reference_shifts(ctx, sf)
    save(rs, f'swap4_reference_shifts_{tag}')
    au = S.auroc_shifts(ctx, sf)
    save(au, f'swap4_auroc_{tag}')
    bd = S.auroc_by_draw(ctx)
    save(bd, f'swap4_auroc_by_draw_{tag}')
    tp = S.temporal_pairs(ctx, frames)
    save(tp, f'swap4_temporal_paired_{tag}')
    paired = S.paired_dominance(ctx, sf)
    save(paired, f'swap4_paired_dominance_{tag}')
    info, info_per = informativeness(manifest, configs, readouts, clfs)
    save(info, f'informativeness_{tag}')
    save(info_per, f'informativeness_per_stay_{tag}')

    if charts:
        for (ro, nm), _ in rs.groupby(['readout', 'clf'], sort=False):
            C.reference_chart(ctx, rs, ro, nm, name=f'swap4_vsref_{C.ftag(ro)}_{C.clf_tag(nm)}')
        for (ro, nm), _ in au.groupby(['readout', 'clf'], sort=False):
            C.auroc_chart(ctx, au, ro, nm, name=f'swap4_auroc_{C.ftag(ro)}_{C.clf_tag(nm)}')
        for ro in S.temporal_readouts(ctx, frames):
            C.temporal_chart(ctx, tp, ro)
        for nm in [c for c in paired.clf.unique() if c != S.RAW]:
            C.dominance_chart(ctx, paired, nm)
    return Report(summary, clean, st, rs, au, bd, tp, paired, info, frames, sf)
