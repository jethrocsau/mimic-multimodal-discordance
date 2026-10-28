"""One model's full analysis on one modality set: probe -> effect tables -> charts.

Tables go to `<cohort_version>/results/<model>/<modset>/<table>.csv|parquet`, charts to `figures/` with the same
layout. Every table carries `model`, so the tables of all models stack (`mmd collect`, `mmd bq link`).

    summary, summary_clean, per_stay      Δ per family and slot (per_stay: every probe score, long)
    scores                                the long scores the effect tables use (vector probes + raw P(Yes))
    selftest                              the label-shuffle test
    reference_shifts, auroc, auroc_by_draw, temporal
    reliance, pairwise_dominance          each modality's share of the effect; every modality pair
    informativeness, informativeness_per_stay   AUROC of each "only modality m" input and of all of them
"""
from dataclasses import dataclass

import pandas as pd
from tqdm.auto import tqdm

from .. import fsio
from ..cohort import Manifest
from ..perturb import REFERENCE, resolve
from . import charts as C
from . import effects as E
from .probe import Embeddings, Probe, default_classifiers, raw_scores


@dataclass
class ModelReport:
    summary: pd.DataFrame
    clean: pd.DataFrame
    selftest: pd.DataFrame
    reference_shifts: pd.DataFrame
    auroc: pd.DataFrame
    auroc_by_draw: pd.DataFrame
    temporal: pd.DataFrame
    reliance: pd.DataFrame
    pairwise: pd.DataFrame
    informativeness: pd.DataFrame
    scores: pd.DataFrame


def build_embeddings(adapter_cls, run, pca_dim) -> list:
    out = []
    for spec in adapter_cls.readouts:
        if run is None or spec.key not in run:
            print(f'  readout {spec.name}: key {spec.key} not stored, skipped')
            continue
        out.append(Embeddings.from_spec(spec, run, pca_dim))
    return out


def narrow(configs: pd.DataFrame, strategies) -> pd.DataFrame:
    """Only some strategies (the reference and the core arms always stay)."""
    if not strategies:
        return configs
    keep = configs.family.isin({s.family for s in resolve(strategies)} | {REFERENCE}) | configs.core
    return configs[keep].reset_index(drop=True)


def analyse_model(adapter_cls, manifest: Manifest, run, cfg, out_dir, fig_dir, clfs=None, extra=(),
                  strategies=None, selftest=True, charts=True) -> ModelReport:
    """extra: model-specific feature-level embeddings (e.g. LLaVA's pre-fusion lookups), scored on every donor
    draw. strategies: analyse only some (default: everything the manifest holds)."""
    clfs = clfs or default_classifiers()
    configs = narrow(manifest.analysis_configs(), strategies)
    feat_configs = narrow(manifest.analysis_configs(model_level=False), strategies)
    embeddings = build_embeddings(adapter_cls, run, cfg.pca_dim)
    ctx = E.AnalysisContext(model_name=adapter_cls.name, modset=manifest.modset, cohorts=manifest.cohorts,
                            configs=configs, raw_embeddings=[e for e in embeddings if e.raw],
                            note_blind=set(adapter_cls.note_blind), out_dir=out_dir, fig_dir=fig_dir)

    def save(df, name):
        if 'model' not in df:
            df = df.assign(model=adapter_cls.name)
        fsio.write_csv(df, f'{out_dir}/{name}.csv')
        fsio.write_parquet(df, f'{out_dir}/{name}.parquet')

    st = pd.DataFrame()
    vec = [e for e in embeddings if not e.raw]
    if selftest and vec:
        lr = 'Logistic regression'
        st = E.label_shuffle_test(Probe(clfs[lr], lr), manifest.cohorts[manifest.levels[0]], configs, vec[0])
        save(st, 'selftest')
        print('label-shuffle test:', st.verdict.value_counts().to_dict())

    jobs = [(lv, e, cf, nm, clf) for lv in manifest.levels
            for e, cf in [(e, configs) for e in embeddings] + [(e, feat_configs) for e in extra]
            for nm, clf in clfs.items()]
    probed = pd.concat([Probe(clf, nm).score(manifest.cohorts[lv], cf, e)
                        for lv, e, cf, nm, clf in tqdm(jobs, desc='probe')], ignore_index=True)
    summary = E.summarise(probed, model=adapter_cls.name)
    save(summary, 'summary')
    save(probed, 'per_stay')
    clean = E.clean_summary(ctx, summary)
    save(clean, 'summary_clean')

    raw = [raw_scores(manifest.cohorts[lv], configs, e) for lv in manifest.levels for e in ctx.raw_embeddings]
    scores = pd.concat([probed[~probed.readout.isin({e.name for e in ctx.raw_embeddings})]] + raw, ignore_index=True)
    save(scores, 'scores')
    rs = E.reference_shifts(ctx, scores)
    save(rs, 'reference_shifts')
    au = E.auroc_shifts(ctx, scores)
    save(au, 'auroc')
    bd = E.auroc_by_draw(ctx)
    save(bd, 'auroc_by_draw')
    tp = E.temporal_pairs(ctx, probed)
    save(tp, 'temporal')
    rel, pw = E.reliance(ctx, scores)
    save(rel, 'reliance')
    save(pw, 'pairwise_dominance')
    info, info_per = E.informativeness(manifest.cohorts, configs, manifest.mods, embeddings, clfs)
    save(info, 'informativeness')
    save(info_per, 'informativeness_per_stay')

    if charts:
        for (ro, nm), _ in rs.groupby(['readout', 'clf'], sort=False):
            C.reference_chart(ctx, rs, ro, nm, name=f'reference_shifts_{C.ftag(ro)}_{C.clf_tag(nm)}')
        for (ro, nm), _ in au.groupby(['readout', 'clf'], sort=False):
            C.auroc_chart(ctx, au, ro, nm, name=f'auroc_{C.ftag(ro)}_{C.clf_tag(nm)}')
        for ro in (tp.readout.unique() if len(tp) else []):
            C.temporal_chart(ctx, tp, ro)
        for (ro, nm), _ in (rel.groupby(['readout', 'clf'], sort=False) if len(rel) else []):
            C.reliance_chart(ctx, rel, ro, nm)
    return ModelReport(summary, clean, st, rs, au, bd, tp, rel, pw, info, scores)
