"""The cross-model table: does each model rely on the note more than the note's information justifies?

Per (model, tier, readout, classifier):
    informativeness   AUROC of the signal-only arm and the note-only arm on their own (same stays, same folds);
                      info_share_note = (AUROC_note − .5)+ / ((AUROC_note − .5)+ + (AUROC_signal − .5)+)
    reliance          paired dominance of the counterfactual swap on |risk change|, as a share:
                      reliance_share_note = (dominance + 1) / 2, with its bootstrap CI
                      (also the label-specific outcome-pull dominance, as a second view)
    bias              reliance_share_note − info_share_note: > 0 the model leans on the note beyond what the
                      note alone tells it; < 0 it leans on the signal

Readouts are matched across models by their stored key (p_yes, ans_emb, post_all, post_signal), not by name.
`stays` restricts everything to a subset (e.g. stays whose images are in the MIMIC-CXR official test split),
recomputed from the per-stay tables; `exclude_exposed` drops models trained on MIMIC data.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import fsio
from .analysis import swap as S
from .analysis.report import info_auroc


def load_model(cfg, model, pair='cxr_notes'):
    d = cfg.results_dir(model, pair)
    meta = fsio.read_json(f'{d}/model.json')
    info_per = fsio.read_parquet(f'{d}/informativeness_per_stay_{pair}.parquet')
    sf = fsio.read_parquet(f'{d}/swap4_score_frames_{pair}.parquet')
    return meta, info_per, sf


def _own_columns(g: pd.DataFrame) -> pd.DataFrame:
    """One frame back out of the concatenated table: keep a configuration's p_/d_/shift_/n_ columns only if the
    frame scored it (its p_ column has values); shift_ stays even when all-NaN (raw P(Yes) has no shift)."""
    g = g.reset_index(drop=True)
    have = {c[2:] for c in g.columns if c.startswith('p_') and g[c].notna().any()}
    drop = [c for c in g.columns
            if any(c.startswith(p) for p in ('p_', 'd_', 'shift_', 'n_')) and c.split('_', 1)[1] not in have]
    return g.drop(columns=drop)


def _share(a, b):
    a, b = max(a - 0.5, 0.0), max(b - 0.5, 0.0)
    return a / (a + b) if (a + b) > 0 else np.nan


def model_table(meta, info_per, score_frames, stays=None) -> pd.DataFrame:
    if stays is not None:
        info_per = info_per[info_per.stay_id.isin(stays)]
        score_frames = score_frames[score_frames.stay_id.isin(stays)]
    info = info_auroc(info_per)
    keys = info.drop_duplicates('readout').set_index('readout').key.to_dict()
    ctx = S.Ctx(model_name=meta['model'], sig=meta['sig'], sig_name=meta['sig'], cohorts={}, configs=None,
                note_blind=set(meta.get('note_blind', [])))
    frames = [_own_columns(g) for _, g in score_frames.groupby(['level_id', 'readout', 'clf'], sort=False)]
    paired = S.paired_dominance(ctx, frames)
    rows = []
    for (lv, ro, clf), g in info.groupby(['level_id', 'readout', 'clf']):
        a = g.set_index('arm').auroc
        if not {'signal only', 'note only'} <= set(a.index):
            continue
        p = paired[(paired.level_id == lv) & (paired.readout == ro) & (paired.clf == clf)]
        cf = p[(p.family == 'counterfactual') & (p.measure == '|risk change|')]
        ls = p[(p.family == 'label-specific') & (p.measure == 'outcome pull')]
        share_info = _share(a['note only'], a['signal only'])
        row = dict(model=meta['model'], **{k: meta.get(k) for k in ('fusion', 'domain', 'params', 'mimic_exposure')},
                   level_id=lv, readout=ro, key=keys.get(ro), clf=clf, n=int(g.n.max()),
                   auroc_signal_only=a['signal only'], auroc_note_only=a['note only'], auroc_both=a.get('both'),
                   info_share_note=share_info)
        if len(cf):
            r = cf.iloc[0]
            row.update(reliance_share_note=(r.dominance + 1) / 2, reliance_lo=(r.dominance_lo + 1) / 2,
                       reliance_hi=(r.dominance_hi + 1) / 2, n_paired=r.n_paired)
            row['bias'] = row['reliance_share_note'] - share_info
        if len(ls):
            row['label_specific_dominance'] = ls.iloc[0].dominance
        rows.append(row)
    return pd.DataFrame(rows)


def zoo_table(cfg, models, pair='cxr_notes', stays_by_model=None, exclude_exposed=False) -> pd.DataFrame:
    out = []
    for m in models:
        meta, info_per, sf = load_model(cfg, m, pair)
        if exclude_exposed and meta.get('mimic_exposure', 'none') != 'none':
            continue
        stays = (stays_by_model or {}).get(m)
        out.append(model_table(meta, info_per, sf, stays).assign(subset='official split' if stays is not None else 'all'))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
