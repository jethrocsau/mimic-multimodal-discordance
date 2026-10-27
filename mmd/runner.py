"""How to run it: a manifest's records -> the model adapter -> the embedding store (the GPU step).

    select()   the records a model pass needs, for some tiers / strategies / orders (core arms always)
    plan()     records, already stored, to do, and estimated GPU-hours (from measured or given s/record)
    smoke()    a few records end to end: shapes, signal span, Yes/No ids, answers, s/record; nothing stored
    run()      every missing record, resumable, optionally one shard of n. Mean-swap records run inside the
               adapter's inject_mean() for each mean modality, with the cohort mean computed once and saved next
               to the run (with any extra arrays the adapter returns, e.g. OpenTSLM's label statistics)
"""
import contextlib
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from . import fsio
from .cohort import Manifest
from .models.base import SampleBuilder
from .perturb import REFERENCE, resolve
from .records import parse_record
from .store import EmbeddingStore, in_shard


def mean_mods(record: str) -> frozenset:
    """The modalities a record replaces with the cohort mean."""
    return frozenset(m for m, s in parse_record(record)[1].items() if s == 'mean')


def mean_path(run_dir: str, mod: str, modset: str) -> str:
    """Where a model's cohort-mean signal of `mod` for one modality set is cached."""
    return f"{run_dir.rstrip('/')}/cohort_mean_{mod}_{modset}.npz"


def select(manifest: Manifest, adapter_cls, levels=None, strategies='all', max_order: int = None,
           factors: bool = True) -> list:
    c = manifest.model_configs(factors)
    if levels:
        c = c[c.level_id.isin(levels)]
    keep = c.family.isin({s.family for s in resolve(strategies)} | {REFERENCE})
    if max_order:
        keep &= c.order <= int(max_order)
    c = c[keep | c.core]
    can = set(adapter_cls.mean_modalities())
    return sorted(r for r in set(c.record) if mean_mods(r) <= can)


def plan(store: EmbeddingStore, records, sec_per_record=None, shard=None) -> pd.DataFrame:
    done = store.done()
    mine = [r for r in records if in_shard(r, shard)]
    todo = sum(r not in done for r in mine)
    return pd.DataFrame([dict(records=len(mine), stored=len(mine) - todo, todo=todo,
                              est_gpu_hours=round(todo * sec_per_record / 3600, 1) if sec_per_record else None)])


def smoke(adapter, builder: SampleBuilder, records, n=3, verbose=True):
    """Run n records of each arm present (which modalities are in the input) without storing anything."""
    picked, seen = [], set()
    for r in records:
        arm = tuple(sorted(k for k, v in parse_record(r)[1].items() if v != 'none'))
        if arm not in seen and '~' not in r and not mean_mods(r):
            picked.append(r); seen.add(arm)
    picked = (picked + [r for r in records if r not in picked])[:max(n, len(picked))]
    rows, times = [], []
    for r in picked:
        s = builder(r)
        t = time.time()
        out = adapter.forward(s)
        times.append(time.time() - t)
        row = dict(record=r, seconds=round(times[-1], 2), answer=out.get('answer_text'),
                   yes_probability=round(float(out['yes_probability']), 4),
                   yes_log_odds=round(float(out.get('yes_log_odds', np.nan)), 2), signal_span=out.get('signal_span'))
        row.update({f'shape:{k}': np.shape(v) for k, v in out.items() if np.ndim(v) > 0})
        rows.append(row)
    df = pd.DataFrame(rows)
    if verbose:
        print(adapter.describe())
        print(df.to_string(index=False))
        print(f'seconds per record (median, after the first): {np.median(times[1:] or times):.2f}')
    return df, float(np.median(times[1:] or times))


def cohort_mean(adapter, store: EmbeddingStore, mod: str, modset: str, reference_signals) -> dict:
    """The cohort-mean signal of `mod`, computed once over the reference signals and cached in the run folder."""
    path = mean_path(store.dir, mod, modset)
    if fsio.exists(path):
        return fsio.load_npz(path)
    sigs = [s for s in reference_signals(mod) if s is not None]
    m = adapter.signal_mean(sigs, mod=mod)
    z = dict(m if isinstance(m, dict) else dict(mean=m), n=np.array(len(sigs)))
    fsio.save_npz(z, path)
    print(f'cohort-mean {mod} over {len(sigs)} reference signals saved: {z["mean"].shape}')
    return z


def run(adapter, manifest: Manifest, store: EmbeddingStore, builder: SampleBuilder, records, cfg,
        shard=None, reference_signals=None, run_ref=''):
    """Store every missing record, then refresh the index. reference_signals(mod) -> the cohort's reference
    signals of that modality (for the cohort mean)."""
    t0 = time.time()
    groups = defaultdict(list)
    for r in records:
        groups[mean_mods(r)].append(r)
    fn = adapter.compute_fn(builder)
    saved = store.run(groups.pop(frozenset(), []), fn, chunk=cfg.chunk, shard=shard)
    for mods, recs in groups.items():
        with contextlib.ExitStack() as stack:
            for m in sorted(mods):
                z = cohort_mean(adapter, store, m, manifest.name, reference_signals)
                stack.enter_context(adapter.inject_mean(z['mean'], mod=m,
                                                        **{k: v for k, v in z.items() if k not in ('mean', 'n')}))
            saved += store.run(recs, fn, chunk=cfg.chunk, shard=shard)
    store.save_index(store.build_index(run_ref))
    store.log_run(**adapter.info(), modset=manifest.name, cohort_hash=manifest.hash, run_ref=run_ref,
                  shard='' if shard is None else f'{shard[0]}/{shard[1]}', saved=saved,
                  minutes=round((time.time() - t0) / 60, 1))
    return saved
