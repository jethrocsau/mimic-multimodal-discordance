"""Generate a model's perturbation records: manifest -> missing records -> adapter -> store -> index.

    plan()   records, already stored, to do, and estimated GPU-hours (from measured or given s/record)
    smoke()  a few records end to end: shapes, signal span, Yes/No ids, answers, s/record
    run()    every missing record, resumable, optionally one shard of n; mean-swap records run inside the
             adapter's inject_mean() with the cohort mean computed once and saved next to the run
             (with any extra arrays the adapter returns, e.g. OpenTSLM's label statistics)
"""
import time

import numpy as np
import pandas as pd

from . import fsio
from . import index as index_mod
from .cohort import Manifest, parse, run_records
from .models.base import SampleFactory
from .store import ChunkStore, in_shard


def is_mean(record: str, sig: str) -> bool:
    return parse(record)[1].get(sig) == 'mean'


def select(manifest: Manifest, adapter, levels=None, families=None, factors=True):
    recs = run_records(manifest, levels, families, factors)
    if not adapter.supports_mean:
        sig = manifest.mods[0]
        recs = [r for r in recs if not is_mean(r, sig)]
    return recs


def plan(store: ChunkStore, records, sec_per_record=None, shard=None) -> pd.DataFrame:
    done = store.done()
    mine = [r for r in records if in_shard(r, shard)]
    todo = sum(r not in done for r in mine)
    return pd.DataFrame([dict(records=len(mine), stored=len(mine) - todo, todo=todo,
                              est_gpu_hours=round(todo * sec_per_record / 3600, 1) if sec_per_record else None)])


def smoke(adapter, factory: SampleFactory, records, n=3, verbose=True):
    """Run n records of each arm present (signal only, note only, both) without storing anything."""
    picked, seen = [], set()
    for r in records:
        sl = parse(r)[1]
        arm = tuple(sorted(k for k, v in sl.items() if v != 'none'))
        if arm not in seen and '~' not in r:
            picked.append(r); seen.add(arm)
        if len(seen) >= 3:
            break
    picked = (picked + [r for r in records if r not in picked])[:max(n, len(picked))]
    rows, times = [], []
    for r in picked:
        s = factory(r)
        t = time.time()
        out = adapter.forward(s)
        times.append(time.time() - t)
        row = dict(record=r, seconds=round(times[-1], 2), answer=out.get('answer'), p_yes=round(float(out['p_yes']), 4),
                   signal_span=out.get('signal_span', out.get('image_span')))
        row.update({f'shape:{k}': np.shape(v) for k, v in out.items() if np.ndim(v) > 0})
        rows.append(row)
    df = pd.DataFrame(rows)
    if verbose:
        print(adapter.describe())
        print(df.to_string(index=False))
        print(f'seconds per record (median, after the first): {np.median(times[1:] or times):.2f}')
    return df, float(np.median(times[1:] or times))


def run(adapter, manifest: Manifest, store: ChunkStore, factory: SampleFactory, records, cfg,
        shard=None, mean_signals=None, run_ref=''):
    """Store every missing record. mean_signals: callable returning the cohort reference signals for the mean."""
    t0 = time.time()
    sig = manifest.mods[0]
    plain = [r for r in records if not is_mean(r, sig)]
    mean_recs = [r for r in records if is_mean(r, sig)]
    fn = adapter.compute_fn(factory)
    saved = store.run(plain, fn, chunk=cfg.chunk, shard=shard)
    if mean_recs and adapter.supports_mean:
        path = f'{store.dir}/cohort_mean_{sig}.npz'
        if fsio.exists(path):
            z = fsio.load_npz(path)
        else:
            sigs = [s for s in mean_signals() if s is not None]
            m = adapter.signal_mean(sigs)
            z = dict(m if isinstance(m, dict) else dict(mean=m), n=np.array(len(sigs)))
            fsio.save_npz(z, path)
            print(f'cohort-mean signal over {len(sigs)} reference signals saved: {z["mean"].shape}')
        with adapter.inject_mean(z['mean'], **{k: v for k, v in z.items() if k not in ('mean', 'n')}):
            saved += store.run(mean_recs, fn, chunk=cfg.chunk, shard=shard)
    idx = index_mod.build(store, manifest.configs, run_ref)
    index_mod.save(idx, store.dir)
    index_mod.log_run(store.dir, **adapter.info(), cohort_hash=manifest.hash, run_ref=run_ref,
                      shard='' if shard is None else f'{shard[0]}/{shard[1]}', saved=saved,
                      minutes=round((time.time() - t0) / 60, 1))
    return saved
