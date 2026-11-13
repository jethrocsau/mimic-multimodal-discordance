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


def smoke(adapter, builder: SampleBuilder, records, n=3, verbose=True, batch=1):
    """Run n records of each arm present (which modalities are in the input) without storing anything. With
    batch > 1, the same records are also run batched and compared with the single-record readouts."""
    picked, seen = [], set()
    for r in records:
        arm = tuple(sorted(k for k, v in parse_record(r)[1].items() if v != 'none'))
        if arm not in seen and '~' not in r and not mean_mods(r):
            picked.append(r); seen.add(arm)
    picked = (picked + [r for r in records if r not in picked])[:max(n, len(picked))]
    rows, times, singles = [], [], {}
    for r in picked:
        s = builder(r)
        t = time.time()
        out = adapter.forward(s)
        times.append(time.time() - t)
        singles[r] = (s, out)
        row = dict(record=r, seconds=round(times[-1], 2), answer=out.get('answer_text'),
                   yes_probability=round(float(out['yes_probability']), 4),
                   yes_log_odds=round(float(out.get('yes_log_odds', np.nan)), 2), signal_span=out.get('signal_span'))
        row.update({f'shape:{k}': np.shape(v) for k, v in out.items() if np.ndim(v) > 0})
        rows.append(row)
    df = pd.DataFrame(rows)
    if verbose and hasattr(adapter, 'prompt'):        # the fullest input, as the model reads it
        r = max(picked, key=lambda r: sum(v != 'none' for v in parse_record(r)[1].values()))
        print(f'prompt of {r}:\n{adapter.prompt(singles[r][0])}\n')
    if verbose:
        print(adapter.describe())
        print(df.to_string(index=False))
        print(f'seconds per record (median, after the first): {np.median(times[1:] or times):.2f}')
    if batch > 1 and adapter.batches:
        batch_check(adapter, builder, records, singles, batch, verbose)
    return df, float(np.median(times[1:] or times))


def batch_check(adapter, builder, records, singles, batch, verbose=True) -> pd.DataFrame:
    """Batched vs single-record readouts on the same records: answers, log-odds and cosine similarity of every
    vector. Batches mix prompt lengths, so padding is exercised; bf16 rounding leaves differences of ~1e-2."""
    by_mods = defaultdict(list)
    for r, (s, _) in singles.items():
        by_mods[frozenset(s.signals)].append(r)
    for r in records:                        # fill each group up to a full batch
        if len(singles) >= 4 * batch:
            break
        if r not in singles and '~' not in r and not mean_mods(r):
            s = builder(r)
            singles[r] = (s, adapter.forward(s))
            by_mods[frozenset(s.signals)].append(r)
    rows, t_batched, n_batched = [], 0.0, 0
    for recs in by_mods.values():
        for b in range(0, len(recs), batch):
            block = recs[b:b + batch]
            t = time.time()
            outs = adapter.forward_batch([singles[r][0] for r in block])
            t_batched += time.time() - t
            n_batched += len(block)
            for r, o in zip(block, outs):
                one = singles[r][1]
                row = dict(record=r, batch=len(block), same_answer=one.get('answer_text') == o.get('answer_text'),
                           log_odds_single=round(float(one.get('yes_log_odds', np.nan)), 3),
                           log_odds_batched=round(float(o.get('yes_log_odds', np.nan)), 3))
                for k, v in one.items():
                    if np.ndim(v) == 1 and np.size(v) > 1 and not np.isnan(v).all():
                        a, c = np.asarray(v, np.float64), np.asarray(o[k], np.float64)
                        row[f'cos:{k}'] = round(float(a @ c / (np.linalg.norm(a) * np.linalg.norm(c) + 1e-12)), 4)
                rows.append(row)
    df = pd.DataFrame(rows)
    if verbose:
        cos = df.filter(like='cos:')
        print(f'\nbatched (--batch {batch}) vs single records:')
        print(df.to_string(index=False))
        print(f'same answer: {df.same_answer.mean():.0%} · max |Δ log-odds|: '
              f'{(df.log_odds_single - df.log_odds_batched).abs().max():.3f} · min cosine: {cos.min().min():.4f}')
        print(f'seconds per record batched: {t_batched / max(n_batched, 1):.2f}')
        print('OK to batch' if cos.min().min() > 0.99 and (df.log_odds_single - df.log_odds_batched).abs().max() < 0.5
              else 'CHECK: batched readouts differ from single ones more than bf16 rounding explains')
    return df


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
        shard=None, reference_signals=None, run_ref='', batch=1):
    """Store every missing record, then refresh the index. reference_signals(mod) -> the cohort's reference
    signals of that modality (for the cohort mean)."""
    t0 = time.time()
    groups = defaultdict(list)
    for r in records:
        groups[mean_mods(r)].append(r)
    if batch > 1 and not adapter.batches:
        print(f'{adapter.name} runs one record at a time (as its stored runs did); --batch {batch} ignored')
        batch = 1
    fn = adapter.compute_fn(builder, batch=batch)
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
                  shard='' if shard is None else f'{shard[0]}/{shard[1]}', batch=batch, saved=saved,
                  minutes=round((time.time() - t0) / 60, 1))
    return saved


def synthetic_sample(mods, record=None):
    """A made-up input carrying `mods` (no MIMIC data): a grey gradient film, a 12-lead sine ECG, a 24 h vitals
    window, a lab panel and a short report. For checking an adapter loads and runs before any data is mirrored."""
    from PIL import Image

    from .data import LAB_PANEL, VITALS, Bundle, LabPanel
    from .models.base import Sample
    sig = {}
    if 'x' in mods:
        g = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (512, 1))
        sig['x'] = Image.fromarray(np.stack([g, g.T, g], -1))
    if 'e' in mods:
        t = np.arange(5000) / 500.0
        leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        sig['e'] = Bundle(np.stack([np.sin(2 * np.pi * 1.2 * t + k) ** 15 for k in range(12)]), leads,
                          ['mV'] * 12, 500.0, info=dict(record='synthetic-ecg'))
    if 'v' in mods:
        v = np.full((len(VITALS), 24), np.nan)
        v[0], v[2], v[3], v[4] = 95 + 5 * np.sin(np.arange(24) / 3), 105.0, 60.0, 75.0
        v[1, ::4] = 99.1
        sig['v'] = Bundle(v, [c for c, _, _ in VITALS.values()], [u for _, u, _ in VITALS.values()], 1 / 3600,
                          np.arange(24, dtype=float))
    if 'l' in mods:
        tab = pd.DataFrame(index=pd.Index(list(LAB_PANEL.values()), name='lab'),
                           columns=['value', 'text', 'unit', 'ref_low', 'ref_high', 'flag'], dtype=object)
        tab.loc['sodium'] = [131.0, '131', 'mEq/L', 133.0, 145.0, 'abnormal']
        tab.loc['creatinine'] = [2.1, '2.1', 'mg/dL', 0.5, 1.2, 'abnormal']
        tab.loc['lactate'] = [3.4, '3.4', 'mmol/L', 0.5, 2.0, 'abnormal']
        for c in ('value', 'ref_low', 'ref_high'):
            tab[c] = tab[c].astype(float)
        sig['l'] = LabPanel(tab, 2.0)
    report = ('FINDINGS: Moderate cardiomegaly. Bilateral pleural effusions. IMPRESSION: Pulmonary edema.'
              if 'n' in mods else None)
    return Sample(record or ('synthetic|' + '+'.join(mods)), sig, {}, report)


def synthetic_smoke(adapter, batch=1, verbose=True):
    """Every arm of the adapter's modalities (each alone, each with the note, all together) on synthetic inputs:
    readout shapes, answers, log-odds, spans, s/record; with batch > 1 also batched vs single. Nothing stored."""
    from .modsets import MOD_ORDER, NOTE
    mods = [m for m in MOD_ORDER if m in adapter.modalities]
    arms = [(m,) for m in mods] + [(m, NOTE) for m in mods if m != NOTE] + [tuple(mods)]
    arms = list(dict.fromkeys(tuple(m for m in MOD_ORDER if m in a) for a in arms))
    rows, singles, times = [], {}, []
    for a in arms:
        s = synthetic_sample(a)
        t = time.time()
        out = adapter.forward(s)
        times.append(time.time() - t)
        singles[s.record] = (s, out)
        row = dict(arm='+'.join(a), seconds=round(times[-1], 2), answer=out.get('answer_text'),
                   yes_probability=round(float(out['yes_probability']), 4),
                   yes_log_odds=round(float(out.get('yes_log_odds', np.nan)), 2), signal_span=out.get('signal_span'))
        row.update({f'shape:{k}': np.shape(v) for k, v in out.items() if np.ndim(v) > 0})
        rows.append(row)
    df = pd.DataFrame(rows)
    if verbose:
        if hasattr(adapter, 'prompt'):
            print(f'prompt of {arms[-1]}:\n{adapter.prompt(synthetic_sample(arms[-1]))}\n')
        print(adapter.describe())
        print(df.to_string(index=False))
        print(f'seconds per record (median, after the first): {np.median(times[1:] or times):.2f}')
    if batch > 1 and adapter.batches:
        full = singles[f"synthetic|{'+'.join(arms[-1])}"][0]
        group = {f'{full.record}#{i}': (synthetic_sample(arms[-1], f'{full.record}#{i}'), None) for i in range(batch)}
        outs = adapter.forward_batch([s for s, _ in group.values()])
        one = singles[full.record][1]
        cos = min(float(np.dot(np.asarray(one[k], float), np.asarray(o[k], float)) /
                        (np.linalg.norm(np.asarray(one[k], float)) * np.linalg.norm(np.asarray(o[k], float)) + 1e-12))
                  for o in outs for k, v in one.items()
                  if np.ndim(v) == 1 and np.size(v) > 1 and not np.isnan(np.asarray(v, float)).all())
        dlo = max(abs(float(one.get('yes_log_odds', 0)) - float(o.get('yes_log_odds', 0))) for o in outs)
        print(f'batched x{batch} (identical inputs) vs single: min cosine {cos:.4f}, max |Δ log-odds| {dlo:.3f} -> '
              + ('OK to batch' if cos > 0.99 and dlo < 0.5 else 'CHECK batching'))
    return df
