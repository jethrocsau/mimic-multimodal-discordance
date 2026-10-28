"""EmbeddingStore: one model's stored readouts, a folder of chunk .npz files (local or gs://), plus its index.

A run folder (`emb/<cohort_version>/<model>/`) holds
    chunk_*.npz (+ .records.txt)   readouts of a block of records: arrays keyed by name, plus `record`
    index.parquet                  record -> chunk, row (a cache, rebuilt from the chunks' record lists)
    runs.parquet                   one row per generation run (code commit, cohort hash, timing)
    manifest.json                  the model's info and the cohort hash of every modality set it ran
    sources.json                   read-only folders (notebook runs) and lookup runs registered by `index import`
    cohort_mean_<mod>_<set>.npz    the cohort-mean signal of the mean swap

Same chunk format as the notebooks, so existing runs are read in place. Resume is keyed on the record ids, not on
chunk numbers, so the record list can change between sessions without redoing work. Chunk names: chunk_<idx>.npz
(unsharded, as the notebooks wrote them) or chunk_s<k>of<n>_<idx>.npz, so shards never collide.

Readouts are loaded under mmd's names: the keys the notebooks stored (NOTEBOOK_KEYS, plus a model's own `keymap`)
are renamed on load, so imported notebook chunks and new chunks read as one set of names.
"""
import re
import subprocess
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import fsio

META = {"record", "layers", "n_nan", "failed", "failed_reason", "missing"}
NOTEBOOK_KEYS = {'p_yes': 'yes_probability', 'post_last': 'prompt_last_state', 'post_all': 'prompt_mean_state',
                 'ans_emb': 'answer_mean_state', 'answer': 'answer_text', 'prefix_mean': 'projector_mean'}
_NAME = re.compile(r'chunk_(?:s(\d+)of(\d+)_)?(\d+)\.npz$')


def in_shard(record: str, shard) -> bool:
    if shard is None:
        return True
    k, n = shard
    return zlib.crc32(record.encode()) % n == k


def _same_width(key, parts):
    """A chunk holding only signal-less records stores NaN placeholders of another width: widen them."""
    shapes = {p.shape[1:] for p in parts if p.dtype.kind == 'f' and not np.isnan(p).all()}
    if len(shapes) != 1:
        if len(shapes) > 1:
            raise ValueError(f'{key}: chunks disagree on shape {shapes}')
        return parts
    shape = shapes.pop()
    return [p if p.shape[1:] == shape or p.dtype.kind != 'f'
            else np.full((len(p),) + shape, np.nan, p.dtype) for p in parts]


def repo_ref(path='.') -> str:
    """The commit the code runs at (recorded with every run); '' outside a git checkout."""
    try:
        out = subprocess.run(['git', '-C', path, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True)
        dirty = subprocess.run(['git', '-C', path, 'status', '--porcelain', '--', 'mmd'], capture_output=True,
                               text=True).stdout.strip()
        return out.stdout.strip() + ('-dirty' if dirty else '')
    except Exception:
        return ''


class EmbeddingStore:
    def __init__(self, out_dir: str, keymap: Dict[str, str] = None, read_dirs=()):
        """keymap: a model's own stored key -> current key (e.g. LLaVA post_image -> signal_mean_state).
        read_dirs: extra folders read but never written (e.g. a notebook run, imported in place)."""
        self.dir = out_dir.rstrip('/')
        self.names = {**NOTEBOOK_KEYS, **(keymap or {})}
        self.read_dirs = [d.rstrip('/') for d in read_dirs]

    @classmethod
    def for_model(cls, run_dir, keymap=None):
        """The model's run folder plus the read-only folders listed in its sources.json."""
        src = f"{run_dir.rstrip('/')}/sources.json"
        extra = fsio.read_json(src).get('read_dirs', []) if fsio.exists(src) else []
        return cls(run_dir, keymap, extra)

    def __repr__(self):
        return f"EmbeddingStore({self.dir}, {len(self.paths())} chunks)"

    def stored_names(self, key) -> list:
        """Every name `key` may be stored under (today's name first)."""
        return [key] + [old for old, new in self.names.items() if new == key and old != key]

    # ── chunks ───────────────────────────────────────────────────────────────
    def paths(self, own_only=False):
        dirs = [self.dir] + ([] if own_only else self.read_dirs)
        return [p for d in dirs for p in fsio.glob(f"{d}/chunk_*.npz") if _NAME.search(p) and '.tmp' not in p]

    def records_of(self, path) -> list:
        side = path[:-4] + '.records.txt'
        if fsio.exists(side):
            return [r for r in fsio.read_text(side).split('\n') if r]
        return fsio.load_npz(path, keys=['record'])['record'].tolist()

    def chunk_records(self, workers=16) -> Dict[str, list]:
        """chunk path -> its record ids (read in parallel: on gs:// each is a small ranged read)."""
        def read(p):
            try:
                return p, self.records_of(p)
            except Exception as e:                      # a chunk from a killed run
                print(f"  ! unreadable, remove it by hand and re-run: {p} ({e})")
                return p, None
        paths = self.paths()
        with ThreadPoolExecutor(workers) as ex:
            got = list(tqdm(ex.map(read, paths), total=len(paths), desc='chunk record ids', disable=len(paths) < 20))
        return {p: r for p, r in got if r is not None}

    def done(self) -> set:
        return {r for recs in self.chunk_records().values() for r in recs}

    def _next_idx(self, shard):
        idx = -1
        for p in self.paths(own_only=True):
            m = _NAME.search(p)
            same = (m.group(1) is None) if shard is None else (m.group(1) == str(shard[0]) and m.group(2) == str(shard[1]))
            if same:
                idx = max(idx, int(m.group(3)))
        return idx + 1

    def _name(self, idx, shard):
        return (f"{self.dir}/chunk_{idx:05d}.npz" if shard is None
                else f"{self.dir}/chunk_s{shard[0]}of{shard[1]}_{idx:05d}.npz")

    def run(self, records: Iterable[str], compute: Callable, chunk: int = 200, shard=None,
            verbose: bool = True, done: set = None):
        """Call compute(block) on each block of not-yet-done records and save the result.

        compute(block) returns a dict of arrays whose first axis matches the records it actually processed,
        plus a 'record' entry naming them. shard=(k, n) keeps only this shard's records.
        """
        done = self.done() if done is None else done
        todo = [r for r in records if r not in done and in_shard(r, shard)]
        idx = self._next_idx(shard)
        if verbose:
            print(f"{self.dir.split('/')[-1]}: {len(done):,} done, {len(todo):,} to do"
                  + (f" (shard {shard[0]} of {shard[1]})" if shard else ''))
        saved = 0
        for b in range(0, len(todo), chunk):
            block = todo[b:b + chunk]
            res = compute(block)
            if res is None or len(res.get("record", [])) == 0:
                if verbose:
                    print(f"  chunk {idx}: nothing usable, skipped")
                continue
            path = self._name(idx, shard)
            fsio.save_npz(res, path)
            fsio.write_text('\n'.join(res['record'].tolist()) + '\n', path[:-4] + '.records.txt')
            saved += len(res['record'])
            if verbose:
                print(f"  chunk {idx}: {len(res['record'])} of {len(block)} saved")
            idx += 1
        return saved

    def load(self, keys=None, records=None, dtype=np.float32, chunks: Optional[Iterable[str]] = None):
        """Read back under today's names, optionally only some readouts, records, or chunks (from the index)."""
        want = set(records) if records is not None else None
        stored = None if keys is None else [n for k in keys for n in self.stored_names(k)] + ['record']
        acc, recs = {}, []
        for p in (list(chunks) if chunks is not None else self.paths()):
            if want is not None and not any(r in want for r in self.records_of(p)):
                continue
            z = fsio.load_npz(p, keys=stored)
            rec = z['record']
            sel = np.ones(len(rec), bool) if want is None else np.array([r in want for r in rec])
            if not sel.any():
                continue
            for k in [k for k in z if k not in META]:
                a = z[k][sel]
                acc.setdefault(self.names.get(k, k), []).append(a.astype(dtype) if a.dtype.kind == 'f' else a)
            recs.append(rec[sel])
        if not recs:
            return None
        out = {k: np.concatenate(_same_width(k, v)) for k, v in acc.items()}
        out["record"] = np.concatenate(recs)
        return out

    # ── index: record -> chunk, row (a cache of the chunks' record lists) ───
    @property
    def index_path(self):
        return f'{self.dir}/index.parquet'

    def build_index(self, run_ref: str = '') -> pd.DataFrame:
        rows = [(r, path, i) for path, recs in self.chunk_records().items() for i, r in enumerate(recs)]
        idx = pd.DataFrame(rows, columns=['record', 'chunk', 'row'])
        return idx.drop_duplicates('record', keep='last').assign(run_ref=run_ref)   # a record re-run later wins

    def save_index(self, idx: pd.DataFrame):
        fsio.write_parquet(idx, self.index_path)

    def load_index(self) -> pd.DataFrame:
        p = self.index_path
        return fsio.read_parquet(p) if fsio.exists(p) else pd.DataFrame(columns=['record', 'chunk', 'row'])

    @staticmethod
    def chunks_for(idx: pd.DataFrame, records) -> Optional[list]:
        """The chunks holding `records` (None = no index yet: read every chunk)."""
        return sorted(set(idx[idx.record.isin(set(records))].chunk)) if len(idx) else None

    # ── run folder metadata ─────────────────────────────────────────────────
    def log_run(self, **fields):
        p = f'{self.dir}/runs.parquet'
        old = fsio.read_parquet(p) if fsio.exists(p) else pd.DataFrame()
        row = pd.DataFrame([dict(logged=datetime.now(timezone.utc).isoformat(timespec='seconds'), **fields)])
        fsio.write_parquet(pd.concat([old, row.astype(str)], ignore_index=True), p)

    def run_info(self) -> dict:
        p = f'{self.dir}/manifest.json'
        return fsio.read_json(p) if fsio.exists(p) else {}

    def record_cohort(self, info: dict, modset: str, cohort_hash: str):
        """Note which cohort manifest this model ran on, per modality set."""
        hashes = {**self.run_info().get('cohort_hashes', {}), modset: cohort_hash}
        fsio.write_json(dict(info, cohort_hashes=hashes), f'{self.dir}/manifest.json')

    def cohort_hash(self, modset: str):
        """The cohort hash this model's records of `modset` were generated on (None = never recorded)."""
        return self.run_info().get('cohort_hashes', {}).get(modset)

    def sources(self) -> dict:
        p = f'{self.dir}/sources.json'
        return fsio.read_json(p) if fsio.exists(p) else {}
