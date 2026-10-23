"""ChunkStore: a run is a folder of chunk .npz files, local or on gs://.

Same chunk format as the notebooks (arrays keyed by name, plus `record`), so existing runs are read in place.
New chunks also get a `<chunk>.records.txt` sidecar, so resuming and indexing read a few KB per chunk instead
of the arrays; legacy chunks without one fall back to reading their `record` array. Resume is keyed on the
record ids, not on chunk numbers, so the record list can change between sessions without redoing work.

Chunk names: chunk_<idx>.npz (unsharded, as the notebooks wrote them) or chunk_s<k>of<n>_<idx>.npz, so
shards running at the same time never collide.
"""
import os
import re
import zlib
from typing import Callable, Dict, Iterable, Optional

import numpy as np

from . import fsio

META = {"record", "layers", "n_nan", "failed", "failed_reason", "missing"}
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


class ChunkStore:
    def __init__(self, out_dir: str, keymap: Dict[str, str] = None, read_dirs=()):
        """keymap: stored key -> common key, applied on load (e.g. LLaVA post_image -> post_signal).
        read_dirs: extra folders read but never written (e.g. a legacy notebook run, imported in place)."""
        self.dir = out_dir.rstrip('/')
        self.keymap = keymap or {}
        self.read_dirs = [d.rstrip('/') for d in read_dirs]

    @classmethod
    def for_model(cls, run_dir, keymap=None):
        """The model's run folder plus the read-only folders listed in its sources.json."""
        src = f"{run_dir.rstrip('/')}/sources.json"
        extra = fsio.read_json(src).get('read_dirs', []) if fsio.exists(src) else []
        return cls(run_dir, keymap, extra)

    def __repr__(self):
        return f"ChunkStore({self.dir}, {len(self.paths())} chunks)"

    def paths(self, own_only=False):
        dirs = [self.dir] + ([] if own_only else self.read_dirs)
        return [p for d in dirs for p in fsio.glob(f"{d}/chunk_*.npz") if _NAME.search(p) and '.tmp' not in p]

    def records_of(self, path) -> list:
        side = path[:-4] + '.records.txt'
        if fsio.exists(side):
            return [r for r in fsio.read_text(side).split('\n') if r]
        return fsio.load_npz(path, keys=['record'])['record'].tolist()

    def index(self) -> Dict[str, list]:
        """chunk path -> its record ids."""
        out = {}
        for p in self.paths():
            try:
                out[p] = self.records_of(p)
            except Exception as e:                      # a chunk from a killed run
                print(f"  ! unreadable, remove it by hand and re-run: {p} ({e})")
        return out

    def done(self) -> set:
        return {r for recs in self.index().values() for r in recs}

    def status(self, records):
        done = self.done()
        return dict(done=len(done), todo=sum(r not in done for r in records), chunks=len(self.paths()))

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
        """Read back, optionally only some arrays, some records, or some chunks (from the index)."""
        inv = {v: k for k, v in self.keymap.items()}
        want = set(records) if records is not None else None
        acc, recs = {}, []
        for p in (list(chunks) if chunks is not None else self.paths()):
            if want is not None:
                rec_here = self.records_of(p)
                if not any(r in want for r in rec_here):
                    continue
            stored = [inv.get(k, k) for k in keys] + ['record'] if keys is not None else None
            z = fsio.load_npz(p, keys=stored)
            rec = z['record']
            sel = np.ones(len(rec), bool) if want is None else np.array([r in want for r in rec])
            if not sel.any():
                continue
            use = [k for k in z if k not in META]
            for k in use:
                a = z[k][sel]
                acc.setdefault(self.keymap.get(k, k), []).append(a.astype(dtype) if a.dtype.kind == 'f' else a)
            recs.append(rec[sel])
        if not recs:
            return None
        out = {k: np.concatenate(_same_width(k, v)) for k, v in acc.items()}
        out["record"] = np.concatenate(recs)
        return out

    @staticmethod
    def row_index(run):
        """record -> row number, for aligning a run to a cohort table."""
        return {r: i for i, r in enumerate(run["record"])}
