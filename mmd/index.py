"""The embedding index: one parquet per model run folder, mapping every stored record to its chunk.

    record, stay_id, level_id, config, family, slot, draw, donor, chunk, row, run_ref

It is rebuilt from the chunks' records.txt sidecars (or their record arrays, for legacy chunks), so it is a
cache, never the only copy of anything. `perturb` uses it to skip finished records, analysis to read only
the chunks it needs, and `mmd bq link` exposes it (and the results) to BigQuery as external tables.
"""
import subprocess
from datetime import datetime, timezone

import pandas as pd

from . import fsio
from .store import ChunkStore


def index_path(run_dir):
    return f"{run_dir.rstrip('/')}/index.parquet"


def runs_path(run_dir):
    return f"{run_dir.rstrip('/')}/runs.parquet"


def build(store: ChunkStore, configs: pd.DataFrame = None, run_ref: str = '') -> pd.DataFrame:
    rows = []
    for path, recs in store.index().items():
        rows += [(r, path, i) for i, r in enumerate(recs)]
    idx = pd.DataFrame(rows, columns=['record', 'chunk', 'row'])
    idx = idx.drop_duplicates('record', keep='last')          # a record re-run later wins
    if configs is not None and len(idx):
        meta = configs.drop_duplicates('record')[['record', 'stay_id', 'level_id', 'config', 'family', 'slot',
                                                  'draw', 'donor']]
        idx = idx.merge(meta, on='record', how='left')        # NaN = stored but not in this manifest
    idx['run_ref'] = run_ref
    return idx


def save(idx: pd.DataFrame, run_dir):
    fsio.write_parquet(idx, index_path(run_dir))


def load(run_dir) -> pd.DataFrame:
    p = index_path(run_dir)
    return fsio.read_parquet(p) if fsio.exists(p) else pd.DataFrame(columns=['record', 'chunk', 'row'])


def chunks_for(idx: pd.DataFrame, records) -> list:
    return sorted(set(idx[idx.record.isin(set(records))].chunk))


def repo_ref(path='.') -> str:
    """The commit the code runs at (recorded with every run); '' outside a git checkout."""
    try:
        out = subprocess.run(['git', '-C', path, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True)
        dirty = subprocess.run(['git', '-C', path, 'status', '--porcelain', '--', 'mmd'], capture_output=True,
                               text=True).stdout.strip()
        return out.stdout.strip() + ('-dirty' if dirty else '')
    except Exception:
        return ''


def log_run(run_dir, **fields):
    p = runs_path(run_dir)
    old = fsio.read_parquet(p) if fsio.exists(p) else pd.DataFrame()
    row = pd.DataFrame([dict(logged=datetime.now(timezone.utc).isoformat(timespec='seconds'), **fields)])
    fsio.write_parquet(pd.concat([old, row.astype(str)], ignore_index=True), p)
