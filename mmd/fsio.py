"""File access for gs:// and local paths alike (fsspec), plus small read/write helpers."""
import io
import json
import os

import fsspec
import numpy as np
import pandas as pd


def fs_of(path: str):
    return fsspec.core.url_to_fs(path)[0]


def is_remote(path: str) -> bool:
    return '://' in path


def exists(path: str) -> bool:
    return fs_of(path).exists(path)


def makedirs(path: str):
    if not is_remote(path):
        os.makedirs(path, exist_ok=True)     # object stores have no directories


def glob(pattern: str):
    fs = fs_of(pattern)
    out = sorted(fs.glob(pattern))
    if is_remote(pattern):
        proto = pattern.split('://', 1)[0]
        out = [p if '://' in p else f'{proto}://{p}' for p in out]
    return out


def open_(path: str, mode='rb'):
    return fsspec.open(path, mode).open()


def read_parquet(path):
    with open_(path) as f:
        return pd.read_parquet(f)


def write_parquet(df: pd.DataFrame, path):
    makedirs(os.path.dirname(path))
    with fsspec.open(path, 'wb') as f:
        df.to_parquet(f, index=False)


def read_csv(path, **kw):
    with open_(path) as f:
        return pd.read_csv(f, **kw)


def write_csv(df: pd.DataFrame, path):
    makedirs(os.path.dirname(path))
    with fsspec.open(path, 'w') as f:
        df.to_csv(f, index=False)


def read_json(path):
    with open_(path, 'r') as f:
        return json.load(f)


def write_json(obj, path):
    makedirs(os.path.dirname(path))
    with fsspec.open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)


def read_text(path) -> str:
    with open_(path, 'r') as f:
        return f.read()


def write_text(text: str, path):
    makedirs(os.path.dirname(path))
    with fsspec.open(path, 'w') as f:
        f.write(text)


def load_npz(path, keys=None):
    """All (or some) arrays of an .npz, read into memory in one go (works the same on gs://)."""
    with open_(path) as f:
        buf = io.BytesIO(f.read())
    with np.load(buf, allow_pickle=False) as z:
        return {k: z[k] for k in (keys or z.files) if k in z.files}


def save_npz(arrays: dict, path):
    """Write then move into place, so a reader never sees half a chunk."""
    makedirs(os.path.dirname(path))
    if is_remote(path):
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        with fsspec.open(path, 'wb') as f:      # GCS objects appear atomically on close
            f.write(buf.getvalue())
    else:
        np.savez(path + '.tmp.npz', **arrays)
        os.replace(path + '.tmp.npz', path)


def savefig(fig, path, dpi=200):
    makedirs(os.path.dirname(path))
    buf = io.BytesIO()
    fig.savefig(buf, dpi=dpi, bbox_inches='tight', format='png')
    with fsspec.open(path, 'wb') as f:
        f.write(buf.getvalue())
    return path
