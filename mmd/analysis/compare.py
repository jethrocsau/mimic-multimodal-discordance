"""Check `mmd analyse` against the notebooks: the same tables, matched row by row, point estimates only.

The notebooks (modality_eda/*/..._ablations.ipynb) wrote swap4_<table>_<set>.csv into their run folder. This maps
their names onto mmd's (readouts, slot 'both' -> 'x+n', signal / note -> value_a / value_b), merges on the table's
keys, and reports per numeric column the largest |difference| and the share of rows that agree. Bootstrap CIs
(*_lo, *_hi) are skipped: they depend on row order. Rows only mmd has (ablate, order-2 arms) are not counted.
"""
import pandas as pd

from .. import fsio
from ..modsets import slot_name

# notebook readout -> mmd readout (names not listed are the same in both)
READOUTS = {
    'answer P(Yes)': 'yes probability', 'answer embedding': 'answer mean state',
    'post_all': 'prompt mean state', 'post_image': 'signal mean state', 'CXR vision mean': 'CXR encoder mean',
    'answer P(Yes) (SP)': 'yes probability (SP)', 'answer embedding (SP)': 'answer mean state (SP)',
    'state at Answer: (SP)': 'prompt last state (SP)', 'ECG prefix mean': 'ECG projector mean',
}
KEYS = ['level_id', 'readout', 'clf', 'family', 'slot']
# (mmd table, notebook file, merge keys)
TABLES = [('summary', 'swap4_summary_{set}.csv', KEYS),
          ('summary_clean', 'swap4_summary_{set}_clean.csv', KEYS),
          ('auroc', 'swap4_auroc_{set}.csv', KEYS),
          ('pairwise_dominance', 'swap4_paired_dominance_{set}.csv', ['level_id', 'readout', 'clf', 'family', 'measure'])]
TOL = (1e-4, 1e-2)


def _notebook(df: pd.DataFrame, modset) -> pd.DataFrame:
    df = df.rename(columns={'signal': 'value_a', 'note': 'value_b'}).copy()
    df['readout'] = df.readout.replace(READOUTS)
    if 'slot' in df:
        df['slot'] = df.slot.replace({'both': slot_name(modset.mods)})
    return df


def _skip(col: str) -> bool:
    return col.endswith(('_lo', '_hi')) or col in ('order', 'prevalence')


def compare_table(old: pd.DataFrame, new: pd.DataFrame, keys) -> dict:
    """{matched, notebook_rows, unmatched (key rows only the notebook has), columns: {col: (max diff, shares)}}."""
    if 'order' in new:
        new = new[new.order <= 2]
    m = old.merge(new, on=keys, suffixes=('_nb', '_mmd'))
    lost = old[keys].drop_duplicates().merge(new[keys].drop_duplicates(), on=keys, how='left', indicator=True)
    cols = {}
    for c in old.columns:
        if c in keys or _skip(c) or f'{c}_mmd' not in m or not pd.api.types.is_numeric_dtype(m[f'{c}_nb']):
            continue
        d = (m[f'{c}_nb'] - m[f'{c}_mmd']).abs()
        both_nan = m[f'{c}_nb'].isna() & m[f'{c}_mmd'].isna()
        d = d[~both_nan].fillna(float('inf'))          # NaN on one side only counts as a mismatch
        if len(d):
            cols[c] = (float(d.max()), *[float((d <= t).mean()) for t in TOL])
    return dict(matched=len(m), notebook_rows=len(old), unmatched=lost[lost._merge == 'left_only'][keys], columns=cols)


def compare_notebook(results_dir: str, notebook_dir: str, modset, tag=None) -> pd.DataFrame:
    """Print the comparison of every table and return it as one long frame (table, column, max_diff, shares)."""
    rows = []
    for table, file, keys in TABLES:
        nb_path, new_path = f'{notebook_dir.rstrip("/")}/{file.format(set=tag or modset.name)}', f'{results_dir}/{table}.csv'
        print(f'\n== {table}')
        missing = [p for p in (nb_path, new_path) if not fsio.exists(p)]
        if missing:
            print('  missing:', ', '.join(missing))
            continue
        old, new = _notebook(fsio.read_csv(nb_path), modset), fsio.read_csv(new_path)
        r = compare_table(old, new, keys)
        print(f'  rows: notebook {r["notebook_rows"]}, matched {r["matched"]}')
        if len(r['unmatched']):
            print(f'  {len(r["unmatched"])} notebook rows with no mmd row, by readout:',
                  r['unmatched'].readout.value_counts().to_dict())
        for c, (mx, *shares) in r['columns'].items():
            print(f'  {c:18} max |diff| {mx:9.2e}   ' + '   '.join(f'≤{t:g}: {s:6.1%}' for t, s in zip(TOL, shares)))
            rows.append(dict(table=table, column=c, matched=r['matched'], notebook_rows=r['notebook_rows'],
                             max_diff=mx, **{f'within_{t:g}': s for t, s in zip(TOL, shares)}))
    out = pd.DataFrame(rows)
    if len(out):
        ok = (out[f'within_{TOL[0]:g}'] > 0.99).all()
        print('\nverdict:', 'REPRODUCED (point estimates within 1e-4)' if ok else
              'differences above 1e-4: see the columns above')
    return out
