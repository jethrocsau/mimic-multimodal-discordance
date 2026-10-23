"""`mmd` command line. Settings come from mmd.toml / MMD_* env vars (see mmd/config.py); --cohort overrides.

    mmd setup check | setup llava-med
    mmd data stage
    mmd migrate drive --src <Drive folder>
    mmd cohort build --pair cxr_notes|ecg_notes [--donor-plans <dir>] [--pins <csv>]
    mmd index import --model llava-med --legacy <run folder> [--mean-from <npz>] [--lookup name=<dir>]
    mmd weights stage --model medgemma-4b
    mmd perturb --model medgemma-4b [--smoke 3 | --plan] [--levels L1] [--families all] [--shard 0/2]
    mmd analyse --model medgemma-4b
    mmd zoo --models llava-med,medgemma-4b [--exclude-exposed] [--split-file <csv>]
    mmd collect
    mmd status
    mmd bq link
"""
import os
import subprocess
from typing import List, Optional

import numpy as np
import typer

from . import config as config_mod
from . import fsio

app = typer.Typer(add_completion=False, no_args_is_help=True)
setup_app, data_app, migrate_app, cohort_app, index_app, weights_app, bq_app = (
    typer.Typer(no_args_is_help=True) for _ in range(7))
for name, sub in (('setup', setup_app), ('data', data_app), ('migrate', migrate_app), ('cohort', cohort_app),
                  ('index', index_app), ('weights', weights_app), ('bq', bq_app)):
    app.add_typer(sub, name=name)

CohortOpt = typer.Option(None, '--cohort', help='cohort version (default from mmd.toml)')
PairOpt = typer.Option('cxr_notes', '--pair', help='cxr_notes | ecg_notes')
ModelPairOpt = typer.Option(None, '--pair', help="default: the model's own pair")


def _pair(A, pair):
    if pair and pair != A.pair:
        raise typer.BadParameter(f'{A.name} runs on --pair {A.pair}, not {pair}')
    return A.pair


def _cfg(cohort=None):
    return config_mod.load(cohort_version=cohort)


def _sh(cmd, check=True):
    print('$', ' '.join(cmd))
    return subprocess.run(cmd, check=check)


# ── setup ────────────────────────────────────────────────────────────────────
@setup_app.command('check')
def setup_check(cohort: Optional[str] = CohortOpt):
    """Print the settings and test access to the store root and the phase0 BigQuery dataset."""
    cfg = _cfg(cohort)
    for k, v in vars(cfg).items():
        print(f'  {k:16} {v}')
    if 'CHANGE-ME' in cfg.root:
        print('! set `root` in mmd.toml (or MMD_ROOT) to your private bucket, e.g. gs://<bucket>')
    try:
        fsio.fs_of(cfg.root).ls(cfg.root)
        print(f'store root reachable: {cfg.root}')
    except Exception as e:
        print(f'! store root not reachable: {cfg.root} ({e})')
    try:
        from google.cloud import bigquery
        rows = bigquery.Client(project=cfg.project).query(
            f'SELECT COUNT(*) n FROM {cfg.table("analysis_dataset")}').to_dataframe()
        print(f'BigQuery {cfg.project}.{cfg.dataset}.analysis_dataset: {int(rows.n[0]):,} rows')
    except Exception as e:
        print(f'! BigQuery not reachable ({e})')
    try:
        import torch
        print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(),
              torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
    except ImportError:
        print('torch not installed (core-only environment)')


@setup_app.command('llava-med')
def setup_llava(commit: str = typer.Option(None, help='LLaVA-Med commit the existing runs used')):
    """Install the LLaVA-Med repo code without its outdated dependency pins (after `uv sync --group llava-med`)."""
    from .models.llava_med import LLAVA_MED_COMMIT
    ref = commit or LLAVA_MED_COMMIT
    _sh(['uv', 'pip', 'install', '--no-deps', f'git+https://github.com/microsoft/LLaVA-Med.git@{ref}'])


# ── data and migration ───────────────────────────────────────────────────────
@data_app.command('stage')
def data_stage(cohort: Optional[str] = CohortOpt):
    """BigQuery phase0 tables -> parquet snapshot under {root}/data/<cohort_version>/."""
    from .data import stage
    stage(_cfg(cohort))


@migrate_app.command('drive')
def migrate_drive(src: str = typer.Option(..., help="Drive's mimic_mmd folder, e.g. /content/drive/MyDrive/mimic_mmd"),
                  cohort: Optional[str] = CohortOpt,
                  raw: bool = typer.Option(False, help='also copy unpacked/ (only if PhysioNet buckets are not readable)'),
                  dry_run: bool = typer.Option(False, '--dry-run')):
    """Copy (never delete) embeddings/ and tables/ from Drive to the bucket with resumable rsync, then verify."""
    cfg = _cfg(cohort)
    pairs = [(f'{src}/embeddings', f'{cfg.root}/emb/legacy'), (f'{src}/tables', cfg.data_dir())]
    if raw:
        pairs.append((f'{src}/unpacked', cfg.raw))
    for a, b in pairs:
        if not os.path.isdir(a):
            print(f'! {a} not found, skipped'); continue
        _sh(['du', '-sh', a], check=False)
        _sh(['gcloud', 'storage', 'rsync', '-r'] + (['--dry-run'] if dry_run else []) + [a, b])
    if dry_run:
        return
    for a, b in pairs:
        if not os.path.isdir(a):
            continue
        n_local = sum(len(f) for _, _, f in os.walk(a))
        n_remote = len([p for p in fsio.fs_of(b).find(b)])
        print(f'{"ok" if n_remote >= n_local else "MISMATCH"}: {a} {n_local:,} files -> {b} {n_remote:,} objects')


# ── cohort manifest ──────────────────────────────────────────────────────────
@cohort_app.command('build')
def cohort_build(pair: str = PairOpt, cohort: Optional[str] = CohortOpt,
                 donor_plans: Optional[str] = typer.Option(None, help='folder with donor_plan_<pair>_<L>.csv to reuse'),
                 pins: Optional[str] = typer.Option(None, help='slot_pins_x.csv from the S3 repair'),
                 note_chest_only: bool = False, force: bool = False):
    """Tiers, fixed folds, donors and every configuration, frozen under {root}/cohort/<version>/<pair>/."""
    import pandas as pd
    from .cohort import build_manifest
    from .data import Tables, slots_for
    cfg = _cfg(cohort)
    out = cfg.cohort_dir(pair)
    if fsio.exists(f'{out}/manifest.json') and not force:
        raise typer.BadParameter(f'{out} already has a manifest; a cohort version is frozen (use a new --cohort)')
    tables = Tables(cfg)
    slots = slots_for(pair, tables, note_chest_only, fsio.read_csv(pins) if pins else None)
    plans = {lv: f'{donor_plans}/donor_plan_{pair}_{lv}.csv' for lv in cfg.levels} if donor_plans else None
    m = build_manifest(cfg, pair, tables.get('analysis_dataset'), slots, plans)
    m.save(out)
    print(pd.concat([c.summary() for c in m.cohorts.values()]).to_string(index=False))
    print(pd.concat([p.coverage() for p in m.plans.values()]).to_string(index=False))
    print(pd.concat([p.balance() for p in m.plans.values()]).round(3).to_string(index=False))
    print(f'saved {out} (hash {m.hash}, {len(m.configs):,} configuration rows)')


def _manifest(cfg, pair):
    from .cohort import Manifest
    return Manifest.load(cfg.cohort_dir(pair))


# ── index ────────────────────────────────────────────────────────────────────
@index_app.command('import')
def index_import(model: str = typer.Option(...), legacy: str = typer.Option(..., help='existing run folder (read only)'),
                 pair: Optional[str] = ModelPairOpt, cohort: Optional[str] = CohortOpt,
                 mean_from: Optional[str] = typer.Option(None, help='cohort-mean npz of the legacy run (key "mean")'),
                 lookup: List[str] = typer.Option([], help='name=<dir>: a single-signal run the adapter looks '
                                                           'pre-fusion vectors up in (OpenTSLM: sp-prefix, flamingo)')):
    """Register a legacy run folder as read-only source of a model and index it. Nothing is written into it."""
    from . import index
    from .cohort import PAIRS
    from .models.base import get
    from .store import ChunkStore
    cfg = _cfg(cohort)
    pair = _pair(get(model), pair)
    run_dir = cfg.emb_dir(model)
    src = f'{run_dir}/sources.json'
    sources = fsio.read_json(src) if fsio.exists(src) else {}
    sources['read_dirs'] = sorted(set(sources.get('read_dirs', [])) | {legacy.rstrip('/')})
    for item in lookup:
        name, _, d = item.partition('=')
        if not d:
            raise typer.BadParameter(f'--lookup {item}: expected name=<dir>')
        sources.setdefault('lookups', {})[name] = d.rstrip('/')
    fsio.write_json(sources, src)
    if mean_from:
        m = fsio.load_npz(mean_from)              # every array (OpenTSLM's also has stat_mean / stat_std)
        m.setdefault('n', np.array(0))
        sig = PAIRS[pair][0]
        fsio.save_npz(m, f'{run_dir}/cohort_mean_{sig}.npz')
    store = ChunkStore.for_model(run_dir, get(model).keymap)
    man = _manifest(cfg, pair)
    idx = index.build(store, man.configs, run_ref='legacy-import')
    index.save(idx, run_dir)
    known = idx.config.notna()
    print(f'{len(idx):,} stored records indexed ({known.sum():,} in the manifest, {(~known).sum():,} not)')
    if (~known).any():
        print('  not in the manifest, e.g.', idx.record[~known].head(3).tolist())


# ── weights ──────────────────────────────────────────────────────────────────
@weights_app.command('stage')
def weights_stage(model: str = typer.Option(...), cohort: Optional[str] = CohortOpt,
                  local: str = typer.Option('~/.cache/mmd/weights', help='local folder on the VM')):
    """Pinned weights: from the bucket if staged before, else from Hugging Face (then copied to the bucket)."""
    from .models.base import get
    a = get(model)
    cfg = _cfg(cohort)
    remote = cfg.weights_dir(model, a.revision)
    dst = f'{os.path.expanduser(local)}/{model}@{a.revision}'
    if fsio.exists(f'{remote}/config.json'):
        _sh(['gcloud', 'storage', 'rsync', '-r', remote, dst])
    else:
        from huggingface_hub import snapshot_download
        snapshot_download(a.hf_id, revision=a.revision, local_dir=dst)
        _sh(['gcloud', 'storage', 'rsync', '-r', dst, remote])
    print('weights at', dst)


# ── perturb ──────────────────────────────────────────────────────────────────
@app.command()
def perturb(model: str = typer.Option(...), pair: Optional[str] = ModelPairOpt, cohort: Optional[str] = CohortOpt,
            levels: str = typer.Option('', help='comma list, default all'),
            families: str = typer.Option('all', help='comma list of temporal,matched,mean,counterfactual or all'),
            factors: bool = typer.Option(True, help='include the prompt-factor records'),
            smoke: int = typer.Option(0, help='run N records without storing'),
            plan: bool = typer.Option(False, '--plan', help='only print records / to do / GPU-hours'),
            sec_per_record: float = typer.Option(None, help='for --plan; measured by --smoke'),
            shard: str = typer.Option('', help='k/n'), device: str = 'cuda',
            weights: str = typer.Option('~/.cache/mmd/weights', help='local weights folder from `weights stage`')):
    """Generate a model's missing perturbation records into its run folder, then refresh the index."""
    from .data import NotesLoader, Tables, signal_loader
    from .index import repo_ref
    from .models.base import SampleFactory, get
    from .perturb import plan as plan_fn, run as run_fn, select, smoke as smoke_fn
    from .store import ChunkStore
    cfg = _cfg(cohort)
    A = get(model)
    pair = _pair(A, pair)
    man = _manifest(cfg, pair)
    adapter = A(cfg)
    store = ChunkStore.for_model(cfg.emb_dir(model), A.keymap)
    recs = select(man, adapter, [l for l in levels.split(',') if l] or None, families.split(','), factors)
    sh = tuple(int(x) for x in shard.split('/')) if shard else None
    if plan:
        print(plan_fn(store, recs, sec_per_record, sh).to_string(index=False)); return
    local = f'{os.path.expanduser(weights)}/{model}@{A.revision}'
    if os.path.isdir(local):
        adapter.hf_id = local
    tables = Tables(cfg)
    load, path = signal_loader(tables, man.mods[0])
    notes = NotesLoader(tables.get('x_notes'), max_chars=cfg.note_chars, keep_sections=True)
    factory = SampleFactory(man.resolver, man.mods[0], load, notes, signal_path=path)
    adapter.load(device)
    if smoke:
        smoke_fn(adapter, factory, recs, n=smoke); return
    ref_ids = sorted({d for c in man.cohorts.values() for d in c.table[f'{man.mods[0]}_first']})
    fsio.write_json(dict(**adapter.info(), cohort_hash=man.hash), f'{store.dir}/manifest.json')
    def safe(d):
        try:
            return load(d)
        except Exception as e:
            print('  skip', d, repr(e))
    run_fn(adapter, man, store, factory, recs, cfg, shard=sh, run_ref=repo_ref(),
           mean_signals=lambda: [safe(d) for d in ref_ids])


# ── analysis ─────────────────────────────────────────────────────────────────
@app.command()
def analyse(model: str = typer.Option(...), pair: Optional[str] = ModelPairOpt, cohort: Optional[str] = CohortOpt,
            charts: bool = True, selftest: bool = True):
    """Probe transfer, swap tables and charts for one model -> {root}/results|figures/<version>/<model>/<pair>/."""
    from . import index
    from .analysis.report import run_report, standard_configs
    from .models.base import get
    from .store import ChunkStore
    cfg = _cfg(cohort)
    A = get(model)
    pair = _pair(A, pair)
    man = _manifest(cfg, pair)
    run_dir = cfg.emb_dir(model)
    store = ChunkStore.for_model(run_dir, A.keymap)
    stored = fsio.read_json(f'{run_dir}/manifest.json') if fsio.exists(f'{run_dir}/manifest.json') else {}
    if stored.get('cohort_hash') not in (None, man.hash):
        raise typer.BadParameter(f'{model} was run on cohort {stored["cohort_hash"]}, the manifest is {man.hash}')
    configs = standard_configs(man)
    idx = index.load(run_dir)
    keys = [k for _, _, k, _ in A.readouts]
    run = store.load(keys=keys, records=set(configs.record),
                     chunks=index.chunks_for(idx, configs.record) if len(idx) else None)
    out, figs = cfg.results_dir(model, pair), cfg.fig_dir(model, pair)
    fsio.write_json(dict(**A(cfg).info(), sig=man.mods[0], note_blind=list(A.note_blind), cohort_hash=man.hash,
                         run_ref=index.repo_ref()), f'{out}/model.json')
    extra = A.extra_readouts(cfg, man, store, run_dir)
    rep = run_report(A, man, run, cfg, out, figs, extra=extra, charts=charts, selftest=selftest)
    print(rep.clean[['level_id', 'readout', 'clf', 'family', 'slot', 'n', 'delta_died', 'delta_survived', 'gap']]
          .round(4).to_string(index=False))


@app.command()
def zoo(models: str = typer.Option(..., help='comma list'), pair: str = PairOpt, cohort: Optional[str] = CohortOpt,
        exclude_exposed: bool = False,
        split_file: Optional[str] = typer.Option(None, help='mimic-cxr-2.0.0-split.csv(.gz): also report exposed '
                                                            'models on stays whose baseline film is not in train')):
    """Informativeness / reliance / bias across models -> {root}/results/<version>/zoo/."""
    import pandas as pd
    from .zoo import zoo_table
    cfg = _cfg(cohort)
    names = [m for m in models.split(',') if m]
    tab = zoo_table(cfg, names, pair, exclude_exposed=exclude_exposed)
    if split_file:
        man = _manifest(cfg, pair)
        split = fsio.read_csv(split_file)
        held = set(split.dicom_id[split.split != 'train'])
        stays = {s for c in man.cohorts.values() for s, d in zip(c.table.stay_id, c.table[f'{man.mods[0]}_first'])
                 if d in held}
        print(f'{len(stays):,} stays have a baseline film outside the MIMIC-CXR train split')
        tab = pd.concat([tab, zoo_table(cfg, names, pair, stays_by_model={m: stays for m in names},
                                        exclude_exposed=exclude_exposed)], ignore_index=True)
    out = cfg._p('results', cfg.cohort_version, 'zoo')
    fsio.write_csv(tab, f'{out}/zoo_{pair}.csv')
    fsio.write_parquet(tab, f'{out}/zoo_{pair}.parquet')
    cols = ['model', 'mimic_exposure', 'subset', 'level_id', 'readout', 'clf', 'auroc_signal_only', 'auroc_note_only',
            'info_share_note', 'reliance_share_note', 'bias']
    print(tab[[c for c in cols if c in tab]].round(3).to_string(index=False))


COLLECT = {'summary': 'swap4_summary_{p}_clean', 'auroc': 'swap4_auroc_{p}', 'auroc_by_draw': 'swap4_auroc_by_draw_{p}',
           'reference_shifts': 'swap4_reference_shifts_{p}', 'paired_dominance': 'swap4_paired_dominance_{p}',
           'temporal_paired': 'swap4_temporal_paired_{p}', 'selftest': 'swap4_selftest_{p}',
           'informativeness': 'informativeness_{p}'}


@app.command()
def collect(cohort: Optional[str] = CohortOpt):
    """Stack every model x pair's tables into one long table per kind -> {root}/results/<version>/all/.

    Each row gains `pair` and `signal` (the swapped modality besides the note), so one filter answers e.g.
    "counterfactual note swap, XGBoost, L1, every model"."""
    import pandas as pd
    cfg = _cfg(cohort)
    base = cfg._p('results', cfg.cohort_version)
    runs = [p.rstrip('/').split('/')[-3:-1] for p in fsio.glob(f'{base}/*/*/model.json')]
    out = f'{base}/all'
    for kind, pat in COLLECT.items():
        frames = []
        for model, pair in runs:
            f = f'{base}/{model}/{pair}/{pat.format(p=pair)}.parquet'
            if fsio.exists(f):
                sig = fsio.read_json(f'{base}/{model}/{pair}/model.json').get('sig', '')
                frames.append(fsio.read_parquet(f).assign(model=model, pair=pair, signal=sig))
        if frames:
            t = pd.concat(frames, ignore_index=True)
            fsio.write_csv(t, f'{out}/{kind}.csv')
            fsio.write_parquet(t, f'{out}/{kind}.parquet')
            print(f'{kind:18} {len(t):>8,} rows  {t.groupby(["model", "pair"]).ngroups} model x pair')
    zoo = [fsio.read_parquet(p).assign(pair=p.rsplit('zoo_', 1)[1][:-len('.parquet')])
           for p in fsio.glob(f'{base}/zoo/zoo_*.parquet')]
    if zoo:
        t = pd.concat(zoo, ignore_index=True)
        fsio.write_csv(t, f'{out}/zoo.csv')
        fsio.write_parquet(t, f'{out}/zoo.parquet')
        print(f'{"zoo":18} {len(t):>8,} rows')


@app.command()
def status(pair: str = PairOpt, cohort: Optional[str] = CohortOpt):
    """Done / to-do per model run folder and family, from each folder's index."""
    import pandas as pd
    from . import index
    from .models.base import get
    cfg = _cfg(cohort)
    man = _manifest(cfg, pair)
    need = man.input_configs()
    rows = []
    for d in fsio.glob(cfg._p('emb', cfg.cohort_version, '*')):
        try:
            if get(d.rstrip('/').split('/')[-1]).pair != pair:
                continue                                # a model of another pair
        except Exception:
            pass                                        # not a registered model: show it
        idx = index.load(d)
        have = set(idx.record)
        for fam, g in need.groupby('family'):
            rows.append(dict(model=d.rstrip('/').split('/')[-1], family=fam, records=g.record.nunique(),
                             stored=len(set(g.record) & have)))
    print(pd.DataFrame(rows).to_string(index=False) if rows else 'no model runs yet')


@bq_app.command('link')
def bq_link(pair: str = PairOpt, cohort: Optional[str] = CohortOpt, dataset: str = 'mmd_results'):
    """BigQuery external tables over the parquet results (nothing loaded or copied)."""
    from google.cloud import bigquery
    cfg = _cfg(cohort)
    client = bigquery.Client(project=cfg.project, location=cfg.location)
    kinds = ['swap4_summary', 'swap4_auroc', 'swap4_paired_dominance', 'swap4_reference_shifts',
             'informativeness', 'swap4_auroc_by_draw']
    for k in kinds:
        ext = bigquery.ExternalConfig('PARQUET')
        ext.source_uris = [cfg._p('results', cfg.cohort_version, f'*/{pair}/{k}_{pair}.parquet')]
        t = bigquery.Table(f'{cfg.project}.{dataset}.{k}_{cfg.cohort_version}_{pair}')
        t.external_data_configuration = ext
        client.create_table(t, exists_ok=True)
        print('linked', t.table_id, '->', ext.source_uris[0])


def main():
    app()


if __name__ == '__main__':
    main()
