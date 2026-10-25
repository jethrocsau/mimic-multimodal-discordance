"""`mmd` command line: a thin layer over mmd.zoo.Zoo. Settings come from mmd.toml / MMD_* env vars (see
mmd/config.py); --cohort overrides the cohort version, --define adds modality sets for this command.

    mmd setup check | setup llava-med
    mmd data stage
    mmd modsets [--define cxr_ecg_notes=e,x,n] [--strategies ...] [--max-order 2]
    mmd cohort build --modset cxr_notes [--strategies all] [--max-order 2] [--donor-plans <dir>] [--pins <csv>]
    mmd index import --model llava-med --run-folder <notebook run> [--mean-from <npz>] [--lookup name=<dir>]
    mmd weights stage --model medgemma-4b
    mmd perturb --model medgemma-4b [--modset cxr_notes] [--smoke 3 | --plan] [--levels L1]
                [--strategies within_patient,matched_label_swap] [--max-order 1] [--shard 0/2]
    mmd analyse --model medgemma-4b [--modset cxr_notes]
    mmd zoo --modset cxr_notes [--models llava-med,medgemma-4b] [--exclude-exposed] [--split-file <csv>]
    mmd collect
    mmd status --modset cxr_notes
    mmd bq link --modset cxr_notes

Strategies: ablate, within_patient, matched_patient, matched_label_swap, cohort_mean, or all.
"""
import subprocess
from typing import List, Optional

import typer

from . import config as config_mod
from . import fsio

app = typer.Typer(add_completion=False, no_args_is_help=True)
setup_app, data_app, cohort_app, index_app, weights_app, bq_app = (
    typer.Typer(no_args_is_help=True) for _ in range(6))
for name, sub, text in (
        ('setup', setup_app, 'check settings and access; install LLaVA-Med code'),
        ('data', data_app, 'step 1: snapshot the phase0 BigQuery tables'),
        ('cohort', cohort_app, 'step 1: build the frozen cohort manifest of a modality set'),
        ('index', index_app, "step 2: register a notebook run's embeddings with a model"),
        ('weights', weights_app, 'step 2: stage pinned model weights'),
        ('bq', bq_app, 'step 4: BigQuery external tables over the results')):
    app.add_typer(sub, name=name, help=text)

CohortOpt = typer.Option(None, '--cohort', help='cohort version (default from mmd.toml)')
ModsetOpt = typer.Option('cxr_notes', '--modset', help='modality set name (mmd modsets lists them)')
ModelModsetOpt = typer.Option(None, '--modset', help="modality set (default: the model's own)")
DefineOpt = typer.Option([], '--define', help='add a modality set: name=e,x,n (repeatable)')
StrategiesOpt = typer.Option(None, '--strategies',
                             help='comma list: ablate, within_patient, matched_patient, matched_label_swap, '
                                  'cohort_mean, or all')
MaxOrderOpt = typer.Option(None, '--max-order', help='perturb at most this many modalities at once')


def _cfg(cohort=None):
    return config_mod.load(cohort_version=cohort)


def _zoo(cohort=None, define=()):
    from .zoo import Zoo
    return Zoo(_cfg(cohort), specs=define)


def _list(s: Optional[str]):
    return [x for x in (s or '').split(',') if x] or None


def _run(fn, *a, **kw):
    """Errors about the user's choices (unknown set, model that cannot take a set) as a usage error, not a trace."""
    try:
        return fn(*a, **kw)
    except (KeyError, ValueError, FileExistsError) as e:
        raise typer.BadParameter(str(e).strip('"'))


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


# ── data ───────────────────────────────────────────────────────
@data_app.command('stage')
def data_stage(cohort: Optional[str] = CohortOpt):
    """BigQuery phase0 tables -> parquet snapshot under {root}/data/<cohort_version>/."""
    from .data import stage
    stage(_cfg(cohort))


# ── modality sets and the cohort manifest ────────────────────────────────────
@app.command()
def modsets(cohort: Optional[str] = CohortOpt, define: List[str] = DefineOpt,
            strategies: Optional[str] = StrategiesOpt, max_order: Optional[int] = MaxOrderOpt):
    """The modality sets, which models can run each, and the configurations per recipient."""
    zoo = _zoo(cohort, define)
    tab = _run(zoo.compatibility, strategies, max_order)
    runs = tab[tab.can_run].groupby(['modset', 'mods', 'configs_per_recipient']).model.agg(', '.join)
    print(runs.reset_index().rename(columns={'model': 'models that can run it'}).to_string(index=False))
    idle = sorted(set(tab.modset) - set(tab.modset[tab.can_run]))
    if idle:
        print('no model takes:', ', '.join(f'{s} ({tab.mods[tab.modset == s].iloc[0]})' for s in idle))
    for s in zoo.sets:
        print(f'\n{s.name} ({s.label}):')
        print(_run(zoo.counts, s.name, strategies, min(max_order or zoo.cfg.max_order, s.k)).to_string(index=False))


@cohort_app.command('build')
def cohort_build(modset: str = ModsetOpt, cohort: Optional[str] = CohortOpt, define: List[str] = DefineOpt,
                 strategies: Optional[str] = StrategiesOpt, max_order: Optional[int] = MaxOrderOpt,
                 donor_plans: Optional[str] = typer.Option(None, help='folder with donor_plan_<set>_<L>.csv to reuse'),
                 pins: Optional[str] = typer.Option(None, help='slot_pins_x.csv from the S3 repair'),
                 note_chest_only: bool = False, force: bool = False):
    """Tiers, fixed folds, donors and every configuration, frozen under {root}/cohort/<version>/<set>/."""
    _run(_zoo(cohort, define).build_cohort, modset, donor_plans, pins, note_chest_only, _list(strategies),
         max_order, force)


# ── index and weights ────────────────────────────────────────────────────────
@index_app.command('import')
def index_import(model: str = typer.Option(...),
                 run_folder: str = typer.Option(..., help='notebook run folder, e.g. gs://<bucket>/emb/legacy/<run> '
                                                         '(read only)'),
                 modset: Optional[str] = ModelModsetOpt, cohort: Optional[str] = CohortOpt,
                 define: List[str] = DefineOpt,
                 mean_from: Optional[str] = typer.Option(None, help='cohort-mean npz of the notebook run (key "mean")'),
                 lookup: List[str] = typer.Option([], help='name=<dir>: a single-signal run the adapter looks '
                                                           'pre-fusion vectors up in (OpenTSLM: sp-prefix, flamingo)')):
    """Register a notebook run folder as a read-only source of a model and index it. Nothing is written into it."""
    _run(_zoo(cohort, define).import_notebook_run, model, run_folder, modset, mean_from, lookup)


@weights_app.command('stage')
def weights_stage(model: str = typer.Option(...), cohort: Optional[str] = CohortOpt,
                  local: str = typer.Option('~/.cache/mmd/weights', help='local folder on the VM')):
    """Pinned weights: from the bucket if staged before, else from Hugging Face (then copied to the bucket)."""
    print('weights at', _zoo(cohort).stage_weights(model, local))


# ── run, analyse, compare ────────────────────────────────────────────────────
@app.command()
def perturb(model: str = typer.Option(...), modset: Optional[str] = ModelModsetOpt,
            cohort: Optional[str] = CohortOpt, define: List[str] = DefineOpt,
            levels: str = typer.Option('', help='comma list, default all'),
            strategies: Optional[str] = StrategiesOpt, max_order: Optional[int] = MaxOrderOpt,
            factors: bool = typer.Option(True, help='include the prompt-factor records'),
            smoke: int = typer.Option(0, help='run N records without storing'),
            plan: bool = typer.Option(False, '--plan', help='only print records / to do / GPU-hours'),
            sec_per_record: float = typer.Option(None, help='for --plan; measured by --smoke'),
            shard: str = typer.Option('', help='k/n'), device: str = 'cuda',
            weights: str = typer.Option('~/.cache/mmd/weights', help='local weights folder from `weights stage`')):
    """Generate a model's missing perturbation records into its run folder, then refresh the index."""
    out = _run(_zoo(cohort, define).perturb, model, modset, _list(levels), _list(strategies), max_order, factors,
               smoke, plan, sec_per_record, tuple(int(x) for x in shard.split('/')) if shard else None, device,
               weights)
    if plan:
        print(out.to_string(index=False))


@app.command()
def analyse(model: str = typer.Option(...), modset: Optional[str] = ModelModsetOpt,
            cohort: Optional[str] = CohortOpt, define: List[str] = DefineOpt,
            strategies: Optional[str] = StrategiesOpt, charts: bool = True, selftest: bool = True):
    """Probe, effect tables and charts for one model -> {root}/results|figures/<version>/<model>/<set>/."""
    _run(_zoo(cohort, define).analyse, model, modset, _list(strategies), charts, selftest)


@app.command()
def zoo(modset: str = ModsetOpt, models: str = typer.Option('', help='comma list (default: every model that can '
                                                                     'run the set and has results)'),
        cohort: Optional[str] = CohortOpt, define: List[str] = DefineOpt, exclude_exposed: bool = False,
        split_file: Optional[str] = typer.Option(None, help='mimic-cxr-2.0.0-split.csv(.gz): also report every '
                                                            'model on stays whose baseline film is not in train')):
    """Information vs reliance per modality across models -> {root}/results/<version>/zoo/<set>."""
    tab = _run(_zoo(cohort, define).table, _list(models), modset, split_file, exclude_exposed)
    cols = ['model', 'mimic_exposure', 'subset', 'level_id', 'readout', 'clf', 'mod', 'auroc_only', 'info_share',
            'reliance_share', 'bias']
    print(tab[[c for c in cols if c in tab]].round(3).to_string(index=False) if len(tab) else 'no results yet')


@app.command()
def collect(cohort: Optional[str] = CohortOpt):
    """Stack every model x set's tables into one long table per kind -> {root}/results/<version>/all/.

    Each row gains `modset` and `mods`, so one filter answers e.g. "label swap of the note, XGBoost, L1, every
    model"."""
    _zoo(cohort).collect()


@app.command()
def status(modset: str = ModsetOpt, cohort: Optional[str] = CohortOpt, define: List[str] = DefineOpt):
    """Done / to-do per model run folder and family, from each folder's index."""
    tab = _run(_zoo(cohort, define).status, modset)
    print(tab.to_string(index=False) if len(tab) else 'no model runs yet')


@bq_app.command('link')
def bq_link(modset: str = ModsetOpt, cohort: Optional[str] = CohortOpt, dataset: str = 'mmd_results'):
    """BigQuery external tables over the parquet results (nothing loaded or copied)."""
    from google.cloud import bigquery
    from .zoo import TABLES
    cfg = _cfg(cohort)
    client = bigquery.Client(project=cfg.project, location=cfg.location)
    for k in TABLES:
        ext = bigquery.ExternalConfig('PARQUET')
        ext.source_uris = [cfg._p('results', cfg.cohort_version, f'*/{modset}/{k}.parquet')]
        t = bigquery.Table(f'{cfg.project}.{dataset}.{k}_{cfg.cohort_version}_{modset}')
        t.external_data_configuration = ext
        client.create_table(t, exists_ok=True)
        print('linked', t.table_id, '->', ext.source_uris[0])


def main():
    app()


if __name__ == '__main__':
    main()
