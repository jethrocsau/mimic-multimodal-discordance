"""The zoo: one orchestration layer every model plugs into, from cohort to cross-model table.

    zoo = Zoo(config.load(), specs=['cxr_ecg_notes=e,x,n'])
    zoo.compatibility()                                  which model can run which modality set, records / recipient
    zoo.build_cohort('cxr_notes')                        define: frozen manifest (perturb.build_manifest)
    zoo.perturb('gemma3-4b', 'cxr_notes')                run:    records -> adapter -> store (runner), GPU
    zoo.analyse('gemma3-4b', 'cxr_notes')                analyse: probe -> effects -> charts (analysis.report)
    zoo.table(['llava-med', 'gemma3-4b'], 'cxr_notes')   compare: information vs reliance per modality
    zoo.collect()                                        stack every model x set's tables

A model is any registered Adapter (models.base): it declares the modalities it takes, and runs every modality set
inside them. The cross-model table reduces each model to three numbers per modality m of the set:

    info_share       (AUROC of "only m" − .5)+ / Σ (AUROC − .5)+      what m tells the model on its own
    reliance_share   m's share of the summed |Δ risk| under the label-swap (counterfactual) perturbation
    bias             reliance_share − info_share: > 0 the model leans on m beyond what m alone tells it

For a two-modality set the note rows equal the notebooks' info_share_note / reliance_share_note.
`stays` restricts everything to a subset (e.g. stays whose images are in the MIMIC-CXR official test split),
recomputed from the per-stay tables; `exclude_exposed` drops models trained on MIMIC data.
"""
import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from . import fsio
from . import runner
from .analysis import effects as E
from .cohort import Manifest
from .models.base import SampleBuilder, all_models, get
from .perturb import build_manifest, count_configs
from .store import EmbeddingStore, repo_ref

TABLES = ['summary_clean', 'auroc', 'auroc_by_draw', 'reference_shifts', 'reliance', 'pairwise_dominance',
          'temporal', 'selftest', 'informativeness']


class Zoo:
    def __init__(self, cfg, specs: Iterable[str] = ()):
        self.cfg = cfg
        self.sets = cfg.modset_registry(specs)
        self._manifests = {}

    # ── models and modality sets ─────────────────────────────────────────────
    def models(self) -> dict:
        return all_models()

    def check(self, model: str, modset: Optional[str] = None):
        """(adapter class, modality set), or a clear error when the model cannot take the set."""
        A = get(model)
        ms = self.sets[modset or A.default_modset]
        if not ms.fits(A.modalities):
            raise ValueError(f'{model} takes {"+".join(A.modalities)}; {ms.name} needs {"+".join(ms.mods)}. '
                             f'Sets it can run: {", ".join(s.name for s in self.sets.compatible(A.modalities))}')
        return A, ms

    def counts(self, modset: str, strategies=None, max_order=None) -> pd.DataFrame:
        ms = self.sets[modset]
        return count_configs(ms.k, strategies or self.cfg.strategies, max_order or self.cfg.max_order,
                             self.cfg.n_draws_model)

    def compatibility(self, strategies=None, max_order=None) -> pd.DataFrame:
        rows = []
        for ms in self.sets:
            n = int(self.counts(ms.name, strategies, min(max_order or self.cfg.max_order, ms.k)).configs.sum())
            for name, A in self.models().items():
                rows.append(dict(modset=ms.name, mods='+'.join(ms.mods), model=name, can_run=ms.fits(A.modalities),
                                 model_takes='+'.join(A.modalities), configs_per_recipient=n))
        return pd.DataFrame(rows)

    def manifest(self, modset: str) -> Manifest:
        if modset not in self._manifests:
            self._manifests[modset] = Manifest.load(self.cfg.cohort_dir(modset))
        return self._manifests[modset]

    # ── define: the frozen cohort manifest ───────────────────────────────────
    def build_cohort(self, modset: str, donor_plan_dir=None, pins=None, note_chest_only=False, strategies=None,
                     max_order=None, force=False) -> Manifest:
        from .data import Tables, slots_for
        ms = self.sets[modset]
        out = self.cfg.cohort_dir(ms.name)
        if fsio.exists(f'{out}/manifest.json') and not force:
            raise FileExistsError(f'{out} already has a manifest; a cohort version is frozen (use a new --cohort)')
        tables = Tables(self.cfg)
        slots = slots_for(ms, tables, note_chest_only, fsio.read_csv(pins) if pins else None)
        m = build_manifest(self.cfg, ms, tables.get('analysis_dataset'), slots, donor_plan_dir, strategies, max_order)
        m.save(out)
        self._manifests[ms.name] = m
        print(pd.concat([c.summary() for c in m.cohorts.values()]).to_string(index=False))
        print(pd.concat([p.coverage() for p in m.plans.values()]).to_string(index=False))
        print(pd.concat([p.balance() for p in m.plans.values()]).round(3).to_string(index=False))
        print(m.configs.groupby(['family', 'order']).config.nunique().rename('configs').to_string())
        print(f'saved {out} (hash {m.hash}, {len(m.configs):,} configuration rows)')
        return m

    # ── run: records -> adapter -> store ─────────────────────────────────────
    def import_notebook_run(self, model: str, run_folder: str, modset=None, mean_from=None,
                            lookups: Iterable[str] = ()):
        """Register a notebook run folder (copied from Drive to emb/legacy/) as a read-only source of a model and
        index it, so its stored embeddings count and are never regenerated. Nothing is written into the folder."""
        A, ms = self.check(model, modset)
        run_dir = self.cfg.emb_dir(model)
        store = EmbeddingStore.for_model(run_dir, A.keymap)
        sources = store.sources()
        sources['read_dirs'] = sorted(set(sources.get('read_dirs', [])) | {run_folder.rstrip('/')})
        for item in lookups:
            name, _, d = item.partition('=')
            if not d:
                raise ValueError(f'--lookup {item}: expected name=<dir>')
            sources.setdefault('lookups', {})[name] = d.rstrip('/')
        fsio.write_json(sources, f'{run_dir}/sources.json')
        if mean_from:
            m = fsio.load_npz(mean_from)              # every array (OpenTSLM's also has stat_mean / stat_std)
            m.setdefault('n', np.array(0))
            fsio.save_npz(m, runner.mean_path(run_dir, ms.signals[0], ms.name))
        store = EmbeddingStore.for_model(run_dir, A.keymap)
        idx = store.build_index(run_ref='notebook-import')
        store.save_index(idx)
        known = idx.record.isin(set(self.manifest(ms.name).configs.record))
        print(f'{len(idx):,} stored records indexed ({known.sum():,} in the {ms.name} manifest, {(~known).sum():,} not)')
        if (~known).any():
            print('  not in the manifest, e.g.', idx.record[~known].head(3).tolist())
        return idx

    def stage_weights(self, model: str, local='~/.cache/mmd/weights') -> str:
        """Pinned weights: from the bucket if staged before, else from Hugging Face (then copied to the bucket)."""
        import subprocess
        A = get(model)
        remote = self.cfg.weights_dir(model, A.revision)
        dst = f'{os.path.expanduser(local)}/{model}@{A.revision}'
        if fsio.exists(f'{remote}/config.json'):
            subprocess.run(['gcloud', 'storage', 'rsync', '-r', remote, dst], check=True)
        else:
            from huggingface_hub import snapshot_download
            snapshot_download(A.hf_id, revision=A.revision, local_dir=dst)
            subprocess.run(['gcloud', 'storage', 'rsync', '-r', dst, remote], check=True)
        return dst

    def perturb(self, model: str, modset=None, levels=None, strategies=None, max_order=None, factors=True,
                smoke: int = 0, plan=False, sec_per_record=None, shard=None, device='cuda',
                weights='~/.cache/mmd/weights'):
        """Generate a model's missing records for one modality set into its run folder, then refresh the index.
        strategies / max_order narrow the manifest's configurations (default: all of them)."""
        from .data import NotesLoader, Tables, signal_loaders
        A, ms = self.check(model, modset)
        man = self.manifest(ms.name)
        store = EmbeddingStore.for_model(self.cfg.emb_dir(model), A.keymap)
        recs = runner.select(man, A, levels, strategies or 'all', max_order, factors)
        if plan:
            return runner.plan(store, recs, sec_per_record, shard)
        adapter = A(self.cfg)
        local = f'{os.path.expanduser(weights)}/{model}@{A.revision}'
        if os.path.isdir(local):
            adapter.hf_id = local
        tables = Tables(self.cfg)
        loaders = signal_loaders(tables, ms.mods)
        notes = NotesLoader(tables.get('x_notes'), max_chars=self.cfg.note_chars, keep_sections=True)
        builder = SampleBuilder(man, loaders, notes)
        adapter.load(device)
        if smoke:
            return runner.smoke(adapter, builder, recs, n=smoke)[0]
        store.record_cohort(A.info(), ms.name, man.hash)

        def reference_signals(mod):
            load = loaders[mod][0]
            for d in sorted({d for c in man.cohorts.values() for d in c.table[f'{mod}_first']}):
                try:
                    yield load(d)
                except Exception as e:
                    print('  skip', d, repr(e))
        return runner.run(adapter, man, store, builder, recs, self.cfg, shard=shard,
                          reference_signals=lambda mod: list(reference_signals(mod)), run_ref=repo_ref())

    # ── analyse ──────────────────────────────────────────────────────────────
    def analyse(self, model: str, modset=None, strategies=None, charts=True, selftest=True):
        from .analysis.report import analyse_model
        A, ms = self.check(model, modset)
        man = self.manifest(ms.name)
        run_dir = self.cfg.emb_dir(model)
        store = EmbeddingStore.for_model(run_dir, A.keymap)
        h = store.cohort_hash(ms.name)
        if h not in (None, man.hash):
            raise ValueError(f'{model} was run on cohort {h}, the {ms.name} manifest is {man.hash}')
        configs = man.analysis_configs()
        run = store.load(keys=[s.key for s in A.readouts], records=set(configs.record),
                         chunks=store.chunks_for(store.load_index(), configs.record))
        out, figs = self.cfg.results_dir(model, ms.name), self.cfg.fig_dir(model, ms.name)
        fsio.write_json(dict(**A.info(), modset=ms.name, mods=list(ms.mods), note_blind=list(A.note_blind),
                             cohort_hash=man.hash, run_ref=repo_ref()), f'{out}/model.json')
        extra = A.extra_readouts(self.cfg, man, store, run_dir)
        rep = analyse_model(A, man, run, self.cfg, out, figs, extra=extra, strategies=strategies, charts=charts,
                            selftest=selftest)
        print(rep.clean[['level_id', 'readout', 'clf', 'family', 'slot', 'n', 'delta_died', 'delta_survived', 'gap']]
              .round(4).to_string(index=False))
        return rep

    def analyse_all(self, models=None, modsets=None, **kw):
        """Analyse every (model, set) pair that fits and has stored records."""
        for name, A in self.models().items():
            if models and name not in models:
                continue
            for ms in self.sets.compatible(A.modalities):
                if (modsets and ms.name not in modsets) or not fsio.exists(f'{self.cfg.cohort_dir(ms.name)}/manifest.json'):
                    continue
                print(f'── {name} · {ms.name}')
                self.analyse(name, ms.name, **kw)

    def status(self, modset: str) -> pd.DataFrame:
        """Records needed vs stored, per model run folder and family."""
        need = self.manifest(modset).model_configs()
        ms = self.sets[modset]
        rows = []
        for d in fsio.glob(self.cfg._p('emb', self.cfg.cohort_version, '*')):
            name = d.rstrip('/').split('/')[-1]
            try:
                if not ms.fits(get(name).modalities):
                    continue                                # a model that cannot take this set
            except KeyError:
                pass                                        # not a registered model: show it
            have = set(EmbeddingStore(d).load_index().record)
            for fam, g in need.groupby('family'):
                rows.append(dict(model=name, family=fam, records=g.record.nunique(),
                                 stored=len(set(g.record) & have)))
        return pd.DataFrame(rows)

    def compare_notebook(self, model: str, notebook_dir: str, modset=None) -> pd.DataFrame:
        """Check this model's analyse tables against the notebook's swap4_* CSVs in notebook_dir; the comparison is
        also saved as notebook_comparison.csv next to the results."""
        from .analysis.compare import compare_notebook
        _, ms = self.check(model, modset)
        out_dir = self.cfg.results_dir(model, ms.name)
        tab = compare_notebook(out_dir, notebook_dir, ms)
        if len(tab):
            fsio.write_csv(tab, f'{out_dir}/notebook_comparison.csv')
        return tab

    # ── compare ──────────────────────────────────────────────────────────────
    def load_results(self, model: str, modset: str):
        d = self.cfg.results_dir(model, modset)
        return (fsio.read_json(f'{d}/model.json'), fsio.read_parquet(f'{d}/informativeness_per_stay.parquet'),
                fsio.read_parquet(f'{d}/scores.parquet'))

    def model_table(self, model: str, modset: str, stays=None) -> pd.DataFrame:
        meta, info_per, scores = self.load_results(model, modset)
        ms = self.sets[modset]
        if stays is not None:
            info_per = info_per[info_per.stay_id.isin(stays)]
            scores = scores[scores.stay_id.isin(stays)]
        shares = E.information_shares(E.arm_auroc(info_per), ms.mods)
        ctx = E.AnalysisContext(model_name=meta['model'], modset=ms, cohorts={},
                                configs=pd.DataFrame(columns=['slot', 'family']),
                                note_blind=set(meta.get('note_blind', [])))
        rel, _ = E.reliance(ctx, scores)
        if shares.empty:
            return shares
        on = ['level_id', 'readout', 'clf', 'mod']
        if len(rel):
            cf = rel[(rel.family == 'counterfactual') & (rel.measure == '|risk change|')]
            ls = rel[(rel.family == E.LABEL_SPECIFIC) & (rel.measure == 'outcome pull')]
            shares = shares.merge(cf[on + ['share', 'share_lo', 'share_hi', 'n_paired']].rename(columns=dict(
                share='reliance_share', share_lo='reliance_lo', share_hi='reliance_hi')), on=on, how='left')
            shares = shares.merge(ls[on + ['mean']].rename(columns={'mean': 'label_specific_pull'}), on=on, how='left')
            shares['bias'] = shares.reliance_share - shares.info_share
        info = {k: meta.get(k) for k in ('model', 'fusion', 'domain', 'params', 'mimic_exposure')}
        return shares.assign(**info, modset=ms.name, mods='+'.join(ms.mods))

    def table(self, models: Iterable[str] = None, modset: str = 'cxr_notes', split_file=None,
              exclude_exposed=False, save=True) -> pd.DataFrame:
        """The cross-model table for one modality set (default: every model that can run it and has results)."""
        ms = self.sets[modset]
        models = list(models or [n for n, A in self.models().items() if ms.fits(A.modalities)])
        models = [m for m in models if fsio.exists(f'{self.cfg.results_dir(m, ms.name)}/model.json')]
        out = []
        for m in models:
            meta = fsio.read_json(f'{self.cfg.results_dir(m, ms.name)}/model.json')
            if exclude_exposed and meta.get('mimic_exposure', 'none') != 'none':
                continue
            out.append(self.model_table(m, ms.name).assign(subset='all'))
            if split_file:
                stays = self.held_out_stays(ms.name, split_file)
                out.append(self.model_table(m, ms.name, stays).assign(subset='official split'))
        tab = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
        if save and len(tab):
            fsio.write_csv(tab, f'{self.cfg.zoo_dir()}/{ms.name}.csv')
            fsio.write_parquet(tab, f'{self.cfg.zoo_dir()}/{ms.name}.parquet')
        return tab

    def held_out_stays(self, modset: str, split_file: str) -> set:
        """Stays whose baseline CXR is outside the MIMIC-CXR train split (for models trained on MIMIC-CXR)."""
        man = self.manifest(modset)
        if 'x' not in man.mods:
            raise ValueError(f'--split-file needs a set with a CXR image; {modset} has {man.mods}')
        split = fsio.read_csv(split_file)
        held = set(split.dicom_id[split.split != 'train'])
        stays = {s for c in man.cohorts.values() for s, d in zip(c.table.stay_id, c.table.x_first) if d in held}
        print(f'{len(stays):,} stays have a baseline film outside the MIMIC-CXR train split')
        return stays

    def collect(self) -> dict:
        """Stack every model x set's tables into one long table per kind -> results/<version>/all/<kind>."""
        base = self.cfg._p('results', self.cfg.cohort_version)
        runs = [p.rstrip('/').split('/')[-3:-1] for p in fsio.glob(f'{base}/*/*/model.json')]
        out = {}
        for kind in TABLES:
            frames = []
            for model, modset in runs:
                f = f'{base}/{model}/{modset}/{kind}.parquet'
                if fsio.exists(f):
                    mods = '+'.join(fsio.read_json(f'{base}/{model}/{modset}/model.json').get('mods', []))
                    frames.append(fsio.read_parquet(f).assign(model=model, modset=modset, mods=mods))
            if frames:
                out[kind] = t = pd.concat(frames, ignore_index=True)
                fsio.write_csv(t, f'{base}/all/{kind}.csv')
                fsio.write_parquet(t, f'{base}/all/{kind}.parquet')
                print(f'{kind:20} {len(t):>8,} rows  {t.groupby(["model", "modset"]).ngroups} model x set')
        zoo = [fsio.read_parquet(p) for p in fsio.glob(f'{self.cfg.zoo_dir()}/*.parquet')]
        if zoo:
            out['zoo'] = t = pd.concat(zoo, ignore_index=True)
            fsio.write_csv(t, f'{base}/all/zoo.csv')
            fsio.write_parquet(t, f'{base}/all/zoo.parquet')
            print(f'{"zoo":20} {len(t):>8,} rows')
        return out
