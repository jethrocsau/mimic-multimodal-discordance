"""Settings: where the data, the store and the results live.

Read, in order of precedence, from MMD_* environment variables, then `mmd.toml` in the working directory, then
the defaults below. `root` is a gs:// bucket path (default) or a local folder (offline / Drive fallback); every
other path is derived from it, so the same commands run on a laptop, Colab or the GPU VM.
"""
import os
import tomllib
from dataclasses import dataclass, fields
from typing import Tuple


@dataclass
class Config:
    project: str = 'ace-apex-508114-n9'        # GCP project with the phase0 BigQuery dataset
    dataset: str = 'phase0'                     # BigQuery dataset built by sql/phase0_dataform
    location: str = 'US'                        # phase0 lives in the US multi-region
    root: str = 'gs://CHANGE-ME-mmd'            # bucket (or local folder) holding data, cohort, emb, results
    cohort_version: str = 'v1'
    raw_root: str = ''                          # where CXR JPG / ECG files live; '' = {root}/physionet
    levels: Tuple[str, ...] = ('L1', 'L2', 'L3')
    n_splits: int = 5
    n_draws: int = 20                           # donors drawn per recipient (feature-level uses all)
    n_draws_input: int = 5                      # donors per recipient that get a model pass
    note_chars: int = 1200
    chunk: int = 200                            # records per saved chunk
    pca_dim: int = 512
    seeds: Tuple[int, ...] = (0, 1, 2)

    # ── derived paths ───────────────────────────────────────────────────────
    def _p(self, *parts):
        return '/'.join([self.root.rstrip('/')] + [str(p).strip('/') for p in parts])

    @property
    def raw(self):
        return self.raw_root or self._p('physionet')

    def data_dir(self):
        return self._p('data', self.cohort_version)

    def cohort_dir(self, pair):
        return self._p('cohort', self.cohort_version, pair)

    def emb_dir(self, model):
        return self._p('emb', self.cohort_version, model)

    def results_dir(self, model, pair):
        return self._p('results', self.cohort_version, model, pair)

    def fig_dir(self, model, pair):
        return self._p('figures', self.cohort_version, model, pair)

    def weights_dir(self, model, revision):
        return self._p('weights', f'{model}@{revision}')

    def table(self, name):
        return f'`{self.project}.{self.dataset}.{name}`'


def load(path: str = 'mmd.toml', **overrides) -> Config:
    vals = {}
    if os.path.exists(path):
        with open(path, 'rb') as f:
            vals.update(tomllib.load(f).get('mmd', {}))
    for f in fields(Config):
        env = os.environ.get(f'MMD_{f.name.upper()}')
        if env is not None:
            vals[f.name] = tuple(env.split(',')) if f.name in ('levels', 'seeds') else env
    vals.update({k: v for k, v in overrides.items() if v is not None})
    cfg = Config(**{k: v for k, v in vals.items() if k in {f.name for f in fields(Config)}})
    for name in ('n_splits', 'n_draws', 'n_draws_input', 'note_chars', 'chunk', 'pca_dim'):
        setattr(cfg, name, int(getattr(cfg, name)))
    cfg.levels = tuple(cfg.levels)
    cfg.seeds = tuple(int(s) for s in cfg.seeds)
    return cfg
