"""The adapter interface every model implements, the model registry, and record -> sample resolution.

An adapter turns one Sample (signals + report text + prompt factors) into a dict of readouts. Every adapter MUST
return the common readouts, which is all the cross-model analysis reads (all final-layer hidden states):

    yes_probability     P(Yes) / (P(Yes) + P(No)) from the logits at "Answer:"
    prompt_last_state   state at the last prompt token ("Answer:")
    prompt_mean_state   state averaged over every prompt token
    signal_mean_state   state averaged over the signal's tokens (image / ECG span; NaN without a signal)
    answer_mean_state   state averaged over the generated answer tokens
    answer_text         the decoded answer

(prompt/signal mean states only where the model exposes them; OpenTSLM's stored runs have yes_probability /
prompt_last_state / answer_mean_state), and MAY add pre-fusion vectors (encoder_mean, projector_mean, ...) and
yes_log_odds (log P(Yes) - log P(No): keeps the ranking where yes_probability saturates at 1.0).
Mean swaps are optional: an adapter that can inject the cohort-mean signal sets `supports_mean` and implements
signal_mean() and inject_mean(). extra_readouts() adds embeddings scored at feature level on every donor draw
(pre-fusion vectors looked up by the record in the signal slot).

A model plugs into the zoo by subclassing Adapter, decorating the class with @register, and adding its module to
MODULES. `modalities` says which modality keys it takes; it can run every modality set inside that.
"""
import contextlib
import importlib
from dataclasses import dataclass, field
from typing import Dict, NamedTuple, Optional

import numpy as np

from ..modsets import NOTE
from ..records import parse_options, parse_record

COMMON = ('yes_probability', 'prompt_last_state', 'prompt_mean_state', 'signal_mean_state', 'answer_mean_state',
          'answer_text')
REGISTRY: Dict[str, type] = {}
MODULES = {'llava-med': 'llava_med', 'gemma3-4b': 'gemma', 'qwen2.5-vl-7b': 'qwen_vl',
           'opentslm-sp': 'opentslm', 'opentslm-sp-gemma': 'opentslm'}   # model name -> module under mmd.models


class ReadoutSpec(NamedTuple):
    name: str                   # shown in tables and charts
    family: str                 # pre-fusion | projector | cross-attention | post-fusion
    key: str                    # the stored array
    pca: Optional[str] = None   # 'pca' = reduce to cfg.pca_dim before probing


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def get(name: str):
    if name not in REGISTRY:
        if name not in MODULES:
            raise KeyError(f'unknown model {name!r}; known: {", ".join(MODULES)}')
        importlib.import_module(f'mmd.models.{MODULES[name]}')
    return REGISTRY[name]


def all_models() -> Dict[str, type]:
    """Every adapter in MODULES (importing them is cheap: torch and model code load in load())."""
    for name in MODULES:
        get(name)
    return dict(REGISTRY)


@dataclass
class Sample:
    record: str
    signals: Dict[str, object] = field(default_factory=dict)        # mod -> PIL image / series bundle
    signal_paths: Dict[str, str] = field(default_factory=dict)      # mod -> file (for models that read paths)
    report: Optional[str] = None
    mean_mods: frozenset = frozenset()   # signals that are placeholders; inject_mean() replaces them
    opts: Dict[str, str] = field(default_factory=dict)

    def _one(self, d):
        if len(d) > 1:
            raise ValueError(f'{self.record}: {len(d)} signals; this adapter takes one (read sample.signals)')
        return next(iter(d.values()), None)

    @property
    def signal(self):
        """The one signal, for adapters that take a single signal modality (None when it is removed)."""
        return self._one(self.signals)

    @property
    def signal_path(self):
        return self._one(self.signal_paths)

    @property
    def mean_signal(self) -> bool:
        return bool(self.mean_mods)

    @property
    def order(self):
        return self.opts.get('order', 'default')

    @property
    def para(self):
        return int(self.opts.get('para', 0))


class SampleBuilder:
    """Record id -> Sample, for every slot type. A mean slot loads the recipient's own first signal only as a
    placeholder of the right shape: the adapter's inject_mean() hook replaces it, so none of it reaches the LLM."""

    def __init__(self, manifest, loaders: Dict[str, tuple], notes):
        """loaders: {mod: (load, path)} for the signal modalities (data.signal_loaders); notes: data.NotesLoader."""
        self.manifest, self.loaders, self.notes = manifest, loaders, notes

    def __call__(self, record) -> Sample:
        sid, slots = parse_record(record)
        signals, paths, mean = {}, {}, set()
        for m, (load, path) in self.loaders.items():
            u = self.manifest.resolve(m, sid, slots.get(m, 'none'))
            if u == 'mean':
                mean.add(m)
                u = self.manifest.resolve(m, sid, 'first')
            if u is not None:
                signals[m] = load(u)
                paths[m] = path(u) if path else None
        n = self.manifest.resolve(NOTE, sid, slots.get(NOTE, 'none')) if NOTE in self.manifest.mods else None
        report = self.notes.text(n) if n is not None else None
        if not signals and report is None:
            raise ValueError(f'{record}: nothing to encode')
        return Sample(record, signals, paths, report, frozenset(mean), parse_options(record))


class Adapter:
    """Subclass, set the class attributes, implement load() and forward()."""
    name = ''
    hf_id = ''
    revision = 'main'
    group = ''                        # the uv dependency group it runs in
    signal = 'cxr'                    # prompt wording for the signal (prompts.SIGNALS): 'cxr' | 'ecg'
    modalities = ('x', 'n')           # the modality keys it takes; it runs every modality set inside them
    default_modset = 'cxr_notes'
    meta = dict(fusion='', domain='', params='', mimic_exposure='none')
    keymap: Dict[str, str] = {}       # stored key -> common key, for runs written before the common names
    readouts = [                      # the embeddings the standard analysis probes
        ReadoutSpec('yes probability', 'post-fusion', 'yes_probability'),
        ReadoutSpec('answer mean state', 'post-fusion', 'answer_mean_state', 'pca'),
        ReadoutSpec('prompt mean state', 'post-fusion', 'prompt_mean_state', 'pca'),
        ReadoutSpec('signal mean state', 'post-fusion', 'signal_mean_state', 'pca'),
    ]
    note_blind = ('signal mean state',)   # readouts that cannot see the note (causal attention, signal first)
    supports_mean = False
    default_order = 'signal first'    # slot order of the unflipped prompt (`~order=flip` is the other one)

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.model = None

    @classmethod
    def mean_modalities(cls):
        """The signal modalities this adapter can replace with the cohort mean."""
        return tuple(m for m in cls.modalities if m != NOTE) if cls.supports_mean else ()

    # ── to implement ─────────────────────────────────────────────────────────
    def load(self, device='cuda'):
        raise NotImplementedError

    def forward(self, sample: Sample) -> dict:
        raise NotImplementedError

    def describe(self) -> dict:
        """Printed by the smoke test: token ids, template, module names... anything worth checking."""
        return {}

    # ── optional: mean swap ──────────────────────────────────────────────────
    def signal_mean(self, signals, mod=None):
        """The cohort-mean encoder output of modality `mod` over the reference signals: an array, or a dict with
        'mean' plus extra arrays (saved together; the extras are passed back to inject_mean as keywords)."""
        raise NotImplementedError(f'{self.name}: no mean swap')

    @contextlib.contextmanager
    def inject_mean(self, mean: np.ndarray, mod=None, **extra):
        raise NotImplementedError(f'{self.name}: no mean swap')
        yield

    # ── optional: feature-level embeddings ───────────────────────────────────
    @classmethod
    def extra_readouts(cls, cfg, manifest, store, run_dir) -> list:
        """Embeddings scored on every donor draw (not only the model-pass ones), e.g. Embeddings.from_lookup."""
        return []

    # ── shared ───────────────────────────────────────────────────────────────
    @classmethod
    def info(cls) -> dict:
        return dict(model=cls.name, hf_id=cls.hf_id, revision=cls.revision, group=cls.group, signal=cls.signal,
                    modalities=list(cls.modalities), default_modset=cls.default_modset,
                    default_order=cls.default_order, **cls.meta)

    def compute_fn(self, builder: SampleBuilder, vector_dtype=np.float16):
        """compute(records) for EmbeddingStore.run: vectors fp16, yes_probability fp32, failures recorded."""
        def compute(records):
            rows, ok, errs = [], [], {}
            for r in records:
                try:
                    rows.append(self.forward(builder(r)))
                    ok.append(r)
                except Exception as e:
                    errs[r] = repr(e)
            if not rows:
                print('    whole chunk failed, e.g.', next(iter(errs.items()), None))
                return dict(record=np.array([], dtype='<U1'))
            out = {}
            for k in dict.fromkeys(k for x in rows for k in x):
                v = [x.get(k) for x in rows]
                if k == 'answer_text':
                    out[k] = np.array(['' if a is None else a for a in v])
                elif k == 'yes_probability':
                    out[k] = np.array(v, np.float32)
                elif all(a is None or np.ndim(a) == 0 for a in v):
                    out[k] = np.array([np.nan if a is None else a for a in v])
                else:
                    out[k] = _stack_vectors(k, v).astype(vector_dtype)
            out['record'] = np.array(ok)
            if errs:
                out.update(failed=np.array(list(errs)), failed_reason=np.array(list(errs.values())))
            return out
        return compute


def _stack_vectors(key, values):
    """Stack per-record arrays; a missing or all-NaN placeholder of another shape becomes NaN of the real shape
    (e.g. a text-only record's pre-fusion vector), so one chunk can mix records with and without a signal."""
    arrs = [None if a is None else np.asarray(a, np.float32) for a in values]
    real = [a for a in arrs if a is not None and not np.isnan(a).all()]
    shape = real[0].shape if real else next(a.shape for a in arrs if a is not None)
    out = []
    for a in arrs:
        if a is None or (a.shape != shape and np.isnan(a).all()):
            out.append(np.full(shape, np.nan, np.float32))
        elif a.shape != shape:
            raise ValueError(f'{key}: records with different shapes {a.shape} vs {shape}')
        else:
            out.append(a)
    return np.stack(out)


def to_np(t):
    """torch tensor / number -> numpy (float32), without importing torch here."""
    if hasattr(t, 'detach'):
        return t.detach().float().cpu().numpy()
    return np.asarray(t)
