"""The adapter interface every model implements, the registry, and record -> sample resolution.

An adapter turns one Sample (signal + report text + prompt factors) into a dict of readouts. Every adapter
MUST return the common readouts, which is all the cross-model analysis reads:

    p_yes        P(Yes) / (P(Yes) + P(No)) from the logits at "Answer:"
    post_last    final-layer state at the last prompt token
    post_all     final-layer state averaged over all prompt positions
    post_signal  final-layer state averaged over the signal-token span (NaN without a signal)
    ans_emb      final-layer state averaged over the generated answer tokens
    answer       decoded answer text (str)

(post_* only where the model exposes them; OpenTSLM's stored runs have p_yes / post_last / ans_emb), and MAY add
model-specific `pre_*` readouts (pre-fusion encoder / projector). Mean swaps are optional: an adapter that can
inject the cohort-mean signal implements signal_mean() and inject_mean(). extra_readouts() adds readouts scored
at feature level on every donor draw (pre-fusion vectors looked up by the record in the signal slot).
"""
import contextlib
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from ..cohort import parse, parse_opts

COMMON = ('p_yes', 'post_last', 'post_all', 'post_signal', 'ans_emb', 'answer')
REGISTRY: Dict[str, type] = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


def get(name: str):
    if name not in REGISTRY:
        import importlib
        mod = {'llava-med': 'llava_med', 'medgemma-4b': 'medgemma', 'chexagent': 'chexagent',
               'opentslm-sp': 'opentslm', 'opentslm-sp-gemma': 'opentslm'}.get(name, name)
        importlib.import_module(f'mmd.models.{mod}')
    return REGISTRY[name]


@dataclass
class Sample:
    record: str
    signal: object = None             # PIL image / series bundle, or None
    signal_path: Optional[str] = None # file path of the signal (for models that read paths themselves)
    report: Optional[str] = None
    mean_signal: bool = False         # the signal is a placeholder; inject_mean() replaces it
    opts: Dict[str, str] = field(default_factory=dict)

    @property
    def order(self):
        return self.opts.get('order', 'default')

    @property
    def para(self):
        return int(self.opts.get('para', 0))


class SampleFactory:
    """Record id -> Sample, for every slot type. For a mean slot the recipient's own first signal is only a
    placeholder of the right shape: the adapter's inject_mean() hook replaces it, so none of it reaches the LLM."""

    def __init__(self, resolver, signal_mod: str, load_signal, notes, signal_path=None):
        self.resolver, self.sig, self.load_signal, self.notes = resolver, signal_mod, load_signal, notes
        self.signal_path = signal_path

    def __call__(self, record) -> Sample:
        sid, sl = parse(record)
        x = self.resolver.underlying(self.sig, sid, sl.get(self.sig, 'none'))
        n = self.resolver.underlying('n', sid, sl.get('n', 'none'))
        mean = x == 'mean'
        if mean:
            x = self.resolver.underlying(self.sig, sid, 'first')
        signal = self.load_signal(x) if x is not None else None
        report = self.notes.text(n) if n is not None else None
        if signal is None and report is None:
            raise ValueError(f'{record}: nothing to encode')
        return Sample(record, signal, self.signal_path(x) if (self.signal_path and x is not None) else None,
                      report, mean, parse_opts(record))


class Adapter:
    """Subclass, set the class attributes, implement load() and forward()."""
    name = ''
    hf_id = ''
    revision = 'main'
    group = ''                        # the uv dependency group it runs in
    signal = 'cxr'                    # 'cxr' | 'ecg'
    pair = 'cxr_notes'
    meta = dict(fusion='', domain='', params='', mimic_exposure='none')
    keymap: Dict[str, str] = {}       # stored key -> common key, for runs written before the common names
    readouts = [                      # (name, family, key, pca) the standard analysis probes
        ('answer P(Yes)', 'post-fusion', 'p_yes', None),
        ('answer embedding', 'post-fusion', 'ans_emb', 'pca'),
        ('post_all', 'post-fusion', 'post_all', 'pca'),
        ('post_signal', 'post-fusion', 'post_signal', 'pca'),
    ]
    note_blind = ('post_signal',)     # readouts that cannot see the note (causal attention, signal first)
    supports_mean = False
    default_order = 'signal first'    # slot order of the unflipped prompt (`~order=flip` is the other one)

    def __init__(self, cfg=None):
        self.cfg = cfg
        self.model = None

    # ── to implement ─────────────────────────────────────────────────────────
    def load(self, device='cuda'):
        raise NotImplementedError

    def forward(self, sample: Sample) -> dict:
        raise NotImplementedError

    def describe(self) -> dict:
        """Printed by the smoke test: token ids, template, module names... anything worth checking."""
        return {}

    # ── optional: mean swap ──────────────────────────────────────────────────
    def signal_mean(self, signals):
        """The cohort-mean encoder output over the reference signals: an array, or a dict with 'mean' plus extra
        arrays (saved together; the extras are passed back to inject_mean as keywords)."""
        raise NotImplementedError(f'{self.name}: no mean swap')

    @contextlib.contextmanager
    def inject_mean(self, mean: np.ndarray, **extra):
        raise NotImplementedError(f'{self.name}: no mean swap')
        yield

    # ── optional: feature-level readouts ─────────────────────────────────────
    @classmethod
    def extra_readouts(cls, cfg, manifest, store, run_dir) -> list:
        """Readouts scored on every donor draw (not only the input-level ones), e.g. LookupReadouts."""
        return []

    # ── shared ───────────────────────────────────────────────────────────────
    def info(self) -> dict:
        return dict(model=self.name, hf_id=self.hf_id, revision=self.revision, group=self.group,
                    signal=self.signal, pair=self.pair, default_order=self.default_order, **self.meta)

    def compute_fn(self, factory: SampleFactory, vector_dtype=np.float16):
        """compute(records) for ChunkStore.run: vectors fp16, p_yes fp32, failures recorded, not raised."""
        def compute(records):
            rows, ok, errs = [], [], {}
            for r in records:
                try:
                    rows.append(self.forward(factory(r)))
                    ok.append(r)
                except Exception as e:
                    errs[r] = repr(e)
            if not rows:
                print('    whole chunk failed, e.g.', next(iter(errs.items()), None))
                return dict(record=np.array([], dtype='<U1'))
            out = {}
            for k in dict.fromkeys(k for x in rows for k in x):
                v = [x.get(k) for x in rows]
                if k == 'answer':
                    out[k] = np.array(['' if a is None else a for a in v])
                elif k == 'p_yes':
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
