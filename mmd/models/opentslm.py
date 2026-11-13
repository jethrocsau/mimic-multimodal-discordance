"""OpenTSLM-SP, ECG soft-prompt checkpoints: Llama-3.2-1B (port of the OpenTSLM notebook's section 10 pass)
and Gemma-3-1B (same code, its own checkpoint; no notebook run to import).

Runs in the `opentslm` uv group. Ported unchanged from the notebook: ModelHub.load, FusedPromptBuilder,
AnswerExtractor (SP path), MeanEncoder (SP). It writes yes_probability, prompt_last_state (the final-layer state at
the last prompt position, "Answer:"), answer_mean_state and answer_text; the notebook-era chunks stored
p_yes / ans_last / ans_emb / answer, renamed on load (`keymap` and store.NOTEBOOK_KEYS).

The prompt is this model's own (the one its stored records used), not the shared wording in prompts.py:
the report sits in the clinical-context block, BEFORE the series. So `~order=flip` moves the report after the
series (into the post-prompt), and `~para=k` swaps in prompts.QUESTIONS[k] (k = 0 is the notebook question).

Differences from the notebook pass, both numerical only:
  - one record per forward by default; `perturb --batch 4` left-pads batches as the notebook did. Either way new
    records can differ from stored ones in the last float digits (bf16, and batch composition);
  - none in the cohort: slot tables, donors and records come from the same manifest.

Pre-fusion embeddings come from the notebook's separate single-ECG runs (prefix_mean of the SP run, read as
projector_mean; media of the Flamingo run), registered with `mmd index import --lookup sp-prefix=<dir> --lookup
flamingo=<dir>` and looked up by the ECG in the slot. A model without such a run (Gemma) stores `projector_mean`
itself in every record that has an ECG, and the ECG projector mean is looked up from its own run instead.
"""
import contextlib
import dataclasses

import numpy as np

from .. import fsio, prompts, series_prompts
from .base import Adapter, ReadoutSpec, Sample, register


def _opentslm():
    """The opentslm package. open-flamingo 2.x (mmd setup open-flamingo) imports open_clip at package import only
    for its CLIP factory, which OpenTSLM never uses: a stand-in module is registered when open_clip is absent."""
    import importlib.util
    import sys
    import types
    if importlib.util.find_spec('open_clip') is None:
        sys.modules.setdefault('open_clip', types.ModuleType('open_clip'))
    import opentslm
    return opentslm


class FusedPromptBuilder:
    """The notebook's three arms (ecg / notes / fused), which differ by exactly one block of the prompt."""
    EXPERT = {
        "ecg":   "You are an expert cardiologist analyzing an ECG (electrocardiogram). ",
        "notes": "You are an expert clinician reviewing a radiology report from an ICU stay. ",
        "fused": "You are an expert clinician reviewing an ECG (electrocardiogram) alongside "
                 "a radiology report. ",
    }
    TASK = {
        "ecg":   "Your task is to examine the ECG signal and answer the following question:",
        "notes": "Your task is to read the report and answer the following question:",
        "fused": "Your task is to examine the ECG signal together with the report and answer "
                 "the following question:",
    }
    DATA = {"ecg": "the ECG data", "notes": "the report", "fused": "the ECG data and the report"}
    LABEL = "This is ECG Lead {name}, it has mean {mean:.4f} and std {std:.4f}:"
    QUESTION = prompts.QUESTIONS[0]
    DIRECT = ("Instructions:\n"
              "- Answer immediately with a single word.\n"
              "- Do not explain, do not give a rationale, do not restate the question.")

    @staticmethod
    def arm(has_ecg: bool, has_note: bool) -> str:
        return "fused" if (has_ecg and has_note) else ("ecg" if has_ecg else "notes")

    def pre_post(self, arm, report, question=None, flip=False):
        ctx = ("12-lead ECG clinical recording. Signal quality: adequate for analysis."
               if arm != "notes" else "ICU admission.")
        block = f"Clinical Context: {ctx}"
        if report and not flip:
            block += f"\n\nRadiology report:\n{report}"
        pre = (f"{self.EXPERT[arm]}\n\n{block}\n\n{self.TASK[arm]}\n\n"
               f"Question: {question or self.QUESTION}\n\n{self.DIRECT}")
        post = (f"Based on your analysis of {self.DATA[arm]}, provide your answer.\n"
                f"Respond with exactly one word, Yes or No.\n\nAnswer:")
        if report and flip:                      # the report after the series
            post = f"Radiology report:\n{report}\n\n" + post
        return pre, post

    def sample(self, bundle, report, kind="sp", question=None, flip=False) -> dict:
        import torch
        arm = self.arm(bundle is not None, bool(report))
        pre, post = self.pre_post(arm, report, question, flip and arm == "fused")
        if bundle is None:                       # notes-only: no series at all
            return {"pre_prompt": pre, "time_series_text": [],
                    "time_series": [] if kind == "sp" else None, "post_prompt": post}
        texts = [self.LABEL.format(name=n, mean=m, std=s) for n, m, s in zip(bundle.names, bundle.mean, bundle.std)]
        ts = torch.from_numpy(bundle.values)
        return {"pre_prompt": pre, "time_series_text": texts,
                "time_series": [ts[i] for i in range(len(texts))] if kind == "sp" else ts,
                "post_prompt": post}


@register
class OpenTSLMSP(Adapter):
    name = 'opentslm-sp'
    hf_id = 'OpenTSLM/llama-3.2-1b-ecg-sp'
    revision = 'main'
    group = 'opentslm'
    batches = True
    signal, modalities, default_modset = 'ecg', ('e', 'v', 'l', 'n'), 'ecg_notes'
    meta = dict(fusion='prefix (soft prompt)', domain='time series', params='1B', mimic_exposure='none',
                base='llama-3.2-1b')
    keymap = {'ans_last': 'prompt_last_state'}
    readouts = [ReadoutSpec('yes probability (SP)', 'post-fusion', 'yes_probability'),
                ReadoutSpec('answer mean state (SP)', 'post-fusion', 'answer_mean_state', 'pca'),
                ReadoutSpec('prompt last state (SP)', 'post-fusion', 'prompt_last_state', 'pca')]
    lookups = [('ECG projector mean', 'pre-fusion', 'projector_mean', 'sp-prefix'),
               ('ECG Flamingo mean', 'cross-attention', 'media', 'flamingo')]
    note_blind = ('ECG projector mean', 'ECG Flamingo mean')
    supports_mean = True
    default_order = 'note first'
    kind = 'sp'
    # the ECG as OpenTSLM trained on it (PTB-XL): 100 Hz, 10 s, gaps filled, each lead z-scored; the vitals at their
    # hourly rate, gaps filled, each charted channel z-scored (never-charted channels are left out, see _vitals)
    series_spec = {'e': dict(hz=100, length=1000, fill=True, znorm=True), 'v': dict(fill=True, znorm=True)}
    answer_tokens = 8
    store_prefix = False                         # True: forward() also writes projector_mean (H,), the projector
                                                 # output averaged over leads and patches

    base_llm = None                              # the base LLM, where opentslm's own name mapping gets it wrong
    # Attention kernel of the LLM: 'eager' (opentslm's own setting, used by every stored run) or 'sdpa'
    # (PyTorch's fused kernel: the same formula, ~3.5x faster on ECG prompts of ~3.5k tokens, but bf16 sums taken
    # in another order, so P(Yes) moves by up to ~0.05 on some records; same answers). A model's records must all
    # use one kernel, so switching a model with stored records means rerunning them. Recorded in the run's
    # manifest (info()['attention']).
    attn = 'eager'

    @classmethod
    def info(cls) -> dict:
        return dict(super().info(), attention=cls.attn)

    def _set_attention(self, lm):
        """Switch the HF language model (unwrapping a LoRA wrapper) to `attn`; returns the kernel now in use."""
        base = lm.get_base_model() if hasattr(lm, 'get_base_model') else lm
        if base.config._attn_implementation != self.attn:
            if hasattr(base, 'set_attn_implementation'):
                base.set_attn_implementation(self.attn)
            else:
                base.config._attn_implementation = self.attn
        return base.config._attn_implementation

    def load(self, device='cuda'):
        import torch
        OpenTSLM = _opentslm().OpenTSLM
        self.torch = torch
        with self._base_llm(OpenTSLM):
            self.model = OpenTSLM.load_pretrained(self.hf_id, device=device, enable_lora=True)
        self.model.eval()
        self.attention = self._set_attention(self.model.llm)
        self.tok = self.model.tokenizer
        first = lambda s: self.tok.encode(s, add_special_tokens=False)[0]
        yes = {first(s) for s in (" Yes", " yes", "Yes", "yes")}
        no = {first(s) for s in (" No", " no", "No", "no")}
        self.yes_ids, self.no_ids = sorted(yes - no), sorted(no - yes)
        self.prompts = FusedPromptBuilder()
        self.stat_mean = self.stat_std = None
        self.hidden = self.model.llm.get_input_embeddings().embedding_dim
        return self

    def describe(self):
        d = lambda ids: [self.tok.decode([i]) for i in ids]
        return dict(yes_ids=self.yes_ids, yes=d(self.yes_ids), no_ids=self.no_ids, no=d(self.no_ids),
                    kind=self.kind, projector=type(self.model.projector).__name__, attention=self.attention)

    @classmethod
    def mean_modalities(cls):
        """Only the ECG has a cohort mean (its projector output); vitals records with a mean slot are skipped."""
        return ('e',) if cls.supports_mean else ()

    @contextlib.contextmanager
    def _base_llm(self, OpenTSLM):
        """opentslm maps 'gemma-3-1b-*' checkpoints to google/gemma-3-1b, which does not exist on Hugging Face (the
        checkpoints were trained on google/gemma-3-1b-pt): `base_llm` overrides the mapping while the model loads."""
        if not self.base_llm:
            yield
            return
        old = OpenTSLM.__dict__['_get_base_llm_id']
        OpenTSLM._get_base_llm_id = staticmethod(lambda repo_id: self.base_llm)
        try:
            yield
        finally:
            OpenTSLM._get_base_llm_id = old

    @staticmethod
    def multi(sample: Sample) -> bool:
        """Records with vitals or labs take series_prompts.OpenTSLMSeriesBuilder; the rest keep FusedPromptBuilder."""
        return 'v' in sample.signals or 'l' in sample.signals

    def _vitals(self, bundle):
        """The vitals window without its never-charted channels, prepared (series_spec['v']); None if nothing charted."""
        if bundle is None:
            return None
        keep = np.isfinite(np.asarray(bundle.values, float)).any(1)
        if not keep.any():
            return None
        units = bundle.units or [''] * len(bundle.names)
        b = dataclasses.replace(bundle, values=np.asarray(bundle.values)[keep],
                                names=[n for n, k in zip(bundle.names, keep) if k],
                                units=[u for u, k in zip(units, keep) if k])
        return self.series(b, 'v')

    def prompt(self, sample: Sample) -> str:
        """The text of the prompt (smoke output): pre-prompt, the series labels with [series], post-prompt."""
        d = self.sample_dict(sample)
        return '\n'.join([d['pre_prompt']] + [f'{t} [series]' for t in d['time_series_text']] + [d['post_prompt']])

    def sample_dict(self, sample: Sample) -> dict:
        if self.multi(sample):
            return self.sample_dict_multi(sample)
        b = self.series(sample.signal, 'e')
        if sample.mean_signal:                   # placeholder series; the label quotes the cohort means
            if self.stat_mean is None:
                raise RuntimeError('mean record outside inject_mean()')
            b = dataclasses.replace(b, mean=self.stat_mean, std=self.stat_std, info=dict(n_nan=0))
        return self.prompts.sample(b, sample.report, self.kind, question=prompts.QUESTIONS[sample.para],
                                   flip=sample.order == 'flip')

    def sample_dict_multi(self, sample: Sample) -> dict:
        e = self.series(sample.signals.get('e'), 'e')
        if e is not None and 'e' in sample.mean_mods:
            if self.stat_mean is None:
                raise RuntimeError('mean record outside inject_mean()')
            e = dataclasses.replace(e, mean=self.stat_mean, std=self.stat_std, info=dict(n_nan=0))
        v = self._vitals(sample.signals.get('v'))
        labs = series_prompts.labs_body(sample.signals['l']) if 'l' in sample.signals else None
        d = series_prompts.OpenTSLMSeriesBuilder().sample(e, v, labs, sample.report, self.kind,
                                                         question=prompts.QUESTIONS[sample.para],
                                                         flip=sample.order == 'flip')
        d['series_mods'] = ['e'] * (len(e.names) if e is not None else 0) + ['v'] * (len(v.names) if v is not None else 0)
        return d

    @staticmethod
    def _decoder(model):
        return model.llm.get_decoder()

    def forward(self, sample: Sample) -> dict:
        """AnswerExtractor.run for one SP record: P(Yes) at "Answer:", the state there, the answer-span mean."""
        return self.forward_batch([sample])[0]

    def forward_batch(self, samples) -> list:
        """AnswerExtractor.run over records that all have, or all lack, the ECG: one generate() call, the rows
        left-padded as the notebook's batches were. One record takes no padding at all."""
        torch = self.torch
        ss = [self.sample_dict(s) for s in samples]
        steps = []
        grab = lambda m, i, o: steps.append((o.last_hidden_state if hasattr(o, 'last_hidden_state') else o[0]).detach())
        proj = []
        kw = dict(max_new_tokens=self.answer_tokens, do_sample=False, num_beams=1, return_dict_in_generate=True,
                  output_logits=True, pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id)
        ecg_means = None
        with torch.no_grad():
            if any('series_mods' in d for d in ss):      # vitals / labs records: one at a time, own lengths
                rows, ecg_means = map(list, zip(*[self._embed_multi(d) for d in ss]))
            else:
                hp = self.model.projector.register_forward_hook(lambda m, i, o: proj.append(o.detach().float()))
                try:
                    e, a = self.model.pad_and_apply_batch(ss)
                finally:
                    hp.remove()
                rows = [e[i][a[i].bool()] for i in range(len(ss))]           # each record's own positions
            L = max(r.shape[0] for r in rows)
            embeds = rows[0].new_zeros(len(rows), L, rows[0].shape[-1])
            attn = torch.zeros(len(rows), L, dtype=torch.long, device=rows[0].device)
            for i, r in enumerate(rows):                                  # left padding: all end at "Answer:"
                embeds[i, L - r.shape[0]:] = r
                attn[i, L - r.shape[0]:] = 1
            h = self._decoder(self.model).register_forward_hook(grab)
            try:
                out = self.model.llm.generate(inputs_embeds=embeds, attention_mask=attn, **kw)
            finally:
                h.remove()
        gen_ids = out.sequences[:, -self.answer_tokens:]
        logits = out.logits[0].float()
        p_yes = torch.sigmoid(torch.logsumexp(logits[:, self.yes_ids], 1) - torch.logsumexp(logits[:, self.no_ids], 1))
        ans_last = steps[0][:, -1].float()
        if len(steps) > 1:
            hs = torch.cat(steps[1:], 1).float()
            v = torch.ones_like(gen_ids, dtype=torch.bool)
            if self.tok.eos_token_id is not None:
                v &= ~((gen_ids == self.tok.eos_token_id).cumsum(1) > 0)
            if self.tok.pad_token_id is not None:
                v &= gen_ids != self.tok.pad_token_id
            m = v[:, :hs.shape[1], None].float()
            ans_emb = (hs * m).sum(1) / m.sum(1).clamp(min=1)
        else:
            ans_emb = ans_last
        texts = self.tok.batch_decode(gen_ids, skip_special_tokens=True)
        # projector output: (B*K, P, H) lead-major per record, in one call or one per record; same K leads each
        prefix = torch.cat(proj).reshape(len(ss), -1, proj[0].shape[-1]) if proj else None
        res = []
        for i in range(len(ss)):
            cpu = lambda t: t[i].float().cpu().numpy()
            r = dict(yes_probability=float(p_yes[i]), prompt_last_state=cpu(ans_last), answer_mean_state=cpu(ans_emb),
                     answer_text=texts[i].strip())
            if self.store_prefix and ecg_means is not None:      # the ECG leads' projector output only
                r['projector_mean'] = (ecg_means[i].cpu().numpy() if ecg_means[i] is not None
                                       else np.full(self.hidden, np.nan, np.float32))
            elif self.store_prefix:              # mean over the record's (K, P) prefix; NaN when there is no ECG
                r['projector_mean'] = (prefix[i].mean(0).cpu().numpy() if prefix is not None
                                       else np.full(self.hidden, np.nan, np.float32))
            res.append(r)
        return res

    def _embed_multi(self, d):
        """pad_and_apply_batch for one vitals / labs record, except that each modality's series are encoded in their
        own call at their own length: OpenTSLM pads every series of a call to the longest, so hourly vitals (24
        points) sent with the ECG (1000) would become mostly padding. The text segments are tokenized together and
        embedded exactly as there. Returns (embeddings (L, H), the ECG leads' mean projector output (H,) or None)."""
        torch, m = self.torch, self.model
        texts = [d['pre_prompt'], *d['time_series_text'], d['post_prompt']]
        tok = m.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        emb = m.llm.get_input_embeddings()(tok.input_ids.to(m.device))
        mask = tok.attention_mask.to(m.device)
        mods, series = d['series_mods'], d['time_series']
        proj, ecg_mean = [None] * len(series), None
        for mod in dict.fromkeys(mods):                 # one call per modality (equal lengths within it)
            idx = [i for i, x in enumerate(mods) if x == mod]
            x = torch.stack([series[i] for i in idx]).to(m.device)
            rem = x.shape[1] % m.patch_size
            if rem:
                x = torch.cat([x, x.new_zeros(x.shape[0], m.patch_size - rem)], 1)
            self._encoding = mod                         # the mean swap replaces only the ECG's projector output
            try:
                p = m.projector(m.encoder(x)).to(emb.dtype)
            finally:
                self._encoding = None
            for j, i in enumerate(idx):
                proj[i] = p[j]
            if mod == 'e':
                ecg_mean = p.float().mean((0, 1))
        seq = [emb[0, :int(mask[0].sum())]]
        for i in range(len(series)):
            seq += [emb[1 + i, :int(mask[1 + i].sum())], proj[i]]
        seq.append(emb[-1, :int(mask[-1].sum())])
        return torch.cat(seq, 0), ecg_mean

    # ── mean swap (MeanEncoder, SP) ──────────────────────────────────────────
    def signal_mean(self, bundles, mod=None, batch=8):
        """Mean projector output (K, P, H) over the cohort reference ECGs, plus the mean per-lead statistics the
        label text quotes for a mean record."""
        torch = self.torch
        total, n, stats, cap = None, 0, [], {}
        hook = lambda m, i, o: cap.__setitem__('o', o.detach().float())
        with torch.no_grad():
            for b in range(0, len(bundles), batch):
                bs = [self.series(x, 'e') for x in bundles[b:b + batch] if x is not None]
                if not bs:
                    continue
                samples = [self.prompts.sample(x, None, self.kind) for x in bs]
                h = self.model.projector.register_forward_hook(hook)
                try:
                    self.model.pad_and_apply_batch(samples)
                finally:
                    h.remove()
                o = cap['o'].reshape(len(samples), -1, *cap['o'].shape[1:])          # (B, K, P, H)
                total = o.sum(0) if total is None else total + o.sum(0)
                n += o.shape[0]
                stats += [(x.mean, x.std) for x in bs]
        return dict(mean=(total / n).cpu().numpy().astype(np.float32),
                    stat_mean=np.mean([s[0] for s in stats], 0).astype(np.float32),
                    stat_std=np.mean([s[1] for s in stats], 0).astype(np.float32))

    @contextlib.contextmanager
    def inject_mean(self, mean, mod=None, stat_mean=None, stat_std=None, **_):
        if stat_mean is None or stat_std is None:
            raise ValueError('opentslm: the cohort mean needs stat_mean / stat_std (the label text)')
        mean_t = self.torch.from_numpy(np.asarray(mean))

        def swap(m, i, o):                       # o: (B*K, P, H), lead-major per sample
            if getattr(self, '_encoding', None) == 'v':
                return o                         # a vitals call of _embed_multi: not the ECG
            t = mean_t.to(o.device, o.dtype)
            assert o.shape[0] % t.shape[0] == 0 and o.shape[1:] == t.shape[1:], (o.shape, t.shape)
            return t.repeat(o.shape[0] // t.shape[0], 1, 1)
        self.stat_mean, self.stat_std = np.asarray(stat_mean), np.asarray(stat_std)
        h = self.model.projector.register_forward_hook(swap)
        try:
            yield
        finally:
            h.remove()
            self.stat_mean = self.stat_std = None

    # ── pre-fusion lookups from the single-ECG runs ──────────────────────────
    @classmethod
    def extra_readouts(cls, cfg, manifest, store, run_dir):
        """The notebook's _pre readouts: per-ECG vectors (mean over patches) from the single-modality runs,
        looked up by the ECG in the slot; the mean slot gets the average over the cohort reference ECGs."""
        from ..analysis.probe import Embeddings, path_key
        from ..store import EmbeddingStore
        src = f'{run_dir}/sources.json'
        dirs = fsio.read_json(src).get('lookups', {}) if fsio.exists(src) else {}
        need = {path_key(d, cfg.raw): d for c in manifest.cohorts.values()
                for col in ('e_first', 'e_last') for d in c.table[col].dropna()}
        ref = {path_key(d, cfg.raw) for c in manifest.cohorts.values() for d in c.table.e_first}
        out = []
        for name, family, key, source in cls.lookups:
            if source not in dirs and cls.store_prefix and key == 'projector_mean':
                out += cls._own_prefix(manifest, store, name, family)
                continue
            if source not in dirs:
                print(f'  readout {name}: no `{source}` lookup run registered (mmd index import --lookup), skipped')
                continue
            lk = EmbeddingStore(dirs[source])
            have = [r for r in lk.done() if path_key(r, cfg.raw) in need]
            run = lk.load(keys=[key], records=set(have))
            if run is None:
                continue
            vec = {need[path_key(r, cfg.raw)]: v for r, v in zip(run['record'], run[key].mean(1))}
            # sorted and in float64, so the mean slot is the same in every session (the notebook averaged over a
            # Python set, whose order moves the last float32 digits)
            mean_vec = np.mean([vec[d].astype(np.float64) for d in sorted(vec) if path_key(d, cfg.raw) in ref],
                               0).astype(np.float32)
            out.append(Embeddings.from_lookup(name, family, manifest, 'e', vec, mean_vec=mean_vec))
        return out

    @staticmethod
    def _own_prefix(manifest, store, name, family):
        """The ECG projector mean from this model's own records (any record whose ECG slot holds a real ECG)."""
        from ..analysis.probe import Embeddings, vectors_by_slot
        run = store.load(keys=['projector_mean'], records=set(manifest.analysis_configs().record))
        if run is None or 'projector_mean' not in run:
            print(f'  readout {name}: projector_mean not stored, skipped')
            return []
        vec = vectors_by_slot(run, manifest, 'e', ['projector_mean'])['projector_mean']
        ref = sorted({d for c in manifest.cohorts.values() for d in c.table.e_first if d in vec})
        mean_vec = np.mean([vec[d].astype(np.float64) for d in ref], 0).astype(np.float32)
        return [Embeddings.from_lookup(name, family, manifest, 'e', vec, mean_vec=mean_vec)]


@register
class OpenTSLMSPGemma(OpenTSLMSP):
    """The same SP pipeline on OpenTSLM's Gemma-3-1B ECG checkpoint (base google/gemma-3-1b-pt: accept its terms
    on Hugging Face first). No legacy run: every record is generated, and the ECG projector mean comes from its own
    records. No Flamingo lookup (OpenTSLM's Gemma Flamingo checkpoint has no single-ECG run here)."""
    name = 'opentslm-sp-gemma'
    hf_id = 'OpenTSLM/gemma-3-1b-pt-ecg-sp'
    base_llm = 'google/gemma-3-1b-pt'
    meta = dict(fusion='prefix (soft prompt)', domain='time series', params='1B', mimic_exposure='none',
                base='gemma-3-1b-pt')
    keymap = {}
    lookups = [('ECG projector mean', 'pre-fusion', 'projector_mean', 'sp-prefix')]
    note_blind = ('ECG projector mean',)
    store_prefix = True


@register
class OpenTSLMSP3B(OpenTSLMSP):
    """The same SP pipeline on OpenTSLM's Llama-3.2-3B ECG checkpoint (base meta-llama/Llama-3.2-3B: accept its
    terms on Hugging Face first): with opentslm-sp, the 1B -> 3B scale step of one recipe. No legacy run: the ECG
    projector mean comes from its own records, as for the Gemma checkpoint."""
    name = 'opentslm-sp-3b'
    hf_id = 'OpenTSLM/llama-3.2-3b-ecg-sp'
    meta = dict(fusion='prefix (soft prompt)', domain='time series', params='3B', mimic_exposure='none',
                base='llama-3.2-3b')
    keymap = {}
    lookups = [('ECG projector mean', 'pre-fusion', 'projector_mean', 'sp-prefix')]
    note_blind = ('ECG projector mean',)
    store_prefix = True


@register
class OpenTSLMFlamingo(OpenTSLMSP):
    """OpenTSLM-Flamingo, Llama-3.2-1B ECG checkpoint: the same series and prompts as the SP model, fused by gated
    cross-attention instead of a soft prompt. Each series is one media item; its CNN encoder + perceiver output
    conditions cross-attention layers inside the LLM, so no series token enters the sequence.

    The series of one record go in as one (K, L) tensor, so with vitals each 24 h channel is zero-padded to the ECG's
    1000 samples (as OpenTSLM's own batching pads); without an ECG they keep their length. OpenTSLMFlamingo.generate()
    returns text only, so forward() repeats its three steps (encode the series, generate, clear the conditioned
    layers) asking generate() for logits and hidden states. One record per forward, no mean swap. Readouts as for SP,
    plus `media_mean`: the perceiver output averaged over the record's series and latents (NaN without a series).
    Needs open-flamingo 2.x: `uv run --no-sync mmd setup open-flamingo` after every sync of the opentslm group."""
    name = 'opentslm-flamingo'
    hf_id = 'OpenTSLM/llama-3.2-1b-ecg-flamingo'
    meta = dict(fusion='gated cross-attention (Flamingo)', domain='time series', params='1B', mimic_exposure='none',
                base='llama-3.2-1b')
    kind = 'flamingo'
    batches = False
    supports_mean = False
    keymap = {}
    lookups = []
    readouts = OpenTSLMSP.readouts + [ReadoutSpec('series media mean', 'cross-attention', 'media_mean', 'pca')]
    note_blind = ()
    store_prefix = False

    def load(self, device='cuda'):
        import torch
        from importlib.metadata import version
        OpenTSLM = _opentslm().OpenTSLM
        if not version('open-flamingo').startswith('2.'):
            raise RuntimeError(f'open-flamingo {version("open-flamingo")} installed; OpenTSLM-Flamingo needs 2.x: '
                               f'run `uv run --no-sync mmd setup open-flamingo`')
        self.torch = torch
        with self._base_llm(OpenTSLM):
            self.model = OpenTSLM.load_pretrained(self.hf_id, device=device)
        self.model.eval()
        self.attention = self._set_attention(self.model.llm.lang_encoder)
        self.tok = self.model.text_tokenizer
        first = lambda s: self.tok.encode(s, add_special_tokens=False)[0]
        yes = {first(s) for s in (" Yes", " yes", "Yes", "yes")}
        no = {first(s) for s in (" No", " no", "No", "no")}
        self.yes_ids, self.no_ids = sorted(yes - no), sorted(no - yes)
        self.prompts = FusedPromptBuilder()
        self.stat_mean = self.stat_std = None
        self.hidden = self.model.llm.lang_encoder.get_input_embeddings().embedding_dim
        return self

    def describe(self):
        d = lambda ids: [self.tok.decode([i]) for i in ids]
        return dict(yes_ids=self.yes_ids, yes=d(self.yes_ids), no_ids=self.no_ids, no=d(self.no_ids), kind=self.kind,
                    attention=self.attention)

    def forward_batch(self, samples) -> list:
        return [self.forward(s) for s in samples]

    def forward(self, sample: Sample) -> dict:
        from .hf_vlm import readouts_from_generate
        torch, fl = self.torch, self.model.llm
        d = self.sample_dict(sample)
        d.pop('series_mods', None)
        has_series = bool(d['time_series_text'])
        if d['time_series'] is None:                 # report only: a dummy series no media token points to
            d['time_series'] = torch.zeros(1, 4)
        ids, vision, att, _ = self.model.pad_and_apply_batch([d], include_labels=True)
        media = {}
        hook = fl.perceiver.register_forward_hook(lambda m, i, o: media.__setitem__('o', o.detach().float()))
        orig = torch._dynamo.config.disable
        torch._dynamo.config.disable = True
        try:
            with torch.no_grad():
                fl._encode_vision_x(vision_x=vision)
                gen = fl.lang_encoder.generate(ids, attention_mask=att, eos_token_id=fl.eoc_token_id,
                                               max_new_tokens=self.answer_tokens, do_sample=False, num_beams=1,
                                               pad_token_id=self.tok.pad_token_id, output_hidden_states=True,
                                               output_logits=True, return_dict_in_generate=True)
        finally:
            fl.lang_encoder.clear_conditioned_layers()
            hook.remove()
            torch._dynamo.config.disable = orig
        r = readouts_from_generate(torch, gen, ids.shape[1], None, self.yes_ids, self.no_ids, self.tok,
                                   stop={fl.eoc_token_id, self.tok.eos_token_id})
        out = dict(yes_probability=r['yes_probability'], yes_log_odds=r['yes_log_odds'],
                   prompt_last_state=r['prompt_last_state'], answer_mean_state=r['answer_mean_state'],
                   answer_text=r['answer_text'])
        m = media.get('o')
        width = m.shape[-1] if m is not None else 128
        out['media_mean'] = (m.reshape(-1, width).mean(0).cpu().numpy() if has_series and m is not None
                             else np.full(width, np.nan, np.float32))
        return out


@register
class OpenTSLMFlamingoGemma(OpenTSLMFlamingo):
    """OpenTSLM-Flamingo on the Gemma-3-1B ECG checkpoint (base google/gemma-3-1b-pt, gated: accept its terms)."""
    name = 'opentslm-flamingo-gemma'
    hf_id = 'OpenTSLM/gemma-3-1b-pt-ecg-flamingo'
    base_llm = 'google/gemma-3-1b-pt'
    meta = dict(fusion='gated cross-attention (Flamingo)', domain='time series', params='1B', mimic_exposure='none',
                base='gemma-3-1b-pt')


@register
class OpenTSLMFlamingo3B(OpenTSLMFlamingo):
    """OpenTSLM-Flamingo on the Llama-3.2-3B ECG checkpoint (base meta-llama/Llama-3.2-3B, gated)."""
    name = 'opentslm-flamingo-3b'
    hf_id = 'OpenTSLM/llama-3.2-3b-ecg-flamingo'
    meta = dict(fusion='gated cross-attention (Flamingo)', domain='time series', params='3B', mimic_exposure='none',
                base='llama-3.2-3b')
