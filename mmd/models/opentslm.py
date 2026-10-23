"""OpenTSLM-SP, ECG soft-prompt checkpoints: Llama-3.2-1B (port of the OpenTSLM notebook's section 10 pass)
and Gemma-3-1B (same code, its own checkpoint; no notebook run to import).

Runs in the `opentslm` uv group. Ported unchanged from the notebook: ModelHub.load, FusedPromptBuilder,
AnswerExtractor (SP path), MeanEncoder (SP). It writes the same keys as the stored swap4 run
(p_yes, ans_last, ans_emb, answer), and `keymap` renames ans_last -> post_last, the final-layer state at the
last prompt position ("Answer:").

The prompt is this model's own (the one its stored records used), not the shared wording in prompts.py:
the report sits in the clinical-context block, BEFORE the series. So `~order=flip` moves the report after the
series (into the post-prompt), and `~para=k` swaps in prompts.QUESTIONS[k] (k = 0 is the notebook question).

Differences from the notebook pass, both numerical only:
  - one record per forward (the notebook left-padded batches of 4), so new records can differ from stored
    ones in the last float digits;
  - none in the cohort: slot tables, donors and records come from the same manifest.

Pre-fusion readouts come from the notebook's separate single-ECG runs (prefix_mean of the SP run, media of the
Flamingo run), registered with `mmd index import --lookup sp-prefix=<dir> --lookup flamingo=<dir>` and looked
up by the ECG in the slot. A model without such a run (Gemma) stores `prefix_mean` itself in every record that
has an ECG, and the ECG prefix mean is looked up from its own run instead.
"""
import contextlib
import dataclasses

import numpy as np

from .. import fsio, prompts
from .base import Adapter, Sample, register


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
        ts = torch.from_numpy(bundle.norm)
        return {"pre_prompt": pre, "time_series_text": texts,
                "time_series": [ts[i] for i in range(len(texts))] if kind == "sp" else ts,
                "post_prompt": post}


@register
class OpenTSLMSP(Adapter):
    name = 'opentslm-sp'
    hf_id = 'OpenTSLM/llama-3.2-1b-ecg-sp'
    revision = 'main'
    group = 'opentslm'
    signal, pair = 'ecg', 'ecg_notes'
    meta = dict(fusion='prefix (soft prompt)', domain='time series', params='1B', mimic_exposure='none',
                base='llama-3.2-1b')
    keymap = {'ans_last': 'post_last'}
    readouts = [('answer P(Yes) (SP)', 'post-fusion', 'p_yes', None),
                ('answer embedding (SP)', 'post-fusion', 'ans_emb', 'pca'),
                ('state at Answer: (SP)', 'post-fusion', 'post_last', 'pca')]
    lookups = [('ECG prefix mean', 'pre-fusion', 'prefix_mean', 'sp-prefix'),
               ('ECG Flamingo mean', 'cross-attention', 'media', 'flamingo')]
    note_blind = ('ECG prefix mean', 'ECG Flamingo mean')
    supports_mean = True
    default_order = 'note first'
    kind = 'sp'
    answer_tokens = 8
    store_prefix = False                         # True: forward() also writes prefix_mean (H,), the projector
                                                 # output averaged over leads and patches

    def load(self, device='cuda'):
        import torch
        from opentslm import OpenTSLM
        self.torch = torch
        self.model = OpenTSLM.load_pretrained(self.hf_id, device=device, enable_lora=True)
        self.model.eval()
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
                    kind=self.kind, projector=type(self.model.projector).__name__)

    def sample_dict(self, sample: Sample) -> dict:
        b = sample.signal
        if sample.mean_signal:                   # placeholder series; the label quotes the cohort means
            if self.stat_mean is None:
                raise RuntimeError('mean record outside inject_mean()')
            b = dataclasses.replace(b, mean=self.stat_mean, std=self.stat_std, info=dict(n_nan=0))
        return self.prompts.sample(b, sample.report, self.kind, question=prompts.QUESTIONS[sample.para],
                                   flip=sample.order == 'flip')

    @staticmethod
    def _decoder(model):
        return model.llm.get_decoder()

    def forward(self, sample: Sample) -> dict:
        """AnswerExtractor.run for one SP record: P(Yes) at "Answer:", the state there, the answer-span mean."""
        torch = self.torch
        s = self.sample_dict(sample)
        steps = []
        grab = lambda m, i, o: steps.append((o.last_hidden_state if hasattr(o, 'last_hidden_state') else o[0]).detach())
        proj = []
        kw = dict(max_new_tokens=self.answer_tokens, do_sample=False, num_beams=1, return_dict_in_generate=True,
                  output_logits=True, pad_token_id=self.tok.pad_token_id, eos_token_id=self.tok.eos_token_id)
        with torch.no_grad():
            hp = self.model.projector.register_forward_hook(lambda m, i, o: proj.append(o.detach().float()))
            try:
                e, a = self.model.pad_and_apply_batch([s])
            finally:
                hp.remove()
            embeds = e[0][a[0].bool()][None]
            attn = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)
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
        cpu = lambda t: t[0].float().cpu().numpy()
        out = dict(p_yes=float(p_yes[0]), ans_last=cpu(ans_last), ans_emb=cpu(ans_emb),
                   answer=self.tok.batch_decode(gen_ids, skip_special_tokens=True)[0].strip())
        if self.store_prefix:                    # (K, P, H) for the record's leads; NaN when there is no ECG
            out['prefix_mean'] = (proj[0].reshape(-1, proj[0].shape[-1]).mean(0).cpu().numpy() if proj
                                  else np.full(self.hidden, np.nan, np.float32))
        return out

    # ── mean swap (MeanEncoder, SP) ──────────────────────────────────────────
    def signal_mean(self, bundles, batch=8):
        """Mean projector output (K, P, H) over the cohort reference ECGs, plus the mean per-lead statistics the
        label text quotes for a mean record."""
        torch = self.torch
        total, n, stats, cap = None, 0, [], {}
        hook = lambda m, i, o: cap.__setitem__('o', o.detach().float())
        with torch.no_grad():
            for b in range(0, len(bundles), batch):
                bs = [x for x in bundles[b:b + batch] if x is not None]
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
    def inject_mean(self, mean, stat_mean=None, stat_std=None, **_):
        if stat_mean is None or stat_std is None:
            raise ValueError('opentslm: the cohort mean needs stat_mean / stat_std (the label text)')
        mean_t = self.torch.from_numpy(np.asarray(mean))

        def swap(m, i, o):                       # o: (B*K, P, H), lead-major per sample
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
        from ..analysis.probe import LookupReadout, path_key
        from ..store import ChunkStore
        src = f'{run_dir}/sources.json'
        dirs = fsio.read_json(src).get('lookups', {}) if fsio.exists(src) else {}
        res = manifest.resolver
        need = {path_key(d, cfg.raw): d for c in manifest.cohorts.values()
                for col in ('e_first', 'e_last') for d in c.table[col].dropna()}
        ref = {path_key(d, cfg.raw) for c in manifest.cohorts.values() for d in c.table.e_first}
        out = []
        for name, family, key, source in cls.lookups:
            if source not in dirs and cls.store_prefix and key == 'prefix_mean':
                out += cls._own_prefix(manifest, store, name, family)
                continue
            if source not in dirs:
                print(f'  readout {name}: no `{source}` lookup run registered (mmd index import --lookup), skipped')
                continue
            lk = ChunkStore(dirs[source])
            have = [r for r in lk.done() if path_key(r, cfg.raw) in need]
            run = lk.load(keys=[key], records=set(have))
            if run is None:
                continue
            vec = {need[path_key(r, cfg.raw)]: v for r, v in zip(run['record'], run[key].mean(1))}
            # sorted and in float64, so the mean slot is the same in every session (the notebook averaged over a
            # Python set, whose order moves the last float32 digits)
            mean_vec = np.mean([vec[d].astype(np.float64) for d in sorted(vec) if path_key(d, cfg.raw) in ref],
                               0).astype(np.float32)
            out.append(LookupReadout(name, family, res, 'e', vec, mean_vec=mean_vec))
        return out

    @staticmethod
    def _own_prefix(manifest, store, name, family):
        """The ECG prefix mean from this model's own records (any record whose ECG slot holds a real ECG)."""
        from ..analysis.probe import LookupReadout, vectors_by_slot
        from ..analysis.report import standard_configs
        run = store.load(keys=['prefix_mean'], records=set(standard_configs(manifest).record))
        if run is None or 'prefix_mean' not in run:
            print(f'  readout {name}: prefix_mean not stored, skipped')
            return []
        res = manifest.resolver
        vec = vectors_by_slot(run, res, 'e', ['prefix_mean'])['prefix_mean']
        ref = sorted({d for c in manifest.cohorts.values() for d in c.table.e_first if d in vec})
        mean_vec = np.mean([vec[d].astype(np.float64) for d in ref], 0).astype(np.float32)
        return [LookupReadout(name, family, res, 'e', vec, mean_vec=mean_vec)]


@register
class OpenTSLMSPGemma(OpenTSLMSP):
    """The same SP pipeline on OpenTSLM's Gemma-3-1B ECG checkpoint (base google/gemma-3-1b-pt: accept its terms
    on Hugging Face first). No legacy run: every record is generated, and the ECG prefix mean comes from its own
    records. No Flamingo lookup (OpenTSLM's Gemma Flamingo checkpoint has no single-ECG run here)."""
    name = 'opentslm-sp-gemma'
    hf_id = 'OpenTSLM/gemma-3-1b-pt-ecg-sp'
    meta = dict(fusion='prefix (soft prompt)', domain='time series', params='1B', mimic_exposure='none',
                base='gemma-3-1b-pt')
    keymap = {}
    readouts = [('answer P(Yes) (SP)', 'post-fusion', 'p_yes', None),
                ('answer embedding (SP)', 'post-fusion', 'ans_emb', 'pca'),
                ('state at Answer: (SP)', 'post-fusion', 'post_last', 'pca')]
    lookups = [('ECG prefix mean', 'pre-fusion', 'prefix_mean', 'sp-prefix')]
    note_blind = ('ECG prefix mean',)
    store_prefix = True

    def forward(self, sample: Sample) -> dict:
        out = super().forward(sample)
        out['post_last'] = out.pop('ans_last')   # common name from the start (no legacy records to match)
        return out
