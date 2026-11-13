"""ChatTS-8B (bytedance-research/ChatTS-8B, VLDB'25): a time-series MLLM on Qwen3-8B, aligned on synthetic series
only, so it has seen no MIMIC data. Fusion: each series is cut into patches of 8, a small MLP encoder maps each patch
to the LLM width, and the patch embeddings are spliced into the token sequence at the series' <ts><ts/> placeholder
(prefix-style, like an image-text model's image tokens). Its processor also writes each series' offset, scale,
length, min, max and first / last values as text just before the placeholder, so absolute values reach the LLM as
text while the shape goes through the encoder.

What it is given (the same records as the OpenTSLM models; prompt from series_prompts.text, not prompts.py):
    e  ECG      the 12 leads as 12 series, 100 Hz x 10 s (data.prepare, gaps filled), in mV: 125 patches each
    v  vitals   one series per charted channel, hourly (gaps filled, never-charted channels left out)
    l  labs     text (Zifan's clinical_text lines)
    n  note     text
Short series are what the 8B release improved over ChatTS-14B (model card), so vitals are in its range; the ECG at
1000 points is within its training lengths.

Readouts: the common ones from generate() (hf_vlm.readouts_from_generate); `signal_mean_state` stays NaN, as the
series positions move inside the model's own merge step. Pre-fusion: `projector_mean_e` / `projector_mean_v`, the
encoder output averaged over that modality's patches (NaN when absent). One record per forward (its merge step
re-pads batches itself; not checked against single records), no mean swap.

Runs in the hf-modern group with trust_remote_code (the remote code was written for transformers 4.52.4; if a newer
transformers breaks it, give it its own uv group pinned there). The chat template is Qwen3's, with thinking off.
"""
import dataclasses

import numpy as np

from .. import series_prompts as SP
from .base import Adapter, ReadoutSpec, Sample, register
from .clinical_text import VITAL_NAMES, nice
from .hf_vlm import _hooks, readouts_from_generate, yes_no_ids

TS_TOKEN = '<ts><ts/>'
UNITS = {'degF': '°F'}


@register
class ChatTS8B(Adapter):
    name = 'chatts-8b'
    hf_id = 'bytedance-research/ChatTS-8B'
    revision = '7216a84034063c2d394901fe45d20a1e8d773a1f'
    group = 'hf-modern'
    batches = False
    signal, modalities, default_modset = 'ecg', ('e', 'v', 'l', 'n'), 'ecg_notes'
    meta = dict(fusion='prefix (time-series patch encoder)', domain='time series', params='8B',
                mimic_exposure='none (synthetic training series)', base='qwen3-8b', presentation='series')
    readouts = [r for r in Adapter.readouts if r.key != 'signal_mean_state'] + [
        ReadoutSpec('ECG series encoder mean', 'projector', 'projector_mean_e'),
        ReadoutSpec('vitals series encoder mean', 'projector', 'projector_mean_v')]
    note_blind = ('ECG series encoder mean', 'vitals series encoder mean')
    torch_dtype = 'bfloat16'
    answer_tokens = 8
    answer_cue = 'Answer:'
    series_spec = {'e': dict(hz=100, length=1000, fill=True), 'v': dict(fill=True)}

    def load(self, device='cuda'):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        self.torch = torch
        kw = dict(revision=self.revision, trust_remote_code=True)
        self.tok = AutoTokenizer.from_pretrained(self.hf_id, **kw)
        self.proc = AutoProcessor.from_pretrained(self.hf_id, tokenizer=self.tok, **kw)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, torch_dtype=getattr(torch, self.torch_dtype), device_map=device, **kw).eval()
        gc = self.model.generation_config
        eos = gc.eos_token_id if isinstance(gc.eos_token_id, (list, tuple)) else [gc.eos_token_id]
        self.stop_ids = {t for t in eos if t is not None}
        self.dev = next(self.model.parameters()).device
        self.yes_ids, self.no_ids = yes_no_ids(self.tok)
        self.hidden = self.model.config.hidden_size
        return self

    def describe(self):
        d = lambda ids: [self.tok.decode([i]) for i in ids]
        return dict(yes_ids=self.yes_ids, yes=d(self.yes_ids), no_ids=self.no_ids, no=d(self.no_ids),
                    dtype=self.torch_dtype, revision=self.revision[:7])

    # ── input ────────────────────────────────────────────────────────────────
    def series_list(self, sample: Sample):
        """[(modality, label, 1-D values)] in prompt order: the ECG leads, then the charted vitals channels."""
        out = []
        e = sample.signals.get('e')
        if e is not None:
            b = self.series(e, 'e')
            out += [('e', f'Lead {n} (mV, 100 Hz, 10 s):', np.asarray(x, float)) for n, x in zip(b.names, b.values)]
        v = sample.signals.get('v')
        if v is not None:
            keep = np.isfinite(np.asarray(v.values, float)).any(1)
            if keep.any():
                units = v.units or [''] * len(v.names)
                v = dataclasses.replace(v, values=np.asarray(v.values)[keep],
                                        names=[n for n, k in zip(v.names, keep) if k],
                                        units=[u for u, k in zip(units, keep) if k])
                b = self.series(v, 'v')
                L = b.values.shape[1]
                out += [('v', f'{nice(n, VITAL_NAMES)} ({UNITS.get(u, u)}, hourly, {L} h):', np.asarray(x, float))
                        for n, u, x in zip(b.names, b.units, b.values)]
        return out

    def user_text(self, sample: Sample) -> str:
        ser = self.series_list(sample)
        labels = {m: [lab for mm, lab, _ in ser if mm == m] for m in ('e', 'v')}
        blocks = {'l': SP.labs_body(sample.signals['l']) if 'l' in sample.signals else None, 'n': sample.report}
        return SP.text(blocks=blocks, series=labels, order=sample.order, para=sample.para).replace(SP.TS_MARK,
                                                                                                    TS_TOKEN)

    def prompt(self, sample: Sample) -> str:
        msgs = [{'role': 'user', 'content': self.user_text(sample)}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False) + self.answer_cue

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, sample: Sample) -> dict:
        torch = self.torch
        ser = self.series_list(sample)
        kw = dict(text=[self.prompt(sample)], padding=True, return_tensors='pt')
        if ser:
            kw['timeseries'] = [x for _, _, x in ser]
        x = self.proc(**kw).to(self.dev)
        if 'timeseries' in x:
            x['timeseries'] = x['timeseries'].to(self.model.dtype)
        cap = {}
        hook = lambda m, i, o: cap.__setitem__('ts', (o[0].detach().float(), o[1].detach()))
        with torch.no_grad(), _hooks([(self.model.ts_encoder, hook)]):
            gen = self.model.generate(**x, max_new_tokens=self.answer_tokens, do_sample=False, use_cache=True,
                                      output_hidden_states=True, output_logits=True, return_dict_in_generate=True)
        r = readouts_from_generate(torch, gen, x['input_ids'].shape[1], None, self.yes_ids, self.no_ids, self.tok,
                                   stop=self.stop_ids)
        for m in ('e', 'v'):
            r[f'projector_mean_{m}'] = np.full(self.hidden, np.nan, np.float32)
        if 'ts' in cap:
            feats, cnt = cap['ts']
            parts = torch.split(feats, [int(c) for c in cnt.tolist()])
            for m in ('e', 'v'):
                mine = [p for (mm, _, _), p in zip(ser, parts) if mm == m]
                if mine:
                    r[f'projector_mean_{m}'] = torch.cat(mine).mean(0).cpu().numpy().astype(np.float32)
        return r

    def forward_batch(self, samples) -> list:
        return [self.forward(s) for s in samples]
