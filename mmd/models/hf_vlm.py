"""Generic adapter for Hugging Face image-text models (AutoModelForImageTextToText + a chat template).

Covers models whose processor takes {"type": "image"} chat content and whose image features come out of a
`multi_modal_projector` (Gemma 3 / MedGemma, LLaVA-HF, ...). A subclass sets hf_id, meta and dtype; it can
override `messages()` for a different template.

One greedy generate() call gives everything:
    prefill final-layer states (hidden_states[0])  -> prompt_last_state / prompt_mean_state / signal_mean_state
    first-step logits                               -> yes_probability
    states of the generated tokens fed back in       -> answer_mean_state, answer_text
Hooks on the vision tower and the projector give encoder_mean / projector_mean. The mean swap replaces the
projector output with the cohort mean (same hook point), as MeanProjector does for LLaVA-Med.
"""
import contextlib

import numpy as np

from .. import prompts
from .base import Adapter, ReadoutSpec, Sample, to_np


def find_module(model, *suffixes):
    for name, m in model.named_modules():
        if any(name.endswith(s) for s in suffixes):
            return name, m
    return None, None


def yes_no_ids(tok):
    first = lambda s: tok.encode(s, add_special_tokens=False)[0]
    yes = {first(s) for s in (' Yes', ' yes', 'Yes', 'yes')}
    no = {first(s) for s in (' No', ' no', 'No', 'no')}
    return sorted(yes - no), sorted(no - yes)


def readouts_from_generate(torch, gen, L, span, yes_ids, no_ids, tok):
    """Common readouts from generate(..., return_dict_in_generate, output_hidden_states, output_logits)."""
    hs = gen.hidden_states[0][-1][0].float()                       # (L, H) prefill, final layer
    logits = (gen.logits if getattr(gen, 'logits', None) is not None else gen.scores)[0][0].float()
    p_yes = torch.sigmoid(torch.logsumexp(logits[yes_ids], 0) - torch.logsumexp(logits[no_ids], 0))
    new = gen.sequences[0, L:].tolist()
    eos = {tok.eos_token_id} if tok.eos_token_id is not None else set()
    n_ans = next((i for i, t in enumerate(new) if t in eos), len(new))
    # hidden_states[t] (t >= 1) is the step that consumed generated token t-1
    states = [gen.hidden_states[t][-1][0, -1].float() for t in range(1, min(n_ans, len(gen.hidden_states) - 1) + 1)]
    nan = torch.full((hs.shape[1],), float('nan'))
    return dict(yes_probability=float(p_yes), prompt_last_state=to_np(hs[-1]), prompt_mean_state=to_np(hs.mean(0)),
                signal_mean_state=to_np(hs[span[0]:span[1]].mean(0)) if span else to_np(nan),
                answer_mean_state=to_np(torch.stack(states).mean(0) if states else hs[-1]),
                answer_text=tok.decode(new[:n_ans], skip_special_tokens=True).strip(),
                signal_span=(span[1] - span[0]) if span else 0)


class HFImageTextAdapter(Adapter):
    group = 'hf-modern'
    torch_dtype = 'bfloat16'
    answer_tokens = 8
    answer_cue = 'Answer:'
    system = None                       # optional system turn
    supports_mean = True
    readouts = Adapter.readouts + [ReadoutSpec('signal projector mean', 'projector', 'projector_mean')]
    note_blind = ('signal mean state', 'signal projector mean')

    def load(self, device='cuda'):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(self.hf_id, revision=self.revision)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.hf_id, revision=self.revision, torch_dtype=getattr(torch, self.torch_dtype), device_map=device).eval()
        self.tok = self.proc.tokenizer
        self.dev = next(self.model.parameters()).device
        self.yes_ids, self.no_ids = yes_no_ids(self.tok)
        cfg = self.model.config
        self.image_token_id = getattr(cfg, 'image_token_index', None) or getattr(cfg, 'image_token_id', None)
        self.vt_name, self.vt = find_module(self.model, 'vision_tower')
        self.pj_name, self.pj = find_module(self.model, 'multi_modal_projector')
        return self

    def describe(self):
        d = lambda ids: [self.tok.decode([i]) for i in ids]
        return dict(yes_ids=self.yes_ids, yes=d(self.yes_ids), no_ids=self.no_ids, no=d(self.no_ids),
                    image_token_id=self.image_token_id, vision_tower=self.vt_name, projector=self.pj_name,
                    dtype=self.torch_dtype)

    def messages(self, sample: Sample):
        body = prompts.text(sample.signal is not None, sample.report, self.signal, sample.order, sample.para)
        content = []
        for i, part in enumerate(body.split(prompts.SIGNAL_MARK)):
            if i:
                content.append({'type': 'image'})
            if part.strip():                      # keep the newlines around the image
                content.append({'type': 'text', 'text': part})
        msgs = [{'role': 'user', 'content': content}]
        if self.system:
            msgs.insert(0, {'role': 'system', 'content': [{'type': 'text', 'text': self.system}]})
        return msgs

    def prompt(self, sample: Sample) -> str:
        return self.proc.apply_chat_template(self.messages(sample), tokenize=False,
                                             add_generation_prompt=True) + self.answer_cue

    def inputs(self, sample: Sample):
        kw = dict(text=[self.prompt(sample)], return_tensors='pt', add_special_tokens=False)
        if sample.signal is not None:
            kw['images'] = [[sample.signal]]
        x = self.proc(**kw).to(self.dev)
        if 'pixel_values' in x:
            x['pixel_values'] = x['pixel_values'].to(self.model.dtype)
        return x

    def _span(self, ids):
        if self.image_token_id is None:
            return None
        pos = (ids[0] == self.image_token_id).nonzero().flatten()
        return (int(pos[0]), int(pos[-1]) + 1) if len(pos) else None

    def forward(self, sample: Sample) -> dict:
        torch = self.torch
        cap = {}
        grab = lambda key: (lambda m, i, o: cap.__setitem__(key, (o[0] if isinstance(o, tuple) else
                                                                   getattr(o, 'last_hidden_state', o)).detach()))
        hooks = [(m, grab(k)) for k, m in (('vision', self.vt), ('proj', self.pj)) if m is not None]
        x = self.inputs(sample)
        L = x['input_ids'].shape[1]
        with torch.no_grad(), _hooks(hooks):
            gen = self.model.generate(**x, max_new_tokens=self.answer_tokens, do_sample=False, use_cache=True,
                                      output_hidden_states=True, output_logits=True, return_dict_in_generate=True)
        r = readouts_from_generate(torch, gen, L, self._span(x['input_ids']), self.yes_ids, self.no_ids, self.tok)
        if sample.signal is not None and 'proj' in cap:
            r['projector_mean'] = to_np(cap['proj'][0].float().reshape(-1, cap['proj'].shape[-1]).mean(0))
            r['encoder_mean'] = to_np(cap['vision'][0].float().reshape(-1, cap['vision'].shape[-1]).mean(0)) \
                if 'vision' in cap else np.full(1, np.nan, np.float32)
        else:
            h = self.model.config.get_text_config().hidden_size if hasattr(self.model.config, 'get_text_config') \
                else r['prompt_mean_state'].shape[0]
            r['projector_mean'] = np.full(h, np.nan, np.float32)
            vh = getattr(getattr(self.model.config, 'vision_config', None), 'hidden_size', 1)
            r['encoder_mean'] = np.full(vh, np.nan, np.float32)
        return r

    # ── mean swap: the projector output replaced by the cohort mean ──────────
    def signal_mean(self, images, mod=None, batch=8):
        torch = self.torch
        total, n = None, 0
        cap = {}
        hook = lambda m, i, o: cap.__setitem__('proj', o.detach())
        for b in range(0, len(images), batch):
            ims = [im for im in images[b:b + batch] if im is not None]
            if not ims:
                continue
            x = self.proc.image_processor(ims, return_tensors='pt')['pixel_values'].to(self.dev, self.model.dtype)
            with torch.no_grad(), _hooks([(self.pj, hook)]):
                self.model.get_image_features(pixel_values=x)
            f = cap['proj'].float()                              # (B, n_tokens, H)
            total = f.sum(0) if total is None else total + f.sum(0)
            n += f.shape[0]
        return (total / n).cpu().numpy().astype(np.float32)

    @contextlib.contextmanager
    def inject_mean(self, mean, mod=None, **_):
        mean_t = self.torch.from_numpy(np.asarray(mean))

        def swap(m, i, o):
            t = mean_t.to(o.device, o.dtype)
            assert o.shape[1:] == t.shape, (o.shape, t.shape)
            return t.unsqueeze(0).expand_as(o).contiguous()
        with _hooks([(self.pj, swap)]):
            yield


@contextlib.contextmanager
def _hooks(pairs):
    handles = [m.register_forward_hook(f) for m, f in pairs]
    try:
        yield
    finally:
        for h in handles:
            h.remove()
