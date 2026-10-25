"""LLaVA-Med v1.5 (Mistral-7B): port of the notebook's LlavaHub / LlavaExtractor / MeanProjector, unchanged.

Runs in the `llava-med` uv group, after `mmd setup llava-med` (the LLaVA-Med repo at a pinned commit, no deps).
It writes the common names; `keymap` renames the keys the notebook's chunks use (post_image, vision_*, proj_*)
on load, so imported notebook records and new records sit in one store and one analysis.

One record at a time (no padding questions at all), one prefill, then a short greedy continuation:
    encoder_mean / encoder_grid       CLIP patch features (576, Cv): mean, and pooled to grid x grid
    projector_mean / projector_grid   mm_projector output (576, H): the soft tokens entering the LLM
    prompt_last_state / signal_mean_state / prompt_mean_state   final-layer prompt states: last / image span / all
    yes_probability, answer_mean_state, answer_text, signal_span
Hooks run in registration order and each sees the previous hook's output, so when inject_mean() is active
(registered first) the projector readout recorded here IS the injected mean.
"""
import contextlib
import gc
import inspect

import numpy as np

from .. import prompts
from .base import Adapter, Sample, register

LLAVA_MED_COMMIT = 'main'          # pin to the commit the existing runs used: `mmd setup llava-med --commit <sha>`
GRID = 4


@register
class LlavaMed(Adapter):
    name = 'llava-med'
    hf_id = 'microsoft/llava-med-v1.5-mistral-7b'
    group = 'llava-med'
    signal, modalities, default_modset = 'cxr', ('x', 'n'), 'cxr_notes'
    meta = dict(fusion='prefix (projector)', domain='medical', params='7B', mimic_exposure='none')
    keymap = {'post_image': 'signal_mean_state', 'vision_mean': 'encoder_mean', 'vision_grid': 'encoder_grid',
              'proj_mean': 'projector_mean', 'proj_grid': 'projector_grid', 'image_span': 'signal_span'}
    note_blind = ('signal mean state', 'CXR encoder mean', 'CXR projector mean')
    supports_mean = True
    conv_mode = 'mistral_instruct'
    load_4bit = False
    answer_tokens = 8

    def load(self, device='cuda'):
        import torch
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
        from llava.model.builder import load_pretrained_model
        self.torch = torch
        self.L = dict(IMAGE_TOKEN_INDEX=IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN,
                      conv_templates=conv_templates, process_images=process_images,
                      tokenizer_image_token=tokenizer_image_token)
        self.tok, self.model, self.proc, _ = load_pretrained_model(
            self.hf_id, None, get_model_name_from_path(self.hf_id), load_4bit=self.load_4bit)
        self.model.eval()
        m = self.model
        self.dev = next(m.parameters()).device
        self.dtype = next(m.get_model().mm_projector.parameters()).dtype
        first = lambda s: self.tok.encode(s, add_special_tokens=False)[0]
        yes = {first(s) for s in (' Yes', ' yes', 'Yes', 'yes')}
        no = {first(s) for s in (' No', ' no', 'No', 'no')}
        self.yes_ids, self.no_ids = sorted(yes - no), sorted(no - yes)
        self.cv = int(getattr(m.config, 'mm_hidden_size', 1024))
        self.h = int(m.config.hidden_size)
        self.lm = self._lm_forward(m)
        self._prep = inspect.signature(m.prepare_inputs_labels_for_multimodal).parameters
        return self

    def describe(self):
        d = lambda ids: [self.tok.decode([i]) for i in ids]
        return dict(yes_ids=self.yes_ids, yes=d(self.yes_ids), no_ids=self.no_ids, no=d(self.no_ids),
                    conv_mode=self.conv_mode, hidden=self.h, vision_hidden=self.cv)

    # ── helpers (LlavaHub) ───────────────────────────────────────────────────
    @staticmethod
    def _lm_forward(model):
        """The language model's own forward, below LLaVA's wrapper, so inputs_embeds go straight in."""
        for c in type(model).__mro__[1:]:
            if c.__name__.endswith('ForCausalLM') and not c.__name__.lower().startswith('llava'):
                return lambda **kw: c.forward(model, **kw)
        raise RuntimeError(f'no causal-LM base class under {type(model).__name__}')

    @staticmethod
    @contextlib.contextmanager
    def _hooks(pairs):
        handles = [m.register_forward_hook(f) for m, f in pairs]
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def free(self):
        self.model = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def prompt(self, sample: Sample) -> str:
        has_image = sample.signal is not None
        body = prompts.text(has_image, sample.report, 'cxr', sample.order, sample.para)
        body = body.replace(prompts.SIGNAL_MARK, self.L['DEFAULT_IMAGE_TOKEN'])
        conv = self.L['conv_templates'][self.conv_mode].copy()
        conv.append_message(conv.roles[0], body)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt() + prompts.ANSWER_CUE

    def _grid(self, x):                                   # (576, C) -> (grid*grid, C)
        side = int(round(x.shape[0] ** 0.5))
        x = x.reshape(side, side, -1).permute(2, 0, 1)[None]
        return self.torch.nn.functional.adaptive_avg_pool2d(x, GRID)[0].permute(1, 2, 0).reshape(GRID * GRID, -1)

    def _embed(self, ids, image):
        m = self.model
        if image is None:
            return m.get_model().embed_tokens(ids), None
        images = self.L['process_images']([image], self.proc, m.config).to(self.dev, self.dtype)
        kw = dict(input_ids=ids, position_ids=None, attention_mask=self.torch.ones_like(ids),
                  past_key_values=None, labels=None, images=images, image_sizes=[image.size])
        res = m.prepare_inputs_labels_for_multimodal(**{k: v for k, v in kw.items() if k in self._prep})
        embeds = next(r for r in res if self.torch.is_tensor(r) and r.dim() == 3)
        start = int((ids[0] == self.L['IMAGE_TOKEN_INDEX']).nonzero()[0, 0])
        n_img = embeds.shape[1] - (ids.shape[1] - 1)
        return embeds, (start, start + n_img)

    def forward(self, sample: Sample) -> dict:
        torch = self.torch
        with torch.no_grad():
            image = sample.signal
            ids = self.L['tokenizer_image_token'](self.prompt(sample), self.tok, self.L['IMAGE_TOKEN_INDEX'],
                                                  return_tensors='pt')[None].to(self.dev)
            cap = {}
            pairs = [(self.model.get_vision_tower(), lambda m, i, o: cap.__setitem__('vision', o.detach())),
                     (self.model.get_model().mm_projector, lambda m, i, o: cap.__setitem__('proj', o.detach()))]
            with self._hooks(pairs):
                embeds, span = self._embed(ids, image)
            L = embeds.shape[1]
            out = self.lm(inputs_embeds=embeds, attention_mask=torch.ones(1, L, dtype=torch.long, device=self.dev),
                          use_cache=True, output_hidden_states=True, return_dict=True)
            hs = out.hidden_states[-1][0].float()                        # (L, H)
            logits = out.logits[0, -1].float()
            p_yes = torch.sigmoid(torch.logsumexp(logits[self.yes_ids], 0) - torch.logsumexp(logits[self.no_ids], 0))
            past, nxt, gen, states = out.past_key_values, int(logits.argmax()), [], []
            for step in range(self.answer_tokens):
                if nxt == self.tok.eos_token_id:
                    break
                gen.append(nxt)
                e = self.model.get_model().embed_tokens(torch.tensor([[nxt]], device=self.dev))
                o = self.lm(inputs_embeds=e, past_key_values=past, use_cache=True, output_hidden_states=True,
                            return_dict=True,
                            attention_mask=torch.ones(1, L + step + 1, dtype=torch.long, device=self.dev))
                states.append(o.hidden_states[-1][0, -1].float())
                past, nxt = o.past_key_values, int(o.logits[0, -1].argmax())
            nan = lambda *s: torch.full(s, float('nan'))
            g2 = GRID ** 2
            r = dict(yes_probability=float(p_yes), prompt_last_state=hs[-1].cpu(), prompt_mean_state=hs.mean(0).cpu(),
                     signal_mean_state=hs[span[0]:span[1]].mean(0).cpu() if span else nan(self.h),
                     answer_mean_state=(torch.stack(states).mean(0) if states else hs[-1]).cpu(),
                     answer_text=self.tok.decode(gen, skip_special_tokens=True).strip(),
                     signal_span=(span[1] - span[0]) if span else 0)
            if image is not None:
                v, p = cap['vision'][0].float(), cap['proj'][0].float()
                r.update(encoder_mean=v.mean(0).cpu(), encoder_grid=self._grid(v).cpu(),
                         projector_mean=p.mean(0).cpu(), projector_grid=self._grid(p).cpu())
            else:
                r.update(encoder_mean=nan(self.cv), encoder_grid=nan(g2, self.cv),
                         projector_mean=nan(self.h), projector_grid=nan(g2, self.h))
        return {k: (v.numpy() if hasattr(v, 'numpy') else v) for k, v in r.items()}

    # ── pre-fusion lookups (the notebook's PRE readouts) ─────────────────────
    @classmethod
    def extra_readouts(cls, cfg, manifest, store, run_dir):
        """CXR encoder / projector means looked up by the image in the slot, scored on every donor draw. The mean
        slot gets the reference-image average (encoder) and the injected cohort mean (projector)."""
        from .. import fsio
        from ..analysis.probe import Embeddings, vectors_by_slot
        from ..runner import mean_path
        run = store.load(keys=['encoder_mean', 'projector_mean'], records=set(manifest.analysis_configs().record))
        if run is None or 'encoder_mean' not in run:
            return []
        v = vectors_by_slot(run, manifest, 'x', ['encoder_mean', 'projector_mean'])
        enc, prj = v['encoder_mean'], v['projector_mean']
        ref = [d for c in manifest.cohorts.values() for d in c.table.x_first if d in enc]
        path = mean_path(run_dir, 'x', manifest.name)
        if fsio.exists(path):
            proj_mean = fsio.load_npz(path)['mean'].mean(0)
        else:
            print(f'  {path} not found: projector mean slot uses the reference-image average')
            proj_mean = np.mean([prj[d] for d in ref], 0)
        return [Embeddings.from_lookup('CXR encoder mean', 'pre-fusion', manifest, 'x', enc,
                                       mean_vec=np.mean([enc[d] for d in ref], 0)),
                Embeddings.from_lookup('CXR projector mean', 'projector', manifest, 'x', prj, mean_vec=proj_mean)]

    # ── mean swap (MeanProjector) ────────────────────────────────────────────
    def signal_mean(self, images, mod=None, batch=16):
        """Mean projector output (576 x H) over the cohort reference images (encode_images only)."""
        torch = self.torch
        total, n = None, 0
        with torch.no_grad():
            for b in range(0, len(images), batch):
                ims = [im for im in images[b:b + batch] if im is not None]
                if not ims:
                    continue
                x = self.model.encode_images(
                    self.L['process_images'](ims, self.proc, self.model.config).to(self.dev, self.dtype)).float()
                total = x.sum(0) if total is None else total + x.sum(0)
                n += x.shape[0]
        return (total / n).cpu().numpy().astype(np.float32)

    @contextlib.contextmanager
    def inject_mean(self, mean, mod=None, **_):
        mean_t = self.torch.from_numpy(np.asarray(mean))

        def swap(m, i, o):
            t = mean_t.to(o.device, o.dtype)
            assert o.shape[1:] == t.shape, (o.shape, t.shape)
            return t.unsqueeze(0).expand_as(o).contiguous()
        with self._hooks([(self.model.get_model().mm_projector, swap)]):
            yield
