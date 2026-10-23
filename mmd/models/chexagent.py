"""CheXagent-2-3b (Stanford AIMI): a chest-X-ray instruction model with its own remote code.

Runs in the `chexagent` uv group (its model card pins an older transformers; check on first build). The
remote code reads images from FILE PATHS inside the prompt (`tokenizer.from_list_format`), so gs:// images are
copied to a local cache first. Trained with MIMIC-CXR among its data, so it is flagged as MIMIC-exposed.

What is not known until the smoke test runs on the VM, and is printed by describe():
  - the image-token span: taken between the tokenizer's image start / end ids when it has them (Qwen-VL style);
    otherwise post_signal is NaN and the zoo table marks it missing;
  - the vision module: the first module whose name ends in 'visual' or 'vision_model'.
The mean swap is not offered (no verified projector hook point): its mean_* configurations are skipped and the
zoo table marks them missing rather than imputing them.
"""
import hashlib
import os
import tempfile

import numpy as np

from .. import fsio, prompts
from .base import Adapter, Sample, register
from .hf_vlm import _hooks, find_module, readouts_from_generate, yes_no_ids


@register
class CheXagent(Adapter):
    name = 'chexagent'
    hf_id = 'StanfordAIMI/CheXagent-2-3b'
    revision = 'main'
    group = 'chexagent'
    signal, pair = 'cxr', 'cxr_notes'
    meta = dict(fusion='prefix (resampler)', domain='radiology', params='3B', mimic_exposure='MIMIC-CXR')
    supports_mean = False
    answer_tokens = 8
    system = 'You are a helpful assistant.'
    cache_dir = os.path.join(tempfile.gettempdir(), 'mmd_images')

    def load(self, device='cuda'):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(self.hf_id, revision=self.revision, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.hf_id, revision=self.revision, device_map=device,
                                                          trust_remote_code=True).to(torch.bfloat16).eval()
        self.dev = next(self.model.parameters()).device
        self.yes_ids, self.no_ids = yes_no_ids(self.tok)
        self.img_ids = (getattr(self.tok, 'img_start_id', None), getattr(self.tok, 'img_end_id', None))
        self.vt_name, self.vt = find_module(self.model, 'visual', 'vision_model')
        return self

    def describe(self):
        d = lambda ids: [self.tok.decode([i]) for i in ids]
        return dict(yes_ids=self.yes_ids, yes=d(self.yes_ids), no_ids=self.no_ids, no=d(self.no_ids),
                    image_start_end_ids=self.img_ids, vision_module=self.vt_name, mean_swap='not offered')

    def local_image(self, path):
        if not fsio.is_remote(path):
            return path
        os.makedirs(self.cache_dir, exist_ok=True)
        dst = os.path.join(self.cache_dir, hashlib.md5(path.encode()).hexdigest() + os.path.splitext(path)[1])
        if not os.path.exists(dst):
            with fsio.open_(path) as f, open(dst + '.part', 'wb') as g:
                g.write(f.read())
            os.replace(dst + '.part', dst)
        return dst

    def input_ids(self, sample: Sample):
        body = prompts.text(sample.signal_path is not None, sample.report, 'cxr', sample.order, sample.para)
        items = []
        for i, part in enumerate(body.split(prompts.SIGNAL_MARK)):
            if i:
                items.append({'image': self.local_image(sample.signal_path)})
            if part.strip():
                items.append({'text': part})
        query = self.tok.from_list_format(items)
        conv = [{'from': 'system', 'value': self.system}, {'from': 'human', 'value': query}]
        ids = self.tok.apply_chat_template(conv, add_generation_prompt=True, return_tensors='pt')
        cue = self.tok(prompts.ANSWER_CUE.strip(), add_special_tokens=False, return_tensors='pt').input_ids
        return self.torch.cat([ids, cue], 1).to(self.dev)

    def _span(self, ids):
        s, e = self.img_ids
        if s is None or e is None:
            return None
        a, b = (ids[0] == s).nonzero().flatten(), (ids[0] == e).nonzero().flatten()
        return (int(a[0]) + 1, int(b[0])) if len(a) and len(b) else None

    def forward(self, sample: Sample) -> dict:
        torch = self.torch
        if sample.mean_signal:
            raise ValueError('chexagent: mean swap not offered')
        ids = self.input_ids(sample)
        cap = {}
        hooks = [(self.vt, lambda m, i, o: cap.__setitem__('vision', (o[0] if isinstance(o, tuple) else o).detach()))] \
            if self.vt is not None else []
        with torch.no_grad(), _hooks(hooks):
            gen = self.model.generate(ids, max_new_tokens=self.answer_tokens, do_sample=False, num_beams=1,
                                      use_cache=True, output_hidden_states=True, output_logits=True,
                                      return_dict_in_generate=True)
        r = readouts_from_generate(torch, gen, ids.shape[1], self._span(ids), self.yes_ids, self.no_ids, self.tok)
        v = cap.get('vision')
        r['pre_vision_mean'] = (v[0].float().reshape(-1, v.shape[-1]).mean(0).cpu().numpy()
                                if (v is not None and sample.signal_path is not None) else np.full(1, np.nan, np.float32))
        return r
