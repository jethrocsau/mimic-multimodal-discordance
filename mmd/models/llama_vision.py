"""Llama-3.2-11B-Vision-Instruct: a general VLM with no MIMIC-CXR training. Cross-attention fusion: the image never
enters the token sequence. A ViT encodes the film in up to 4 tiles of 560x560 (1601 patches each, 5 layers'
states concatenated), `multi_modal_projector` maps them to the LLM width, and gated cross-attention layers
inserted into the Llama 3.1 decoder read them. The prompt holds a single <|image|> token, which marks where the
text starts to attend to the image.

This is the zoo's only cross-attention model (the others are prefix fusion), so it tests whether reliance depends on
the fusion design rather than on the modality.

It is also the zoo's model for vitals and labs: it takes x + v + l + n, with vitals and labs as text blocks in the
prompt (models/clinical_text.py), so it runs `cxr_notes`, `cxr_vitals_labs_notes`, `vitals_notes`, `labs_notes`.

Differences from the Gemma-style models in hf_vlm:
- `signal_mean_state` is the final-layer state at the single <|image|> token, not a mean over image tokens: there
  are no image tokens in the sequence. It is kept so the readout set matches the other models;
- encoder_mean / projector_mean average only the film's real tiles (aspect_ratio_mask), not the padding tiles
  the processor adds to reach 4;
- the encoder module is `vision_model` (hf_vlm looks for `vision_tower`);
- the tokenizer has no pad token, so batches pad with <|finetune_right_pad_id|> (or EOS if absent);
- no mean swap yet: the projector output depends on the film's tile layout, so one cohort mean does not fit every
  film. Records with a `mean` slot are skipped (runner.select).

Gated on Hugging Face: accept the Llama 3.2 Community License on the model page with the account whose token the
VM uses. bf16 weights are ~21 GB, so run it on an A100 40GB or larger (an L4's 24 GB is too tight).
"""
import numpy as np

from .base import ReadoutSpec, register, to_np
from .hf_vlm import HFImageTextAdapter, find_module

PAD_TOKEN = '<|finetune_right_pad_id|>'


@register
class Llama32Vision11B(HFImageTextAdapter):
    name = 'llama3.2-vision-11b'
    hf_id = 'meta-llama/Llama-3.2-11B-Vision-Instruct'
    revision = 'main'                 # pin a commit once the smoke test passes
    signal, modalities, default_modset = 'cxr', ('x', 'v', 'l', 'n'), 'cxr_notes'
    meta = dict(fusion='cross-attention', domain='general', params='11B', mimic_exposure='none')
    torch_dtype = 'bfloat16'
    supports_mean = False
    readouts = HFImageTextAdapter.readouts[:-1] + [
        ReadoutSpec('signal projector mean', 'cross-attention', 'projector_mean')]

    def load(self, device='cuda'):
        super().load(device)
        self.vt_name, self.vt = find_module(self.model, 'vision_model')
        if self.tok.pad_token is None:
            vocab = self.tok.get_vocab()
            self.tok.pad_token = PAD_TOKEN if PAD_TOKEN in vocab else self.tok.eos_token
        gc = self.model.generation_config
        if gc.pad_token_id is None:
            gc.pad_token_id = self.tok.pad_token_id
            self.stop_ids |= {gc.pad_token_id}
        self._tiles = None
        return self

    def describe(self):
        d = super().describe()
        d.update(pad_token=self.tok.pad_token, pad_token_id=self.tok.pad_token_id,
                 max_tiles=getattr(self.proc.image_processor, 'max_image_tiles', None))
        return d

    def _generate(self, x):
        # which of the (up to 4) tiles of each film are real; read back by _prefusion
        self._tiles = x.get('aspect_ratio_mask')
        return super()._generate(x)

    def _tile_mean(self, t, i):
        """Mean over film i's real tiles of a (B, images, tiles, patches, H) hook output."""
        f = t[i].float()                                         # (images, tiles, patches, H)
        if self._tiles is not None:
            keep = self._tiles[i].bool().to(f.device)            # (images, tiles)
            if keep.any():
                f = f[keep]                                      # (real tiles, patches, H)
        return f.reshape(-1, f.shape[-1]).mean(0)

    def _prefusion(self, r, cap, has_signal, i=0, sizes=None):
        if not (has_signal and 'proj' in cap) or cap['proj'].dim() != 5:
            r = super()._prefusion(r, cap, has_signal, i, sizes)
            if not has_signal:   # the encoder output concatenates several layers: vision_output_dim, not hidden_size
                vd = getattr(self.model.config.vision_config, 'vision_output_dim', None)
                if vd:
                    r['encoder_mean'] = np.full(vd, np.nan, np.float32)
            return r
        r['projector_mean'] = to_np(self._tile_mean(cap['proj'], i))
        v = cap.get('vision')
        r['encoder_mean'] = to_np(self._tile_mean(v, i)) if v is not None and v.dim() == 5 \
            else super()._prefusion(dict(r), {}, False)['encoder_mean']
        return r
