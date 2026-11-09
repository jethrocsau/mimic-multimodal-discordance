"""InternVL3-8B (the transformers-native `-hf` checkpoint): a general VLM with no MIMIC-CXR training. Prefix fusion:
InternViT-300M features, pixel-shuffled to 256 tokens per 448x448 tile, through an MLP `multi_modal_projector` into
a Qwen2.5-7B LLM.

Differences from the Gemma-style models in hf_vlm:
- InternVL normally cuts a film into up to 12 tiles plus a thumbnail. Here every film is one 448x448 tile
  (crop_to_patches=False), so each film gives exactly 256 image tokens, like gemma3-4b, and the projector output
  comes out (B, 256, H);
- no mean swap yet (not checked against stored runs). It also takes vitals and labs (text blocks).
"""
from .base import register
from .hf_vlm import HFImageTextAdapter


@register
class InternVL3_8B(HFImageTextAdapter):
    name = 'internvl3-8b'
    hf_id = 'OpenGVLab/InternVL3-8B-hf'
    revision = 'main'                 # pin a commit once the smoke test passes
    signal, modalities, default_modset = 'cxr', ('x', 'v', 'l', 'n'), 'cxr_notes'
    meta = dict(fusion='prefix (MLP projector, pixel shuffle)', domain='general', params='8B', mimic_exposure='none')
    torch_dtype = 'bfloat16'
    supports_mean = False
    proc_kwargs = dict(crop_to_patches=False)        # one 448x448 tile per film: 256 image tokens
