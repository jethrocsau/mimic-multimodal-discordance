"""Gemma-3-4B-it: a general VLM with no MIMIC-CXR training. Prefix fusion: SigLIP image features, 256 tokens
through `multi_modal_projector`.

Gated on Hugging Face: accept the terms on the model page with the account whose token the VM uses.
"""
from .base import register
from .hf_vlm import HFImageTextAdapter


@register
class Gemma3_4B(HFImageTextAdapter):
    name = 'gemma3-4b'
    hf_id = 'google/gemma-3-4b-it'
    revision = 'main'                 # pin a commit once the smoke test passes
    signal, modalities, default_modset = 'cxr', ('x', 'n'), 'cxr_notes'
    meta = dict(fusion='prefix (projector)', domain='general', params='4B', mimic_exposure='none')
    torch_dtype = 'bfloat16'
