"""MedGemma-4B-it (Gemma 3 + MedSigLIP): prefix fusion, 256 image tokens through `multi_modal_projector`.

Gated on Hugging Face: accept the terms on the model page with the account whose token the VM uses.
Trained with MIMIC-CXR among its data, so it is flagged as MIMIC-exposed for the zoo table.
"""
from .base import register
from .hf_vlm import HFImageTextAdapter


@register
class MedGemma4B(HFImageTextAdapter):
    name = 'medgemma-4b'
    hf_id = 'google/medgemma-4b-it'
    revision = 'main'                 # pin a commit once the smoke test passes
    signal, pair = 'cxr', 'cxr_notes'
    meta = dict(fusion='prefix (projector)', domain='medical', params='4B', mimic_exposure='MIMIC-CXR')
    torch_dtype = 'bfloat16'
