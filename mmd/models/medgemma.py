"""MedGemma-1.5-4B-it: Gemma 3 4B further trained on medical data, including MIMIC-CXR. Prefix fusion as in
gemma3-4b (SigLIP features, 256 tokens through `multi_modal_projector`), so the two form a matched pair: same
architecture and size, general vs medical training.

MIMIC exposure: it has seen MIMIC-CXR films and reports in training, so it may recall the pairing rather than read
the inputs. It is flagged `mimic_exposure`; keep it out of the main comparison (`exclude_exposed`) and analyse it
as a sensitivity check, ideally on stays whose films are in the MIMIC-CXR official test split.

It also takes vitals and labs (as text blocks, models/clinical_text.py).

Gated on Hugging Face (Health AI Developer Foundations terms): accept them on the model page with the account whose
token the VM uses.
"""
from .base import register
from .gemma import Gemma3_4B


@register
class MedGemma15_4B(Gemma3_4B):
    name = 'medgemma1.5-4b'
    hf_id = 'google/medgemma-1.5-4b-it'
    revision = 'main'                 # pin a commit once the smoke test passes
    signal, modalities, default_modset = 'cxr', ('x', 'v', 'l', 'n'), 'cxr_notes'
    meta = dict(fusion='prefix (projector)', domain='medical', params='4B',
                mimic_exposure='MIMIC-CXR images and reports (training)')
