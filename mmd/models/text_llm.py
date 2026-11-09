"""Text-only LLMs: the same question over vitals, labs and the report, with no image (no signal encoder at all).

Llama-3.1-8B-Instruct is the language model inside llama3.2-vision-11b (whose text weights were kept frozen when
the vision layers were added), so the pair separates what the image pathway changes from what the language model
already does with the same text: compare llama3.1-8b on vitals_labs_notes with llama3.2-vision-11b's
image-ablated arm of cxr_vitals_labs_notes.

Readouts: yes_probability / yes_log_odds and the prompt / answer states, from the same greedy generate() call as the
VLMs (hf_vlm.readouts_from_generate). There is no signal, so no signal / pre-fusion readouts.

Check in --smoke: the Llama 3.1 chat template adds a system header (knowledge cutoff and a fixed date) that the
Llama 3.2 Vision template may not; describe() prints the start of the prompt.

Gated on Hugging Face: accept the Llama 3.1 Community License on the model page with the account whose token the
VM uses.
"""
from .. import prompts
from .base import Adapter, Sample, register
from .clinical_text import labs_text, vitals_text
from .hf_vlm import HFImageTextAdapter, yes_no_ids

PAD_TOKEN = '<|finetune_right_pad_id|>'


class TextLLMAdapter(HFImageTextAdapter):
    """Any chat LLM (AutoModelForCausalLM + its chat template) on the text modalities."""
    modalities = ('v', 'l', 'n')
    default_modset = 'vitals_labs_notes'
    supports_mean = False
    readouts = [r for r in Adapter.readouts if r.key != 'signal_mean_state']
    note_blind = ()

    def load(self, device='cuda'):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = self.proc = AutoTokenizer.from_pretrained(self.hf_id, revision=self.revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id, revision=self.revision, torch_dtype=getattr(torch, self.torch_dtype), device_map=device).eval()
        self.tok.padding_side = 'left'
        if self.tok.pad_token is None:
            self.tok.pad_token = PAD_TOKEN if PAD_TOKEN in self.tok.get_vocab() else self.tok.eos_token
        gc = self.model.generation_config
        if gc.pad_token_id is None:
            gc.pad_token_id = self.tok.pad_token_id
        self.stop_ids = set(gc.eos_token_id if isinstance(gc.eos_token_id, (list, tuple)) else
                            [gc.eos_token_id] if gc.eos_token_id is not None else []) | {gc.pad_token_id}
        self.dev = next(self.model.parameters()).device
        self.yes_ids, self.no_ids = yes_no_ids(self.tok)
        self.image_token_id = None
        self.vt_name = self.vt = self.pj_name = self.pj = None
        return self

    def describe(self):
        d = lambda ids: [self.tok.decode([i]) for i in ids]
        demo = Sample('0|n:first', {}, {}, 'FINDINGS: none.', frozenset(), {})
        return dict(yes_ids=self.yes_ids, yes=d(self.yes_ids), no_ids=self.no_ids, no=d(self.no_ids),
                    pad_token=self.tok.pad_token, dtype=self.torch_dtype, prompt_start=self.prompt(demo)[:300])

    def image(self, sample: Sample):
        return None

    def messages(self, sample: Sample):
        body = prompts.text(False, sample.report, self.signal, sample.order, sample.para,
                            vitals=vitals_text(sample.signals.get('v')), labs=labs_text(sample.signals.get('l')))
        msgs = [{'role': 'user', 'content': body}]
        if self.system:
            msgs.insert(0, {'role': 'system', 'content': self.system})
        return msgs

    def _generate(self, x):
        x.pop('token_type_ids', None)    # some tokenizers return them; causal LMs' generate() rejects them
        return super()._generate(x)

    def _prefusion(self, r, cap, has_signal, i=0, sizes=None):
        return r                      # no signal encoder: nothing before fusion


@register
class Llama31_8B(TextLLMAdapter):
    name = 'llama3.1-8b'
    hf_id = 'meta-llama/Llama-3.1-8B-Instruct'
    revision = 'main'                 # pin a commit once the smoke test passes
    meta = dict(fusion='none (text only)', domain='general', params='8B', mimic_exposure='none')
    torch_dtype = 'bfloat16'
