"""Text-only baselines of the series models: the ECG and the vitals written out as text (models/series_text.py),
read by a chat LLM with no series encoder at all.

They answer one question: does a model rely on a signal differently when it reads numbers instead of an encoder
output? Each backbone is matched to a series model:
    qwen3-8b-tsfeat             Qwen3-8B, the LLM inside chatts-8b (same weights before ChatTS's alignment)
    llama3.1-8b-tsfeat          Llama-3.1-8B-Instruct, the backbone of Zifan's text-only llama3.1-8b, which reads a
                                short vitals summary and no ECG; it is left as it is
and the ECG as measurements from the tracing (rate, rhythm, PR, QRS, QT / QTc, axis): its content, not its form.
(TextSeriesLLM can also write the 12 leads out as numbers, ecg_text = 'raw', but no model is registered with it
for now: ~10k tokens per ECG, ~25x the cost.)
Vitals are the hourly grid (every hourly value), labs Zifan's clinical_text lines, the report as written.

Built on Zifan's TextLLMAdapter (loading, batching, readouts unchanged); only the prompt differs, and it comes from
series_prompts.text, so prompts.py and every existing model's prompt stay as they are. No mean swap (there is no
encoder output to replace).
"""
from .. import series_prompts as SP
from .base import Sample, register
from .series_text import ecg_features_text, ecg_raw_text, vitals_grid
from .text_llm import TextLLMAdapter


class TextSeriesLLM(TextLLMAdapter):
    """A chat LLM on the ECG, vitals, labs and report, all as text."""
    signal, modalities, default_modset = 'ecg', ('e', 'v', 'l', 'n'), 'ecg_notes'
    ecg_text = 'raw'                   # 'raw' (numbers) | 'features' (measurements)
    chat_kwargs: dict = {}             # extra apply_chat_template arguments (Qwen3: thinking off)

    def blocks(self, sample: Sample) -> dict:
        e = sample.signals.get('e')
        ecg = None if e is None else (ecg_raw_text(e) if self.ecg_text == 'raw' else ecg_features_text(e))
        return {'e': ecg, 'v': vitals_grid(sample.signals.get('v')),
                'l': SP.labs_body(sample.signals['l']) if 'l' in sample.signals else None, 'n': sample.report}

    def messages(self, sample: Sample):
        msgs = [{'role': 'user', 'content': SP.text(blocks=self.blocks(sample), order=sample.order,
                                                    para=sample.para)}]
        if self.system:
            msgs.insert(0, {'role': 'system', 'content': self.system})
        return msgs

    def prompt(self, sample: Sample) -> str:
        return self.proc.apply_chat_template(self.messages(sample), tokenize=False, add_generation_prompt=True,
                                             **self.chat_kwargs) + self.answer_cue


def _meta(params, ecg):
    return dict(fusion='none (text only)', domain='general', params=params, mimic_exposure='none',
                presentation=f'text-serialized (ECG {ecg})')


# The raw-number variants (ECG as 12 x 250 values, ~10k tokens, ~6.5 s per record on a GB10: ~25x the cost of
# -tsfeat) are not registered for now. To bring one back: a subclass with ecg_text = 'raw', registered and added to
# base.MODULES, e.g. `qwen3-8b-ts`.


@register
class Llama31_8B_TSFeat(TextSeriesLLM):
    name = 'llama3.1-8b-tsfeat'
    hf_id = 'meta-llama/Llama-3.1-8B-Instruct'
    revision = 'main'                 # pin a commit once the smoke test passes
    ecg_text = 'features'
    meta = _meta('8B', 'as measurements')


@register
class Qwen3_8B_TSFeat(TextSeriesLLM):
    name = 'qwen3-8b-tsfeat'
    hf_id = 'Qwen/Qwen3-8B'
    revision = 'main'                 # pin a commit once the smoke test passes
    chat_kwargs = dict(enable_thinking=False)
    ecg_text = 'features'
    meta = _meta('8B', 'as measurements')
