"""Shared prompt wording, so every model is asked the same question in the same words.

Para 0 is exactly the LLaVA-Med notebook's LlavaPromptBuilder text (so new LLaVA-Med records match the stored
ones). The three arms differ by one block: image (signal + question), notes (report + question), fused (both).
Vitals and labs (text blocks from models/clinical_text.py) go between the expert line and the report; when either is present the
expert line names every input the prompt holds. Without them the text is unchanged, byte for byte.
Each adapter wraps `text()` in its own chat template and puts its own signal token where `compose()` says.

Factors (in the record id as ~order / ~para):
    order  'default' | 'flip': the model's usual slot order, or the other one. For the shared wording the default
           is signal first, so 'flip' puts the signal after the report; an adapter with its own layout
           (OpenTSLM: report before the series) flips its own default
    para   0 = the original question; 1, 2 = paraphrases
"""
from typing import Optional

SIGNALS = {
    'cxr': dict(expert_signal='You are an expert radiologist reviewing a chest radiograph from an ICU stay.',
                expert_fused='You are an expert clinician reviewing a chest radiograph alongside a radiology '
                             'report from the same ICU stay.'),
    'ecg': dict(expert_signal='You are an expert cardiologist reviewing a 12-lead ECG from an ICU stay.',
                expert_fused='You are an expert clinician reviewing a 12-lead ECG alongside a radiology '
                             'report from the same ICU stay.'),
}
EXPERT_NOTES = 'You are an expert clinician reviewing a radiology report from an ICU stay.'
QUESTIONS = [
    'Is this patient at high risk of dying within one year of this ICU admission?',
    'Will this patient most likely die within one year of this ICU admission?',
    'Is death within one year of this ICU admission likely for this patient?',
]
DIRECT = ('Instructions:\n'
          '- Answer immediately with a single word.\n'
          '- Do not explain, do not give a rationale, do not restate the question.\n'
          'Respond with exactly one word, Yes or No.')
ANSWER_CUE = ' Answer:'
SIGNAL_MARK = '<<SIGNAL>>'          # where the adapter puts its image / series token(s)


def arm(has_signal: bool, report: Optional[str]) -> str:
    return 'fused' if (has_signal and report) else ('signal' if has_signal else 'notes')


SIGNAL_NOUN = {'cxr': 'a chest radiograph', 'ecg': 'a 12-lead ECG'}


def _join(parts):
    return parts[0] if len(parts) == 1 else ', '.join(parts[:-1]) + ' and ' + parts[-1]


def expert_line(has_signal: bool, report: Optional[str], signal: str, vitals: Optional[str],
                labs: Optional[str]) -> str:
    """The expert sentence. Without vitals / labs: the original three sentences."""
    if not (vitals or labs):
        return {'fused': SIGNALS[signal]['expert_fused'], 'signal': SIGNALS[signal]['expert_signal'],
                'notes': EXPERT_NOTES}[arm(has_signal, report)]
    parts = ([SIGNAL_NOUN[signal]] if has_signal else []) + (['vital signs'] if vitals else []) \
        + (['laboratory results'] if labs else []) + (['a radiology report'] if report else [])
    where = 'from the same ICU stay' if len(parts) > 1 else 'from an ICU stay'
    return f'You are an expert clinician reviewing {_join(parts)} {where}.'


def text(has_signal: bool, report: Optional[str], signal: str = 'cxr', order: str = 'default', para: int = 0,
         vitals: Optional[str] = None, labs: Optional[str] = None) -> str:
    """The user turn, with SIGNAL_MARK where the signal goes (absent when there is no signal). `vitals` / `labs`:
    the text blocks of models.clinical_text, or None."""
    body = expert_line(has_signal, report, signal, vitals, labs)
    for block in (vitals, labs):
        if block:
            body += f'\n\n{block}'
    if report:
        body += f'\n\nRadiology report:\n{report}'
    body += f'\n\nQuestion: {QUESTIONS[int(para)]}\n\n{DIRECT}'
    if not has_signal:
        return body
    if order == 'flip' and report:                       # the signal right after the report block
        head, tail = body.split('\n\nQuestion: ', 1)
        return f'{head}\n\n{SIGNAL_MARK}\n\nQuestion: {tail}'
    return f'{SIGNAL_MARK}\n{body}'                    # signal first, as in the LLaVA-Med runs
