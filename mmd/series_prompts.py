"""Prompts for the series models and their text-serialized baselines, kept apart from prompts.py.

Every existing prompt is left exactly as it is: prompts.py (the shared wording of the image-text models),
models/clinical_text.py (Zifan's vitals / labs blocks) and opentslm.FusedPromptBuilder (the ECG prompt of the
stored opentslm runs) are not edited. This module only borrows the question and the answer instructions
(prompts.QUESTIONS, prompts.DIRECT), so every model is still asked the same thing in the same words.

Who uses what:
    text()                   the user turn of ChatTS (series placeholders) and of the -ts text LLMs (text blocks)
    OpenTSLMSeriesBuilder    OpenTSLM records that carry vitals or labs (records with only ECG / note keep
                             FusedPromptBuilder)
    labs_body()              Zifan's lab lines (clinical_text.labs_text) without their title, under our header

Layout of text(): the expert line naming every input given, then one section per modality in the order
ECG -> vitals -> labs -> report, then the question and the instructions. `order='flip'` puts the series-bearing
sections (ECG, vitals) after the text-only ones (labs, report), as the image-text prompts flip the image.
A section is either a text block (the -ts models: the series written out as numbers) or a list of labelled
series placeholders (ChatTS: one TS_MARK per series, which the adapter turns into the model's own token).
"""
from typing import Dict, List, Optional, Sequence

import numpy as np

from .prompts import DIRECT, QUESTIONS

ORDER = ('e', 'v', 'l', 'n')
SERIES_MODS = ('e', 'v')
NOUN = {'e': 'a 12-lead ECG', 'v': 'bedside vital signs', 'l': 'laboratory results', 'n': 'a radiology report'}
HEADER = {'e': '12-lead ECG', 'v': 'Vital signs', 'l': 'Laboratory results', 'n': 'Radiology report'}
TS_MARK = '<<TS>>'               # one per series; the adapter replaces each with its series token(s)


def _join(parts: Sequence[str]) -> str:
    parts = list(parts)
    return parts[0] if len(parts) == 1 else ', '.join(parts[:-1]) + ' and ' + parts[-1]


def expert_line(mods: Sequence[str]) -> str:
    where = 'from the same ICU stay' if len(mods) > 1 else 'from an ICU stay'
    return f'You are an expert clinician reviewing {_join([NOUN[m] for m in mods])} {where}.'


def labs_body(panel) -> Optional[str]:
    """The lab lines of clinical_text.labs_text (value, unit, low / high), without its own title; None if no lab
    has a value."""
    from .models.clinical_text import labs_text
    t = labs_text(panel)
    return t.split('\n', 1)[1] if t else None


def text(blocks: Dict[str, Optional[str]] = None, series: Dict[str, List[str]] = None, order: str = 'default',
         para: int = 0) -> str:
    """The user turn. blocks: {mod: text}, a written-out section (None / '' = not given); series: {mod: [label per
    series]}, sections of labelled TS_MARK placeholders. A modality comes from one or the other."""
    blocks = {m: b for m, b in (blocks or {}).items() if b}
    series = {m: s for m, s in (series or {}).items() if s}
    if set(blocks) & set(series):
        raise ValueError(f'{sorted(set(blocks) & set(series))}: given both as text and as series')
    mods = [m for m in ORDER if m in blocks or m in series]
    if not mods:
        raise ValueError('nothing to put in the prompt')
    if order == 'flip':
        mods_shown = [m for m in mods if m not in SERIES_MODS] + [m for m in mods if m in SERIES_MODS]
    else:
        mods_shown = mods
    parts = [expert_line(mods)]
    for m in mods_shown:
        body = blocks[m] if m in blocks else '\n'.join(f'{label} {TS_MARK}' for label in series[m])
        parts.append(f'{HEADER[m]}:\n{body}')
    parts.append(f'Question: {QUESTIONS[int(para)]}')
    parts.append(DIRECT)
    return '\n\n'.join(parts)


class OpenTSLMSeriesBuilder:
    """OpenTSLM's prompt for records with vitals and / or labs, in the same layout and wording style as
    opentslm.FusedPromptBuilder (expert line, clinical context, task, question, instructions; the series with their
    labels; the post-prompt ending in "Answer:"), and the same sample() contract (pre_prompt, time_series_text,
    time_series, post_prompt). The ECG leads keep FusedPromptBuilder's label; each charted vitals channel is one more
    series with its own label (raw mean / std quoted, as for the leads). `flip` moves the text blocks (labs, report)
    after the series, into the post-prompt."""
    ECG_LABEL = "This is ECG Lead {name}, it has mean {mean:.4f} and std {std:.4f}:"
    VITAL_LABEL = "This is the {name} vital sign ({unit}), hourly over {hours} h, it has mean {mean:.4f} and std {std:.4f}:"
    DATA = {'e': 'the ECG data', 'v': 'the vital signs', 'l': 'the laboratory results', 'n': 'the report'}
    UNITS = {'degF': '°F'}

    def sample(self, ecg=None, vitals=None, labs: Optional[str] = None, report: Optional[str] = None, kind='sp',
               question: Optional[str] = None, flip: bool = False) -> dict:
        """ecg / vitals: prepared Bundles (z-scored, gaps filled; vitals without never-charted channels) or None;
        labs: labs_body() text or None."""
        import torch

        from .models.clinical_text import VITAL_NAMES, nice
        have = [m for m, x in (('e', ecg), ('v', vitals), ('l', labs), ('n', report)) if x is not None and
                (not isinstance(x, str) or x)]
        ctx = 'ICU admission.'
        if 'e' in have:
            ctx += ' 12-lead ECG clinical recording. Signal quality: adequate for analysis.'
        text_blocks = []
        if labs:
            text_blocks.append(f'{HEADER["l"]}:\n{labs}')
        if report:
            text_blocks.append(f'{HEADER["n"]}:\n{report}')
        block = f'Clinical Context: {ctx}'
        if text_blocks and not flip:
            block += '\n\n' + '\n\n'.join(text_blocks)
        data = _join([self.DATA[m] for m in have])
        pre = (f"{expert_line(have)}\n\n{block}\n\nYour task is to examine {data} and answer the following "
               f"question:\n\nQuestion: {question or QUESTIONS[0]}\n\n{DIRECT.rsplit(chr(10), 1)[0]}")
        post = (f"Based on your analysis of {data}, provide your answer.\n"
                f"Respond with exactly one word, Yes or No.\n\nAnswer:")
        if text_blocks and flip:
            post = '\n\n'.join(text_blocks) + '\n\n' + post
        texts, rows = [], []
        if ecg is not None:
            for i, n in enumerate(ecg.names):
                texts.append(self.ECG_LABEL.format(name=n, mean=ecg.mean[i], std=ecg.std[i]))
                rows.append(ecg.values[i])
        if vitals is not None:
            units = vitals.units or [''] * len(vitals.names)
            for i, n in enumerate(vitals.names):
                texts.append(self.VITAL_LABEL.format(name=nice(n, VITAL_NAMES).lower(),
                                                     unit=self.UNITS.get(units[i], units[i]),
                                                     hours=vitals.values.shape[1], mean=vitals.mean[i],
                                                     std=vitals.std[i]))
                rows.append(vitals.values[i])
        rows = [torch.from_numpy(np.ascontiguousarray(r, dtype=np.float32)) for r in rows]
        if kind == 'sp':
            ts = rows
        elif rows:                               # Flamingo: one (K, L) tensor, shorter series zero-padded
            L = max(len(r) for r in rows)
            ts = torch.stack([torch.nn.functional.pad(r, (0, L - len(r))) for r in rows])
        else:                                    # no series at all: a dummy no media token points to
            ts = torch.zeros(1, 4)
        return {'pre_prompt': pre, 'time_series_text': texts, 'time_series': ts, 'post_prompt': post}
