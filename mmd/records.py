"""The record id: one string that names a whole input configuration.

    <stay>|<mod>:<slot>[|<mod>:<slot>]...[|~<option>=<value>]...
    mod  ::= e (ECG) | x (CXR) | v (vitals) | l (labs) | n (notes)
    slot ::= first | last | none | mean | d<donor_stay>:first

    12345|x:first|n:first                       reference: the stay's own baseline records
    12345|x:last|n:first                        within-patient (temporal) swap of the image
    12345|x:d9876:first|n:first                 donor swap: stay 9876's baseline image, own note
    12345|e:first|x:d9876:first|n:d9876:first   joint donor swap: image and note both from stay 9876
    12345|x:mean|n:first                        image replaced by the cohort-mean encoder output
    12345|x:first|n:none                        note removed (ablation)
    12345|x:first|n:first|~order=flip|~para=1   prompt factors: other slot order, paraphrase 1

Modality keys appear in MOD_ORDER and options sorted by name, so a configuration has exactly one spelling: the
manifest, the embedding store, resuming, the index and the probe all key on this string.
"""
from typing import Dict, Tuple

from .modsets import MOD_ORDER


def record_id(stay_id, slots: Dict[str, str], options: Dict[str, str] = None) -> str:
    bad = set(slots) - set(MOD_ORDER)
    if bad:
        raise ValueError(f'unknown modality key(s) {sorted(bad)}: use {MOD_ORDER}')
    r = f"{int(stay_id)}|" + "|".join(f"{m}:{slots[m]}" for m in MOD_ORDER if m in slots)
    if options:
        r += "".join(f"|~{k}={options[k]}" for k in sorted(options))
    return r


def parse_record(record: str) -> Tuple[int, Dict[str, str]]:
    """'12345|e:d9876:first|n:none' -> (12345, {'e': 'd9876:first', 'n': 'none'}). Options are ignored."""
    head, *parts = record.split('|')
    return int(head), {p.split(':', 1)[0]: p.split(':', 1)[1] for p in parts if not p.startswith('~')}


def parse_options(record: str) -> Dict[str, str]:
    return dict(p[1:].split('=', 1) for p in record.split('|')[1:] if p.startswith('~'))


def donor_slot(donor: int, slot: str = 'first') -> str:
    return f'd{int(donor)}:{slot}'
