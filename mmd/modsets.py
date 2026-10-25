"""Modality sets: which modalities a cohort, a perturbation run and an analysis cover.

A modality set is a name plus a tuple of modality keys, e.g. `cxr_ecg_notes = ('e', 'x', 'n')`. Every cohort
manifest, run folder and result table is keyed by the set's name, so the same model can run several sets.

    e  ECG          x  CXR image       v  vitals        l  labs        n  radiology note

Sets come from the built-ins below, merged with a `name -> modalities` mapping from (later wins):
    mmd.toml        [mmd.modsets]  cxr_ecg_notes = ["e", "x", "n"]
    environment     MMD_MODSETS="cxr_ecg_notes=e,x,n;cxr_labs_notes=x,l,n"
    command line    --define cxr_ecg_notes=e,x,n      (repeatable)
    Python          ModalitySets({'cxr_ecg_notes': ('e', 'x', 'n')})
"""
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Sequence, Tuple

MOD_ORDER = ('e', 'x', 'v', 'l', 'n')          # the one spelling order of modality keys in a record id
MOD_NAMES = {'e': 'ECG', 'x': 'CXR image', 'v': 'vitals', 'l': 'labs', 'n': 'note'}
MOD_SHORT = {'e': 'ecg', 'x': 'cxr', 'v': 'vitals', 'l': 'labs', 'n': 'notes'}
MOD_BIT = {'v': 1, 'l': 2, 'e': 4, 'n': 8, 'x': 16}  # analysis_dataset bitmask (base_mask / rep_mask)
NOTE = 'n'
MAX_MODS = len(MOD_ORDER)

BUILTIN = {'cxr_notes': ('x', 'n'), 'ecg_notes': ('e', 'n')}


def canonical(mods: Iterable[str]) -> Tuple[str, ...]:
    """Modality keys validated, de-duplicated and put in MOD_ORDER."""
    mods = [m.strip() for m in mods if m.strip()]
    bad = sorted(set(mods) - set(MOD_ORDER))
    if bad:
        raise ValueError(f'unknown modality key(s) {bad}: use {MOD_ORDER} ({MOD_NAMES})')
    return tuple(m for m in MOD_ORDER if m in set(mods))


def auto_name(mods: Iterable[str]) -> str:
    """('n', 'x', 'e') -> 'ecg_cxr_notes'."""
    return '_'.join(MOD_SHORT[m] for m in canonical(mods))


def slot_name(mods: Iterable[str]) -> str:
    """The perturbed modalities as one slot label: ('n', 'x') -> 'x+n'."""
    return '+'.join(canonical(mods))


def slot_mods(slot: str) -> Tuple[str, ...]:
    """'x+n' -> ('x', 'n'); '' -> ()."""
    return tuple(slot.split('+')) if slot else ()


def slot_label(slot: str) -> str:
    """'x+n' -> 'CXR image + note'."""
    return ' + '.join(MOD_NAMES[m] for m in slot_mods(slot))


@dataclass(frozen=True)
class ModalitySet:
    name: str
    mods: Tuple[str, ...]

    def __post_init__(self):
        mods = canonical(self.mods)
        if len(mods) < 2:
            raise ValueError(f'modality set {self.name}: needs at least two modalities, got {mods}')
        object.__setattr__(self, 'mods', mods)

    @property
    def k(self) -> int:
        return len(self.mods)

    @property
    def signals(self) -> Tuple[str, ...]:
        """The non-note modalities (the ones a model encodes as a signal)."""
        return tuple(m for m in self.mods if m != NOTE)

    @property
    def label(self) -> str:
        return ' + '.join(MOD_NAMES[m] for m in self.mods)

    def fits(self, accepted: Sequence[str]) -> bool:
        """Can a model that accepts `accepted` take every modality of this set?"""
        return set(self.mods) <= set(accepted)


def parse_spec(spec: str) -> Tuple[str, Tuple[str, ...]]:
    """'cxr_ecg_notes=e,x,n' -> ('cxr_ecg_notes', ('e', 'x', 'n')); 'e,x,n' alone gets an auto name."""
    name, sep, mods = spec.partition('=')
    if not sep:
        name, mods = '', name
    mods = canonical(mods.replace('+', ',').split(','))
    return (name.strip() or auto_name(mods)), mods


def parse_env(value: str) -> Dict[str, Tuple[str, ...]]:
    return dict(parse_spec(s) for s in value.split(';') if s.strip())


class ModalitySets:
    """The registry: built-ins plus any user mapping. Look a set up by name with `sets[name]`."""

    def __init__(self, mapping: Mapping[str, Sequence[str]] = None, specs: Iterable[str] = ()):
        self._sets: Dict[str, ModalitySet] = {}
        for name, mods in {**BUILTIN, **dict(mapping or {})}.items():
            self.add(name, mods)
        for spec in specs:
            self.add(*parse_spec(spec))

    def add(self, name: str, mods: Sequence[str]) -> ModalitySet:
        if isinstance(mods, str):
            mods = mods.replace('+', ',').split(',')
        s = ModalitySet(name, tuple(mods))
        self._sets[name] = s
        return s

    def __getitem__(self, name: str) -> ModalitySet:
        if name not in self._sets:
            raise KeyError(f'unknown modality set {name!r}; defined: {", ".join(self.names())} '
                           f'(define one with --define {name}=x,n or [mmd.modsets] in mmd.toml)')
        return self._sets[name]

    get = __getitem__

    def __contains__(self, name) -> bool:
        return name in self._sets

    def __iter__(self):
        return iter(self._sets.values())

    def names(self):
        return list(self._sets)

    def compatible(self, accepted: Sequence[str]):
        """The sets a model that accepts `accepted` can run."""
        return [s for s in self if s.fits(accepted)]
