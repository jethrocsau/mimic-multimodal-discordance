"""Vitals and labs as text, for models that read them in the prompt (llama3.2-vision-11b).

The loaders return raw data (data.VitalsLoader: an hourly Bundle with NaN where nothing was charted;
data.LabsLoader: a LabPanel, one lab draw). This is one model-side view of it: a short block per modality that
prompts.text() puts between the expert line and the report. No times are printed, so a baseline and a follow-up
record read alike and a temporal swap changes only the values.

    Vital signs:
    - Heart rate: 96 bpm (range 82-118 over 14 h)
    - Non-invasive BP systolic: 104 mmHg (range 92-131 over 12 h)

    Laboratory results:
    - Sodium: 131 mEq/L (low)
    - Creatinine: 2.4 mg/dL (high)

Vitals: per channel, the last charted hour, and the range and number of charted hours when there are several.
Labs: per panel lab drawn, the value, flagged low / high against its reference range (or 'abnormal' when only
MIMIC's flag says so). Channels and labs with no value are left out.
"""
from typing import Optional

import numpy as np

VITAL_NAMES = {
    'heart_rate': 'Heart rate', 'temperature_f': 'Temperature', 'nibp_systolic': 'Non-invasive BP systolic',
    'nibp_diastolic': 'Non-invasive BP diastolic', 'nibp_mean': 'Non-invasive BP mean',
    'resp_rate': 'Respiratory rate', 'spo2': 'SpO2',
}
UNIT_NAMES = {'degF': '°F', 'degC': '°C'}
LAB_NAMES = {
    'bun': 'BUN', 'wbc': 'WBC', 'inr': 'INR', 'alt': 'ALT', 'ast': 'AST', 'ntprobnp': 'NT-proBNP',
    'troponin_t': 'Troponin T', 'anion_gap': 'Anion gap',
}


def fmt(v) -> str:
    """3 significant figures, plain notation."""
    v = float(v)
    if v == 0:
        return '0'
    s = f'{v:.3g}'
    if 'e' in s:
        s = f'{v:.0f}' if abs(v) >= 1 else f'{v:.4f}'.rstrip('0').rstrip('.')
    return s


def nice(name: str, table: dict) -> str:
    return table.get(name, name.replace('_', ' ').capitalize())


def vitals_text(bundle, title: str = 'Vital signs') -> Optional[str]:
    """data.Bundle of vitals (K channels x hours, NaN = not charted) -> text block, or None if nothing charted."""
    if bundle is None:
        return None
    units = bundle.units or [''] * len(bundle.names)
    lines = []
    for name, unit, row in zip(bundle.names, units, np.asarray(bundle.values, float)):
        got = row[~np.isnan(row)]
        if not len(got):
            continue
        u = UNIT_NAMES.get(unit, unit)
        line = f'- {nice(name, VITAL_NAMES)}: {fmt(got[-1])}{" " + u if u else ""}'
        if len(got) > 1:
            line += f' (range {fmt(got.min())}-{fmt(got.max())} over {len(got)} h)'
        lines.append(line)
    return f'{title}:\n' + '\n'.join(lines) if lines else None


def lab_flag(value, low, high, flag) -> str:
    if low == low and low is not None and value < low:        # low == low: not NaN
        return 'low'
    if high == high and high is not None and value > high:
        return 'high'
    return 'abnormal' if isinstance(flag, str) and flag.strip().lower() == 'abnormal' else ''


def labs_text(panel, title: str = 'Laboratory results') -> Optional[str]:
    """data.LabPanel (one draw; NaN rows = not drawn) -> text block, or None if no lab has a value."""
    if panel is None:
        return None
    lines = []
    for lab, r in panel.table.iterrows():
        if r.value != r.value or r.value is None:              # not drawn (or no numeric value)
            continue
        unit = r.unit if isinstance(r.unit, str) and r.unit.strip() else ''
        line = f'- {nice(lab, LAB_NAMES)}: {fmt(r.value)}{" " + unit if unit else ""}'
        f = lab_flag(float(r.value), r.ref_low, r.ref_high, r.flag)
        if f:
            line += f' ({f})'
        lines.append(line)
    return f'{title}:\n' + '\n'.join(lines) if lines else None
