"""Series modalities written out as text, for the text-only baselines of the series models (models/ts_text.py).

A text LLM reads the same records the series models read, with no series encoder: the ECG and the vitals become
numbers or measurements in the prompt. Every function is deterministic (fixed precision, '–' for missing).

    vitals_grid(bundle)        the vitals window, one line per charted channel: every hourly value in order
    ecg_raw_text(bundle)       the 12 leads at 25 Hz (bin means of the 500 Hz record), 250 values per lead in mV
    ecg_features_text(bundle)  measurements from the raw ECG (neurokit2 on lead II, axis from I and aVF): heart rate,
                               rhythm regularity, PR, QRS, QT / QTc, frontal axis

The raw-number form is as close as text gets to what OpenTSLM and ChatTS are given; the measurements are what a
reader of the tracing would take from it. Comparing the two separates the format (numbers an LLM cannot read as a
waveform) from the content (what the waveform says).
"""
from collections import OrderedDict
from threading import Lock

import numpy as np

from .clinical_text import VITAL_NAMES, nice

UNITS = {'degF': '°F', 'degC': '°C'}
VITAL_DIGITS = {'temperature_f': 1}            # the rest in whole units
ECG_HZ = 25
_cache, _lock, CACHE = OrderedDict(), Lock(), 256


def _num(x, d=0):
    return '–' if x is None or not np.isfinite(x) else f'{x:.{d}f}'


def vitals_grid(bundle):
    """data.Bundle of vitals (K channels x L hours, NaN = not charted) -> one line per charted channel, or None."""
    if bundle is None:
        return None
    v = np.asarray(bundle.values, float)
    units = bundle.units or [''] * len(bundle.names)
    L = v.shape[1]
    lines = []
    for name, unit, row in zip(bundle.names, units, v):
        if not np.isfinite(row).any():
            continue
        u = UNITS.get(unit, unit)
        d = VITAL_DIGITS.get(name, 0)
        head = f'{nice(name, VITAL_NAMES)}{f" ({u})" if u else ""}, hourly, h0–h{L - 1}'
        lines.append(f'{head}: ' + ', '.join(_num(x, d) for x in row))
    if not lines:
        return None
    return '\n'.join([f'Hourly means over a {L} h window, in time order ("–" = not charted).'] + lines)


def ecg_raw_text(bundle, hz: int = ECG_HZ):
    """Raw 12-lead Bundle (mV, 500 Hz) -> 12 lines of `hz` x 10 s values (mean of each bin, gaps filled), 2 decimals."""
    if bundle is None:
        return None
    from ..data import prepare
    b = prepare(bundle, hz=hz, length=int(hz * 10), how='mean', fill=True)
    lines = [f'{len(b.names)} leads, {b.values.shape[1]} samples each at {hz} Hz over 10 s, in mV.']
    lines += [f'Lead {n}: ' + ', '.join(f'{x:.2f}' for x in row) for n, row in zip(b.names, b.values)]
    return '\n'.join(lines)


def ecg_features(bundle) -> dict:
    """Measurements from a raw 12-lead Bundle: hr (bpm), rr_cv, pr / qrs / qt / qtc (ms), axis (degrees); NaN where
    the delineation fails. Cached by the record (bundle.info['record'])."""
    key = (bundle.info or {}).get('record')
    if key is not None:
        with _lock:
            if key in _cache:
                _cache.move_to_end(key)
                return _cache[key]
    out = _measure(bundle)
    if key is not None:
        with _lock:
            _cache[key] = out
            while len(_cache) > CACHE:
                _cache.popitem(last=False)
    return out


def _measure(bundle) -> dict:
    import warnings

    import neurokit2 as nk
    fs = float(bundle.fs)
    sig = {n.lower(): np.nan_to_num(np.asarray(x, float)) for n, x in zip(bundle.names, bundle.values)}
    nan = float('nan')
    out = dict(hr=nan, rr_cv=nan, pr=nan, qrs=nan, qt=nan, qtc=nan, axis=nan)
    ii = sig.get('ii')
    if ii is None or not np.any(ii):
        return out
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            clean = nk.ecg_clean(ii, sampling_rate=fs)
            _, info = nk.ecg_peaks(clean, sampling_rate=fs)
            r = np.asarray(info['ECG_R_Peaks'], int)
        except Exception:
            return out
        if len(r) < 3:
            return out
        rr = np.diff(r) / fs
        out['hr'] = 60.0 / float(np.median(rr))
        out['rr_cv'] = float(np.std(rr) / np.mean(rr))
        try:
            _, w = nk.ecg_delineate(clean, r, sampling_rate=fs, method='dwt')
            ms = lambda a, b: float(np.nanmedian((np.asarray(w[b], float) - np.asarray(w[a], float)) / fs * 1000))
            out['pr'] = ms('ECG_P_Onsets', 'ECG_R_Onsets')
            out['qrs'] = ms('ECG_R_Onsets', 'ECG_R_Offsets')
            out['qt'] = ms('ECG_R_Onsets', 'ECG_T_Offsets')
            out['qtc'] = out['qt'] / np.sqrt(float(np.median(rr)))
        except Exception:
            pass
        lead_i, avf = sig.get('i'), sig.get('avf')
        if lead_i is not None and avf is not None:
            h = int(0.06 * fs)                       # +-60 ms around each R peak: the QRS complex
            net = lambda x: float(np.mean([x[max(p - h, 0):p + h].max() + x[max(p - h, 0):p + h].min() for p in r]))
            out['axis'] = float(np.degrees(np.arctan2(net(avf), net(lead_i))))
    return {k: (v if v is None or np.isfinite(v) else nan) for k, v in out.items()}


def ecg_features_text(bundle):
    """ecg_features() as text lines, 'not measurable' where a value is missing."""
    if bundle is None:
        return None
    f = ecg_features(bundle)
    ms = lambda k: f'{f[k]:.0f} ms' if np.isfinite(f[k]) else 'not measurable'
    rhythm = ('not measurable' if not np.isfinite(f['rr_cv']) else
              f"{'regular' if f['rr_cv'] < 0.10 else 'irregular'} (RR interval variation {100 * f['rr_cv']:.0f}%)")
    axis = f"{f['axis']:+.0f}°" if np.isfinite(f['axis']) else 'not measurable'
    hr = f"{f['hr']:.0f} bpm" if np.isfinite(f['hr']) else 'not measurable'
    return '\n'.join([
        'Measured from the 12-lead tracing (10 s):',
        f'Heart rate: {hr}',
        f'Rhythm: {rhythm}',
        f'PR interval: {ms("pr")}',
        f'QRS duration: {ms("qrs")}',
        f'QT interval: {ms("qt")}',
        f'QTc (Bazett): {ms("qtc")}',
        f'Frontal QRS axis: {axis}',
    ])
