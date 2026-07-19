"""Canonical twelve-lead WFDB conversion shared by all cohorts."""
from __future__ import annotations

import numpy as np


CANONICAL_LEADS = (
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"
)


def _lead_key(value: str) -> str:
    return str(value).strip().replace(" ", "").upper()


def unit_scale_to_mv(unit: str | None) -> float:
    if unit is None or not str(unit).strip():
        raise ValueError("ECG channels require an explicit physical unit")
    key = str(unit).strip().replace("μ", "u").replace("µ", "u").lower()
    if key in {"mv", "millivolt", "millivolts"}:
        return 1.0
    if key in {"uv", "microvolt", "microvolts"}:
        return 1e-3
    if key in {"v", "volt", "volts"}:
        return 1e3
    raise ValueError(f"unsupported ECG unit: {unit!r}")


def canonicalize_wfdb_record(record) -> tuple[np.ndarray, dict]:
    """Return ``(T,12)`` mV samples and a conversion audit from a WFDB record."""

    names = tuple(str(value).strip() for value in (record.sig_name or ()))
    if len(names) != len(set(_lead_key(value) for value in names)):
        raise ValueError("duplicate lead names")
    lookup = {_lead_key(name): index for index, name in enumerate(names)}
    missing = [lead for lead in CANONICAL_LEADS if _lead_key(lead) not in lookup]
    if missing:
        raise ValueError(f"missing canonical leads: {missing}")

    raw = np.asarray(record.p_signal, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != len(names):
        raise ValueError(f"unexpected WFDB signal shape {raw.shape}")
    order = [lookup[_lead_key(lead)] for lead in CANONICAL_LEADS]
    raw_units = getattr(record, "units", None)
    if raw_units is None:
        raise ValueError("WFDB record does not declare physical channel units")
    units = tuple(str(value) for value in raw_units)
    if len(units) != len(names):
        raise ValueError("WFDB units do not match channel count")
    scales = np.asarray([unit_scale_to_mv(units[index]) for index in order], dtype=float)
    signal = raw[:, order] * scales[None, :]
    if not np.all(np.isfinite(signal)):
        raise ValueError("non-finite physical samples")
    source_rate_hz = float(record.fs)
    if not np.isfinite(source_rate_hz) or source_rate_hz <= 0:
        raise ValueError("WFDB record has an invalid sampling rate")
    return signal, {
        "input_leads": names,
        "input_units": units,
        "canonical_leads": CANONICAL_LEADS,
        "source_channel_indices": tuple(int(index) for index in order),
        "unit_scales_to_mv": tuple(float(value) for value in scales),
        "output_unit": "mV",
        "source_rate_hz": source_rate_hz,
    }
