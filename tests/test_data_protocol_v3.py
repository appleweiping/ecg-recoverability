from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ecgcert.data.audit import AuditTrail, SignalAudit
from ecgcert.data.common import CANONICAL_LEADS, canonicalize_wfdb_record
from ecgcert.data.manifest import build_wfdb_manifest
from ecgcert.data.ptbxl import PTBXL, sample_segment_timepoints
from ecgcert.protocol import (
    PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD,
    SEGMENT_SAMPLING_SEED,
    SENSITIVITY_SEGMENT_SAMPLE_CAP_PER_RECORD,
)


def test_wfdb_conversion_reorders_and_converts_to_mv():
    names = list(reversed(CANONICAL_LEADS))
    values = np.tile(np.arange(12, dtype=float), (20, 1))
    record = SimpleNamespace(sig_name=names, p_signal=values, units=["uV"] * 12, fs=500)
    signal, audit = canonicalize_wfdb_record(record)
    expected = np.asarray([names.index(lead) for lead in CANONICAL_LEADS]) * 1e-3
    assert np.allclose(signal[0], expected)
    assert audit["canonical_leads"] == CANONICAL_LEADS
    assert audit["source_channel_indices"] == tuple(
        names.index(lead) for lead in CANONICAL_LEADS
    )
    assert audit["unit_scales_to_mv"] == (1e-3,) * 12
    assert audit["output_unit"] == "mV"


def test_wfdb_conversion_rejects_missing_units():
    record = SimpleNamespace(
        sig_name=CANONICAL_LEADS,
        p_signal=np.zeros((20, 12)),
        units=None,
        fs=500,
    )
    with pytest.raises(ValueError, match="physical channel units"):
        canonicalize_wfdb_record(record)


def test_strict_delineation_retains_neurokit_failure(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("fixture delineator failure")

    monkeypatch.setitem(sys.modules, "neurokit2", SimpleNamespace(ecg_peaks=fail))
    signal = np.zeros((5000, 12))
    legacy = PTBXL.segment_indices(signal, fs=500)
    assert all(not values.size for values in legacy.values())
    with pytest.raises(RuntimeError, match="fixture delineator failure"):
        PTBXL.segment_indices(signal, fs=500, strict=True)


def test_keyed_timepoint_cap_is_deterministic_and_nested_for_sensitivity():
    indices = np.arange(137, dtype=np.int64)
    kwargs = {
        "seed": SEGMENT_SAMPLING_SEED,
        "namespace": "PTB-XL/folds1-7/spatial-map-fit",
        "record_id": 123,
        "segment": "QRS",
    }
    primary = sample_segment_timepoints(
        indices,
        cap=PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD,
        **kwargs,
    )
    sensitivity = sample_segment_timepoints(
        indices,
        cap=SENSITIVITY_SEGMENT_SAMPLE_CAP_PER_RECORD,
        **kwargs,
    )

    assert len(primary) == PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD
    assert len(sensitivity) == SENSITIVITY_SEGMENT_SAMPLE_CAP_PER_RECORD
    assert np.array_equal(primary, np.sort(primary))
    assert set(primary) < set(sensitivity)
    assert np.array_equal(
        primary,
        sample_segment_timepoints(
            indices,
            cap=PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD,
            **kwargs,
        ),
    )
    other_record = sample_segment_timepoints(
        indices,
        cap=PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD,
        **{**kwargs, "record_id": 124},
    )
    assert not np.array_equal(primary, other_record)


def test_audit_trail_counts_explicit_exclusions():
    trail = AuditTrail()
    trail.append(SignalAudit("x", "1", "p1", "included", None, 500))
    trail.append(SignalAudit("x", "2", "p2", "excluded", "bad leads", 500))
    summary = trail.summary()
    assert summary["n_included"] == summary["n_excluded"] == 1
    assert summary["n_patients_total"] == 2
    assert summary["n_patients_with_included_records"] == 1
    assert summary["n_patients_all_records_excluded"] == 1
    assert summary["exclusion_reasons"] == {"bad leads": 1}
    assert len(summary["sha256"]) == 64


def test_wfdb_manifest_is_stable_and_patient_split_is_disjoint(tmp_path: Path):
    for index in range(30):
        (tmp_path / f"r{index}.hea").write_text(f"r{index} 12 500 5000\n", encoding="utf-8")
        (tmp_path / f"r{index}.mat").write_bytes(bytes([index]))
    patients = {f"r{index}": f"p{index // 2}" for index in range(30)}
    kwargs = {
        "cohort": "test",
        "version": "1",
        "source_url": "https://example.invalid",
        "root": tmp_path,
        "patient_by_record": patients,
        "split_salt": "test-v1",
    }
    manifest = build_wfdb_manifest(**kwargs)
    assert len(manifest.records) == 30
    assert manifest.sha256() == build_wfdb_manifest(**kwargs).sha256()
    manifest.split().validate()
