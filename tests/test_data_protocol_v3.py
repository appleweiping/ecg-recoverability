from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ecgcert.data.audit import AuditTrail, SignalAudit
from ecgcert.data.common import CANONICAL_LEADS, canonicalize_wfdb_record
from ecgcert.data.manifest import build_wfdb_manifest


def test_wfdb_conversion_reorders_and_converts_to_mv():
    names = list(reversed(CANONICAL_LEADS))
    values = np.tile(np.arange(12, dtype=float), (20, 1))
    record = SimpleNamespace(sig_name=names, p_signal=values, units=["uV"] * 12, fs=500)
    signal, audit = canonicalize_wfdb_record(record)
    expected = np.asarray([names.index(lead) for lead in CANONICAL_LEADS]) * 1e-3
    assert np.allclose(signal[0], expected)
    assert audit["unit_scales_to_mv"] == (1e-3,) * 12


def test_audit_trail_counts_explicit_exclusions():
    trail = AuditTrail()
    trail.append(SignalAudit("x", "1", "p1", "included", None, 500))
    trail.append(SignalAudit("x", "2", "p2", "excluded", "bad leads", 500))
    summary = trail.summary()
    assert summary["n_included"] == summary["n_excluded"] == 1
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
