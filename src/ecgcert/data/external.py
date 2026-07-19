"""Canonical full-cohort Chapman/CPSC WFDB loading with patient-level audits."""
from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from ecgcert.data.audit import AuditTrail, SignalAudit
from ecgcert.data.common import canonicalize_wfdb_record
from ecgcert.data.manifest import DatasetManifest
from ecgcert.data.ptbxl import PTBXL


class ExternalWFDBCohort:
    def __init__(self, manifest: DatasetManifest):
        self.manifest = manifest
        self.root = Path(manifest.root)
        self._records = {record.record_id: record for record in manifest.records}

    def signal_with_audit(self, record_id: str, rate: int = 500):
        import wfdb

        if record_id not in self._records:
            raise KeyError(record_id)
        item = self._records[record_id]
        # Read the exact header authenticated by the manifest.  ``record_id`` is
        # an identity, not an independently trusted filesystem path.
        source = (self.root / item.relative_header).with_suffix("")
        record = wfdb.rdrecord(str(source))
        signal, conversion = canonicalize_wfdb_record(record)
        source_rate = int(round(float(conversion["source_rate_hz"])))
        if source_rate <= 0:
            raise ValueError(f"invalid source rate {source_rate}")
        if source_rate != rate:
            divisor = gcd(source_rate, rate)
            signal = resample_poly(signal, rate // divisor, source_rate // divisor, axis=0)
        audit = SignalAudit(
            cohort=self.manifest.cohort,
            record_id=record_id,
            patient_id=item.patient_id,
            status="included",
            reason=None,
            requested_rate_hz=rate,
            source_rate_hz=conversion["source_rate_hz"],
            n_samples=int(signal.shape[0]),
            input_leads=conversion["input_leads"],
            input_units=conversion["input_units"],
            canonical_leads=conversion["canonical_leads"],
            source_channel_indices=conversion["source_channel_indices"],
            unit_scales_to_mv=conversion["unit_scales_to_mv"],
            output_unit=conversion["output_unit"],
        )
        return signal, audit

    def collect_all_segments_audited(
        self,
        record_ids,
        *,
        rate: int = 500,
        max_per_record: int = 40,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        rows = {segment: [] for segment in ("P", "QRS", "ST", "T")}
        record_groups = {segment: [] for segment in rows}
        patient_groups = {segment: [] for segment in rows}
        trail = AuditTrail()
        for record_id in record_ids:
            item = self._records[str(record_id)]
            base_audit = None
            try:
                signal, base_audit = self.signal_with_audit(str(record_id), rate=rate)
                if signal.shape[0] < 10 * rate:
                    raise ValueError("record shorter than 10 seconds")
                segments = PTBXL.segment_indices(signal, fs=rate, strict=True)
                counts = {segment: int(index.size) for segment, index in segments.items()}
                if not any(counts.values()):
                    raise ValueError("no valid delineated segments")
                trail.append(SignalAudit(**{**base_audit.__dict__, "segment_counts": counts}))
            except Exception as exc:
                retained = {}
                if base_audit is not None:
                    retained = {
                        "source_rate_hz": base_audit.source_rate_hz,
                        "n_samples": base_audit.n_samples,
                        "input_leads": base_audit.input_leads,
                        "input_units": base_audit.input_units,
                        "canonical_leads": base_audit.canonical_leads,
                        "source_channel_indices": base_audit.source_channel_indices,
                        "unit_scales_to_mv": base_audit.unit_scales_to_mv,
                        "output_unit": base_audit.output_unit,
                    }
                trail.append(
                    SignalAudit(
                        cohort=self.manifest.cohort,
                        record_id=str(record_id),
                        patient_id=item.patient_id,
                        status="excluded",
                        reason=f"{type(exc).__name__}: {exc}",
                        requested_rate_hz=rate,
                        **retained,
                    )
                )
                continue
            for segment, index in segments.items():
                if index.size == 0:
                    continue
                if index.size > max_per_record:
                    index = rng.choice(index, max_per_record, replace=False)
                rows[segment].append(signal[index])
                record_groups[segment].append(np.full(index.size, str(record_id), dtype=object))
                patient_groups[segment].append(np.full(index.size, item.patient_id, dtype=object))

        out = {}
        for segment in rows:
            if rows[segment]:
                out[segment] = (
                    np.vstack(rows[segment]),
                    np.concatenate(record_groups[segment]),
                    np.concatenate(patient_groups[segment]),
                )
            else:
                out[segment] = (
                    np.zeros((0, 12)),
                    np.zeros(0, dtype=object),
                    np.zeros(0, dtype=object),
                )
        return out, trail
