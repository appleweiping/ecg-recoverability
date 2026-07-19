"""PTB-XL loader: metadata, diagnostic superclasses, and per-segment sampling.

PTB-XL provides 21 799 twelve-lead records at 100 Hz (``records100/``) and 500 Hz
(``records500/``), a metadata table ``ptbxl_database.csv`` (with SCP codes,
recommended 10-fold split ``strat_fold``, and acquisition ``device``), and
``scp_statements.csv`` mapping each SCP code to a diagnostic superclass.

We expose:

* :attr:`PTBXL.meta` -- the metadata frame with a parsed ``superclass`` column,
* :meth:`PTBXL.signal` -- a (T, 12) lead array in the standard lead order,
* :meth:`PTBXL.segment_indices` -- P/QRS/ST/T time indices via NeuroKit2,
* :meth:`PTBXL.collect_segment_samples` -- pooled per-segment 12-lead sample
  vectors used to fit the dipolar subspace.

Lead order in PTB-XL WFDB files is exactly the clinical order
``[I, II, III, aVR, aVL, aVF, V1..V6]`` matching :data:`ecgcert.physics.LEADS`,
so signal columns are used positionally.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ecgcert.data.audit import AuditTrail, SignalAudit
from ecgcert.data.common import canonicalize_wfdb_record
from ecgcert.protocol import (
    PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD,
    SEGMENT_SAMPLING_ALGORITHM,
)

# PTB-XL diagnostic superclasses (scp_statements.diagnostic_class).
SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")

_DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "ptbxl"


def sample_segment_timepoints(
    indices: np.ndarray,
    *,
    cap: int,
    seed: int,
    namespace: str,
    record_id: int | str,
    segment: str,
) -> np.ndarray:
    """Select a deterministic, nested uniform-without-replacement prefix.

    SHA-256 derives an independent PCG64 stream for each
    ``namespace x record x segment`` cell.  A full permutation is generated and
    its first ``cap`` entries are retained, so a larger-cap sensitivity contains
    every timepoint selected by the primary cap when all other inputs agree.
    Returned indices are sorted back into temporal order.
    """

    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError("segment timepoint cap must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("segment sampling seed must be an integer")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("segment sampling namespace must be non-empty")
    if not isinstance(segment, str) or not segment:
        raise ValueError("segment name must be non-empty")
    raw = np.asarray(indices)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("segment indices must be a one-dimensional integer array")
    canonical = np.sort(raw.astype(np.int64, copy=False))
    if canonical.size != np.unique(canonical).size:
        raise ValueError("segment indices must be unique")
    if canonical.size <= cap:
        return canonical.copy()

    key = json.dumps(
        [
            SEGMENT_SAMPLING_ALGORITHM,
            int(seed),
            namespace,
            str(record_id),
            segment,
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(key).digest()[:16], "big")
    generator = np.random.Generator(np.random.PCG64(derived_seed))
    return np.sort(generator.permutation(canonical)[:cap])


class PTBXL:
    def __init__(self, root: str | Path = _DEFAULT_ROOT):
        self.root = Path(root)
        db = pd.read_csv(self.root / "ptbxl_database.csv", index_col="ecg_id")
        db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
        scp = pd.read_csv(self.root / "scp_statements.csv", index_col=0)
        scp = scp[scp["diagnostic"] == 1]
        self._code2class = scp["diagnostic_class"].to_dict()
        db["superclass"] = db["scp_codes"].apply(self._superclasses)
        self.meta = db

    def _superclasses(self, codes: dict) -> list[str]:
        out = {self._code2class.get(c) for c in codes}
        return sorted(x for x in out if isinstance(x, str))

    # ------------------------------------------------------------------ selection
    def ids_with_superclass(self, superclass: str, exclusive: bool = True,
                            folds=None) -> np.ndarray:
        """ECG ids whose diagnostic superclass set contains ``superclass``.

        ``exclusive`` keeps only records whose superclass set is exactly
        ``{superclass}`` (cleaner strata for the dipolarity contrast).
        ``folds`` optionally restricts to a set of ``strat_fold`` values.
        """
        m = self.meta
        if folds is not None:
            m = m[m["strat_fold"].isin(list(folds))]
        if exclusive:
            sel = m["superclass"].apply(lambda s: s == [superclass])
        else:
            sel = m["superclass"].apply(lambda s: superclass in s)
        return np.asarray(m.index[sel])

    # ---------------------------------------------------------------------- signal
    def patient_id(self, ecg_id: int) -> str:
        """Stable patient identifier used for clustering and leakage checks."""

        value = self.meta.loc[ecg_id, "patient_id"] if "patient_id" in self.meta else ecg_id
        return str(value)

    def signal_with_audit(self, ecg_id: int, rate: int = 100) -> tuple[np.ndarray, SignalAudit]:
        """Load, reorder and convert one record to canonical twelve-lead mV."""

        if rate not in {100, 500}:
            raise ValueError("PTB-XL provides only 100 or 500 Hz records")
        import wfdb

        col = "filename_lr" if rate == 100 else "filename_hr"
        rel = self.meta.loc[ecg_id, col]
        rec = wfdb.rdrecord(str(self.root / rel))
        signal, conversion = canonicalize_wfdb_record(rec)
        if not np.isclose(float(rec.fs), rate):
            raise ValueError(f"record {ecg_id} reports {rec.fs} Hz, expected {rate} Hz")
        audit = SignalAudit(
            cohort="PTB-XL",
            record_id=str(ecg_id),
            patient_id=self.patient_id(ecg_id),
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

    def signal(self, ecg_id: int, rate: int = 100) -> np.ndarray:
        """Return canonical ``(T,12)`` mV in ``[I,II,III,aVR,aVL,aVF,V1..V6]``."""

        return self.signal_with_audit(ecg_id, rate=rate)[0]

    # --------------------------------------------------------------- delineation
    @staticmethod
    def segment_indices(sig: np.ndarray, fs: int, rpeak_lead: int = 1,
                        method: str | None = None,
                        strict: bool = False) -> dict[str, np.ndarray]:
        """P/QRS/ST/T time indices via NeuroKit2 delineation.

        Delineation runs on one lead (default II) to locate fiducials shared across
        all leads.  Segments:

            P   : P onset -> P offset
            QRS : R/QRS onset -> QRS offset (S offset)
            ST  : QRS offset (J point) -> T onset
            T   : T onset -> T offset

        Returns ``{segment: int array of sample indices}``; missing waves yield
        empty arrays.  By default NeuroKit failures also return empties for legacy
        exploratory callers. Audited release paths set ``strict=True`` so the
        underlying segmentation failure is retained as an exclusion reason.
        """
        import neurokit2 as nk

        x = sig[:, rpeak_lead]
        try:
            _, rpeaks = nk.ecg_peaks(x, sampling_rate=fs)
            import os
            meth = method or os.environ.get("ECG_DELINEATOR", "dwt")   # robustness: ECG_DELINEATOR=peak
            _, waves = nk.ecg_delineate(x, rpeaks, sampling_rate=fs, method=meth)
        except Exception as exc:
            if strict:
                raise RuntimeError(f"NeuroKit2 delineation failed: {exc}") from exc
            return {s: np.array([], dtype=int) for s in ("P", "QRS", "ST", "T")}

        def arr(key):
            v = np.asarray(waves.get(key, []), dtype=float)
            return v[~np.isnan(v)].astype(int)

        p_on, p_off = arr("ECG_P_Onsets"), arr("ECG_P_Offsets")
        r_on, r_off = arr("ECG_R_Onsets"), arr("ECG_R_Offsets")
        t_on, t_off = arr("ECG_T_Onsets"), arr("ECG_T_Offsets")

        def spans(on, off, n):
            idx = []
            for a, b in _pair(on, off, n):
                idx.extend(range(a, b))
            return np.asarray(sorted(set(idx)), dtype=int)

        n = sig.shape[0]
        segs = {
            "P": spans(p_on, p_off, n),
            "QRS": spans(r_on, r_off, n),
            "T": spans(t_on, t_off, n),
        }
        # ST = J point (QRS offset) -> T onset, paired by nearest following T onset.
        st_idx = []
        for j in r_off:
            future = t_on[t_on > j]
            if future.size:
                st_idx.extend(range(int(j), int(future[0])))
        segs["ST"] = np.asarray(sorted(set(i for i in st_idx if 0 <= i < n)), dtype=int)
        return segs

    def collect_all_segments(
        self,
        ecg_ids,
        rate: int = 100,
        max_per_record: int = PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD,
        max_records: int | None = None,
        seed: int = 0,
        sampling_namespace: str = "PTB-XL",
    ) -> dict[str, np.ndarray]:
        """Pool per-segment 12-lead sample vectors, delineating each record ONCE.

        Returns ``{segment: (N, 12)}``.  Much faster than calling
        :meth:`collect_segment_samples` per segment (which re-delineates).
        """
        ids = list(ecg_ids)
        if max_records is not None:
            ids = ids[:max_records]
        rows: dict[str, list] = {s: [] for s in ("P", "QRS", "ST", "T")}
        for eid in ids:
            try:
                sig = self.signal(int(eid), rate=rate)
            except Exception:
                continue
            segs = self.segment_indices(sig, fs=rate)
            for s, idx in segs.items():
                if idx.size == 0:
                    continue
                idx = sample_segment_timepoints(
                    idx,
                    cap=max_per_record,
                    seed=seed,
                    namespace=sampling_namespace,
                    record_id=eid,
                    segment=s,
                )
                rows[s].append(sig[idx])
        return {s: (np.vstack(v) if v else np.zeros((0, 12))) for s, v in rows.items()}

    def collect_all_segments_with_ids(
        self,
        ecg_ids,
        rate: int = 100,
        max_per_record: int = PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD,
        max_records: int | None = None,
        seed: int = 0,
        sampling_namespace: str = "PTB-XL",
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Like :meth:`collect_all_segments` but also returns the source record id per sample.

        Returns ``{segment: (X (N,12), rec_ids (N,))}``. Needed for record-level bootstrap:
        resample records (not pooled samples), then re-pool the segments of the chosen
        records and refit -- pooled-sample bootstrap understates uncertainty because
        multiple samples from one record are not exchangeable.
        """
        ids = list(ecg_ids)
        if max_records is not None:
            ids = ids[:max_records]
        rows: dict[str, list] = {s: [] for s in ("P", "QRS", "ST", "T")}
        rids: dict[str, list] = {s: [] for s in ("P", "QRS", "ST", "T")}
        for eid in ids:
            try:
                sig = self.signal(int(eid), rate=rate)
            except Exception:
                continue
            segs = self.segment_indices(sig, fs=rate)
            for s, idx in segs.items():
                if idx.size == 0:
                    continue
                idx = sample_segment_timepoints(
                    idx,
                    cap=max_per_record,
                    seed=seed,
                    namespace=sampling_namespace,
                    record_id=eid,
                    segment=s,
                )
                rows[s].append(sig[idx])
                rids[s].append(np.full(idx.size, int(eid), dtype=np.int64))
        out = {}
        for s in rows:
            if rows[s]:
                out[s] = (np.vstack(rows[s]), np.concatenate(rids[s]))
            else:
                out[s] = (np.zeros((0, 12)), np.zeros(0, dtype=np.int64))
        return out

    def collect_all_segments_audited(
        self,
        ecg_ids,
        rate: int = 500,
        max_per_record: int = PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD,
        max_records: int | None = None,
        seed: int = 0,
        sampling_namespace: str = "PTB-XL",
    ):
        """Collect samples with record/patient clusters and explicit exclusions.

        Returns ``({segment: (X, record_ids, patient_ids)}, AuditTrail)``.
        """

        ids = list(ecg_ids)
        if max_records is not None:
            ids = ids[:max_records]
        rows = {segment: [] for segment in ("P", "QRS", "ST", "T")}
        record_groups = {segment: [] for segment in rows}
        patient_groups = {segment: [] for segment in rows}
        trail = AuditTrail()
        for eid in ids:
            patient_id = self.patient_id(int(eid))
            base_audit = None
            try:
                signal, base_audit = self.signal_with_audit(int(eid), rate=rate)
                segments = self.segment_indices(signal, fs=rate, strict=True)
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
                        cohort="PTB-XL",
                        record_id=str(eid),
                        patient_id=patient_id,
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
                index = sample_segment_timepoints(
                    index,
                    cap=max_per_record,
                    seed=seed,
                    namespace=sampling_namespace,
                    record_id=eid,
                    segment=segment,
                )
                rows[segment].append(signal[index])
                record_groups[segment].append(np.full(index.size, int(eid), dtype=np.int64))
                patient_groups[segment].append(np.full(index.size, patient_id, dtype=object))

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
                    np.zeros(0, dtype=np.int64),
                    np.zeros(0, dtype=object),
                )
        return out, trail

    def collect_segment_samples(
        self,
        ecg_ids,
        segment: str,
        rate: int = 100,
        max_per_record: int = PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD,
        max_records: int | None = None,
        seed: int = 0,
        sampling_namespace: str = "PTB-XL",
    ) -> np.ndarray:
        """Pool per-segment 12-lead sample vectors across records -> (N, 12)."""
        return self.collect_all_segments(
            ecg_ids,
            rate=rate,
            max_per_record=max_per_record,
            max_records=max_records,
            seed=seed,
            sampling_namespace=sampling_namespace,
        )[segment]


def _pair(on: np.ndarray, off: np.ndarray, n: int):
    """Pair each onset with the next offset after it (both within [0, n))."""
    on = on[(on >= 0) & (on < n)]
    off = off[(off >= 0) & (off < n)]
    for a in on:
        later = off[off > a]
        if later.size:
            yield int(a), int(later[0])
