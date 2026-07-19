"""Versioned manifests for local PTB-XL, Chapman and CPSC WFDB cohorts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
from typing import Iterable, Mapping

from ecgcert.protocol import PatientSplit, patient_hash_split


PTBXL_RELEASE_CONTRACT: dict[str, object] = {
    "version": "1.0.3",
    "source_url": "https://physionet.org/content/ptb-xl/1.0.3/",
    "n_records": 21_799,
    "n_patients": 18_869,
    "population": "all_records_no_diagnosis_filter",
    "split_algorithm": "official-strat-folds-1-7_8_9_10-v1",
}


RELEASE_COHORT_CONTRACTS: dict[str, dict[str, object]] = {
    "chapman": {
        "version": "1.0.0",
        "source_url": "https://physionet.org/content/ecg-arrhythmia/1.0.0/",
        "n_records": 45_152,
        "n_patient_ids": 45_152,
        "patient_id_strategy": "official_one_patient_ecg_per_unique_wfdb_record_name",
    },
    "cpsc2018": {
        "version": "challenge-2020/1.0.2",
        "source_url": (
            "https://physionet.org/content/challenge-2020/1.0.2/"
            "training/cpsc_2018/"
        ),
        "n_records": 6_877,
        "n_patient_ids": 6_877,
        "patient_id_strategy": (
            "wfdb_record_name_pseudopatient_no_public_cross_record_patient_key"
        ),
    },
}


@dataclass(frozen=True)
class ManifestRecord:
    record_id: str
    patient_id: str
    relative_header: str
    header_sha256: str
    signal_file: str | None
    signal_size_bytes: int | None
    signal_sha256: str | None = None


@dataclass(frozen=True)
class DatasetManifest:
    cohort: str
    version: str
    source_url: str
    root: str
    records: tuple[ManifestRecord, ...]
    split_salt: str
    patient_id_strategy: str = "wfdb_record_name_or_explicit_mapping"

    def __post_init__(self) -> None:
        for name in ("cohort", "version", "source_url", "root", "split_salt"):
            if not str(getattr(self, name)):
                raise ValueError(f"dataset manifest {name} must be non-empty")
        if not self.patient_id_strategy:
            raise ValueError("dataset manifest patient_id_strategy must be non-empty")
        if not (
            Path(self.root).is_absolute()
            or PurePosixPath(self.root).is_absolute()
            or PureWindowsPath(self.root).is_absolute()
        ):
            raise ValueError("dataset manifest root must be absolute")
        record_ids = tuple(str(record.record_id) for record in self.records)
        if not record_ids or any(not value for value in record_ids):
            raise ValueError("dataset manifest must contain non-empty record identifiers")
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("dataset manifest contains duplicate record identifiers")
        if any(not str(record.patient_id) for record in self.records):
            raise ValueError("dataset manifest contains an empty patient identifier")
        for record in self.records:
            for label, raw_path in (
                ("record_id", record.record_id),
                ("relative_header", record.relative_header),
                ("signal_file", record.signal_file),
            ):
                if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
                    raise ValueError(f"dataset manifest {label} must be canonical POSIX relative")
                path = PurePosixPath(raw_path)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"dataset manifest {label} escapes its root")
            if record.relative_header != f"{record.record_id}.hea":
                raise ValueError("dataset manifest record/header identity mismatch")
            if record.signal_file not in {
                f"{record.record_id}.dat",
                f"{record.record_id}.mat",
            }:
                raise ValueError("dataset manifest record/signal identity mismatch")
            for label, digest in (
                ("header", record.header_sha256),
                ("signal", record.signal_sha256),
            ):
                if not isinstance(digest, str) or len(digest) != 64 or any(
                    character not in "0123456789abcdef" for character in digest.lower()
                ):
                    raise ValueError(f"dataset manifest {label} SHA-256 is invalid")
            if (
                isinstance(record.signal_size_bytes, bool)
                or not isinstance(record.signal_size_bytes, int)
                or record.signal_size_bytes <= 0
            ):
                raise ValueError("dataset manifest signal size must be a positive integer")

    @classmethod
    def from_dict(cls, value: Mapping) -> "DatasetManifest":
        """Load a versioned JSON representation and verify its embedded hashes."""

        if not isinstance(value, Mapping):
            raise ValueError("dataset manifest must be an object")
        schema = value.get("schema_version", "dataset-manifest-v3")
        if schema != "dataset-manifest-v3":
            raise ValueError(f"unsupported dataset manifest schema {schema!r}")
        required = {
            "schema_version",
            "cohort",
            "version",
            "source_url",
            "root",
            "records",
            "split_salt",
            "patient_id_strategy",
            "split_algorithm",
            "split_ratios",
            "structure",
            "manifest_sha256",
            "split_sha256",
        }
        missing = required - set(value)
        if missing:
            raise ValueError(f"dataset manifest is missing fields: {sorted(missing)}")
        try:
            records = tuple(ManifestRecord(**record) for record in value["records"])
        except (TypeError, ValueError) as exc:
            raise ValueError("dataset manifest contains an invalid record") from exc
        manifest = cls(
            cohort=str(value["cohort"]),
            version=str(value["version"]),
            source_url=str(value["source_url"]),
            root=str(value["root"]),
            records=records,
            split_salt=str(value["split_salt"]),
            patient_id_strategy=str(value["patient_id_strategy"]),
        )
        from ecgcert.protocol import EXTERNAL_SPLIT_ALGORITHM, EXTERNAL_SPLIT_RATIOS

        if value["split_algorithm"] != EXTERNAL_SPLIT_ALGORITHM:
            raise ValueError("dataset manifest uses an unsupported split algorithm")
        if value["split_ratios"] != {
            "train": EXTERNAL_SPLIT_RATIOS[0],
            "tune": EXTERNAL_SPLIT_RATIOS[1],
            "test": EXTERNAL_SPLIT_RATIOS[2],
        }:
            raise ValueError("dataset manifest uses unsupported split ratios")
        expected_manifest = value["manifest_sha256"]
        if expected_manifest != manifest.sha256():
            raise ValueError("dataset manifest SHA-256 does not match its content")
        expected_split = value["split_sha256"]
        if expected_split != manifest.split().sha256():
            raise ValueError("dataset split SHA-256 does not match its records")
        if value["structure"] != manifest.structure():
            raise ValueError("dataset manifest structure counts do not match its records")
        return manifest

    @classmethod
    def from_path(cls, path: str | Path) -> "DatasetManifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def split(self) -> PatientSplit:
        mapping = {record.record_id: record.patient_id for record in self.records}
        return patient_hash_split(mapping, salt=self.split_salt)

    def structure(self) -> dict[str, object]:
        """Return complete record/patient and frozen split counts."""

        split = self.split()
        record_to_patient = {
            str(record.record_id): str(record.patient_id) for record in self.records
        }
        patient_counts: dict[str, int] = {}
        for patient_id in record_to_patient.values():
            patient_counts[patient_id] = patient_counts.get(patient_id, 0) + 1
        split_counts = {}
        for role in ("train", "tune", "calibration", "test"):
            record_ids = tuple(str(value) for value in getattr(split, role))
            split_counts[role] = {
                "n_records": len(record_ids),
                "n_patients": len({record_to_patient[value] for value in record_ids}),
            }
        counts = tuple(patient_counts.values())
        return {
            "n_records": len(self.records),
            "n_patients": len(patient_counts),
            "records_per_patient_min": min(counts),
            "records_per_patient_max": max(counts),
            "split": split_counts,
        }

    def validate_release_contract(self, cohort: str | None = None) -> None:
        """Reject partial or identity-ambiguous official external mirrors."""

        expected_cohort = self.cohort if cohort is None else cohort
        contract = RELEASE_COHORT_CONTRACTS.get(expected_cohort)
        if contract is None:
            raise ValueError(f"no release cohort contract is registered for {expected_cohort!r}")
        mismatches = []
        for field in ("version", "source_url", "patient_id_strategy"):
            if getattr(self, field) != contract[field]:
                mismatches.append(field)
        if self.cohort != expected_cohort:
            mismatches.append("cohort")
        structure = self.structure()
        if structure["n_records"] != contract["n_records"]:
            mismatches.append("n_records")
        if structure["n_patients"] != contract["n_patient_ids"]:
            mismatches.append("n_patient_ids")
        if structure["records_per_patient_min"] != 1 or structure[
            "records_per_patient_max"
        ] != 1:
            mismatches.append("record_to_patient_cardinality")
        split = self.split()
        if not split.train or not split.tune or not split.test or split.calibration:
            mismatches.append("split_roles")
        if mismatches:
            raise ValueError(
                f"{expected_cohort} release manifest violates its official contract: "
                f"{sorted(set(mismatches))}"
            )

    def to_dict(self) -> dict:
        from ecgcert.protocol import EXTERNAL_SPLIT_ALGORITHM, EXTERNAL_SPLIT_RATIOS

        out = asdict(self)
        out["schema_version"] = "dataset-manifest-v3"
        out["split_algorithm"] = EXTERNAL_SPLIT_ALGORITHM
        out["split_ratios"] = {
            "train": EXTERNAL_SPLIT_RATIOS[0],
            "tune": EXTERNAL_SPLIT_RATIOS[1],
            "test": EXTERNAL_SPLIT_RATIOS[2],
        }
        out["structure"] = self.structure()
        out["manifest_sha256"] = self.sha256()
        out["split_sha256"] = self.split().sha256()
        return out

    def verify_files(self) -> None:
        """Fail closed if a header or signal differs from the frozen manifest."""

        root = Path(self.root)
        for record in self.records:
            header = root / record.relative_header
            if not header.is_file() or _sha256(header) != record.header_sha256:
                raise ValueError(f"header hash mismatch for {record.record_id}")
            if not record.signal_file or record.signal_size_bytes is None or not record.signal_sha256:
                raise ValueError(f"manifest lacks complete signal provenance for {record.record_id}")
            signal = root / record.signal_file
            if not signal.is_file() or signal.stat().st_size != record.signal_size_bytes:
                raise ValueError(f"signal size mismatch for {record.record_id}")
            if _sha256(signal) != record.signal_sha256:
                raise ValueError(f"signal hash mismatch for {record.record_id}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_files(
    paths: Iterable[str | Path], *, workers: int | None = None
) -> dict[Path, str]:
    """Hash a deterministic path set with bounded I/O concurrency.

    ``ThreadPoolExecutor.map`` preserves input order, so concurrency cannot alter
    the resulting manifest. The DAG exports ``ECGCERT_NUM_WORKERS`` from the
    node's declared CPU allocation; direct calls default to at most four workers.
    """

    ordered = tuple(sorted({Path(path).resolve() for path in paths}, key=lambda p: str(p)))
    if workers is None:
        raw = os.environ.get("ECGCERT_NUM_WORKERS", "4")
        try:
            workers = int(raw)
        except ValueError as exc:
            raise ValueError("ECGCERT_NUM_WORKERS must be an integer") from exc
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 10:
        raise ValueError("manifest hash workers must be in [1, 10]")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        digests = tuple(executor.map(_sha256, ordered))
    return dict(zip(ordered, digests, strict=True))


def build_wfdb_manifest(
    *,
    cohort: str,
    version: str,
    source_url: str,
    root: str | Path,
    patient_by_record: Mapping[str, str] | None = None,
    split_salt: str | None = None,
    patient_id_strategy: str | None = None,
    expected_record_count: int | None = None,
    expected_patient_count: int | None = None,
) -> DatasetManifest:
    """Scan a complete local WFDB cohort without loading its signal arrays."""

    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)
    for label, expected in (
        ("record", expected_record_count),
        ("patient", expected_patient_count),
    ):
        if expected is not None and (
            isinstance(expected, bool) or not isinstance(expected, int) or expected < 1
        ):
            raise ValueError(f"expected {label} count must be a positive integer")
    explicit_patient_mapping = patient_by_record is not None
    patient_by_record = {} if patient_by_record is None else patient_by_record
    pending: list[tuple[str, str, Path, Path]] = []
    for header in sorted(root_path.rglob("*.hea")):
        relative = header.relative_to(root_path).as_posix()
        record_id = relative[:-4]
        lines = header.read_text(encoding="utf-8", errors="replace").splitlines()
        first_line = lines[0].split() if lines else []
        if len(first_line) < 4:
            raise ValueError(f"WFDB header has an incomplete record line: {header}")
        signal_name = first_line[0]
        if signal_name != Path(record_id).name:
            raise ValueError(
                f"WFDB header identity {signal_name!r} disagrees with record {record_id!r}"
            )
        try:
            n_signals = int(first_line[1])
            sample_rate_hz = float(first_line[2].split("/", 1)[0])
            n_samples = int(first_line[3])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"WFDB header has invalid dimensions: {header}") from exc
        if (
            n_signals != 12
            or not math.isfinite(sample_rate_hz)
            or sample_rate_hz <= 0
            or n_samples <= 0
        ):
            raise ValueError(f"WFDB header violates the twelve-lead signal contract: {header}")
        candidates = [header.with_suffix(extension) for extension in (".dat", ".mat")]
        present = [candidate for candidate in candidates if candidate.is_file()]
        if len(present) != 1:
            raise ValueError(
                f"WFDB header must have exactly one .dat or .mat signal file: {header}"
            )
        signal = present[0]
        if explicit_patient_mapping and record_id not in patient_by_record:
            raise ValueError(f"patient mapping lacks WFDB record {record_id}")
        patient_id = str(patient_by_record[record_id]) if explicit_patient_mapping else signal_name
        if not patient_id:
            raise ValueError(f"WFDB record {record_id} has no patient identity")
        pending.append((record_id, patient_id, header, signal))
    if not pending:
        raise ValueError(f"no WFDB headers found under {root_path}")
    if explicit_patient_mapping:
        scanned = {record_id for record_id, _patient_id, _header, _signal in pending}
        extra = {str(record_id) for record_id in patient_by_record} - scanned
        if extra:
            raise ValueError(f"patient mapping contains unknown WFDB records: {sorted(extra)[:5]}")
    if expected_record_count is not None and len(pending) != expected_record_count:
        raise ValueError(
            f"{cohort} is incomplete: expected {expected_record_count} records, found {len(pending)}"
        )
    n_patients = len({patient_id for _record_id, patient_id, _header, _signal in pending})
    if expected_patient_count is not None and n_patients != expected_patient_count:
        raise ValueError(
            f"{cohort} patient count mismatch: expected {expected_patient_count}, found {n_patients}"
        )
    hashes = hash_files(
        path for _record_id, _patient_id, header, signal in pending for path in (header, signal)
    )
    records: list[ManifestRecord] = []
    for record_id, patient_id, header, signal in pending:
        records.append(
            ManifestRecord(
                record_id=record_id,
                patient_id=patient_id,
                relative_header=header.relative_to(root_path).as_posix(),
                header_sha256=hashes[header.resolve()],
                signal_file=signal.relative_to(root_path).as_posix(),
                signal_size_bytes=signal.stat().st_size,
                signal_sha256=hashes[signal.resolve()],
            )
        )
    return DatasetManifest(
        cohort=cohort,
        version=version,
        source_url=source_url,
        root=str(root_path),
        records=tuple(records),
        split_salt=split_salt or f"ecgcert-{cohort}-{version}-v1",
        patient_id_strategy=patient_id_strategy
        or ("explicit_mapping" if explicit_patient_mapping else "wfdb_record_name"),
    )
