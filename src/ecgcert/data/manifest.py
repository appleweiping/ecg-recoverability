"""Versioned manifests for local PTB-XL, Chapman and CPSC WFDB cohorts."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from ecgcert.protocol import PatientSplit, patient_hash_split


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

    @classmethod
    def from_dict(cls, value: Mapping) -> "DatasetManifest":
        """Load a versioned JSON representation and verify its embedded hashes."""

        if not isinstance(value, Mapping):
            raise ValueError("dataset manifest must be an object")
        schema = value.get("schema_version", "dataset-manifest-v3")
        if schema != "dataset-manifest-v3":
            raise ValueError(f"unsupported dataset manifest schema {schema!r}")
        required = {"cohort", "version", "source_url", "root", "records", "split_salt"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"dataset manifest is missing fields: {sorted(missing)}")
        records = tuple(ManifestRecord(**record) for record in value["records"])
        manifest = cls(
            cohort=str(value["cohort"]),
            version=str(value["version"]),
            source_url=str(value["source_url"]),
            root=str(value["root"]),
            records=records,
            split_salt=str(value["split_salt"]),
        )
        expected_manifest = value.get("manifest_sha256")
        if expected_manifest is not None and expected_manifest != manifest.sha256():
            raise ValueError("dataset manifest SHA-256 does not match its content")
        expected_split = value.get("split_sha256")
        if expected_split is not None and expected_split != manifest.split().sha256():
            raise ValueError("dataset split SHA-256 does not match its records")
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

    def to_dict(self) -> dict:
        out = asdict(self)
        out["schema_version"] = "dataset-manifest-v3"
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


def build_wfdb_manifest(
    *,
    cohort: str,
    version: str,
    source_url: str,
    root: str | Path,
    patient_by_record: Mapping[str, str] | None = None,
    split_salt: str | None = None,
) -> DatasetManifest:
    """Scan a complete local WFDB cohort without loading its signal arrays."""

    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(root_path)
    patient_by_record = {} if patient_by_record is None else patient_by_record
    records: list[ManifestRecord] = []
    for header in sorted(root_path.rglob("*.hea")):
        relative = header.relative_to(root_path).as_posix()
        record_id = relative[:-4]
        lines = header.read_text(encoding="utf-8", errors="replace").splitlines()
        first_line = lines[0].split() if lines else []
        signal_name = first_line[0] if first_line else Path(record_id).name
        candidates = [header.with_suffix(extension) for extension in (".dat", ".mat")]
        signal = next((candidate for candidate in candidates if candidate.exists()), None)
        if signal is None:
            raise ValueError(f"WFDB header has no .dat or .mat signal file: {header}")
        records.append(
            ManifestRecord(
                record_id=record_id,
                patient_id=str(patient_by_record.get(record_id, signal_name)),
                relative_header=relative,
                header_sha256=_sha256(header),
                signal_file=signal.relative_to(root_path).as_posix(),
                signal_size_bytes=signal.stat().st_size,
                signal_sha256=_sha256(signal),
            )
        )
    if not records:
        raise ValueError(f"no WFDB headers found under {root_path}")
    return DatasetManifest(
        cohort=cohort,
        version=version,
        source_url=source_url,
        root=str(root_path),
        records=tuple(records),
        split_salt=split_salt or f"ecgcert-{cohort}-{version}-v1",
    )
