"""Create immutable external-cohort manifests and 60/20/20 patient splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ecgcert import lineage
from ecgcert.data.manifest import (
    PTBXL_RELEASE_CONTRACT,
    RELEASE_COHORT_CONTRACTS,
    build_wfdb_manifest,
    hash_files,
)
from ecgcert.data.ptbxl import PTBXL, SUPERCLASSES
from ecgcert.protocol import ptbxl_split


COHORTS = {
    "ptbxl": (
        "1.0.3",
        "https://physionet.org/content/ptb-xl/1.0.3/",
    ),
    "chapman": (
        "1.0.0",
        "https://physionet.org/content/ecg-arrhythmia/1.0.0/",
    ),
    "cpsc2018": (
        "challenge-2020/1.0.2",
        "https://physionet.org/content/challenge-2020/1.0.2/training/cpsc_2018/",
    ),
}

# Official version-specific completeness contracts.  A release manifest is not
# emitted for a partial mirror, including the historic 350-record Chapman cache.
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ptbxl_manifest(root: Path, version: str, source_url: str) -> dict:
    """Create the fold-aware PTB-XL manifest used by every primary node."""

    root = root.resolve()
    db = PTBXL(root)
    expected = PTBXL_RELEASE_CONTRACT
    if len(db.meta) != expected["n_records"]:
        raise ValueError(
            f"PTB-XL is incomplete: expected {expected['n_records']} records, "
            f"found {len(db.meta)}"
        )
    patient_ids = db.meta["patient_id"].astype(str)
    if patient_ids.nunique() != expected["n_patients"]:
        raise ValueError(
            f"PTB-XL patient count mismatch: expected {expected['n_patients']}, "
            f"found {patient_ids.nunique()}"
        )
    pending = []
    for ecg_id, row in db.meta.sort_index().iterrows():
        paths = {}
        for rate, column in ((100, "filename_lr"), (500, "filename_hr")):
            stem = root / str(row[column])
            header = stem.with_suffix(".hea")
            signal = stem.with_suffix(".dat")
            if not header.is_file() or not signal.is_file():
                raise FileNotFoundError(f"incomplete PTB-XL record {ecg_id} at {stem}")
            paths[str(rate)] = (stem, header, signal)
        pending.append((ecg_id, row, paths))
    hashes = hash_files(
        path
        for _ecg_id, _row, paths in pending
        for _rate, (_stem, header, signal) in paths.items()
        for path in (header, signal)
    )
    records = []
    for ecg_id, row, paths in pending:
        files = {
            rate: {
                "record": stem.relative_to(root).as_posix(),
                "header_sha256": hashes[header.resolve()],
                "signal_sha256": hashes[signal.resolve()],
                "signal_size_bytes": signal.stat().st_size,
            }
            for rate, (stem, header, signal) in paths.items()
        }
        records.append(
            {
                "record_id": str(ecg_id),
                "patient_id": db.patient_id(int(ecg_id)),
                "strat_fold": int(row["strat_fold"]),
                # Preserve the full multilabel diagnostic membership in the
                # authenticated manifest.  Supplementary subgroup analyses can
                # then consume frozen fold-10 evidence without reopening or
                # reparsing mutable metadata files.
                "diagnostic_superclasses": sorted(
                    str(label) for label in row["superclass"]
                ),
                "files": files,
            }
        )
    split = ptbxl_split(db)
    split_roles = {
        "train": list(split.train),
        "tune": list(split.tune),
        "calibration": list(split.calibration),
        "test": list(split.test),
    }

    def role_counts(record_ids) -> dict[str, int]:
        selected = db.meta.loc[list(record_ids)]
        return {
            "n_records": len(selected),
            "n_patients": int(selected["patient_id"].astype(str).nunique()),
        }

    superclass_record_counts = {
        superclass: int(
            db.meta["superclass"].apply(lambda labels: superclass in labels).sum()
        )
        for superclass in SUPERCLASSES
    }
    fold_counts = {
        str(fold): {
            "n_records": int((db.meta["strat_fold"] == fold).sum()),
            "n_patients": int(
                db.meta.loc[db.meta["strat_fold"] == fold, "patient_id"]
                .astype(str)
                .nunique()
            ),
        }
        for fold in range(1, 11)
    }
    payload = {
        "schema_version": "ptbxl-manifest-v3",
        "cohort": "PTB-XL",
        "version": version,
        "source_url": source_url,
        "root": str(root),
        "metadata_sha256": _sha256(root / "ptbxl_database.csv"),
        "scp_statements_sha256": _sha256(root / "scp_statements.csv"),
        "records": records,
        "population": "all_records_no_diagnosis_filter",
        "split_algorithm": "official-strat-folds-1-7_8_9_10-v1",
        "structure": {
            "n_records": len(records),
            "n_patients": int(patient_ids.nunique()),
            "folds": fold_counts,
            "split": {
                role: role_counts(record_ids)
                for role, record_ids in split_roles.items()
            },
            "diagnostic_superclass_record_counts_multilabel": superclass_record_counts,
            "n_records_with_multiple_diagnostic_superclasses": int(
                db.meta["superclass"].apply(lambda labels: len(labels) > 1).sum()
            ),
            "n_records_without_diagnostic_superclass": int(
                db.meta["superclass"].apply(lambda labels: len(labels) == 0).sum()
            ),
        },
        "split": split_roles,
        "split_sha256": split.sha256(),
    }
    payload["manifest_sha256"] = lineage.canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    version, source_url = COHORTS[args.cohort]
    if args.cohort == "ptbxl":
        payload = _ptbxl_manifest(Path(args.root), version, source_url)
    else:
        expected = RELEASE_COHORT_CONTRACTS[args.cohort]
        manifest = build_wfdb_manifest(
            cohort=args.cohort,
            version=version,
            source_url=source_url,
            root=args.root,
            split_salt=f"ecgcert-{args.cohort}-{version}-v1",
            patient_id_strategy=str(expected["patient_id_strategy"]),
            expected_record_count=expected["n_records"],
            expected_patient_count=expected["n_patient_ids"],
        )
        manifest.validate_release_contract(args.cohort)
        payload = manifest.to_dict()
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(f"[{args.cohort}] {len(payload['records'])} records -> {destination}")
    print(f"manifest_sha256={payload['manifest_sha256']}")
    print(f"split_sha256={payload['split_sha256']}")
    print(f"split_algorithm={payload['split_algorithm']}")
    print(f"structure={json.dumps(payload['structure'], sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
