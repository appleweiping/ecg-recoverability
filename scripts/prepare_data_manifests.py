"""Create immutable external-cohort manifests and 60/20/20 patient splits."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ecgcert import lineage
from ecgcert.data.manifest import build_wfdb_manifest
from ecgcert.data.ptbxl import PTBXL
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
    records = []
    for ecg_id, row in db.meta.sort_index().iterrows():
        files = {}
        for rate, column in ((100, "filename_lr"), (500, "filename_hr")):
            stem = root / str(row[column])
            header = stem.with_suffix(".hea")
            signal = stem.with_suffix(".dat")
            if not header.is_file() or not signal.is_file():
                raise FileNotFoundError(f"incomplete PTB-XL record {ecg_id} at {stem}")
            files[str(rate)] = {
                "record": stem.relative_to(root).as_posix(),
                "header_sha256": _sha256(header),
                "signal_sha256": _sha256(signal),
                "signal_size_bytes": signal.stat().st_size,
            }
        records.append(
            {
                "record_id": str(ecg_id),
                "patient_id": db.patient_id(int(ecg_id)),
                "strat_fold": int(row["strat_fold"]),
                "files": files,
            }
        )
    split = ptbxl_split(db)
    payload = {
        "schema_version": "ptbxl-manifest-v3",
        "cohort": "PTB-XL",
        "version": version,
        "source_url": source_url,
        "root": str(root),
        "metadata_sha256": _sha256(root / "ptbxl_database.csv"),
        "scp_statements_sha256": _sha256(root / "scp_statements.csv"),
        "records": records,
        "split": {
            "train": list(split.train),
            "tune": list(split.tune),
            "calibration": list(split.calibration),
            "test": list(split.test),
        },
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
        manifest = build_wfdb_manifest(
            cohort=args.cohort,
            version=version,
            source_url=source_url,
            root=args.root,
            split_salt=f"ecgcert-{args.cohort}-{version}-v1",
        )
        payload = manifest.to_dict()
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[{args.cohort}] {len(payload['records'])} records -> {destination}")
    print(f"manifest_sha256={payload['manifest_sha256']}")
    print(f"split_sha256={payload['split_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
