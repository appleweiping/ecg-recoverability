import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ecgcert import lineage
from ecgcert.benchmarking import RELEASE_NEURAL_SEEDS
from ecgcert.protocol import PatientSplit
from experiments import diagnostic_subgroup_v3 as subgroup
from experiments import meta_analysis_v3 as meta
from scripts import prepare_data_manifests as manifest_builder


def _prediction_moments(
    patients: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sufficient_rows = []
    paired_rows = []
    for method_index, method in enumerate(meta.COMMON_PANEL_METHODS):
        seeds = (
            tuple(RELEASE_NEURAL_SEEDS)
            if method in meta.NEURAL_METHODS
            else (0,)
        )
        for patient_index, patient_id in enumerate(patients):
            base = np.asarray(
                [
                    -1.2 + 0.17 * patient_index,
                    -0.35 + 0.09 * method_index,
                    0.55 + 0.11 * patient_index + 0.03 * method_index,
                ],
                dtype=float,
            )
            truths = np.stack(
                [base + 0.002 * seed * np.asarray((1.0, -0.5, 0.75)) for seed in seeds]
            )
            point_truth = truths.mean(axis=0)
            simple_prediction = point_truth + np.asarray((0.34, -0.23, 0.27))
            augmented_prediction = point_truth + np.asarray((0.09, -0.06, 0.07))

            def sufficient_row(truth: np.ndarray, *, seed: int, estimand: str) -> dict:
                return {
                    "schema_version": meta.META_SUFFICIENT_SCHEMA_VERSION,
                    "cohort": "PTB-XL",
                    "patient_id": patient_id,
                    "method": method,
                    "model_seed": seed,
                    "estimand": estimand,
                    "row_count": len(truth),
                    "truth_sum": float(truth.sum()),
                    "truth_square_sum": float(truth @ truth),
                    "simple_square_error": float(
                        np.square(truth - simple_prediction).sum()
                    ),
                    "augmented_square_error": float(
                        np.square(truth - augmented_prediction).sum()
                    ),
                }

            sufficient_rows.append(
                sufficient_row(
                    point_truth,
                    seed=meta.POINT_SEED_SENTINEL,
                    estimand="point_seed_mean",
                )
            )
            sufficient_rows.extend(
                sufficient_row(truth, seed=seed, estimand="seed_specific")
                for seed, truth in zip(seeds, truths, strict=True)
            )
            paired_rows.append(
                {
                    "schema_version": (
                        meta.META_PAIRED_SEED_SUFFICIENT_SCHEMA_VERSION
                    ),
                    "cohort": "PTB-XL",
                    "patient_id": patient_id,
                    "method": method,
                    "model_seeds_json": json.dumps(list(seeds), separators=(",", ":")),
                    "row_count": len(point_truth),
                    "truth_sums_json": json.dumps(
                        truths.sum(axis=1).tolist(), separators=(",", ":")
                    ),
                    "truth_crossproducts_json": json.dumps(
                        (truths @ truths.T).reshape(-1).tolist(),
                        separators=(",", ":"),
                    ),
                    "simple_truth_products_json": json.dumps(
                        (truths @ simple_prediction).tolist(), separators=(",", ":")
                    ),
                    "augmented_truth_products_json": json.dumps(
                        (truths @ augmented_prediction).tolist(),
                        separators=(",", ":"),
                    ),
                    "simple_prediction_square_sum": float(
                        simple_prediction @ simple_prediction
                    ),
                    "augmented_prediction_square_sum": float(
                        augmented_prediction @ augmented_prediction
                    ),
                }
            )
    sufficient = pd.DataFrame(sufficient_rows, columns=meta.SUFFICIENT_COLUMNS)
    paired = pd.DataFrame(paired_rows, columns=meta.PAIRED_SEED_SUFFICIENT_COLUMNS)
    meta._validate_sufficient_contract(sufficient, cohort="PTB-XL")
    meta._validate_paired_sufficient_contract(paired, cohort="PTB-XL")
    return sufficient, paired


def _manifest(tmp_path: Path) -> tuple[Path, dict]:
    definitions = (
        ("train", 1, "train-patient", ["NORM"]),
        ("tune", 8, "tune-patient", ["MI"]),
        ("calibration", 9, "cal-patient", ["STTC"]),
        ("test-0", 10, "p0", ["MI", "NORM"]),
        ("test-1", 10, "p1", ["NORM"]),
        ("test-2", 10, "p2", ["MI"]),
        ("test-3", 10, "p3", ["STTC"]),
        ("test-4", 10, "p4", ["CD"]),
    )
    records = [
        {
            "record_id": record_id,
            "patient_id": patient_id,
            "strat_fold": fold,
            "diagnostic_superclasses": labels,
            "files": {},
        }
        for record_id, fold, patient_id, labels in definitions
    ]
    split = {
        "train": ["train"],
        "tune": ["tune"],
        "calibration": ["calibration"],
        "test": ["test-0", "test-1", "test-2", "test-3", "test-4"],
    }
    split_sha256 = PatientSplit(
        train=("train",),
        tune=("tune",),
        calibration=("calibration",),
        test=("test-0", "test-1", "test-2", "test-3", "test-4"),
    ).sha256()
    record_counts = {
        name: sum(name in record["diagnostic_superclasses"] for record in records)
        for name in subgroup.SUPERCLASSES
    }
    payload = {
        "schema_version": "ptbxl-manifest-v3",
        "cohort": "PTB-XL",
        "root": str(tmp_path.resolve()),
        "records": records,
        "split": split,
        "split_sha256": split_sha256,
        "structure": {
            "diagnostic_superclass_record_counts_multilabel": record_counts,
            "n_records_with_multiple_diagnostic_superclasses": 1,
            "n_records_without_diagnostic_superclass": 0,
        },
    }
    payload["manifest_sha256"] = lineage.canonical_sha256(payload)
    path = tmp_path / "ptbxl.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def _meta_bundle(
    tmp_path: Path,
    manifest_path: Path,
    manifest: dict,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    root = tmp_path / "meta"
    root.mkdir()
    sufficient, paired = _prediction_moments(("p0", "p1", "p2", "p3"))
    paths = {
        "ptbxl_predictions": root / "ptbxl_predictions.parquet",
        "ptbxl_seed_predictions": root / "ptbxl_seed_predictions.parquet",
        "ptbxl_sufficient_stats": root / "ptbxl_sufficient_stats.parquet",
        "ptbxl_paired_seed_sufficient": (
            root / "ptbxl_paired_seed_sufficient.parquet"
        ),
    }
    pd.DataFrame({"authenticated_fixture": [True]}).to_parquet(
        paths["ptbxl_predictions"], index=False
    )
    pd.DataFrame({"authenticated_fixture": [True]}).to_parquet(
        paths["ptbxl_seed_predictions"], index=False
    )
    sufficient.to_parquet(paths["ptbxl_sufficient_stats"], index=False)
    paired.to_parquet(paths["ptbxl_paired_seed_sufficient"], index=False)
    summary = {
        "schema_version": meta.SCHEMA_VERSION,
        "status": "complete",
        "release_contract_verified": False,
        "common_panel_methods": list(meta.COMMON_PANEL_METHODS),
        "seed_prediction_schema_version": meta.META_SEED_PREDICTION_SCHEMA_VERSION,
        "sufficient_stat_schema_version": meta.META_SUFFICIENT_SCHEMA_VERSION,
        "paired_seed_sufficient_schema_version": (
            meta.META_PAIRED_SEED_SUFFICIENT_SCHEMA_VERSION
        ),
        "exact_model_seed_contract": {
            method: list(meta._expected_common_seeds(method))
            for method in meta.COMMON_PANEL_METHODS
        },
        "ptbxl": {"point": 0.5, "ci95": [0.1, 0.8], "replicates": 100, "seed": 1},
        "release_lineage": {
            "source_manifest_sha256": manifest["manifest_sha256"],
            "source_manifest_artifact_sha256": lineage.artifact_sha256(
                manifest_path
            ),
        },
        "artifacts": {
            key: {"path": path.name, "sha256": lineage.artifact_sha256(path)}
            for key, path in paths.items()
        },
    }
    (root / "summary.v3.json").write_text(json.dumps(summary), encoding="utf-8")
    return root, sufficient, paired


def _arguments(meta_root: Path, manifest_path: Path, output: Path) -> SimpleNamespace:
    return SimpleNamespace(
        meta_analysis=meta_root,
        ptbxl_manifest=manifest_path,
        output_dir=output,
        bootstrap_replicates=100,
        seed=meta.META_BOOTSTRAP_SEED,
        release=False,
    )


def test_subgroup_analysis_reuses_frozen_paired_estimator_and_reports_every_class(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    meta_root, sufficient, paired = _meta_bundle(tmp_path, manifest_path, manifest)
    output = tmp_path / "subgroups"

    subgroup.analyze(_arguments(meta_root, manifest_path, output))

    summary = json.loads((output / "summary.v3.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == subgroup.SCHEMA_VERSION
    assert summary["analysis_role"] == "extended_only"
    assert summary["stage15_eligible"] is False
    assert not any(summary["model_or_score_refit"].values())
    assert [row["superclass"] for row in summary["subgroups"]] == list(
        subgroup.SUPERCLASSES
    )
    by_class = {row["superclass"]: row for row in summary["subgroups"]}
    assert by_class["NORM"]["status"] == "estimated"
    assert by_class["MI"]["status"] == "estimated"
    assert by_class["STTC"]["status"] == "not_estimable"
    assert by_class["CD"]["status"] == "not_estimable"
    assert by_class["HYP"]["status"] == "not_estimable"
    assert by_class["CD"]["n_patients"] == 0
    assert by_class["HYP"]["not_estimable_reason_code"] == (
        "fewer_than_two_analyzable_patients"
    )

    norm_patients = {"p0", "p1"}
    expected, _draws = meta._bootstrap_effect_and_draws_from_sufficient(
        sufficient[sufficient["patient_id"].isin(norm_patients)].reset_index(drop=True),
        paired_sufficient=paired[
            paired["patient_id"].isin(norm_patients)
        ].reset_index(drop=True),
        cohort="PTB-XL",
        replicates=100,
        seed=meta.META_BOOTSTRAP_SEED + subgroup.BOOTSTRAP_SEED_OFFSET,
    )
    actual = by_class["NORM"]["effect"]
    assert actual["point"] == pytest.approx(expected.point, abs=1e-14)
    assert actual["ci95"] == pytest.approx(expected.ci95, abs=1e-14)

    status = pd.read_parquet(output / "subgroup_status.parquet")
    effects = pd.read_parquet(output / "subgroup_effects.parquet")
    draws = pd.read_parquet(output / "bootstrap_draws.parquet")
    membership = pd.read_parquet(output / "patient_membership.parquet")
    assert set(status["superclass"]) == set(subgroup.SUPERCLASSES)
    assert set(effects["superclass"]) == {"NORM", "MI"}
    assert len(draws) == 200
    assert set(draws["schema_version"]) == {subgroup.DRAW_SCHEMA_VERSION}
    p0 = membership[membership["patient_id"] == "p0"].iloc[0]
    assert json.loads(p0["diagnostic_superclasses_json"]) == ["MI", "NORM"]
    p4 = membership[membership["patient_id"] == "p4"].iloc[0]
    assert bool(p4["included_in_frozen_meta_evidence"]) is False
    assert summary["lineage"]["upstream_sha256"]["ptbxl_manifest"] == (
        lineage.artifact_sha256(manifest_path)
    )


def test_manifest_labels_are_hash_bound_and_structure_audited(tmp_path: Path) -> None:
    manifest_path, payload = _manifest(tmp_path)
    subgroup._manifest_patient_membership(manifest_path, release=False)

    payload["records"][3]["diagnostic_superclasses"] = ["NORM"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA-256"):
        subgroup._manifest_patient_membership(manifest_path, release=False)

    payload["manifest_sha256"] = lineage.canonical_sha256(
        {key: value for key, value in payload.items() if key != "manifest_sha256"}
    )
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="structure counts"):
        subgroup._manifest_patient_membership(manifest_path, release=False)


def test_meta_artifact_hash_tamper_fails_closed(tmp_path: Path) -> None:
    manifest_path, manifest = _manifest(tmp_path)
    meta_root, _sufficient, _paired = _meta_bundle(
        tmp_path, manifest_path, manifest
    )
    pd.DataFrame({"tampered": [True]}).to_parquet(
        meta_root / "ptbxl_sufficient_stats.parquet", index=False
    )
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        subgroup.analyze(_arguments(meta_root, manifest_path, tmp_path / "out"))


def test_ptbxl_manifest_builder_embeds_multilabel_membership_in_content_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rows = []
    for fold in range(1, 11):
        low = Path("records100") / f"record-{fold}"
        high = Path("records500") / f"record-{fold}"
        for stem in (low, high):
            (tmp_path / stem).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / stem.with_suffix(".hea")).write_text(
                f"record-{fold} fixture\n", encoding="utf-8"
            )
            (tmp_path / stem.with_suffix(".dat")).write_bytes(bytes((fold,)))
        rows.append(
            {
                "patient_id": f"patient-{fold}",
                "strat_fold": fold,
                "filename_lr": low.as_posix(),
                "filename_hr": high.as_posix(),
                "superclass": ["MI", "NORM"] if fold == 10 else ["NORM"],
            }
        )
    metadata = pd.DataFrame(rows, index=pd.Index(range(1, 11), name="ecg_id"))
    (tmp_path / "ptbxl_database.csv").write_text("fixture\n", encoding="utf-8")
    (tmp_path / "scp_statements.csv").write_text("fixture\n", encoding="utf-8")

    class FakePTBXL:
        def __init__(self, _root):
            self.meta = metadata

        def patient_id(self, ecg_id: int) -> str:
            return str(self.meta.loc[ecg_id, "patient_id"])

    monkeypatch.setattr(manifest_builder, "PTBXL", FakePTBXL)
    monkeypatch.setattr(
        manifest_builder,
        "PTBXL_RELEASE_CONTRACT",
        {"n_records": 10, "n_patients": 10},
    )
    payload = manifest_builder._ptbxl_manifest(tmp_path, "fixture", "fixture-url")

    test_record = next(
        record for record in payload["records"] if record["record_id"] == "10"
    )
    assert test_record["diagnostic_superclasses"] == ["MI", "NORM"]
    assert payload["structure"]["diagnostic_superclass_record_counts_multilabel"] == {
        "NORM": 10,
        "MI": 1,
        "STTC": 0,
        "CD": 0,
        "HYP": 0,
    }
    unhashed = dict(payload)
    unhashed.pop("manifest_sha256")
    assert payload["manifest_sha256"] == lineage.canonical_sha256(unhashed)
