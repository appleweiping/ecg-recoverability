"""Round-trip contracts for replayable robust-map bootstrap evidence."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ecgcert import lineage
from ecgcert.recoverability import (
    BOOTSTRAP_ATTEMPT_SCHEMA_VERSION,
    BOOTSTRAP_MOMENTS_SCHEMA_VERSION,
    BOOTSTRAP_REPLAY_SCHEMA_VERSION,
    bootstrap_attempts_table,
    bootstrap_moments_table,
    bootstrap_spatial_model_bank,
    rebuild_model_bank_from_artifacts,
)
from experiments.robust_maps_v3 import _AtomicParquetWriter, _write_bootstrap_design


RANKS = (2, 3)
VARIANTS = ("raw12_pca", "independent8_lifted")


def _population() -> tuple[np.ndarray, tuple[str, ...]]:
    rng = np.random.default_rng(470)
    rows = []
    patient_ids = []
    for patient_index in range(8):
        patient_rows = 4 + patient_index % 3
        rows.append(
            rng.normal(scale=0.4, size=(patient_rows, 12))
            + rng.normal(scale=0.8, size=12)
        )
        patient_ids.extend([f"patient-{patient_index}"] * patient_rows)
    return np.vstack(rows), tuple(patient_ids)


def _write_bundle(bank, root: Path) -> tuple[Path, Path, dict[str, str]]:
    moments_path = root / "bootstrap_moments.parquet"
    attempts_path = root / "bootstrap_attempts.parquet"
    pq.write_table(
        bootstrap_moments_table(bank, segment="QRS"),
        moments_path,
        compression="zstd",
    )
    pq.write_table(
        bootstrap_attempts_table(bank, segment="QRS"),
        attempts_path,
        compression="zstd",
    )
    hashes = {
        "bootstrap_moments": lineage.artifact_sha256(moments_path),
        "bootstrap_attempts": lineage.artifact_sha256(attempts_path),
    }
    return moments_path, attempts_path, hashes


def _rebuild(bank, moments_path: Path, attempts_path: Path, hashes: dict[str, str]):
    return rebuild_model_bank_from_artifacts(
        moments_path,
        attempts_path,
        artifact_sha256=hashes,
        segment="QRS",
        ranks=bank.ranks,
        basis_variants=bank.basis_variants,
        fit_cohort=bank.fit_cohort,
        seed=bank.seed,
    )


def _assert_models_equal(original, replayed) -> None:
    assert original.rank == replayed.rank
    assert original.basis_variant == replayed.basis_variant
    np.testing.assert_allclose(original.M, replayed.M, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(original.mu, replayed.mu, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        original.covariance,
        replayed.covariance,
        rtol=1e-12,
        atol=1e-12,
    )


def test_patient_moments_and_every_bootstrap_model_round_trip(tmp_path: Path) -> None:
    X, patient_ids = _population()
    bank = bootstrap_spatial_model_bank(
        X,
        patient_ids,
        ranks=RANKS,
        basis_variants=VARIANTS,
        fit_cohort="PTB-XL/folds1-7/QRS",
        n_boot=5,
        seed=81,
    )
    moments_path, attempts_path, hashes = _write_bundle(bank, tmp_path)
    replayed = _rebuild(bank, moments_path, attempts_path, hashes)

    assert replayed.patient_ids == bank.patient_ids
    np.testing.assert_array_equal(replayed.statistics.origin, bank.statistics.origin)
    np.testing.assert_array_equal(replayed.statistics.counts, bank.statistics.counts)
    np.testing.assert_array_equal(replayed.statistics.sums, bank.statistics.sums)
    np.testing.assert_array_equal(
        replayed.statistics.crossproducts,
        bank.statistics.crossproducts,
    )
    np.testing.assert_array_equal(
        replayed.attempt_ledger.multiplicities,
        bank.attempt_ledger.multiplicities,
    )
    np.testing.assert_array_equal(
        replayed.attempt_ledger.accepted,
        bank.attempt_ledger.accepted,
    )
    np.testing.assert_array_equal(
        replayed.attempt_ledger.accepted_bootstrap_index,
        bank.attempt_ledger.accepted_bootstrap_index,
    )
    for original, reconstructed in zip(bank.point_models, replayed.point_models):
        _assert_models_equal(original, reconstructed)
    for original_group, reconstructed_group in zip(
        bank.bootstrap_models,
        replayed.bootstrap_models,
    ):
        for original, reconstructed in zip(original_group, reconstructed_group):
            _assert_models_equal(original, reconstructed)


def test_rank_deficient_rejections_survive_and_replay(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    X = rng.normal(size=(7, 12))
    patient_ids = tuple(f"singleton-{index}" for index in range(7))
    bank = bootstrap_spatial_model_bank(
        X,
        patient_ids,
        ranks=(5,),
        basis_variants=VARIANTS,
        fit_cohort="rank-deficient-fixture",
        n_boot=3,
        seed=2,
    )
    assert bank.rejected_draws > 0
    assert bank.attempt_ledger.n_attempts == bank.n_boot + bank.rejected_draws
    assert np.count_nonzero(~bank.attempt_ledger.accepted) == bank.rejected_draws
    assert bank.attempt_ledger.accepted_bootstrap_index[~bank.attempt_ledger.accepted].tolist() == [
        -1
    ] * bank.rejected_draws

    moments_path, attempts_path, hashes = _write_bundle(bank, tmp_path)
    replayed = _rebuild(bank, moments_path, attempts_path, hashes)
    assert replayed.rejected_draws == bank.rejected_draws
    np.testing.assert_array_equal(
        replayed.attempt_ledger.accepted,
        bank.attempt_ledger.accepted,
    )
    np.testing.assert_array_equal(
        replayed.bootstrap_multiplicities,
        bank.bootstrap_multiplicities,
    )


def test_replay_fails_closed_on_hash_and_semantic_attempt_tampering(tmp_path: Path) -> None:
    X, patient_ids = _population()
    bank = bootstrap_spatial_model_bank(
        X,
        patient_ids,
        ranks=RANKS,
        basis_variants=VARIANTS,
        fit_cohort="tamper-fixture",
        n_boot=4,
        seed=19,
    )
    moments_path, attempts_path, hashes = _write_bundle(bank, tmp_path)

    original_bytes = attempts_path.read_bytes()
    attempts_path.write_bytes(original_bytes + b"tamper")
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        _rebuild(bank, moments_path, attempts_path, hashes)
    attempts_path.write_bytes(original_bytes)

    table = pq.read_table(attempts_path)
    multiplicities = table.column("multiplicities").to_pylist()
    first = list(multiplicities[0])
    source = next(index for index, value in enumerate(first) if value > 0)
    destination = (source + 1) % len(first)
    first[source] -= 1
    first[destination] += 1
    multiplicities[0] = first
    tampered_column = pa.array(
        multiplicities,
        type=table.schema.field("multiplicities").type,
    )
    table = table.set_column(
        table.schema.get_field_index("multiplicities"),
        "multiplicities",
        tampered_column,
    )
    pq.write_table(table, attempts_path, compression="zstd")
    tampered_hashes = {
        **hashes,
        "bootstrap_attempts": lineage.artifact_sha256(attempts_path),
    }
    with pytest.raises(ValueError, match="declared RNG seed"):
        _rebuild(bank, moments_path, attempts_path, tampered_hashes)


def test_replay_rejects_extra_columns_even_after_hash_is_updated(tmp_path: Path) -> None:
    X, patient_ids = _population()
    bank = bootstrap_spatial_model_bank(
        X,
        patient_ids,
        ranks=RANKS,
        basis_variants=VARIANTS,
        fit_cohort="extra-column-fixture",
        n_boot=3,
        seed=27,
    )
    moments_path, attempts_path, hashes = _write_bundle(bank, tmp_path)
    table = pq.read_table(attempts_path).append_column(
        "unrecognized_evidence",
        pa.array(["ignored?"] * bank.attempt_ledger.n_attempts, type=pa.string()),
    )
    pq.write_table(table, attempts_path, compression="zstd")
    updated_hashes = {
        **hashes,
        "bootstrap_attempts": lineage.artifact_sha256(attempts_path),
    }
    with pytest.raises(ValueError, match="schema columns must match exactly"):
        _rebuild(bank, moments_path, attempts_path, updated_hashes)


def test_producer_writes_uint16_compatible_and_complete_artifacts(tmp_path: Path) -> None:
    X, patient_ids = _population()
    bank = bootstrap_spatial_model_bank(
        X,
        patient_ids,
        ranks=RANKS,
        basis_variants=("raw12_pca",),
        fit_cohort="producer-fixture",
        n_boot=3,
        seed=4,
    )
    paths = {
        name: tmp_path / f"{name}.parquet"
        for name in ("patients", "multiplicities", "moments", "attempts")
    }
    writers = {name: _AtomicParquetWriter(path) for name, path in paths.items()}
    _write_bootstrap_design(
        bank=bank,
        segment="QRS",
        patient_writer=writers["patients"],
        multiplicity_writer=writers["multiplicities"],
        moments_writer=writers["moments"],
        attempt_writer=writers["attempts"],
    )
    for writer in writers.values():
        writer.close(publish=True)

    accepted_type = pq.read_schema(paths["multiplicities"]).field(
        "multiplicities"
    ).type.value_type
    attempt_type = pq.read_schema(paths["attempts"]).field(
        "multiplicities"
    ).type.value_type
    assert accepted_type == pa.uint16()
    assert attempt_type == pa.uint16()
    attempts = pq.read_table(paths["attempts"])
    moments = pq.read_table(paths["moments"])
    assert attempts.num_rows == bank.n_boot + bank.rejected_draws
    assert moments.num_rows == bank.statistics.n_patients
    assert set(attempts.column("schema_version").to_pylist()) == {
        BOOTSTRAP_ATTEMPT_SCHEMA_VERSION
    }
    assert set(moments.column("schema_version").to_pylist()) == {
        BOOTSTRAP_MOMENTS_SCHEMA_VERSION
    }
    descriptor = json.loads(attempts.column("descriptor_json")[0].as_py())
    assert descriptor["schema_version"] == BOOTSTRAP_REPLAY_SCHEMA_VERSION
