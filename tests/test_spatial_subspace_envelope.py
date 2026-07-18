"""Record-cluster bootstrap and conservative rank-path envelopes."""

from collections import defaultdict

import numpy as np

from ecgcert.recoverability import (
    DEFAULT_RANK_GRID,
    aggregate_recoverability_envelope,
    bootstrap_rank_path,
)


def _clustered_population(records: int = 8, rows_per_record: int = 6):
    rng = np.random.default_rng(29)
    record_effect = rng.normal(scale=0.8, size=(records, 12))
    rows = []
    record_ids = []
    for record_id in range(records):
        rows.append(
            record_effect[record_id]
            + rng.normal(scale=0.25, size=(rows_per_record, 12))
        )
        record_ids.extend([record_id] * rows_per_record)
    return np.vstack(rows), np.asarray(record_ids)


def _bootstrap(seed: int = 101):
    X, record_ids = _clustered_population()
    return bootstrap_rank_path(
        X,
        record_ids,
        ["I", "II", "V2"],
        ranks=(2, 3),
        basis_variants=("raw12_pca", "independent8_lifted"),
        fit_cohort="development",
        n_boot=7,
        seed=seed,
        observation_variance_mv2=1e-4,
    )


def test_default_rank_grid_is_preregistered_two_through_five():
    assert DEFAULT_RANK_GRID == (2, 3, 4, 5)


def test_patient_cluster_bootstrap_is_reproducible_and_keeps_cluster_draws():
    first = _bootstrap()
    second = _bootstrap()

    assert first.seed == second.seed == 101
    assert first.n_boot == second.n_boot == 7
    assert first.record_ids == second.record_ids == tuple(range(8))
    assert first.patient_ids == second.patient_ids == tuple(range(8))
    assert len(first.point) == 4
    assert len(first.replicates) == 7 * 4

    for left, right in zip(first.point + first.replicates, second.point + second.replicates):
        assert left.model_key == right.model_key
        assert left.fit_ids == right.fit_ids
        assert left.bootstrap_index == right.bootstrap_index
        assert left.effective_rank == right.effective_rank
        assert left.kappa_global == right.kappa_global
        for field_name in ("eta", "eta_normalized", "kappa_per_lead", "ambiguity"):
            np.testing.assert_array_equal(getattr(left, field_name), getattr(right, field_name))

    draws_by_replicate = defaultdict(set)
    for entry in first.replicates:
        assert len(entry.fit_ids) == len(first.record_ids)
        draws_by_replicate[entry.bootstrap_index].add(entry.fit_ids)
    # Every rank/variant fit in one replicate uses the same whole-record draw.
    assert all(len(draws) == 1 for draws in draws_by_replicate.values())
    # Sampling is with replacement; at least one reproducible draw repeats a record.
    assert any(
        len(set(next(iter(draws)))) < len(first.record_ids)
        for draws in draws_by_replicate.values()
    )


def test_envelope_contains_every_full_sample_rank_path_member():
    path = _bootstrap()
    envelope = aggregate_recoverability_envelope(path, confidence=0.90)

    assert envelope.observed_leads == ("I", "II", "V2")
    assert envelope.confidence == 0.90
    assert len(envelope.worst_eta_member) == 12
    assert envelope.kappa_global_upper >= max(entry.kappa_global for entry in path.point)

    for entry in path.point:
        finite_eta = np.isfinite(entry.eta_normalized)
        assert np.all(
            envelope.eta_normalized_lower[finite_eta]
            <= entry.eta_normalized[finite_eta]
        )
        assert np.all(
            entry.eta_normalized[finite_eta]
            <= envelope.eta_normalized_upper[finite_eta]
        )
        assert np.all(envelope.kappa_lower <= entry.kappa_per_lead)
        assert np.all(entry.kappa_per_lead <= envelope.kappa_upper)
        assert np.all(envelope.ambiguity_lower <= entry.ambiguity)
        assert np.all(entry.ambiguity <= envelope.ambiguity_upper)

    np.testing.assert_allclose(
        envelope.recoverability_lower,
        np.clip(1.0 - envelope.eta_normalized_upper, 0.0, 1.0),
    )
    assert np.all(envelope.model_sensitivity_span >= 0.0)
    assert not envelope.eta_normalized_upper.flags.writeable
