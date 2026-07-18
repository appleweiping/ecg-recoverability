"""Robust recoverability paths across spatial-model uncertainty."""

from ecgcert.recoverability.rank_path import (
    DEFAULT_RANK_GRID,
    BootstrapRankPath,
    RankPathEntry,
    RecoverabilityEnvelope,
    aggregate_recoverability_envelope,
    bootstrap_rank_path,
    compute_rank_path,
)
from ecgcert.recoverability.model_bank import (
    PatientBootstrapModelBank,
    PatientClusterSufficientStatistics,
    bootstrap_spatial_model_bank,
    cache_patient_cluster_statistics,
    rank_path_from_model_bank,
)
from ecgcert.recoverability.gaussian import (
    REGULARIZATION_GRID_MV2,
    RegularizationSelection,
    gaussian_conditional_mean,
    gaussian_posterior_covariance,
    gaussian_prior_ambiguity_per_lead,
    tune_gaussian_regularization,
)

__all__ = [
    "DEFAULT_RANK_GRID",
    "BootstrapRankPath",
    "RankPathEntry",
    "RecoverabilityEnvelope",
    "aggregate_recoverability_envelope",
    "bootstrap_rank_path",
    "compute_rank_path",
    "PatientBootstrapModelBank",
    "PatientClusterSufficientStatistics",
    "bootstrap_spatial_model_bank",
    "cache_patient_cluster_statistics",
    "rank_path_from_model_bank",
    "REGULARIZATION_GRID_MV2",
    "RegularizationSelection",
    "gaussian_conditional_mean",
    "gaussian_posterior_covariance",
    "gaussian_prior_ambiguity_per_lead",
    "tune_gaussian_regularization",
]
