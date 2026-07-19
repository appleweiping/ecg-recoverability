import numpy as np
import pandas as pd
import pytest

from ecgcert.physics.dipolar_subspace import INDEPENDENT_LEADS
from ecgcert.protocol import (
    BOOTSTRAP_REPLICATES,
    CONFIG_PANEL_SHA256,
    CONFIG_PANEL_SALT,
    PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD,
    SEGMENT_SAMPLING_ALGORITHM,
    SEGMENT_SAMPLING_SEED,
    SENSITIVITY_SEGMENT_SAMPLE_CAP_PER_RECORD,
    PatientSplit,
    StudyProtocol,
    all_independent_configurations,
    configuration_panel_sha256,
    deep_configuration_panel,
    patient_hash_split,
    ptbxl_split,
)


def test_study_protocol_accepts_only_the_preregistered_primary_contract():
    StudyProtocol().validate()
    invalid_overrides = (
        {"rate_hz": 100},
        {"primary_segments": ("QRS", "T")},
        {"supplementary_segments": ()},
        {"rank_grid": (3,)},
        {"bootstrap_replicates": BOOTSTRAP_REPLICATES + 1},
        {"basis_variant": "raw12_pca"},
        {"sensitivity_basis_variant": "independent8_lifted"},
        {"configuration_salt": "post-hoc"},
        {"configuration_panel_sha256": "0" * 64},
        {
            "primary_segment_sample_cap_per_record": (
                PRIMARY_SEGMENT_SAMPLE_CAP_PER_RECORD + 1
            )
        },
        {
            "sensitivity_segment_sample_cap_per_record": (
                SENSITIVITY_SEGMENT_SAMPLE_CAP_PER_RECORD + 1
            )
        },
        {"segment_sampling_seed": SEGMENT_SAMPLING_SEED + 1},
        {"segment_sampling_algorithm": SEGMENT_SAMPLING_ALGORITHM + "-changed"},
        {"external_map_primary_metric": "recoverability_lower"},
        {"external_map_primary_order": "higher_is_more_recoverable"},
        {"external_map_secondary_diagnostic": "ambiguity_robust_mv"},
    )
    for override in invalid_overrides:
        with pytest.raises(ValueError):
            StudyProtocol(**override).validate()


def test_configuration_universe_and_locked_panel_are_deterministic():
    assert INDEPENDENT_LEADS == ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
    universe = all_independent_configurations()
    panel = deep_configuration_panel()
    assert len(universe) == 255
    assert len(set(universe)) == 255
    assert len(panel) == 64
    assert panel == deep_configuration_panel(CONFIG_PANEL_SALT)
    assert configuration_panel_sha256(panel) == CONFIG_PANEL_SHA256
    assert sum(len(config) == 1 for config in panel) == 8
    assert sum(len(config) == 2 for config in panel) == 28
    assert sum(len(config) == 8 for config in panel) == 1


def test_patient_hash_split_keeps_all_records_from_patient_together():
    mapping = {f"r{i}-{j}": f"p{i}" for i in range(100) for j in range(2)}
    split = patient_hash_split(mapping, salt="external-v1")
    split.validate()
    memberships = {}
    for name in ("train", "tune", "test"):
        for record in getattr(split, name):
            memberships.setdefault(mapping[record], set()).add(name)
    assert all(len(parts) == 1 for parts in memberships.values())
    assert set(split.train) | set(split.tune) | set(split.test) == set(mapping)


def test_patient_hash_split_realizes_exact_patient_proportions():
    mapping = {f"r{i}-{j}": f"p{i}" for i in range(100) for j in range(2)}
    split = patient_hash_split(mapping, salt="external-proportions-v1")
    patient_counts = {
        role: len({mapping[record] for record in getattr(split, role)})
        for role in ("train", "tune", "test")
    }
    assert patient_counts == {"train": 60, "tune": 20, "test": 20}
    assert not split.calibration


def test_patient_hash_split_rejects_ambiguous_serialized_record_ids():
    with pytest.raises(ValueError, match="collide"):
        patient_hash_split({1: "p1", "1": "p2"}, salt="external-v1")


def test_patient_split_rejects_leakage():
    with pytest.raises(ValueError, match="leakage"):
        PatientSplit(train=(1,), tune=(2,), calibration=(3,), test=(1,)).validate()


class _FakePTBXL:
    def __init__(self):
        self.meta = pd.DataFrame(
            {
                "strat_fold": np.repeat(np.arange(1, 11), 2),
                "patient_id": np.arange(20),
                "superclass": [["NORM"]] * 20,
            },
            index=np.arange(100, 120),
        )

    def ids_with_superclass(self, superclass, exclusive, folds):
        del superclass, exclusive
        return self.meta[self.meta["strat_fold"].isin(folds)].index.to_numpy()


def test_ptbxl_locked_fold_roles():
    split = ptbxl_split(_FakePTBXL())
    assert len(split.train) == 14
    assert len(split.tune) == len(split.calibration) == len(split.test) == 2
    assert split.sha256() == ptbxl_split(_FakePTBXL()).sha256()


def test_ptbxl_norm_sensitivity_uses_multilabel_membership():
    db = _FakePTBXL()
    split = ptbxl_split(db, population="norm")
    assert set(split.train) == set(db.meta.index[:14])


def test_ptbxl_split_rejects_patient_leakage_even_with_distinct_records():
    db = _FakePTBXL()
    db.meta.loc[100, "patient_id"] = db.meta.loc[119, "patient_id"]
    with pytest.raises(ValueError, match="patient leakage"):
        ptbxl_split(db)
