import numpy as np
import pandas as pd
import pytest

from ecgcert.protocol import (
    CONFIG_PANEL_SALT,
    PatientSplit,
    all_independent_configurations,
    configuration_panel_sha256,
    deep_configuration_panel,
    patient_hash_split,
    ptbxl_split,
)


def test_configuration_universe_and_locked_panel_are_deterministic():
    universe = all_independent_configurations()
    panel = deep_configuration_panel()
    assert len(universe) == 255
    assert len(set(universe)) == 255
    assert len(panel) == 64
    assert panel == deep_configuration_panel(CONFIG_PANEL_SALT)
    assert len(configuration_panel_sha256(panel)) == 64
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
