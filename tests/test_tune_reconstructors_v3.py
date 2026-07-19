import hashlib

import numpy as np
import pandas as pd
import pytest

from experiments.tune_reconstructors_v3 import (
    CANDIDATE_SCHEMA_VERSION,
    TUNING_SEEDS,
    candidate_grid,
    select_tuning_configuration,
)


MANIFEST_SHA = "a" * 64
SPLIT_SHA = "b" * 64


def _official_config():
    return {
        "schema_version": "official-reconstruction-config-v3",
        "methods": {
            "imputeecg": {
                "selection": "official_fixed",
                "training_parameters": {"epochs": 100, "batch_size": 128},
            },
            "ecgrecover": {
                "selection": "official_fixed",
                "training_parameters": {"published_task": "single-input"},
            },
        },
    }


def _row(method, candidate_id, patient, *, seed, epoch, score):
    rmse = float(np.exp(score + 0.01 * patient))
    checkpoint = hashlib.sha256(f"{candidate_id}|{seed}".encode()).hexdigest()
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "cohort": "PTB-XL",
        "train_partition": "folds1-7/train",
        "partition": "fold8/tune",
        "manifest_sha256": MANIFEST_SHA,
        "split_sha256": SPLIT_SHA,
        "method": method,
        "candidate_id": candidate_id,
        "patient_id": f"patient-{patient}",
        "segment": "QRS",
        "configuration": "I",
        "target": "V2",
        "model_seed": seed,
        "epoch": epoch,
        "rmse_mv": rmse,
        "log_rmse_mv": float(np.log(rmse)),
        "checkpoint_path": f"checkpoints/{candidate_id}-seed-{seed}.ckpt",
        "checkpoint_sha256": checkpoint,
        "observed_integrity": True,
    }


def _candidate_metrics():
    grid = candidate_grid()
    preferred_lowrank = next(
        candidate
        for candidate in grid["lowrank"]
        if candidate.parameters == {"rank": 3, "noise_variance": 1e-6}
    )
    preferred_ridge = next(
        candidate
        for candidate in grid["ridge"]
        if candidate.parameters == {"ridge_lambda": 1e-2}
    )
    preferred_unet = grid["masked-unet"][1]
    rows = []
    for method in ("lowrank", "ridge"):
        preferred = preferred_lowrank if method == "lowrank" else preferred_ridge
        for candidate in grid[method]:
            penalty = 0.0 if candidate == preferred else 0.5
            for patient in (1, 2):
                rows.append(
                    _row(
                        method,
                        candidate.candidate_id,
                        patient,
                        seed=0,
                        epoch=0,
                        score=-2.0 + penalty,
                    )
                )
    for candidate in grid["masked-unet"]:
        candidate_penalty = 0.0 if candidate == preferred_unet else 0.4
        for seed in TUNING_SEEDS:
            for epoch in range(1, 13):
                epoch_penalty = 0.02 * (epoch - 4) ** 2
                for patient in (1, 2):
                    rows.append(
                        _row(
                            "masked-unet",
                            candidate.candidate_id,
                            patient,
                            seed=seed,
                            epoch=epoch,
                            score=-1.5 + candidate_penalty + epoch_penalty,
                        )
                    )
    return pd.DataFrame(rows), preferred_lowrank, preferred_ridge, preferred_unet


def test_fold8_selector_uses_complete_grid_patient_balance_and_early_stopping():
    frame, preferred_lowrank, preferred_ridge, preferred_unet = _candidate_metrics()
    selected = select_tuning_configuration(
        frame,
        manifest_sha256=MANIFEST_SHA,
        split_sha256=SPLIT_SHA,
        official_config=_official_config(),
    )
    assert selected["source_role"] == "fold8/tune"
    assert selected["methods"]["lowrank"] == preferred_lowrank.parameters
    assert selected["methods"]["ridge"] == preferred_ridge.parameters
    assert selected["selection_audit"]["masked-unet"]["candidate_id"] == (
        preferred_unet.candidate_id
    )
    assert selected["methods"]["masked-unet"]["epochs"] == 4
    assert selected["selection_audit"]["masked-unet"]["stopped_epoch"] == 12
    assert selected["methods"]["imputeecg"] == {"epochs": 100, "batch_size": 128}
    assert selected["selection_audit"]["ecgrecover"]["selection"].startswith(
        "official_fixed"
    )


def test_selector_rejects_holdout_rows_missing_grid_and_official_outcome_tuning():
    frame, *_ = _candidate_metrics()
    holdout = frame.copy()
    holdout.loc[holdout.index[0], "partition"] = "fold10/test"
    with pytest.raises(ValueError, match="partition"):
        select_tuning_configuration(
            holdout,
            manifest_sha256=MANIFEST_SHA,
            split_sha256=SPLIT_SHA,
            official_config=_official_config(),
        )

    incomplete = frame.iloc[1:].copy()
    with pytest.raises(ValueError, match="identical fold-8 cells"):
        select_tuning_configuration(
            incomplete,
            manifest_sha256=MANIFEST_SHA,
            split_sha256=SPLIT_SHA,
            official_config=_official_config(),
        )

    official_row = frame.iloc[[0]].copy()
    official_row["method"] = "imputeecg"
    with pytest.raises(ValueError, match="only lowrank, ridge, and masked-unet"):
        select_tuning_configuration(
            pd.concat([frame, official_row], ignore_index=True),
            manifest_sha256=MANIFEST_SHA,
            split_sha256=SPLIT_SHA,
            official_config=_official_config(),
        )


def test_selector_rejects_unpinned_official_configuration():
    frame, *_ = _candidate_metrics()
    official = _official_config()
    official["methods"]["imputeecg"]["selection"] = "fold8_best"
    with pytest.raises(ValueError, match="official_fixed"):
        select_tuning_configuration(
            frame,
            manifest_sha256=MANIFEST_SHA,
            split_sha256=SPLIT_SHA,
            official_config=official,
        )
