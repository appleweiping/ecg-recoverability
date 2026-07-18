# ecg-recoverability

Robust, target-specific, **model-conditional** recoverability maps for reduced-lead ECG
reconstruction.

> **Research status — fail closed.** Stage 15 has no reviewed decision. Final empirical values,
> model rankings, external-transfer conclusions, and manuscript effect sizes are **PENDING**. The
> repository currently provides the frozen question, implementation, validation design, and evidence
> gate; it does not yet provide an empirical conclusion.

The project is being rebuilt around one falsifiable question:

> Does a robust target-specific recoverability score add held-out predictive value for per-lead ECG
> reconstruction error beyond simple configuration and signal-scale variables, across reconstruction
> methods and on external cohorts?

The score is evaluated as a property of a fitted spatial model and its assumptions. It is not a
model-free impossibility result, a guarantee for every reconstructor, or a clinical conclusion.

## Scientific scope

### Primary object

For each waveform segment, observed-lead configuration, and target lead, the primary pipeline fits a
bank of empirical spatial models over the preregistered rank grid `{2, 3, 4, 5}`. It evaluates two
basis definitions:

- `independent8_lifted`: fit the eight algebraically independent measured channels and lift them to
  the displayed 12 leads; this is the primary definition;
- `raw12_pca`: fit the displayed 12 channels directly; this is a weighting and acquisition
  sensitivity analysis.

Each fitted model produces target-specific geometry and a Gaussian posterior ambiguity in mV. That
ambiguity is **model-conditional**: it depends on the fitted subspace, coordinate covariance,
observation model, and fold-8 regularization. The primary `ambiguity_robust_mv` score takes a
conservative envelope over ranks and patient-cluster bootstrap uncertainty instead of selecting a
favorable rank after seeing test outcomes.

The primary analysis asks whether that score predicts patient-level held-out reconstruction error
after accounting for:

- number of observed leads;
- observed-configuration rank and global conditioning;
- target RMS;
- maximum observed-to-target correlation;
- reconstruction method, waveform segment, and target lead.

The comparison is the out-of-sample change in predictive fit between a simple meta-model and the same
model augmented with `ambiguity_robust_mv`. Direct reconstruction benchmarking remains necessary;
the map complements it rather than replacing it.

### Validation design

The primary protocol is encoded in `src/ecgcert/protocol.py`:

- primary segments: QRS, ST, and T; P is supplementary;
- primary sampling rate: 500 Hz;
- rank grid: `{2, 3, 4, 5}`;
- primary uncertainty: at least 2,000 patient-cluster bootstrap replicates;
- primary basis: eight independent leads lifted to the displayed 12-lead system;
- sensitivity basis: direct 12-channel PCA;
- observed configurations: an outcome-independent, hashed panel over the independent leads.

PTB-XL uses patient-disjoint official fold roles:

| Role | Fold(s) | Permitted use |
|---|---:|---|
| Train | 1–7 | fit spatial models and reconstructors |
| Tune | 8 | select regularization and other frozen hyperparameters |
| Calibration/meta-fit | 9 | fit the frozen comparison model |
| Test | 10 | one held-out primary evaluation |

Patient identity is the split and resampling unit. Beat- or window-level resampling is not accepted
as primary uncertainty.

External validation is zero-transfer on Chapman–Shaoxing–Ningbo and CPSC 2018. The PTB-XL score
definition, preprocessing contract, comparison variables, and outcome direction remain frozen.
External cohort-specific map refits are sensitivity analyses; they do not substitute for the
zero-transfer score–error test.

### Reconstruction panel

All methods share the same observed masks, patient splits, units, sample rate, and per-patient scorer:

- low-rank Gaussian conditional mean;
- ridge regression;
- mask-conditioned 1-D U-Net;
- official-method adapters for ImputeECG and ECGrecover when their pinned upstream artifacts and
  protocol requirements are satisfied.

Unavailable or failed public methods remain in the run ledger with a reason. They are not silently
replaced by an easier in-house surrogate.

## Stage 15 decision gate

`src/ecgcert/evaluation.py::stage15_decision` implements a nondiscretionary, fail-closed decision.
`PROCEED` requires all of the following:

1. the PTB-XL fold-10 incremental `ΔR²` patient-bootstrap interval has a positive lower bound;
2. at least one external zero-transfer cohort has a positive lower bound for the same comparison;
3. at least three reconstruction methods have a positive method-specific point estimate.

If any condition fails, the coded decision is `PIVOT`, with machine-readable reasons. Every
automatic decision then requires a signed author review within 24 hours. A failed automatic gate
cannot be reviewed as `PROCEED`, and there is no override that turns incomplete, unstable,
single-method, or internal-only evidence into a positive headline. The broader artifact, leakage,
citation, and submission requirements are defined
in [the evidence-gated protocol](docs/research_protocol.md) and
[the stage-gate register](arc_audit/STAGE_GATES.md).

A signed `PROCEED` releases the positive result macros; a signed `PIVOT` releases the actual effects
with a transparent negative-result headline and conclusion. Until one of those reviewed decisions is
bound to the artifacts, the project-level and paper-level status remains `PENDING`.

## What is not claimed

- No single spatial rank is treated as the true cardiac representation.
- No fitted row-space result is promoted to a model-free statement about all waveform information.
- No result is assumed to hold independently of the reconstruction method.
- No reconstruction metric is treated as diagnostic equivalence, clinical utility, or patient
  safety evidence.
- No ST-event, conformal-calibration, active-selection, or generative-model branch supports the
  primary headline.
- No current JSON, plot, checkpoint, or historical manuscript value is a final submission result.

## Repository map

```text
src/ecgcert/
  protocol.py                 frozen ranks, segments, patient splits, configuration panel
  physics/spatial_subspace.py rank-generic empirical spatial models and basis variants
  recoverability/
    gaussian.py               model-conditional posterior ambiguity and conditional mean
    model_bank.py             reusable patient-cluster bootstrap model banks
    rank_path.py              per-rank paths and robust cross-rank envelopes
  estimators/                 shared reconstructor API, baselines, U-Net, official adapters
  evaluation.py               held-out meta-model comparison and Stage 15 decision
  data/                       manifests, cohort adapters, and patient-level audit checks
  execution/                  isolated artifact and execution-envelope primitives

experiments/
  robust_maps_v3.py           primary rank-robust target map
  reconstruction_candidates_v3.py
                               fold-8-only candidate training and checkpoint traces
  tune_reconstructors_v3.py   frozen candidate selection
  reconstruction_benchmark_v3.py
  stage_gates_v3.py           Stage 5, 9, and 20 fail-closed quality gates
  external_validation_v3.py  external zero-transfer analysis
  meta_analysis_v3.py         incremental comparison and Stage 15 artifact

scripts/experiment_manifest.yaml  primary, extended, and legacy DAG profiles
scripts/dag_runner.py              profile-aware DAG validation and execution
scripts/prepare_official_baselines_v3.py
                                   audited official-method data preparation
scripts/claim_sync_v3.py           reviewed Stage-15 values to paper/registry binding
paper/main_v2.tex                  evidence-gated short-paper source
paper/arxiv_long.tex               extended protocol plus quarantined legacy supplement
```

## Setup and checks

Windows PowerShell:

```powershell
uv venv --python 3.11.2 .venv
uv pip install --python .venv/Scripts/python.exe --require-hashes -r environments/cpu.lock.txt
uv pip install --python .venv/Scripts/python.exe -e . --no-deps

# Unit and static checks; these do not create scientific evidence.
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe paper/check_submission_claims.py

# Validate the primary DAG without running experiments.
.venv/Scripts/python.exe scripts/dag_runner.py --profile icassp --validate-only
```

The validation-only command should list the primary lineage from dataset manifests through robust
maps, reconstruction benchmarks, external validation, meta-analysis, `stage15_gate`, claim sync, and
submission compilation. Real execution requires the public datasets, pinned upstream methods, and
the declared CPU/GPU environment. A successful command is not sufficient by itself: claim-bearing
artifacts must also satisfy the lineage and human-gate contract.

GPU workers install the self-contained Linux/CUDA 12.8 lock into a fresh Python 3.11.2
environment. The runner records that lock's SHA-256 together with the actual driver, CUDA runtime,
GPU UUID, and memory in every result envelope:

```bash
uv venv --python 3.11.2 .venv
uv pip install --python .venv/bin/python --require-hashes -r environments/gpu.lock.txt
uv pip install --python .venv/bin/python -e . --no-deps
```

Four additional DAG nodes wait for hash-authenticated official AutoResearchClaw v0.5.0 co-pilot
receipt bundles at Stages 5, 9, 15, and 20. A configured checkout, console log, failed probe, or
project-authored review cannot satisfy those nodes. The official receipt and the separately signed
author decision are both required.

The review nodes do not accept a JSON field that merely says `signed`. They verify an Ed25519
signature against [the repository-pinned reviewer public key](security/reviewer_ed25519.pub). The
private key must remain outside the repository, and approval signs the exact gate and evidence
hashes. See [the security gate](security/README.md) for the record/review command.

## Current execution blockers

The implementation is deliberately ahead of the claim-bearing run. As of 2026-07-19, the following
conditions still fail closed:

- the exposed server password must be rotated; the project SSH public key must be installed and a
  separately verified host key written to a fixed `known_hosts` file before any new remote command;
- the exact-commit ECGrecover checkout is incomplete because its upstream GitLab blobs were
  unavailable, so the reviewed `config/ecgrecover.integration.v3.json` is intentionally absent;
- the three public datasets and five-method checkpoints have not yet produced a clean primary
  artifact tree, and Stage 15 therefore has no decision to review;
- a real bounded AutoResearchClaw v0.5.0 co-pilot probe reached Stage 01 but failed with
  `Queue owner disconnected before prompt completion`, producing no ARC artifact. This is recorded
  in [the ARC status](arc_audit/ARC_STATUS.md). Consequently all four official ARC receipt nodes
  remain blocked, and the paper does not claim that ARC reviewed it.

The 15 pre-refactor local result files were copied without altering their originals and recorded by
SHA-256 in [the legacy archive inventory](security/legacy_artifact_archive.v1.json). The previously
observed remote checkpoint is hash-recorded there but has not been copied because password fallback
and automatic host-key acceptance are prohibited.

## Legacy policy

The repository retains earlier fixed-subspace theory, calibration, ST-event, active-selection, and
generative-model branches for provenance and negative-result transparency. They are isolated in the
`legacy` DAG profile and the labeled legacy supplement.

Legacy artifacts are not primary evidence, are not inputs to the Stage 15 decision, and **do not
block the primary submission** merely because a legacy-only experiment or maintenance test is
unavailable. A legacy issue becomes blocking only if it contaminates primary artifacts, breaks the
declared primary build, violates licensing or security, or makes a claim-bearing result
unreproducible.

This separation prevents two opposite errors: reviving an unsupported historical claim, and allowing
non-claim-bearing historical code to hold the refactored submission hostage.

## Citation and license

Preprint in preparation for a conference-quality submission. Final empirical values are **PENDING**
the Stage 15 decision. Code is released under the MIT License; each dataset and upstream model keeps
its own license and citation requirements.
