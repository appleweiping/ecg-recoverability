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
- primary uncertainty: exactly 2,000 patient-cluster bootstrap replicates;
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
as primary uncertainty. The public CPSC2018 WFDB release has no verifiable cross-record patient key,
so its unique record name is disclosed as a pseudopatient identifier; no stronger CPSC
patient-disjointness claim is made.

External validation is zero-transfer on Chapman–Shaoxing–Ningbo and CPSC 2018. The PTB-XL score
definition, preprocessing contract, comparison variables, and outcome direction remain frozen.
External cohort-specific map refits are sensitivity analyses; they do not substitute for the
zero-transfer score–error test. Their primary cross-cohort ranking comparison uses `A_robust`
(lower ambiguity is more recoverable); `R_lower` is retained only as a secondary diagnostic and
cannot be promoted to the headline map-stability result.

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
2. Chapman zero transfer has a positive lower bound for the same comparison; CPSC2018 is still
   reported in full but cannot trigger the hard gate because its public release has no cross-record
   patient identity key;
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
environment. On the experiment server, the exact interpreter is first materialized in the
persistent tool volume and its explicit Conda package list and binary hash are retained; the
repository virtual environment is then created from that interpreter with pinned uv 0.11.29. The
runner records the scientific lock's SHA-256 together with the actual interpreter, driver, CUDA
runtime, GPU UUID, and memory in every result envelope:

```bash
PY311=/root/autodl-fs/ecg-tools/python/conda-cpython-3.11.2/bin/python
UV=/root/autodl-fs/ecg-tools/bootstrap/uv-0.11.29/uv
"$UV" venv --python "$PY311" .venv
"$UV" pip install --python .venv/bin/python \
  --require-hashes --index-strategy unsafe-best-match \
  -r environments/gpu.lock.txt
"$UV" pip install --python .venv/bin/python -e . --no-deps
```

The GPU lock is the sole exception to uv's safer default `first-index` policy because the
PyTorch CUDA index republishes a partial set of ordinary PyPI package names; with the frozen lock,
`first-index` makes the exact `certifi` pin unsatisfiable. A named PyTorch index has the same
first-index failure and does not provide package-level explicit-index routing in a requirements
file. `unsafe-best-match` is therefore used only together with complete exact `==` pins and
`--require-hashes`; an unhashed or floating GPU install is not an admissible environment.

The complete mixed-resource profile is launched by that one interpreter. Node `resource.kind`
controls scheduling and accounting only; it never selects a different Python implicitly. Execution
therefore requires `--environment-lock gpu`, and the runner verifies every applicable pin before it
reserves compute. The server also needs `pdflatex` and `bibtex`; preflight records their resolved
binary hashes and versions so a missing paper toolchain fails before training.

Deploy all three cohorts to the persistent volume with PhysioNet's official public S3 bucket
(preferred) or the resumable HTTPS fallback. The downloader never accepts the historical
350-record Chapman cache as a complete cohort:

```bash
python scripts/download_external_data.py \
  --dataset all \
  --destination /root/autodl-fs/ecg-data \
  --transport auto

python scripts/link_server_data.py \
  --repo "$PWD" \
  --data-root /root/autodl-fs/ecg-data

# Bootstrap the persistent official sources once, before the frozen DAG run.
python scripts/checkout_upstreams.py \
  --model all \
  --destination /root/autodl-fs/ecg-recoverability-tools/upstreams

python scripts/link_server_upstreams.py \
  --repo "$PWD" \
  --tools-root /root/autodl-fs/ecg-recoverability-tools
```

The link command is idempotent and fail closed: it creates only the three ignored dataset links,
never replaces an existing repository path, and rolls back only unchanged links and empty parent
directories created by the failing invocation.

The upstream link is likewise ignored, idempotent, and non-overwriting. It accepts the persistent
source only after both official checkouts match their frozen commit, origin, root tree, and clean
state. Server preflight reads `--tools-root/upstreams`, while the DAG reads the exact same directory
through the absolute repository `upstreams` link. The `public_baseline_checkouts` DAG node then uses
only a local, no-hardlink clone into `artifacts/upstreams`; it neither fetches nor contacts an
upstream host during an evidence run.

After the data, exact-commit upstream checkouts, and locked environment are present—but before
reserving a training budget—run the read-only server preflight from the frozen GPU environment.
Point `--storage-root` and `--tools-root` at the persistent volume and pass explicit dataset roots
when they are outside this repository's ignored `data/` tree:

```bash
# Copy the reviewed 40-hex commit from the local freeze/PR record. Do not derive
# this value from the remote checkout being tested.
: "${REVIEWED_SOURCE_COMMIT:?export the reviewed 40-hex source commit}"
FROZEN_COMMIT="$REVIEWED_SOURCE_COMMIT"
python scripts/server_preflight.py \
  --repo "$PWD" \
  --expected-commit "$FROZEN_COMMIT" \
  --storage-root /root/autodl-fs \
  --tools-root /root/autodl-fs/ecg-recoverability-tools \
  --ptbxl /root/autodl-fs/ecg-data/ptbxl \
  --chapman /root/autodl-fs/ecg-data/chapman \
  --cpsc2018 /root/autodl-fs/ecg-data/cpsc2018 \
  --environment-lock gpu \
  --output /root/autodl-fs/ecg-preflight/server-preflight.v2.json
```

The command always emits the versioned JSON to stdout and optionally writes the same bytes to
`--output`. It returns `0` only for a clean exact commit with the frozen Python and lock hashes, a
working CUDA-enabled PyTorch runtime, sufficient CPU/RAM/GPU-memory/persistent disk, the paper
toolchain, the pinned official-model checkouts, and complete structural counts for PTB-XL (21,799),
Chapman (45,152), and CPSC2018 (6,877). ARC/acpx are verified by the local control bridge rather
than treated as server prerequisites. Missing fields return `2` with `ok=false`. The allowlisted probe does not inspect
environment variables, SSH files, home directories, remotes, or credential stores, and it never
serializes dirty filenames or raw subprocess diagnostics.

The primary DAG then runs `cuda_contract_tests` before any reconstruction training. That node uses
the exact offline checkouts to execute a one-epoch native U-Net smoke, a one-epoch official
ImputeECG train/load/batched-inference smoke, and the official ECGrecover bridge smoke. It requires
real CUDA, verifies observed-lead bitwise preservation, revalidates both pristine upstream trees,
removes its isolated temporary checkpoints, and publishes hash-bound stdout/stderr plus the actual
GPU/CUDA fingerprint. A skip is a hard failure on the server.

Launch the frozen profile with the same absolute remote interpreter; relative `python` values are
rejected. `launch` starts an idempotent supervisor and DAG in separate server-side sessions, then
returns immediately. The SSH channel is therefore not the lifetime owner of a multi-day run:

```bash
python scripts/remote_runner.py \
  --action launch \
  --host connect.weste.seetacloud.com --port 22886 --user root \
  --repo /root/autodl-tmp/ecg-recoverability-icassp27 \
  --run-root /root/autodl-fs/ecg-recoverability-runs \
  --run-id icassp27-frozen-001 --profile icassp \
  --environment-lock gpu \
  --remote-python /root/autodl-tmp/ecg-recoverability-icassp27/.venv/bin/python \
  --known-hosts /absolute/local/known_hosts --key /absolute/local/id_ed25519
```

After any client restart or network disconnect, query the authenticated server-side job record or
attach to bounded log tails without starting a second DAG process:

```bash
python scripts/remote_runner.py --action status \
  --host connect.weste.seetacloud.com --port 22886 --user root \
  --repo /root/autodl-tmp/ecg-recoverability-icassp27 \
  --run-root /root/autodl-fs/ecg-recoverability-runs \
  --run-id icassp27-frozen-001 \
  --remote-python /root/autodl-tmp/ecg-recoverability-icassp27/.venv/bin/python \
  --known-hosts /absolute/local/known_hosts --key /absolute/local/id_ed25519

python scripts/remote_runner.py --action attach --tail-bytes 65536 \
  --host connect.weste.seetacloud.com --port 22886 --user root \
  --repo /root/autodl-tmp/ecg-recoverability-icassp27 \
  --run-root /root/autodl-fs/ecg-recoverability-runs \
  --run-id icassp27-frozen-001 \
  --remote-python /root/autodl-tmp/ecg-recoverability-icassp27/.venv/bin/python \
  --known-hosts /absolute/local/known_hosts --key /absolute/local/id_ed25519
```

If the DAG itself finishes in `failed` state, repeat the launch command with `--resume`. A live
initial/resume attempt is recovered idempotently rather than duplicated. The supervisor records
process boot/start identity, exact job spec, DAG exit code, run-status hash, and separate log files
under the persistent run root.

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

- the project owner accepted the residual password risk; experiment automation nevertheless uses
  only the dedicated project key. Batch-mode key-only login now succeeds with automatic host-key
  acceptance disabled and the historical host key pinned outside the repository;
- ECGrecover is now pinned to official GitLab commit `ed49dddf8e5e599b8af702e871a1f66b1d628518`
  with an audited, hashed single-input bridge. The source has no declared license (`NOASSERTION`),
  is not vendored, and the project owner's reported author permission must be reviewed at Stage 9
  before redistribution;
- the three public datasets and five-method checkpoints have not yet produced a clean primary
  artifact tree, and Stage 15 therefore has no decision to review;
- the earlier AutoResearchClaw v0.5.0 probe that disconnected at Stage 01 remains invalid historical
  evidence. The queue-owner bridge has been repaired with one-time, hash-bound operator responses;
  a fresh single-session formal run must still produce the four official receipts before the paper
  may claim that ARC reviewed it.

The 15 pre-refactor local result files were recorded by SHA-256 in
[the legacy archive inventory](security/legacy_artifact_archive.v1.json), verified 15/15 against the
repository-external copy, and then the tracked workspace paths were restored to the frozen commit.
The previously observed remote checkpoint and all remote legacy result files are also retained in a
read-only, hash-inventoried archive on the persistent server volume.  The old checkout was not reset,
overwritten, or deleted; no archived artifact is an input to the primary DAG.

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
