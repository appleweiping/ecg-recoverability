# Evidence-gated research protocol

Status: **control-plane draft; no Stage 15 evidence freeze has occurred**.

This document is the shared scientific contract for the refactored project and the additional
guidance file for AutoResearchClaw stages 5, 9, 15, 20, and 23. It does not certify that the
AutoResearchClaw pipeline has run. AutoResearchClaw is pinned to v0.5.0 and its ACP backend is pinned
to `acpx` 0.12.0.  A successful ARC run manifest is still required before any ARC-generated-review
claim is permitted. Stages 5, 9, 15, and 20 each require a separately hash-authenticated official
co-pilot receipt plus the project's Ed25519-signed author decision.

## 1. Single research question

Do robust target-specific recoverability scores estimated only from training-set ECG spatial
geometry predict held-out, per-lead reduced-lead reconstruction error beyond simple configuration
and signal-scale baselines, across reconstruction families and both Chapman and CPSC external
cohorts?

The only intended positive headline is:

> Across prespecified plausible spatial-subspace fits, a continuous target-specific recoverability
> score predicts held-out per-lead reconstruction error beyond lead count, configuration rank,
> target scale, pairwise correlation, and global conditioning; the direction is not confined to one
> reconstructor and has zero-transfer support on an external cohort.

Until Stage 15 records `PROCEED`, the statement above is a hypothesis, not a result. The manuscript
must use “map”, “score”, or “diagnostic”. It must not use finite-sample guarantee language, clinical
safety language, “any SNR”, or a fixed-rank algebraic result as the headline.

## 2. Scope and non-claims

In scope:

- target lead × observed configuration × waveform segment recoverability scores;
- robustness to candidate rank, independent-vs-displayed-lead weighting, and patient resampling;
- held-out prediction of patient-level reconstruction error;
- transparent and learned reconstruction families;
- external validation of the score–error relationship.

Out of scope for the primary paper:

- a fixed rank-3 physical-dipole claim;
- minimax floors or impossibility certificates;
- conformalized-quantile-regression coverage as a primary contribution;
- ST-threshold clinical-safety conclusions;
- active lead/electrode selection;
- diffusion fabrication or hallucination claims;
- diagnostic equivalence, clinical utility, or deployment safety.

Historical work on those topics may remain only in a clearly labeled legacy supplement and cannot
be used in the primary claim–evidence matrix.

## 3. Frozen data and split roles

Primary cohort: the complete PTB-XL diagnostic population using its official patient-level folds.

- folds 1–7: fit spatial representations and reconstructors;
- fold 8: choose the Gaussian observation variance and reconstructor/meta-model hyperparameters on
  prespecified grids; it never chooses one rank from the primary rank set;
- fold 9: fit the two frozen error meta-models;
- fold 10: open exactly once for the frozen primary analysis.

External cohorts: complete Chapman–Shaoxing–Ningbo and CPSC 2018 cohorts.  Each patient is assigned
to 60/20/20 train/validation/test by the frozen patient hash.  The primary external result applies
the PTB-XL score, checkpoints, and meta-model directly to the external 20% test set without
fine-tuning.  The 60% split is used only for the secondary cohort-specific map and cross-cohort rank
stability.  Any unavoidable cohort-specific preprocessing must be declared before outcomes are
inspected and reported as a deviation.

The patient is the split and resampling unit. No beat, time window, or duplicate record from one
patient may cross roles. The eight algebraically independent measured channels define the primary
spatial fit. A 12-displayed-lead weighting is a sensitivity analysis, not silent duplication of the
limb-lead information.

The primary analysis is 500 Hz and uses QRS, ST, and T segments.  P-wave, 100 Hz, alternative
delineator, and direct raw-12 PCA analyses are supplement-only sensitivities.  NORM and the major
diagnostic superclasses are multilabel-stratified sensitivities.  Every exclusion, channel reorder,
unit conversion, and segmentation failure is retained in a patient-level audit record.

## 4. Frozen score family

For segment `s`, observed set `S`, target lead `l`, and rank `r`, folds 1--7 fit the independent-eight
spatial representation `[I, II, V1--V6]`, then lift it to displayed twelve-lead coordinates through
the fixed algebraic transform.  With a Gaussian latent prior and fold-8-frozen observation variance,
the target ambiguity is the posterior standard deviation

`A_r(S,l,s) = sqrt(m_l^T Sigma_post(S,s,r) m_l)` in mV.

The preregistered rank set is exactly `{2,3,4,5}`; fold 8 never selects a primary rank.  Each rank
uses 2,000 patient bootstrap resamples, with configurations and targets retained as correlated cells
inside the patient resample.  The primary envelope is

`A_robust = max_r Q97.5_bootstrap(A_r)`.

The lower recoverability diagnostic `R_lower = 1 - max_r Q97.5(eta_normalized)`, `log10(kappa)`, and
the across-rank span are decomposition diagnostics, not guarantees.  The primary structural map
traverses all 255 nonempty subsets of the independent eight leads.  Learned reconstructions use the
frozen 64-configuration SHA-256 panel with salt `ecgcert-icassp27-v1`.  Direct raw-12 PCA is a
sensitivity only.  Units, signs, regularization grid, segment windows, failure rules, and aggregation
are frozen at Stage 9.

## 5. Reconstructors and comparison variables

Required reconstruction families are low-rank Gaussian conditional mean, fold-8-tuned ridge, the
existing arbitrary-mask 1-D U-Net, the official-code ImputeECG adapter, and the official-code
ECGrecover adapter.  Learned methods use five frozen seeds.  The first four share the frozen
64-configuration panel; ECGrecover is evaluated separately on its public single-input-lead task and
is not silently converted into an arbitrary-mask architecture.  Public upstream commits,
checkpoints, preprocessing, licenses, masks, and input/output compatibility must be recorded.  If a
public method cannot be run, the reason and attempted pinned version stay in the evidence ledger; it
cannot be replaced by a vague “representative” claim.

All methods receive the same observed samples, masks, target definitions, split roles, and scoring
code. Required simple comparison variables are lead count, configuration rank, target RMS, maximum
pairwise observed–target correlation, and global conditioning.

## 6. Endpoints and statistical analysis

Primary outcome: patient-level `log(RMSE)` in mV on missing target samples within segment.

Secondary outcomes include normalized RMSE and waveform correlation, reported separately and never
substituted for the primary outcome after inspection.

Primary analysis unit: `(observed configuration, segment, target lead, reconstructor)` cell, with
patient-level observations retained for uncertainty estimation.

Fold 9 fits two frozen meta-models.  The simple model contains method, segment, target lead, observed
lead count, global configuration rank/condition, training-only target RMS, and training-only maximum
target--observed correlation.  The augmented model adds `A_robust`.  Leave-one-configuration-out
prediction is evaluated on fold 10.  The primary effect is
`Delta R^2 = R^2_augmented - R^2_simple`, with a 95% patient-cluster bootstrap interval that nests
neural-seed resampling.  The same frozen PTB-XL meta-model is tested without refitting on both
external test sets.

Stage 15 `PROCEED` requires the fold-10 interval lower bound to exceed zero; at least one external
zero-transfer interval lower bound to exceed zero; and at least three of the four common-panel
reconstructors (low-rank, ridge, masked U-Net, ImputeECG) to have positive point estimates.  Failure
cannot trigger rank, score, or hyperparameter retuning: the paper must transparently PIVOT to a
negative benchmark result or the submission must be abandoned.  Effect sizes and uncertainty take
precedence over thresholded significance; subspace-angle similarity alone is not validation.  The
primary bootstrap count is fixed at 2,000.

## 7. Artifact and lineage contract

Every claim-bearing result must trace to an immutable run ID containing:

- clean source commit and a recorded clean-worktree assertion;
- dataset version, file hashes, patient IDs hash, and fold-role hash;
- exact Python 3.11.2 CPU/GPU lock-file hash and actual environment hash;
- immutable source-tree hash checked against the selected commit before and after every node;
- fully resolved configuration and command;
- CPU/GPU hardware and software inventory;
- preprocessing and score-code hashes;
- random seed and, for learned models, checkpoint hash;
- stdout/stderr, exit status, runtime, and failure reason;
- raw per-patient predictions sufficient to regenerate aggregates;
- leakage-test report and observed-mask integrity test;
- generated tables/figures linked to the run IDs.

No credential may appear in a config, log, manuscript, or run manifest.  Because the original remote
password was exposed, it must be rotated and cannot be used by this workflow.  Remote execution is
key-only with an explicitly supplied private key and pinned `known_hosts`; automatic host-key
acceptance and password fallback fail closed.

The repository-external execution control root contains an atomic run lease and a hash-chained
cross-run budget ledger. Normal reservations may consume at most 400 of the 500 GPU-hours, leaving
100 GPU-hours for audited reruns/final reproduction; cumulative CPU and artifact limits are 4,000
core-hours and 100 GiB. A crashed or unreconciled lease blocks new work rather than being stolen.

## 8. Human gates

### Stage 5 — literature screen

Approval requires:

- primary-source verification of all claim-bearing citations;
- a citation status ledger with `verified_primary`, `verified_secondary`, or `verified_pending`;
- explicit differentiation from reconstruction models and prior reduced-lead benchmarks;
- no invented venue, year, DOI, author, dataset, or performance metadata;
- verified and cited coverage of the 2026 full-configuration benchmark, ImputeECG, and ECGrecover;
- the benchmark scope is stated precisely: LR, ridge, and LightCNN enumerate 4,094 subsets, while
  its Transformer evaluation uses 32 representative configurations.

### Stage 9 — experiment design

Approval freezes:

- the single claim and non-claims;
- split roles and patient grouping;
- score formula, fixed rank set, basis variants, tolerances, and segment rules;
- reconstructors, public-baseline acceptance rules, seeds, and checkpoints policy;
- endpoints, simple covariates, nested models, uncertainty method, multiplicity handling;
- external-cohort harmonization rules;
- `PROCEED` / `REFINE` / `PIVOT` decision criteria;
- compute budget and the remote-execution security plan.

Opening fold 10 before approval invalidates the primary analysis and requires a new untouched holdout
or an explicit exploratory relabeling.

### Stage 15 — research decision (hard evidence gate)

`PROCEED` requires all of the following:

- real, non-simulated execution completed successfully;
- immutable lineage contract satisfied for every claim-bearing value;
- final fold opened only after the Stage 9 freeze;
- leakage and mask-integrity checks pass;
- patient-level uncertainty and seed variability are reported;
- at least one strong public baseline is successfully and fairly evaluated;
- incremental comparison against all frozen simple covariates is complete;
- robustness is not driven by one rank, weighting, segment, or reconstructor;
- external score–error validation is complete;
- negative, failed, and null runs remain available;
- claim–evidence matrix has no unresolved cell.

`REFINE` is required when evidence is incomplete but the same frozen question remains viable.
`PIVOT` is required when the central association, incremental value, cross-model direction, or
external direction is not supported. Neither decision permits post-hoc replacement of fold 10.

Only a recorded `PROCEED` permits numerical replacement of the `PENDING--STAGE 15` manuscript
macros.

### Stage 20 — submission quality gate

Approval requires:

- all pending macros resolved from immutable Stage 15 artifacts;
- the paper-domain claim check passes;
- manuscript and supplement compile cleanly with readable, non-clipped figures and tables;
- abstract, title, conclusion, and limitations match the claim–evidence matrix;
- all claim-bearing citations verified and the pending 2026 benchmark either resolved or omitted;
- AI-tool provenance ledger reconciled with venue policy;
- ethics, funding, conflicts, licenses, and data statements reconfirmed;
- strict release-clean check passes and no secret or private server detail is present;
- independent adversarial review findings are resolved or explicitly accepted.

## 9. Manuscript result placeholders

Before Stage 15 `PROCEED`, these macros must expand visibly to `PENDING--STAGE 15`:

- `\ResultPrimaryAssociation`
- `\ResultIncrementalValue`
- `\ResultRankWeightStability`
- `\ResultExternalAssociation`
- `\ResultModelCoverage`
- `\ResultBootstrapUncertainty`

They may only be replaced by an automated evidence-freeze step that records the source run IDs. No
final empirical number is hand-typed into the primary manuscript.
