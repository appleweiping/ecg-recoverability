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
- robustness to candidate rank, independent-vs-displayed-lead weighting, the timepoint cap, and
  patient resampling;
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

External cohorts: complete Chapman–Shaoxing–Ningbo (45,152 records) and CPSC 2018 (6,877 records)
cohorts.  Salted SHA-256 orders patient identifiers before an exact largest-remainder 60/20/20
allocation.  Chapman provides one unique record identifier per patient.  The public CPSC WFDB
release provides no verifiable cross-record patient key, so its unique record name is explicitly
treated as a pseudopatient identifier; CPSC patient-disjointness beyond that public identity cannot
be claimed.  The primary external result applies the PTB-XL score, checkpoints, and meta-model
directly to the external 20% test set without fine-tuning.  The 60% split is used only for the
secondary cohort-specific map and cross-cohort rank stability.  Any unavoidable cohort-specific
preprocessing must be declared before outcomes are inspected and reported as a deviation.
For that secondary map comparison, the preregistered primary ranking metric is `A_robust` in mV,
ordered with lower ambiguity as more recoverable.  Spearman agreement is computed across matched
missing-target cells.  `R_lower` is reported separately as a secondary decomposition diagnostic and
is not eligible to replace the `A_robust` ranking in a headline result.

The patient is the split and resampling unit wherever a public patient key exists; the disclosed
CPSC pseudopatient limitation is the sole exception. No beat, time window, or duplicate record from
one known patient may cross roles. The eight algebraically independent measured channels in locked
order `[I, II, V1, V2, V3, V4, V5, V6]` define the primary spatial fit. A 12-displayed-lead weighting
is a sensitivity analysis, not silent duplication of the limb-lead information.

The primary analysis is 500 Hz and uses QRS, ST, and T segments.  P-wave, 100 Hz, alternative
delineator, and direct raw-12 PCA analyses are supplement-only sensitivities.  NORM and the major
diagnostic superclasses are multilabel-stratified sensitivities.  Every exclusion, channel reorder,
unit conversion, and segmentation failure is retained in a patient-level audit record.

One authenticated folds-1--7 inclusion artifact owns reconstruction-training eligibility for all
five methods. A record is included only when strict delineation yields nonempty QRS, ST, and T
windows; no consumer may make a second method-specific exclusion. The artifact binds the ordered
record sequence, the aligned per-record patient sequence (including repeated patient IDs), each
canonical float32 signal hash, the source audit, and the folds-1--7 simple-predictor table. Any later
signal/audit drift or artifact mismatch fails closed before a training array is consumed. Folds
8--10 are invalid inputs to this artifact.

For spatial-map fitting, each included record contributes at most 40 timepoints to each waveform
segment. This outcome-independent cap prevents records with longer delineated intervals from
dominating the spatial covariance and bounds the folds-1--7 working set; it is not an estimate of 40
independent observations, because the patient remains the bootstrap unit. If a record/segment has
more than 40 eligible indices, SHA-256 of the locked algorithm identifier, base seed `20260719`,
role namespace, record identifier, and segment name seeds an independent NumPy PCG64 stream. The
first 40 entries of its full without-replacement permutation are retained and returned in temporal
order. The two role namespaces are `PTB-XL/folds1-7/spatial-map-fit` and
`PTB-XL/fold8/regularization-tuning`; the algorithm identifier is
`sha256-namespace-record-segment-pcg64-permutation-prefix-v1`. An extended-only, preregistered
cap-80 analysis repeats the map while reusing the primary fold-8 regularizer. Its selected indices
contain every cap-40 selected index and add points wherever more than 40 are eligible, isolating the
effect of admitting more timepoints. This sensitivity cannot feed the frozen primary claim or change
the primary cap.

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

ECGrecover is fixed to official commit `ed49dddf8e5e599b8af702e871a1f66b1d628518`. Its U-Net,
MSE-minus-0.1-Pearson loss, and training loop remain upstream code. The project-owned bridge makes
one necessary disclosed adaptation: the upstream record-and-target-specific min-max preprocessing
would expose a missing target's amplitude, so it is replaced by a folds-1--7-only fixed per-lead
scale for truth-free raw-mV scoring. Missing inputs retain the published random-valued masking,
restricted to lead I, and the observed lead is copied exactly at the evaluator boundary. The
frozen execution adapter uses 128 records per bridge process and 64 records per GPU micro-batch;
mask noise remains record-derived so batch membership cannot change its inputs. The
upstream repository contains no license declaration (`NOASSERTION`) and is not redistributed here;
the owner's reported author permission is an ARC Stage 9 review item.

All methods receive the same observed samples, masks, target definitions, split roles, and scoring
code. Required simple comparison variables are lead count, configuration rank, target RMS, maximum
pairwise observed–target correlation, and global conditioning. Target RMS and correlation are
computed once from the authenticated folds-1--7 `training_predictors.parquet`; the four common-panel
methods must have identical artifact hashes and values, and external evaluation reuses that exact
PTB-XL table. Any similarly named rank-map columns are diagnostics only and cannot overwrite the
benchmark predictors used by either meta-model.

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
neural-seed resampling.  Within every replicate and neural method, five seeds are drawn with
replacement from that method's five preregistered fitted runs and their outcomes are averaged again
at the patient/configuration/segment/target cell level; one patient draw is shared across all methods.
This makes the bootstrap estimator match the five-run mean used by the point estimate.  The same
frozen PTB-XL meta-model is tested without refitting on both
external test sets.

The diagnostic supplement contains two scientifically different multilabel sensitivities and keeps
their artifacts separate.  The diagnosis-specific spatial-map nodes refit a structural map within
NORM, MI, STTC, CD, and HYP records; they do not test the frozen prediction claim.  A separate
extended-only fold-10 subgroup node performs no map, reconstructor, hyperparameter, or meta-model
fit.  It unions the authenticated record-level `diagnostic_superclasses` in the PTB-XL manifest at
the patient level, subsets the already-frozen point/seed/paired sufficient evidence, and applies the
same shared-patient plus five-seed-mean bootstrap estimator as the primary analysis.  Every
prespecified superclass is emitted with its point estimate, 95% interval, and patient count, or an
explicit not-estimable reason when the subgroup cannot support the estimator.  These results are
supplementary and cannot alter or satisfy Stage 15.

Stage 15 `PROCEED` requires the fold-10 interval lower bound to exceed zero; the Chapman
zero-transfer interval lower bound to exceed zero; and at least three of the four common-panel
reconstructors (low-rank, ridge, masked U-Net, ImputeECG) to have positive point estimates.  Failure
cannot trigger rank, score, or hyperparameter retuning: the paper must transparently PIVOT to a
negative benchmark result or the submission must be abandoned.  Effect sizes and uncertainty take
precedence over thresholded significance; subspace-angle similarity alone is not validation.  The
primary bootstrap count is fixed at 2,000.  CPSC2018 remains a required, fully reported
zero-transfer sensitivity, but its public WFDB record names are treated as pseudopatients because no
cross-record patient key is available; therefore CPSC2018 cannot by itself satisfy the patient-level
external hard gate.

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

No credential may appear in a config, log, manuscript, or run manifest.  The project owner explicitly
declined rotation of the exposed remote password and accepted the resulting residual risk; this
deviation is recorded in the versioned security status and must be bound by the signed Stage-9
review.  The password is nevertheless forbidden to this workflow.  Remote execution is key-only
with an explicitly supplied private key and a host key matched against the pre-existing historical
`known_hosts` entry; automatic host-key acceptance and password fallback fail closed.

The repository-external execution control root contains an atomic run lease and a hash-chained
cross-run budget ledger. Normal reservations may consume at most 400 of the 500 GPU-hours, leaving
100 GPU-hours for audited reruns/final reproduction; cumulative CPU and artifact limits are 4,000
core-hours and 100 GiB. A crashed or unreconciled lease blocks new work rather than being stolen.

## 8. Human gates

Each of Stages 5, 9, 15, and 20 executes first, emits and validates its declared artifacts, and
then pauses exactly once for named-user review.  There is no pre-stage approval at these gates.
Stage 15 is evidence review only: it consumes the already-generated, immutable Stage 14 and native
DAG evidence and must not launch, rerun, tune, or open any server test itself.
Its automatic decision has exactly three hard gates: positive PTB-XL lower CI, at least one positive
external lower CI, and at least three positive common-panel method point estimates. The prohibition
on post-test retuning is a study policy, not an extra outcome-dependent hard gate.
All four reviews belong to one continuous official ARC process with one native `run_id` and one
HITL `session_id`.  ARC v0.5.0 restart and `--from-stage` entrypoints create new identities, so they
are not valid continuation mechanisms for this evidence run; a process loss invalidates the run.

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
- score formula, fixed rank set, basis variants, tolerances, segment rules, and the keyed cap-40
  timepoint-sampling contract plus cap-80 sensitivity;
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
