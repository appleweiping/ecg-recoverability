# ecg-recoverability

**Target-Specific Recoverability Maps for Reduced-Lead ECG Reconstruction** — a per-lead,
per-feature *identifiability + conditioning* certificate, with distribution-free calibrated
intervals for the predictable residual. Targeting **IEEE ICASSP** (Biomedical Signal
Processing). Real public data (PTB-XL); every theorem is cross-checked by simulation.

> **What this is (and an honest note on what it is not).** Reconstructing missing ECG leads
> from a reduced set is ill-posed, and a scalar error says nothing about *which* named
> feature, on *which* lead, a clinician can trust. We give, per waveform segment (P/QRS/ST/T),
> observed lead set `S`, and **target lead `ℓ`**, two closed-form numbers from one truncated
> SVD of the segment's empirical rank-3 spatial basis `M_s`: an **identifiability**
> `η_{s,ℓ}(S)` (zero ⇔ the target's low-rank component is recoverable from `S`) and a
> **conditioning** `κ_{s,ℓ}(S)`. Plus a calibrated interval for the empirically predictable
> residual. This project began with a stronger "certified hallucination / fabrication is the
> objective" claim; **rigorous re-analysis (see [Honest history](#8-honest-history)) showed
> that claim did not survive**, and the paper was rebuilt around the parts that do.

---

## 1. The problem

A 12-lead ECG is expensive; wearables record one, three, or six leads, and *lead
reconstruction* fills in the rest. A low average error certifies nothing about which
morphological features are trustworthy — a reconstruction that gets the mean right but
invents an ST deviation invents a finding. We ask a reconstructor-independent question:
**for a given observed lead set, which target lead's morphology is even identifiable, and
how well-conditioned is its recovery?**

## 2. The certificate (per target lead)

Each segment's instantaneous potential is approximately low rank across leads. We estimate a
per-segment rank-3 spatial basis `M_s` (top-3 singular vectors of the population lead
covariance). For observed leads `S` (sub-matrix `M_{s,S}`), from a **single** truncated SVD
at relative tolerance `ϱ`:

| quantity | meaning |
|---|---|
| `η_{s,ℓ}(S) = ‖eₗᵀ M_s (I − M_{s,S}⁺M_{s,S})‖` | **identifiability**: `η=0` ⇒ lead `ℓ`'s dipolar component is recoverable from `S`; `η>0` ⇒ an unobserved direction changes lead `ℓ` (unrecoverable at any SNR) |
| `κ_{s,ℓ}(S) = ‖eₗᵀ M_s M_{s,S}⁺‖` | **conditioning**: noise / observed-residual gain into the identifiable part of lead `ℓ` |

The global `κ_s(S) = ‖M_s M_{s,S}⁺‖₂` (spectral norm) is a *configuration-level worst case*
that **upper-bounds** every per-lead conditioning — `κ_{s,ℓ}(S) ≤ κ_s(S)` (a row norm never
exceeds the spectral norm) — it is **not** their maximum. It is not a per-lead certificate. All quantities come from one truncated SVD (unified numerics),
so a lead is either observed or not, everywhere.

**On real PTB-XL** (`results/recoverability_map.png`): a single lead leaves every other lead
unidentifiable; a dipole-spanning triplet `{I,II,V2}` or `{I,II,V1,V3,V5}` makes all targets
**exactly** identifiable (η=0). The map is **graded**, not binary. Since absolute η conflates
lead amplitude with observation geometry, we grade the shortfall by **normalized identifiability**
`η̃ = η/‖eₗᵀ Mₛ‖` (the *fraction* of a lead's dipolar content that is unobservable) and by the
**prior-conditional expected ambiguity in mV** — the residual a Bayes reconstructor still
incurs after conditioning the unobserved dipole coordinate on the observed one through the
fitted Gaussian dipole prior (a marginal footprint would ignore that correlation). For limb-6 no
precordial lead is exactly identifiable; both measures fall from strongly unrecoverable anterior
leads (V1–V4) to much-less-ambiguous lateral V5/V6. So the honest a-priori statement is:
**anterior precordial ST (V1–V4) is far from identifiable from limb leads**, while lateral V5/V6
are much less ambiguous — *but none is exactly identifiable*. A physiological "precordial vs.
limb" split cannot make this graded call; the empirical map does. (Exact numbers:
`results/recoverability_maps.json`, `results/st_safety.json`.)

**Truncation-tolerance sensitivity.** Rank/`κ` of well-posed configurations are stable across
`ϱ∈{1e-4,…,1e-1}`; near-rank-deficient sets are `ϱ`-sensitive (synthetic `{V1,V2,V3}`: rank 2
at `ϱ=1e-2` vs rank 3, `κ~200` at `ϱ=1e-4`). We report a single `κ` only for well-conditioned
sets and give the `ϱ`-sweep + bootstrap CIs otherwise — a single `10⁴`/`10⁵` figure for a
degenerate set is not meaningful.

## 3. Calibrated intervals (predictable residual)

For the empirically predictable non-dipolar residual we train a **real quantile regressor**
(gradient-boosted pinball loss) of each target lead's off-dipole residual on the observed
leads, and apply conformalized quantile regression per Mondrian group `(S,s,ℓ)` with **strict
fold discipline** (basis+model on folds 1–7, conformal calibration on fold 9, a single
evaluation on fold 10). We report **within-group marginal coverage under exchangeability**
(not per-example conditional coverage): on PTB-XL, fold-10 coverage clusters near the nominal
0.90 (e.g. `{I,II,V1,V3,V5}` QRS/V2: 0.92, bootstrap CI [0.89,0.95], width 0.165 mV).

## 4. Baselines and safety

**Baselines** (`results/fair_baselines.json`, one identical per-timepoint waveform-RMSE
protocol for all methods — same shared split (train folds 1–7, ridge λ chosen on fold 8, test
on NORM fold-10), same test records, **paired** record-bootstrap CIs on the step-to-step
deltas, and a **representative** arbitrary-mask 1-D U-Net (3 seeds)): on a spanning set the
methods form a ladder prior-mean → dipolar → ridge → U-Net (the neural margin over a simple
ridge is small), so there is recoverable non-dipolar content. On **limb-6 the U-Net still lowers
error substantially** by recovering the predictable structure, but **plateaus well above its
spanning-set error and no reconstructor closes that gap** — a residual **consistent with** the
unobserved dipolar coordinate `η>0` flags (we do not decompose the error into projected
components, so we claim consistency, not equality). Capacity recovers the predictable part, not
the missing coordinate; we scope the identifiability claim to the dipolar component, not a
blanket "not recoverable". (Exact values + paired-Δ CIs: `results/fair_baselines.json`.)

**ST safety across reconstructors** (`results/st_safety.json`): we reconstruct the full
10-second waveform with three real continuous reconstructors (observed leads kept exact; ST
measured at J+60 ms vs. a PR baseline on fiducials located once on the observed Lead II and
shared between truth and reconstruction; event = `|ST| ≥ 0.1 mV`, absolute). The **total**
wrong-event rate (false-positive + false-negative crossings) is **empirically similar across
the evaluated reconstructors**; all three **miss more true crossings than they invent** (FN > FP
— limb→precordial reconstruction smooths ST toward the population), and the FP/FN balance shifts
with the reconstructor (dipolar has the largest false-positive share, OLS the smallest). We
report the whole matrix and — because we do not
derive a minimax/Bayes lower bound — describe the shared total as an *empirically similar total
error rate across these reconstructors*, **not** a certified floor. We report ST-threshold
events, not diagnoses.

## 5. `M_s` is an *empirical* subspace, not the physical dipole

On real PTB-XL the estimated `M_s` has **three graded principal angles** to the classical
inverse-Dower vectorcardiographic space: one **close** (≈2–6°), one **moderate**, and one
**substantially different** (max ≈43–55°, bootstrap CIs excluding small angles). We therefore
call `M_s` an **empirical rank-3 spatial subspace**, not a physical cardiac dipole. (An earlier
synthetic "matches inverse-Dower" check *generated* the data with inverse-Dower — self-
consistency, not evidence on real ECG.)

## 6. Repository layout

```
src/ecgcert/
  physics/dipolar_subspace.py   # unified SVD: observed_dipole, kappa/kappa_per_lead, eta_per_lead
  certify/tier_decomposition.py # off_dipole_projector/energy, per-lead tier_report (eta)
  conformal/mondrian_cqr.py     # CQR, Mondrian (tuple-safe), conformal risk control
  estimators/diffusion.py       # arbitrary-mask conditional DDPM (application study)
  lineage.py                    # result provenance (commit/dataset/id-hash) + assert_consistent
experiments/
  protocol.py                   # shared train/test split + loader (identical across methods)
  recoverability_maps.py        # per-lead eta/eta_tilde/ambiguity + record-level bootstrap CI
  tier2_conformal.py            # real quantile model + fold discipline (1-7 / tune 8 / cal 9 / test 10)
  baselines_physics.py          # baselines + physics-vs-PCA (empirical subspace)
  fair_baselines.py             # like-for-like per-timepoint baselines + PAIRED delta CIs
  st_safety.py                  # continuous ST-threshold-event safety across reconstructors
  neural_baseline.py            # representative neural (arbitrary-mask U-Net) baseline, 3 seeds
  lead_weighting.py             # 8-independent-lead vs 12-lead fit sensitivity
  maps_figure.py                # the recoverability-map figure
tests/                          # 57 checks (16 cited JSONs) incl. per-lead certificate, rcond sensitivity,
                                #   graded/lineage units, paper-number consistency
paper/main_v2.tex               # ICASSP 4-page draft (target-specific recoverability)
paper/arxiv_long.tex            # extended version: full proofs, cross-dataset transfer,
                                #   the abandoned-hypothesis negative result (8 pp)
paper/emit_baseline_table.py    # regenerate all paper numbers from results/*.json
env.lock.txt                    # exact environment used for results/*.json
```

**Two paper versions.** `paper/main_v2.tex` is the four-page IEEE ICASSP submission (only the
rigorously-supported core). `paper/arxiv_long.tex` is the extended preprint: full statement and
proof of the certificate, the cross-dataset transfer study (QRS subspace transfers PTB-XL↔
Chapman; ST/T third direction does not), and a complete account of the abandoned
"fabrication-objective" claim (§8). Both compile clean; every table number is auto-generated
from result JSON (no hand-typed values).

## 7. Reproduce

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/Scripts/python.exe -e ".[dev,torch]"
.venv/Scripts/python.exe -m pytest -q                         # 51 tests (CPU, no data); 57 with ECG_RELEASE=1
.venv/Scripts/python.exe scripts/download_data.py --dataset ptbxl   # PTB-XL via HF mirror
.venv/Scripts/python.exe experiments/recoverability_maps.py   # the map (CPU)
.venv/Scripts/python.exe experiments/tier2_conformal.py       # calibrated intervals (CPU)
.venv/Scripts/python.exe experiments/baselines_physics.py     # baselines + physics (CPU)
.venv/Scripts/python.exe experiments/st_safety.py             # ST safety (CPU)
# GPU (server): experiments/neural_baseline.py, experiments/diffusion.py-based study
```
CI (`.github/workflows/tests.yml`) runs the test suite on every push.

## 8. Honest history

This project first claimed a diffusion model proves "fabrication is a property of the
objective" (a "certified hallucination" flagged with a false-flag guarantee). A rigorous
pre-submission rebuild found three things that did **not** survive scrutiny and removed them:

1. **Configuration leakage** in the generative exhibit (one model, fixed observed set, was
   reused to score a different configuration) inflated the effect. Fixed via an arbitrary-mask
   model with explicit leakage tests.
2. With leakage fixed and **5 training seeds**, the "achievability gap" is monotone but
   **marginal** (+0.058 ± 0.044 at high guidance, ~1.3σ) — the old "4.3σ" was a leakage
   artifact.
3. **Realism did not improve with guidance** (PSD and amplitude distances *worsen*), so the
   causal "realism → fabrication" premise is false; high guidance simply degrades the model
   on every axis.

So the fabrication-objective story was **deleted**, and "physical cardiac dipole" was
**downgraded** to "empirical rank-3 subspace" (§5). What remains is the honest core above.

## 9. Citation & license

Preprint in preparation (target IEEE ICASSP). MIT License. Built with
[NeuroKit2](https://github.com/neuropsychology/NeuroKit) and
[PTB-XL](https://physionet.org/content/ptb-xl/).
