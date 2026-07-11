# ecg-recoverability

**Certified Reduced-Lead ECG Reconstruction — a per-feature recoverability certificate with distribution-free calibration.**

Targeting the **IEEE ICASSP** *Biomedical Signal Processing* track. The certificate and
every theory-validation experiment are **CPU-only and fully reproducible** on real public
data (PTB-XL); the single generative-fabrication exhibit trains a conditional diffusion
model and needs a **GPU** (`experiments/gpu_diffusion_clean.py`). Every theorem in the
paper is cross-checked against Monte-Carlo simulation by the test suite before it is
allowed into the manuscript.

> **TL;DR** — Reconstructing missing ECG leads from a reduced set is an ill-posed
> inverse problem, and a **scalar reconstruction error says nothing about *which*
> features a clinician can trust**. We output, **per waveform feature (P/QRS/ST/T) and
> per lead, a certificate** of what is *provably recoverable* (with a closed-form
> conditioning κ that warns, before any reconstruction, which lead configurations can
> support a feature), what is *statistically recoverable* (distribution-free calibrated
> interval), and what is *provably unrecoverable* (flagged, held-out false-flag ≤ α).
> The engine is a fact the ECG has and generic inverse-problem theory does not: **each
> waveform segment is an approximately rank-3 cardiac dipole plus a non-dipolar
> residual.** A real conditional **diffusion model**, scored against a held-out
> recoverability oracle, shows **fabrication is a choice of *objective***, not an accident
> of the network: raising the realism knob injects a *growing recoverability deficit* that
> the certificate localizes and a scalar RMSE cannot see.

---

## Table of contents
1. [The problem in plain language](#1-the-problem-in-plain-language)
2. [The idea: three tiers of recoverability](#2-the-idea-three-tiers-of-recoverability)
3. [The theorem (and what is *not* ours)](#3-the-theorem-and-what-is-not-ours)
4. [The certified estimator](#4-the-certified-estimator)
5. [Results](#5-results)
6. [Repository layout](#6-repository-layout)
7. [Installation and full reproduction](#7-installation-and-full-reproduction)
8. [Data access (important under GFW)](#8-data-access-important-under-gfw)
9. [What each test proves](#9-what-each-test-proves)
10. [Relation to prior work](#10-relation-to-prior-work)
11. [Honest limitations](#11-honest-limitations)
12. [中文速览](#12-中文速览)
13. [Citation & license](#13-citation--license)

---

## 1. The problem in plain language

A 12-lead ECG is expensive to acquire; wearables and monitors often record only one,
three, or six leads. **Lead reconstruction** fills in the rest. Dozens of recent
methods — linear, U-Net, transformer, diffusion — report a low mean-squared error
(MSE) and declare success.

But a low average error certifies **nothing about which features are trustworthy**.
A cardiologist does not read "average millivolts"; they read the P wave, the QRS
complex, the ST segment, the T wave — each on specific leads. If a reconstruction
gets the average right but *invents* an ST elevation on V2, it invents a heart attack.

Generative models are especially dangerous here, and for a principled reason. To look
realistic on an ill-posed problem you must **add content you cannot know** (the
*perception-distortion tradeoff*). On an image that yields a plausible-but-wrong
texture; on an ECG it yields a plausible-but-wrong **diagnosis**.

## 2. The idea: three tiers of recoverability

Write reduced-lead reconstruction as an inverse problem `y_S = A_S · L + noise`, where
`L` is the full 12-lead signal and `A_S` selects the observed leads `S`. Because the
standard 12 leads obey exact algebra (Einthoven / Goldberger: `III = II − I`,
`aVR = −(I+II)/2`, …), the operator `A_S` is **known** — we never estimate it.

For each waveform segment `s ∈ {P, QRS, ST, T}` the instantaneous heart potential is,
to good approximation, a **3-D cardiac dipole** `M_s·d` plus a **non-dipolar residual**
`r_s`. We estimate the per-segment dipolar basis `M_s ∈ ℝ^{12×3}` from a population
(the top-3 left singular vectors of the segment's lead covariance). This splits every
reconstructed feature into three tiers:

| Tier | What it is | Guarantee |
|------|------------|-----------|
| **I — recoverable** | the dipolar component, when `S` spans the dipole | **exact** up to a closed-form noise gain `κ_s(S)` |
| **II — statistical** | population-correlated non-dipolar content | **distribution-free calibrated interval** |
| **III — unrecoverable** | non-dipolar content independent of `S` (local, e.g. a fractionated QRS under one precordial electrode) | any reconstruction of it is **fabrication → flagged** with false-flag rate ≤ α |

The number `κ_s(S) = ‖M_s M_{s,S}⁺‖₂` is the key object. It says **geometry beats lead
count**: three *coplanar* limb leads see only 2 of the 3 dipole directions, and six
frontal-plane limb leads recover the (transverse) dipole far worse than three
well-chosen leads. Measured on PTB-XL QRS:

| configuration | `κ` | note |
|---|---|---|
| Lead-I | rank 1 | 2 of 3 dipole directions are Tier III |
| `{I, II, V2}` (spread) | **3.1** | well-conditioned |
| `{V1, V2, V3}` (adjacent) | 4.9 | spans, but worse |
| `{I, II, III}` (coplanar) | **3.9×10⁵** | looks like 3 leads, is really 2 |
| limb-6 | **6.7×10⁴** | 6 leads, all frontal-plane |

## 3. The theorem (and what is *not* ours)

**Theorem (per-feature dipolar recoverability).** With `M_{s,S}` the observed rows of
`M_s`: if `rank(M_{s,S}) = 3`, the population-dipolar projection of **every** lead is
recovered by `L̂_s = M_s M_{s,S}⁺ y_S` with error `M_s M_{s,S}⁺ n`, so the error is
`≤ κ_s(S)·‖n‖`. If `rank < 3`, the unobserved dipole directions are unrecoverable at
any SNR. The residual splits into a `y_S`-predictable (Tier II) and a `y_S`-independent
(Tier III) part.

This is a **positive** guarantee — *which* named feature, on *which* lead, is
recoverable and with *what* noise gain — that a generic inverse-problem analysis cannot
produce, because it has no `M`. That is what makes the work non-derivative rather than
"apply known UQ to ECG."

**What we do not claim.** The *negative* side — that Tier III content is unrecoverable
and any estimator returns the prior mean with error ≥ `Var(u)` — is the standard
non-identifiability limit, established in general form for inverse problems by
[Iagaru & Gottschling et al., arXiv:2605.13146] and [Kim & Fridovich-Keil,
arXiv:2510.10947]. We **credit and use** it; our contribution is the ECG-physical
instantiation, the positive `κ_s(S)` certificate, and the clinical metering.

## 4. The certified estimator

- **Tier II — calibrated intervals.** Group-conditional (Mondrian) *conformalized
  quantile regression* (CQR): a separate finite-sample conformal correction per group
  `(segment, lead)`, so coverage holds *conditional on the feature and lead*, not just
  marginally.
- **Tier III — the flag.** The hallucination energy `h_{s,ℓ}` is the reconstruction's
  energy in the certified-unrecoverable subspace. We threshold it at the one-sided
  `(1−α)` conformal quantile of `h` over *faithful* reconstructions, so the false-flag
  rate is `≤ α`, finite-sample and distribution-free. Conformal Risk Control sets `α`
  against the clinical loss (a missed STEMI ≫ a false alarm).
- **Device shift.** Under a genuine device shift (Schiller CS-family → AT-family within
  PTB-XL), Tier I exactness and Tier III soundness are distribution-free by
  construction; only Tier II *coverage level* is shift-sensitive, and it is restored by
  weighted conformal + a small-slice recalibration.

## 5. Results

**Synthetic validation** (`results/synthetic_validation.png`) — all three claims hold
exactly: Tier I error grows linearly with noise at slope `κ`; the Bayes-optimal
reconstructor's Tier III error equals the `Var(u)` lower bound while a hallucinator
doubles it; the flag's false-flag rate is 0.099 ≤ α=0.1 with power → 1.0; Mondrian-CQR
attains 0.90 coverage.

**PTB-XL — the certificate localises and explains.** Reconstructing from `{I, II, V2}`
on the held-out fold, a scalar RMSE *ranks* three reconstructors (the generative model
is worst) but does not say **which** feature is wrong or **why**. The certificate does:

| segment | method | RMSE (mV) | `h` (mV) | ρ (oracle) |
|---|---|---|---|---|
| QRS | dipolar | 0.196 | 0.000 | +0.00 |
| QRS | OLS (learned linear) | 0.196 | 0.030 | **+0.13** |
| QRS | generative | 0.228 | 0.097 | **+0.00** |
| ST | dipolar | 0.125 | 0.000 | +0.00 |
| ST | OLS | 0.112 | 0.016 | **+0.28** |
| ST | generative | 0.149 | 0.063 | **−0.00** |
| T | dipolar | 0.140 | 0.000 | +0.00 |
| T | OLS | 0.105 | 0.028 | **+0.44** |
| T | generative | 0.163 | 0.067 | **−0.00** |

`h` is the **deployable** hallucination energy (no ground truth); ρ is an **oracle**
diagnostic (needs the true leads) used to *validate* that flagged energy is fabricated.
The dipolar baseline never fabricates (`h=0`) but recovers only Tier I; OLS's
non-dipolar energy is correlated with truth (it genuinely recovers Tier II); the
generative model's is not. Honest caveat: `h` alone need not separate blur from
fabrication when energies match — the dependable, deployable signal is the
**configuration-level** certificate (κ, tiers) plus the held-out flag.

**Fabrication is the objective, not the network — a real diffusion model, measured
non-circularly** (`results/gpu_diffusion_frontier.png`). A subtlety makes the naive test
circular: on the certified-unrecoverable subspace, correlation-with-truth ρ≈0 for *any*
sampler (even Bayes-optimal), so ρ≈0 there measures the *definition*, not fabrication. We
defeat this with a **held-out recoverability oracle** — a supervised ridge predictor whose
out-of-sample correlation `ρ_oracle` is a *lower bound* on recoverability: `ρ_oracle>0`
*proves* content is recoverable (a linear map attains it). For `S={I,II,V1,V3,V5}→{V2,V4,V6}`
the QRS non-dipolar content is provably recoverable (`ρ_oracle=+0.33`). We then train a real
conditional 1-D **DDPM** (classifier-free guidance + RePaint) on PTB-XL and score it with the
certificate, comparing the diffusion **posterior mean** (variance-free) to a single deployed
draw to avoid a mean-vs-sample confound. At the honest end (guidance `w=1`) the posterior
mean *matches* the oracle (Δρ≈0 — the model is competent); as guidance pushes toward realism,
the mean develops a **growing recoverability deficit** (Δρ: 0 → +0.10 at `w=4`) while staying
faithful on the recoverable dipole subspace (ρ_recoverable≈0.83) and while **RMSE stays flat**
— the deficit is invisible to a scalar and localized only by the certificate. On the limb-6
negative control (QRS, `ρ_oracle≈0`) the deficit never turns positive: no false alarm. (A
complementary `results/neural_perception_distortion.png` shows the same phenomenon in a CNN
swept over an explicit distortion–perception weight.)

**Safety case (honest).** Reconstructing precordial leads from limb-6 (κ=6.7×10⁴,
effective rank 2 → precordial ST is Tier III) is *certified unsafe upfront*. Over 799
records the danger is bidirectional and RMSE-invisible (~0.075 mV ST error all methods):
the generative model **fabricates 263 phantom STEMIs**, OLS **masks 360 real ones**, and
even the dipolar reconstructor fabricates 226 (inside the recoverable subspace, where
`h` is blind — its flag correctly never fires). The null-space flag catches only a
minority of fabrications (18/263, held-out threshold). The
reliable safety signal is the **configuration-level κ warning**, not the per-record flag.

**Device shift.** CS→AT device shift drops Tier II coverage 0.90→0.82; a 300-record
recalibration restores it to 0.93 at modest width increase (0.15→0.24 mV). (Naive
likelihood-ratio weighting over-widens to trivial 1.00 coverage — not a repair.) Only
the Tier II *level* is shift-sensitive; Tier I exactness and Tier III soundness hold.

## 6. Repository layout

```
src/ecgcert/
  physics/dipolar_subspace.py   # lead algebra, per-segment M_s, kappa_s(S) certificate
  certify/tier_decomposition.py # Tier I/II/III projectors, hallucination energy h
  conformal/mondrian_cqr.py     # Mondrian-CQR, conformal risk control, weighted conformal
  estimators/reconstructors.py  # dipolar / Bayes / OLS / generative reconstructors
  models.py                     # fit per-segment (M_s, mu_s, Sigma_r) from a population
  clinical.py                   # ST-deviation measurement + STEMI flip counting
  data/ptbxl.py                 # PTB-XL loader + NeuroKit2 P/QRS/ST/T delineation
estimators/diffusion.py         # conditional 1-D DDPM (CFG + RePaint), GPU
experiments/
  synthetic_dipole_injection.py # M3: validates all theorems (figures)
  ptbxl_reduced_lead.py         # M4: 3 configs, baselines, hallucination quantification
  ptbxl_stemi_safety.py         # M5: fabricated/masked STEMI + abstention (held-out flag)
  cross_device.py               # M5b: CS->AT device-shift coverage + repair
  neural_baselines.py           # M6: real CNN distortion->perception sweep (CPU)
  gpu_fabrication.py            # M7: oracle gate (Band A/B) + diffusion scoring (GPU)
  gpu_diffusion_clean.py        # M8: de-noised CRAFT exhibit -> gpu_diffusion.json (GPU)
  gpu_diffusion_figure.py       # M8: recoverability-deficit frontier figure
scripts/
  download_data.py              # PTB-XL via HuggingFace mirror (GFW-friendly)
  precheck_dipolarity.py        # risk-2 gate: per-segment dipolarity by diagnosis
tests/                          # 19 theorem-vs-simulation checks
paper/                          # ICASSP spconf LaTeX source
```

## 7. Installation and full reproduction

```bash
# 1. environment (Python 3.11)
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -e .

# 2. data (PTB-XL, ~1.8 GB, via HuggingFace mirror — see §8)
.venv/Scripts/python.exe scripts/download_data.py --dataset ptbxl

# 3. the risk-2 gate + all experiments
.venv/Scripts/python.exe scripts/precheck_dipolarity.py
.venv/Scripts/python.exe experiments/synthetic_dipole_injection.py
.venv/Scripts/python.exe experiments/ptbxl_reduced_lead.py
.venv/Scripts/python.exe experiments/ptbxl_stemi_safety.py
.venv/Scripts/python.exe experiments/cross_device.py

# 4. tests (all theorems cross-checked by simulation)
.venv/Scripts/python.exe -m pytest -q
```

The synthetic experiment and calibration run in seconds on a CPU; the PTB-XL
experiments are dominated by NeuroKit2 delineation (a few minutes each).

## 8. Data access (important under GFW)

PhysioNet's HTTPS endpoint is unreachable from some networks (direct connections are
dropped; some proxies reset the TLS handshake to `physionet.org` specifically). The
downloader therefore pulls PTB-XL from the byte-complete HuggingFace mirror
[`longisland3/ptb-xl`](https://huggingface.co/datasets/longisland3/ptb-xl) via
`hf-mirror.com` with `curl` (resumable). Raw data is **not** committed to git; run the
downloader. If your network reaches PhysioNet directly, use `--source physionet`.

## 9. What each test proves

| test | claim |
|---|---|
| `test_physics::test_lead_algebra_rank8_and_relations` | the encoded 12-lead algebra is exact (rank 8, Einthoven/Goldberger) |
| `test_physics::test_dipolar_subspace_matches_dower` | the data-estimated dipolar subspace equals the inverse-Dower column space |
| `test_physics::test_tier1_exact_recovery_noiseless` | Tier I recovers all 12 leads exactly for a dipole-spanning set |
| `test_physics::test_kappa_geometry_not_leadcount` | `κ` / rank distinguish spanning from coplanar/collinear configs |
| `test_physics::test_tier1_noise_amplification_matches_kappa` | reconstruction error obeys `‖·‖ ≤ κ‖n‖` and attains it |
| `test_certify::*` | Tier projectors, `h = 0` for faithful dipolar, `h` grows with injection, supported-reconstruction strips fabrication |
| `test_conformal::*` | marginal + group-conditional coverage, false-flag control, weighted conformal recovers coverage under shift |
| `test_synthetic_theory::*` | the paper's synthetic figures match their theorems (Tier I `κ` law, Tier III `Var(u)` bound, flag, coverage) |

## 10. Relation to prior work

- **Reduced-lead reconstruction** (U-Net/GAN/diffusion, e.g. arXiv:2502.00559,
  arXiv:2401.05388): reports aggregate MSE; qualitatively notes non-dipolar content is
  hard. We turn that into a per-feature *certificate* with calibration and a flag.
- **Hallucination in inverse problems** (arXiv:2605.13146, arXiv:2510.10947): general
  theory of unrecoverability and distribution-free assessment. We **use** it and add
  the ECG-physical positive certificate `κ_s(S)` and clinical metering.
- **Vectorcardiography / inverse-Dower** (Edenbrandt & Pahlm 1988): the classical
  whole-beat dipole map, which is our Tier I *baseline*, not our contribution — ours is
  per-segment, per-configuration, and calibrated.
- **Conformal prediction** (CQR, conformal risk control, weighted conformal): the
  calibration layer; the novelty is the physics that tells conformal *what* to calibrate.

## 11. Honest limitations

- The dipolar approximation is *weakest* in some pathologies; we never assume exact
  rank 3, and the conformal layer makes subspace approximation affect interval *width*,
  not *validity* (decreasing dipolarity correctly *widens intervals and raises flags*).
- Dipolarity is feature-**and-pathology** specific; we do **not** claim a monotone
  "disease lowers dipolarity" (it does not — see `results/precheck_dipolarity.json`).
- PTB-XL has no per-lead ST-elevation millivolt label; the STEMI safety case uses a
  *measured* ST-deviation endpoint (J+60 ms, 0.1 mV), stated as such.
- The deployable hallucination energy `h` detects fabrication **in the null space**; it
  is blind to error **inside** the recoverable subspace (a contaminated dipole estimate),
  which is instead warned by the conditioning κ. The dependable, deployable signal is the
  **configuration-level** certificate (κ + tiers) plus the held-out flag; the
  correlation ρ used in the results table is an **oracle** diagnostic (needs ground
  truth) that only *validates* what `h` measures.
- The sampled "generative" baseline is an idealised perceptual endpoint; the honest
  fabrication evidence is the **real trained-CNN perception-distortion sweep**. The
  certificate is reconstructor-agnostic and wraps any method.
- Theorem 1 is honest about scope: recovery is *exact* only in the purely-dipolar limit;
  otherwise the observed non-dipolar residual contaminates the estimate, amplified by the
  same κ (which is why a large κ is a warning, not just a noise bound).

## 12. 中文速览

减少导联的 ECG 重建是病态逆问题。现有方法只报一个平均误差，而**生成式重建会编造有临床意义的形态**（伪造 ST 抬高等）。本项目按 **P/QRS/ST/T 逐特征、逐导联**给出可恢复性**证书**：Tier I（偶极分量，可证精确，噪声增益 `κ_s(S)` 闭式）、Tier II（人群相关非偶极内容，分布无关校准区间）、Tier III（与观测独立的局部内容，任何"重建"都是幻觉→有保证地标记，假标记率 ≤ α）。理论内核是通用逆问题理论没有的 ECG 物理结构：**每个波形段≈秩3心脏偶极+非偶极残差**。在 PTB-XL 上，RMSE 几乎相同的重建器被证书清晰区分：生成式在"已认证不可恢复子空间"放入大量能量却**与真值零相关**（自信地幻觉）。所有定理都由测试套件的蒙特卡洛仿真交叉验证（19 个测试）。

## 13. Citation & license

Preprint in preparation (target IEEE ICASSP). MIT License (see `LICENSE`). Built with
[NeuroKit2](https://github.com/neuropsychology/NeuroKit) and
[PTB-XL](https://physionet.org/content/ptb-xl/).
