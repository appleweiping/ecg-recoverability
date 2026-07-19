# ARXIV NEW-CONTENT VERIFICATION BUNDLE

## paper/arxiv_long.tex (full)
```tex
\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,bm,amsthm}
\usepackage{cite}
\usepackage[hidelinks]{hyperref}

% result macros (auto-generated from results/*.json by paper/emit_baseline_table.py)
\input{auto/fair_baselines_macros}

\title{\textbf{Target-Specific Recoverability Maps for Reduced-Lead ECG
Reconstruction}\\[2pt]\large A graded per-lead identifiability + conditioning certificate with
calibrated intervals, and an honest account of a retracted fabrication claim}
\author{Weiping Yan\\ University of Minnesota, Twin Cities}
\date{}

\begin{document}
\maketitle

\begin{abstract}
Reconstructing missing electrocardiogram (ECG) leads from a reduced set is ill-posed, and a
scalar reconstruction error says nothing about \emph{which} named feature, on \emph{which}
lead, a clinician can trust. We give a \emph{target-specific recoverability map}: for a
waveform segment $s$ (P/QRS/ST/T), an observed lead set $S$, and a target lead $\ell$, two
closed-form numbers from one truncated SVD of the segment's empirical rank-3 spatial
sub-matrix---a graded \emph{identifiability} $\eta_{s,\ell}(S)$ (zero iff the target's low-rank
component is recoverable from $S$) and a \emph{conditioning} $\kappa_{s,\ell}(S)$. The per-lead
criterion is the classical estimability condition specialized to ECG; the contribution is its
use as a graded, \emph{empirically-estimated} map (the estimated subspace differs
$43$--$55^\circ$ from the textbook inverse-Dower dipole), the distribution-free calibrated
intervals we attach to it, and a reconstructor-\emph{invariant} safety certificate. On PTB-XL
the map ranks target leads on a continuum---from limb leads, anterior precordial ST (V1--V4) is
unidentifiable while lateral V5/V6 are near-identifiable---and the total ST-threshold error
floor is near-invariant across reconstructors ($\TotWrongLo$--$\TotWrongHi\%$) while the
false-positive/false-negative split is a design choice. The certificate transfers across PTB-XL
and Chapman for QRS (aligned subspaces, reproduced $\kappa$ ordering) but not in the ST/T third
direction. Finally, we document in full a claim this project retracted---that a diffusion
reconstructor proves ``fabrication is a property of the objective''---which a leakage-corrected,
multi-seed re-analysis reduced to a marginal ($\approx1.3\sigma$) effect with no supporting
realism trend. This long version accompanies a four-page conference paper and contains the full
proofs, ablations, cross-dataset study, and the negative result.
\end{abstract}

\section{Introduction}
\label{sec:intro}
Wearable and reduced-lead ECG acquisition makes lead \emph{reconstruction} ubiquitous: recover
the standard 12 leads from one, three, or six observed leads
\cite{gradowski2025reveals,cardoso2024bayesian}. A low aggregate error certifies nothing about
\emph{which} morphological features are trustworthy: a cardiologist reads the P wave, the QRS
complex, the ST segment and the T wave, each on specific leads, and a reconstruction that gets
the mean right but invents an ST deviation invents a finding.

We ask a sharper, reconstructor-independent question: \emph{for a given observed lead set,
which target lead's low-rank morphology is even identifiable, and how well-conditioned is its
recovery?} Each waveform segment's instantaneous potential is approximately low rank across
leads; we estimate a per-segment rank-3 spatial basis $\bm M_s$ (top-3 singular vectors of the
population lead covariance). We stress that $\bm M_s$ is a \emph{population-estimated (PCA)}
object: on real PTB-XL it shares two directions with the classical vectorcardiographic
(inverse-Dower) dipole but its third direction differs materially, so we call it an empirical
rank-3 subspace rather than a physical dipole (Sec.~\ref{sec:exp}).

This document is the extended version of a four-page conference submission. It adds: the full
statement and proof of the certificate (Sec.~\ref{sec:cert}, App.~\ref{app:proof}); the
cross-dataset transfer study (Sec.~\ref{sec:cross}); and a complete account of a claim we
retracted during pre-submission review (Sec.~\ref{sec:generative}). We report the retraction in
detail on purpose: hiding the earlier, more exciting numbers would be precisely the selective
reporting our certificate is designed to expose.

\section{Related Work}
\label{sec:related}
\textbf{Lead reconstruction and VCG synthesis.} Fixed linear transforms synthesize the
vectorcardiogram or missing leads from a standard set (inverse-Dower \cite{edenbrandt1988dower},
Kors regression \cite{kors1990vcg}), and reduced-lead reconstruction has a long clinical
literature \cite{nelwan2004reduced}; recent deep models push reconstruction accuracy
\cite{gradowski2025reveals,cardoso2024bayesian}. These target the reconstruction; we target the
\emph{prior} question of what a given configuration can support, independent of the
reconstructor. \textbf{Identifiability and lead selection.} Whether a linear functional is
recoverable from a sub-observation is the classical estimability question
\cite{scheffe1959anova}; choosing which leads to observe is optimal sensor selection
\cite{joshi2009sensor}. Our per-lead $\eta$ is estimability specialized to the ECG dipolar
functional; the novelty is not the identity but the graded, empirically-estimated,
per-target-lead map with calibrated per-record error attached. \textbf{Uncertainty for inverse
problems.} Conformal prediction \cite{romano2019cqr,angelopoulos2023gentle} gives
distribution-free intervals under exchangeability; we apply Mondrian CQR per $(S,s,\ell)$ group.

\section{The Recoverability Certificate}
\label{sec:cert}
\input{theorem_corrected}

\section{Distribution-Free Calibration}
\label{sec:calib}
\input{results_v2_calib}

\section{Experiments}
\label{sec:exp}
\textbf{Data.} PTB-XL \cite{wagner2020ptbxl} (21{,}799 twelve-lead records, 100/500\,Hz,
10-fold split) with P/QRS/ST/T delineation by NeuroKit2 \cite{makowski2021neurokit}; the map is
evaluated on records disjoint from the folds used to estimate $\bm\mu_s,\bm M_s$. All table
numbers regenerate from result JSON via \texttt{paper/emit\_baseline\_table.py}.
\input{results_v2_maps}
\input{results_v2_baseline}

\input{results_long_crossdata}

\input{results_long_generative}

\section{Limitations}
\label{sec:limits}
The subspace $\bm M_s$ is estimated, so $\eta/\kappa$ inherit its estimation error (reported via
record-bootstrap CIs) and its cohort dependence (Sec.~\ref{sec:cross}); the ST/T third
direction is cohort-specific. Identifiability is stated for the low-rank (dipolar) coordinate:
a reconstructor may still recover predictable non-dipolar structure (Sec.~\ref{sec:exp}), so
$\eta{>}0$ bounds the dipolar component, not every waveform detail. Conformal coverage is
within-$(S,s,\ell)$-group marginal under exchangeability, not conditional on diagnosis; rare
diagnostic subgroups are under-covered (Sec.~\ref{sec:calib}). Delineation uses NeuroKit2 and
inherits its errors. The certificate is a pre-reconstruction screen, not a substitute for
clinical validation.

\section{Conclusion}
Reduced-lead ECG reconstruction has a known low-rank structure per segment. Exploiting it gives
a graded, per-target-lead recoverability map---what is identifiable, how well-conditioned, and
(via calibration) what interval the predictable residual admits---before and independent of any
reconstructor, plus a reconstructor-invariant error floor. It converts ``trust the
reconstruction'' into a per-feature, per-lead, calibrated statement, and it told us honestly
when one of our own earlier claims was not supported.

\appendix
\section{Proof of Proposition~\ref{thm:perlead}}
\label{app:proof}
Write $\bm M_{s,S}=\bm{\mathrm{Sel}}_S\bm M_s$ with truncated SVD
$\bm M_{s,S}=\bm U\bm\Sigma\bm V^{\top}$ at tolerance $\varrho$, so
$\bm M_{s,S}^{+}=\bm V\bm\Sigma^{-1}\bm U^{\top}$ and
$\bm P_{\mathrm{obs}}=\bm M_{s,S}^{+}\bm M_{s,S}=\bm V\bm V^{\top}$ is the orthogonal projector
onto $\operatorname{row}(\bm M_{s,S})=\operatorname{span}(\bm V)$.

\textbf{(i) Identifiability.} The dipolar part of lead $\ell$ is the linear functional
$f_\ell(\bm d)=\bm e_\ell^{\top}\bm M_s\bm d$. From $S$ we observe
$\bm{\mathrm{Sel}}_S\bm M_s\bm d=\bm M_{s,S}\bm d$, i.e.\ the coordinates $\bm V^{\top}\bm d$
(the components of $\bm d$ in $\operatorname{row}(\bm M_{s,S})$) up to noise; the orthogonal
complement $(\bm I-\bm P_{\mathrm{obs}})\bm d$ is unobserved. Decompose
$\bm e_\ell^{\top}\bm M_s=\bm e_\ell^{\top}\bm M_s\bm P_{\mathrm{obs}}
+\bm e_\ell^{\top}\bm M_s(\bm I-\bm P_{\mathrm{obs}})$. If
$\eta_{s,\ell}(S)=\|\bm e_\ell^{\top}\bm M_s(\bm I-\bm P_{\mathrm{obs}})\|_2=0$ then
$f_\ell(\bm d)=\bm e_\ell^{\top}\bm M_s\bm P_{\mathrm{obs}}\bm d$ depends on $\bm d$ only through
the observed coordinates $\bm P_{\mathrm{obs}}\bm d$, hence is determined by $\bm M_{s,S}\bm d$:
lead $\ell$'s dipolar component is identifiable. Conversely if $\eta_{s,\ell}(S)>0$ there is a
unit $\bm\delta\in\ker\bm M_{s,S}$ (an unobserved dipole direction, so $\bm M_{s,S}\bm\delta=0$)
with $\bm e_\ell^{\top}\bm M_s\bm\delta\neq0$; the two dipoles $\bm d$ and $\bm d+t\bm\delta$
produce identical observations for all $t$ but change $f_\ell$ by $t\,\bm e_\ell^{\top}\bm
M_s\bm\delta$, so no estimator recovers $f_\ell$ at any SNR. This is the estimability criterion
\cite{scheffe1959anova} for $f_\ell$.

\textbf{(ii) Conditioning.} The mean-centred estimate is
$\widehat{\bm L}_s-\bm\mu_s=\bm M_s\bm M_{s,S}^{+}(\bm y_S-\bm\mu_{s,S})$ with
$\bm y_S-\bm\mu_{s,S}=\bm M_{s,S}\bm d+\bm{\mathrm{Sel}}_S\bm r_s+\bm n$. On the identifiable
part its lead-$\ell$ error from the observed non-dipolar residual and noise is
$\bm e_\ell^{\top}\bm M_s\bm M_{s,S}^{+}(\bm{\mathrm{Sel}}_S\bm r_s+\bm n)$, and by
Cauchy--Schwarz
$|\bm e_\ell^{\top}\bm M_s\bm M_{s,S}^{+}(\bm{\mathrm{Sel}}_S\bm r_s+\bm n)|
\le\|\bm e_\ell^{\top}\bm M_s\bm M_{s,S}^{+}\|_2(\|\bm{\mathrm{Sel}}_S\bm r_s\|+\|\bm n\|)
=\kappa_{s,\ell}(S)(\|\bm{\mathrm{Sel}}_S\bm r_s\|+\|\bm n\|)$. Taking the max over $\ell$ gives
$\kappa_s(S)=\|\bm M_s\bm M_{s,S}^{+}\|_2$. \hfill$\square$

\section{Reproducibility}
\label{app:repro}
Code, fixed seeds, an exact environment lock (\texttt{env.lock.txt}), and a one-command
pipeline (\texttt{experiments/run\_all.py}) are released; every table number is regenerated
from result JSON (\texttt{paper/emit\_baseline\_table.py}), and a CI workflow runs the test
suite on every push. The neural baseline is a 24$\to$12-channel 1-D U-Net (base width 64,
residual GroupNorm blocks, 60 epochs, Adam $2{\times}10^{-4}$, MSE on masked leads, random-mask
training); the residual predictor is a gradient-boosted quantile regressor (pinball loss) with
strict fold discipline (folds 1--7 train, 9 calibrate, 10 test).

\bibliographystyle{IEEEtran}
\bibliography{refs}
\end{document}```

## paper/theorem_corrected.tex (the Proposition being proved)
```tex
% Per-target-lead identifiability + conditioning. No residual-independence claim on
% real data; the strict independence / Var(u) lower bound is stated only for the
% synthetic model (Proposition 2).

\newtheorem{theoremc}{Proposition}
\newtheorem{propc}[theoremc]{Proposition}

% ---------------------------------------------------------------- setup
\noindent\textbf{Model.} Within segment $s$ write the population 12-lead vector as
$\bm L_s = \bm\mu_s + \bm M_s\bm d + \bm r_s$, where $\bm M_s\in\mathbb R^{12\times3}$
is the (population-estimated, PCA) dipolar basis, $\bm d$ the dipole coordinates, and
$\bm r_s$ the non-dipolar residual. Observing $S$ gives $\bm y_S=\bm{\mathrm{Sel}}_S\bm L_s
+\bm n$. Let $\bm M_{s,S}=\bm{\mathrm{Sel}}_S\bm M_s$ have truncated SVD at relative
tolerance $\varrho$; $\bm M_{s,S}^{+}$ its truncated pseudo-inverse and
$\bm P_{\mathrm{obs}}=\bm M_{s,S}^{+}\bm M_{s,S}$ the projector onto the observed dipole
coordinate directions.

% ---------------------------------------------------------------- theorem
\begin{theoremc}[Per-target-lead dipolar identifiability and conditioning]
\label{thm:perlead}
The mean-centred dipolar estimate
$\widehat{\bm L}_s=\bm\mu_s+\bm M_s\bm M_{s,S}^{+}(\bm y_S-\bm\mu_{s,S})$ satisfies, for
each target lead $\ell$ with $\bm e_\ell$ the $\ell$-th unit vector:
\begin{enumerate}
\item[(i)] \emph{Identifiability.} Define
$\eta_{s,\ell}(S)=\big\|\bm e_\ell^{\top}\bm M_s(\bm I-\bm P_{\mathrm{obs}})\big\|_2$.
If $\eta_{s,\ell}(S)=0$, the dipolar component of lead $\ell$ is identifiable from $S$:
$\bm e_\ell^{\top}\bm M_s\bm d$ is determined by $\bm{\mathrm{Sel}}_S\bm M_s\bm d$. If
$\eta_{s,\ell}(S)>0$, there is a dipole direction unobserved by $S$ that changes lead
$\ell$; its dipolar component is not identifiable at any SNR.
\item[(ii)] \emph{Conditioning.} On the identifiable part, the error from observation
noise and from the observed non-dipolar residual is amplified into lead $\ell$ by
$\kappa_{s,\ell}(S)=\big\|\bm e_\ell^{\top}\bm M_s\bm M_{s,S}^{+}\big\|_2$:
$\big|\bm e_\ell^{\top}(\widehat{\bm L}_s-\bm\mu_s-\bm M_s\bm P_{\mathrm{obs}}\bm d)\big|
\le \kappa_{s,\ell}(S)\,\big(\|\bm{\mathrm{Sel}}_S\bm r_s\|+\|\bm n\|\big).$
\end{enumerate}
The global $\kappa_s(S)=\|\bm M_s\bm M_{s,S}^{+}\|_2=\max_\ell\kappa_{s,\ell}(S)$ is a
configuration-level worst case. Both $\eta_{s,\ell}$ and $\kappa_{s,\ell}$ depend on the
estimated $\bm M_s$ and on $\varrho$; near-rank-deficient $S$ are reported with an
$\varrho$-sensitivity sweep and record-bootstrap confidence intervals.
\end{theoremc}

\noindent\emph{Relation to classical estimability.} Part~(i) is the classical estimability
criterion (Gauss--Markov / \cite{scheffe1959anova}) specialized to the per-lead linear
functional $\bm e_\ell^{\top}\bm M_s\bm d$: $\eta_{s,\ell}(S)=0$ iff that functional lies in
the row space of $\bm M_{s,S}$, and $\kappa_{s,\ell}$ is its coefficient norm. We claim no
novelty for this identity. The contribution (Sec.~\ref{sec:exp}) is its use as a
\emph{graded}, per-target-lead, \emph{empirically-estimated} recoverability map: a continuous
$\eta$ that ranks target leads (not a binary rank test), a subspace $\bm M_s$ that differs
materially from the textbook inverse-Dower dipole, and per-record calibrated error rates
(Sec.~\ref{sec:calib}) attached to it---quantities not read off lead geometry.

% ---------------------------------------------------------------- residual (honest)
\noindent\textbf{The non-dipolar residual (no independence claim on real data).}
Let $\bm m_s(\bm y_S)=\mathbb E[\bm r_s\mid\bm y_S]$ and
$\bm u_s=\bm r_s-\bm m_s(\bm y_S)$, so $\mathbb E[\bm u_s\mid\bm y_S]=\bm 0$. We do
\emph{not} assume $\bm u_s\perp\bm y_S$. The Bayes-optimal reconstruction error of the
residual is
\[
\inf_{\hat{\bm r}}\ \mathbb E\big\|\bm r_s-\hat{\bm r}(\bm y_S)\big\|^2
=\mathbb E\big[\operatorname{Var}(\bm r_s\mid\bm y_S)\big].
\]
Part of $\bm r_s$ (the \emph{empirically predictable residual}) is recovered by a
predictor trained on $\bm y_S$ and metered with distribution-free calibrated intervals;
the remainder is an \emph{achievability gap}, established by a held-out achievability
analysis for a stated predictor family---\emph{not} declared unrecoverable a priori.

\begin{propc}[Strict independence limit -- synthetic model only]
\label{prop:synth}
If the data are generated so that a component $\bm u$ is statistically independent of
$\bm y_S$ (an explicit synthetic construction), then $p(\bm u\mid\bm y_S)=p(\bm u)$, the
Bayes estimate of $\bm u$ is its prior mean, and
$\mathbb E\|\widehat{\bm u}-\bm u\|^2\ge\operatorname{Var}(\bm u)$ for any estimator. This
strict limit is invoked \emph{only} for the synthetic model; on real ECG we make no such
independence claim.
\end{propc}
```

## paper/results_long_crossdata.tex
```tex
\section{Cross-Dataset Transfer of the Certificate}
\label{sec:cross}
The certificate is a function of the estimated subspace $\bm M_s$ and the selection $S$; a
fair question is whether it is an artifact of one cohort. We re-estimate $\bm M_s$ and the
per-configuration conditioning on the independent Chapman-Shaoxing 12-lead database
\cite{zheng2020chapman} ($\sim$10{,}000 patients, different country, devices, and sampling)
and compare to PTB-XL (\texttt{results/cross\_dataset.json}). We deliberately do \emph{not}
claim ``population independence''; we report what actually transfers.

\textbf{QRS subspace is transfer-stable.} For QRS the two datasets' rank-3 subspaces are nearly
aligned (principal angles $20.3^\circ/9.1^\circ/1.6^\circ$; explained-variance ratios
$0.56/0.27/0.08$ vs.\ $0.55/0.27/0.06$). The per-configuration conditioning transfers: $\kappa$
for $\{$I,II,V2$\}$ is $3.1$ (PTB-XL) vs.\ $4.1$ (Chapman), for $\{$I,II,V1,V3,V5$\}$ $2.5$
vs.\ $2.3$, and the limb-6 ill-conditioning is reproduced on both ($\kappa\!\sim\!2\times10^{5}$),
i.e.\ the \emph{qualitative} identifiable/unidentifiable verdict and the \emph{quantitative}
$\kappa$ ordering are stable across cohorts.

\textbf{ST/T third direction does not transfer.} For the low-amplitude segments (P, ST, T)
two of three subspace directions align but the third differs sharply between datasets
(max principal angle $88$--$89^\circ$). We therefore restrict the transfer claim: the QRS map
and the limb-6 conditioning verdict are cohort-stable; the ST/T \emph{third} spatial direction
is cohort-specific and its map should be re-estimated per dataset. This is the honest scope of
transfer, not a blanket invariance.```

## paper/results_long_generative.tex
```tex
\section{A Negative Result: the Generative ``Fabrication-Objective'' Story Did Not Survive}
\label{sec:generative}
This project began with a stronger claim---that a diffusion reconstructor proves
``fabrication is a property of the objective,'' a certified hallucination. A pre-submission
re-analysis retracted it. We document the retraction in full because the negative result is
itself informative and because selective reporting of the earlier (favourable) numbers would
be exactly the failure the certificate is meant to prevent.

\textbf{Setup.} An arbitrary-mask conditional DDPM (12-channel binary mask $+$ observed values;
random-mask training so one model serves any configuration, with explicit leakage tests)
reconstructs $\{$V2,V4,V6$\}$ from $\{$I,II,V1,V3,V5$\}$ under classifier-free-style guidance
weight $w\in\{1,2,4,6\}$ (\texttt{results/gpu\_diffusion\_leakfixed.json},
\texttt{realism\_metrics.json}, \texttt{gpu\_deficit\_ci.json}).

\textbf{(1) The original effect was a configuration-leakage artifact.} The earlier exhibit
reused one model trained on a fixed observed set to score a \emph{different} configuration.
With leakage removed (arbitrary-mask model $+$ leakage guards) and $5$ training seeds, the
``achievability gap'' $\Delta\rho=\rho_{\text{achievable}}-\rho_{\text{model}}$ is monotone in
$w$ but \emph{marginal}: mean $\Delta\rho=-0.031,-0.024,+0.024,+0.058$ for $w=1,2,4,6$, with
$\pm0.044$ at $w{=}6$ ($\approx1.3\sigma$). The previously reported ``$4.3\sigma$'' does not
reproduce.

\textbf{(2) Realism does not improve with guidance---the causal premise is false.} The story
required higher guidance to buy realism at the cost of fidelity. Instead both realism metrics
\emph{worsen} monotonically with $w$: the target-lead PSD log-distance rises (V2:
$1.95\!\to\!2.35$ from $w{=}1$ to $6$) and the amplitude-Wasserstein distance rises (V2:
$0.29\!\to\!0.67$). High guidance simply degrades the model on every axis; there is no
realism/fidelity trade to exploit.

\textbf{(3) What is true instead.} The predictable non-dipolar residual is real but modest:
oracle correlations (\texttt{gpu\_oracle\_gate.json}) are $\rho\!\approx\!0.49$--$0.55$ for V2
(QRS/ST) and $0.2$--$0.3$ for V4/V6---consistent with the calibrated Tier-II intervals of
Sec.~\ref{sec:calib}. So the honest statement is a \emph{graded predictability} of the
residual, calibrated per feature and lead, not a proof of fabrication. We removed the
fabrication-objective claim and rebuilt the paper around the parts that withstand scrutiny.```

## results/cross_dataset.json
```json
{
  "n_chapman_records": 349,
  "processing": "100Hz, lead-II dwt, top-3 spatial eig",
  "segments": {
    "P": {
      "principal_angles_deg": [
        88.38,
        31.83,
        6.1
      ],
      "max_angle_deg": 88.38,
      "ptbxl_evr3": [
        0.238,
        0.175,
        0.127
      ],
      "chapman_evr3": [
        0.315,
        0.231,
        0.151
      ],
      "n_ptb": 59940,
      "n_chap": 13781
    },
    "QRS": {
      "principal_angles_deg": [
        20.28,
        9.06,
        1.61
      ],
      "max_angle_deg": 20.28,
      "ptbxl_evr3": [
        0.558,
        0.273,
        0.076
      ],
      "chapman_evr3": [
        0.545,
        0.272,
        0.064
      ],
      "n_ptb": 59917,
      "n_chap": 13954
    },
    "ST": {
      "principal_angles_deg": [
        88.34,
        12.97,
        10.61
      ],
      "max_angle_deg": 88.34,
      "ptbxl_evr3": [
        0.382,
        0.237,
        0.098
      ],
      "chapman_evr3": [
        0.417,
        0.246,
        0.1
      ],
      "n_ptb": 59913,
      "n_chap": 13867
    },
    "T": {
      "principal_angles_deg": [
        89.15,
        22.39,
        6.75
      ],
      "max_angle_deg": 89.15,
      "ptbxl_evr3": [
        0.513,
        0.143,
        0.085
      ],
      "chapman_evr3": [
        0.518,
        0.147,
        0.094
      ],
      "n_ptb": 59945,
      "n_chap": 13782
    }
  },
  "kappa_QRS": {
    "{I,II,V2}": {
      "ptbxl": 3.14,
      "chapman": 4.11,
      "rank_ptb": 3,
      "rank_chap": 3
    },
    "{I,II,V1,V3,V5}": {
      "ptbxl": 2.54,
      "chapman": 2.26,
      "rank_ptb": 3,
      "rank_chap": 3
    },
    "{V1,V2,V3}": {
      "ptbxl": 3.94,
      "chapman": 6.74,
      "rank_ptb": 3,
      "rank_chap": 3
    },
    "limb-6": {
      "ptbxl": 206991.81,
      "chapman": 184162.7,
      "rank_ptb": 3,
      "rank_chap": 3
    }
  }
}```

## results/gpu_deficit_ci.json (achievability gap, 5 seeds)
```json
{
  "n_seeds": 5,
  "n_test": 500,
  "epochs": 70,
  "guidances": [
    1.0,
    2.0,
    4.0,
    6.0
  ],
  "oracle_agg": 0.3073480083545236,
  "per_seed_delta_rho": {
    "1.0": [
      -0.052583378443445974,
      -0.02680629711601997,
      -0.027829352588199283,
      -0.005359271301941229,
      -0.043073033427526566
    ],
    "2.0": [
      -0.051227266455580844,
      -0.019376494103704622,
      0.01428198281960514,
      -0.002026613189888421,
      -0.0592839442568832
    ],
    "4.0": [
      -0.016305432126908753,
      0.043819240270207116,
      0.08407003240651033,
      0.031215961367023117,
      -0.020911679244257724
    ],
    "6.0": [
      0.010948531249432339,
      0.09034607500067346,
      0.12296707753763586,
      0.058188840487927246,
      0.009661282196017409
    ]
  },
  "mean": {
    "1.0": -0.031130266575426602,
    "2.0": -0.02352646703729039,
    "4.0": 0.024377624534514815,
    "6.0": 0.058422361294337255
  },
  "std": {
    "1.0": 0.016103432126899724,
    "2.0": 0.028124282522093196,
    "4.0": 0.03922793638897761,
    "6.0": 0.04430938004004163
  }
}```

## results/realism_metrics.json
```json
{
  "n_test": 300,
  "guidances": [
    1.0,
    2.0,
    4.0,
    6.0
  ],
  "obs": [
    "I",
    "II",
    "V1",
    "V3",
    "V5"
  ],
  "targets": [
    "V2",
    "V4",
    "V6"
  ],
  "per_w": {
    "1.0": {
      "psd_logdist": {
        "V2": 1.9461422006289164,
        "V4": 1.7235506451129914,
        "V6": 1.7571495215098063
      },
      "amp_wasserstein": {
        "V2": 0.29380000164111447,
        "V4": 0.6441409550110498,
        "V6": 0.5823854563633597
      },
      "qrs_width_wasserstein": 0.0,
      "gen_qrs_delineation_rate": 10.846666666666666
    },
    "2.0": {
      "psd_logdist": {
        "V2": 1.8688957659403482,
        "V4": 1.63987104733785,
        "V6": 1.6942854948838553
      },
      "amp_wasserstein": {
        "V2": 0.1641430090864499,
        "V4": 0.604516767859459,
        "V6": 0.553678226669629
      },
      "qrs_width_wasserstein": 0.0,
      "gen_qrs_delineation_rate": 10.846666666666666
    },
    "4.0": {
      "psd_logdist": {
        "V2": 2.027515626748403,
        "V4": 1.7656704533100127,
        "V6": 1.8427314205964407
      },
      "amp_wasserstein": {
        "V2": 0.28434679597616197,
        "V4": 0.5645589091380437,
        "V6": 0.6176233494281764
      },
      "qrs_width_wasserstein": 0.0,
      "gen_qrs_delineation_rate": 10.846666666666666
    },
    "6.0": {
      "psd_logdist": {
        "V2": 2.3512677995363873,
        "V4": 2.012936324675878,
        "V6": 2.1028463701407114
      },
      "amp_wasserstein": {
        "V2": 0.6736846698323885,
        "V4": 0.6254885850350063,
        "V6": 0.7786913079023356
      },
      "qrs_width_wasserstein": 0.0,
      "gen_qrs_delineation_rate": 10.846666666666666
    }
  }
}```

## results/gpu_oracle_gate.json (head)
```json
{
  "n_train": 2000,
  "n_test": 800,
  "configs": {
    "precordial-interp": {
      "QRS": {
        "V2": {
          "rho_oracle": 0.4917352760285542,
          "true_nondip_rms_mV": 0.15041848197933305
        },
        "V4": {
          "rho_oracle": 0.3072603814985097,
          "true_nondip_rms_mV": 0.16530451314254307
        },
        "V6": {
          "rho_oracle": 0.34074420322598964,
          "true_nondip_rms_mV": 0.182116327365601
        }
      },
      "ST": {
        "V2": {
          "rho_oracle": 0.5537673357195662,
          "true_nondip_rms_mV": 0.09138137061730273
        },
        "V4": {
          "rho_oracle": 0.20752460971032125,
          "true_nondip_rms_mV": 0.11612855523163612
        },
        "V6": {
          "rho_oracle": 0.2638188675166644,
          "true_nondip_rms_mV": 0.14478502562357942
        }
      },
      "T": {
        "V2": {
          "rho_oracle": 0.48620815427798386,
          "true_nondip_rms_mV": 0.09353277102299112
        },
        "V4": {
          "rho_oracle": 0.2225073045115776,
          "true_nondip_rms_mV": 0.1180315033469179
        },
        "V6": {
          "rho_oracle": 0.28224821199097694,
          "true_nondip_rms_mV": 0.13233268764472086
        }
      },
      "P": {
        "V2": {
          "rho_oracle": 0.2658679713430947,
          "true_nondip_rms_mV": 0.09637378487383792
        },
        "V4": {
          "rho_oracle": 0.10780603672649731,
          "true_nondip_rms_mV": 0.08889502934653662
        },
        "V6": {
          "rho_oracle": 0.8221124071794557,
          "true_nondip_rms_mV": 0.05804843271750684
        }
      }
    },
    "classic-3lead": {
      "QRS": {
        "V1": {
          "rho_oracle": 0.1538532366193525,
          "true_nondip_rms_mV": 0.15341474392045648
        },
        "V3": {
          "rho_oracle": 0.08737022563517167,
          "true_nondip_rms_mV": 0.18065753290202796
    ```
