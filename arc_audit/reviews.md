# Adversarial review findings → fixing commits

**These reviews were produced by Claude Code sub-agents / a multi-agent workflow, applying ARC's
peer-review + adversarial-verification methodology. They are NOT AutoResearchClaw output (ARC's real
pipeline could not run here — see `ARC_STATUS.md`).** Branch: `presubmission-rebuild`.

## Round A — adversarial correctness reviewer (CC general-purpose sub-agent)

| # | Finding | Severity | Fixed in |
|---|---|---|---|
| A1 | Paper claimed reconstructors "sit on/above the floor"; `certificate_validation.json` shows dipolar violates the worst-case floor ~21% and ridge/OLS ~60% of η>0 cells | CRITICAL | `3329f6d`, tightened `fb567bf` |
| A2 | `\CertValFloorGap` "median gap 0.009" is an η=0-diluted artifact (ridge η>0 median gap −0.005) | CRITICAL | `3329f6d` → η>0 stats; macro removed `fb567bf` |
| A3 | DDPM "0/39 cells" oversold (21 trivial η=0 cells); DDPM avoids violations only by being looser than the floor | MODERATE | `3329f6d`, demoted `fb567bf` |
| A4 | Transfer "stable Lipschitz constant across a sixfold range" rests on n=2 with an empty ST CI | MODERATE | `3329f6d`, demoted to empirical sensitivity `fb567bf` |
| A5 | Fabrication docstring "a minimax bound must be obeyed by a specific estimator" — logically wrong | MODERATE | `fb567bf` |

Verified-correct by the same review (unchanged): rotation-invariance of η; `2 sin(θ*/2)` alignment;
constant-2 Stewart–Wedin projector bound; `κ=‖M_S⁺‖`; the exact-floor Schur-complement algebra.

## Round B — 5-agent verification workflow + final front-matter reviewer (CC)

| # | Finding | Fixed in |
|---|---|---|
| B1 | Intro topic sentence "lower-bounds every reconstructor" un-hedged | `8e46ae9` |
| B2 | arXiv "the certificate binds even a learned reconstructor" ("binds" overclaims) | `8e46ae9` |
| B3 | Abstract parenthetical implied the diffusion model was inside the linear-reconstructor Spearman/AUC | `8e46ae9` |
| B4 | Robustness contribution pointed to `sec:exp`; it lives in `sec:limits` | `8e46ae9` |
| B5 | `Thm.~\ref` to `theoremc` (renders as "Proposition") — label inconsistency | `8e46ae9` |

## Round C — owner directive P0-1…P0-7 (human review; not an agent)

Recorded here for completeness (source: repository owner, not ARC):
`fb567bf` (P0-1 fabrication semantics + rename φ→R_Q, P0-2 exact-minimax-only theorem + direct proof,
P0-3 transfer bound demoted/vacuous, P0-4 active-selection demoted / lead-channel budget / drop
1−1/e); `8415f9e` (P0-5 ICASSP exact-vs-ρ + ST claims + paired total-δ, P0-7 README); `2e37441`
(P0-6 release-gate infra).
