# Frozen ICASSP 2027 Stage 1 claim boundary

Write the SMART goal for the already-frozen project titled **Robust
Target-Specific Recoverability Maps for Reduced-Lead ECG Reconstruction**.
The target venue is ICASSP 2027. Do not redefine or narrow the protocol.

The sole central claim is: a **cross-rank robust, target-lead-specific,
model-conditional** recoverability score predicts unseen-patient
reconstruction error beyond simple configuration features, with external
zero-transfer evidence. Keep every claim explicitly conditional on the fitted
model and empirical protocol. Do not add universal guarantees, formal proof
language, medical deployment assurances, or a negative claim-boundary list.

Bind the goal to these frozen facts:

- PTB-XL is the full-population primary cohort. Folds 1-7 train, fold 8 tunes,
  fold 9 fits the error meta-model, and fold 10 is used once for final testing.
- The primary 500 Hz windows are QRS, ST, and T. The lifted independent-lead
  representation starts from `[I, II, V1-V6]`.
- The preregistered rank set is `{2,3,4,5}`. The main score is the cross-rank
  bootstrap envelope `A_robust = max_r Q97.5(A_r)` using 2,000 patient
  resamples; no final-test rank selection is allowed.
- The structural map covers all 255 non-empty independent-lead subsets. Deep
  models use the frozen 64-configuration panel.
- The shared benchmark includes low-rank conditional mean, tuned ridge, the
  arbitrary-mask U-Net, pinned ImputeECG, and the permission-authorized pinned
  ECGrecover upstream; neural methods use five seeds.
- The primary outcome is patient-level `log(RMSE)` on missing targets. Fold 9
  fits simple and augmented meta-models; the main effect is Delta R^2 on fold
  10 with patient-clustered uncertainty and nested neural-seed resampling.
- Both Chapman and CPSC receive zero-transfer test evaluation. Report both.
- Stage 15 may PROCEED only when the fold-10 Delta R^2 CI lower bound is above
  zero, Chapman has a zero-transfer CI lower bound above zero, and at least
  three of four common-panel primary reconstructors have a positive point
  estimate. CPSC remains mandatory reported sensitivity but cannot trigger the
  hard gate without a public cross-record patient key. Otherwise PIVOT
  transparently; do not retune.
- Compute ceilings are 500 GPU-hours, 4,000 CPU-core-hours, and 100 GB of
  artifacts. The intended submission date is 2026-09-16.

Success criteria must be empirical and falsifiable. Do not promise that both
external cohorts pass; the registered gate requires Chapman while both must be
reported. Do not substitute Spearman correlation for Delta R^2 as the
primary endpoint.

In the `Success Criteria` section, write exactly three list items: (1) the
PTB-XL fold-10 Delta R^2 confidence-interval gate, (2) the at-least-one
external zero-transfer confidence-interval gate, and (3) the three-of-four
reconstructor direction gate. Add no fourth publishability condition. In
particular, agreement across QRS, ST, and T is sensitivity/interpretation
evidence only and must never determine PROCEED or PIVOT. After the three-item
list, state the no-retuning failure-to-PIVOT policy as prose, not as another
list item. Make every gate item self-contained: the external item must
explicitly name zero-transfer Delta R^2 and its confidence-interval lower
bound rather than referring to "the same comparison."
