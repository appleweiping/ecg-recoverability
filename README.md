# ecg-recoverability

**Certified Reduced-Lead ECG Reconstruction: a Per-Feature Recoverability Certificate with Distribution-Free Calibration.**

> Work in progress. This README is a stub; the full, detailed write-up lands with the paper (milestone M6).

Reconstructing missing ECG leads from a reduced set is an ill-posed inverse problem.
Existing methods output a point reconstruction and an aggregate error number; generative
reconstructors can *hallucinate* clinically critical morphology (a fabricated ST elevation,
an invented fractionated QRS) while the global error looks fine.

This project instead outputs, **per waveform feature (P / QRS / ST / T) and per lead**, a
*recoverability certificate*:

- **Tier I — provably recoverable** (the cardiac-dipole component; exact up to a closed-form
  noise gain `κ_s(S)`),
- **Tier II — statistically recoverable** (population-correlated residual; distribution-free
  calibrated interval),
- **Tier III — provably unrecoverable** (non-dipolar local content in the observation
  null space; any reconstruction of it is hallucination, and we flag it with a guarantee).

The theoretical core is the ECG *dipolar* structure (a physical low-rank + a physically
meaningful null space) — something generic imaging inverse-problem theory does not have.

## Status
- [x] Physics core: lead algebra + per-segment dipolar subspace + `κ_s(S)` certificate (tested)
- [ ] Tier decomposition + hallucination energy
- [ ] Mondrian-CQR calibration + conformal-risk-control flag
- [ ] Experiments (PTB-XL + PhysioNet/CinC-2021)
- [ ] Paper (IEEE ICASSP, spconf)

## License
MIT (see `LICENSE`).
