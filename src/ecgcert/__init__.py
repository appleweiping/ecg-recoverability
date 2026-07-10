"""ecgcert -- Certified Reduced-Lead ECG Reconstruction.

A per-feature (P/QRS/ST/T), per-lead recoverability *certificate* for the ill-posed
reduced-lead ECG reconstruction inverse problem, built from three pieces:

    physics/   -- the ECG dipolar subspace and the closed-form recoverability
                  certificate (Tier I exact recovery + noise-amplification kappa_s(S)).
    certify/   -- the Tier I / II / III decomposition and the null-space
                  hallucination energy h_{s,l}.
    conformal/ -- Mondrian conformalized quantile regression (per feature x lead)
                  and conformal risk control for the hallucination flag.

The scientific claim is a *positive* physical certificate (which named cardiac
feature, on which lead, is exactly recoverable and with what noise gain), not a
generic null-space impossibility bound (that is credited to the imaging inverse-
problem literature and only *used* here).
"""

__version__ = "0.1.0"
