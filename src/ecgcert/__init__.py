"""Robust, model-conditional recoverability maps for reduced-lead ECG.

The primary research path evaluates target-specific scores over the preregistered
rank grid ``(2, 3, 4, 5)``, empirical basis variants, and patient-cluster
bootstrap refits.  It tests whether those scores add held-out predictive value
for patient-level reconstruction error across reconstruction methods and on
external zero-transfer cohorts.

Scientific conclusions are fail-closed: Stage 15 must record ``PROCEED`` before
final empirical values or a positive headline can be released.  Historical
fixed-subspace, calibration, event-analysis, selection, and generative modules
remain available for compatibility and provenance, but they are not primary
evidence and do not block the refactored submission path.
"""

__version__ = "0.1.0"
