"""ST-segment measurement and the ST-threshold-event endpoint (no diagnosis language).

PTB-XL has no per-lead ST-deviation millivolt annotation, so we use a *measured* physical
endpoint: the ST deviation at the J point + 60 ms relative to the ISOELECTRIC PR-segment
baseline, with the classic 0.1 mV threshold applied as an *absolute* deviation
``|ST| >= 0.1 mV`` (elevation or depression). We report ST-threshold events only; no clinical
diagnosis is asserted.

The PR baseline is the mean over the isoelectric PR SEGMENT ``[P_offset, R_onset)`` -- i.e.
*after* the P wave ends and *before* the QRS begins -- so a nonzero P wave does not bias it.
When P_offset is unavailable we fall back to a short (20 ms) window just before QRS onset and
record how often the fallback fired. Fiducials are located ONCE on the observed Lead II and
shared between truth and reconstruction; beat values are aggregated by MEDIAN over beats
(robust to a single bad beat).
"""
from __future__ import annotations

import numpy as np

ST_THRESHOLD_MV = 0.1        # classic 0.1 mV threshold (absolute)
ST_OFFSET_MS = 60.0          # measure ST at J point + 60 ms
PR_FALLBACK_MS = 20.0        # fallback baseline window before QRS onset
BEAT_AGGREGATION = "median"  # aggregate per-beat ST deviations by median over beats


def fiducials(lead_signal: np.ndarray, fs: int):
    """Locate P/R/J fiducials on ONE lead (observed Lead II). Returns None on failure.

    Includes P offsets (needed for the isoelectric PR-segment baseline). Compute once and
    pass to :func:`st_deviation` for both truth and reconstruction.
    """
    import neurokit2 as nk

    try:
        _, rpeaks = nk.ecg_peaks(lead_signal, sampling_rate=fs)
        _, waves = nk.ecg_delineate(lead_signal, rpeaks, sampling_rate=fs, method="dwt")
    except Exception:
        return None

    def arr(k):
        v = np.asarray(waves.get(k, []), float)
        return v[~np.isnan(v)].astype(int)

    return {"P_on": arr("ECG_P_Onsets"), "P_off": arr("ECG_P_Offsets"),
            "R_on": arr("ECG_R_Onsets"), "R_off": arr("ECG_R_Offsets"),
            "rpeaks": np.asarray(rpeaks["ECG_R_Peaks"])}


def _pr_baseline(sig: np.ndarray, fid: dict, r_off: int, fs: int):
    """Isoelectric PR-segment baseline [P_offset, R_onset) for the beat ending at ``r_off``.

    Returns ``(baseline_vec (12,), used_fallback: bool)``. Fallback = a 20 ms window just
    before QRS onset when P_offset is unavailable.
    """
    r_on = fid["R_on"]; p_off = fid["P_off"]
    prior_ron = r_on[r_on <= r_off]
    if prior_ron.size == 0:
        return np.zeros(sig.shape[1]), True
    ron = int(prior_ron[-1])
    prior_poff = p_off[p_off < ron]
    if prior_poff.size:
        poff = int(prior_poff[-1])
        if ron - poff >= 2:                                  # a real PR segment exists
            return sig[poff:ron].mean(axis=0), False
    w = max(2, int(round(PR_FALLBACK_MS * fs / 1000.0)))     # fallback: 20 ms pre-QRS
    a = max(0, ron - w)
    return (sig[a:ron].mean(axis=0) if ron > a else sig[ron]), True


def st_deviation(sig: np.ndarray, fs: int, ref_lead: int = 1, fid: dict | None = None):
    """Per-lead ST deviation (mV), MEDIAN over beats, plus the fallback fraction.

    ``sig`` is (T, 12). Returns ``(dev (12,), fallback_frac)``. For each beat: baseline = mean
    over the isoelectric PR segment [P_offset, R_onset); ST value = signal at J+60 ms; deviation
    = ST value - baseline. Aggregated by median across beats.
    """
    if fid is None:
        fid = fiducials(sig[:, ref_lead], fs)
    if fid is None or np.asarray(fid["R_off"]).size == 0:
        return np.full(12, np.nan), 1.0
    off = int(round(ST_OFFSET_MS * fs / 1000.0))
    T = sig.shape[0]
    devs, nfb = [], 0
    for j in fid["R_off"]:
        st_t = int(j) + off
        if st_t >= T:
            continue
        base, fb = _pr_baseline(sig, fid, int(j), fs)
        nfb += int(fb)
        devs.append(sig[st_t] - base)                        # (12,)
    if not devs:
        return np.full(12, np.nan), 1.0
    return np.median(np.stack(devs, axis=0), axis=0), nfb / len(devs)


def st_threshold_positive(dev: np.ndarray, leads=None, thr: float = ST_THRESHOLD_MV) -> bool:
    """ST-threshold positivity: any monitored lead with |ST deviation| >= ``thr`` mV."""
    d = dev if leads is None else dev[list(leads)]
    d = d[~np.isnan(d)]
    return bool(np.any(np.abs(d) >= thr))


def st_threshold_events(true_sig: np.ndarray, recon_sig: np.ndarray, fs: int,
                        leads=None, thr: float = ST_THRESHOLD_MV,
                        fid: dict | None = None) -> dict:
    """Compare |ST| >= thr positivity on true vs reconstructed signals on SHARED fiducials.

    Returns ``false_positive`` (recon crosses, truth does not), ``false_negative``
    (truth crosses, recon does not), the mean absolute per-lead ST error, and the baseline
    ``fallback_frac`` (shared, from the observed Lead II fiducials). No diagnosis is asserted.
    """
    dt, fbt = st_deviation(true_sig, fs, fid=fid)
    dr, _ = st_deviation(recon_sig, fs, fid=fid)
    if np.all(np.isnan(dt)) or np.all(np.isnan(dr)):
        return {"valid": False}
    pos_t = st_threshold_positive(dt, leads, thr)
    pos_r = st_threshold_positive(dr, leads, thr)
    lead_idx = list(range(12)) if leads is None else list(leads)
    err = np.nanmean(np.abs(dr[lead_idx] - dt[lead_idx]))
    return {"valid": True, "true_positive": pos_t, "recon_positive": pos_r,
            "false_positive": (pos_r and not pos_t), "false_negative": (pos_t and not pos_r),
            "st_error_mv": float(err), "fallback_frac": float(fbt)}
