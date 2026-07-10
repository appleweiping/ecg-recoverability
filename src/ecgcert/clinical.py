"""Clinical ST-segment measurement and the STEMI safety endpoint.

PTB-XL provides MI / STTC diagnostic labels but no per-lead ST-elevation
millivolt annotation and no clean STEMI/NSTEMI split, so we use a *measured*
physical endpoint: the ST deviation at the J point + 60 ms relative to the
isoelectric PR baseline, with the classic 0.1 mV significance threshold.

The safety experiment reconstructs the precordial leads and asks: does the
reconstruction *fabricate* a significant ST deviation the true signal does not
have (a phantom STEMI), or *mask* one that is truly present?  These are the
clinically dangerous failure modes that global RMSE hides.
"""
from __future__ import annotations

import numpy as np

ST_THRESHOLD_MV = 0.1        # classic significance threshold for ST deviation
ST_OFFSET_MS = 60.0          # measure ST at J point + 60 ms


def _fiducials(lead_signal: np.ndarray, fs: int):
    import neurokit2 as nk

    try:
        _, rpeaks = nk.ecg_peaks(lead_signal, sampling_rate=fs)
        _, waves = nk.ecg_delineate(lead_signal, rpeaks, sampling_rate=fs, method="dwt")
    except Exception:
        return None
    def arr(k):
        v = np.asarray(waves.get(k, []), float)
        return v[~np.isnan(v)].astype(int)
    return {"P_on": arr("ECG_P_Onsets"), "R_on": arr("ECG_R_Onsets"),
            "R_off": arr("ECG_R_Offsets"), "rpeaks": np.asarray(rpeaks["ECG_R_Peaks"])}


def st_deviation(sig: np.ndarray, fs: int, ref_lead: int = 1) -> np.ndarray:
    """Per-lead ST deviation (mV), averaged over beats.

    ``sig`` is (T, 12).  Fiducials are located on ``ref_lead`` (II) and applied to
    all leads.  For each beat: baseline = mean over the PR segment (P onset .. R
    onset); ST value = signal at J point (R offset) + 60 ms; deviation = ST value -
    baseline.  Returns a length-12 vector of mean deviations.
    """
    fid = _fiducials(sig[:, ref_lead], fs)
    if fid is None or fid["R_off"].size == 0:
        return np.full(12, np.nan)
    off = int(round(ST_OFFSET_MS * fs / 1000.0))
    T = sig.shape[0]
    devs = []
    for j in fid["R_off"]:
        st_t = int(j) + off
        if st_t >= T:
            continue
        # baseline: nearest preceding PR segment (P onset .. following R onset).
        base = _pr_baseline(sig, fid, j)
        devs.append(sig[st_t] - base)                 # (12,)
    if not devs:
        return np.full(12, np.nan)
    return np.mean(np.stack(devs, axis=0), axis=0)     # (12,)


def _pr_baseline(sig: np.ndarray, fid: dict, r_off: int) -> np.ndarray:
    """Isoelectric baseline from the PR segment preceding this beat's QRS."""
    p_on = fid["P_on"]
    r_on = fid["R_on"]
    # find the R onset for this beat (the largest R_on <= r_off)
    prior_ron = r_on[r_on <= r_off]
    if prior_ron.size == 0:
        return np.zeros(sig.shape[1])
    ron = int(prior_ron[-1])
    prior_pon = p_on[p_on < ron]
    if prior_pon.size == 0:
        # fall back to a short window just before QRS onset
        a = max(0, ron - max(1, sig.shape[0] // 100))
        return sig[a:ron].mean(axis=0) if ron > a else sig[ron]
    pon = int(prior_pon[-1])
    seg = sig[pon:ron]
    return seg.mean(axis=0) if seg.shape[0] > 0 else sig[ron]


def stemi_positive(dev: np.ndarray, leads=None, thr: float = ST_THRESHOLD_MV) -> bool:
    """A (very simplified) ST-elevation positivity rule: any monitored lead with
    ST elevation >= ``thr`` mV.  ``leads`` restricts to a lead-index subset."""
    d = dev if leads is None else dev[list(leads)]
    d = d[~np.isnan(d)]
    return bool(np.any(d >= thr))


def count_stemi_flips(true_sig: np.ndarray, recon_sig: np.ndarray, fs: int,
                      leads=None, thr: float = ST_THRESHOLD_MV) -> dict:
    """Compare ST positivity on true vs reconstructed signals.

    Returns dict with ``fabricated`` (recon positive, truth negative) and
    ``masked`` (truth positive, recon negative) booleans plus the per-lead
    deviation error on the monitored leads.
    """
    dt = st_deviation(true_sig, fs)
    dr = st_deviation(recon_sig, fs)
    if np.all(np.isnan(dt)) or np.all(np.isnan(dr)):
        return {"valid": False}
    pos_t = stemi_positive(dt, leads, thr)
    pos_r = stemi_positive(dr, leads, thr)
    lead_idx = list(range(12)) if leads is None else list(leads)
    err = np.nanmean(np.abs(dr[lead_idx] - dt[lead_idx]))
    return {"valid": True, "true_positive": pos_t, "recon_positive": pos_r,
            "fabricated": (pos_r and not pos_t), "masked": (pos_t and not pos_r),
            "st_error_mv": float(err)}
