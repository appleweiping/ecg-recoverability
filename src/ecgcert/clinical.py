"""ST-segment measurement and the ST-threshold-event endpoint (no diagnosis language).

PTB-XL has no per-lead ST-deviation millivolt annotation, so we use a *measured*
physical endpoint: the ST deviation at the J point + 60 ms relative to the isoelectric
PR baseline, with the classic 0.1 mV significance threshold, applied as an *absolute*
deviation ``|ST| >= 0.1 mV`` (elevation or depression). We report ST-threshold events only;
no clinical diagnosis is asserted anywhere in this module.

The safety experiment reconstructs precordial leads and asks whether a reconstruction
produces a threshold crossing the truth does not have (a *false positive* ST-threshold
event) or misses one that is present (a *false negative*). Fiducials are located ONCE on
the observed Lead II and shared between truth and reconstruction, so the two are measured
on an identical time grid (fair comparison; the observed lead is exact in both).
"""
from __future__ import annotations

import numpy as np

ST_THRESHOLD_MV = 0.1        # classic 0.1 mV significance threshold (absolute)
ST_OFFSET_MS = 60.0          # measure ST at J point + 60 ms


def fiducials(lead_signal: np.ndarray, fs: int):
    """Locate P/R/J fiducials on ONE lead (the observed Lead II). Returns None on failure.

    Compute once on the observed lead and pass to :func:`st_deviation` for both truth and
    reconstruction so both are measured on the same beats and offsets.
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

    return {"P_on": arr("ECG_P_Onsets"), "R_on": arr("ECG_R_Onsets"),
            "R_off": arr("ECG_R_Offsets"), "rpeaks": np.asarray(rpeaks["ECG_R_Peaks"])}


def st_deviation(sig: np.ndarray, fs: int, ref_lead: int = 1, fid: dict | None = None) -> np.ndarray:
    """Per-lead ST deviation (mV), averaged over beats.

    ``sig`` is (T, 12). If ``fid`` (from :func:`fiducials` on the observed lead) is given it
    is used directly; otherwise fiducials are located on ``ref_lead`` (II). For each beat:
    baseline = mean over the PR segment (P onset .. R onset); ST value = signal at J point
    (R offset) + 60 ms; deviation = ST value - baseline. Returns a length-12 vector.
    """
    if fid is None:
        fid = fiducials(sig[:, ref_lead], fs)
    if fid is None or np.asarray(fid["R_off"]).size == 0:
        return np.full(12, np.nan)
    off = int(round(ST_OFFSET_MS * fs / 1000.0))
    T = sig.shape[0]
    devs = []
    for j in fid["R_off"]:
        st_t = int(j) + off
        if st_t >= T:
            continue
        base = _pr_baseline(sig, fid, j)
        devs.append(sig[st_t] - base)                 # (12,)
    if not devs:
        return np.full(12, np.nan)
    return np.mean(np.stack(devs, axis=0), axis=0)     # (12,)


def _pr_baseline(sig: np.ndarray, fid: dict, r_off: int) -> np.ndarray:
    """Isoelectric baseline from the PR segment preceding this beat's QRS."""
    p_on = fid["P_on"]
    r_on = fid["R_on"]
    prior_ron = r_on[r_on <= r_off]
    if prior_ron.size == 0:
        return np.zeros(sig.shape[1])
    ron = int(prior_ron[-1])
    prior_pon = p_on[p_on < ron]
    if prior_pon.size == 0:
        a = max(0, ron - max(1, sig.shape[0] // 100))
        return sig[a:ron].mean(axis=0) if ron > a else sig[ron]
    pon = int(prior_pon[-1])
    seg = sig[pon:ron]
    return seg.mean(axis=0) if seg.shape[0] > 0 else sig[ron]


def st_threshold_positive(dev: np.ndarray, leads=None, thr: float = ST_THRESHOLD_MV) -> bool:
    """ST-threshold positivity: any monitored lead with |ST deviation| >= ``thr`` mV
    (elevation OR depression). ``leads`` restricts to a lead-index subset."""
    d = dev if leads is None else dev[list(leads)]
    d = d[~np.isnan(d)]
    return bool(np.any(np.abs(d) >= thr))


def st_threshold_events(true_sig: np.ndarray, recon_sig: np.ndarray, fs: int,
                        leads=None, thr: float = ST_THRESHOLD_MV,
                        fid: dict | None = None) -> dict:
    """Compare |ST| >= thr positivity on true vs reconstructed signals on SHARED fiducials.

    ``fid`` (from the observed Lead II) is used for BOTH signals so the comparison is on an
    identical time grid. Returns ``false_positive`` (recon crosses, truth does not),
    ``false_negative`` (truth crosses, recon does not), and the mean absolute per-lead ST
    error on the monitored leads. No diagnosis is asserted -- this is a threshold-crossing
    agreement measure only.
    """
    dt = st_deviation(true_sig, fs, fid=fid)
    dr = st_deviation(recon_sig, fs, fid=fid)
    if np.all(np.isnan(dt)) or np.all(np.isnan(dr)):
        return {"valid": False}
    pos_t = st_threshold_positive(dt, leads, thr)
    pos_r = st_threshold_positive(dr, leads, thr)
    lead_idx = list(range(12)) if leads is None else list(leads)
    err = np.nanmean(np.abs(dr[lead_idx] - dt[lead_idx]))
    return {"valid": True, "true_positive": pos_t, "recon_positive": pos_r,
            "false_positive": (pos_r and not pos_t), "false_negative": (pos_t and not pos_r),
            "st_error_mv": float(err)}
