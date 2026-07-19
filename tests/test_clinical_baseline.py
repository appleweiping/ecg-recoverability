"""Synthetic validation of the ST reference baseline (P0-B).

Proves (1) a nonzero P wave does not bias the isoelectric PR-segment baseline, and
(2) a known ST elevation/depression is recovered. Uses explicit fiducials so the test does
not depend on NeuroKit delineation.
"""
import numpy as np

from ecgcert.clinical import st_deviation, _pr_baseline, ST_OFFSET_MS

FS = 100
OFF = int(round(ST_OFFSET_MS * FS / 1000.0))   # 6 samples


def _synth(st_level_mv, p_amp_mv=0.5, n_beats=3, period=100):
    """Build a (T,12) signal + explicit fiducials. Isoelectric everywhere except a P-wave
    bump [P_on,P_off) and an ST level at J+60ms. Same on all 12 leads."""
    T = period * (n_beats + 1)
    sig = np.zeros((T, 12))
    P_on, P_off, R_on, R_off = [], [], [], []
    for b in range(n_beats):
        t0 = period * (b + 1)                 # R onset
        pon, poff, ron, roff = t0 - 40, t0 - 25, t0, t0 + 10
        sig[pon:poff] += p_amp_mv             # large P wave BEFORE the PR segment
        sig[roff + OFF] = st_level_mv         # ST value at J+60ms
        P_on.append(pon); P_off.append(poff); R_on.append(ron); R_off.append(roff)
    fid = {"P_on": np.array(P_on), "P_off": np.array(P_off),
           "R_on": np.array(R_on), "R_off": np.array(R_off), "rpeaks": np.array(R_on)}
    return sig, fid


def test_p_wave_does_not_bias_baseline():
    # Large P wave (0.8 mV) but isoelectric PR segment -> baseline must be ~0, not pulled up.
    sig, fid = _synth(st_level_mv=0.0, p_amp_mv=0.8)
    base, fb = _pr_baseline(sig, fid, int(fid["R_off"][0]), FS)
    assert not fb, "PR segment available -> no fallback"
    assert np.max(np.abs(base)) < 1e-9, "nonzero P wave must NOT bias the PR-segment baseline"
    dev, _ = st_deviation(sig, FS, fid=fid)
    assert np.max(np.abs(dev)) < 1e-9, "flat ST with large P wave -> ~0 deviation"


def test_known_st_elevation_recovered():
    for lvl in (0.20, -0.20, 0.35):
        sig, fid = _synth(st_level_mv=lvl, p_amp_mv=0.6)
        dev, fb = st_deviation(sig, FS, fid=fid)
        assert abs(float(np.median(dev)) - lvl) < 1e-6, f"ST {lvl} mV must be recovered"
        assert fb == 0.0, "no fallback when P_off present"


def test_fallback_when_no_p_offset():
    sig, fid = _synth(st_level_mv=0.1)
    fid["P_off"] = np.array([], dtype=int)     # remove P offsets -> force fallback
    base, fbflag = _pr_baseline(sig, fid, int(fid["R_off"][0]), FS)
    assert fbflag, "missing P_offset must trigger the fallback window"
    dev, frac = st_deviation(sig, FS, fid=fid)
    assert frac == 1.0, "all beats used the fallback"
    # fallback window (20 ms pre-QRS) is still isoelectric here -> ST recovered
    assert abs(float(np.median(dev)) - 0.1) < 1e-6
