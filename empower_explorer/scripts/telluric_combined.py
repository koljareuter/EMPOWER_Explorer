#!/usr/bin/env python
"""Telluric correction of already COMBINED KMOS cubes (DATACUBE_COMBINED_*.fits).

A combined cube is the (k-sigma clipped) mean of many single frames, often from
several OBs and nights.  Each OB has its own TELLURIC_*.fits (one transmission
spectrum per IFU), each frame its own barycentric velocity (the frames were
resampled to the barycentric frame by pyspark rvcorr before combining) and its
own airmass / water vapour.  The correct transmission for a combined cube is
therefore the mean of the per-frame transmissions:

    T_eff(lambda) = sum_i w_i T_i(lambda) / sum_i w_i
    T_i  = transmission of the IFU that observed this galaxy in frame i, from the
           TELLURIC file of frame i's OB, shifted by frame i's barycentric
           velocity exactly like the data (rvcorr_cube)
    w_i  = on-source time of frame i

Frames are identified from the PROVn keywords of the combined cube; the IFU per
frame from that frame's OCS ARMn NAME (= target id) and whether that IFU holds
data.  DATA and STAT are both divided by the transmission.

What the TELLURIC files are (docs/telluric-correction-v2.md)
  IFU.n.DATA = static DRP response RESPONSE_<B>_2021 (IFU n) x a noise-free MODEL atmosphere, one template per
  detector per night, its depth computed for the OB's first science frame.  No stellar lines, no noise.

Methods
  best        (default) the improvements of October 2026, validated on 16 bright in-field stars:
              * wavelength alignment (IZ, YJ): the atmospheric template of each (standard night, detector) is offset
                by -31..+45 km/s from the science (OH-calibrated) wavelength scale; the offset is measured against
                the DRP model kmos_atmos_<b> in fixed windows, minus the science reference SCI_REF, and removed when
                the frame's T is moved into the barycentric frame.  H: plain rv (gain not shown there).
              * line-depth factor k: tau = -ln(T_al / R_eff), tau_hf = tau - Gauss(tau, 6 px),
                T' = T_al * exp(-(k - 1) tau_hf).  k from the bright in-field stars of the same target list and band
                (recenter arms etc.) where they exist, else the band default (IZ 1.13, YJ 1.075, H 1).  Empirical
                line-contrast factor (in IZ mostly a line-width mismatch of the model, not proven extra water).
              * floor 0.02 decided on the plain rv T_eff (the old 0.1 cut H2O pixels with Halpha/[NII] at 1.135 um)
              * flux-scale keywords FLUXK (F_lambda = DATA * FLUXK) - the TELLURIC normalisation differs per band
              * TELLEFF file with T_RV / ATM / TAU / SIGTREL extensions and QC header values
  rv          T_i shifted by the frame's barycentric velocity (version 1 default)
  noshift     T_i unshifted (reference only)
  rv_airmass, rv_airm_iwv
              absorption scaled to the frame airmass (and IWV) relative to the STANDARD's header values - kept for
              reproducibility only: the model in the file is already computed for the OB's first frame, so the
              reference is wrong (they gave no gain, 2026-10-13)

Usage
  # 1. index the single frames once (headers only; cached as JSON)
  python scripts/telluric_combined.py index --frames kmos_ob_proc_server_BACKUP \
         --out telluric_frame_index.json
  # 2. correct all combined cubes of a folder into a NEW mirror folder
  python scripts/telluric_combined.py apply --comb kmos_comb_ohlin --index telluric_frame_index_ohlin.json \
         --tell telluric_all --out kmos_comb_ohlin_tell2 [--method best] [--bands YJ,IZ,H]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import warnings

import numpy as np
from astropy.io import fits

warnings.filterwarnings("ignore")
C_KMS = 299792.458
_TS = re.compile(r"KMOS\.(\d{4}-\d\d-\d\dT\d\d)[-:](\d\d)[-:](\d\d\.\d{3})")
DRY_BANDS = {"YJ": [(1.255, 1.285)], "H": [(1.565, 1.620), (1.630, 1.680)], "IZ": []}
STATIC_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "telluric_static")
BEST = "best"


def tkey(name):
    """Normalised timestamp key 'KMOS.YYYY-MM-DDTHH-MM-SS.sss' from any file/folder name."""
    m = _TS.search(str(name))
    return None if m is None else "KMOS.%s-%s-%s" % m.groups()


def wave_of(hdr, axis):
    n = hdr["NAXIS%d" % axis]
    d = hdr.get("CDELT%d" % axis) or hdr.get("CD%d_%d" % (axis, axis))
    return (np.arange(n) + 1 - hdr["CRPIX%d" % axis]) * d + hdr["CRVAL%d" % axis]


def _mean_kw(h, a, b):
    v = [h.get(a), h.get(b)]
    v = [float(x) for x in v if x is not None]
    return float(np.mean(v)) if v else None


# ── 1. frame index ────────────────────────────────────────────────────────────
def build_index(frame_bases, verbose=True):
    """{frame key: {...}} from all *_s_i_bkg2d_rv.fits under <base>/<OB>/science/."""
    idx = {}
    paths = []
    for base in frame_bases:
        paths += glob.glob(os.path.join(base, "*", "science", "*_s_i_bkg2d_rv.fits"))
    for n, p in enumerate(sorted(paths)):
        k = tkey(os.path.basename(p))
        ob = os.path.basename(os.path.dirname(os.path.dirname(p)))
        if k is None:
            continue
        with fits.open(p) as h:
            h0 = h[0].header
            ifus_data = []
            for i in range(1, min(25, len(h))):
                hd = h[i].header
                if hd.get("NAXIS", 0) == 3:
                    ifus_data.append(i)
            dit = h0.get("HIERARCH ESO DET SEQ1 DIT") or h0.get("HIERARCH ESO DET DIT")
            ndit = h0.get("HIERARCH ESO DET SEQ1 NDIT") or h0.get("HIERARCH ESO DET NDIT") or 1
            idx[k] = dict(
                path=p, ob=ob, ob_key=tkey(ob),
                band=(h0.get("HIERARCH ESO INS GRAT1 ID") or "").strip(),
                targ=(h0.get("HIERARCH ESO OBS TARG NAME") or "").strip(),
                vhelio=h0.get("HIERARCH ESO PYSPARK VHELIO"),
                airmass=_mean_kw(h0, "HIERARCH ESO TEL AIRM START", "HIERARCH ESO TEL AIRM END"),
                iwv=_mean_kw(h0, "HIERARCH ESO TEL AMBI IWV START", "HIERARCH ESO TEL AMBI IWV END"),
                mjd=h0.get("MJD-OBS"),
                texp=float(dit) * float(ndit) if dit else None,
                arms={str(i): str(h0.get("HIERARCH ESO OCS ARM%d NAME" % i, "")).strip()
                      for i in range(1, 25)},
                ifus_data=ifus_data)
        if verbose and (n + 1) % 100 == 0:
            print("  indexed %d/%d frames" % (n + 1, len(paths)), flush=True)
    return idx


# ── 2. telluric files ─────────────────────────────────────────────────────────
class TelluricStore:
    """Per-OB telluric transmission spectra, loaded lazily."""

    def __init__(self, tell_bases):
        self.paths = {}
        for base in tell_bases:
            cand = glob.glob(os.path.join(base, "*", "*TELLURIC*.fits")) or \
                glob.glob(os.path.join(base, "**", "*TELLURIC*.fits"), recursive=True)
            for t in cand:
                k = tkey(os.path.basename(t)) or tkey(t)
                band = os.path.basename(t).rsplit("_", 1)[-1].replace(".fits", "")
                if k:
                    self.paths[(k, band)] = t
        self._cache = {}

    def has(self, ob_key, band):
        return (ob_key, band) in self.paths

    def get(self, ob_key, band):
        key = (ob_key, band)
        if key not in self._cache:
            p = self.paths[key]
            T = {}
            with fits.open(p) as h:
                h0 = h[0].header
                wave = None
                for ifu in range(1, 25):
                    try:
                        e = h["IFU.%d.DATA" % ifu]
                    except KeyError:
                        continue
                    if e.data is None:
                        continue
                    wave = wave_of(e.header, 1) if wave is None else wave
                    T[ifu] = np.asarray(e.data, float)
            self._cache[key] = dict(
                T=T, wave=wave, path=p,
                airmass=_mean_kw(h0, "HIERARCH ESO TEL AIRM START", "HIERARCH ESO TEL AIRM END"),
                iwv=_mean_kw(h0, "HIERARCH ESO TEL AMBI IWV START", "HIERARCH ESO TEL AMBI IWV END"),
                env={})
        return self._cache[key]


def envelope(T, wave, absorb_cut=0.97, order=4, n_iter=6):
    """Smooth upper envelope (response x stellar continuum) of a transmission spectrum."""
    from scipy.ndimage import maximum_filter1d, median_filter
    ok = np.isfinite(T) & (T > 0)
    x = (wave - wave.mean()) / np.ptp(wave)
    hi = median_filter(maximum_filter1d(np.where(ok, T, 0.0), 41), 61)
    use = ok & (T > 0.9 * hi)
    env = None
    for _ in range(n_iter):
        env = np.polyval(np.polyfit(x[use], T[use], order), x)
        r = T / env
        use = ok & (r > absorb_cut) & (r < 1.0 + 2 * (1.0 - absorb_cut))
    return env


def to_rv_frame(spec, wave, vhelio):
    """Shift a telescope-frame spectrum exactly like pyspark rvcorr_cube shifts the data."""
    return np.interp(wave, wave * (1.0 + vhelio / C_KMS), spec, left=np.nan, right=np.nan)


def frame_transmission(tel, ifu, fr, band, method):
    """Transmission for one frame/IFU in the frame of the combined (rv) data (version-1 methods)."""
    T = tel["T"][ifu]
    wave = tel["wave"]
    if method in ("rv_airmass", "rv_airm_iwv"):
        if ifu not in tel["env"]:
            env = envelope(T, wave)
            tel["env"][ifu] = (env, np.clip(T / env, 1e-4, None))
        env, A = tel["env"][ifu]
        k_air = (fr["airmass"] / tel["airmass"]) if (fr.get("airmass") and tel.get("airmass")) else 1.0
        k_h2o = k_air
        if method == "rv_airm_iwv" and fr.get("iwv") and tel.get("iwv"):
            k_h2o = k_air * fr["iwv"] / tel["iwv"]
        dry = np.zeros(wave.shape, bool)
        for lo, hi in DRY_BANDS.get(band, []):
            dry |= (wave > lo) & (wave < hi)
        T = env * A ** np.where(dry, k_air, k_h2o)
    if method == "noshift":
        return T
    return to_rv_frame(T, wave, float(fr.get("vhelio") or 0.0))


# ── 3. method 'best': wavelength alignment of the standards' atmospheric template ─
ALIGN_BANDS = ("IZ", "YJ")
ALIGN_WINDOWS = {   # um, windows with model absorption (H CO2 1.56-1.625 excluded: two-valued fits)
    "IZ": [(0.890, 0.918), (0.918, 0.940), (0.940, 0.962), (0.962, 0.992)],
    "YJ": [(1.105, 1.128), (1.128, 1.150), (1.150, 1.180), (1.180, 1.215), (1.255, 1.285),
           (1.295, 1.322), (1.322, 1.345)],
    "H": [(1.450, 1.480), (1.480, 1.520), (1.720, 1.760), (1.760, 1.800), (1.800, 1.835)],
}
# science (OH-calibrated) minus kmos_atmos model per window, km/s, from the 16 in-field stars; re-measure if the
# science wavelength calibration changes (e.g. after a new OH re-reduction)
SCI_REF = {"IZ": [2.5, 2.7, 2.8, 6.0], "YJ": [1.4, -0.6, 1.0, -1.5, -1.9, -1.0, -0.2],
           "H": [-0.4, 3.1, 0.2, 1.6, 2.2]}
SCI_REF_VERSION = "2026-10-24 (kmos_comb_ohlin, 16 stars)"
DV_GRID = np.arange(-100.0, 100.1, 2.0)
SIG_GRID = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])   # extra Gaussian LSF sigma on the native model, km/s
MAX_MAD = 3.0          # km/s, IFU spread of a window inside one detector
MIN_IFUS = 4
QC_SPAN, QC_JUMP = 50.0, 30.0      # km/s: flag offset curves with a larger span / neighbour jump


class AtmosModel:
    """DRP model atmosphere kmos_atmos_<b>.fits (vacuum), native and smoothed by an extra Gaussian sigma."""

    def __init__(self, static_dir):
        self.dir = static_dir
        self._raw, self._sig = {}, {}

    def raw(self, band):
        if band not in self._raw:
            with fits.open(os.path.join(self.dir, "kmos_atmos_%s.fits" % band.lower())) as h:
                e = h["SPEC"]
                m = np.asarray(e.data, float)
                mw = (np.arange(m.size) + 1 - e.header.get("CRPIX1", 1.0)) * e.header["CDELT1"] + e.header["CRVAL1"]
            self._raw[band] = (mw, m)
        return self._raw[band]

    def get(self, band, sig):
        from scipy.ndimage import gaussian_filter1d
        key = (band, float(sig))
        if key not in self._sig:
            mw, m = self.raw(band)
            self._sig[key] = (mw, gaussian_filter1d(m, sig / C_KMS * np.median(mw) / np.median(np.diff(mw))))
        return self._sig[key]

    def at_kmos(self, band, wave, res={"IZ": 3400, "YJ": 3600, "H": 4000}):
        """Model convolved to KMOS resolution on 'wave' (NaN outside the model range)."""
        from scipy.ndimage import gaussian_filter1d
        mw, m = self.raw(band)
        sig_pix = (np.median(mw) / res.get(band, 3500) / 2.3548) / np.median(np.diff(mw))
        return np.interp(wave, mw, gaussian_filter1d(m, sig_pix), left=np.nan, right=np.nan)


def fit_shift(spec, wave, band, lo, hi, atm, order=3, min_model=0.03):
    """dv (km/s) of the absorption in spec vs the model in [lo, hi]:
       ln T = Legendre_order(lambda) + a * ln M_sigma(lambda/(1+dv/c)), grid in (dv, sigma), parabolic refinement.
       None if < 40 usable pixels, at the grid edge or not a minimum."""
    sel = (wave > lo) & (wave < hi) & np.isfinite(spec) & (spec > 0)
    if sel.sum() < 40:
        return None
    x, y = wave[sel], np.log(spec[sel])
    B = np.polynomial.legendre.legvander((x - x.mean()) / (0.5 * np.ptp(x)), order)
    best = None
    for sg in SIG_GRID:
        mw, mm = atm.get(band, sg)
        Mx = np.interp(x[None, :] / (1.0 + DV_GRID[:, None] / C_KMS), mw, mm, left=np.nan, right=np.nan)
        ok = np.all(np.isfinite(Mx), 0) & np.all(Mx > min_model, 0)
        if ok.sum() < 40:
            continue
        rel = np.exp(y - np.polyval(np.polyfit(x[ok], y[ok], 2), x))
        sw = np.where(ok, np.clip(rel, 0.05, 1.5), 0.0)          # down-weight saturated cores
        Q, _ = np.linalg.qr(B * sw[:, None])
        yp = y * sw - Q @ (Q.T @ (y * sw))
        L = np.log(np.where(np.isfinite(Mx) & (Mx > 0), Mx, 1.0)) * sw[None, :]
        Lp = L - (L @ Q) @ Q.T
        num = Lp @ yp
        den = np.einsum("ij,ij->i", Lp, Lp)
        chi2 = yp @ yp - num ** 2 / np.where(den > 0, den, np.nan)
        i = int(np.nanargmin(chi2))
        if best is None or chi2[i] < best[0]:
            best = (chi2[i], chi2, i)
    if best is None:
        return None
    _, chi2, i = best
    if not 0 < i < len(DV_GRID) - 1:
        return None
    c0, c1, c2 = chi2[i - 1], chi2[i], chi2[i + 1]
    a = 0.5 * (c0 + c2 - 2 * c1)
    if not a > 0:
        return None
    return float(DV_GRID[i] - 0.5 * (c2 - c0) / (2 * a) * (DV_GRID[1] - DV_GRID[0]))


class StdOffsets:
    """Per TELLURIC file: {det: {k: [median dv, MAD, n_ifu]}} against the model, cached in a JSON file
    (key: basename + md5 of IFU.1.DATA + fingerprint of the fit configuration and model), ~0.5 s per file uncached."""

    def __init__(self, atm, cache_json=None):
        self.atm = atm
        self.cache_json = cache_json
        self.cache = {}
        self._fp = {}
        if cache_json and os.path.exists(cache_json):
            try:
                with open(cache_json) as fh:
                    self.cache = json.load(fh)
            except (ValueError, OSError) as err:
                print("  WARNING: unreadable offset cache %s (%s) - recomputing" % (cache_json, err))

    def fingerprint(self, band):
        if band not in self._fp:
            mw, m = self.atm.raw(band)
            h = hashlib.md5(json.dumps([ALIGN_WINDOWS[band], SIG_GRID.tolist(), DV_GRID.tolist(), 0.03]).encode())
            h.update(np.ascontiguousarray(m).tobytes())
            self._fp[band] = h.hexdigest()[:8]
        return self._fp[band]

    def get(self, tel, band):
        T, wave = tel["T"], tel["wave"]
        first = T[min(T)]
        key = "%s|%s|%s" % (os.path.basename(tel["path"]),
                            hashlib.md5(np.ascontiguousarray(first).tobytes()).hexdigest()[:12], self.fingerprint(band))
        if key not in self.cache:
            out = {}
            for det in (1, 2, 3):
                ifus = [i for i in range(8 * det - 7, 8 * det + 1) if i in T]
                dd = {}
                for k, (lo, hi) in enumerate(ALIGN_WINDOWS[band]):
                    v = np.array([x for x in (fit_shift(T[i], wave, band, lo, hi, self.atm) for i in ifus)
                                  if x is not None])
                    if v.size:
                        m = float(np.median(v))
                        dd[str(k)] = [m, float(1.4826 * np.median(np.abs(v - m))), int(v.size)]
                out[str(det)] = dd
            self.cache[key] = out
            if self.cache_json:          # atomic rewrite: an interrupted run cannot leave a broken cache
                os.makedirs(os.path.dirname(os.path.abspath(self.cache_json)), exist_ok=True)
                tmp = self.cache_json + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump(self.cache, fh)
                os.replace(tmp, self.cache_json)
        return self.cache[key]


def dv_curves(off, band, sci_ref=None):
    """({det: (lam_k, d_k)}, {det: [qc messages]}) with d_k = dv_k - SCI_REF_k over the accepted windows
    (IFU MAD < 3 km/s, >= 4 IFUs).  QC: span > 50 km/s, neighbour jump > 30 km/s, interpolation across a rejected
    window.  A detector without accepted window gets no curve (plain rv)."""
    ref = SCI_REF[band] if sci_ref is None else sci_ref
    out, qc = {}, {}
    for det, dd in off.items():
        ks = sorted(int(k) for k, v in dd.items() if v[1] < MAX_MAD and v[2] >= MIN_IFUS)
        msg = []
        if ks:
            lam = np.array([0.5 * sum(ALIGN_WINDOWS[band][k]) for k in ks])
            d = np.array([dd[str(k)][0] - ref[k] for k in ks])
            out[int(det)] = (lam, d)
            if np.ptp(d) > QC_SPAN:
                msg.append("span %.0f km/s" % np.ptp(d))
            if d.size > 1 and np.max(np.abs(np.diff(d))) > QC_JUMP:
                msg.append("jump %.0f km/s" % np.max(np.abs(np.diff(d))))
            if ks[-1] - ks[0] + 1 > len(ks):
                msg.append("interpolated across rejected window(s)")
        else:
            msg.append("no accepted window (plain rv)")
        if msg:
            qc[int(det)] = msg
    return out, qc


def shift_to_frame(spec, wave, vhelio, d=0.0):
    """Move a standard spectrum into the barycentric frame of one science frame, removing the standard's
    wavelength offset d(lambda) (km/s, > 0 = standard too red).  d = 0 is exactly to_rv_frame."""
    dst = wave * (1.0 + vhelio / C_KMS) / (1.0 + np.asarray(d) / C_KMS)
    o = np.argsort(dst)
    return np.interp(wave, dst[o], spec[o], left=np.nan, right=np.nan)


# ── 4. method 'best': line-depth factor k from in-field stars ─────────────────
K_HF_DEFAULT = {"IZ": 1.13, "YJ": 1.075, "H": 1.0}   # median of the in-field run values (2026-10-24)
K_RANGE = (1.0, 1.25)
K_BANDS = ("IZ", "YJ")                 # H: k = 1 (no gain shown)
SIG_LF = 6.0                           # px: tau_hf = tau - Gauss(tau, SIG_LF)
CALIB_RE = r"^(recenter\d+|R_2[89]|H12[89]R)$"   # bright in-field stars (target ids) of this data set
MIN_FRAME_OVERLAP = 0.9


def split_tau(T, R, sig_lf=SIG_LF):
    """tau = -ln(T/R) (>= 0, 0 where unknown), its smooth part (Gauss sig_lf px) and the line-scale rest."""
    from scipy.ndimage import gaussian_filter1d
    with np.errstate(invalid="ignore", divide="ignore"):
        A = T / R
    tau = -np.log(np.clip(A, 1e-3, None))
    tau = np.where(np.isfinite(tau), np.clip(tau, 0, None), 0.0)
    lf = gaussian_filter1d(tau, sig_lf, mode="nearest")
    return tau, lf, tau - lf, A


def apply_khf(T, R, k_hf):
    return T * np.exp(-(k_hf - 1.0) * split_tau(T, R)[2])


def _bspline_basis(wave, dk):
    from scipy.interpolate import BSpline
    lo, hi = wave[0] - 1e-6, wave[-1] + 1e-6
    n = max(int(np.ceil((hi - lo) / dk)), 1)
    t = np.r_[[lo] * 3, np.linspace(lo, hi, n + 1), [hi] * 3]
    return BSpline.design_matrix(np.clip(wave, lo, hi), t, 3).toarray()


def fit_k(stars, dk_cont=0.05, s0=0.005, kappa=0.05, clip=4.0, n_iter=6):
    """Joint robust fit over the stars of one run:
       ln(spec/T_al) = own B-spline continuum - (k_lf-1) tau_lf - (k_hf-1) tau_hf,
       weights 1/(sigma^2 + s0^2 + kappa^2 tau), iterative clipping (stellar lines, OH, cosmics).
       Only k_hf is used (k_lf is degenerate with the continuum).  stars: [dict(spec, err, T, R, wave)]."""
    rows_X, rows_y, rows_w, n_own = [], [], [], []
    blocks = []
    for s in stars:
        tau, lf, hf, A = split_tau(s["T"], s["R"])
        with np.errstate(invalid="ignore", divide="ignore"):
            y = np.log(s["spec"] / s["T"])
            sy = s["err"] / s["spec"]
        ok = np.isfinite(y) & np.isfinite(sy) & (s["spec"] > 0) & np.isfinite(s["T"]) & (A > 0.05)
        B = _bspline_basis(s["wave"], dk_cont)
        B = B[:, B[ok].sum(0) > 3]
        blocks.append((y, B, np.vstack([-lf, -hf]).T, sy ** 2 + s0 ** 2 + kappa ** 2 * tau, ok))
        n_own.append(B.shape[1])
    ncol = sum(n_own) + 2
    off = 0
    for (y, B, Xk, var, ok), no in zip(blocks, n_own):
        X = np.zeros((ok.sum(), ncol))
        X[:, off:off + no] = B[ok]
        X[:, -2:] = Xk[ok]
        rows_X.append(X); rows_y.append(y[ok]); rows_w.append(1.0 / var[ok])
        off += no
    X, y, w = np.vstack(rows_X), np.concatenate(rows_y), np.concatenate(rows_w)
    keep = np.ones(y.size, bool)
    s = 1.0
    for _ in range(n_iter):
        sw = np.sqrt(w * keep)
        coef = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        r = (y - X @ coef) * np.sqrt(w)
        s = 1.4826 * np.median(np.abs(r[keep]))
        new = np.abs(r) < clip * s
        if np.array_equal(new, keep):
            break
        keep = new
    sw = np.sqrt(w * keep)
    cov = np.linalg.pinv((X * sw[:, None]).T @ (X * sw[:, None])) * s ** 2
    return float(1 + coef[-1]), float(np.sqrt(cov[-1, -1]))


def star_spectrum(cube, stat, box=2, ann_min=5):
    """5x5 aperture spectrum (and sigma) of a bright point source, with the per-spaxel additive offset of the
    pyspark bkg2d constant undone (outer annulus = alpha * core + beta -> add -beta per spaxel)."""
    col = np.nansum(cube[500:1500], 0)
    y, x = np.unravel_index(np.nanargmax(np.where(np.isfinite(col), col, -np.inf)), col.shape)
    yy, xx = np.mgrid[:cube.shape[1], :cube.shape[2]]
    rr = np.maximum(np.abs(yy - y), np.abs(xx - x))
    sa = np.nanmean(cube[:, (rr >= ann_min) & np.isfinite(col)], 1)
    sc = np.nanmean(cube[:, rr <= box], 1)
    ok = np.isfinite(sa) & np.isfinite(sc)
    coef = np.linalg.lstsq(np.vstack([sc[ok], np.ones(ok.sum())]).T, sa[ok], rcond=None)[0]
    sub = cube[:, max(0, y - 2):y + 3, max(0, x - 2):x + 3]
    cover = np.where(np.isfinite(sub).sum((1, 2)) > 0.6 * sub[0].size, 1.0, np.nan)   # planes without data -> NaN
    spec = (np.nansum(sub, (1, 2)) - coef[1] * sub[0].size) * cover
    err = np.sqrt(np.nansum(stat[:, max(0, y - 2):y + 3, max(0, x - 2):x + 3] ** 2, (1, 2))) * cover
    return spec, err


# ── 5. method 'best': first-frame conditions, flux scale, uncertainty terms, guard ─
SBAR_EXP = {"IZ": (0.65, 0.61), "YJ": (0.64, 0.62), "H": (0.66, 0.67)}   # depth ~ X^a IWV^b of the TELLURIC models
SBAR_FLAG = 0.05
SYS_A = 0.05            # SIGTREL = SYS_A * max(sqrt(tau), tau): frame-common (correlated) model error
SUBRES_C = 0.06         # sigma_F/F = SUBRES_C (80 km/s / sigma_int)^1.6 * tau(line): provisional
# flux scale: F_lambda [1e-17 erg/s/cm2/A] = DATA (corrected, ADU/s/pixel) * FLUXK; q = per-um band scale vs YJ from
# the same standard star observed in two bands (QC ZPOINT, MAG and the DRP band normalisation), K_YJ from the typical
# YJ zero point 17.461 and Vega J; validated on the in-field stars observed in IZ and YJ (0.98/1.00)
FLUX_Q = {"IZ": 0.430, "YJ": 1.0, "H": 1.874}
FLUX_K = {"IZ": 8.73, "YJ": 3.24, "H": 1.405}
FLUX_KERR = 0.10
XBAND_ERR = {"IZ": 0.06, "YJ": 0.0, "H": 0.05}
STL_LIMIT = 0.15        # Angstrom: stellar-line EW in T above which the standard must be cleaned first


def first_frame_conditions(index):
    """{(ob_key, band): (airmass, iwv)} of the OB's first science frame (the index entry named like the OB)."""
    by_ob = {}
    for k, fr in index.items():
        by_ob.setdefault((fr["ob_key"], fr["band"]), []).append((k, fr))
    out = {}
    for key, lst in by_ob.items():
        f0 = dict(lst).get(key[0]) or sorted(lst, key=lambda kv: kv[1].get("mjd") or 0)[0][1]
        out[key] = (f0.get("airmass"), f0.get("iwv"))
    return out


_RH = 10.9677583       # Rydberg constant for hydrogen, 1/um (vacuum)


def _paschen(n):
    return 1.0 / (_RH * (1 / 9 - 1 / n ** 2))


def _brackett(n):
    return 1.0 / (_RH * (1 / 16 - 1 / n ** 2))


STELLAR_GUARD_LINES = {   # vacuum um; lines a B/A/G standard would leave and that sit in clean windows
    "IZ": dict([("Pa%d" % n, _paschen(n)) for n in range(11, 18)] +
               [("Pa-delta", _paschen(7)), ("CaII8544", 0.85445), ("CaII8665", 0.86625)]),
    "YJ": {"SiI10588": 1.05881, "KI12436": 1.24357},
    "H": {"Br14": _brackett(14), "Br16": _brackett(16)},
}


def stellar_line_check(T, wave, band, atm_kmos, res={"IZ": 3400, "YJ": 3600, "H": 4000}, rv_window=150.0):
    """Max |EW| (Angstrom) of intrinsic stellar lines in a transmission spectrum, windows +-(2 FWHM + 150 km/s)
    where the model atmosphere is > 0.98.  ~0.01-0.1 on the current model files; real stellar lines 0.5-10."""
    dw = np.median(np.diff(wave)) * 1e4
    worst = 0.0
    for lc in STELLAR_GUARD_LINES.get(band, {}).values():
        if not (wave[0] + 0.005 < lc < wave[-1] - 0.005):
            continue
        fwhm = lc / res.get(band, 3500)
        half = 2 * fwhm + lc * rv_window / C_KMS
        win = np.abs(wave - lc) < half
        side = (np.abs(wave - lc) > half) & (np.abs(wave - lc) < half + 3 * fwhm)
        if not np.nanmin(atm_kmos[win]) > 0.98:
            continue
        ok = side & np.isfinite(T)
        if ok.sum() < 4 or not np.isfinite(T[win]).all():
            continue
        p = np.polyfit(wave[ok] - lc, T[ok], 1)
        worst = max(worst, abs(float(np.sum(1 - T[win] / np.polyval(p, wave[win] - lc)) * dw)))
    return worst


class BestContext:
    """Shared state of method 'best': static files, standard offsets, responses, first-frame conditions."""

    def __init__(self, static_dir, index, align_cache=None):
        if not os.path.isdir(static_dir):
            raise IOError("static folder with kmos_atmos_<b>.fits and RESPONSE_<B>_2021.fits not found: %s"
                          % static_dir)
        self.static = static_dir
        self.atm = AtmosModel(static_dir)
        self.offsets = StdOffsets(self.atm, align_cache)
        self.cond0 = first_frame_conditions(index)
        self._resp, self._curves = {}, {}
        self.qc_log = {}

    def has_band(self, band):
        """Static response and model atmosphere available for this band (IZ, YJ, H)."""
        return band in K_HF_DEFAULT and os.path.isfile(os.path.join(self.static, "RESPONSE_%s_2021.fits" % band)) \
            and os.path.isfile(os.path.join(self.static, "kmos_atmos_%s.fits" % band.lower()))

    def resp(self, band):
        if band not in self._resp:
            p = os.path.join(self.static, "RESPONSE_%s_2021.fits" % band)
            with fits.open(p) as h:
                self._resp[band] = {i: np.asarray(h["IFU.%d.DATA" % i].data, float) for i in range(1, 25)}
        return self._resp[band]

    def d_curve(self, tel, band, ifu):
        """Standard wavelength offset d(lambda) (km/s) for this file and IFU's detector, 0 outside ALIGN_BANDS."""
        if band not in ALIGN_BANDS:
            return 0.0, None
        if tel["path"] not in self._curves:
            curves, qc = dv_curves(self.offsets.get(tel, band), band)
            self._curves[tel["path"]] = (curves, qc)
            for det, msg in qc.items():
                self.qc_log["%s det%d" % (os.path.basename(tel["path"]), det)] = msg
        curves, qc = self._curves[tel["path"]]
        det = (ifu - 1) // 8 + 1
        d = np.interp(tel["wave"], *curves[det]) if det in curves else 0.0
        return d, qc.get(det)


def best_transmissions(h0, index, store, wave, ctx):
    """For one combined cube: texp-weighted T_rv (plain rv), T_al (aligned), R_al (response, same shifts), sbar
    (mean model-depth ratio frame/first frame), frame keys and info.  Same frames and 50 %-weight coverage rule as
    effective_transmission."""
    band = (h0.get("HIERARCH ESO INS GRAT1 ID") or "").strip()
    tid = str(h0.get("HIERARCH ESO PRO REFLEX SUFFIX") or "").strip()
    prov = [h0[k] for k in h0 if re.fullmatch(r"PROV\d+", k)]
    resp = ctx.resp(band)
    acc, den = None, 0.0
    s_num = 0.0
    n_used, n_noframe, n_notarget, n_nodata, n_notell, n_qc = 0, 0, 0, 0, 0, 0
    used_obs, frames = set(), set()
    for p in prov:
        k = tkey(p)
        fr = index.get(k)
        if fr is None:
            n_noframe += 1
            continue
        ifus = [int(i) for i, nm in fr["arms"].items() if nm == tid]
        if not ifus:
            n_notarget += 1
            continue
        ifu = ifus[0]
        if ifu not in fr["ifus_data"]:
            n_nodata += 1
            continue
        if not store.has(fr["ob_key"], band):
            n_notell += 1
            continue
        tel = store.get(fr["ob_key"], band)
        if ifu not in tel["T"]:
            n_notell += 1
            continue
        if len(tel["wave"]) != len(wave) or abs(tel["wave"][0] - wave[0]) > 1e-6:
            raise ValueError("wavelength grid of %s differs from the combined cube" % tel["path"])
        v = float(fr.get("vhelio") or 0.0)
        d, qc = ctx.d_curve(tel, band, ifu)
        n_qc += qc is not None
        arrs = dict(rv=to_rv_frame(tel["T"][ifu], tel["wave"], v),
                    al=shift_to_frame(tel["T"][ifu], tel["wave"], v, d),
                    r=shift_to_frame(resp[ifu], tel["wave"], v, d))
        w = fr.get("texp") or 1.0
        if acc is None:
            acc = {n: [np.zeros_like(wave, dtype=float), np.zeros_like(wave, dtype=float)] for n in arrs}
        for n, a in arrs.items():
            g = np.isfinite(a)
            acc[n][0][g] += w * a[g]
            acc[n][1][g] += w
        x0, i0 = ctx.cond0.get((fr["ob_key"], band), (None, None))
        a_x, a_w = SBAR_EXP.get(band, (0.65, 0.62))
        s = 1.0
        if fr.get("airmass") and x0:
            s *= (fr["airmass"] / x0) ** a_x
        if fr.get("iwv") and i0:
            s *= (fr["iwv"] / i0) ** a_w
        s_num += w * s
        den += w
        n_used += 1
        used_obs.add(fr["ob_key"])
        frames.add(k)
    info = dict(band=band, tid=tid, n_prov=len(prov), n_used=n_used, n_noframe=n_noframe,
                n_notarget=n_notarget, n_nodata=n_nodata, n_notell=n_notell, obs=sorted(used_obs), n_qc=n_qc)
    contributing = n_used + n_notell
    info["coverage"] = n_used / contributing if contributing else 0.0
    if acc is None:
        return None, info
    out = dict(frames=frozenset(frames), sbar=s_num / den)
    with np.errstate(invalid="ignore", divide="ignore"):
        for n, (num, da) in acc.items():
            out[n] = np.where(da > 0.5 * den, num / da, np.nan)
    return out, info


# ── 6. effective transmission of one combined cube (version-1 methods) ───────
def effective_transmission(h0, index, store, method="rv", wave=None):
    """(T_eff, info) for a combined cube with primary header h0 (methods noshift, rv, rv_airmass, rv_airm_iwv)."""
    band = (h0.get("HIERARCH ESO INS GRAT1 ID") or "").strip()
    tid = str(h0.get("HIERARCH ESO PRO REFLEX SUFFIX") or "").strip()
    prov = [h0[k] for k in h0 if re.fullmatch(r"PROV\d+", k)]
    num, den = None, 0.0
    n_used, n_noframe, n_notarget, n_nodata, n_notell = 0, 0, 0, 0, 0
    used_obs = set()
    for p in prov:
        k = tkey(p)
        fr = index.get(k)
        if fr is None:
            n_noframe += 1
            continue
        ifus = [int(i) for i, nm in fr["arms"].items() if nm == tid]
        if not ifus:
            n_notarget += 1
            continue
        ifu = ifus[0]
        if ifu not in fr["ifus_data"]:
            n_nodata += 1
            continue
        if not store.has(fr["ob_key"], band):
            n_notell += 1
            continue
        tel = store.get(fr["ob_key"], band)
        if ifu not in tel["T"]:
            n_notell += 1
            continue
        if wave is not None and (len(tel["wave"]) != len(wave) or abs(tel["wave"][0] - wave[0]) > 1e-6):
            raise ValueError("wavelength grid of %s differs from the combined cube" % tel["path"])
        Ti = frame_transmission(tel, ifu, fr, band, method)
        w = fr.get("texp") or 1.0
        good = np.isfinite(Ti)
        if num is None:
            num = np.zeros_like(Ti)
            den_arr = np.zeros_like(Ti)
        num[good] += w * Ti[good]
        den_arr[good] += w
        den += w
        n_used += 1
        used_obs.add(fr["ob_key"])
    info = dict(band=band, tid=tid, n_prov=len(prov), n_used=n_used, n_noframe=n_noframe,
                n_notarget=n_notarget, n_nodata=n_nodata, n_notell=n_notell, obs=sorted(used_obs))
    contributing = n_used + n_notell          # frames that really hold this galaxy
    info["coverage"] = n_used / contributing if contributing else 0.0
    if num is None:
        return None, info
    with np.errstate(invalid="ignore", divide="ignore"):
        T_eff = np.where(den_arr > 0.5 * den, num / den_arr, np.nan)
    return T_eff, info


# ── 7. apply to a folder of combined cubes ────────────────────────────────────
def _teff_name(dst):
    # the name must NOT start with DATACUBE_COMBINED_ or the notebooks would read it as a cube
    return os.path.join(os.path.dirname(dst), "TELLEFF_" + os.path.basename(dst).replace("DATACUBE_COMBINED_", ""))


def _list_cubes(comb_base, out_base, bands, overwrite, calib_re):
    """All input cubes of the selected bands: dict(f, rel, dst, band, tid, targ, write, calib).  write = output
    missing or overwrite; calib = bright in-field star (needed for k even when its output already exists)."""
    files = sorted(glob.glob(os.path.join(comb_base, "**", "DATACUBE_COMBINED_*.fits"), recursive=True))
    out = []
    for f in files:
        rel = os.path.relpath(f, comb_base)
        dst = os.path.join(out_base, rel)
        if "_SKY" in os.path.basename(f):
            continue
        h0 = fits.getheader(f, 0)
        band = (h0.get("HIERARCH ESO INS GRAT1 ID") or "").strip()
        if bands and band not in bands:
            continue
        tid = str(h0.get("HIERARCH ESO PRO REFLEX SUFFIX") or "").strip()
        out.append(dict(f=f, rel=rel, dst=dst, band=band, tid=tid,
                        targ=(h0.get("HIERARCH ESO OBS TARG NAME") or "").strip(),
                        write=overwrite or not os.path.exists(dst), calib=bool(re.match(calib_re, tid))))
    return out


def apply_folder(comb_base, out_base, index, store, method=BEST, floor=None, min_coverage=0.999,
                 bands=None, overwrite=False, verbose=True, static=STATIC_DEFAULT, calib_re=CALIB_RE,
                 align_cache=None, stellar_guard=True):
    if method != BEST:
        return _apply_folder_v1(comb_base, out_base, index, store, method, 0.1 if floor is None else floor,
                                min_coverage, bands, overwrite, verbose)
    floor = 0.02 if floor is None else floor
    log = []
    ctx = BestContext(static, index, align_cache or os.path.join(out_base, "telluric_align_offsets.json"))
    cubes = _list_cubes(comb_base, out_base, bands, overwrite, calib_re)
    runs_to_write = {(c["targ"], c["band"]) for c in cubes if c["write"]}
    for c in cubes:
        if not c["write"]:
            log.append((c["rel"], "exists"))

    # pass 1: transmissions of the cubes to write and of the calibrator stars of their runs (k is a per-run quantity,
    # so the calibrators enter even when their own output already exists)
    entries = []
    for c in cubes:
        if not (c["write"] or (c["calib"] and (c["targ"], c["band"]) in runs_to_write)):
            continue
        if not ctx.has_band(c["band"]):
            log.append((c["rel"], "SKIPPED band %r (no static response / model atmosphere)" % c["band"]))
            if verbose and c["write"]:
                print("  skip %-70s band %r not supported by method best" % (c["rel"][-70:], c["band"]))
            continue
        with fits.open(c["f"]) as h:
            h0 = h[0].header.copy()
            wave = wave_of(h["DATA"].header, 3)
        tr, info = best_transmissions(h0, index, store, wave, ctx)
        if tr is None or info["coverage"] < min_coverage:
            if c["write"]:
                log.append((c["rel"], "SKIPPED coverage %.2f %s" % (info["coverage"], info)))
                if verbose:
                    print("  skip %-70s coverage %.2f (frames used %d, no telluric %d, not found %d)" % (
                        c["rel"][-70:], info["coverage"], info["n_used"], info["n_notell"], info["n_noframe"]))
            continue
        entries.append(dict(c, wave=wave, tr=tr, info=info))

    # k per run (target list x band) from its bright in-field stars
    runs = {}
    for e in entries:
        runs.setdefault((e["targ"], e["band"]), []).append(e)
    run_k = {}
    for key, members in runs.items():
        band = key[1]
        cals = [e for e in members if e["calib"]]
        if band not in K_BANDS or not cals:
            continue
        stars = []
        try:
            for e in cals:
                with fits.open(e["f"]) as h:
                    cube = np.asarray(h["DATA"].data, float)
                    stat = np.asarray(h["STAT"].data, float)
                spec, err = star_spectrum(cube, stat)
                stars.append(dict(spec=spec, err=err, T=e["tr"]["al"], R=e["tr"]["r"], wave=e["wave"]))
            k, kerr = fit_k(stars)
        except (KeyError, ValueError, np.linalg.LinAlgError) as err:       # no STAT, no usable pixels ...
            print("  k(%s, %s): in-field fit failed (%s) -> band default" % (key[0], band, err))
            continue
        k0 = K_HF_DEFAULT[band]
        ok = K_RANGE[0] <= k <= K_RANGE[1] and abs(k - k0) <= 3 * max(kerr, 0.01)
        frames = frozenset().union(*[e["tr"]["frames"] for e in cals])
        run_k[key] = (k, kerr, [e["tid"] for e in cals], frames, ok)
        if verbose:
            print("  k(%s, %s) = %.3f +- %.3f from %s%s" % (key[0], band, k, kerr, ",".join(e["tid"] for e in cals),
                                                         "" if ok else "  -> QC failed, band default %.3f" % k0))

    # pass 2: apply and write (calibrators that only entered for k are not rewritten)
    for e in entries:
        if not e["write"]:
            continue
        band, tr = e["band"], e["tr"]
        k, kerr, ksrc = K_HF_DEFAULT.get(band, 1.0), np.nan, "band default"
        rk = run_k.get((e["targ"], band))
        if rk is not None and rk[4]:
            overlap = len(tr["frames"] & rk[3]) / max(len(tr["frames"]), 1)
            if overlap >= MIN_FRAME_OVERLAP:
                k, kerr, ksrc = rk[0], rk[1], "in-field " + ",".join(rk[2])
        T_al = np.where(np.isfinite(tr["al"]), tr["al"], tr["rv"])
        R_al = np.where(np.isfinite(tr["r"]), tr["r"], np.nan)
        T_fin = apply_khf(T_al, R_al, k) if k != 1.0 else T_al
        T_fin = np.where(np.isfinite(T_fin), T_fin, T_al)
        cut = ~(tr["rv"] > floor)                   # floor on the plain rv T_eff only, never after alignment / k
        Tc = np.where(cut, np.nan, T_fin)
        with np.errstate(invalid="ignore", divide="ignore"):
            atm = T_fin / R_al
        tau = np.where(np.isfinite(atm), np.clip(-np.log(np.clip(atm, 1e-6, None)), 0, None), np.nan)
        sigtrel = np.sqrt((SYS_A * np.maximum(np.sqrt(tau), tau)) ** 2 + (abs(tr["sbar"] - 1.0) * tau) ** 2)
        stl = stellar_line_check(T_fin, e["wave"], band, ctx.atm.at_kmos(band, e["wave"]))
        if stellar_guard and stl > STL_LIMIT:
            raise RuntimeError("%s: stellar line of %.2f A in the transmission - the TELLURIC files contain a real "
                               "star spectrum; clean the standards first (or --no-stellar-guard)" % (e["rel"], stl))
        with fits.open(e["f"]) as h:
            h["DATA"].data = np.asarray(h["DATA"].data / Tc[:, None, None], np.float32)
            if "STAT" in h:
                h["STAT"].data = np.asarray(h["STAT"].data / Tc[:, None, None], np.float32)
            hh = h[0].header
            info = e["info"]
            hh["TELLMETH"] = (BEST, "telluric method (scripts/telluric_combined.py)")
            hh["TELLALGN"] = ("SCI_REF " + SCI_REF_VERSION if band in ALIGN_BANDS else "none",
                              "wavelength alignment of the standards")
            hh["TELLAQC"] = (info["n_qc"], "frames whose offset curve has a QC flag")
            hh["TELLKHF"] = (round(k, 4), "line-scale optical-depth factor k")
            hh["TELLKERR"] = (round(kerr, 4) if np.isfinite(kerr) else -1.0, "error of k (-1: band default)")
            hh["TELLKSRC"] = (ksrc[:68], "source of k")
            hh["TELLSBAR"] = (round(tr["sbar"], 4), "mean model depth ratio frame/first frame")
            hh["TELLSFLG"] = (bool(abs(tr["sbar"] - 1) > SBAR_FLAG), "|TELLSBAR-1| > 0.05: depth unvalidated")
            hh["TELLNFR"] = (info["n_used"], "frames contributing to T_eff")
            hh["TELLNOB"] = (len(info["obs"]), "OBs (standard stars) in T_eff")
            hh["TELLCOV"] = (round(info["coverage"], 3), "fraction of frames with telluric")
            hh["TELLFLOR"] = (floor, "plain-rv T_eff <= floor set to NaN")
            hh["TELLSTL"] = (round(stl, 3), "max stellar-line EW in T [A] (guard)")
            hh["FLUXK"] = (FLUX_K.get(band, -1.0), "F_lambda[1e-17 erg/s/cm2/A] = DATA * FLUXK")
            hh["FLUXQ"] = (FLUX_Q.get(band, -1.0), "per-um band scale rel. to YJ")
            hh["FLUXKERR"] = (FLUX_KERR, "rel. absolute uncertainty of FLUXK")
            hh["XBANDERR"] = (XBAND_ERR.get(band, -1.0), "rel. cross-band (vs YJ) uncertainty")
            hh["HISTORY"] = "Telluric corrected (%s, k=%.3f %s) from %d frames of %d OBs" % (
                BEST, k, ksrc[:30], info["n_used"], len(info["obs"]))
            os.makedirs(os.path.dirname(e["dst"]), exist_ok=True)
            h.writeto(e["dst"], overwrite=True)
        p = fits.PrimaryHDU(np.asarray(T_fin, np.float32))
        p.header["TELLMETH"] = (BEST, "primary = transmission divided out")
        p.header["SYS_A"] = (SYS_A, "SIGTREL=sqrt((SYS_A*max(sqrt(TAU),TAU))^2+((SBAR-1)TAU)^2)")
        p.header["SBAR"] = (round(tr["sbar"], 4), "mean model depth ratio frame/first frame")
        p.header["SUBRES_C"] = (SUBRES_C, "provisional: sigma_F/F=c(80/sig_kms)^1.6*TAU_line")
        hdus = [p]
        for name, arr, com in [("T_RV", tr["rv"], "plain rv T_eff (version 1, floor reference)"),
                               ("ATM", atm, "atmospheric part: primary / static response"),
                               ("TAU", tau, "-ln ATM (>= 0)"),
                               ("SIGTREL", sigtrel, "rel. telluric model error, frame-common/correlated")]:
            x = fits.ImageHDU(np.asarray(arr, np.float32), name=name)
            x.header["COMMENT"] = com
            hdus.append(x)
        fits.HDUList(hdus).writeto(_teff_name(e["dst"]), overwrite=True)
        log.append((e["rel"], "ok %d frames, %d OBs, k %.3f (%s)" % (info["n_used"], len(info["obs"]), k, ksrc)))
        if verbose:
            print("  ok   %-62s %3d fr %2d OB k %.3f %s" % (e["rel"][-62:], info["n_used"], len(info["obs"]), k,
                                                         "infield" if ksrc.startswith("in") else "default"), flush=True)
    for name, msg in sorted(ctx.qc_log.items()):
        print("  QC offset curve %s: %s" % (name, "; ".join(msg)))
    return log


def _apply_folder_v1(comb_base, out_base, index, store, method="rv", floor=0.1, min_coverage=0.999,
                     bands=None, overwrite=False, verbose=True):
    files = sorted(glob.glob(os.path.join(comb_base, "**", "DATACUBE_COMBINED_*.fits"), recursive=True))
    log = []
    for f in files:
        rel = os.path.relpath(f, comb_base)
        dst = os.path.join(out_base, rel)
        with fits.open(f) as h:
            h0 = h[0].header
            band = (h0.get("HIERARCH ESO INS GRAT1 ID") or "").strip()
            if bands and band not in bands:
                continue
            if "_SKY" in os.path.basename(f):
                continue
            if os.path.exists(dst) and not overwrite:
                log.append((rel, "exists"))
                continue
            wave = wave_of(h["DATA"].header, 3)
            T, info = effective_transmission(h0, index, store, method, wave)
            if T is None or info["coverage"] < min_coverage:
                log.append((rel, "SKIPPED coverage %.2f %s" % (info["coverage"], info)))
                if verbose:
                    print("  skip %-70s coverage %.2f (frames used %d, no telluric %d, not found %d)" % (
                        rel[-70:], info["coverage"], info["n_used"], info["n_notell"], info["n_noframe"]))
                continue
            Tc = T.copy()
            Tc[~(Tc > floor)] = np.nan
            h["DATA"].data = np.asarray(h["DATA"].data / Tc[:, None, None], np.float32)
            if "STAT" in h:
                h["STAT"].data = np.asarray(h["STAT"].data / Tc[:, None, None], np.float32)
            hh = h[0].header
            hh["TELLMETH"] = (method, "telluric method (scripts/telluric_combined.py)")
            hh["TELLNFR"] = (info["n_used"], "frames contributing to T_eff")
            hh["TELLNOB"] = (len(info["obs"]), "OBs (standard stars) in T_eff")
            hh["TELLCOV"] = (round(info["coverage"], 3), "fraction of frames with telluric")
            hh["TELLFLOR"] = (floor, "T_eff <= floor set to NaN")
            hh["HISTORY"] = "Telluric corrected (%s) from %d frames of %d OBs" % (
                method, info["n_used"], len(info["obs"]))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            h.writeto(dst, overwrite=True)
            # keep the effective transmission next to the cube for inspection
            fits.PrimaryHDU(np.asarray(T, np.float32)).writeto(_teff_name(dst), overwrite=True)
            log.append((rel, "ok %d frames, %d OBs" % (info["n_used"], len(info["obs"]))))
            if verbose:
                print("  ok   %-70s %3d frames %2d OBs" % (rel[-70:], info["n_used"], len(info["obs"])), flush=True)
    return log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("index")
    a1.add_argument("--frames", nargs="+", required=True)
    a1.add_argument("--out", default="telluric_frame_index.json")
    a2 = sub.add_parser("apply")
    a2.add_argument("--comb", required=True)
    a2.add_argument("--index", required=True)
    a2.add_argument("--tell", nargs="+", required=True)
    a2.add_argument("--out", required=True)
    a2.add_argument("--method", default=BEST, choices=[BEST, "noshift", "rv", "rv_airmass", "rv_airm_iwv"])
    a2.add_argument("--floor", type=float, default=None, help="default 0.02 (best) / 0.1 (version-1 methods)")
    a2.add_argument("--bands", default=None)
    a2.add_argument("--static", default=STATIC_DEFAULT,
                    help="folder with kmos_atmos_<b>.fits and RESPONSE_<B>_2021.fits (method best)")
    a2.add_argument("--calibrators", default=CALIB_RE, help="regex of the target ids of bright in-field stars")
    a2.add_argument("--align-cache", default=None, help="JSON cache of the standard offsets (default: in --out)")
    a2.add_argument("--no-stellar-guard", action="store_true")
    a2.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    if a.cmd == "index":
        idx = build_index(a.frames)
        with open(a.out, "w") as f:
            json.dump(idx, f)
        print("%d frames indexed -> %s" % (len(idx), a.out))
    else:
        idx = json.load(open(a.index))
        store = TelluricStore(a.tell)
        print("%d frames in index, %d telluric files" % (len(idx), len(store.paths)))
        if os.path.abspath(a.out) == os.path.abspath(a.comb):
            sys.exit("refusing to overwrite the input folder")
        log = apply_folder(a.comb, a.out, idx, store, a.method, a.floor,
                           bands=a.bands.split(",") if a.bands else None, overwrite=a.overwrite,
                           static=a.static, calib_re=a.calibrators, align_cache=a.align_cache,
                           stellar_guard=not a.no_stellar_guard)
        n_ok = sum(1 for _, s in log if s.startswith("ok"))
        print("\n%d cubes corrected, %d skipped -> %s" % (n_ok, len(log) - n_ok, a.out))


if __name__ == "__main__":
    main()
