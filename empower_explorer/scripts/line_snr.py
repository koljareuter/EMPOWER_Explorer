"""
Robust signal-to-noise measurement for emission lines in 1-D (KMOS) spectra.

Why not peak / noise?
    The peak pixel is exactly what a cosmic, hot pixel or OH-sky residual
    dominates.  This module instead:

    1. fits the continuum robustly (iterative sigma-clipped polynomial) in
       side-bands that exclude the line *and* any companion lines,
    2. flags single-pixel spikes (pixel far above its 3-pixel running median
       AND that excess makes up most of the pixel value) and masks them,
    3. fits Gaussian(s) with physical priors: centre within +-dv_max of the
       catalogue redshift, width >= instrumental LSF (a spike narrower than
       the LSF cannot be fitted well -> flagged 'unresolved'),
       companions (e.g. [NII]) share velocity and width,
    4. primary S/N = fitted line flux / its error.  The error comes from
       the per-pixel noise (error spectrum if given -> sky lines are
       down-weighted), is inflated for pixel-to-pixel correlation of the
       resampled cubes (autocorrelation of the side-band residuals, lags
       1-4) and includes the continuum-level uncertainty,
    5. cross-check: model-independent flux in a +-int_nsig*sigma aperture
       around the fitted centre (companions subtracted, spikes replaced by
       the model, aperture-corrected); fit and aperture S/N must agree
       within a factor 2.  Quality flags + one `reliable` boolean.

Wavelengths in micron (as in the KMOS notebooks), velocities in km/s.

Calibration (2026-10-13, null and injection tests on all 253 stacks)
--------------------------------------------------------------------
measure_line_snr with a free velocity/width and a uniform noise is NOT a
calibrated S/N: fitted at wrong redshifts (no line there) it gives median
S/N +1.5, NMAD 1.5 and S/N >= 3 in ~21 % of the cases (the fit finds the
best noise bump within +-500 km/s; OH residual pixels get full weight).
measure_lines() fixes this: per-pixel noise (noise_from_cube) + one
freely searched anchor line that must pass a calibrated threshold + all
other lines FORCED to the anchor's velocity and width.  Null test (eval and
held-out offsets): median S/N 0.0, NMAD 1.15, detected in 2.6-2.9 %;
injected lines: unbiased fluxes, errors with unit pull width; all clear real
lines kept.  Lines offset from z by > ~150 km/s are found only via an anchor.

Usage
-----
    import sys; sys.path.insert(0, "scripts")
    from line_snr import measure_lines, measure_line_snr, noise_from_cube, REST_UM

    noise = noise_from_cube(cube, aperture_mask)      # same units as spec
    res, info = measure_lines(wave, spec, z, {"ha": REST_UM["Ha"], "nii": REST_UM["NII6583"]},
                              noise=noise, band="YJ")
    print(res["ha"].snr, res["ha"].detected, info["anchor"])

    # single line, free fit (not calibrated - see above):
    r = measure_line_snr(wave, spec, z, REST_UM["Ha"], band="YJ")
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.special import erf

__all__ = ["LineSNR", "measure_line_snr", "measure_lines", "lines_from_cube", "box_mask", "ring_mask",
           "line_snr", "noise_from_cube",
           "default_companions", "LINE_COMPANIONS", "REST_UM",
           "plot_line_snr", "BAND_RESOLUTION"]

C_KMS = 299792.458
MAD2STD = 1.4826

# approximate KMOS resolving power per band
BAND_RESOLUTION = {"IZ": 3400, "YJ": 3600, "H": 4000, "K": 4200, "HK": 2000}

# Rest wavelengths in VACUUM (micron) - the one table for all scripts and notebooks.
# KMOS cubes have a vacuum wavelength axis (CTYPE3 = 'WAVE'; air would be 'AWAV'),
# so air values (Halpha 6562.80 A) put every line ~83 km/s (~3 A at z = 0.7) too blue.
REST_UM = {
    "OII3726":  0.372709,   # 3727.09 A
    "OII3729":  0.372988,   # 3729.88 A
    "Hg":       0.434169,   # 4341.69 A
    "Hb":       0.486268,   # 4862.68 A
    "OIII4959": 0.496030,   # 4960.30 A
    "OIII5007": 0.500824,   # 5008.24 A
    "HeI5876":  0.587725,   # 5877.25 A
    "OI6300":   0.630204,   # 6302.04 A
    "OI6364":   0.636554,   # 6365.54 A
    "NII6548":  0.654986,   # 6549.86 A
    "Ha":       0.656461,   # 6564.61 A
    "NII6583":  0.658527,   # 6585.27 A
    "SII6716":  0.671829,   # 6718.29 A
    "SII6731":  0.673267,   # 6732.67 A
}

# nearby lines fitted together with the target line (rest micron)
LINE_COMPANIONS = {
    REST_UM["Ha"]:       (REST_UM["NII6548"], REST_UM["NII6583"]),   # Hα <- [NII] doublet
    REST_UM["NII6583"]:  (REST_UM["Ha"], REST_UM["NII6548"]),        # [NII]6583
    REST_UM["OIII5007"]: (REST_UM["OIII4959"],),                     # [OIII]5007
    REST_UM["OIII4959"]: (REST_UM["OIII5007"],),                     # [OIII]4959
    REST_UM["SII6716"]:  (REST_UM["SII6731"],),                      # [SII]6716
    REST_UM["SII6731"]:  (REST_UM["SII6716"],),                      # [SII]6731
}


def default_companions(line_rest, tol=0.0003):
    """Companion rest wavelengths for a known line, () otherwise.  Scaled by
    line_rest / table value, so a caller still passing an air wavelength gets
    companions on the same (air) scale and the line spacing stays exact."""
    for lr, comp in LINE_COMPANIONS.items():
        if abs(line_rest - lr) < tol:
            return tuple(c * line_rest / lr for c in comp)
    return ()


# flags that make a measurement unreliable (others are informational)
HARD_FLAGS = {"no_coverage", "too_few_pixels", "fit_failed", "unresolved",
              "dv_at_bound", "inconsistent", "bad_shape"}


@dataclass
class LineSNR:
    snr: float = np.nan          # primary: Gaussian-fit flux / its error
    snr_ap: float = np.nan       # cross-check: aperture flux / aperture noise
    snr_peak: float = np.nan     # legacy peak / sigma_pix (spikes masked)
    flux: float = np.nan         # fitted line flux [spec units * micron]
    flux_err: float = np.nan
    flux_ap: float = np.nan      # aperture flux (aperture-corrected)
    flux_ap_err: float = np.nan
    center: float = np.nan       # fitted observed wavelength [micron]
    dv_kms: float = np.nan       # offset from catalogue redshift
    sigma_kms: float = np.nan    # observed (LSF-convolved) Gaussian sigma
    chi2_red: float = np.nan
    shape_mismatch: float = np.nan  # unexplained rms in aperture / model peak
    noise_corr: float = np.nan   # correlated-noise inflation factor
    rho: np.ndarray | None = field(default=None, repr=False)  # autocorr lags 1..
    n_spikes: int = 0
    flags: list = field(default_factory=list)
    reliable: bool = False
    detected: bool = False       # set by measure_lines (calibrated rule)
    forced: bool = False         # measure_lines: velocity and width fixed
    # arrays for diagnostics / plotting (local region only)
    wave: np.ndarray | None = field(default=None, repr=False)
    resid: np.ndarray | None = field(default=None, repr=False)
    cont: np.ndarray | None = field(default=None, repr=False)
    model: np.ndarray | None = field(default=None, repr=False)
    err: np.ndarray | None = field(default=None, repr=False)
    spikes: np.ndarray | None = field(default=None, repr=False)
    aperture: np.ndarray | None = field(default=None, repr=False)


# ── helpers ────────────────────────────────────────────────────────────────────
def _robust_std(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan
    return MAD2STD * np.median(np.abs(x - np.median(x)))


def _fit_continuum(w, s, mask, order=1, kappa=3.0, n_iter=5):
    """Iterative sigma-clipped polynomial fit; returns (poly, used_mask)."""
    use = mask & np.isfinite(s)
    if use.sum() < order + 3:
        return None, use
    x0 = np.mean(w[use])
    for _ in range(n_iter):
        p = np.polyfit(w[use] - x0, s[use], order)
        r = s - np.polyval(p, w - x0)
        sd = _robust_std(r[use])
        if not np.isfinite(sd) or sd == 0:
            break
        new = mask & np.isfinite(s) & (np.abs(r) < kappa * sd)
        if new.sum() < order + 3 or np.array_equal(new, use):
            break
        use = new
    return (lambda ww: np.polyval(p, ww - x0)), use


def _find_spikes(resid, err, kappa=5.0, frac=0.7):
    """Pixels that stand out from their 3-px running median by > kappa*err
    AND where that excess is > frac of the pixel value.  A resolved line
    (FWHM >= 2 px) has neighbours carrying a large fraction of the peak,
    so it fails the second criterion; a 1-px artefact passes both."""
    r = np.where(np.isfinite(resid), resid, 0.0)
    d = r - median_filter(r, size=3, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.abs(d) > kappa * err) & (np.abs(d) > frac * np.abs(r)) \
            & np.isfinite(resid)


def _autocorr(z, used, max_lag=4):
    """Autocorrelation rho_1..rho_max_lag of the normalised residuals z,
    using only pixel pairs where both pixels are side-band pixels."""
    z = np.where(used & np.isfinite(z), z, 0.0)
    var = np.sum(z[used] ** 2) / max(used.sum(), 1)
    rho = []
    for l in range(1, max_lag + 1):
        pairs = used[:-l] & used[l:]
        if pairs.sum() < 20 or var <= 0:
            break
        rho.append(np.sum((z[:-l] * z[l:])[pairs]) / pairs.sum() / var)
    return np.array(rho)


def _corr_factor(rho, n):
    """Noise inflation of an n-pixel sum for autocorrelation rho:
    Var(sum) = n * sigma^2 * [1 + 2 * sum_l (1 - l/n) rho_l].
    The resampled KMOS cubes correlate only neighbouring pixels, so a few
    lags suffice and they are measured far more precisely than the scatter
    of a handful of independent n-pixel window sums."""
    lags = np.arange(1, min(len(rho), n - 1) + 1)
    if lags.size == 0:
        return 1.0
    f2 = 1.0 + 2.0 * np.sum((1.0 - lags / n) * rho[:lags.size])
    return float(np.clip(np.sqrt(max(f2, 0.0)), 1.0, 3.0))


def _make_model(lams_rest, z):
    """Gaussians sharing dv and sigma; params = (dv, sigma_kms, F_0, F_1, ...)."""
    lams_rest = np.asarray(lams_rest, float)

    def model(w, dv, sig, *fluxes):
        out = np.zeros_like(w, dtype=float)
        for lr, f in zip(lams_rest, fluxes):
            lc = lr * (1 + z) * (1 + dv / C_KMS)
            s = lc * sig / C_KMS
            out += f / (np.sqrt(2 * np.pi) * s) * np.exp(-0.5 * ((w - lc) / s) ** 2)
        return out
    return model


# ── main ───────────────────────────────────────────────────────────────────────
def measure_line_snr(wave, spec, z, line_rest, noise=None, companions=None,
                     band=None, resolution=3500.0,
                     win_half=0.0025, cont_in=0.006, cont_out=0.022,
                     dv_max=500.0, sigma_max=600.0, int_nsig=2.5,
                     spike_kappa=5.0, spike_frac=0.7, cont_order=1,
                     shape_max=0.6, snr_min=3.0, sigma_min=0.0):
    """Measure a robust S/N of one emission line.

    Parameters
    ----------
    wave, spec : 1-D arrays (micron, any flux unit)
    z          : catalogue redshift
    line_rest  : rest wavelength of the line [micron]
    noise      : optional 1-D 1-sigma error spectrum (same units as spec).
                 Strongly recommended (e.g. from `noise_from_cube`) - it is
                 the only way to down-weight OH sky-line residuals.  It is
                 rescaled to match the side-band scatter.
    companions : rest wavelengths of nearby lines fitted simultaneously
                 (same dv & sigma), e.g. (REST_UM["NII6548"], REST_UM["NII6583"]) for [NII].
                 None -> looked up in LINE_COMPANIONS; () -> none.
    band       : 'IZ','YJ','H','K','HK' -> sets resolution
    win_half   : half width of the search window [micron, observed]
    cont_in/out: side-band inner/outer distance from the line [micron]
    dv_max     : allowed |velocity offset| from z [km/s]
    sigma_max  : maximum observed Gaussian sigma [km/s]
    sigma_min  : minimum INTRINSIC sigma [km/s]; the observed sigma is at least
                 sqrt(sigma_lsf^2 + sigma_min^2) (0 = LSF only, as before)
    int_nsig   : aperture half width in fitted sigmas
    shape_max  : max. shape mismatch (rms of data - model in the aperture,
                 noise subtracted in quadrature, / model peak).  Real lines on
                 KMOS data stay < 0.25 at S/N > 8; a narrow artefact fitted by a
                 broad Gaussian gives ~1.  Above -> 'bad_shape'.
    snr_min    : threshold used for `reliable`
    """
    res = LineSNR()
    if companions is None:
        companions = default_companions(line_rest)
    if band is not None:
        resolution = BAND_RESOLUTION.get(band.upper(), resolution)
    sig_lsf = C_KMS / (resolution * 2.3548)

    wave = np.asarray(wave, float)
    spec = np.asarray(spec, float)
    lam0 = line_rest * (1 + z)
    lam_c = [c * (1 + z) for c in companions]
    fin = np.isfinite(wave) & np.isfinite(spec)
    if not fin.any() or not (wave[fin].min() < lam0 < wave[fin].max()):
        res.flags.append("no_coverage")
        return res

    # local region: everything within cont_out of the line or a companion
    lam_all = [lam0] + lam_c
    reg = np.zeros_like(wave, bool)
    for l in lam_all:
        reg |= np.abs(wave - l) <= cont_out
    w, s = wave[reg], spec[reg]
    e_in = None if noise is None else np.asarray(noise, float)[reg]
    dlam = np.abs(np.gradient(w))

    near_line = np.zeros_like(w, bool)
    for l in lam_all:
        near_line |= np.abs(w - l) < cont_in
    cont_mask = ~near_line & np.isfinite(s)
    if e_in is not None:
        cont_mask &= np.isfinite(e_in) & (e_in > 0)

    win = (np.abs(w - lam0) <= win_half) & np.isfinite(s)
    if win.sum() < 3 or cont_mask.sum() < 10:
        res.flags.append("too_few_pixels")
        return res

    # 1) robust continuum
    cfun, cont_used = _fit_continuum(w, s, cont_mask, order=cont_order)
    if cfun is None:
        res.flags.append("too_few_pixels")
        return res
    cont = cfun(w)
    resid = s - cont

    # 2) per-pixel error, scaled to the observed side-band scatter
    sig_pix = _robust_std(resid[cont_used])
    if not np.isfinite(sig_pix) or sig_pix <= 0:
        res.flags.append("too_few_pixels")
        return res
    if e_in is not None:
        scale = _robust_std(resid[cont_used] / e_in[cont_used])
        err = e_in * (scale if np.isfinite(scale) and scale > 0 else 1.0)
        err = np.where(np.isfinite(err) & (err > 0), err, np.inf)
        if np.nanmedian(err[win]) > 1.5 * np.nanmedian(err[cont_used]):
            res.flags.append("skyline")
    else:
        err = np.full_like(w, sig_pix)

    # 3) spikes (in the whole local region; continuum fit already clipped them)
    spikes = _find_spikes(resid, err, spike_kappa, spike_frac)
    good = np.isfinite(resid) & np.isfinite(err) & ~spikes

    # 4) Gaussian fit (primary + companions)
    fit_reg = np.zeros_like(w, bool)
    for l in lam_all:
        fit_reg |= np.abs(w - l) <= 2 * win_half
    fit_reg &= good
    model_fn = _make_model([line_rest, *companions], z)
    n_lines = 1 + len(companions)
    popt = pcov = None
    if fit_reg.sum() >= 2 + n_lines + 2:
        sm = median_filter(np.where(good, resid, 0.0), size=3, mode="nearest")
        i0 = np.argmax(np.where(win & good, sm, -np.inf))
        dv0 = np.clip((w[i0] / lam0 - 1) * C_KMS, -0.9 * dv_max, 0.9 * dv_max)
        sig_lo = float(np.hypot(sig_lsf, sigma_min)) if sigma_min > 0 else 0.9 * sig_lsf
        sig0 = np.clip(max(1.5 * sig_lsf, 80.0), sig_lo * 1.01, sigma_max * 0.99)
        f0 = max(np.nansum(np.where(win & good, resid, 0) * dlam), sig_pix * dlam.mean())
        p0 = [dv0, sig0, f0] + [0.3 * f0] * len(companions)
        lo = [-dv_max, sig_lo, -np.inf] + [0.0] * len(companions)
        hi = [dv_max, sigma_max, np.inf] + [np.inf] * len(companions)
        try:
            popt, pcov = curve_fit(model_fn, w[fit_reg], resid[fit_reg], p0=p0,
                                   sigma=err[fit_reg], absolute_sigma=True,
                                   bounds=(lo, hi), maxfev=20000)
        except (RuntimeError, ValueError):
            popt = None
    model = np.zeros_like(w)
    if popt is None or not np.all(np.isfinite(pcov)):
        res.flags.append("fit_failed")
        popt = None
    else:
        model = model_fn(w, *popt)
        chi = (resid[fit_reg] - model[fit_reg]) / err[fit_reg]
        dof = max(fit_reg.sum() - len(popt), 1)
        res.chi2_red = float(np.sum(chi ** 2) / dof)
        # NOT scaled by sqrt(chi2_red): bright real lines are never perfectly
        # Gaussian, so chi2_red >> 1 there would wrongly deflate their S/N.
        # Shape mismatch is tested separately ('bad_shape').
        perr = np.sqrt(np.diag(pcov))
        res.dv_kms, res.sigma_kms = float(popt[0]), float(popt[1])
        res.center = lam0 * (1 + popt[0] / C_KMS)
        res.flux, res.flux_err = float(popt[2]), float(perr[2])
        if popt[1] < 1.05 * 0.9 * sig_lsf:
            res.flags.append("unresolved")
        if abs(popt[0]) > 0.97 * dv_max:
            res.flags.append("dv_at_bound")
        if res.chi2_red > 3:
            res.flags.append("poor_fit")

    # 5) aperture flux with companion model removed, spikes -> model
    if popt is not None:
        s_obs = res.center * res.sigma_kms / C_KMS
        ap = np.abs(w - res.center) <= int_nsig * s_obs
        if ap.sum() < 3:  # make it at least 3 px
            ap = np.zeros_like(w, bool)
            ap[np.argsort(np.abs(w - res.center))[:3]] = True
        comp_model = model - model_fn(w, popt[0], popt[1], popt[2],
                                      *([0.0] * len(companions)))
        prim_model = model - comp_model
    else:
        ap = win.copy()
        comp_model = np.zeros_like(w)
        prim_model = median_filter(np.where(good, resid, 0.0), size=3,
                                   mode="nearest")
    data = np.where(spikes | ~np.isfinite(resid), prim_model + comp_model, resid)
    data = data - comp_model
    ap &= np.isfinite(data) & np.isfinite(err)
    if ap.sum() < 3:
        res.flags.append("too_few_pixels")
        return res

    # 6) errors: correlated noise + uncertainty of the continuum level
    n_ap = int(ap.sum())
    rho = _autocorr(resid / err, cont_used)
    corr = _corr_factor(rho, n_ap)
    n_cont = int(cont_used.sum())
    dc = sig_pix * _corr_factor(rho, n_cont) / np.sqrt(n_cont)  # error of continuum level
    res.noise_corr, res.rho = corr, rho
    if popt is not None:
        # a constant offset dc changes a Gaussian LSQ flux by dc * 2*sqrt(pi)*sigma_lambda
        e_cont = dc * 2 * np.sqrt(np.pi) * s_obs
        res.flux_err = float(np.hypot(res.flux_err * corr, e_cont))
        res.snr = res.flux / res.flux_err if res.flux_err > 0 else np.nan
    ap_frac = erf(int_nsig / np.sqrt(2)) if popt is not None else 1.0
    res.flux_ap = float(np.sum(data[ap] * dlam[ap])) / ap_frac
    res.flux_ap_err = float(np.hypot(np.sqrt(np.sum((err[ap] * dlam[ap]) ** 2)) * corr,
                                     dc * np.sum(dlam[ap]))) / ap_frac
    res.snr_ap = res.flux_ap / res.flux_ap_err

    pk = np.where(win & good, resid, -np.inf)
    res.snr_peak = float(np.max(pk) / sig_pix) if np.isfinite(pk).any() else np.nan
    res.n_spikes = int(np.sum(spikes & (np.abs(w - lam0) <= cont_in)))
    if res.n_spikes:
        res.flags.append("spike_masked")

    # 7) shape: does the Gaussian explain the signal in the aperture?
    if popt is not None:
        sel = ap & ~spikes & np.isfinite(resid)
        pk_model = np.max(prim_model[sel]) if sel.any() else np.nan
        if sel.any() and pk_model > 0:
            excess = np.mean((resid[sel] - model[sel]) ** 2 - err[sel] ** 2)
            res.shape_mismatch = float(np.sqrt(max(excess, 0.0)) / pk_model)
            if res.shape_mismatch > shape_max:
                res.flags.append("bad_shape")

    # 8) consistency: aperture vs. fit S/N must roughly agree for a real line
    if np.isfinite(res.snr) and max(res.snr, res.snr_ap) >= snr_min:
        ratio = res.snr_ap / res.snr if res.snr > 0 else np.inf
        if not (0.5 <= ratio <= 2.0):
            res.flags.append("inconsistent")

    res.reliable = (res.snr >= snr_min and not (HARD_FLAGS & set(res.flags)))
    res.wave, res.resid, res.cont, res.model = w, resid, cont, model
    res.err, res.spikes, res.aperture = err, spikes, ap
    return res


def line_snr(wave, spec, z, line_rest=REST_UM["Ha"], noise=None, companions=None,
             require_reliable=False, **kw):
    """Drop-in replacement for the old notebook `line_snr` (returns a float).
    NaN only if the line is not covered or the fit failed.  Artefact flags
    (unresolved, dv_at_bound, bad_shape, inconsistent) do NOT reject the value
    (project decision 2026-10-07: flagged lines count as normal detections);
    pass require_reliable=True to get NaN whenever `reliable` is False."""
    r = measure_line_snr(wave, spec, z, line_rest, noise=noise,
                         companions=companions, **kw)
    if require_reliable and not r.reliable:
        return np.nan
    if {"no_coverage", "too_few_pixels", "fit_failed"} & set(r.flags):
        return np.nan
    return float(r.snr)


ANCHOR_RESTS = (REST_UM["Ha"], REST_UM["OIII5007"], REST_UM["Hb"])   # vacuum
T_ANCHOR = 5.0        # anchor threshold: ~2 % false positives, like forced lines at S/N 3 (null test)
SIG_DEFAULT = 80.0    # km/s, intrinsic sigma used without an anchor (typical integrated value)
ANCHOR_SEARCH = dict(dv_max=500.0, sigma_min=40.0, sigma_max=350.0)


def _is_anchor_line(rest, tol=0.0003):
    return any(abs(rest - a) < tol for a in ANCHOR_RESTS)


def measure_lines(wave, spec, z, lines, noise=None, band=None, win_half=None,
                  anchor=None, t_anchor=T_ANCHOR, snr_min=3.0, sig_default=SIG_DEFAULT,
                  cont_in=0.006, cont_out=0.022, free_dv=0.0):
    """Calibrated S/N of several lines of one spectrum (two stages).

    1. anchor: Halpha, [OIII]5007 and Hbeta among `lines` are fitted with a free
       velocity (|dv| <= 500 km/s) and width (sigma_int 40-350 km/s).  The best
       one is the anchor if it is `reliable` and min(fit S/N, aperture S/N) >=
       t_anchor (5.0: ~2 % false positives of this searched statistic, as for a
       forced line at S/N 3; 6.5 gives 1 %).
    2. forced: every other line is fitted with velocity and width FIXED to the
       anchor's (or to `anchor` given by the caller, e.g. from another band of
       the same galaxy, or else to the catalogue z with sig_default) - no search,
       so its S/N is an honest flux / error.

    lines   : {name: rest wavelength [micron]}
    noise   : per-pixel 1-sigma spectrum (noise_from_cube), strongly recommended
    win_half: {name: micron} or one value (default 0.0025 for Halpha, else 0.002)
    anchor  : optional (dv_kms, sigma_int_kms) used when no line of this
              spectrum qualifies as anchor
    free_dv : km/s velocity freedom of the forced fits when there is no anchor
              (default 0 = fixed; 150 doubles the false positives - not calibrated)
    Returns (results, info): results {name: LineSNR} with .detected (anchor
    passed, or forced S/N >= snr_min) and .forced; info dict(anchor=name or
    'external' or None, dv, sigma_int, anchor_stat).
    """
    sig_lsf = C_KMS / (BAND_RESOLUTION.get((band or "").upper(), 3500.0) * 2.3548)
    wh = (lambda n, r: win_half[n]) if isinstance(win_half, dict) else \
         (lambda n, r: win_half if win_half is not None else (0.0025 if abs(r - REST_UM["Ha"]) < 3e-4 else 0.002))
    common = dict(noise=noise, band=band, cont_in=cont_in, cont_out=cont_out, snr_min=snr_min)

    # 1. anchor search
    best, best_stat = None, -np.inf
    for name, rest in lines.items():
        if not _is_anchor_line(rest):
            continue
        r = measure_line_snr(wave, spec, z, rest, win_half=wh(name, rest), **ANCHOR_SEARCH, **common)
        if {"no_coverage", "too_few_pixels"} & set(r.flags):
            continue
        stat = min(r.snr, r.snr_ap) if (r.reliable and np.isfinite(r.snr_ap)) else -np.inf
        if stat > best_stat:
            best, best_stat = (name, r), stat
    info = dict(anchor=None, dv=0.0, sigma_int=sig_default, anchor_stat=best_stat)
    if best is not None and best_stat >= t_anchor:
        info.update(anchor=best[0], dv=best[1].dv_kms,
                    sigma_int=float(np.sqrt(max(best[1].sigma_kms**2 - sig_lsf**2, 1.0))))
    elif anchor is not None:
        info.update(anchor="external", dv=float(anchor[0]), sigma_int=float(anchor[1]))

    # 2. forced photometry at the anchor kinematics
    z0 = (1 + z) * (1 + info["dv"] / C_KMS) - 1
    sig_hi = float(np.hypot(sig_lsf, info["sigma_int"])) * 1.02 + 1.0
    out = {}
    for name, rest in lines.items():
        if name == info["anchor"]:
            r = best[1]
            r.detected = True
        else:
            dvm = free_dv if (info["anchor"] is None and free_dv > 0) else 2.0
            r = measure_line_snr(wave, spec, z0, rest, win_half=wh(name, rest), dv_max=dvm,
                                 sigma_min=info["sigma_int"], sigma_max=sig_hi, **common)
            r.forced = True
            # kinematics are fixed: bound/width flags carry no information here
            r.flags = [f for f in r.flags if f not in ("dv_at_bound", "unresolved")]
            if np.isfinite(r.dv_kms):
                r.dv_kms = info["dv"] + r.dv_kms * (1 + z0) / (1 + z)
            r.reliable = bool(np.isfinite(r.snr) and r.snr >= snr_min and not (HARD_FLAGS & set(r.flags)))
            r.detected = bool(np.isfinite(r.snr) and r.snr >= snr_min
                              and not ({"no_coverage", "too_few_pixels"} & set(r.flags)))
        out[name] = r
    return out, info


def box_mask(shape, centre, box=5):
    """2-D boolean mask of a box x box aperture at centre (y, x), clipped to the cube."""
    m = np.zeros(shape, bool)
    h = box // 2
    m[max(centre[0] - h, 0):centre[0] + h + 1, max(centre[1] - h, 0):centre[1] + h + 1] = True
    return m


def ring_mask(shape, centre, r_in, r_out):
    """2-D boolean mask of spaxels with r_in <= distance < r_out from centre (y, x)."""
    yy, xx = np.indices(shape)
    d2 = (yy - centre[0]) ** 2 + (xx - centre[1]) ** 2
    return (d2 >= r_in ** 2) & (d2 < r_out ** 2)


def lines_from_cube(wave, spec, z, lines, cube, mask, band=None, **kw):
    """measure_lines with the per-pixel noise of the spaxels `mask` of `cube`
    (noise_from_cube).  `spec` may be normalised in any way: only the shape of
    the noise matters, it is rescaled to the side-band scatter."""
    return measure_lines(wave, spec, z, lines, noise=noise_from_cube(cube, mask), band=band, **kw)


def noise_from_cube(cube, aperture_mask, bg_mask=None, min_bg=8):
    """Per-wavelength 1-sigma noise of an aperture-summed spectrum.

    Uses the spaxel-to-spaxel robust scatter of background spaxels at each
    wavelength (captures OH residuals) times sqrt(N_aperture).

    cube          : (nz, ny, nx)
    aperture_mask : (ny, nx) bool, spaxels summed into the spectrum
    bg_mask       : (ny, nx) bool, default = finite spaxels not in the
                    aperture dilated by 2 px
    If you normalise the spectrum (e.g. spec / median), divide the noise by
    the same factor.
    """
    from scipy.ndimage import binary_dilation
    cube = np.asarray(cube, float)
    ap = np.asarray(aperture_mask, bool)
    if bg_mask is None:
        bg_mask = ~binary_dilation(ap, iterations=2) & np.isfinite(cube).any(0)
    bg = cube[:, bg_mask]                                   # (nz, n_bg)
    nbg = np.isfinite(bg).sum(1)
    med = np.nanmedian(bg, axis=1, keepdims=True)
    sd = MAD2STD * np.nanmedian(np.abs(bg - med), axis=1)
    sd[nbg < min_bg] = np.nan
    n_ap = np.isfinite(cube[:, ap]).sum(1)
    return sd * np.sqrt(np.maximum(n_ap, 1))


def plot_line_snr(res, ax=None, title=None):
    """Quick diagnostic plot of a LineSNR result."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.5))
    if res.wave is None:
        ax.text(0.5, 0.5, ", ".join(res.flags) or "no data",
                transform=ax.transAxes, ha="center")
        return ax
    w = res.wave
    ax.step(w, res.resid, where="mid", color="0.3", lw=0.8, label="data - cont")
    ax.fill_between(w, -res.err, res.err, step="mid", color="0.85",
                    label="1σ", zorder=0)
    ax.plot(w, res.model, color="C3", lw=1.3, label="fit")
    if res.aperture is not None and res.aperture.any():
        ax.axvspan(w[res.aperture].min(), w[res.aperture].max(),
                   color="C0", alpha=0.12, label="aperture")
    if res.spikes is not None and res.spikes.any():
        ax.plot(w[res.spikes], res.resid[res.spikes], "x", color="m",
                ms=7, label="spike")
    ax.axhline(0, color="k", lw=0.5)
    fin = np.isfinite(res.err) & np.isfinite(res.resid) & ~res.spikes
    lo = np.nanmin(np.r_[res.resid[fin], -3 * res.err[fin]])
    hi = np.nanmax(np.r_[res.resid[fin], res.model[fin], 3 * res.err[fin]])
    ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.15 * (hi - lo))
    ax.set_xlabel("λ [µm]")
    ax.set_title(title or (f"S/N={res.snr:.1f}  aperture={res.snr_ap:.1f}  "
                           f"peak={res.snr_peak:.1f}  "
                           f"{'OK' if res.reliable else 'UNRELIABLE'} "
                           f"{' '.join(res.flags)}"), fontsize=9)
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    return ax


# ── self-test on synthetic spectra ─────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(1)
    z, ha = 0.85, REST_UM["Ha"]
    wave = np.arange(1.10, 1.30, 0.000175)       # ~KMOS YJ sampling
    sig_kms = 120.0

    def gauss(l0, f):
        s = l0 * sig_kms / C_KMS
        return f / (np.sqrt(2 * np.pi) * s) * np.exp(-0.5 * ((wave - l0) / s) ** 2)

    def old_peak_snr(spec):
        lam = ha * (1 + z)
        sm = np.abs(wave - lam) <= 0.0025
        cm = (np.abs(wave - lam) >= 0.006) & (np.abs(wave - lam) <= 0.022)
        p = np.polyfit(wave[cm], spec[cm], 1)
        r = spec - np.polyval(p, wave)
        return np.max(r[sm]) / (MAD2STD * np.median(np.abs(r[cm] - np.median(r[cm]))))

    noise = np.full_like(wave, 1.0)
    cases = {}
    base = 5 + rng.normal(0, 1, wave.size)
    cases["real line"] = base + gauss(ha * (1 + z), 0.006) \
        + gauss(REST_UM["NII6583"] * (1 + z), 0.002)
    cases["no line, spike"] = base.copy()
    i = np.argmin(np.abs(wave - ha * (1 + z) * 1.0004))
    cases["no line, spike"][i] += 15
    sky = base.copy()
    j = np.argmin(np.abs(wave - ha * (1 + z) * 0.9997))
    sky[j - 1:j + 2] += [6, 12, 6]                # OH residual, 3 px
    nsky = noise.copy()
    nsky[j - 1:j + 2] = [5, 8, 5]
    cases["no line, sky residual"] = sky
    cases["weak line + spike"] = base + gauss(ha * (1 + z), 0.0015)
    cases["weak line + spike"][i] += 15

    print(f"{'case':24s} {'old peak':>8s} {'S/N':>6s} {'ap':>6s} rel  flags")
    for name, sp in cases.items():
        nz = nsky if "sky" in name else noise
        r = measure_line_snr(wave, sp, z, ha, noise=nz,
                             companions=(REST_UM["NII6548"], REST_UM["NII6583"]), band="YJ")
        print(f"{name:24s} {old_peak_snr(sp):8.1f} {r.snr:6.1f} "
              f"{r.snr_ap:6.1f} {'Y' if r.reliable else 'n'}    "
              f"{','.join(r.flags)}")
