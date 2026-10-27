# -*- coding: utf-8 -*-
"""
EMPOWER Voronoi binning + multi-Gaussian emission-line maps (pipeline version 2).

Successor of tools/Fitting_Voronoi.py (KMOS3D: H-alpha only, air wavelengths, PSF from extension 4),
for the EMPOWER KMOS combined cubes.  Kept from it: continuum from line-free side bands, single and
multiple Gaussians, the [NII] doublet, model choice by BIC, integrated-flux S/N.  Taken from
kmos_best3_single_gauss.ipynb so the numbers agree with the notebooks: vacuum rest wavelengths
(scripts/line_snr.py), spaxel noise from the scatter of sky spaxels, measured correlated-noise factor
c(N), Voronoi binning on the anchor-line S/N (target 15, fall-backs 10/8/6), weak lines forced to the
anchor kinematics, null-test error calibration, [SII] density (Sanders+16), N2 metallicity (Marino+13).

What it does
  * line groups Hb | [OIII]4959,5007 | Ha+[NII]6548,6583 | [SII]6716,6731 with 1..3 Gaussian
    components each (velocity and width shared by the lines of a group; [NII] and [OIII] doublet
    ratios fixed, the [SII] ratio kept inside its physical range)
  * the number of components per bin from BIC - chi2 rescaled by the reduced chi2 of the simpler
    model, so spikes and template mismatch do not buy components - plus a significance cut on every
    component's own flux
  * one set of bins for all bands: bins are made on the anchor band (Ha, else [OIII], else Hb) and
    carried to the other bands through the WCS (sub-pixel sampling), so ratios across bands use the
    same sky area; ratios across bands use the cubes' FLUXK calibration
  * galaxy region = footprint interior (outer 1-px frame and spaxels noisier than 1.7x the median
    spaxel removed) connected to the centre; a
    galaxy without detection is fitted in one central aperture and flagged, never mapped from edge
    artefacts
  * detection (DETECT, STATUS): the anchor line fitted (free, clipped) in the central aperture or the
    whole region with calibrated errors must reach S/N >= 5 after dividing by max(1, NULLDET / 3),
    where NULLDET is the largest S/N of the same fit at line-free wavelengths of the same spectrum
    (look-elsewhere effect, continuum structure, sky residuals), and its narrowest component must have
    sigma <= 250 km/s; undetected galaxies get one central aperture with forced fits (upper limits)
  * noise = robust scatter of the sky spaxels (galaxy and frame excluded; if fewer than 30 remain,
    all spaxels >= 6 px from the centre) x the relative noise r of every spaxel (its own pixel-to-
    pixel scatter: field edges with fewer exposures are 2-3x noisier than the centre) x c(N), calibrated per band with null tests BEFORE the bin
    fits as a function of region size (bins, sky boxes when there are few bins, and the TOTAL
    spectrum: NOISE_<band> ERRSCN), and in the anchor band already before the detection.  The galaxy
    region (S/N >= 1) and the Voronoi target S/N use the uncalibrated sky noise, as the notebooks do
    (TARGETSN is nominal); every fit decision uses calibrated errors
  * residual background subtracted: a smooth pedestal (running median of the sky spaxels without the
    +-1000 km/s around every emission line, then interpolated across them); its uncertainty enters the
    continuum and EW errors
  * fit, clip (single-pixel spikes; > 5 sigma outliers away from the lines), refit; a free component
    that is unresolved (sigma < 0.5 sigma_LSF) on a bright OH line (scripts/sky_lines_uves_oliva.csv)
    is a sky residual and rejected
  * forced bins carry NO kinematics in the maps (the template is kept as *_TPL_*); weak outer bins
    are forced to the interpolated velocity field of the detected bins, not to v = 0

Fit: only velocity and width of each component are non-linear; the line fluxes and the linear
continuum follow from a weighted linear solve at every step ("variable projection"): no flux start
values, and the fluxes of multi-component fits are >= 0.

Library (no Qt):
    from empower_explorer.pipeline import voronoi_binning as vb
    prod = vb.run_galaxy(stacks, z, vb.Config(), name='COSMOS 92754')   # stacks: {band: BandStack}
    vb.write_product(prod, path)
    p = vb.read_product(path)      # dict: maps, bins, total, header, config, lines, rest, noise, clipped
    wave, spec, err = vb.product_bin_spectrum(p, stack, band, bin_id)   # exactly what was fitted
    total, parts = vb.model_curve(row, 'ha', wave, p['header']['Z_REF'], components=True, rest=p['rest'],
                                  resolving_power=p['noise'][band]['resolving_power'])
    used = vb.used_pixels(p, wave, bin_id, 'ha') & np.isfinite(spec) & (err > 0)   # pixels of the final fit

Command line (all galaxies of a reduction, in parallel):
    python -m empower_explorer.pipeline.voronoi_binning --all
    python -m empower_explorer.pipeline.voronoi_binning --galaxy "COSMOS 92754" --max-components 3 --overwrite
Products: <EMPOWER>/kmos_voronoi_maps/<reduction>/<FIELD>_<tid>_voronoi.fits + index.csv / index.json.

Naming (schema EMPOWER-VORONOI-2; unique ignoring case; map EXTNAMEs = BINS column names)
  lines   HB OIII4959 OIII5007 NII6548 HA NII6583 SII6716 SII6731:
          <L>_FLUX _FLUX_ERR _SNR _FORCED _FLUX_CGS _FLUX_CGS_ERR _C<k>_FLUX _C<k>_FLUX_ERR
  groups  HBETA OIII HANII SII:
          <G>_BAND _NCOMP _FORCED _CHI2NU _DBIC_NCOMP _DBIC_FREE _NCLIP _QFLAG _X0 _CONT_A _CONT_B
          _CONT_MAIN _CONT_MAIN_ERR _V50 _W80 _V10 _V90 _TPL_V _TPL_SIG
          _C<k>_V _VERR _SIG _SIGERR _FRAC _UNRES (published, NaN for forced or weak fits)
          _C<k>_MV _MSIG _MFRAC (model parameters, also for forced fits: model_curve uses them)
  ratios  LOG_N2 LOG_O3 LOG_S2 LOG_O3N2 (+ _ERR, from the fit covariance; + _XBAND: 1 across bands,
          0 one band, -1 undefined), SII_RATIO (+ _ERR), NE_SII, OH_N2_M13, EW_HA (+ _ERR)
  tables  BINS, TOTAL (sum of all bins), CLIPPED (BIN, GROUP, BAND, WAVE of every clipped pixel), LINES,
          CONFIG, NOISE_<band>; every column carries TUNITn
Units (FITS): fluxes 'adu s-1 um' (cube unit x micron; the models rebuild in these units); *_FLUX_CGS
'10**(-17) erg s-1 cm-2' via the band's FLUXK (about 10 % absolute).  Velocities 'km s-1' relative to
Z_REF; sigma intrinsic (LSF removed).  Pixel coordinates 0-based.  Provenance: PIPEVER, CODEHASH
(sha256 of this file), GITHASH, package versions, input cubes (HISTORY).
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, binary_opening, label
from scipy.optimize import least_squares, lsq_linear

SCHEMA = 'EMPOWER-VORONOI-2'
PIPELINE_VERSION = '2.5.0'        # bump on every change of behaviour (CODEHASH identifies the exact code)
C_KMS = 299792.458
MAD2STD = 1.4826
FWHM_W80 = 2.5631          # W80 of a Gaussian in units of sigma

# vacuum rest wavelengths [micron]; replaced by scripts/line_snr.py REST_UM via sync_rest()
REST = {'OII3726': 0.372709, 'OII3729': 0.372988, 'Hg': 0.434169, 'Hb': 0.486268,
        'OIII4959': 0.496030, 'OIII5007': 0.500824, 'HeI5876': 0.587725, 'OI6300': 0.630204,
        'OI6364': 0.636554, 'NII6548': 0.654986, 'Ha': 0.656461, 'NII6583': 0.658527,
        'SII6716': 0.671829, 'SII6731': 0.673267}
EXTRA_REST = (0.667999, 0.706715, 0.713777, 0.732201, 0.733275, 0.775319)   # HeI, [ArIII], [OII]7320/30
BAND_RESOLUTION = {'IZ': 3400, 'YJ': 3600, 'H': 4000, 'K': 4200, 'HK': 2000}

# line groups; 'tag' = column / EXTNAME prefix, distinct from every line name ignoring case
GROUPS = {
    'hb':   dict(tag='HBETA', label='Hβ', lines=['Hb'], main='Hb', ties={}),
    'oiii': dict(tag='OIII', label='[OIII]', lines=['OIII4959', 'OIII5007'], main='OIII5007',
                 ties={'OIII4959': ('OIII5007', 1 / 2.98)}),
    'ha':   dict(tag='HANII', label='Hα+[NII]', lines=['NII6548', 'Ha', 'NII6583'], main='Ha',
                 ties={'NII6548': ('NII6583', 1 / 2.94)}),
    'sii':  dict(tag='SII', label='[SII]', lines=['SII6716', 'SII6731'], main='SII6716', ties={}),
}
ANCHORS = ('ha', 'oiii', 'hb')
LINE_GROUP = {k: g for g, d in GROUPS.items() for k in d['lines']}
SII_RANGE = (0.4375, 1.4484)        # [SII] 6716/6731 between the high- and the low-density limit

# <G>_QFLAG bits
QF_CHI2 = 1             # reduced chi2 above Config.chi2nu_max
QF_CLIPPED = 2          # pixels clipped before the final fit
QF_UNRESOLVED = 4       # a free component narrower than 5 % of the LSF (its sigma error is NaN)
QF_SII_BOUND = 8        # [SII] ratio pinned to a physical limit
QF_BROAD = 16           # free fits rejected because the line is broader than allowed: forced flux is a lower limit
QF_PARTIAL = 32         # a line of the group is not covered (masked / band edge): left out of the model


def group_tag(group):
    return GROUPS[group]['tag']


def line_tag(line):
    return line.upper()


@dataclass
class Config:
    # binning
    binning: str = 'voronoi'        # 'voronoi', 'spaxel' (every spaxel) or 'grid' (k x k blocks)
    target_sn: float = 15.0
    target_fallback: tuple = (10.0, 8.0, 6.0)
    min_bins: int = 25
    min_sn: float = 1.0
    grid: int = 3
    frame: int = 1                  # px of the footprint edge not used (noisy dither frame)
    rel_max: float = 1.7            # spaxels noisier than this x the median spaxel are not used
    max_offset: float = 4.0         # px: centre peak / galaxy region must lie this close to the centre
    # fit
    groups: tuple = ('hb', 'oiii', 'ha', 'sii')
    max_components: int = 3
    bic_min: float = 10.0
    comp_snr_min: float = 3.0
    dv_max: float = 500.0
    dv_tie: float = 200.0
    sig_int_max: float = 400.0
    sig_extra_max: float = 1500.0
    sig_min_frac: float = 0.3       # x sigma_LSF: minimum width of extra components / free non-anchor lines
    win_pad: float = 0.0025         # micron rest (25 A) beyond the outermost line of a group
    min_cover: float = 0.6
    force_below: float = 5.0
    det_null_ref: float = 3.0       # detection: S/N / max(1, max line-free-window S/N / det_null_ref) >= force_below
    sig_det_max: float = 250.0      # detection: narrowest component sigma_int <= this [km/s]
    ref_recentre: float = 100.0     # km/s: recentre Z_REF on the bins if their median velocity is further off
    kin_snr_min: float = 5.0
    ratio_snr_min: float = 3.0
    sig_default: float = 80.0
    chi2nu_max: float = 3.0
    clip_sigma: float = 5.0
    sky_guard: bool = True          # reject free components unresolved (< 0.5 sigma_LSF) on a bright OH line
    # noise / background
    noise_rmin: float = 6.0
    min_sky: int = 30               # fewer sky spaxels left after excluding the galaxy: sigma_1, c(N) from all
    min_finite: float = 0.5
    edge_margin: int = 4
    bg_subtract: bool = True
    bg_line_kms: float = 1000.0
    bg_window: int = 41
    null_test: bool = True
    null_min: int = 20
    err_scale_range: tuple = (0.5, 5.0)
    null_offsets: tuple = (-6000.0, -5200.0, -4400.0, 3600.0, 4400.0, 5200.0)
    centre_aperture: float = 2.5

    def __post_init__(self):
        if isinstance(self.groups, str):
            self.groups = tuple(self.groups.split(','))
        self.groups = tuple(str(g).strip().lower() for g in self.groups if str(g).strip())
        bad = [g for g in self.groups if g not in GROUPS]
        if bad or not self.groups:
            raise ValueError(f'unknown line groups {bad or "(none)"}; use {list(GROUPS)}')
        if not 1 <= int(self.max_components) <= 3:
            raise ValueError('max_components must be 1, 2 or 3')
        self.max_components = int(self.max_components)
        if self.binning not in ('voronoi', 'spaxel', 'grid'):
            raise ValueError("binning must be 'voronoi', 'spaxel' or 'grid'")
        if not self.target_sn > 0:
            raise ValueError('target_sn must be > 0')
        if int(self.grid) < 1 or int(self.frame) < 0 or int(self.bg_window) < 3:
            raise ValueError('grid >= 1, frame >= 0 and bg_window >= 3 required')
        self.grid, self.frame, self.bg_window = int(self.grid), int(self.frame), int(self.bg_window)
        self.target_fallback = tuple(float(t) for t in self.target_fallback)
        self.null_offsets = tuple(float(t) for t in self.null_offsets)
        self.err_scale_range = tuple(float(t) for t in self.err_scale_range)
        if len(self.err_scale_range) != 2 or not 0 < self.err_scale_range[0] <= self.err_scale_range[1]:
            raise ValueError('err_scale_range must be (low, high) with 0 < low <= high')

    def to_json(self):
        return json.dumps({k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()})

    @classmethod
    def from_json(cls, text):
        d = json.loads(text) if isinstance(text, str) else dict(text)
        known = cls.__dataclass_fields__
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in known})


def sync_rest(rest_um=None, band_resolution=None):
    """Use the notebooks' tables (scripts/line_snr.py REST_UM, BAND_RESOLUTION)."""
    if rest_um:
        REST.update({k: float(v) for k, v in rest_um.items()})
    if band_resolution:
        BAND_RESOLUTION.update(band_resolution)


SKY_UM = np.zeros(0)     # vacuum wavelengths [micron] of the bright OH sky lines (sync_sky)


def sync_sky(root, keep_fraction=0.3):
    """Load the bright OH sky lines from <root>/scripts/sky_lines_uves_oliva.csv (the brightest
    keep_fraction of each catalogue, air -> vacuum).  Without them the sky-residual guard is off."""
    global SKY_UM
    path = os.path.join(root, 'scripts', 'sky_lines_uves_oliva.csv') if root else ''
    try:
        import pandas as pd
        df = pd.read_csv(path)
    except Exception:
        SKY_UM = np.zeros(0)
        return SKY_UM
    lam = pd.to_numeric(df['lambda_A'], errors='coerce').to_numpy(float) * 1e-4
    flux = pd.to_numeric(df['flux'], errors='coerce').to_numpy(float)
    air = df['medium'].astype(str).str.lower().eq('air').to_numpy()
    with np.errstate(all='ignore'):
        s2 = (1.0 / lam) ** 2
        lam = np.where(air, lam * (1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2)), lam)
    src = df['source'].astype(str).to_numpy()
    keep = np.zeros(len(df), bool)
    for c in np.unique(src):
        sel = (src == c) & np.isfinite(lam) & np.isfinite(flux)
        if sel.sum() >= 5:
            keep[sel] = flux[sel] >= np.nanpercentile(flux[sel], 100 * (1 - keep_fraction))
    SKY_UM = np.sort(lam[keep])
    return SKY_UM


# ======================================================================== noise, background, spectra
def erode_valid(finite, margin, min_run=8):
    """Valid mask, trimmed by `margin` px next to long invalid stretches along axis 0 (band
    edges, empty spaxels); short gaps (masked telluric pixels) are not widened (notebooks)."""
    inv = ~finite
    src = binary_opening(inv, structure=np.ones((min_run,) + (1,) * (inv.ndim - 1), bool))
    eroded = inv.copy()
    for s in range(1, margin + 1):
        eroded[s:] |= src[:-s]
        eroded[:-s] |= src[s:]
    return ~eroded


def footprint(coverage, cfg):
    """Spaxels with data, minus the outer `frame` px (noisy dither edge)."""
    fp = np.asarray(coverage) > cfg.min_finite
    if cfg.frame > 0:
        inner = binary_erosion(fp, iterations=cfg.frame, border_value=0)
        if inner.sum() >= 9:
            fp = inner
    return fp


def sky_mask(shape, fp, centre_yx, cfg, exclude=None):
    """Sky spaxels: footprint, >= noise_rmin from the centre and outside the galaxy (`exclude`, dilated)."""
    yy, xx = np.indices(shape)
    d = np.hypot(yy - centre_yx[0], xx - centre_yx[1])
    m = fp & (d >= cfg.noise_rmin)
    if exclude is not None:
        m &= ~binary_dilation(exclude, iterations=1)
    if m.sum() < 8:                       # small field: everything outside 3 px and the galaxy
        m = fp & (d >= 3)
        if exclude is not None:
            m &= ~exclude
    if m.sum() < 8:
        m = fp.copy()
    return m


def running_nanmedian(x, window):
    """Median over +-window//2 px ignoring NaN (no extrapolation beyond the finite values)."""
    from numpy.lib.stride_tricks import sliding_window_view
    half = window // 2
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.nanmedian(sliding_window_view(np.pad(np.asarray(x, float), half, constant_values=np.nan),
                                                2 * half + 1), axis=1)


def relative_noise(cube, wave, z, fp, window=15):
    """r(y, x): robust pixel-to-pixel scatter of every spaxel (high-pass along wavelength, +-1500
    km/s around the lines masked) relative to the median spaxel of the footprint.  Spaxels at the
    field edge, covered by fewer exposures, are 2-3x noisier than the centre; a single sigma_1 for
    all spaxels would make edge residuals look like lines.  NaN outside the footprint."""
    from scipy.ndimage import median_filter
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        med = np.nanmedian(cube, axis=0)
        filled = np.where(np.isfinite(cube), cube, med[None])
        hp = cube - median_filter(np.nan_to_num(filled), size=(window, 1, 1), mode='nearest')
        hp[_line_mask(wave, z, 1500.0)] = np.nan
        s = MAD2STD * np.nanmedian(np.abs(hp - np.nanmedian(hp, axis=0)), axis=0)
        s[np.isfinite(hp).sum(0) < 50] = np.nan
        good = fp & np.isfinite(s) & (s > 0)
        ref = np.nanmedian(s[good]) if good.any() else np.nan
    if not (np.isfinite(ref) and ref > 0):
        return np.where(fp, 1.0, np.nan)
    r = np.clip(s / ref, 0.3, 10.0)
    r = np.where(np.isfinite(r), r, np.nanmax(r) if np.isfinite(r).any() else 1.0)
    return np.where(fp, r, np.nan)


def spaxel_noise(cube, sky, floor=0.5, window=51, dead=0.2):
    """Per-wavelength 1-sigma noise of ONE spaxel: robust scatter of the sky spaxels.
    NaN (pixel not used) where fewer than half of the sky spaxels (or < 8) have data, or where the
    scatter is below `dead` x its running median - the near-constant ramps at the band edges are not
    measurements; elsewhere never below `floor` x the running median over `window` px."""
    bg = cube[:, sky]
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        med = np.nanmedian(bg, axis=1, keepdims=True)
        sd = MAD2STD * np.nanmedian(np.abs(bg - med), axis=1)
    nfin = np.isfinite(bg).sum(1)
    sd[(nfin < max(8, 0.5 * bg.shape[1])) | ~(sd > 0)] = np.nan
    if np.isfinite(sd).sum() > window:
        smooth = running_nanmedian(sd, window)
        sd = np.where(np.isfinite(sd) & (sd >= dead * smooth), np.maximum(sd, floor * smooth), np.nan)
    return sd


def _line_mask(wave, z, kms):
    near = np.zeros(wave.shape, bool)
    for r in _known_lines():
        near |= np.abs(wave / (r * (1 + z)) - 1) * C_KMS < kms
    return near


def background_spectrum(cube, wave, sky, sd1, z, cfg, c_sky=1.0):
    """Residual background per wavelength and spaxel and its 1-sigma uncertainty: a smooth pedestal
    = running median over bg_window px of the sky-spaxel median, computed WITHOUT the pixels within
    +-bg_line_kms of any emission line and then interpolated (smoothed values) across those zones and
    held constant to the band edges - no galaxy line flux can be removed and no single pixel sets it.
    (bg, bg_err); bg_err ~ 1.25 sigma_1 c / sqrt(n_sky x bg_window / 2)."""
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sub = cube[:, sky]
        raw = np.nanmedian(sub, axis=1)
        nval = np.isfinite(sub).sum(1)
    zeros = np.zeros_like(wave, dtype=float)
    near = _line_mask(wave, z, cfg.bg_line_kms)
    ok = np.isfinite(raw) & np.isfinite(sd1) & ~near
    if ok.sum() < cfg.bg_window:
        return zeros, zeros
    idx = np.arange(wave.size)
    smooth = running_nanmedian(np.where(ok, raw, np.nan), cfg.bg_window)
    fin = np.isfinite(smooth) & ok
    smooth = np.interp(idx, idx[fin], smooth[fin])          # bridged with smoothed values
    with np.errstate(all='ignore'):
        sd_s = np.interp(idx, idx[np.isfinite(sd1)], running_nanmedian(sd1, cfg.bg_window)[np.isfinite(sd1)])
        bg_err = 1.2533 * sd_s * c_sky / np.sqrt(np.maximum(np.nanmedian(nval[ok]), 1) * cfg.bg_window / 2)
    return np.where(np.isfinite(smooth), smooth, 0.0), np.where(np.isfinite(bg_err), bg_err, 0.0)


def noise_correlation(cube, sd1, sky, sizes=(1, 2, 3, 4), min_boxes=15, rel=None):
    """[(n spaxels, noise excess c, n boxes)]: robust scatter of k x k box sums of sky spaxels
    divided by their expected uncorrelated noise sigma_1 sqrt(sum r^2) (1 = uncorrelated; the
    resampled cubes give c(9) ~ 1.3-1.8).  rel: relative noise map r (None: all 1)."""
    nz, ny, nx = cube.shape
    rel = np.ones((ny, nx)) if rel is None else np.nan_to_num(rel, nan=1.0)
    out = []
    for k in sizes:
        boxes = [cube[:, y:y + k, x:x + k].reshape(nz, -1).sum(1) / np.sqrt(np.sum(rel[y:y + k, x:x + k] ** 2))
                 for y in range(ny - k + 1) for x in range(nx - k + 1) if sky[y:y + k, x:x + k].all()]
        if len(boxes) < min_boxes:
            continue
        B = np.array(boxes)
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sd = MAD2STD * np.nanmedian(np.abs(B - np.nanmedian(B, 0)), 0)
            ratio = sd / sd1
        out.append((k * k, float(np.nanmedian(ratio)), len(boxes)))
    return out


def corr_factor(ncorr, n):
    """Noise excess of an n-spaxel sum, interpolated in log n (c(1) = 1, constant beyond)."""
    nn = [1.0] + [o[0] for o in ncorr if o[0] > 1]
    rr = [1.0] + [max(o[1], 1.0) for o in ncorr if o[0] > 1]
    return float(np.interp(np.log10(max(n, 1)), np.log10(nn), rr))


NCORR_DEFAULT = [(4, 1.2, 0), (9, 1.45, 0), (16, 1.6, 0)]    # typical c(N) of the cubes, if not measurable


@dataclass
class BandNoise:
    sd1: np.ndarray                 # 1-sigma noise of one spaxel per wavelength
    bg: np.ndarray                  # residual background per spaxel and wavelength
    ncorr: list                     # [(n, c(n), n boxes)]
    err_scale: float = 1.0          # null-test calibration (all sizes together)
    n_null: int = 0
    null_nmad: float = np.nan
    n_sky: int = 0
    sky: object = None              # (ny, nx) bool sky spaxels the model was measured on
    scale_n: list = field(default_factory=list)   # [(N spaxels, err scale)] null tests by region size
    bg_err: object = None           # 1-sigma uncertainty of bg per spaxel and wavelength
    sky_src: str = 'sky'            # 'sky' (galaxy excluded) or 'all' (too few sky spaxels left)
    ncorr_src: str = 'sky'          # 'sky', 'all' or 'default'
    rel: object = None              # (ny, nx) relative noise r of every spaxel (sigma_1 is for r = 1)

    def scale(self, n):
        """Error scale of an n-spaxel region: interpolated in log N between the null-test size
        classes (constant beyond them); the overall err_scale without such a table."""
        if not self.scale_n:
            return self.err_scale
        nn = np.log10([max(a, 1) for a, _ in self.scale_n])
        return float(np.interp(np.log10(max(n, 1)), nn, [s for _, s in self.scale_n]))


def region_spectrum(cube, sel, noise, edge_margin=4):
    """Background-subtracted sum spectrum of the spaxels `sel` and its 1-sigma error
    sigma_1 sqrt(sum of r^2 over the valid spaxels) c(N) x scale(N) (noise: BandNoise; r = 1 for all
    spaxels when it has no relative noise map)."""
    patch = cube[:, sel]
    valid = erode_valid(np.isfinite(patch), edge_margin)
    n = valid.sum(1)
    npix = int(np.sum(sel))
    spec = np.where(n > 0, np.nansum(np.where(valid, patch, 0.0), 1) - noise.bg * n, np.nan)
    if noise.rel is not None and noise.rel.shape == sel.shape:
        w = np.sqrt(valid.astype(float) @ (np.nan_to_num(noise.rel[sel], nan=1.0) ** 2))
    else:
        w = np.sqrt(np.maximum(n, 1))
    err = noise.sd1 * np.where(n > 0, w, 1.0) * corr_factor(noise.ncorr, npix) * noise.scale(npix)
    return spec, np.where(n > 0, err, np.nan)


def band_noise(st, fp, centre, exclude, z, cfg, rel=None):
    """Noise model of one band: relative noise map r of the spaxels, sky spaxels (galaxy excluded;
    if fewer than min_sky remain, sigma_1 and c(N) come from all spaxels >= noise_rmin from the
    centre), sigma_1 (of an r = 1 spaxel), c(N), background."""
    shape = st.cube.shape[1:]
    rel = relative_noise(st.cube, st.wave, z, fp) if rel is None else rel
    sky = sky_mask(shape, fp, centre, cfg, exclude)
    sky_n, src = sky, 'sky'
    if exclude is not None and sky.sum() < cfg.min_sky:
        sky_n, src = sky_mask(shape, fp, centre, cfg, None), 'all'
    with np.errstate(all='ignore'):
        sd1 = spaxel_noise(st.cube / np.nan_to_num(rel, nan=1.0)[None], sky_n)
    ncorr, nsrc = noise_correlation(st.cube, sd1, sky_n, rel=rel), src
    if not any(o[0] > 1 for o in ncorr) and src == 'sky' and exclude is not None:
        ncorr, nsrc = noise_correlation(st.cube, sd1, sky_mask(shape, fp, centre, cfg, None), rel=rel), 'all'
    if not any(o[0] > 1 for o in ncorr):
        ncorr, nsrc = list(NCORR_DEFAULT), 'default'
    if cfg.bg_subtract:
        r_sky = float(np.sqrt(np.nanmean(rel[sky] ** 2))) if sky.any() else 1.0
        bg, bg_err = background_spectrum(st.cube, st.wave, sky, sd1 * r_sky, z, cfg,
                                         corr_factor(ncorr, int(sky.sum())))
    else:
        bg = bg_err = np.zeros_like(st.wave, dtype=float)
    return BandNoise(sd1=sd1, bg=bg, ncorr=ncorr, n_sky=int(sky_n.sum()), sky=sky_n, bg_err=bg_err,
                     sky_src=src, ncorr_src=nsrc, rel=rel)


def sigma_lsf_kms(band, resolving_power=None):
    R = resolving_power or BAND_RESOLUTION.get((band or '').upper(), 3500.0)
    return C_KMS / (R * 2.3548)


def _unit_gauss(x, mu, sg):
    return np.exp(-0.5 * ((x - mu) / sg) ** 2) / (np.sqrt(2 * np.pi) * sg)


def _known_lines():
    return np.array(sorted(list(REST.values()) + list(EXTRA_REST)))


def _nmad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return MAD2STD * np.median(np.abs(x - np.median(x))) if x.size else np.nan


# ======================================================================== band choice
def band_score(stack, group, z, cfg):
    """None if the main line of the group is not in the band, else (main line window complete,
    number of complete line windows, exposure) - a band where the line sits at an edge ranks lower."""
    g = GROUPS[group]
    lam_main = REST[g['main']] * (1 + z)
    if not (stack.wave_min <= lam_main <= stack.wave_max):
        return None
    pad = cfg.win_pad * (1 + z)

    def full(k):
        lam = REST[k] * (1 + z)
        return stack.wave_min <= lam - pad and lam + pad <= stack.wave_max
    t = stack.exptime_s if np.isfinite(stack.exptime_s) else 0.0
    return (int(full(g['main'])), sum(full(k) for k in g['lines']), t)


def pick_band(stacks, group, z, cfg):
    scored = [(band_score(st, group, z, cfg), b) for b, st in stacks.items()]
    scored = [(s, b) for s, b in scored if s is not None]
    return max(scored)[1] if scored else None


# ======================================================================== one group fit
class GroupData:
    """Spectrum window of one line group: covered lines, fit pixels and the tie structure."""

    def __init__(self, gkey, wave, spec, err, z, band, cfg, extra_ties=None):
        g = GROUPS[gkey]
        self.key, self.band, self.z, self.cfg = gkey, band, z, cfg
        self.main = g['main']
        self.lsf = sigma_lsf_kms(band)
        self.nclip = 0
        self.clipped_x = np.zeros(0)          # wavelengths of the pixels clipped before the fit
        good = np.isfinite(spec) & np.isfinite(err) & (err > 0)
        cov = []
        for k in g['lines']:
            lam = REST[k] * (1 + z)
            core = np.abs(wave - lam) < 3 * lam * self.lsf / C_KMS
            if core.any() and good[core].mean() >= cfg.min_cover:
                cov.append(k)
        # modelled lines: the covered ones plus their doublet partners (a tied line adds no free
        # parameter, so a partly masked member keeps the tie and is modelled on the pixels there are)
        model = set(cov)
        for k, (ref, _) in g['ties'].items():
            if k in model:
                model.add(ref)
        for k, (ref, _) in g['ties'].items():
            if ref in model:
                model.add(k)
        self.keys = [k for k in g['lines'] if k in model]
        self.partial = len(cov) < len(g['lines'])
        self.ok = self.main in cov
        if not self.ok:
            return
        lo = min(REST[k] for k in self.keys) - cfg.win_pad
        hi = max(REST[k] for k in self.keys) + cfg.win_pad
        sel = good & (wave > lo * (1 + z)) & (wave < hi * (1 + z))
        for k in g['lines']:               # an unmodelled (uncovered free) line must not sit in the window
            if k not in model:
                lam = REST[k] * (1 + z)
                sel &= np.abs(wave - lam) > (3 * np.hypot(self.lsf, cfg.sig_default) + cfg.dv_tie) * lam / C_KMS
        self.x, self.y, self.e = wave[sel], spec[sel], err[sel]
        self.x0 = float(np.mean(self.x)) if self.x.size else np.nan
        # continuum on both sides of the covered lines is needed for an honest fit
        lam_lo, lam_hi = min(REST[k] for k in cov) * (1 + z), max(REST[k] for k in cov) * (1 + z)
        side = 3 * lam_hi * self.lsf / C_KMS
        self.ok = bool(self.x.size >= 8 and (self.x < lam_lo - side).sum() >= 3
                       and (self.x > lam_hi + side).sum() >= 3)
        self.lam = {k: REST[k] * (1 + z) for k in self.keys}
        ties = {k: v for k, v in g['ties'].items() if k in self.keys and v[0] in self.keys}
        ties.update({k: v for k, v in (extra_ties or {}).items() if k in self.keys and v[0] in self.keys})
        self._set_ties(ties)

    def _set_ties(self, ties):
        self.ties = ties
        self.free = [k for k in self.keys if k not in ties]
        self.members = {f: [(f, 1.0)] + [(k, r) for k, (ref, r) in ties.items() if ref == f] for f in self.free}

    def _copy(self):
        new = object.__new__(GroupData)
        new.__dict__.update(self.__dict__)
        return new

    def subset(self, keep):
        """The same window with only the pixels `keep` (outlier clipping)."""
        new = self._copy()
        new.x, new.y, new.e = self.x[keep], self.y[keep], self.e[keep]
        new.nclip = int((~keep).sum()) + self.nclip
        new.clipped_x = np.concatenate([self.clipped_x, self.x[~keep]])
        return new

    def with_ties(self, extra):
        """A copy with additional ties (e.g. the [SII] ratio pinned to a limit)."""
        new = self._copy()
        ties = dict(self.ties)
        ties.update({k: v for k, v in extra.items() if k in self.keys and v[0] in self.keys})
        new._set_ties(ties)
        return new

    def core(self, comps, width=3.0):
        """Pixels within +-width sigma_obs (+50 km/s) of any line for the components [(v, sigma)]."""
        m = np.zeros(self.x.shape, bool)
        for k in self.keys:
            for v, s in comps:
                if not (np.isfinite(v) and np.isfinite(s)):
                    continue
                mu = self.lam[k] * (1 + v / C_KMS)
                m |= np.abs(self.x - mu) < (width * np.hypot(self.lsf, s) + 50.0) * mu / C_KMS
        return m

    def profile(self, line, v, s):
        mu = self.lam[line] * (1 + v / C_KMS)
        return _unit_gauss(self.x, mu, mu * np.hypot(self.lsf, s) / C_KMS)

    def on_sky_line(self, v):
        """True if a modelled line at velocity v falls within half an LSF FWHM of a bright OH line."""
        for k in self.keys:
            mu = self.lam[k] * (1 + v / C_KMS)
            if np.min(np.abs(SKY_UM - mu)) < 0.5 * 2.3548 * self.lsf * mu / C_KMS:
                return True
        return False

    def design(self, comps, fracs=None):
        """[1, x - x0, one column per component and free line]; with fracs (forced fit) ONE
        template column per free line."""
        cols = [np.ones_like(self.x), self.x - self.x0]
        if fracs is None:
            for v, s in comps:
                for f in self.free:
                    cols.append(sum(r * self.profile(k, v, s) for k, r in self.members[f]))
        else:
            for f in self.free:
                cols.append(sum(w * sum(r * self.profile(k, v, s) for k, r in self.members[f])
                                for (v, s), w in zip(comps, fracs)))
        return np.column_stack(cols)


@dataclass
class GroupFit:
    group: str
    band: str
    ncomp: int = 0
    forced: bool = False
    ok: bool = False
    chi2: float = np.nan
    ndata: int = 0
    nparam: int = 0
    bic: float = np.nan
    dbic_ncomp: float = np.nan       # BIC(1 component) - BIC(chosen), chi2-rescaled
    dbic_free: float = np.nan        # BIC(forced to the anchor) - BIC(free), non-anchor groups
    nclip: int = 0
    qflag: int = 0
    x0: float = np.nan
    cont: tuple = (np.nan, np.nan)
    cont_main: tuple = (np.nan, np.nan)            # continuum at the main line and its error
    comps: list = field(default_factory=list)       # [(v, v_err, s, s_err, unresolved)], narrowest first
    template: tuple = (np.nan, np.nan)              # (v, sigma) of the template of a forced fit
    comp_flux: dict = field(default_factory=dict)   # line -> [flux per component]
    comp_flux_err: dict = field(default_factory=dict)
    flux: dict = field(default_factory=dict)
    flux_err: dict = field(default_factory=dict)
    flux_cov: dict = field(default_factory=dict)    # (line a, line b) -> covariance of the total fluxes
    fracs: list = field(default_factory=list)
    clipped: list = field(default_factory=list)     # wavelengths of the clipped pixels
    fail: str = ''                                  # why a free fit was rejected ('width', 'velocity', 'spike')

    @property
    def chi2nu(self):
        return self.chi2 / max(self.ndata - self.nparam, 1)

    def snr(self, line):
        f, e = self.flux.get(line, np.nan), self.flux_err.get(line, np.nan)
        return f / e if (np.isfinite(e) and e > 0) else np.nan


def _solve(A, y, e, nonneg_from=None):
    """Weighted linear least squares; coefficients >= index nonneg_from are kept >= 0."""
    Aw, yw = A / e[:, None], y / e
    if nonneg_from is None:
        coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    else:
        lb = np.full(A.shape[1], -np.inf)
        lb[nonneg_from:] = 0.0
        coef = lsq_linear(Aw, yw, bounds=(lb, np.full(A.shape[1], np.inf)), method='bvls').x
    return coef, yw - Aw @ coef


def _finish(gd, comps, coef, A, nonlin_jac, chi2, forced, fracs=None, template=(np.nan, np.nan)):
    n = gd.x.size
    J = A / gd.e[:, None]
    if nonlin_jac is not None and nonlin_jac.size:
        J = np.column_stack([J, nonlin_jac])
    npar = J.shape[1]
    dof = max(n - npar, 1)
    with np.errstate(all='ignore'):
        cov = np.linalg.pinv(J.T @ J) * max(chi2 / dof, 1.0)
    gf = GroupFit(group=gd.key, band=gd.band, ncomp=len(comps), forced=forced, ok=True, chi2=float(chi2),
                  ndata=n, nparam=npar, bic=float(chi2 + npar * np.log(n)), x0=gd.x0, nclip=gd.nclip,
                  cont=(float(coef[0]), float(coef[1])), template=template)
    d = gd.lam[gd.main] - gd.x0
    var_c = cov[0, 0] + d * d * cov[1, 1] + 2 * d * cov[0, 1]
    gf.cont_main = (float(coef[0] + coef[1] * d), float(np.sqrt(var_c)) if var_c >= 0 else np.nan)
    nf = len(gd.free)
    nl = len(coef)
    C = cov[nl:, nl:] if npar > nl else np.zeros((0, 0))
    for k, (v, s) in enumerate(comps):
        unres = bool(s < 0.05 * gd.lsf)
        if forced or C.shape[0] < 2 * k + 2:
            gf.comps.append((float(v), np.nan, float(s), np.nan, unres))
            continue
        si = [2 * j + 1 for j in range(k + 1)]          # s_k = s_1 + ds_2 + ... + ds_k
        var_s = float(np.sum(C[np.ix_(si, si)]))
        ve = float(np.sqrt(C[2 * k, 2 * k])) if C[2 * k, 2 * k] >= 0 else np.nan
        se = float(np.sqrt(var_s)) if (var_s >= 0 and not unres) else np.nan
        gf.comps.append((float(v), ve, float(s), se, unres))
    gvec = {}                                           # total flux of a line = gvec[line] . parameters
    for f in gd.free:
        for line, r in gd.members[f]:
            g = np.zeros(npar)
            if fracs is None:
                ii = [2 + k * nf + gd.free.index(f) for k in range(len(comps))]
                per = [r * coef[i] for i in ii]
                per_err = [abs(r) * np.sqrt(cov[i, i]) if cov[i, i] >= 0 else np.nan for i in ii]
                g[ii] = r
            else:
                i = 2 + gd.free.index(f)
                per = [r * coef[i] * w for w in fracs]
                per_err = [abs(r * w) * np.sqrt(cov[i, i]) if cov[i, i] >= 0 else np.nan for w in fracs]
                g[i] = r * float(np.sum(fracs))
            var = float(g @ cov @ g)
            gvec[line] = g
            gf.comp_flux[line] = [float(p) for p in per]
            gf.comp_flux_err[line] = [float(p) for p in per_err]
            gf.flux[line] = float(np.sum(per))
            gf.flux_err[line] = float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else np.nan
    for a in gvec:                                      # covariances (tied lines: fully correlated)
        for b in gvec:
            if a < b:
                gf.flux_cov[(a, b)] = gf.flux_cov[(b, a)] = float(gvec[a] @ cov @ gvec[b])
    gf.clipped = [float(x) for x in getattr(gd, 'clipped_x', [])]
    main = gf.comp_flux.get(gd.main)
    if main is not None:
        tot = np.sum(main)
        gf.fracs = [float(m / tot) if tot != 0 else 1.0 / len(main) for m in main]
    if gf.chi2nu > gd.cfg.chi2nu_max:
        gf.qflag |= QF_CHI2
    if gd.nclip:
        gf.qflag |= QF_CLIPPED
    if any(c[4] for c in gf.comps) and not forced:
        gf.qflag |= QF_UNRESOLVED
    if getattr(gd, 'partial', False):
        gf.qflag |= QF_PARTIAL
    return gf


def fit_forced(gd, comps, fracs):
    """Fluxes with the kinematics FIXED to a template: one free amplitude per line (linear)."""
    A = gd.design(comps, fracs)
    coef, r = _solve(A, gd.y, gd.e)
    tpl_v = float(np.sum([w * c[0] for c, w in zip(comps, fracs)])) if comps else np.nan
    tpl_s = float(comps[0][1]) if len(comps) == 1 else float(np.nan)
    return _finish(gd, comps, coef, A, None, float(np.sum(r ** 2)), True, fracs, template=(tpl_v, tpl_s))


def fit_free(gd, ncomp, cfg, v0=0.0, s0=80.0, vlim=None, s_min=0.0):
    """Free fit with `ncomp` Gaussian components (variable projection, several starts).
    Components are ordered by width (s_k = s_1 + ds_2 + ...).  A velocity on its limit, a width on
    its upper limit or an extra component narrower than sig_min_frac x LSF marks the fit as failed
    (it chases noise, a band edge or a spike); the caller then tries other component numbers."""
    vlo, vhi = vlim if vlim is not None else (-cfg.dv_max, cfg.dv_max)
    v0 = float(np.clip(v0, vlo, vhi))
    s_extra_min = cfg.sig_min_frac * gd.lsf
    lb = [vlo, s_min] + [vlo, 0.0] * (ncomp - 1)
    ub = [vhi, cfg.sig_int_max] + [vhi, cfg.sig_extra_max] * (ncomp - 1)
    nonneg = None if ncomp == 1 else 2

    def comps_of(t):
        out, s = [], 0.0
        for k in range(ncomp):
            s = t[1] if k == 0 else s + t[2 * k + 1]
            out.append((t[2 * k], s))
        return out

    def resid(t):
        return _solve(gd.design(comps_of(t)), gd.y, gd.e, nonneg)[1]

    s0 = float(np.clip(s0, max(5.0, s_min), cfg.sig_int_max * 0.9))
    dv = min(1.2 * s0, 0.5 * (vhi - vlo))
    if ncomp == 1:
        starts = [[v0, s0], [v0, max(30.0, 0.5 * s0, s_min)], [v0, min(2 * s0, 0.9 * cfg.sig_int_max)]]
        # chi2(v) on a velocity grid (emission only, linear solves): the two best minima as extra
        # starts, so a line offset from the expected velocity is not missed (local minima)
        sg = max(s0, s_min)
        vg = np.linspace(vlo, vhi, 33)[1:-1]
        chi = np.array([np.sum(_solve(gd.design([(v, sg)]), gd.y, gd.e, nonneg_from=2)[1] ** 2) for v in vg])
        mins = [i for i in range(len(vg)) if (i == 0 or chi[i] <= chi[i - 1]) and (i == len(vg) - 1 or chi[i] <= chi[i + 1])]
        for i in sorted(mins, key=lambda i: chi[i])[:2]:
            if abs(vg[i] - v0) > 0.5 * (vg[1] - vg[0]):
                starts.append([vg[i], sg])
    elif ncomp == 2:
        starts = [[v0, 0.8 * s0, v0, 0.7 * s0], [v0, 0.8 * s0, v0 - 0.3 * dv, 250.0],
                  [v0 - dv, 0.7 * s0, v0 + dv, 5.0], [v0 + dv, 0.7 * s0, v0 - dv, 5.0]]
    else:
        starts = [[v0, 0.7 * s0, v0, 0.5 * s0, v0 - 0.3 * dv, 300.0],
                  [v0 - dv, 0.6 * s0, v0 + dv, 5.0, v0, 300.0],
                  [v0, 0.7 * s0, v0 - 0.5 * dv, 0.5 * s0, v0, 400.0]]
    best = None
    for t0 in starts:
        t0 = np.clip(np.asarray(t0, float), np.array(lb) + 1e-6, np.array(ub) - 1e-6)
        try:
            res = least_squares(resid, t0, bounds=(lb, ub), x_scale='jac', max_nfev=300 * len(t0))
        except (ValueError, np.linalg.LinAlgError):
            continue
        chi2 = float(np.sum(res.fun ** 2))
        if np.isfinite(chi2) and (best is None or chi2 < best[0]):
            best = (chi2, res)
    if best is None:
        return GroupFit(group=gd.key, band=gd.band, ok=False)
    chi2, res = best
    x = res.x
    tol = 1e-3 * (np.array(ub) - np.array(lb))
    fail = ''
    for k in range(ncomp):
        if abs(x[2 * k] - lb[2 * k]) < tol[2 * k] or abs(x[2 * k] - ub[2 * k]) < tol[2 * k]:
            fail = fail or 'velocity'                         # velocity on its limit
        if abs(x[2 * k + 1] - ub[2 * k + 1]) < tol[2 * k + 1]:
            fail = fail or 'width'                            # width on its upper limit
    comps = comps_of(x)
    if ncomp > 1 and any(s < s_extra_min for _, s in comps):
        fail = fail or 'spike'                                # a component of a multi-Gaussian fit as narrow as a spike
    if cfg.sky_guard and SKY_UM.size and any(gd.on_sky_line(v) for v, s in comps if s < 0.5 * gd.lsf):
        fail = fail or 'sky'                                  # unresolved and on a bright OH line: sky residual
    if fail:
        return GroupFit(group=gd.key, band=gd.band, ok=False, fail=fail)
    A = gd.design(comps)
    coef, _ = _solve(A, gd.y, gd.e, nonneg)
    jac = []
    for i in range(len(x)):
        tp, tm = x.copy(), x.copy()
        tp[i] += 1.0
        tm[i] -= 1.0
        jac.append((gd.design(comps_of(tp)) @ coef - gd.design(comps_of(tm)) @ coef) / 2.0 / gd.e)
    return _finish(gd, comps, coef, A, np.column_stack(jac), chi2, False)


def fit_group_best(gd, cfg, v0=0.0, s0=80.0, vlim=None, s_min=0.0, base=None):
    """1..max_components free fits.  One more component only if the BIC - chi2 rescaled by the
    reduced chi2 of the simpler model - drops by >= bic_min and EVERY component's own main-line flux
    has S/N >= comp_snr_min.  base: the fit (e.g. forced) compared with when 1 component fails."""
    fits = {}
    f1 = fit_free(gd, 1, cfg, v0, s0, vlim, s_min)
    if f1.ok:
        fits[1] = f1
    best = f1 if f1.ok else None
    for n in range(2, cfg.max_components + 1):
        prev = best if best is not None else base
        seed = prev.comps[0] if (prev is not None and prev.ok and prev.comps) else (v0, 0, s0)
        sd = seed[2] if np.isfinite(seed[2]) else s0
        f = fit_free(gd, n, cfg, seed[0] if np.isfinite(seed[0]) else v0, sd, vlim, s_min)
        if not f.ok:
            if best is not None and best.ncomp == n - 1:
                break
            continue
        fits[n] = f
        ref = best if best is not None else base
        if ref is None or not ref.ok:
            continue
        scale = max(1.0, ref.chi2nu)
        dbic = (ref.chi2 - f.chi2) / scale - (f.nparam - ref.nparam) * np.log(f.ndata)
        comp_ok = all(np.isfinite(e) and e > 0 and m / e >= cfg.comp_snr_min
                      for m, e in zip(f.comp_flux.get(gd.main, []), f.comp_flux_err.get(gd.main, [])))
        if dbic >= cfg.bic_min and comp_ok:
            best = f
        elif best is not None and best.ncomp == n - 1:
            break
    if best is None:
        return GroupFit(group=gd.key, band=gd.band, ok=False, fail=f1.fail)
    if 1 in fits and best is not fits[1]:
        scale = max(1.0, fits[1].chi2nu)
        best.dbic_ncomp = float((fits[1].chi2 - best.chi2) / scale
                                - (best.nparam - fits[1].nparam) * np.log(best.ndata))
    elif best.ncomp == 1:
        best.dbic_ncomp = 0.0
    return best


def _model_of(gd, gf):
    """Model of a fitted group on the window pixels (from its fluxes)."""
    y = gf.cont[0] + gf.cont[1] * (gd.x - gd.x0)
    for line in gd.keys:
        fl = gf.comp_flux.get(line, [])
        for k, c in enumerate(gf.comps):
            if k < len(fl) and np.isfinite(fl[k]):
                y = y + fl[k] * gd.profile(line, c[0], c[2])
    return y


def clip_mask(gd, gf, cfg, max_frac=0.1):
    """Pixels to keep after a fit.  Pulls are measured in units of their robust scatter (so
    underestimated errors do not clip half the window); clipped are single-pixel spikes
    (|pull| > clip_sigma, both neighbours < clip_sigma / 2) anywhere and |pull| > clip_sigma
    (+-1 px) away from the fitted lines - at most max_frac of the window, the worst first."""
    pull = (gd.y - _model_of(gd, gf)) / gd.e
    pull = pull / max(1.0, float(_nmad(pull)))
    big = np.abs(pull) > cfg.clip_sigma
    if not big.any():
        return np.ones(pull.shape, bool)
    nb = np.zeros_like(big)
    nb[1:] |= np.abs(pull[:-1]) > cfg.clip_sigma / 2
    nb[:-1] |= np.abs(pull[1:]) > cfg.clip_sigma / 2
    spike = big & ~nb
    away = binary_dilation(big, iterations=1) & ~gd.core([(c[0], c[2]) for c in gf.comps])
    cand = np.flatnonzero(spike | away)
    nmax = max(2, int(max_frac * pull.size))
    if cand.size > nmax:
        cand = cand[np.argsort(-np.abs(pull[cand]))[:nmax]]
    keep = np.ones(pull.shape, bool)
    keep[cand] = False
    return keep


def _sii_bounded(gd, gf, refit):
    """[SII] 6716/6731 outside its physical range: refit with the ratio pinned to that limit."""
    if gd.key != 'sii' or not gf.ok or 'SII6731' not in gf.flux or 'SII6716' not in gf.flux:
        return gf
    a, b = gf.flux['SII6716'], gf.flux['SII6731']
    if not (b > 0 and a > 0) or SII_RANGE[0] <= a / b <= SII_RANGE[1]:
        return gf
    lim = SII_RANGE[0] if a / b < SII_RANGE[0] else SII_RANGE[1]
    g2 = refit(gd.with_ties({'SII6716': ('SII6731', lim)}))
    if g2.ok:
        g2.qflag |= QF_SII_BOUND
        g2.dbic_free = gf.dbic_free
        return g2
    return gf


def _broad_flag(forced, free, gd, cfg):
    """Forced fit used because the free single Gaussian ran into its width limit while the forced
    fit sees the line: the line is broader than the model allows, the forced flux a lower limit."""
    if not free.ok and free.fail == 'width' and forced.snr(gd.main) >= cfg.ratio_snr_min:
        forced.qflag |= QF_BROAD
    return forced


def _choose_anchor(gd, cfg, template, allow_free, vlim=None):
    forced = fit_forced(gd, *template)
    if not allow_free:
        return forced
    v0, s0 = template[0][0]
    free = fit_group_best(gd, cfg, v0, s0, vlim=vlim, base=forced)
    if free.ok and free.snr(gd.main) >= cfg.force_below:
        return free
    return _broad_flag(forced, free, gd, cfg)


def fit_anchor(gd, cfg, template, allow_free=True, vlim=None):
    """Anchor group of one bin: free (1..max_components) if the line is detected, else forced to
    `template` ([(v, sigma)], fracs); fitted, clipped against that fit, refitted.  vlim: velocity
    range of the free components (default +-dv_max)."""
    fit = _choose_anchor(gd, cfg, template, allow_free, vlim)
    keep = clip_mask(gd, fit, cfg)
    if (~keep).any() and keep.sum() >= 8:
        gdc = gd.subset(keep)
        if gdc.ok:
            fit = _choose_anchor(gdc, cfg, template, allow_free, vlim)
    return fit


def fit_other(gd, cfg, template, anchor_detected):
    """Non-anchor group: forced to the anchor template of its bin; its own kinematics only when
    the anchor is detected, the line is strong (S/N >= force_below), stays within +-dv_tie of the
    anchor and the data demand it (BIC better than the forced fit by >= bic_min)."""
    forced = fit_forced(gd, *template)
    keep = clip_mask(gd, forced, cfg)
    if (~keep).any() and keep.sum() >= 8 and gd.subset(keep).ok:
        gd = gd.subset(keep)
        forced = fit_forced(gd, *template)

    def refit_forced(g):
        return fit_forced(g, *template)
    forced = _sii_bounded(gd, forced, refit_forced)
    if not anchor_detected:
        return forced
    v0, s0 = template[0][0]
    vlim = (v0 - cfg.dv_tie, v0 + cfg.dv_tie)
    s_min = cfg.sig_min_frac * gd.lsf

    def refit_free(g):
        return fit_group_best(g, cfg, v0, s0, vlim, s_min, base=fit_forced(g, *template))
    free = refit_free(gd)
    if free.ok:
        scale = max(1.0, forced.chi2nu)
        free.dbic_free = float((forced.chi2 - free.chi2) / scale - (free.nparam - forced.nparam) * np.log(free.ndata))
        if free.snr(gd.main) >= cfg.force_below and free.dbic_free >= cfg.bic_min:
            return _sii_bounded(gd, free, refit_free)
        forced.dbic_free = free.dbic_free
    return _broad_flag(forced, free, gd, cfg)


# ======================================================================== bins
def line_sn_map(cube, wave, sd1, z, rest, sig_kms, band, fp, rel=None):
    """Continuum-subtracted narrow-band signal of one line per spaxel and its noise (+-3 sigma_obs;
    continuum = median of line-free side bands 1500-4000 km/s away).  (None, None) if not covered."""
    lam = rest * (1 + z)
    sg = lam * np.hypot(sigma_lsf_kms(band), sig_kms) / C_KMS
    line = np.abs(wave - lam) <= 3.0 * sg
    dv = (wave / lam - 1) * C_KMS
    side = (np.abs(dv) > 1500) & (np.abs(dv) < 4000) & ~_line_mask(wave, z, 1000.0)
    if line.sum() < 2 or side.sum() < 10:
        return None, None
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cont = np.nanmedian(cube[side], axis=0)
        fin = np.isfinite(cube[line]) & np.isfinite(sd1[line])[:, None, None]
        S = np.sum(np.where(fin, cube[line] - cont, 0.0), axis=0)
        N = np.sqrt(np.sum(np.where(fin, sd1[line][:, None, None] ** 2, 0.0), axis=0))
        if rel is not None:                    # noisier spaxels (field edge) get their own noise
            N = N * np.nan_to_num(rel, nan=np.inf)
    bad = (fin.sum(0) < 0.7 * line.sum()) | ~np.isfinite(cont) | ~fp | ~np.isfinite(N) | (N <= 0)
    S[bad] = np.nan
    N[bad] = np.nan
    return S, N


def find_centre(S, fp, max_offset=4.0, min_nbr=0.08):
    """Brightest footprint spaxel of the line map with real neighbours, within max_offset of the
    grid centre (notebooks' ha_peak_center); else the grid centre.  ((y, x), found)."""
    ny, nx = S.shape
    cy, cx = ny // 2, nx // 2
    im = np.where(fp, S, np.nan)
    flat = im.ravel()
    for idx in np.argsort(np.where(np.isfinite(flat), flat, -np.inf))[::-1][:60]:
        v = flat[idx]
        if not np.isfinite(v) or v <= 0:
            break
        py, px = np.unravel_index(idx, (ny, nx))
        if np.hypot(py - cy, px - cx) > max_offset:
            continue
        nb = [im[py + a, px + b] for a in (-1, 0, 1) for b in (-1, 0, 1)
              if (a, b) != (0, 0) and 0 <= py + a < ny and 0 <= px + b < nx]
        nb = [q for q in nb if np.isfinite(q)]
        if nb and np.mean(nb) >= min_nbr * v:
            return (int(py), int(px)), True
    return (cy, cx), False


def galaxy_region(S, N, centre, fp, cfg):
    """Connected S/N >= min_sn part of the footprint that contains the centre, else the one coming
    closest to it (within max_offset).  (mask or None, distance px, found)."""
    with np.errstate(all='ignore'):
        good = fp & np.isfinite(S) & np.isfinite(N) & (N > 0) & (S / N >= cfg.min_sn)
    lab, nlab = label(good)
    if lab[centre] > 0:
        return lab == lab[centre], 0.0, True
    yy, xx = np.indices(S.shape)
    d = np.hypot(yy - centre[0], xx - centre[1])
    best = None
    for c in range(1, nlab + 1):
        m = lab == c
        key = (float(d[m].min()), -float(np.nansum(S[m])))
        if key[0] <= cfg.max_offset and (best is None or key < best[0]):
            best = (key, c)
    if best is None:
        return None, np.nan, False
    return lab == best[1], best[0][0], True


def make_bins(S, N, region, ncorr, cfg):
    """(binmap with -1 = not binned, target S/N reached or NaN, mode)."""
    ny, nx = S.shape
    bm = np.full((ny, nx), -1, int)
    yy, xx = np.nonzero(region)
    per_spaxel = bm.copy()
    per_spaxel[yy, xx] = np.arange(yy.size)
    single = bm.copy()
    single[yy, xx] = 0
    if yy.size == 1:
        return single, np.nan, 'single'
    if cfg.binning == 'spaxel':
        return per_spaxel, np.nan, 'spaxel'
    if cfg.binning == 'grid':
        k = cfg.grid
        ids = {}
        for y, x in zip(yy, xx):
            bm[y, x] = ids.setdefault((y // k, x // k), len(ids))
        return bm, np.nan, f'grid{k}'
    with np.errstate(all='ignore'):
        sn_pix = S[region] / N[region]
    if not np.all(np.isfinite(sn_pix)):
        return single, np.nan, 'single'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from vorbin.voronoi_2d_binning import voronoi_2d_binning

    def sn_func(index, signal, noise):
        return np.sum(signal[index]) / (np.sqrt(np.sum(noise[index] ** 2)) * corr_factor(ncorr, np.size(index)))
    best = None
    for target in (cfg.target_sn,) + tuple(t for t in cfg.target_fallback if t < cfg.target_sn):
        if np.all(sn_pix >= target):              # every spaxel above the target: no binning needed
            return per_spaxel, target, 'spaxel'
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                bin_num, *_ = voronoi_2d_binning(xx.astype(float), yy.astype(float), S[region], N[region],
                                                 target, pixelsize=1.0, plot=False, quiet=True, sn_func=sn_func)
        except (ValueError, IndexError, ZeroDivisionError) as e:
            if 'binning is not needed' in str(e):
                return per_spaxel, target, 'spaxel'
            continue                               # not enough S/N in the whole region: lower target
        b = bm.copy()
        b[yy, xx] = np.asarray(bin_num, int)
        best = (b, target)
        if b.max() + 1 >= cfg.min_bins:
            break
    if best is None:
        return single, np.nan, 'single'
    return best[0], best[1], 'voronoi' if best[0].max() > 0 else 'single'


def transfer_binmap(binmap, wcs_from, shape_to, wcs_to, fp_to=None, sub=5):
    """Carry the bins to another grid through the WCS: every target spaxel is sampled at sub x sub
    points and takes the bin covering most of it (if > half of it is binned); a bin left without
    any spaxel gets the free target spaxel nearest to its centroid."""
    nb = int(binmap.max()) + 1 if (binmap >= 0).any() else 0
    if wcs_from is None or wcs_to is None:
        return binmap.copy() if binmap.shape == tuple(shape_to) else np.full(shape_to, -1, int)
    ny, nx = shape_to
    off = (np.arange(sub) + 0.5) / sub - 0.5
    yy, xx = np.indices((ny, nx))
    ys = (yy[..., None, None] + off[:, None] + 0 * off[None, :]).ravel()
    xs = (xx[..., None, None] + 0 * off[:, None] + off[None, :]).ravel()
    ra, dec = wcs_to.pixel_to_world_values(xs, ys)
    px, py = wcs_from.world_to_pixel_values(ra, dec)
    ix, iy = np.rint(px).astype(int), np.rint(py).astype(int)
    inside = (ix >= 0) & (iy >= 0) & (ix < binmap.shape[1]) & (iy < binmap.shape[0])
    lab = np.full(ix.size, -1, int)
    lab[inside] = binmap[iy[inside], ix[inside]]
    lab = lab.reshape(ny * nx, sub * sub)
    out = np.full(ny * nx, -1, int)
    for i, row in enumerate(lab):
        vals = row[row >= 0]
        if vals.size * 2 > sub * sub:
            out[i] = np.bincount(vals).argmax()
    out = out.reshape(ny, nx)
    if fp_to is not None:
        out[~fp_to] = -1
    have = set(np.unique(out[out >= 0]).tolist())
    yy_t, xx_t = np.indices((ny, nx))
    for b in range(nb):
        if b in have:
            continue
        ys_b, xs_b = np.nonzero(binmap == b)
        ra, dec = wcs_from.pixel_to_world_values(xs_b.mean(), ys_b.mean())
        tx, ty = wcs_to.world_to_pixel_values(ra, dec)
        free = (out < 0) if fp_to is None else ((out < 0) & fp_to)
        if not free.any():
            continue
        d = np.hypot(yy_t - float(ty), xx_t - float(tx))
        d[~free] = np.inf
        j = np.unravel_index(np.argmin(d), d.shape)
        if d[j] <= 1.5:                            # only if the bin really lies on this grid
            out[j] = b
    return out


# ======================================================================== null tests
def _null_snrs(wave, spec, err, z, rest, sig_kms, band, cfg, offsets):
    """Forced fits of an anchor-shaped line at line-free wavelengths: their S/N must be N(0, 1)."""
    out = []
    known = _known_lines() * (1 + z)
    lsf = sigma_lsf_kms(band)
    half = cfg.win_pad * (1 + z) * 0.8
    for off in offsets:
        lam = rest * (1 + z) * (1 + off / C_KMS)
        sg = lam * np.hypot(lsf, sig_kms) / C_KMS
        if np.min(np.abs(known - lam)) < half + 5 * sg:
            continue
        sel = (np.abs(wave - lam) < half) & np.isfinite(spec) & np.isfinite(err) & (err > 0)
        core = np.abs(wave - lam) < 2 * sg
        if (sel.sum() < 10 or not core.any() or sel[core].mean() < cfg.min_cover
                or (sel & (wave < lam - 2 * sg)).sum() < 3 or (sel & (wave > lam + 2 * sg)).sum() < 3):
            continue                       # the null line must lie on data, with continuum on both sides
        x, y, e = wave[sel], spec[sel], err[sel]
        A = np.column_stack([np.ones_like(x), x - x.mean(), _unit_gauss(x, lam, sg)])
        coef, _ = _solve(A, y, e)
        Aw = A / e[:, None]
        with np.errstate(all='ignore'):
            cov = np.linalg.pinv(Aw.T @ Aw)
        if cov[2, 2] > 0:
            out.append(coef[2] / np.sqrt(cov[2, 2]))
    return out


def _dense_offsets():
    """Null-test offsets for galaxies with few bins: every 600 km/s within +-9000 km/s."""
    return tuple(float(o) for o in np.arange(-9000.0, 9001.0, 600.0) if abs(o) >= 2400.0)


def _box_nulls(st, nz, z, rest, sig_kms, band, cfg, k=3, max_boxes=30):
    """Null tests (dense offsets) on non-overlapping k x k boxes of sky spaxels (packed greedily):
    the noise of bin-sized regions without galaxy light.  nz.err_scale must still be 1."""
    if nz.sky is None:
        return []
    ny, nx = nz.sky.shape
    free = nz.sky.copy()
    out, n = [], 0
    for y in range(ny - k + 1):
        for x in range(nx - k + 1):
            if n >= max_boxes:
                return out
            if not free[y:y + k, x:x + k].all():
                continue
            free[y:y + k, x:x + k] = False
            sel = np.zeros((ny, nx), bool)
            sel[y:y + k, x:x + k] = True
            s, e = region_spectrum(st.cube, sel, nz, cfg.edge_margin)
            out += _null_snrs(st.wave, s, e, z, rest, sig_kms, band, cfg, _dense_offsets())
            n += 1
    return out


def _free_null_snrs(wave, spec, err, z, anchor, band, cfg, template):
    """The detection fit (fit_anchor: free, clipped, 1..max_components) repeated at line-free
    wavelengths of the SAME spectrum (offsets >= 4500 km/s, no known line inside the window):
    a detection must beat all of them (look-elsewhere effect, continuum structure, sky residuals)."""
    g = GROUPS[anchor]
    known = _known_lines() * (1 + z)
    lo_rest = min(REST[k] for k in g['lines']) - cfg.win_pad
    hi_rest = max(REST[k] for k in g['lines']) + cfg.win_pad
    out = []
    for off in np.arange(-15000.0, 15001.0, 1500.0):
        if abs(off) < 4500.0:
            continue
        zo = (1 + z) * (1 + off / C_KMS) - 1
        lo, hi = lo_rest * (1 + zo) * (1 - 600 / C_KMS), hi_rest * (1 + zo) * (1 + 600 / C_KMS)
        if np.any((known > lo) & (known < hi)):
            continue
        gd = GroupData(anchor, wave, spec, err, zo, band, cfg)
        if not gd.ok:
            continue
        forced = fit_forced(gd, *template)    # the best FREE fit, whatever its S/N (no censoring at 5)
        free = fit_group_best(gd, cfg, template[0][0][0], template[0][0][1], base=forced)
        snr = free.snr(g['main']) if free.ok else forced.snr(g['main'])
        if np.isfinite(snr):
            out.append(float(snr))
    return out


def _detected(fit, main, free_nulls, cfg):
    """(detected, max null S/N) of a free reference fit: S/N / max(1, max(null S/N) / det_null_ref)
    >= force_below - the line-free windows of the same spectrum set the effective noise when they
    show more than pure noise would (det_null_ref = typical largest free-fit S/N of pure noise in ~8
    windows) - and its narrowest component no broader than sig_det_max (a broader single 'line' is
    continuum structure absorbed by a Gaussian)."""
    nm = max(free_nulls) if free_nulls else np.nan
    s = max(1.0, nm / cfg.det_null_ref) if np.isfinite(nm) else 1.0
    snr = fit.snr(main)
    narrow = bool(fit.comps) and fit.comps[0][2] <= cfg.sig_det_max
    return bool(np.isfinite(snr) and snr / s >= cfg.force_below and narrow), nm


SIZE_CLASSES = ((1, 1), (2, 3), (4, 8), (9, 16), (17, 40), (41, 100), (101, 100000))


def _scale_table(pairs, cfg):
    """[(N, NMAD)] of the null-test S/N in size classes of the summed region (each needs
    >= max(10, null_min / 2) tests; NMAD clipped to err_scale_range) - errors of big bins and of
    the TOTAL spectrum are calibrated on regions of their own size."""
    out = []
    need = max(10, cfg.null_min // 2)
    for lo, hi in SIZE_CLASSES:
        sub = [(n, s) for n, s in pairs if lo <= n <= hi and np.isfinite(s)]
        if len(sub) >= need:
            out.append((float(np.median([n for n, _ in sub])),
                        float(np.clip(_nmad([s for _, s in sub]), *cfg.err_scale_range))))
    return out


def _apply_nulls(nz, nulls, cfg):
    """err_scale = NMAD of the null-test S/N (clipped to err_scale_range) if there are enough."""
    nz.n_null = len(nulls)
    nz.null_nmad = float(_nmad(nulls)) if nulls else np.nan
    if len(nulls) >= cfg.null_min and np.isfinite(nz.null_nmad):
        nz.err_scale = float(np.clip(nz.null_nmad, *cfg.err_scale_range))
        return True
    return False


# ======================================================================== one galaxy
def run_galaxy(stacks, z, cfg=None, name='', anchor=None, progress=None):
    """Bins, fits and maps of one galaxy.  stacks: {band: BandStack} (empower_explorer.data.stack:
    wave, cube, wcs, exptime_s, coverage, wave_min/max, band, bunit, header0, paths).  Returns the
    product dict for write_product."""
    cfg = cfg or Config()
    t0 = time.time()
    if z is None or not np.isfinite(z) or z <= 0:
        raise ValueError('no redshift')
    groups = list(cfg.groups)
    if anchor is None:                # Ha, else [OIII], else Hb - independent of the groups asked for
        anchor = next((a for a in ANCHORS if pick_band(stacks, a, z, cfg)), None)
    if anchor is None:
        raise ValueError('none of Hα, [OIII], Hβ is covered by the bands')
    if anchor not in groups:
        groups = [anchor] + groups
    aband = pick_band(stacks, anchor, z, cfg)
    if aband is None:
        raise ValueError(f'{GROUPS[anchor]["label"]} is not covered by the bands')
    A = stacks[aband]
    main = GROUPS[anchor]['main']
    rest_a = REST[main]
    fps = {b: footprint(st.coverage, cfg) for b, st in stacks.items()}
    rels = {}                         # relative noise per spaxel; noisy edges (r > rel_max) not used
    for b, st in stacks.items():
        rel = relative_noise(st.cube, st.wave, z, fps[b])
        quiet = fps[b] & (np.nan_to_num(rel, nan=np.inf) <= cfg.rel_max)
        if quiet.sum() >= 0.5 * fps[b].sum():
            fps[b] = quiet
        rels[b] = np.where(fps[b], rel, np.nan)
    fp_a = fps[aband]
    ny, nx = A.cube.shape[1:]
    if not fp_a.any():
        raise ValueError(f'no spaxels with data in {aband}')
    yy, xx = np.indices((ny, nx))

    # 1. centre (anchor-line peak near the grid centre) and the noise around it
    grid_c = (ny // 2, nx // 2)
    noise_a = band_noise(A, fp_a, grid_c, None, z, cfg, rels[aband])
    S0, _ = line_sn_map(A.cube, A.wave, noise_a.sd1, z, rest_a, cfg.sig_default, aband, fp_a, noise_a.rel)
    centre, centred = find_centre(S0, fp_a, cfg.max_offset) if S0 is not None else (grid_c, False)
    if centre != grid_c:
        noise_a = band_noise(A, fp_a, centre, None, z, cfg, rels[aband])

    # 2. error scale of the anchor band BEFORE any decision (null tests in the central aperture and
    #    in sky boxes), then the reference kinematics: central aperture, then the whole galaxy region
    #    (REFSRC), each fitted, clipped and refitted like a bin; detected = S/N >= force_below (calibrated)
    ap = (np.hypot(yy - centre[0], xx - centre[1]) <= cfg.centre_aperture) & fp_a
    if not ap.any():
        ap = fp_a & (np.hypot(yy - centre[0], xx - centre[1]) <= np.hypot(yy - centre[0], xx - centre[1])[fp_a].min())
    z_ref, sig_ref, ref_snr, ref_src = z, cfg.sig_default, np.nan, 'default'
    err_scale_det = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if cfg.null_test:
            spec, err = region_spectrum(A.cube, ap, noise_a, cfg.edge_margin)
            nulls = _null_snrs(A.wave, spec, err, z, rest_a, cfg.sig_default, aband, cfg, _dense_offsets())
            nulls += _box_nulls(A, noise_a, z, rest_a, cfg.sig_default, aband, cfg)
            _apply_nulls(noise_a, nulls, cfg)
            err_scale_det = noise_a.err_scale
        spec, err = region_spectrum(A.cube, ap, noise_a, cfg.edge_margin)
        gd = GroupData(anchor, A.wave, spec, err, z, aband, cfg)
        null_max = np.nan
        if gd.ok:
            tpl0 = ([(0.0, cfg.sig_default)], [1.0])
            f = fit_anchor(gd, cfg, tpl0)
            ref_snr = f.snr(main) if f.ok else np.nan
            if f.ok and not f.forced:
                ok, null_max = _detected(f, main, _free_null_snrs(A.wave, spec, err, z, anchor, aband, cfg, tpl0), cfg)
                v50, w80, _, _ = profile_stats(f.comps, f.fracs)
                if np.isfinite(v50) and ok:
                    z_ref = (1 + z) * (1 + v50 / C_KMS) - 1
                    sig_ref = float(np.clip(w80 / FWHM_W80 if f.ncomp > 1 else f.comps[0][2], 20.0, cfg.sig_int_max))
                    ref_src = 'aperture'

    def regions(zz, ss, sd1):
        S, N = line_sn_map(A.cube, A.wave, sd1, zz, rest_a, ss, aband, fp_a, noise_a.rel)
        if S is None:
            nan = np.full((ny, nx), np.nan)
            return nan, nan.copy(), None, np.nan, False
        reg, dist, found = galaxy_region(S, N, centre, fp_a, cfg)
        return S, N, reg, dist, found

    S, N, region, reg_dist, found = regions(z_ref, sig_ref, noise_a.sd1)
    if found and region.sum() > ap.sum():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            spec, err = region_spectrum(A.cube, region, noise_a, cfg.edge_margin)
            gd = GroupData(anchor, A.wave, spec, err, z_ref, aband, cfg)
            tpl1 = ([(0.0, sig_ref)], [1.0])
            ft = fit_anchor(gd, cfg, tpl1) if gd.ok else GroupFit(anchor, aband)
            if ft.ok and not ft.forced:
                ok, nm = _detected(ft, main, _free_null_snrs(A.wave, spec, err, z_ref, anchor, aband, cfg, tpl1), cfg)
                if not ok:                             # not convincingly above the line-free windows
                    ft = GroupFit(anchor, aband)
                null_max = float(np.nanmax([null_max, nm])) if np.isfinite([null_max, nm]).any() else np.nan
        if ft.ok and not ft.forced:
            v50, w80, _, _ = profile_stats(ft.comps, ft.fracs)
            if np.isfinite(v50):
                z_ref = (1 + z_ref) * (1 + v50 / C_KMS) - 1
                sig_ref = float(np.clip(w80 / FWHM_W80 if ft.ncomp > 1 else ft.comps[0][2], 20.0, cfg.sig_int_max))
                ref_snr, ref_src = ft.snr(main), 'total'
                S, N, region, reg_dist, found = regions(z_ref, sig_ref, noise_a.sd1)
    # anchor line detected: S/N >= force_below (calibrated errors) in the aperture or the region AND
    # above the same fit in every line-free window of that spectrum
    detect = ref_src != 'default'
    if not (found and detect):        # no detection: one central aperture, forced fits, flagged
        region, reg_dist = ap.copy(), (reg_dist if found else np.nan)
    allow_free = bool(detect)

    # 3. noise of the anchor band with the galaxy excluded; S/N map; bins
    noise_a = band_noise(A, fp_a, centre, region, z_ref, cfg, rels[aband])
    S2, N2 = line_sn_map(A.cube, A.wave, noise_a.sd1, z_ref, rest_a, sig_ref, aband, fp_a, noise_a.rel)
    if S2 is not None:
        S, N = S2, N2
        fin = region & np.isfinite(S) & np.isfinite(N)
        if fin.any():
            region = fin
    if detect:
        binmap, target, bin_mode = make_bins(S, N, region, noise_a.ncorr, cfg)
    else:                             # undetected: the central aperture is one bin
        binmap = np.where(region, 0, -1)
        target, bin_mode = np.nan, 'single'
    nbins = int(binmap.max() + 1) if (binmap >= 0).any() else 0
    if nbins == 0:
        raise ValueError('no spaxels to bin')
    if progress:
        progress(0, nbins)

    # 4. every band: bins, noise without the galaxy; null tests -> error scale (before any bin fit)
    bands = {}
    for b, st in stacks.items():
        if b == aband:
            bands[b] = dict(binmap=binmap, noise=noise_a)
            continue
        bm = transfer_binmap(binmap, A.wcs, st.cube.shape[1:], st.wcs, fps[b])
        if (bm >= 0).any():
            cb = np.argwhere(bm >= 0).mean(0)
            cb = (int(round(cb[0])), int(round(cb[1])))
        else:
            cb = (st.cube.shape[1] // 2, st.cube.shape[2] // 2)
        bands[b] = dict(binmap=bm, noise=band_noise(st, fps[b], cb, bm >= 0, z_ref, cfg, rels[b]))
    group_band = {g: pick_band(stacks, g, z_ref, cfg) for g in groups}
    group_band[anchor] = aband
    if cfg.null_test:
        offsets = cfg.null_offsets if nbins >= 4 else _dense_offsets()
        for b, info in bands.items():
            rest_b = next((REST[GROUPS[g]['main']] for g in [anchor] + groups if group_band.get(g) == b), None)
            if rest_b is None:
                continue
            pairs = []                                 # (region size, null S/N)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for b_id in range(nbins):
                    sel = info['binmap'] == b_id
                    if sel.any():
                        s, e = region_spectrum(stacks[b].cube, sel, info['noise'], cfg.edge_margin)
                        pairs += [(int(sel.sum()), v) for v in
                                  _null_snrs(stacks[b].wave, s, e, z_ref, rest_b, sig_ref, b, cfg, offsets)]
                if len(pairs) < cfg.null_min:          # few bins: add sky boxes of this band
                    pairs += [(9, v) for v in _box_nulls(stacks[b], info['noise'], z_ref, rest_b, sig_ref, b, cfg)]
                sel = info['binmap'] >= 0
                if nbins > 1 and sel.any():            # the TOTAL spectrum calibrates itself
                    s, e = region_spectrum(stacks[b].cube, sel, info['noise'], cfg.edge_margin)
                    pairs += [(int(sel.sum()), v) for v in
                              _null_snrs(stacks[b].wave, s, e, z_ref, rest_b, sig_ref, b, cfg, _dense_offsets())]
            nz = info['noise']
            if not _apply_nulls(nz, [v for _, v in pairs], cfg) and b == aband:
                nz.err_scale = err_scale_det
            nz.scale_n = _scale_table(pairs, cfg) if len(pairs) >= cfg.null_min else []

    def spectra(sel_of):
        out = {}
        for b, info in bands.items():
            sel = sel_of(info['binmap'])
            if sel.any():
                s, e = region_spectrum(stacks[b].cube, sel, info['noise'], cfg.edge_margin)
                out[b] = (stacks[b].wave, s, e)
        return out

    ref_tpl = ([(0.0, sig_ref)], [1.0])

    def template_of(fa):
        if fa is None or not fa.ok:
            return ref_tpl
        if fa.forced:
            return [(fa.template[0], sig_ref)], [1.0]
        return [(c[0], c[2]) for c in fa.comps], fa.fracs

    def fit_others(sp, fa, fits):
        detected = fa is not None and fa.ok and not fa.forced
        for g in groups:
            if g == anchor or group_band.get(g) is None or group_band[g] not in sp:
                continue
            gd = GroupData(g, *sp[group_band[g]], z_ref, group_band[g], cfg)
            if gd.ok:
                fits[g] = fit_other(gd, cfg, template_of(fa), detected)

    # 5. anchor of every bin: free where detected; the others forced at the interpolated velocity
    spec_of = {}
    for b_id in range(nbins):
        spec_of[b_id] = spectra(lambda bm, i=b_id: bm == i)

    def anchor_pass():
        out = []
        for b_id in range(nbins):
            sel = binmap == b_id
            ys, xs = np.nonzero(sel)
            row = dict(bin=b_id, npix=int(sel.sum()), x=float(xs.mean()), y=float(ys.mean()), fits={})
            sp = spec_of[b_id].get(aband)
            if sp is not None:
                gd = GroupData(anchor, *sp, z_ref, aband, cfg)
                if gd.ok:
                    row['fits'][anchor] = fit_anchor(gd, cfg, ref_tpl, allow_free=allow_free)
            out.append(row)
        d = [(r['x'], r['y'], profile_stats(r['fits'][anchor].comps, r['fits'][anchor].fracs)[0],
              r['fits'][anchor].snr(main), r['fits'][anchor].flux.get(main, np.nan)) for r in out
             if anchor in r['fits'] and r['fits'][anchor].ok and not r['fits'][anchor].forced]
        return out, [x for x in d if np.isfinite(x[2])]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rows, det = anchor_pass()
        # the reference must describe the galaxy: if the flux-weighted median velocity of the detected
        # bins is far from it (the summed spectrum was pulled by a coherent sky residual), recentre once
        if len(det) >= 3:
            v, wt = np.array([x[2] for x in det]), np.clip([x[4] for x in det], 0, None)
            if wt.sum() > 0:
                o = np.argsort(v)
                vmed = float(v[o][np.searchsorted(np.cumsum(wt[o]), 0.5 * wt.sum())])
                if abs(vmed) > cfg.ref_recentre:
                    z_ref = (1 + z_ref) * (1 + vmed / C_KMS) - 1
                    ref_src += '+bins'
                    rows, det = anchor_pass()
        det = [x[:4] for x in det]
        if det:
            for r in rows:
                fa = r['fits'].get(anchor)
                if fa is None or not fa.forced:
                    continue
                dist = np.array([np.hypot(r['x'] - x, r['y'] - y) for x, y, _, _ in det])
                nn = np.argsort(dist)[:4]
                w = np.array([max(det[i][3], 1.0) / max(dist[i], 0.5) ** 2 for i in nn])
                v_i = float(np.sum(w * np.array([det[i][2] for i in nn])) / w.sum())
                gd = GroupData(anchor, *spec_of[r['bin']][aband], z_ref, aband, cfg)
                r['fits'][anchor] = fit_anchor(gd, cfg, ([(v_i, sig_ref)], [1.0]), allow_free=False)
        # 6. the other groups with the anchor template of their bin
        for i, r in enumerate(rows):
            fit_others(spec_of[r['bin']], r['fits'].get(anchor), r['fits'])
            if progress:
                progress(i + 1, nbins)
        # 7. the sum of all bins (closure, integrated values)
        total = None
        if nbins > 1:
            sp = spectra(lambda bm: bm >= 0)
            total = {}
            if aband in sp:
                gd = GroupData(anchor, *sp[aband], z_ref, aband, cfg)
                vt = float(np.clip(max([abs(x[2]) for x in det], default=0.0) + 100.0, 150.0, cfg.dv_max))
                if gd.ok:                  # components only within the velocities the bins show (+100 km/s)
                    total[anchor] = fit_anchor(gd, cfg, ref_tpl, allow_free=allow_free, vlim=(-vt, vt))
            fit_others(sp, total.get(anchor), total)

    prod = dict(name=name, z=float(z), z_ref=float(z_ref), sig_ref=float(sig_ref), ref_snr=float(ref_snr),
                ref_src=ref_src, anchor=anchor, anchor_band=aband, centre=centre, centred=bool(centred),
                reg_dist=float(reg_dist), detect=bool(detect), found=bool(found), bin_mode=bin_mode,
                err_scale_det=float(err_scale_det), null_max=float(null_max),
                target_sn=float(target) if np.isfinite(target) else np.nan, nbins=nbins, binmap=binmap,
                band_binmaps={b: v['binmap'] for b, v in bands.items()},
                noise={b: v['noise'] for b, v in bands.items()}, group_band=group_band, S=S, N=N, rows=rows,
                cfg=cfg, stacks=stacks, groups=groups, seconds=time.time() - t0,
                wcs=A.wcs, shape=(ny, nx), bunit=getattr(A, 'bunit', '') or 'ADU/sec',
                fluxk={b: _fluxk(st) for b, st in stacks.items()})
    prod['table'] = _bins_table(prod)
    prod['total_row'] = _row_of(prod, -1, total, int((binmap >= 0).sum())) if total else None
    prod['total_fits'] = total or {}
    col = f'{line_tag(main)}_FLUX'
    tot = prod['total_row']
    if tot is not None and np.isfinite(tot.get(col, np.nan)) and tot[col] != 0 and col in prod['table']:
        prod['closure'] = float(np.nansum(prod['table'][col]) / tot[col])
    else:
        prod['closure'] = np.nan
    prod['maps'] = _maps(prod)
    prod['status'] = 'no_detection' if not detect else ('ok' if nbins > 1 else 'single')
    return prod


def _fluxk(st):
    h = getattr(st, 'header0', None) or {}
    try:
        fk = float(h.get('FLUXK'))
    except (TypeError, ValueError):
        return np.nan, np.nan
    try:
        fke = float(h.get('FLUXKERR'))
    except (TypeError, ValueError):
        fke = np.nan
    return fk, fke


# ======================================================================== table and maps
def profile_stats(comps, fracs, ngrid=4001):
    """v50, W80 (= v90 - v10), v10, v90 [km/s] of the intrinsic (LSF-free) line profile.
    comps: [(v, ..., sigma, ...)] (GroupFit.comps) or [(v, 0, sigma, 0)]."""
    if not comps or not fracs:
        return np.nan, np.nan, np.nan, np.nan
    v = np.array([c[0] for c in comps], float)
    s = np.array([max(c[2], 1.0) for c in comps], float)
    w = np.clip(np.asarray(fracs, float)[:len(comps)], 0, None)
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(s)) and np.isfinite(w).all()) or w.sum() <= 0:
        return np.nan, np.nan, np.nan, np.nan
    w = w / w.sum()
    vgrid = np.linspace(v.min() - 6 * s.max(), v.max() + 6 * s.max(), ngrid)
    prof = sum(wk * _unit_gauss(vgrid, vk, sk) for wk, vk, sk in zip(w, v, s))
    cdf = np.cumsum(prof)
    cdf /= cdf[-1]
    v10, v50, v90 = np.interp([0.1, 0.5, 0.9], cdf, vgrid)
    return float(v50), float(v90 - v10), float(v10), float(v90)


def ne_sii(R):
    """[SII]6716/6731 -> n_e [cm^-3] (Sanders+16, T_e = 1e4 K); NaN outside the valid range."""
    if not np.isfinite(R) or not (SII_RANGE[0] < R < SII_RANGE[1]):
        return np.nan
    a, b, c = 0.4315, 2107.0, 627.1
    return float(max((c * R - a * b) / (a - R), 1.0))


def oh_m13(n2):
    """12 + log(O/H) from N2 (Marino+13), valid for -1.6 < N2 < -0.2."""
    return float(8.743 + 0.462 * n2) if (np.isfinite(n2) and -1.6 < n2 < -0.2) else np.nan


def _bg_cont_err(prod, gk, band, b_id):
    """Continuum uncertainty at the main line of group gk from the subtracted background:
    (spaxels of the bin in that band) x bg_err."""
    bm, nz = prod['band_binmaps'].get(band), prod['noise'].get(band)
    if bm is None or nz is None or nz.bg_err is None:
        return 0.0
    npix = int((bm >= 0).sum()) if b_id < 0 else int((bm == b_id).sum())
    w = prod['stacks'][band].wave
    i = int(np.clip(np.searchsorted(w, REST[GROUPS[gk]['main']] * (1 + prod['z_ref'])), 0, len(w) - 1))
    return float(npix * nz.bg_err[i])


def _row_of(prod, b_id, fits, npix=0, x=np.nan, y=np.nan):
    cfg = prod['cfg']
    fits = fits or {}
    r = dict(BIN=int(b_id), NPIX=int(npix), XBAR=float(x), YBAR=float(y))
    band_of = {}
    for gk in prod['groups']:
        f = fits.get(gk)
        G = group_tag(gk)
        main_snr = f.snr(GROUPS[gk]['main']) if f else np.nan
        kin_ok = bool(f and f.ok and not f.forced and main_snr >= cfg.kin_snr_min)
        r[f'{G}_BAND'] = f.band if f else ''
        r[f'{G}_NCOMP'] = f.ncomp if f else 0
        r[f'{G}_FORCED'] = int(f.forced) if f else -1
        r[f'{G}_CHI2NU'] = f.chi2nu if f else np.nan
        r[f'{G}_DBIC_NCOMP'] = f.dbic_ncomp if f else np.nan
        r[f'{G}_DBIC_FREE'] = f.dbic_free if f else np.nan
        r[f'{G}_NCLIP'] = f.nclip if f else 0
        r[f'{G}_QFLAG'] = f.qflag if f else 0
        r[f'{G}_X0'] = f.x0 if f else np.nan
        r[f'{G}_CONT_A'], r[f'{G}_CONT_B'] = f.cont if f else (np.nan, np.nan)
        r[f'{G}_CONT_MAIN'], r[f'{G}_CONT_MAIN_ERR'] = f.cont_main if f else (np.nan, np.nan)
        if f:                          # + uncertainty of the subtracted background (n spaxels x bg_err)
            r[f'{G}_CONT_MAIN_ERR'] = float(np.hypot(r[f'{G}_CONT_MAIN_ERR'], _bg_cont_err(prod, gk, f.band, b_id)))
        r[f'{G}_TPL_V'], r[f'{G}_TPL_SIG'] = f.template if (f and f.forced) else (np.nan, np.nan)
        for k in range(cfg.max_components):
            has = bool(f and k < len(f.comps))
            c = f.comps[k] if has else (np.nan, np.nan, np.nan, np.nan, False)
            r[f'{G}_C{k + 1}_MV'], r[f'{G}_C{k + 1}_MSIG'] = c[0], c[2]          # model parameters
            r[f'{G}_C{k + 1}_V'] = c[0] if kin_ok else np.nan                    # published kinematics
            r[f'{G}_C{k + 1}_VERR'] = c[1] if kin_ok else np.nan
            r[f'{G}_C{k + 1}_SIG'] = c[2] if kin_ok else np.nan
            r[f'{G}_C{k + 1}_SIGERR'] = c[3] if kin_ok else np.nan
            r[f'{G}_C{k + 1}_UNRES'] = int(bool(c[4])) if (has and kin_ok) else 0
            frac = f.fracs[k] if (f and k < len(f.fracs)) else np.nan
            r[f'{G}_C{k + 1}_MFRAC'] = frac                                       # model (template) weight
            r[f'{G}_C{k + 1}_FRAC'] = frac if kin_ok else np.nan                 # measured flux share
        v50, w80, v10, v90 = profile_stats(f.comps, f.fracs) if kin_ok else (np.nan,) * 4
        r[f'{G}_V50'], r[f'{G}_W80'], r[f'{G}_V10'], r[f'{G}_V90'] = v50, w80, v10, v90
        fk = prod['fluxk'].get(f.band, (np.nan, np.nan))[0] if f else np.nan
        for line in GROUPS[gk]['lines']:
            L = line_tag(line)
            fitted = bool(f and line in f.flux)
            fl = f.flux[line] if fitted else np.nan
            fe = f.flux_err.get(line, np.nan) if fitted else np.nan
            r[f'{L}_FLUX'], r[f'{L}_FLUX_ERR'] = fl, fe
            r[f'{L}_SNR'] = fl / fe if (np.isfinite(fe) and fe > 0) else np.nan
            r[f'{L}_FORCED'] = int(f.forced) if fitted else -1
            # 1e-17 erg/s/cm2 = flux [BUNIT um] x 1e4 A/um x FLUXK [1e-17 erg/s/cm2/A per BUNIT]
            r[f'{L}_FLUX_CGS'] = fl * 1e4 * fk if np.isfinite(fk) else np.nan
            r[f'{L}_FLUX_CGS_ERR'] = fe * 1e4 * fk if np.isfinite(fk) else np.nan
            if fitted:
                band_of[line] = f.band
            cf = f.comp_flux.get(line, []) if f else []
            ce = f.comp_flux_err.get(line, []) if f else []
            for k in range(cfg.max_components):
                r[f'{L}_C{k + 1}_FLUX'] = cf[k] if k < len(cf) else np.nan
                r[f'{L}_C{k + 1}_FLUX_ERR'] = ce[k] if k < len(ce) else np.nan

    def ok(line):
        return r.get(f'{line_tag(line)}_SNR', np.nan) >= cfg.ratio_snr_min

    def cov(a, b):
        """Covariance of two total line fluxes [native units]: from the fit if both lines belong to
        the same group (doublet ties, shared kinematics, pinned [SII] ratio), else 0."""
        ga, gb = LINE_GROUP.get(a), LINE_GROUP.get(b)
        if ga != gb or fits.get(ga) is None:
            return 0.0
        c = fits[ga].flux_cov.get((a, b), 0.0)
        return c if np.isfinite(c) else 0.0

    def xband(*lines):
        """1 if the lines were fitted in different bands, 0 if in one band, -1 if one is missing."""
        bs = [band_of.get(k) for k in lines]
        return -1 if not all(bs) else int(len(set(bs)) > 1)

    def pair(a, b):
        """(fa, ea, fb, eb, across bands) in common units: native in one band, else CGS."""
        La, Lb = line_tag(a), line_tag(b)
        if band_of.get(a) and band_of.get(a) == band_of.get(b):
            return r[f'{La}_FLUX'], r[f'{La}_FLUX_ERR'], r[f'{Lb}_FLUX'], r[f'{Lb}_FLUX_ERR'], False
        return r[f'{La}_FLUX_CGS'], r[f'{La}_FLUX_CGS_ERR'], r[f'{Lb}_FLUX_CGS'], r[f'{Lb}_FLUX_CGS_ERR'], True

    xcal = 0.4343 * np.sqrt(2) * 0.1     # two independent absolute calibrations (FLUXKERR ~ 10 % each)

    def lg(a, b):
        if not (ok(a) and ok(b)):
            return np.nan, np.nan
        fa, ea, fb, eb, xb = pair(a, b)
        if not (np.isfinite(fa) and np.isfinite(fb) and fa > 0 and fb > 0):
            return np.nan, np.nan
        var = (ea / fa) ** 2 + (eb / fb) ** 2 - (0.0 if xb else 2 * cov(a, b) / (fa * fb))
        e = 0.4343 * np.sqrt(max(var, 0.0))
        return float(np.log10(fa / fb)), float(np.hypot(e, xcal) if xb else e)
    r['LOG_N2'], r['LOG_N2_ERR'] = lg('NII6583', 'Ha')
    r['LOG_N2_XBAND'] = xband('NII6583', 'Ha')
    r['LOG_O3'], r['LOG_O3_ERR'] = lg('OIII5007', 'Hb')
    r['LOG_O3_XBAND'] = xband('OIII5007', 'Hb')
    s2 = s2e = np.nan
    if ok('SII6716') and ok('SII6731') and ok('Ha'):
        fa, ea, fh, eh, xb = pair('SII6716', 'Ha')
        fb, eb, _, _, _ = pair('SII6731', 'Ha')
        if np.all(np.isfinite([fa, fb, fh])) and fa + fb > 0 and fh > 0:
            k = 1e4 * prod['fluxk'].get(band_of.get('SII6716'), (np.nan,))[0] if xb else 1.0
            var_s = ea ** 2 + eb ** 2 + 2 * cov('SII6716', 'SII6731') * k * k
            s2 = float(np.log10((fa + fb) / fh))
            s2e = float(0.4343 * np.sqrt(max(var_s, 0.0) / (fa + fb) ** 2 + (eh / fh) ** 2))
            if xb:
                s2e = float(np.hypot(s2e, xcal))
    r['LOG_S2'], r['LOG_S2_ERR'] = s2, s2e
    r['LOG_S2_XBAND'] = xband('SII6716', 'Ha')
    if np.isfinite(r['LOG_O3']) and np.isfinite(r['LOG_N2']):
        r['LOG_O3N2'] = r['LOG_O3'] - r['LOG_N2']
        r['LOG_O3N2_ERR'] = float(np.hypot(r['LOG_O3_ERR'], r['LOG_N2_ERR']))
    else:
        r['LOG_O3N2'] = r['LOG_O3N2_ERR'] = np.nan
    r['LOG_O3N2_XBAND'] = max(r['LOG_O3_XBAND'], r['LOG_N2_XBAND']) if min(r['LOG_O3_XBAND'], r['LOG_N2_XBAND']) >= 0 else -1
    R = Re = np.nan
    if ok('SII6716') and ok('SII6731'):
        a, ea, b, eb = r['SII6716_FLUX'], r['SII6716_FLUX_ERR'], r['SII6731_FLUX'], r['SII6731_FLUX_ERR']
        if a > 0 and b > 0:
            var = (ea / a) ** 2 + (eb / b) ** 2 - 2 * cov('SII6716', 'SII6731') / (a * b)
            R, Re = a / b, a / b * np.sqrt(max(var, 0.0))
            if fits.get('sii') is not None and fits['sii'].qflag & QF_SII_BOUND:
                Re = np.nan                 # pinned to a physical limit: a limit, no error, no density
    r['SII_RATIO'], r['SII_RATIO_ERR'] = R, Re
    r['NE_SII'] = ne_sii(R) if (np.isfinite(Re) and Re < 0.1) else np.nan
    r['OH_N2_M13'] = oh_m13(r['LOG_N2'])
    G = group_tag('ha')
    ew = ew_err = np.nan
    c, ce = r.get(f'{G}_CONT_MAIN', np.nan), r.get(f'{G}_CONT_MAIN_ERR', np.nan)
    if (ok('Ha') and 'ha' in prod['groups'] and np.isfinite(c) and np.isfinite(ce) and ce > 0
            and c / ce >= cfg.ratio_snr_min):
        ew = r['HA_FLUX'] / c / (1 + prod['z_ref']) * 1e4          # rest-frame Angstrom
        ew_err = abs(ew) * np.hypot(r['HA_FLUX_ERR'] / r['HA_FLUX'], ce / c)
    r['EW_HA'], r['EW_HA_ERR'] = ew, ew_err
    return r


def _bins_table(prod):
    import pandas as pd
    return pd.DataFrame([_row_of(prod, r['bin'], r['fits'], r['npix'], r['x'], r['y']) for r in prod['rows']])


CGS_UNIT = '10**(-17) erg s-1 cm-2'


def fits_unit(bunit):
    """FITS-standard form of the cube unit ('ADU/sec' -> 'adu s-1')."""
    u = (bunit or 'ADU/sec').strip()
    return u.replace('ADU', 'adu').replace('/sec', ' s-1').replace('/s', ' s-1')


def column_unit(col, native):
    """FITS unit string of a BINS / TOTAL column ('' = dimensionless); native = fits_unit(cube BUNIT)."""
    import re
    if re.search(r'_FLUX_CGS(_ERR)?$', col):
        return CGS_UNIT
    if re.search(r'_FLUX(_ERR)?$', col):
        return f'{native} um'
    if re.search(r'_(V50|W80|V10|V90|TPL_V|TPL_SIG|C\d_(V|VERR|SIG|SIGERR|MV|MSIG))$', col):
        return 'km s-1'
    if col.endswith('_X0'):
        return 'um'
    if col.endswith('_CONT_B'):
        return f'{native} um-1'
    if re.search(r'_CONT_(A|MAIN|MAIN_ERR)$', col):
        return native
    if col in ('XBAR', 'YBAR'):
        return 'pix'
    if col.startswith('EW_HA'):
        return 'Angstrom'
    if col == 'NE_SII':
        return 'cm-3'
    return ''


def _map_columns(prod):
    """BINS columns painted as maps -> unit."""
    cfg = prod['cfg']
    names = []
    for gk in prod['groups']:
        G = group_tag(gk)
        names += [f'{G}_NCOMP', f'{G}_FORCED', f'{G}_QFLAG', f'{G}_CHI2NU', f'{G}_V50', f'{G}_W80', f'{G}_V10',
                  f'{G}_V90']
        for k in range(cfg.max_components):
            names += [f'{G}_C{k + 1}_V', f'{G}_C{k + 1}_SIG', f'{G}_C{k + 1}_FRAC']
        for line in GROUPS[gk]['lines']:
            L = line_tag(line)
            names += [f'{L}_FLUX', f'{L}_FLUX_ERR', f'{L}_SNR', f'{L}_FLUX_CGS']
            names += [f'{L}_C{k + 1}_FLUX' for k in range(cfg.max_components)]
    names += ['LOG_N2', 'LOG_O3', 'LOG_S2', 'LOG_O3N2', 'SII_RATIO', 'NE_SII', 'OH_N2_M13', 'EW_HA']
    native = fits_unit(prod['bunit'])
    return {n: column_unit(n, native) for n in names}


def _maps(prod):
    """{EXTNAME: (2-D image on the anchor grid, unit)}; maps without information are left out.
    Number of components and component fluxes are not painted in forced bins (template, not a
    measurement of that group)."""
    import re
    tab, bm = prod['table'], prod['binmap']
    with np.errstate(all='ignore'):
        sn = np.where(np.isfinite(prod['S']) & (prod['N'] > 0), prod['S'] / prod['N'], np.nan).astype(np.float32)
    out = {'BINMAP': (bm.astype(np.int16), ''), 'SN_LINE': (sn, '')}
    if tab.empty:
        return out
    ok = bm >= 0
    for col, unit in _map_columns(prod).items():
        if col not in tab.columns:
            continue
        vals = tab[col].to_numpy(dtype=float)
        m = re.match(r'^(\w+?)_(NCOMP|C\d_FLUX)$', col)
        if m:
            G = m.group(1) if m.group(2) == 'NCOMP' else group_tag(LINE_GROUP.get(
                next((k for k in REST if line_tag(k) == m.group(1)), ''), 'ha'))
            if f'{G}_FORCED' in tab.columns:
                vals = np.where(tab[f'{G}_FORCED'].to_numpy() == 1, np.nan, vals)
        if not np.isfinite(vals).any():
            continue
        if (col.endswith('_NCOMP') or col.endswith('_QFLAG')) and not np.any(vals > 0):
            continue
        if col.endswith('_FORCED') and not np.any(vals >= 0):
            continue
        img = np.full(bm.shape, np.nan, np.float32)
        img[ok] = vals[bm[ok]]
        out[col] = (img, unit)
    return out


# ======================================================================== product I/O
_VERSIONS = {}


def _versions():
    """Package versions and the git hash of this code (cached; no console window on Windows)."""
    if _VERSIONS:
        return _VERSIONS
    import hashlib
    import importlib
    import importlib.metadata as md
    for mod in ('numpy', 'scipy', 'astropy', 'vorbin'):
        try:
            _VERSIONS[mod] = md.version(mod)
        except Exception:
            try:                       # frozen app: no dist-info, but the modules know their version
                _VERSIONS[mod] = str(getattr(importlib.import_module(mod), '__version__', '?'))
            except Exception:
                _VERSIONS[mod] = '?'
    try:                               # content hash of this file: identifies the exact pipeline code
        with open(os.path.abspath(__file__), 'rb') as f:
            _VERSIONS['code'] = hashlib.sha256(f.read()).hexdigest()[:12]
    except OSError:
        _VERSIONS['code'] = '?'
    here = os.path.dirname(os.path.abspath(__file__))
    kw = dict(capture_output=True, text=True, timeout=5)
    if os.name == 'nt':
        kw['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    try:
        h = subprocess.run(['git', '-C', here, 'rev-parse', '--short', 'HEAD'], **kw).stdout.strip()
        dirty = subprocess.run(['git', '-C', here, 'status', '--porcelain', '--', here], **kw).stdout.strip()
        _VERSIONS['git'] = (h + ('-dirty' if dirty else '')) if h else '?'
    except Exception:
        _VERSIONS['git'] = '?'
    try:
        from empower_explorer import __version__ as appver
        _VERSIONS['app'] = appver
    except Exception:
        _VERSIONS['app'] = '?'
    return _VERSIONS


def write_product(prod, path, meta=None):
    """One FITS file per galaxy (schema EMPOWER-VORONOI-2), written atomically:
      PRIMARY        OBJECT, STATUS, DETECT, Z, Z_REF, SIG_REF, REFSRC, REF_SNR, ANCHOR, ANCHBAND, BANDS,
                     BINMODE, NBINS, TARGETSN, CENTX/Y, CENTRED, REGDIST, BGSUB, ERRSC_/NNULL_/FLUXK_<band>,
                     ERRSCALE, CLOSURE, versions (PIPEVER, APPVER, GITHASH, NUMPYVER, ...), inputs (HISTORY)
      image HDUs     maps on the anchor-band grid (celestial WCS); EXTNAME = BINS column name
      BINMAP_<band>  the bins on the grid of every other band
      RELNOISE_<band> relative noise r of every spaxel (sigma_1 of NOISE_<band> is for r = 1)
      BINS / TOTAL   one row per bin / the sum of all bins (every parameter; models via model_curve)
      LINES          rest wavelengths, groups, ties, band and LSF used
      CONFIG         the configuration (KEY, json VALUE)
      NOISE_<band>   WAVE, SD1, BG of the noise model (ERRSCALE, NCORR, RESOLV in its header)"""
    from astropy.io import fits
    from astropy.table import Table
    import pandas as pd
    meta = dict(meta or {})
    vers = _versions()
    cfg = prod['cfg']
    h0 = fits.Header()
    h0['OBJECT'] = prod['name']
    h0['SCHEMA'] = (SCHEMA, 'product format')
    h0['PIPEVER'] = (PIPELINE_VERSION, 'empower_explorer.pipeline.voronoi_binning')
    h0['APPVER'] = (vers.get('app', '?'), 'EMPOWER Explorer version')
    h0['GITHASH'] = (vers.get('git', '?'), 'git commit of the repository')
    h0['CODEHASH'] = (vers.get('code', '?'), 'sha256[:12] of voronoi_binning.py')
    for k, key in (('numpy', 'NUMPYVER'), ('scipy', 'SCIPYVER'), ('astropy', 'ASTROVER'), ('vorbin', 'VORBINV')):
        h0[key] = vers.get(k, '?')
    h0['DATE'] = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S')
    h0['STATUS'] = (prod['status'], 'ok / single / no_detection')
    h0['DETECT'] = (prod['detect'], 'anchor line detected (calibrated S/N, null test)')
    h0['Z'] = (prod['z'], 'catalogue redshift')
    h0['Z_REF'] = (prod['z_ref'], 'velocities refer to this z')
    h0['SIG_REF'] = (prod['sig_ref'], 'km/s intrinsic sigma of the reference')
    h0['REFSRC'] = (prod['ref_src'], 'total/aperture/default; +bins: recentred on bins')
    if np.isfinite(prod['ref_snr']):
        h0['REF_SNR'] = (prod['ref_snr'], 'anchor S/N of the reference fit')
    h0['ANCHOR'] = (prod['anchor'], 'line group defining bins and kinematics')
    h0['ANCHBAND'] = (prod['anchor_band'], 'band of the anchor = grid of the maps')
    h0['BANDS'] = '+'.join(prod['band_binmaps'])
    h0['REGFOUND'] = (prod['found'], 'S/N >= MIN_SN region connected to the centre')
    h0['BINMODE'] = prod['bin_mode']
    h0['NBINS'] = prod['nbins']
    h0['TARGETSN'] = (prod['target_sn'] if np.isfinite(prod['target_sn']) else -1, 'S/N per bin, -1 n/a')
    h0['CENTX'], h0['CENTY'] = int(prod['centre'][1]), int(prod['centre'][0])
    h0['CENTRED'] = (prod['centred'], 'centre on the anchor-line peak')
    h0['REGDIST'] = (prod['reg_dist'] if np.isfinite(prod['reg_dist']) else -1, 'px region to centre, -1 none')
    h0['BGSUB'] = (bool(cfg.bg_subtract), 'residual background subtracted')
    for b, nz in prod['noise'].items():
        h0[f'ERRSC_{b}'] = (nz.err_scale, f'error scale {b} (null tests; by size: NOISE_{b})')
        h0[f'NNULL_{b}'] = (nz.n_null, f'null tests {b}')
        h0[f'NSKY_{b}'] = (nz.n_sky, f'sky spaxels of the noise model {b} ({nz.sky_src})')
        if np.isfinite(nz.null_nmad):
            h0[f'NMAD_{b}'] = (nz.null_nmad, f'NMAD of the null-test S/N {b}')
        fk = prod['fluxk'].get(b, (np.nan, np.nan))[0]
        if np.isfinite(fk):
            h0[f'FLUXK_{b}'] = (fk, '1e-17 erg/s/cm2/A per BUNIT')
    h0['ERRSCALE'] = (prod['noise'][prod['anchor_band']].err_scale, 'error scale of the anchor band')
    h0['ERRSCDET'] = (prod.get('err_scale_det', 1.0), 'anchor error scale of the detection decision')
    if np.isfinite(prod.get('null_max', np.nan)):
        h0['NULLDET'] = (prod['null_max'], 'max S/N of the detection fit in line-free windows')
    if np.isfinite(prod.get('closure', np.nan)):
        h0['CLOSURE'] = (prod['closure'], 'sum of the bin fluxes / fit of their sum')
    h0['MAXCOMP'] = cfg.max_components
    native = fits_unit(prod['bunit'])
    h0['BUNIT'] = (f'{native} um', 'unit of the line fluxes (cube unit x micron)')
    h0['FITSEC'] = round(prod['seconds'], 1)
    for line in (f'Pixel coordinates (CENTX/CENTY, XBAR/YBAR, BINMAP) are 0-based (numpy order y, x).',
                 f'<G>_QFLAG bits: {QF_CHI2} reduced chi2 > {cfg.chi2nu_max:g}, {QF_CLIPPED} pixels clipped '
                 f'(table CLIPPED), {QF_UNRESOLVED} a free component unresolved, {QF_SII_BOUND} [SII] ratio at a limit, '
                 f'{QF_BROAD} line broader than the model (forced flux = lower limit), {QF_PARTIAL} a line of '
                 'the group not covered.',
                 'Component errors (_C<k>_VERR/_SIGERR/_FLUX_ERR) are Fisher errors: optimistic for components '
                 'with S/N below ~10.',
                 'Kinematics (_V, _SIG, _V50, _W80, _FRAC) are NaN where the group was forced to the anchor '
                 f'template or its S/N < {cfg.kin_snr_min:g}; _MV/_MSIG/_MFRAC are the model parameters.',
                 '*_FLUX_CGS = flux x 1e4 x FLUXK_<band> (absolute calibration ~10 %); ratios across bands use '
                 'them (<RATIO>_XBAND = 1) and include the calibration error.',
                 'Velocities relative to Z_REF; sigma intrinsic (LSF removed). Format: '
                 'empower_explorer/pipeline/voronoi_binning.py docstring.'):
        h0.add_comment(line)
    for k, v in meta.items():
        try:
            h0[k[:8].upper()] = v
        except (ValueError, TypeError):
            pass
    for b, st in prod['stacks'].items():
        for p in list(getattr(st, 'paths', []))[:20]:
            h0.add_history(f'input {b}: {os.path.basename(p)}')
    hdus = [fits.PrimaryHDU(header=h0)]
    wcs_hdr = prod['wcs'].to_header() if prod['wcs'] is not None else fits.Header()
    for ext, (img, unit) in prod['maps'].items():
        h = wcs_hdr.copy()
        if unit:
            h['BUNIT'] = unit
        hdus.append(fits.ImageHDU(img, header=h, name=ext))
    for b, bm in prod['band_binmaps'].items():
        if b == prod['anchor_band']:
            continue
        st = prod['stacks'][b]
        hdus.append(fits.ImageHDU(bm.astype(np.int16), header=st.wcs.to_header() if st.wcs is not None else None,
                                  name=f'BINMAP_{b}'))
    for b, nz in prod['noise'].items():           # relative noise of the spaxels (grid of that band)
        if nz.rel is not None:
            st = prod['stacks'][b]
            hdus.append(fits.ImageHDU(np.asarray(nz.rel, np.float32),
                                      header=st.wcs.to_header() if st.wcs is not None else None, name=f'RELNOISE_{b}'))
    def table_hdu(df, name, units):
        t = fits.BinTableHDU(Table.from_pandas(df), name=name)
        for i, c in enumerate(df.columns):
            u = units(c) if callable(units) else units.get(c, '')
            if u:
                t.header[f'TUNIT{i + 1}'] = u
        return t

    def col_unit(c):
        return column_unit(c, native)
    tab = prod['table']
    names = list(tab.columns)
    up = [n.upper() for n in names]
    if len(set(up)) != len(up):
        raise RuntimeError('BINS column names collide ignoring case')
    if not tab.empty:
        hdus.append(table_hdu(tab, 'BINS', col_unit))
    if prod.get('total_row'):
        hdus.append(table_hdu(pd.DataFrame([prod['total_row']]), 'TOTAL', col_unit))
    clipped = []                       # pixels removed before the final fits (bin -1 = TOTAL)
    for b_id, fits_ in [(r['bin'], r['fits']) for r in prod['rows']] + [(-1, prod.get('total_fits') or {})]:
        for g, f in fits_.items():
            clipped += [dict(BIN=int(b_id), GROUP=group_tag(g), BAND=f.band, WAVE=float(w)) for w in f.clipped]
    if clipped:
        hdus.append(table_hdu(pd.DataFrame(clipped), 'CLIPPED', {'WAVE': 'um'}))
    lines = []
    for g in prod['groups']:
        b = prod['group_band'].get(g) or ''
        for k in GROUPS[g]['lines']:
            tie = GROUPS[g]['ties'].get(k)
            lines.append(dict(LINE=line_tag(k), REST_UM=REST[k], GROUP=group_tag(g), BAND=b,
                              TIED_TO=line_tag(tie[0]) if tie else '', RATIO=tie[1] if tie else np.nan,
                              LSF_KMS=sigma_lsf_kms(b) if b else np.nan))
    hdus.append(table_hdu(pd.DataFrame(lines), 'LINES', {'REST_UM': 'um', 'LSF_KMS': 'km s-1'}))
    cfgd = json.loads(cfg.to_json())
    hdus.append(fits.BinTableHDU(Table.from_pandas(pd.DataFrame(
        dict(KEY=list(cfgd), VALUE=[json.dumps(v) for v in cfgd.values()]))), name='CONFIG'))
    for b, nz in prod['noise'].items():
        bge = nz.bg_err if nz.bg_err is not None else np.zeros_like(nz.bg)
        t = table_hdu(pd.DataFrame(dict(WAVE=prod['stacks'][b].wave, SD1=nz.sd1, BG=nz.bg, BG_ERR=bge)),
                      f'NOISE_{b}', {'WAVE': 'um', 'SD1': native, 'BG': native, 'BG_ERR': native})
        t.header['ERRSCALE'] = (nz.err_scale, 'null-test NMAD, all region sizes')
        t.header['ERRSCN'] = (json.dumps([[round(float(a), 1), round(float(s), 4)] for a, s in nz.scale_n]),
                              '[[N, scale]] by region size')
        t.header['NSKY'] = (nz.n_sky, 'spaxels of the noise model')
        t.header['SKYSRC'] = (nz.sky_src, 'sky: galaxy excluded; all: too few left')
        t.header['NCORRSRC'] = (nz.ncorr_src, 'c(N) from: sky / all / default')
        t.header['RESOLV'] = (float(BAND_RESOLUTION.get(b, 3500.0)), 'resolving power used for the LSF')
        t.header['NCORR'] = json.dumps([[int(a), round(float(c), 4)] for a, c, _ in nz.ncorr])
        hdus.append(t)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f'{path}.{os.getpid()}.tmp'
    try:
        fits.HDUList(hdus).writeto(tmp, overwrite=True)
        _replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def _replace(src, dst, tries=6, wait=0.5):
    """os.replace, retried while the destination is open elsewhere (Windows, OneDrive, the app)."""
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(wait * (i + 1))


def read_product(path):
    """dict(maps={EXTNAME: 2-D}, bins, total (DataFrames or None), header (dict), config (Config),
    lines (DataFrame), rest {line: micron}, noise={band: dict(wave, sd1, bg, err_scale, ncorr,
    resolving_power)}, clipped={(bin, group tag): array of clipped wavelengths}, path)."""
    from astropy.io import fits
    from astropy.table import Table
    out = dict(maps={}, bins=None, total=None, header={}, config=None, lines=None, rest=None, noise={},
               clipped={}, path=path)
    with fits.open(path) as h:
        out['header'] = {k: h[0].header[k] for k in h[0].header if k and k not in ('HISTORY', 'COMMENT')}
        for e in h[1:]:
            if isinstance(e, fits.BinTableHDU):
                t = Table.read(e).to_pandas()
                for c in t.columns:                       # FITS strings come back as bytes
                    if t[c].dtype == object:
                        t[c] = [v.decode() if isinstance(v, bytes) else v for v in t[c]]
                if e.name == 'BINS':
                    out['bins'] = t
                elif e.name == 'TOTAL':
                    out['total'] = t
                elif e.name == 'LINES':
                    out['lines'] = t
                    out['rest'] = rest_from_lines(t)
                elif e.name == 'CLIPPED':
                    for (b_id, g), sub in t.groupby(['BIN', 'GROUP']):
                        out['clipped'][(int(b_id), str(g).strip())] = sub['WAVE'].to_numpy(float)
                elif e.name == 'CONFIG':
                    try:
                        out['config'] = Config.from_json({k: json.loads(v) for k, v in zip(t['KEY'], t['VALUE'])})
                    except (ValueError, TypeError):
                        out['config'] = None
                elif e.name.startswith('NOISE_'):
                    nc = json.loads(e.header.get('NCORR', '[]'))
                    out['noise'][e.name[6:]] = dict(
                        wave=t['WAVE'].to_numpy(), sd1=t['SD1'].to_numpy(), bg=t['BG'].to_numpy(),
                        bg_err=t['BG_ERR'].to_numpy() if 'BG_ERR' in t else None,
                        err_scale=float(e.header.get('ERRSCALE', 1.0)), ncorr=[(a, c, 0) for a, c in nc],
                        scale_n=[tuple(x) for x in json.loads(e.header.get('ERRSCN', '[]'))],
                        resolving_power=float(e.header.get('RESOLV', np.nan)))
            elif e.data is not None and e.data.ndim == 2:
                out['maps'][e.name] = np.array(e.data)
    for b, nz in out['noise'].items():
        nz['rel'] = out['maps'].get(f'RELNOISE_{b}')
    return out


def used_pixels(product, wave, bin_id, group, z_ref=None):
    """Bool mask of `wave` with the pixels the final fit of (bin, group) used: the group's fit window
    minus the clipped pixels (bin_id < 0: TOTAL).  Finite data are checked by the caller."""
    cfg = product.get('config') or Config()
    hdr = product['header']
    z = hdr.get('Z_REF') if z_ref is None else z_ref
    rest = product.get('rest') or REST
    lines = GROUPS[group]['lines']
    lo = (min(rest[k] for k in lines) - cfg.win_pad) * (1 + z)
    hi = (max(rest[k] for k in lines) + cfg.win_pad) * (1 + z)
    m = (wave > lo) & (wave < hi)
    clip = product.get('clipped', {}).get((int(bin_id), group_tag(group)))
    if clip is not None and len(clip):
        m &= ~np.isin(wave, clip)
    return m


def noise_from_product(nz):
    """BandNoise from a read_product noise entry (what region_spectrum needs)."""
    rel = nz.get('rel')
    return BandNoise(sd1=np.asarray(nz['sd1'], float), bg=np.asarray(nz['bg'], float), ncorr=nz['ncorr'],
                     err_scale=float(nz['err_scale']), scale_n=list(nz.get('scale_n') or []),
                     rel=None if rel is None else np.asarray(rel, float))


def product_bin_spectrum(product, stack, band, bin_id, edge_margin=None):
    """(wave, spec, err) of one bin (bin_id < 0: all bins) in `band`, exactly as fitted:
    background-subtracted sum of the bin's spaxels and its calibrated error; None if the bin has no
    spaxel in that band or the stack does not match the product."""
    maps, hdr = product['maps'], product['header']
    bm = maps.get('BINMAP') if band == hdr.get('ANCHBAND') else maps.get(f'BINMAP_{band}')
    nz = product['noise'].get(band)
    if bm is None or nz is None or bm.shape != stack.cube.shape[1:] or len(nz['sd1']) != stack.wave.size:
        return None
    sel = (bm >= 0) if bin_id < 0 else (bm == bin_id)
    if not sel.any():
        return None
    cfg = product.get('config') or Config()
    noise = noise_from_product(nz)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        spec, err = region_spectrum(stack.cube, sel, noise, cfg.edge_margin if edge_margin is None else edge_margin)
    return stack.wave, spec, err


def model_curve(row, group, wave, z_ref, components=False, rest=None, resolving_power=None):
    """Fitted model of one bin and group on `wave` (continuum + lines), from a BINS (or TOTAL) row.
    components=True returns (total, [continuum + component k]).  rest {line: micron} and the
    resolving power default to this module's tables; pass the product's (read_product: 'rest',
    noise[band]['resolving_power']) to rebuild exactly what was fitted."""
    rest = rest or REST
    G = group_tag(group)
    band = str(row.get(f'{G}_BAND', '') or '').strip()
    x0 = row.get(f'{G}_X0', np.nan)
    if not band or not np.isfinite(x0):
        return (None, []) if components else None
    lsf = sigma_lsf_kms(band, resolving_power if (resolving_power and np.isfinite(resolving_power)) else None)
    cont = row[f'{G}_CONT_A'] + row[f'{G}_CONT_B'] * (wave - x0)
    total = np.array(cont, float)
    parts = []
    k = 1
    while np.isfinite(row.get(f'{G}_C{k}_MV', np.nan)):
        v, s = row[f'{G}_C{k}_MV'], row[f'{G}_C{k}_MSIG']
        comp = np.zeros_like(total)
        for line in GROUPS[group]['lines']:
            fl = row.get(f'{line_tag(line)}_C{k}_FLUX', np.nan)
            if not np.isfinite(fl):
                continue
            mu = rest[line] * (1 + z_ref) * (1 + v / C_KMS)
            comp += fl * _unit_gauss(wave, mu, mu * np.hypot(lsf, s) / C_KMS)
        total += comp
        parts.append(cont + comp)
        k += 1
    return (total, parts) if components else total


def rest_from_lines(lines_df):
    """{line key: rest micron} from a product's LINES table."""
    if lines_df is None or 'LINE' not in lines_df:
        return None
    by_tag = {line_tag(k): k for k in REST}
    return {by_tag.get(str(t).strip(), str(t).strip()): float(r) for t, r in zip(lines_df['LINE'], lines_df['REST_UM'])}


# ======================================================================== command line / batch
PIPELINE_DIR = 'kmos_voronoi_maps'


def default_out_dir(root, variant):
    return os.path.join(root, PIPELINE_DIR, variant)


def product_name(key):
    return key.replace(' ', '_') + '_voronoi.fits'


_W = {}


def _worker_init(root, variant):
    from empower_explorer import engine, lines
    from empower_explorer.data.catalog import Catalog
    warnings.filterwarnings('ignore')
    engine.load(root)
    lines.sync(engine.REST_UM)
    sync_rest(engine.REST_UM, engine.BAND_RESOLUTION)
    sync_sky(root)
    _W['cat'] = Catalog.build(root, variant)


_HEADER_FIELDS = {'nbins': 'NBINS', 'anchor': 'ANCHOR', 'anchor_band': 'ANCHBAND', 'target_sn': 'TARGETSN',
                  'err_scale': 'ERRSCALE', 'closure': 'CLOSURE', 'bin_mode': 'BINMODE', 'status': 'STATUS',
                  'detect': 'DETECT', 'z': 'Z', 'z_ref': 'Z_REF', 'ref_src': 'REFSRC', 'ref_snr': 'REF_SNR',
                  'pipeline_version': 'PIPEVER', 'schema': 'SCHEMA', 'date': 'DATE', 'seconds': 'FITSEC',
                  'variant': 'VARIANT'}


def record_from_header(path):
    """Index record of one product, from its primary header."""
    from astropy.io import fits
    h = fits.getheader(path, 0)
    rec = dict(key=str(h.get('OBJECT', '')), file=os.path.basename(path), message='')
    for k, hk in _HEADER_FIELDS.items():
        v = h.get(hk)
        rec[k] = v.item() if hasattr(v, 'item') else v
    return rec


def _run_one(key, out_dir, cfg_json, overwrite):
    from empower_explorer.data.stack import load_band_stack
    cat = _W['cat']
    path = os.path.join(out_dir, product_name(key))
    rec = dict(key=key, file=os.path.basename(path), status='', nbins=0, message='')
    if os.path.exists(path) and not overwrite:
        try:
            r = record_from_header(path)
            if str(r.get('schema', '')) == SCHEMA and str(r.get('pipeline_version', '')) == PIPELINE_VERSION:
                return r
        except Exception:
            pass
    try:
        zr = cat.redshift(key)
        if not zr.known:
            rec['status'] = 'no_redshift'
            return rec
        cfg = Config.from_json(cfg_json)
        stacks = {b: load_band_stack(infos) for b, infos in cat.cubes[key].items()}
        prod = run_galaxy(stacks, zr.z, cfg, name=key)
        write_product(prod, path, meta=dict(VARIANT=cat.variant, ZTYPE=zr.z_type))
        rec = record_from_header(path)
    except Exception as e:
        rec.update(status='error', message=f'{type(e).__name__}: {e}', traceback=traceback.format_exc())
    return rec


def write_index(out_dir, records=(), variant='', cfg=None):
    """index.json / index.csv of out_dir: one record per product (read from its header), the
    galaxies without a product (no redshift, errors) of the previous index, and - winning over
    everything - this run's no_redshift / error `records` (a file left from an earlier run is then
    reported as stale_file / stale_date, never as this run's result); written atomically."""
    import glob
    import pandas as pd
    path_json = os.path.join(out_dir, 'index.json')
    recs = {}
    try:
        with open(path_json, encoding='utf-8') as f:
            for r in json.load(f).get('products', []):
                if r.get('status') in ('no_redshift', 'error') and r.get('key'):
                    recs[r['key']] = r
    except (OSError, ValueError, AttributeError):
        pass
    for p in sorted(glob.glob(os.path.join(out_dir, '*_voronoi.fits'))):
        try:
            r = record_from_header(p)
        except Exception:
            continue
        if r.get('key'):
            recs[r['key']] = r
    for r in records:
        if r.get('status') in ('no_redshift', 'error'):
            old = recs.get(r['key'])
            new = {k: v for k, v in r.items() if k != 'traceback'}
            if old is not None and old.get('file') and old.get('status') not in ('no_redshift', 'error'):
                new.update(stale_file=old['file'], stale_date=old.get('date'), stale_status=old.get('status'))
            recs[r['key']] = new

    def clean(v):
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v.item() if hasattr(v, 'item') else v
    rows = [{k: clean(v) for k, v in r.items()} for r in sorted(recs.values(), key=lambda r: r['key'])]
    doc = dict(schema=SCHEMA, pipeline_version=PIPELINE_VERSION, variant=variant,
               updated=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
               config=json.loads(cfg.to_json()) if cfg is not None else None, products=rows)
    for name, write in (('index.json', lambda f: json.dump(doc, f, indent=1)),
                        ('index.csv', lambda f: pd.DataFrame(rows).to_csv(f, index=False, lineterminator='\n'))):
        tmp = os.path.join(out_dir, f'{name}.{os.getpid()}.tmp')
        try:
            with open(tmp, 'w', encoding='utf-8', newline='') as f:
                write(f)
            _replace(tmp, os.path.join(out_dir, name))
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


def main(argv=None):
    ap = argparse.ArgumentParser(description='EMPOWER Voronoi binning + multi-Gaussian line maps')
    ap.add_argument('--root', help='EMPOWER_Analysis_and_presentation folder (default: EMPOWER_ROOT or auto)')
    ap.add_argument('--variant', help='reduction folder (default: the app default)')
    ap.add_argument('--galaxy', action='append', default=[], help="galaxy key, e.g. 'COSMOS 92754' (repeatable)")
    ap.add_argument('--all', action='store_true', help='all galaxies of the reduction')
    ap.add_argument('--out', help=f'output folder (default: <root>/{PIPELINE_DIR}/<variant>)')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--overwrite', action='store_true', help='refit galaxies that already have a product')
    ap.add_argument('--max-components', type=int, default=Config.max_components)
    ap.add_argument('--target-sn', type=float, default=Config.target_sn)
    ap.add_argument('--binning', choices=['voronoi', 'spaxel', 'grid'], default=Config.binning)
    ap.add_argument('--grid', type=int, default=Config.grid)
    ap.add_argument('--groups', default=','.join(Config.groups), help='line groups, e.g. ha,oiii,hb,sii')
    ap.add_argument('--no-bgsub', action='store_true', help='do not subtract the residual background')
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):   # 'Hα' in messages must not kill a redirected run (cp1252)
        try:
            stream.reconfigure(errors='backslashreplace')
        except (AttributeError, ValueError):
            pass

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from empower_explorer import config as ee_config
    root = a.root or os.environ.get('EMPOWER_ROOT') or ee_config.empower_root()
    if not root or not ee_config.is_empower_root(root):
        ap.error('EMPOWER folder not found: use --root')
    variant = a.variant or ee_config.DEFAULT_VARIANT
    out_dir = a.out or default_out_dir(root, variant)
    os.makedirs(out_dir, exist_ok=True)
    try:
        cfg = Config(max_components=a.max_components, target_sn=a.target_sn, binning=a.binning, grid=a.grid,
                     groups=tuple(g for g in a.groups.split(',') if g.strip()),
                     target_fallback=tuple(t for t in Config.target_fallback if t < a.target_sn),
                     bg_subtract=not a.no_bgsub)
    except ValueError as e:
        ap.error(str(e))
    if not a.all and not a.galaxy:
        ap.error('use --all or --galaxy')
    _worker_init(root, variant)
    keys = list(_W['cat'].galaxies.index) if a.all else a.galaxy
    unknown = [k for k in keys if k not in _W['cat'].galaxies.index]
    if unknown:
        ap.error(f'unknown galaxies: {unknown}')
    print(f'{len(keys)} galaxies · {variant} · max {cfg.max_components} components · target S/N {cfg.target_sn} '
          f'· {a.workers} workers -> {out_dir}', flush=True)
    t0 = time.time()
    records = []

    def index():
        try:
            write_index(out_dir, records, variant, cfg)
        except OSError as e:                     # a locked index must not stop the batch
            print(f'(index not written: {e})', flush=True)

    def report(i, r):
        records.append(r)
        print(f'[{i + 1}/{len(keys)}] {r["key"]:22s} {str(r.get("status")):13s} bins={str(r.get("nbins")):<4} '
              f'{r.get("message", "")}', flush=True)
        if (i + 1) % 20 == 0:
            index()
    if a.workers > 1 and len(keys) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(a.workers, initializer=_worker_init, initargs=(root, variant)) as ex:
            futs = {ex.submit(_run_one, k, out_dir, cfg.to_json(), a.overwrite): k for k in keys}
            for i, fu in enumerate(as_completed(futs)):
                report(i, fu.result())
    else:
        for i, k in enumerate(keys):
            report(i, _run_one(k, out_dir, cfg.to_json(), a.overwrite))
    index()
    n_err = sum(r.get('status') == 'error' for r in records)
    print(f'done: {len(records) - n_err}/{len(records)} without error in {time.time() - t0:.0f} s · index: {out_dir}',
          flush=True)
    for r in records:
        if r.get('status') == 'error':
            print(f'--- {r["key"]}\n{r.get("traceback", r.get("message"))}')
    return records


if __name__ == '__main__':
    main()
