"""Science functions (pure numpy, no Qt) - the line S/N, noise and stacking are the notebooks' own
code (engine.lsnr = scripts/line_snr.py, engine.ks = scripts/kmos_sample.py).

Pixel coordinates: x = column, y = row, 0-based, of a BandStack grid (cube[:, y, x]).

    spec = extract_spectrum(stack, x, y, r)            # aperture spectrum (sum), STAT error, fit noise
    fits = fit_lines(spec, z)                           # {line key: LineFit} + FitInfo (anchor)
    gfit = fit_galaxy(spectra_by_band, stacks, z)       # all lines of a galaxy, each from the deepest band
    img  = line_map(stack, z, 'Ha')                     # continuum-subtracted narrow-band map
    x, y = line_peak_center(stack, z, 'Ha')             # notebook Hα-peak centring
    maps = spaxel_fit_maps(stack, z, 'Ha')              # per-spaxel flux/S/N/Δv/σ/W80 (slow: worker)
"""
import warnings
from dataclasses import dataclass, field

import numpy as np

from empower_explorer import config, engine, lines
from empower_explorer.lines import C_KMS, FIT_LINES, REST

W80_PER_SIGMA = 2.5631        # W80 of a Gaussian = 2 x 1.2816 sigma
EDGE_MARGIN = 4               # notebooks: band-edge erosion [px]
MAX_CENTER_OFFSET = 4.0       # notebooks: line peak farther than this from the centre -> centre


# NaN-heavy cube arithmetic warns constantly ('All-NaN slice', 'Mean of empty slice', ...). These
# warnings carry no information here; they are silenced once, globally, because
# warnings.catch_warnings() is not thread-safe and fits run in worker threads.
warnings.filterwarnings('ignore', category=RuntimeWarning)


class _NoOp:
    def __exit__(self, *a):
        return False


def _quiet():
    return _NoOp()


# ------------------------------------------------------------------ apertures and spectra
def aperture_mask(shape, x, y, r, r_in=0.0):
    """Circular aperture (radius r px) at (x, y) on a (ny, nx) grid; with r_in > 0 the annulus
    r_in < d <= r, so neighbouring annuli (and the circle inside) never share a spaxel."""
    yy, xx = np.indices(shape)
    d2 = (xx - x) ** 2 + (yy - y) ** 2
    return (d2 <= r ** 2) & (d2 > r_in ** 2) if r_in > 0 else d2 <= r ** 2


COSMO_H0, COSMO_OM0 = 70.0, 0.3      # flat LambdaCDM of the notebooks (kmos_snr_annuli_reff.ipynb)
_KPC_PER_ARCSEC = {}


def kpc_per_arcsec(z):
    """Proper kpc per arcsec at redshift z (flat LambdaCDM, H0 = 70, Om0 = 0.3); NaN without z."""
    if z is None or not np.isfinite(z) or z <= 0:
        return np.nan
    key = round(float(z), 6)
    if key not in _KPC_PER_ARCSEC:
        from astropy.cosmology import FlatLambdaCDM
        cosmo = FlatLambdaCDM(H0=COSMO_H0, Om0=COSMO_OM0)
        _KPC_PER_ARCSEC[key] = float(cosmo.kpc_proper_per_arcmin(key).value / 60.0)
    return _KPC_PER_ARCSEC[key]


def half_light_radius(image, x, y):
    """Radius [px] around (x, y) that encloses half of the total flux of `image` inside the largest
    circle that fits on the grid (curve of growth, as reff_from_image of kmos_snr_annuli_reff.ipynb);
    NaN if the image has no positive total."""
    if image is None or not np.isfinite(image).any():
        return np.nan
    ny, nx = image.shape
    r_max = min(x, y, nx - 1 - x, ny - 1 - y)
    if r_max < 1:
        return np.nan
    yy, xx = np.indices(image.shape)
    dist = np.hypot(xx - x, yy - y)
    im = np.where(np.isfinite(image), image, 0.0)
    steps = np.linspace(0.0, float(r_max), int(r_max * 10) + 2)
    cum = np.array([im[dist <= r].sum() for r in steps])
    total = cum[-1]
    if not np.isfinite(total) or total <= 0:
        return np.nan
    i = int(np.searchsorted(cum, 0.5 * total))
    if i == 0:
        return float(steps[0])
    if i >= len(steps):
        return float(steps[-1])
    f0, f1, r0, r1 = cum[i - 1], cum[i], steps[i - 1], steps[i]
    return float(r0) if f1 == f0 else float(r0 + (0.5 * total - f0) / (f1 - f0) * (r1 - r0))


@dataclass
class Spectrum:
    band: str
    wave: np.ndarray              # observed micron
    flux: np.ndarray              # aperture sum (method 'sum') or weighted mean ('mean')
    err: np.ndarray | None        # propagated from STAT (None if the stack has no STAT)
    noise: np.ndarray | None      # per-pixel 1 sigma from the spaxel scatter (line_snr.noise_from_cube)
    n_pix: int
    method: str = 'sum'
    bg_subtracted: bool = False
    atm: np.ndarray | None = None

    @property
    def snr(self):
        e = self.err if self.err is not None else self.noise
        if e is None:
            return np.full_like(self.flux, np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(e > 0, self.flux / e, np.nan)


def residual_background(stack, x, y, r, r_min=6.0, min_spaxels=20):
    """Per-wavelength median of the spaxels >= max(r_min, r + 2) px from (x, y): the additive
    sky/background pedestal (kmos_sample.subtract_residual_background); None if too few spaxels."""
    yy, xx = np.indices(stack.cube.shape[1:])
    far = (np.hypot(xx - x, yy - y) >= max(r_min, r + 2)) & (stack.coverage > 0.5)
    if far.sum() < min_spaxels:
        return None
    w = _quiet()
    try:
        bg = np.nanmedian(stack.cube[:, far], axis=1)
    finally:
        w.__exit__(None, None, None)
    return np.where(np.isfinite(bg), bg, 0.0)


def _ring_noise(stack, mask, x, y, r, min_bg=8):
    """Fit noise of an annulus from the spaxels outside its outer radius (>= r + 2 px, else >= r + 1 px), as
    kmos_snr_distribution.ipynb: the default background (everything outside the dilated mask) would include
    the galaxy inside the ring.  None if fewer than min_bg spaxels are left (measure_lines then estimates it)."""
    yy, xx = np.indices(stack.cube.shape[1:])
    d = np.hypot(xx - x, yy - y)
    for gap in (2.0, 1.0):
        bg = (d >= r + gap) & (stack.coverage > 0)
        if bg.sum() >= min_bg:
            return engine.lsnr.noise_from_cube(stack.cube, mask, bg_mask=bg)
    return None


def extract_spectrum(stack, x, y, r, method='sum', subtract_bg=False, min_coverage=0.5, r_in=0.0):
    """Aperture spectrum of a BandStack (the annulus r_in < d <= r if r_in > 0).

    method 'sum'  : total flux in the aperture; spaxels missing at a wavelength are filled with the
                    mean of the valid ones (no dips at partially covered wavelengths); NaN where
                    fewer than half of the aperture spaxels are valid.
    method 'mean' : inverse-variance weighted mean per spaxel (STAT), as the old KMOS viewer.
    The fit noise (`noise`) is line_snr.noise_from_cube of the same aperture, scaled like `flux` (for an
    annulus with the spaxels outside it as background, _ring_noise)."""
    mask = aperture_mask(stack.cube.shape[1:], x, y, r, r_in) & (stack.coverage > min_coverage)
    n = int(mask.sum())
    nz = stack.cube.shape[0]
    if n == 0:
        nan = np.full(nz, np.nan)
        return Spectrum(stack.band, stack.wave, nan, None, None, 0, method, False, stack.atm)
    patch = stack.cube[:, mask].astype(np.float64)
    sig = stack.stat[:, mask].astype(np.float64) if stack.stat is not None else None
    fin = np.isfinite(patch) & (np.isfinite(sig) if sig is not None else True)
    nfin = fin.sum(axis=1)
    ok = nfin >= max(1, 0.5 * n)
    w = _quiet()
    try:
        if method == 'mean' and sig is not None:
            wt = np.where(fin & (sig > 0), 1.0 / np.where(sig > 0, sig, np.inf) ** 2, 0.0)
            sw = wt.sum(axis=1)
            flux = np.where(sw > 0, np.nansum(np.where(fin, patch, 0) * wt, axis=1) / np.where(sw > 0, sw, 1), np.nan)
            err = np.where(sw > 0, 1.0 / np.sqrt(np.where(sw > 0, sw, 1)), np.nan)
            scale = 1.0 / n
        else:
            mean = np.nanmean(np.where(fin, patch, np.nan), axis=1)
            flux = mean * n
            err = None
            if sig is not None:
                err = np.sqrt(np.nanmean(np.where(fin, sig, np.nan) ** 2, axis=1) / np.maximum(nfin, 1)) * n
            scale = 1.0
        flux = np.where(ok, flux, np.nan)
        if err is not None:
            err = np.where(ok, err, np.nan)
        if r_in > 0:
            noise = _ring_noise(stack, mask, x, y, r)
            noise = noise * scale if noise is not None else None
        else:
            noise = engine.lsnr.noise_from_cube(stack.cube, mask) * scale
    finally:
        w.__exit__(None, None, None)
    bg_done = False
    if subtract_bg:
        bg = residual_background(stack, x, y, r)
        if bg is not None:
            flux = flux - bg * (n if method == 'sum' else 1)
            bg_done = True
    return Spectrum(stack.band, stack.wave, flux, err, noise, n, method, bg_done, stack.atm)


def bin_spectrum(stack, binmap, bin_id, method='sum'):
    """Spectrum of all spaxels with binmap == bin_id (Voronoi products; binmap on the stack grid)."""
    if binmap is None or binmap.shape != stack.cube.shape[1:]:
        return None
    mask = binmap == bin_id
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    patch = stack.cube[:, mask].astype(np.float64)
    w = _quiet()
    try:
        flux = np.nanmean(patch, axis=1) * mask.sum() if method == 'sum' else np.nanmean(patch, axis=1)
        err = None
        if stack.stat is not None:
            s = stack.stat[:, mask].astype(np.float64)
            err = np.sqrt(np.nansum(s ** 2, axis=1)) if method == 'sum' else np.sqrt(np.nansum(s ** 2, axis=1)) / mask.sum()
        noise = engine.lsnr.noise_from_cube(stack.cube, mask) * (1.0 if method == 'sum' else 1.0 / mask.sum())
    finally:
        w.__exit__(None, None, None)
    return Spectrum(stack.band, stack.wave, flux, err, noise, int(mask.sum()), method, False, stack.atm)


# ------------------------------------------------------------------ line fits
@dataclass
class LineFit:
    key: str
    band: str
    snr: float = np.nan
    flux: float = np.nan              # spec units x micron
    flux_err: float = np.nan
    dv_kms: float = np.nan            # w.r.t. the catalogue redshift
    sigma_obs_kms: float = np.nan     # observed (LSF-convolved)
    sigma_int_kms: float = np.nan     # intrinsic (LSF removed)
    w80_kms: float = np.nan           # 2.563 sigma_int (single Gaussian)
    center: float = np.nan            # observed micron
    detected: bool = False
    forced: bool = False
    reliable: bool = False
    flags: list = field(default_factory=list)
    kinematics: str = ''              # where velocity/width came from: 'free', 'Hα (IZ)', 'catalogue z'
    wave: np.ndarray | None = None    # fit window
    model: np.ndarray | None = None   # continuum + line model in the fit window
    cont: np.ndarray | None = None
    resid: np.ndarray | None = None   # data - continuum
    err: np.ndarray | None = None

    @property
    def covered(self):
        return not ({'no_coverage', 'too_few_pixels'} & set(self.flags))


@dataclass
class FitInfo:
    anchor: str | None = None         # line that set the kinematics ('Ha', ..., 'external', None)
    dv: float = 0.0
    sigma_int: float = np.nan
    anchor_band: str = ''
    anchor_stat: float = -np.inf      # min(fit S/N, aperture S/N) of the anchor search
    forced_from: str = ''             # set when every line was forced to these kinematics (annulus: the centre)


def sigma_lsf(band):
    return C_KMS / (engine.BAND_RESOLUTION.get((band or '').upper(), 3500.0) * 2.3548)


def _to_linefit(key, band, r):
    lf = LineFit(key=key, band=band, snr=float(r.snr), flux=float(r.flux), flux_err=float(r.flux_err),
                 dv_kms=float(r.dv_kms), sigma_obs_kms=float(r.sigma_kms), center=float(r.center),
                 detected=bool(r.detected), forced=bool(r.forced), reliable=bool(r.reliable),
                 flags=list(r.flags))
    if np.isfinite(lf.sigma_obs_kms):
        lf.sigma_int_kms = float(np.sqrt(max(lf.sigma_obs_kms ** 2 - sigma_lsf(band) ** 2, 0.0)))
        lf.w80_kms = W80_PER_SIGMA * lf.sigma_int_kms
    if r.wave is not None and r.model is not None and r.cont is not None:
        lf.wave = np.asarray(r.wave)
        lf.cont = np.asarray(r.cont)
        lf.model = lf.cont + np.asarray(r.model)
        lf.resid = None if r.resid is None else np.asarray(r.resid)
        lf.err = None if r.err is None else np.asarray(r.err)
    return lf


def covered_lines(stack_or_wave, z, keys=FIT_LINES, margin=0.004):
    """Line keys whose observed wavelength lies inside the data range (with a margin)."""
    if hasattr(stack_or_wave, 'wave_min'):
        lo, hi = stack_or_wave.wave_min, stack_or_wave.wave_max
    else:
        w = np.asarray(stack_or_wave)
        lo, hi = np.nanmin(w), np.nanmax(w)
    return [k for k in keys if lo + margin <= REST[k] * (1 + z) <= hi - margin]


def fit_lines(spec, z, keys=None, anchor=None, search=True):
    """Calibrated line measurement of one spectrum (line_snr.measure_lines: one free anchor line,
    all others forced to its velocity and width).  anchor=(dv, sigma_int) from another band;
    search=False: no free anchor, every line forced to `anchor` (or to the catalogue z without one).
    Returns ({key: LineFit}, FitInfo)."""
    keys = keys if keys is not None else covered_lines(spec.wave[np.isfinite(spec.flux)], z) \
        if np.isfinite(spec.flux).any() else []
    if not keys or z is None:
        return {}, FitInfo()
    w = _quiet()
    try:
        extra = {} if search else {'t_anchor': np.inf}
        res, info = engine.lsnr.measure_lines(spec.wave, spec.flux, z, {k: REST[k] for k in keys},
                                              noise=spec.noise, band=spec.band, anchor=anchor, **extra)
    finally:
        w.__exit__(None, None, None)
    out = {k: _to_linefit(k, spec.band, r) for k, r in res.items()}
    stat = info.get('anchor_stat', -np.inf)
    fi = FitInfo(anchor=info.get('anchor'), dv=float(info.get('dv', 0.0)),
                 sigma_int=float(info.get('sigma_int', np.nan)), anchor_band=spec.band,
                 anchor_stat=float(stat) if stat is not None and np.isfinite(stat) else -np.inf)
    own = fi.anchor not in (None, 'external')
    for k, lf in out.items():
        if k == fi.anchor:
            lf.kinematics = 'free'
        elif own:
            lf.kinematics = f'{lines.LINE_LABELS.get(fi.anchor, fi.anchor)} ({spec.band})'
        elif fi.anchor == 'external':
            lf.kinematics = 'other band'
        else:
            lf.kinematics = 'catalogue z'
    return out, fi


def pick_band(stacks, lam_obs):
    """Band used for a line: the deepest band that covers it (notebooks: by exposure, never by S/N)."""
    cands = [b for b, s in stacks.items() if s.covers(lam_obs)]
    if not cands:
        return None
    return max(cands, key=lambda b: (stacks[b].exptime_s if np.isfinite(stacks[b].exptime_s) else 0.0))


def fit_galaxy(spectra, stacks, z, forced=None):
    """All FIT_LINES of a galaxy, each measured in the deepest band that covers it.
    As kmos_catalogue_statistics.ipynb: every band is fitted on its own first; the galaxy's best
    anchor (highest anchor_stat over all bands) is then passed to every band that found no anchor of
    its own, which is re-fitted with those kinematics (measure_lines anchor=(dv, sigma_int)).
    forced=FitInfo: no anchor search, every line forced to forced.dv / forced.sigma_int (to the
    catalogue z if forced.anchor is None) - the lines of an annulus (fit_annulus).
    spectra: {band: Spectrum}; returns ({key: LineFit}, FitInfo of the best anchor)."""
    if z is None or not spectra:
        return {}, FitInfo()
    use = {}
    for k in FIT_LINES:
        b = pick_band({b: s for b, s in stacks.items() if b in spectra}, REST[k] * (1 + z))
        if b is not None:
            use.setdefault(b, []).append(k)
    if forced is not None:
        kin = (forced.dv, forced.sigma_int) if forced.anchor is not None else None
        src = ('catalogue z' if not kin else forced.forced_from if forced.anchor == 'external' else
               f'{lines.LINE_LABELS.get(forced.anchor, forced.anchor)} ({forced.anchor_band}), {forced.forced_from}')
        fits = {}
        for b, keys in use.items():
            res, _ = fit_lines(spectra[b], z, keys=keys, anchor=kin, search=False)
            for lf in res.values():
                lf.kinematics = src
            fits.update(res)
        return fits, forced
    first = {b: fit_lines(spectra[b], z, keys=keys) for b, keys in use.items()}
    own = [info for _, info in first.values() if info.anchor not in (None, 'external')]
    best = max(own, key=lambda i: i.anchor_stat) if own else FitInfo()
    fits = {}
    for b, (res, info) in first.items():
        if info.anchor is None and own:
            res, _ = fit_lines(spectra[b], z, keys=use[b], anchor=(best.dv, best.sigma_int))
            src = f'{lines.LINE_LABELS.get(best.anchor, best.anchor)} ({best.anchor_band})'
            for lf in res.values():
                lf.kinematics = src
        fits.update(res)
    return fits, best


CENTRAL_R = 2.5      # px: the integrated aperture whose kinematics the lines of an annulus are forced to


def central_kinematics(stacks, ref_band, x, y, z, method='sum', subtract_bg=False, r=CENTRAL_R, prefer=None):
    """FitInfo of the integrated aperture (radius r px of ref_band) at (x, y): its free anchor (dv,
    sigma_int) with forced_from naming it; anchor None if no line is strong enough.  prefer='Ha': that
    line's own kinematics when it was fitted freely (the bins' anchor), whatever line won elsewhere."""
    ref = stacks[ref_band]
    spectra = {}
    for b, st in stacks.items():
        bx, by, br = transfer(ref, st, x, y, r)
        sp = extract_spectrum(st, bx, by, br, method=method, subtract_bg=subtract_bg)
        if sp.n_pix > 0:
            spectra[b] = sp
    fits, info = fit_galaxy(spectra, stacks, z)
    where = f'central r = {r:g} px aperture'
    lf = fits.get(prefer) if prefer else None
    if lf is not None and lf.kinematics == 'free':
        return FitInfo(anchor=prefer, dv=lf.dv_kms, sigma_int=lf.sigma_int_kms, anchor_band=lf.band, forced_from=where)
    if info.anchor in (None, 'external'):
        return FitInfo(forced_from=where)
    info.forced_from = where
    return info


def fit_annulus(spectra, stacks, z, ref_band, x, y, method='sum', subtract_bg=False):
    """Lines of an annulus as kmos_snr_annuli_reff / kmos_snr_distribution: no own anchor search, every
    line forced to the kinematics of the integrated central aperture (catalogue z if that has none)."""
    if z is None or not spectra:
        return {}, FitInfo()
    return fit_galaxy(spectra, stacks, z, forced=central_kinematics(stacks, ref_band, x, y, z, method, subtract_bg))


# ------------------------------------------------------------------ maps and centring
def line_map(stack, z, key, half_px=3):
    """Continuum-subtracted narrow-band map of a line (notebooks' narrowband_image): sum of
    2*half_px+1 channels at the line minus the local continuum (median 40-120 Å away)."""
    lam = REST[key] * (1 + z)
    wave = stack.wave
    if not stack.covers(lam):
        return None
    hp = int(np.argmin(np.abs(wave - lam)))
    p0, p1 = max(hp - half_px, 0), min(hp + half_px + 1, stack.cube.shape[0])
    w = _quiet()
    try:
        img = np.nansum(stack.cube[p0:p1], axis=0)
        img[~np.isfinite(stack.cube[p0:p1]).any(axis=0)] = np.nan
        side = (np.abs(wave - lam) > 0.004) & (np.abs(wave - lam) < 0.012)
        if side.sum() >= 10:
            img = img - (p1 - p0) * np.nanmedian(stack.cube[side], axis=0)
    finally:
        w.__exit__(None, None, None)
    return img


def snr_map(stack):
    """Median S/N per spaxel over the band (DATA / STAT)."""
    if stack.stat is None:
        return None
    w = _quiet()
    try:
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.nanmedian(np.where(stack.stat > 0, stack.cube / stack.stat, np.nan), axis=0)
    finally:
        w.__exit__(None, None, None)


def line_peak_center(stack, z, key='Ha', min_nbr=0.08):
    """(x, y) of the brightest line spaxel with real neighbours (notebooks' ha_peak_center); the
    cube centre pixel (nx//2, ny//2, as the notebooks' ifu_center) if the line is not covered or the
    peak lies > MAX_CENTER_OFFSET px from it.  Returns (x, y, found)."""
    ny, nx = stack.cube.shape[1:]
    cx, cy = float(nx // 2), float(ny // 2)
    lam = REST[key] * (1 + z) if z else None
    if lam is None or not stack.covers(lam):
        return cx, cy, False
    hp = int(np.argmin(np.abs(stack.wave - lam)))
    im = np.nansum(stack.cube[max(hp - 3, 0):hp + 4], axis=0)
    im[stack.coverage <= 0.5] = np.nan
    flat = im.ravel()
    order = np.argsort(np.where(np.isfinite(flat), flat, -np.inf))[::-1]
    for idx in order[:50]:
        v = flat[idx]
        if not np.isfinite(v) or v <= 0:
            break
        py, px = np.unravel_index(idx, (ny, nx))
        nb = [im[py + dy, px + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)
              if (dy, dx) != (0, 0) and 0 <= py + dy < ny and 0 <= px + dx < nx]
        nb = [b for b in nb if np.isfinite(b)]
        if nb and np.mean(nb) >= min_nbr * v:
            if np.hypot(px - cx, py - cy) > MAX_CENTER_OFFSET:
                return cx, cy, False
            return float(px), float(py), True
    return cx, cy, False


def default_center(stack):
    """Target position: CRPIX of the stacked grid (0-based)."""
    if stack.wcs is not None:
        return float(stack.wcs.wcs.crpix[0] - 1), float(stack.wcs.wcs.crpix[1] - 1)
    ny, nx = stack.cube.shape[1:]
    return (nx - 1) / 2.0, (ny - 1) / 2.0


def transfer(src, dst, x, y, r=1.0):
    """Aperture (x, y, r) in pixels of stack `src` -> same sky position and size in `dst` (WCS)."""
    if src is dst or src.wcs is None or dst.wcs is None:
        return x, y, r
    try:
        ra, dec = src.wcs.pixel_to_world_values(x, y)
        px, py = dst.wcs.world_to_pixel_values(ra, dec)
        return float(px), float(py), r * src.pixscale / dst.pixscale
    except Exception:
        return x, y, r


def match_band_scales(spectra, ref_band, min_overlap=0.005):
    """{band: factor} putting all bands on the flux scale of ref_band via the ratio of medians in
    the wavelength overlap (>= 50 Å; notebooks' band_scales).  Bands without overlap keep 1."""
    scale = {ref_band: 1.0} if ref_band in spectra else {}
    todo, changed = [b for b in spectra if b not in scale], True
    rng = {b: (np.nanmin(s.wave[np.isfinite(s.flux)]), np.nanmax(s.wave[np.isfinite(s.flux)]))
           for b, s in spectra.items() if np.isfinite(s.flux).any()}
    w = _quiet()
    try:
        while todo and changed:
            changed = False
            for b in list(todo):
                for o in list(scale):
                    if b not in rng or o not in rng:
                        continue
                    lo, hi = max(rng[b][0], rng[o][0]), min(rng[b][1], rng[o][1])
                    if not hi - lo >= min_overlap:
                        continue
                    so, sb = spectra[o], spectra[b]
                    a = np.nanmedian(so.flux[(so.wave > lo) & (so.wave < hi)]) * scale[o]
                    c = np.nanmedian(sb.flux[(sb.wave > lo) & (sb.wave < hi)])
                    if np.isfinite(a) and np.isfinite(c) and a > 0 and c > 0:
                        scale[b] = a / c
                        todo.remove(b)
                        changed = True
                        break
    finally:
        w.__exit__(None, None, None)
    return {b: scale.get(b, 1.0) for b in spectra}


def telluric_spans(stack, thr=0.5):
    """[(lo, hi)] observed micron where the atmosphere transmits < thr (telluric variants only)."""
    if stack.atm is None:
        return []
    return engine.ks.low_transmission_spans(stack.wave, stack.atm, thr)


_SKY = {}


def sky_lines(keep_fraction=0.3):
    """Vacuum wavelengths [micron] of the bright OH sky lines in scripts/sky_lines_uves_oliva.csv
    (shipped with the app): the brightest `keep_fraction` of each source catalogue (the file mixes
    absolute Hanuschik 2003 fluxes with relative Oliva+15 intensities, so one flux threshold would not work)."""
    path = config.bundled('sky_lines_uves_oliva.csv')
    if path in _SKY:
        return _SKY[path]
    out = np.array([])
    try:
        import pandas as pd
        df = pd.read_csv(path)
        lam_um = pd.to_numeric(df['lambda_A'], errors='coerce').to_numpy(float) * 1e-4
        flux = pd.to_numeric(df['flux'], errors='coerce').to_numpy(float) if 'flux' in df.columns \
            else np.ones(len(df))
        air = df['medium'].astype(str).str.lower().eq('air').to_numpy() if 'medium' in df.columns \
            else np.ones(len(df), bool)
        lam_um = np.where(air, air_to_vacuum(lam_um), lam_um)
        source = df['source'].astype(str).to_numpy() if 'source' in df.columns else np.zeros(len(df), str)
        keep = np.zeros(len(df), bool)
        for s in np.unique(source):
            sel = (source == s) & np.isfinite(lam_um)
            f = flux[sel]
            if np.isfinite(f).sum() >= 5:
                cut = np.nanpercentile(f, 100 * (1 - keep_fraction))
                keep[sel] = ~np.isfinite(f) | (f >= cut)
            else:
                keep[sel] = True
        out = np.sort(lam_um[keep])
    except Exception:
        pass
    _SKY[path] = out
    return out


def air_to_vacuum(lam_um):
    """Air -> vacuum wavelength (Ciddor 1996 / Morton 2000 refractive index), micron."""
    s2 = (1.0 / np.asarray(lam_um, float)) ** 2
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2)
    return lam_um * n


def spaxel_fit_maps(stack, z, key='Ha', binning=1, min_coverage=0.5, progress=None):
    """Per-spaxel (or binning x binning block) single-line fit maps with line_snr.measure_line_snr
    (free velocity within ±500 km/s, companions tied, noise from the spaxel scatter).
    Returns {'flux', 'flux_err', 'snr', 'dv', 'sigma_int', 'w80', 'detected'} arrays on the stack
    grid (blocks filled with the block value).  'snr' is the SEARCHED (uncalibrated) fit S/N;
    'detected' applies the calibrated anchor threshold.  ~5 ms per fit: run in a worker."""
    ny, nx = stack.cube.shape[1:]
    names = ('flux', 'flux_err', 'snr', 'dv', 'sigma_int', 'w80', 'detected')
    out = {n: np.full((ny, nx), np.nan) for n in names}
    lam = REST[key] * (1 + z)
    if not stack.covers(lam):
        return out
    b = max(1, int(binning))
    blocks = [(y0, x0) for y0 in range(0, ny, b) for x0 in range(0, nx, b)]
    lsf = sigma_lsf(stack.band)
    t_anchor = float(getattr(engine.lsnr, 'T_ANCHOR', 5.0))
    w = _quiet()
    try:
        # per-wavelength spaxel scatter once for the whole cube (line_snr.noise_from_cube computes
        # the same per aperture; for 1-9 px apertures excluding them changes nothing) -> 20x faster
        good = stack.coverage > min_coverage
        bg = stack.cube[:, good].astype(np.float64)
        med = np.nanmedian(bg, axis=1, keepdims=True)
        sd = 1.4826 * np.nanmedian(np.abs(bg - med), axis=1)
        sd[np.isfinite(bg).sum(axis=1) < 8] = np.nan
        for i, (y0, x0) in enumerate(blocks):
            if progress is not None and i % 8 == 0:
                progress(i, len(blocks))
            m = np.zeros((ny, nx), bool)
            m[y0:y0 + b, x0:x0 + b] = True
            m &= good
            if not m.any():
                continue
            spec = np.nanmean(stack.cube[:, m], axis=1) * m.sum()
            noise = sd * np.sqrt(m.sum())
            r = engine.lsnr.measure_line_snr(stack.wave, spec, z, REST[key], noise=noise, band=stack.band)
            if {'no_coverage', 'too_few_pixels'} & set(r.flags):
                continue
            sig_int = np.sqrt(max(r.sigma_kms ** 2 - lsf ** 2, 0.0)) if np.isfinite(r.sigma_kms) else np.nan
            # a free velocity/width search is not a calibrated S/N (line_snr docstring): 'detected' uses
            # the anchor rule of measure_lines instead, min(fit S/N, aperture S/N) >= T_ANCHOR (5)
            stat = min(r.snr, r.snr_ap) if np.isfinite(r.snr_ap) else -np.inf
            vals = dict(flux=r.flux, flux_err=r.flux_err, snr=r.snr, dv=r.dv_kms, sigma_int=sig_int,
                        w80=W80_PER_SIGMA * sig_int, detected=float(bool(r.reliable and stat >= t_anchor)))
            for n, v in vals.items():
                out[n][m] = v
        if progress is not None:
            progress(len(blocks), len(blocks))
    finally:
        w.__exit__(None, None, None)
    return out


def rest_wave(wave, z):
    return wave / (1.0 + z) if z else wave


# ------------------------------------------------------------------ radial profiles
def registered_stack(stack, vmap):
    """Copy of `stack` with every spaxel shifted to rest at its velocity vmap[y, x] [km/s] (NaN: not
    shifted), so the rotation is taken out before spaxels are summed (velocity-registered rings, as the
    annular stacks of kmos_best3_single_gauss.ipynb).  Masked pixels stay masked after the shift; the
    stack passed in is not changed."""
    import copy
    wave = stack.wave
    arrays = [np.array(stack.cube, dtype=np.float32, copy=True)]
    if stack.stat is not None:
        arrays.append(np.array(stack.stat, dtype=np.float32, copy=True))
    ys, xs = np.nonzero(np.isfinite(vmap) & (vmap != 0))
    for y, x in zip(ys, xs):
        for arr in arrays:
            arr[:, y, x] = _shift(wave, arr[:, y, x], vmap[y, x])
    st = copy.copy(stack)
    st.cube = arrays[0]
    st.stat = arrays[1] if len(arrays) > 1 else None
    return st


def _shift(wave, col, v):
    """col(wave) of a spaxel at velocity v [km/s], shifted to rest: the value at wave comes from
    wave (1 + v/c); NaN where the source pixel is masked."""
    col = np.asarray(col, np.float64)
    ok = np.isfinite(col)
    if ok.sum() < 10:
        return col
    src = wave * (1.0 + v / C_KMS)
    val = np.interp(src, wave, np.where(ok, col, 0.0), left=np.nan, right=np.nan)
    good = np.interp(src, wave, ok.astype(float), left=0.0, right=0.0)
    return np.where(good > 0.999, val, np.nan)


def _outer_bg(stack, x, y, r, min_bg=8):
    """Background spaxels for the noise of a ring: on the data and >= r + 2 px (else r + 1 px) from (x, y)."""
    yy, xx = np.indices(stack.cube.shape[1:])
    d = np.hypot(xx - x, yy - y)
    for gap in (2.0, 1.0):
        bg = (d >= r + gap) & (stack.coverage > 0)
        if bg.sum() >= min_bg:
            return bg
    return None


def _registered_noise(raw, mask, vmap, x, y, r, min_bg=8):
    """Noise of a velocity-registered ring sum: the per-wavelength scatter of the background spaxels of the
    UNshifted cube (where the sky lines of all spaxels line up), shifted with every ring spaxel by its own
    velocity - like the data - and added in quadrature.  None without background spaxels."""
    bg = _outer_bg(raw, x, y, r, min_bg)
    if bg is None:
        return None
    w = _quiet()
    try:
        b = raw.cube[:, bg].astype(np.float64)
        med = np.nanmedian(b, axis=1, keepdims=True)
        sd = engine.lsnr.MAD2STD * np.nanmedian(np.abs(b - med), axis=1)
        sd[np.isfinite(b).sum(1) < min_bg] = np.nan
    finally:
        w.__exit__(None, None, None)
    var = np.zeros_like(sd)
    nfin = np.zeros_like(sd)
    for yy, xx in zip(*np.nonzero(mask)):
        v = vmap[yy, xx] if vmap is not None and np.isfinite(vmap[yy, xx]) else 0.0
        s = _shift(raw.wave, sd, v) if v else sd
        ok = np.isfinite(s) & np.isfinite(raw.cube[:, yy, xx])
        var[ok] += s[ok] ** 2
        nfin += ok
    return np.where(nfin > 0, np.sqrt(var), np.nan)


def _fluxk(stack):
    """(FLUXK [1e-17 erg/s/cm2/A per cube unit], relative error) of a stack's first cube; NaN if absent."""
    h = getattr(stack, 'header0', None) or {}
    try:
        fk = float(h.get('FLUXK'))
    except (TypeError, ValueError):
        return np.nan, np.nan
    try:
        fe = float(h.get('FLUXKERR'))
    except (TypeError, ValueError):
        fe = 0.1
    return fk, fe


def _log_ratio(fits, num, den, stacks=None, snr_min=3.0):
    """(log10 of the summed fluxes num / den, error); NaN unless every line has S/N >= snr_min.  Lines from
    different bands are put on one scale with each band's FLUXK first (+ its calibration error), as the
    pipeline's LOG_*_XBAND ratios; without FLUXK such a ratio is NaN."""
    num = (num,) if isinstance(num, str) else num
    den = (den,) if isinstance(den, str) else den
    fa = [fits.get(k) for k in num]
    fb = [fits.get(k) for k in den]
    if any(f is None or not (f.snr >= snr_min) for f in fa + fb):
        return np.nan, np.nan
    bands = {f.band for f in fa + fb}
    scale, cal = {b: 1.0 for b in bands}, 0.0
    if len(bands) > 1:
        ks = {b: _fluxk(stacks[b]) if stacks is not None and b in stacks else (np.nan, np.nan) for b in bands}
        if not all(np.isfinite(k[0]) for k in ks.values()):
            return np.nan, np.nan
        scale = {b: k[0] for b, k in ks.items()}
        cal = 0.4343 * float(np.sqrt(sum(k[1] ** 2 for k in ks.values())))
    a = sum(f.flux * scale[f.band] for f in fa)
    b = sum(f.flux * scale[f.band] for f in fb)
    ea = np.sqrt(sum((f.flux_err * scale[f.band]) ** 2 for f in fa))
    eb = np.sqrt(sum((f.flux_err * scale[f.band]) ** 2 for f in fb))
    if not (a > 0 and b > 0):
        return np.nan, np.nan
    return float(np.log10(a / b)), float(np.hypot(0.4343 * np.hypot(ea / a, eb / b), cal))


def _ew(lf, z):
    """(rest-frame EW [Angstrom], error) of a line fit; NaN unless line S/N >= 3 and continuum S/N >= 3
    (the continuum error from the scatter of the window pixels away from the line, as the pipeline's EW_HA cut)."""
    if lf is None or lf.cont is None or lf.wave is None or lf.resid is None or not (lf.snr >= 3):
        return np.nan, np.nan
    c = float(np.interp(lf.center, lf.wave, lf.cont))
    s_obs = lf.center * lf.sigma_obs_kms / C_KMS if np.isfinite(lf.sigma_obs_kms) else 0.0005
    side = np.abs(lf.wave - lf.center) > 3 * s_obs
    r = lf.resid[side]
    r = r[np.isfinite(r)]
    if c <= 0 or r.size < 5:
        return np.nan, np.nan
    c_err = 1.4826 * np.median(np.abs(r - np.median(r))) / np.sqrt(r.size)
    if not (c_err > 0 and c / c_err >= 3):
        return np.nan, np.nan
    ew = lf.flux / c * 1e4 / (1 + z)
    return float(ew), float(abs(ew) * np.hypot(lf.flux_err / lf.flux, c_err / c))


def _common_background(stack, x, y, radii):
    """The residual-sky pedestal used for every ring of a profile in one band: measured outside the largest
    ring radius for which enough spaxels are left (residual_background), so all rings get the same
    correction (None if no radius works: then no ring of that band is corrected)."""
    for r in sorted(radii, reverse=True):
        bg = residual_background(stack, x, y, r)
        if bg is not None:
            return bg, r
    return None, np.nan


def ring_profile(stacks, ref_band, x, y, edges, z, method='sum', subtract_bg=False, vmaps=None, sep_fit=None,
                 anchor_line='Ha', fallback=None, progress=None):
    """Spectra and line fits of the rings edges[i] < d <= edges[i+1] (px of ref_band, edges[0] = 0: a circle)
    around (x, y), every band at the same sky position, fitted at redshift z (with vmaps: the frame the
    spaxels were shifted to).  A ring's kinematics are those of `anchor_line` (the bins' anchor line) when
    that line was fitted freely in the ring; otherwise every line of the ring is forced to the central
    aperture's kinematics (fluxes only).  vmaps {band: km/s map}: velocity-registered rings (registered_stack,
    noise shifted the same way).  subtract_bg: one residual-sky pedestal per band for all rings.
    sep_fit(wave, spec, err, band, start): optional separate Halpha / [NII] fit of each ring.
    fallback=(dv, sigma_int, name): kinematics to force the rings to when the centre has no anchor of its own
    (the Voronoi fit's reference Z_REF / SIG_REF), instead of the catalogue z with a default width.
    Returns ([ring dict], FitInfo of the central aperture, {band: radius [px] of the background region})."""
    use = {b: registered_stack(st, vmaps[b]) if vmaps is not None and vmaps.get(b) is not None else st
           for b, st in stacks.items()}
    ref = use[ref_band]
    centre = central_kinematics(use, ref_band, x, y, z, method, subtract_bg, prefer=anchor_line)
    if centre.anchor is None and fallback is not None and all(np.isfinite(fallback[:2])):
        centre = FitInfo(anchor='external', dv=float(fallback[0]), sigma_int=float(fallback[1]),
                         forced_from=fallback[2])
    pix = ref.pixscale
    pairs = list(zip(edges[:-1], edges[1:]))
    geo = {b: transfer(ref, st, x, y, 1.0) for b, st in use.items()}       # (x, y, px of b per px of ref)
    bgs, bg_r = {}, {}
    if subtract_bg:
        for b, st in use.items():
            bx, by, k = geo[b]
            bgs[b], bg_r[b] = _common_background(st, bx, by, [r_out * k for _, r_out in pairs])
    out = []
    for i, (r_in, r_out) in enumerate(pairs):
        if progress is not None:
            progress(i, len(pairs))
        spectra = {}
        for b, st in use.items():
            bx, by, k = geo[b]
            br, bri = r_out * k, r_in * k
            sp = extract_spectrum(st, bx, by, br, method=method, subtract_bg=False, r_in=bri)
            if sp.n_pix == 0:
                continue
            if st is not stacks[b]:          # registered: noise shifted with the spaxels, like the data
                mask = aperture_mask(st.cube.shape[1:], bx, by, br, bri) & (st.coverage > 0.5)
                nz = _registered_noise(stacks[b], mask, vmaps.get(b), bx, by, br)
                sp.noise = None if nz is None else nz * (1.0 if method == 'sum' else 1.0 / sp.n_pix)
            if bgs.get(b) is not None:
                sp.flux = sp.flux - bgs[b] * (sp.n_pix if method == 'sum' else 1)
                sp.bg_subtracted = True
            spectra[b] = sp
        n = spectra[ref_band].n_pix if ref_band in spectra else 0
        ideal = np.pi * (r_out ** 2 - r_in ** 2)
        rec = dict(r_in=float(r_in), r_out=float(r_out), npix=int(n), area=n * pix ** 2,
                   coverage=n / ideal if ideal > 0 else np.nan, kin='', anchor='', dv=np.nan, sigma=np.nan,
                   w80=np.nan, bg_subtracted=all(s.bg_subtracted for s in spectra.values()) if spectra else False,
                   method=method, npix_b={b: sp.n_pix for b, sp in spectra.items()},
                   pix_b={b: use[b].pixscale for b in spectra})      # a line's area is that of its own band
        if spectra and z is not None:
            fits, info = fit_galaxy(spectra, use, z)
            lf = fits.get(anchor_line)
            own = lf is not None and lf.kinematics == 'free'
            if not own:
                fits, info = fit_galaxy(spectra, use, z, forced=centre)
            rec['kin'] = ('free' if own else 'reference' if centre.anchor == 'external' else
                          'centre' if centre.anchor else 'catalogue z')
            if own:
                rec.update(anchor=anchor_line, dv=lf.dv_kms, sigma=lf.sigma_int_kms, w80=lf.w80_kms)
            for k in FIT_LINES:
                f = fits.get(k)
                rec[f'{k}_flux'] = f.flux if f is not None else np.nan
                rec[f'{k}_err'] = f.flux_err if f is not None else np.nan
                rec[f'{k}_snr'] = f.snr if f is not None else np.nan
                rec[f'{k}_det'] = bool(f.detected) if f is not None else False
                rec[f'{k}_band'] = f.band if f is not None else ''
            rec['n2'], rec['n2_err'] = _log_ratio(fits, 'NII6583', 'Ha', use)
            rec['s2'], rec['s2_err'] = _log_ratio(fits, ('SII6716', 'SII6731'), 'Ha', use)
            rec['o3'], rec['o3_err'] = _log_ratio(fits, 'OIII5007', 'Hb', use)
            rec['ew'], rec['ew_err'] = _ew(fits.get('Ha'), z)
            if sep_fit is not None:
                band = pick_band({b: s for b, s in stacks.items() if b in spectra}, REST['Ha'] * (1 + z))
                if band is not None:
                    sp = spectra[band]
                    e = sp.noise if sp.noise is not None else sp.err
                    start = ((rec['dv'], rec['sigma']) if own else
                             (centre.dv, centre.sigma_int) if centre.anchor else (0.0, 80.0))
                    start = tuple(v if np.isfinite(v) else d for v, d in zip(start, (0.0, 80.0)))
                    rec['sep'] = sep_fit(sp.wave, sp.flux, e, band, start) if e is not None else None
        out.append(rec)
    return out, centre, bg_r
