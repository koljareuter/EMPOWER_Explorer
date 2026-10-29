"""One stacked cube per (galaxy, band), exactly as kmos_sample.load_galaxy_band_stacks
(normalise=False) / load_stack_with_stat: all cubes of the galaxy in that band, aligned on the
sky via the WCS and weighted by texp (x A(lambda)^2 for telluric-corrected cubes, kmos_sample.
_stack_weights); STAT (1 sigma) propagated; transmission from the TELLEFF_ files (ATM extension).
The pixel grid is the one the resolved map products were made on.

Extensions are found by a part of their name (find_ext): the cube is the first 3-D extension whose EXTNAME
contains DATA ('DATA', 'CUBE-DATA', 'IFU.3.DATA', ...), its 1-sigma error the one containing STAT (else NOISE
or ERR), the transmission the one containing ATM.
"""
import os
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from empower_explorer import engine


@dataclass
class BandStack:
    band: str
    wave: np.ndarray            # (nz,) observed vacuum wavelength [micron]
    cube: np.ndarray            # (nz, ny, nx) DATA, ADU/s (telluric-corrected variants: / T_eff)
    stat: np.ndarray | None     # (nz, ny, nx) 1-sigma, same units, or None
    wcs: object                 # celestial astropy WCS of the stacked grid (or None)
    exptime_s: float            # on-source time, summed over the cubes (NCOMBINE x DIT x NDIT)
    n_cubes: int
    n_frames: int
    paths: list
    folders: list
    target_lists: list
    teff: np.ndarray | None = None   # instrument response x atmosphere (telluric variants only)
    atm: np.ndarray | None = None    # atmosphere only, ~1 = clear sky
    bunit: str = ''
    header0: dict = field(default_factory=dict)   # primary header of the first cube (QA)
    # derived
    coverage: np.ndarray = None     # (ny, nx) fraction of valid wavelengths per spaxel
    white: np.ndarray = None        # (ny, nx) nanmedian over wavelength
    wave_min: float = np.nan        # wavelength range with data
    wave_max: float = np.nan
    pixscale: float = 0.2           # arcsec / pixel (from the WCS)

    def covers(self, lam):
        return bool(self.wave_min <= lam <= self.wave_max)

    @property
    def shape(self):
        return self.cube.shape

    @property
    def texp_h(self):
        return self.exptime_s / 3600.0 if np.isfinite(self.exptime_s) else np.nan

    def describe(self):
        t = self.texp_h
        tt = (f'{t:.1f} h' if t >= 1 else f'{t * 60:.0f} min') if np.isfinite(t) else ''
        n = f' · {self.n_cubes} cubes' if self.n_cubes > 1 else ''
        return f'{self.band} {tt}{n}'.strip()


ERROR_KEYS = ('STAT', 'NOISE', 'ERR')


def find_ext(hdul, key, ndim=None, avoid=()):
    """First extension whose EXTNAME contains `key` (case-insensitive), with `ndim` data axes if given; an
    exact match wins, then names without any of `avoid`; None if there is none."""
    def name(e):
        return str(e.header.get('EXTNAME', '')).strip().upper()
    key = key.upper()
    hits = [e for e in hdul[1:] if key in name(e) and (ndim is None or e.header.get('NAXIS') == ndim)]
    exact = [e for e in hits if name(e) == key]
    clean = [e for e in hits if not any(a in name(e) for a in avoid)]
    return (exact or clean or hits or [None])[0]


def data_ext(hdul):
    """The cube extension: 3-D, EXTNAME containing DATA (not one of the error ones, e.g. DATA_ERR)."""
    return find_ext(hdul, 'DATA', ndim=3, avoid=ERROR_KEYS)


def error_ext(hdul, ndim=3):
    """The 1-sigma extension of a cube: EXTNAME containing STAT, else NOISE, else ERR; None without one."""
    for key in ERROR_KEYS:
        e = find_ext(hdul, key, ndim)
        if e is not None:
            return e
    return None


def _telluric_atm(path):
    """Atmosphere-only transmission from the TELLEFF_ file next to a cube (extension containing ATM), else None
    (kmos_sample.telluric_atm_of, with the extension found by name part)."""
    p = os.path.join(os.path.dirname(path), 'TELLEFF_' + os.path.basename(path)[len('DATACUBE_COMBINED_'):])
    if not os.path.exists(p):
        return None
    with fits.open(p) as h:
        e = find_ext(h, 'ATM')
        return np.asarray(e.data, float) if e is not None and e.data is not None else None


def load_band_stack(infos):
    """Stack the cubes `infos` (scan infos of ONE galaxy and band) -> BandStack."""
    ks = engine.ks
    infos = sorted(infos, key=lambda i: i['path'])        # same order as kmos_sample (sorted glob)
    cubes, stats, wcss, wave, bunit, header0 = [], [], [], None, '', {}
    for k, m in enumerate(infos):
        with fits.open(m['path'], memmap=True) as hf:
            dext = data_ext(hf)
            if dext is None:
                raise ValueError(f'{os.path.basename(m["path"])}: no 3-D extension with DATA in its name')
            sext = error_ext(hf)
            hdr = dext.header
            cube = np.array(dext.data, dtype=np.float32)
            stat = np.array(sext.data, dtype=np.float32) if sext is not None else None
            if k == 0:
                bunit = str(hdr.get('BUNIT', '')).strip()
                header0 = {key: hf[0].header[key] for key in hf[0].header
                           if key and isinstance(hf[0].header[key], (str, int, float, bool))}
        w = ks.wavelength_axis(hdr)
        if wave is None:
            wave = w
        elif not np.allclose(wave, w):
            raise ValueError(f"wavelength axes differ for {m['key']} {m['band']}")
        cubes.append(cube)
        stats.append(stat)
        wcss.append(WCS(hdr).celestial)
    # weights exactly as load_galaxy_band_stacks(normalise=False) / load_stack_with_stat: texp, and
    # texp * A(lambda)^2 for telluric-corrected cubes (inverse variance after the telluric division)
    texps = [m['exptime_s'] if np.isfinite(m['exptime_s']) and m['exptime_s'] > 0 else 1.0 for m in infos]
    tells = [ks.telluric_of(m['path']) for m in infos]
    if not all(t is not None and t.shape == wave.shape for t in tells):
        tells = [None] * len(infos)
    atms = []                          # kmos_sample._member_atm: the ATM extension, else derived from T_eff
    for m, t in zip(infos, tells):
        a = _telluric_atm(m['path']) if t is not None else None
        atms.append(a if a is not None else (ks.atmospheric_transmission(t, wave) if t is not None else None))
    weights = ks._stack_weights(texps, tells, atms, False)
    have_stat = all(s is not None and s.shape == c.shape for s, c in zip(stats, cubes))
    if have_stat:
        cube, (sy0, sx0), stat = ks._stack(cubes, weights, wcss, stats)
    else:
        cube, (sy0, sx0) = ks._stack(cubes, weights, wcss)
        stat = None
    wcs = wcss[0].deepcopy()
    wcs.wcs.crpix = wcs.wcs.crpix + np.array([sx0, sy0])

    # transmission of the stack (T_eff and atmosphere-only A from the TELLEFF ATM extension),
    # combined with the same weights as the cubes
    teff = atm = None
    if tells[0] is not None:
        if len(infos) == 1:
            teff, atm = tells[0], atms[0]
        else:
            wl = [np.broadcast_to(np.asarray(w, float).reshape(-1), wave.shape) for w in weights]
            T = np.vstack(tells)
            wt = np.vstack(wl) * np.isfinite(T)
            with np.errstate(invalid='ignore', divide='ignore'):
                teff = np.where(wt.sum(0) > 0, np.nansum(np.nan_to_num(T) * wt, 0) / wt.sum(0), np.nan)
                if all(a is not None for a in atms):
                    Am = np.vstack(atms)
                    wa = np.vstack(wl) * np.isfinite(Am)
                    atm = np.where(wa.sum(0) > 0, np.nansum(np.nan_to_num(Am) * wa, 0) / wa.sum(0), np.nan)

    st = BandStack(band=infos[0]['band'], wave=wave, cube=np.asarray(cube, np.float32),
                   stat=None if stat is None else np.asarray(stat, np.float32), wcs=wcs,
                   exptime_s=float(np.nansum([m['exptime_s'] for m in infos])),
                   n_cubes=len(infos), n_frames=int(sum(m['n_frames'] for m in infos)),
                   paths=[m['path'] for m in infos], folders=sorted({m['folder'] for m in infos}),
                   target_lists=sorted({m['target_list'] for m in infos}),
                   teff=teff, atm=atm, bunit=bunit, header0=header0)
    finite = np.isfinite(st.cube)
    st.coverage = finite.mean(axis=0)
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            st.white = np.nanmedian(st.cube, axis=0)
    has = finite.any(axis=(1, 2))
    if has.any():
        st.wave_min, st.wave_max = float(wave[has].min()), float(wave[has].max())
    try:
        st.pixscale = float(np.mean(proj_plane_pixel_scales(wcs))) * 3600.0
    except Exception:
        st.pixscale = 0.2
    return st
