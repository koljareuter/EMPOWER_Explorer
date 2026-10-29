"""Maps tab - resolved maps of the current galaxy (successor of the old app's 'Kinematic Maps').

Layout
    controls : Source | Band | Line | Binning | Compute (+ progress, Auto) | Mask S/N ≥
    hint     : what the maps are and on which pixel grid they live
    left     : three map panels, each with its own map chooser and a hover value readout in
               the image corner; below them the spectrum of the clicked spaxel (or of its
               Voronoi / kinematic bin or block) around the relevant line with the line fit, a
               residual panel ((data - model)/σ with a ±1σ band; pixels used by the fit bright)
               and a text summary (S/N, Δv, σ_int, W80, flags, map values).  Extraction + fit
               run in a worker (~0.2 s) and are cached; only the latest click is drawn.
    right    : histogram of the first map (log y toggle) and the residual distribution of the
               pixels the selected fit used, with a unit-Gaussian overlay

Map sources
    'Voronoi fit'                    empower_explorer.pipeline.voronoi_binning products (schema
                                     EMPOWER-VORONOI-2; MapProduct.is_pipeline, listed first and shown by
                                     default): Voronoi bins on the anchor-line S/N, 1-3 Gaussian
                                     components per line group (Hβ, [OIII], Hα+[NII], [SII]), ratios;
                                     maps on the anchor-band grid.  A click shows the bin's spectrum
                                     around the chosen line group, taken from the band of that group
                                     (BINMAP_<band>) with the noise model stored in the product
                                     (vb.region_spectrum: exactly the spectrum that was fitted), and the
                                     fitted model rebuilt from the BINS row (vb.model_curve: total +
                                     every component; no refit).  'Compute Voronoi fit' runs
                                     vb.run_galaxy for the current galaxy in a worker and writes the
                                     product to products.pipeline_dir(root, variant)
    MapProduct of state.products()   resolved maps of the notebooks ('Spaxel fits', 'Voronoi',
                                     'Resolved maps'); on the grid of the stack of their band
                                     (header BAND; without BAND: the band with Hα)
    'Cube: line maps'                white light, continuum S/N and continuum-subtracted line
                                     maps straight from the stack of the chosen band (instant)
    'Live fit: per spaxel'           analysis.spaxel_fit_maps for one line and binning, run in a
                                     worker on 'Compute' and cached per (galaxy, band, line, binning)
Product files are read in a worker (first use per galaxy), so the window never waits for the disk.

The spectrum of a clicked pixel is taken from the band that covers the relevant line (the line
of the clicked map, or the one picked in the spectrum header); pixels are carried between band
grids with analysis.transfer (same sky position).
"""
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
                             QProgressBar, QPushButton, QScrollArea, QSpinBox, QSplitter,
                             QStackedWidget, QVBoxLayout, QWidget)

from empower_explorer import __version__, analysis, engine, theme
from empower_explorer.data.products import (PIPELINE_DIR, MapProduct, is_integer_map, is_velocity_map,
                                            pipeline_dir, pretty_map_name)
from empower_explorer.lines import FIT_LINES, LINE_COLORS, LINE_LABELS, PANELS, REST, panel_window
from empower_explorer.theme import C
from empower_explorer.widgets.plots import (EmptyState, ImagePanel, add_spans, add_vline, colormap,
                                            make_plot, robust_levels)
from empower_explorer.workers import run_async

# The Voronoi pipeline (Qt-free, empower_explorer/pipeline).  Without it (e.g. vorbin missing) the tab
# still works, minus the spectra of Voronoi-fit products and the Compute button (the reason is shown).
try:
    from empower_explorer.pipeline import voronoi_binning as vb
    _VB_ERROR = ''
except Exception as _e:          # pragma: no cover - shown in the tab
    vb, _VB_ERROR = None, f'{type(_e).__name__}: {_e}'
_VB_NAME = 'the Voronoi pipeline (empower_explorer/pipeline/voronoi_binning.py)'

LIVE_LINES = ['Ha', 'OIII5007', 'Hb', 'SII6716', 'NII6583']
BINNINGS = [(1, '1×1'), (2, '2×2'), (3, '3×3')]
LIVE_ORDER = ['flux', 'snr', 'dv', 'sigma_int', 'w80', 'flux_err', 'detected']
N_PANELS = 3
MAX_CATEGORIES = 64

# Map-name tokens -> line whose spectrum explains the map (products, ratios included)
_NAME_LINES = [('OIII5007', 'OIII5007'), ('OIII4959', 'OIII4959'), ('NII6583', 'NII6583'),
               ('NII6548', 'NII6548'), ('SII6716', 'SII6716'), ('SII6731', 'SII6731'),
               ('HA', 'Ha'), ('HB', 'Hb'), ('O3', 'OIII5007'), ('N2', 'NII6583'), ('S2', 'SII6716'),
               ('LOG_N2', 'NII6583'), ('LOG_O3HA', 'OIII5007'), ('LOG_S2HA', 'SII6716'),
               ('NE_SII', 'SII6716'), ('OH_M13', 'NII6583'), ('EW_HA', 'Ha'),
               # Voronoi-fit products: group maps ('HANII_V50', 'OIII_V50') and ratios
               ('HANII', 'Ha'), ('HBETA', 'Hb'), ('OIII', 'OIII5007'), ('SII', 'SII6716'),
               ('LOG_O3', 'OIII5007'), ('LOG_S2', 'SII6716'), ('LOG_O3N2', 'OIII5007'),
               ('OH_N2_M13', 'NII6583'), ('SII_RATIO', 'SII6716')]

# Line groups of the Voronoi-fit products (vb.GROUPS; copy for when the pipeline can not be imported)
GROUPS = vb.GROUPS if vb is not None else {
    'hb': dict(tag='HBETA', label='Hβ', lines=['Hb'], main='Hb'),
    'oiii': dict(tag='OIII', label='[OIII]', lines=['OIII4959', 'OIII5007'], main='OIII5007'),
    'ha': dict(tag='HANII', label='Hα+[NII]', lines=['NII6548', 'Ha', 'NII6583'], main='Ha'),
    'sii': dict(tag='SII', label='[SII]', lines=['SII6716', 'SII6731'], main='SII6716')}
GROUP_KEYS = ('hb', 'oiii', 'ha', 'sii')                 # wavelength order (spectrum chooser)
COMP_COLORS = ('#A3E635', '#22D3EE', '#E879F9')           # Gaussian components 1..3 (dashed)
_GROUP_TOKENS = sorted([('OIII4959', 'oiii'), ('OIII5007', 'oiii'), ('OIII', 'oiii'), ('NII6548', 'ha'),
                        ('NII6583', 'ha'), ('HA', 'ha'), ('HANII', 'ha'), ('HB', 'hb'), ('HBETA', 'hb'),
                        ('SII6716', 'sii'), ('SII6731', 'sii'), ('SII', 'sii')], key=lambda t: -len(t[0]))
# <G>_QFLAG bits of the pipeline (vb.QF_*)
_QFLAGS = ((1, 'χ²ν high'), (2, 'pixels clipped'), (4, 'a component unresolved'), (8, '[SII] ratio at a limit'))


def _gtag(g):
    """Column prefix of a line group in the products ('ha' -> 'HANII')."""
    return GROUPS[g].get('tag', g.upper())


def _ltag(line):
    """Column prefix of a line in the products ('Ha' -> 'HA')."""
    return line.upper()
_RATIOS = ['LOG_N2', 'LOG_O3', 'LOG_S2', 'LOG_O3N2', 'OH_N2_M13', 'SII_RATIO', 'NE_SII', 'EW_HA']
# maps of the separate Halpha / [NII] fit (vb.hanii_separate): map name -> (result key, label, tooltip)
_SEP_MAPS = {
    'HA_SEP_V50': ('HA_V50', 'Hα v50, separate fit [km/s]', 'Hα velocity from a fit with its own kinematics'),
    'HA_SEP_W80': ('HA_W80', 'Hα W80, separate fit [km/s]', 'Hα W80 = 2.563 σ_int (one Gaussian, own kinematics)'),
    'HA_SEP_SNR': ('HA_SNR', 'Hα S/N, separate fit', 'Hα S/N of the fit with separate kinematics'),
    'NII_SEP_V50': ('NII_V50', '[NII] v50, separate fit [km/s]',
                    '[NII]6548+6583 velocity from a fit with its own kinematics (6548 = 6583 / 2.94)'),
    'NII_SEP_W80': ('NII_W80', '[NII] W80, separate fit [km/s]', '[NII] W80 = 2.563 σ_int (own kinematics)'),
    'NII_SEP_SNR': ('NII_SNR', '[NII]6583 S/N, separate fit', '[NII]6583 S/N of the fit with separate kinematics'),
    'DV_NII_HA': ('DV_NII_HA', 'Δv [NII] − Hα, separate fit [km/s]',
                  'Velocity of [NII] minus that of Hα (both from the separate fit); ≠ 0: [NII] moves differently '
                  '(outflow, AGN, shocks)'),
}
_SEP_SNR = {'HA_SEP_V50': 'HA_SEP_SNR', 'HA_SEP_W80': 'HA_SEP_SNR', 'HA_SEP_SNR': 'HA_SEP_SNR',
            'NII_SEP_V50': 'NII_SEP_SNR', 'NII_SEP_W80': 'NII_SEP_SNR', 'NII_SEP_SNR': 'NII_SEP_SNR',
            'DV_NII_HA': 'NII_SEP_SNR'}


def _paint_bins(binmap, values):
    """2-D map: every spaxel of a bin gets values[bin] (NaN outside the bins or without a value)."""
    bm = np.asarray(binmap)
    lut = np.full(max(int(bm.max()), 0) + 1, np.nan)
    for b, v in values.items():
        if 0 <= int(b) < lut.size:
            lut[int(b)] = v
    img = np.full(bm.shape, np.nan)
    ok = bm >= 0
    img[ok] = lut[bm[ok]]
    return img


def _separate_job(path, stacks, date):
    """Worker: Halpha and [NII] with separate kinematics in every bin of a Voronoi product - of the file version
    the maps were read from (DATE), else an error (the file was rewritten meanwhile)."""
    P = vb.read_product(path)
    if str(P['header'].get('DATE', '')) != date:
        raise RuntimeError(f'{os.path.basename(path)} was rewritten after it was read: select the galaxy again')
    return vb.hanii_separate(P, stacks)
_RATIO_GROUP = {'LOG_N2': 'ha', 'OH_N2_M13': 'ha', 'EW_HA': 'ha', 'LOG_O3': 'oiii', 'LOG_O3N2': 'oiii',
                'LOG_S2': 'sii', 'SII_RATIO': 'sii', 'NE_SII': 'sii'}


# ---------------------------------------------------------------------------- small helpers
def _fmt(v, unit=''):
    """Compact number for readouts ('–' for NaN)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return '–'
    if not np.isfinite(v):
        return '–'
    s = f'{v:.2e}' if (v != 0 and (abs(v) >= 1e4 or abs(v) < 1e-2)) else f'{v:.3g}'
    return f'{s} {unit}'.strip()


def _bunit(stack):
    u = (getattr(stack, 'bunit', '') or 'ADU/s').replace('/sec', '/s')
    return u


def _line_of_map(name):
    """Line key a map is about ('HA_DV_KMS' -> 'Ha', 'LOG_N2' -> 'NII6583'), None if unknown."""
    if name.startswith('LINE_'):
        return name[5:] if name[5:] in REST else None
    n = name.upper()
    for p in ('SPAX_', 'BIN_', 'KIN_'):
        if n.startswith(p):
            n = n[len(p):]
    for tok, key in sorted(_NAME_LINES, key=lambda kv: -len(kv[0])):
        if n == tok or n.startswith(tok + '_'):
            return key
    return None


def _categorical_cmap(n, binary=False):
    """Stepped colour map with n distinct colours (BINMAP, flags); 0/1 maps grey / emerald."""
    cols = [C.BG_HOVER, C.EMERALD] if binary else \
        [theme.CATEGORY_COLORS[i % len(theme.CATEGORY_COLORS)] for i in range(n)]
    pos, rgb = [], []
    for i, col in enumerate(cols):
        c = pg.mkColor(col).getRgb()[:3]
        pos += [i / n, (i + 1) / n - 1e-6]
        rgb += [c, c]
    return pg.ColorMap(np.array(pos), np.array(rgb, dtype=np.ubyte))


def _finite_runs(mask):
    """[(start, stop)] index ranges where mask is True."""
    idx = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(idx[::2], idx[1::2]))


def _fit_pixels(lf, line, z=None):
    """Pixels of lf.wave that the Gaussian fit actually used (line_snr.measure_line_snr): within
    2 x win_half of the line and its tied companions, finite data and error, 1-px spikes excluded.
    The local region around it (continuum side bands, band edges with a tiny noise estimate) is
    only shown, never counted in the residual statistics."""
    w = np.asarray(lf.wave, float)
    with np.errstate(invalid='ignore', divide='ignore'):
        good = np.isfinite(lf.resid) & np.isfinite(lf.err) & (lf.err > 0) & np.isfinite(lf.model)
    try:
        good &= ~np.asarray(engine.lsnr._find_spikes(lf.resid, lf.err), bool)
    except Exception:
        pass
    rest = REST.get(line)
    if rest is None:
        return good
    try:
        comps = tuple(engine.lsnr.default_companions(rest))
    except Exception:
        comps = ()
    centre = lf.center if np.isfinite(lf.center) else (rest * (1 + z) if z else None)
    if centre is None:
        return good
    half = 2 * (0.0025 if line == 'Ha' else 0.002)       # measure_lines' default win_half, x 2
    reg = np.zeros(w.shape, bool)
    for r in (rest,) + comps:
        reg |= np.abs(w - centre * r / rest) <= half
    return good & reg


def _constant_in_bins(a, b):
    """True if map `a` has one value per bin of the integer bin map `b` (bins with >= 2 spaxels
    exist): then its values belong to the bins and a clicked spaxel stands for its whole bin."""
    if a.shape != b.shape:
        return False
    ok = (b >= 0) & np.isfinite(a)
    if not ok.any():
        return False
    ids, v = b[ok], a[ok]
    order = np.argsort(ids, kind='stable')
    ids, v = ids[order], v[order]
    starts = np.flatnonzero(np.r_[True, np.diff(ids) != 0])
    counts = np.diff(np.r_[starts, ids.size])
    if not (counts > 1).any():
        return False
    spread = np.maximum.reduceat(v, starts) - np.minimum.reduceat(v, starts)
    return bool(np.all(spread <= 1e-6 * max(1.0, float(np.abs(v).max()))))


def _fill_band(plot, x, lo, hi, color, alpha=60):
    """Shaded band between lo and hi, split at NaN gaps (FillBetweenItem dislikes NaN)."""
    ok = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi)
    brush = pg.mkBrush(*pg.mkColor(color).getRgb()[:3], alpha)
    for a, b in _finite_runs(ok):
        if b - a < 2:
            continue
        c1 = pg.PlotDataItem(x[a:b], lo[a:b], pen=None)
        c2 = pg.PlotDataItem(x[a:b], hi[a:b], pen=None)
        fill = pg.FillBetweenItem(c1, c2, brush=brush)
        fill.setZValue(-5)
        plot.addItem(fill)


def _num(v, default=np.nan):
    """float(v), `default` if v is not a number."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _fkms(v, e=np.nan, signed=False):
    """'+12 ± 5' (km/s values of the info card), '–' for NaN, no error part if e is NaN."""
    v, e = _num(v), _num(e)
    if not np.isfinite(v):
        return '–'
    s = f'{v:+.0f}' if signed else f'{v:.0f}'
    return s + (f' ± {e:.0f}' if np.isfinite(e) else '')


# ---------------------------------------------------------------------------- Voronoi-fit products
def _group_of_map(name):
    """Line group of a Voronoi-fit map ('OIII_V50' -> 'oiii', 'LOG_N2' -> 'ha'), else None."""
    n = (name or '').upper()
    if n in _RATIO_GROUP:
        return _RATIO_GROUP[n]
    if n in _SEP_MAPS:                  # Halpha / [NII] fitted separately
        return 'ha'
    for tok, g in _GROUP_TOKENS:
        if n == tok or n.startswith(tok + '_'):
            return g
    return None


def _pipe_order(names, anchor):
    """Chooser order of Voronoi-fit maps: the anchor group first, then the other groups in wavelength
    order (each: main-line flux, S/N, v50, W80, components, ... first), ratios, bins, the rest."""
    groups = [anchor] + [g for g in GROUP_KEYS if g != anchor]
    pos = {n: i for i, n in enumerate(names)}

    def key(n):
        if n in _RATIOS:
            return (len(groups), _RATIOS.index(n), 0)
        if n in ('BINMAP', 'SN_LINE'):
            return (len(groups) + 1, ('BINMAP', 'SN_LINE').index(n), 0)
        g = _group_of_map(n)
        if g not in groups:
            return (len(groups) + 2, pos[n], 0)
        G, M = _gtag(g), _ltag(GROUPS[g]['main'])
        first = ([f'{M}_FLUX', f'{M}_SNR', f'{G}_V50', f'{G}_W80']
                 + (list(_SEP_MAPS) if g == 'ha' else [])
                 + [f'{G}_C{k}_{q}' for k in (1, 2, 3) for q in ('V', 'SIG')]
                 + [f'{G}_NCOMP', f'{G}_CHI2NU', f'{M}_FLUX_ERR'] + [f'{M}_C{k}_FLUX' for k in (1, 2, 3)])
        return (groups.index(g), first.index(n) if n in first else len(first), pos[n])
    return sorted(names, key=key)


def _pipe_defaults(names, anchor):
    """Default maps of a Voronoi fit: anchor-line flux, anchor v50 (else C1 velocity), anchor W80."""
    G, M = _gtag(anchor), _ltag(GROUPS[anchor]['main'])
    out = []
    for cands in ((f'{M}_FLUX', 'SN_LINE'), (f'{G}_V50', f'{G}_C1_V'), (f'{G}_W80', f'{G}_C1_SIG')):
        n = next((c for c in cands if c in names and c not in out), None)
        if n is None:
            n = next((c for c in names if c not in out and c != 'BINMAP'), None)
        out.append(n)
    return out


def _flux_base_unit(prod):
    """Cube unit of a product ('ADU/s'); Voronoi-fit headers give the flux unit '<unit> um'
    (FITS form 'adu s-1 um' since pipeline 2.1)."""
    u = prod.bunit
    if prod.is_pipeline and u.endswith(' um'):
        u = u[:-3]
    return u.replace('/sec', '/s').replace('adu s-1', 'ADU/s')


def _config_from(d):
    """vb.Config from a dict (unknown keys ignored, lists -> tuples); None without the pipeline."""
    if vb is None:
        return None
    known = getattr(vb.Config, '__dataclass_fields__', {})
    kw = {k: tuple(v) if isinstance(v, list) else v for k, v in (d or {}).items() if k in known}
    try:
        return vb.Config(**kw)
    except (TypeError, ValueError):
        return vb.Config()


def _rows_of(df):
    """{bin: row dict} of a BINS / TOTAL table (strings stripped)."""
    rows = {}
    if df is not None and 'BIN' in df.columns:
        for r in df.to_dict('records'):
            rows[int(r['BIN'])] = {k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
    return rows


def _load_product_data(prod):
    """Worker: {'maps'} of a MapProduct; Voronoi fits also 'rows' {bin: BINS row}, 'total' (TOTAL row
    or None), 'cfg' (vb.Config), 'noise' {band: noise model}, 'rest' {line: micron}, 'clipped'
    {(bin, group tag): wavelengths} and 'header' - all read now from the file (vb.read_product), so a
    product rewritten after the catalogue was built is shown consistently.  'cfg' stays None for
    products of another schema (their columns differ): no spectra then."""
    out = {'maps': prod.load(), 'header': dict(prod.meta)}
    if not prod.is_pipeline:
        return out
    P = vb.read_product(prod.path) if vb is not None else None
    if P is not None and str(P['header'].get('SCHEMA', '')).strip() == vb.SCHEMA:
        prod._maps.clear()
        prod._maps.update(P['maps'])
        prod.meta.update({k: v for k, v in P['header'].items() if isinstance(v, (str, int, float, bool))})
        tot = _rows_of(P['total'])
        out.update(maps=prod._maps, header=P['header'], rows=_rows_of(P['bins']), total=tot.get(-1),
                   cfg=P['config'] or vb.Config(), noise=P['noise'], rest=P['rest'], clipped=P['clipped'])
    else:
        out.update(rows=_rows_of(prod.bins_table()), total=None, cfg=None, noise={}, rest=None, clipped={})
    return out


def _product_from_file(path):
    """MapProduct of one Voronoi-fit file, built as data.products.discover_products does."""
    from astropy.io import fits
    with fits.open(path) as h:
        h0 = h[0].header
        names = [e.name for e in h[1:] if e.header.get('NAXIS', 0) == 2]
        meta = {k: h0[k] for k in h0 if k and isinstance(h0[k], (str, int, float, bool))}
    return MapProduct(key=str(h0.get('OBJECT') or '').strip(), kind='Voronoi fit', path=path, names=names,
                      band=str(h0.get('ANCHBAND') or '').strip(), spxbin=int(h0.get('SPXBIN', 1) or 1),
                      meta=meta)


def _update_index(out_dir, rec, variant, cfg):
    """Refresh index.json / index.csv of the product folder (for servers; rebuilt from the product
    headers).  The document-level config stays the batch one; this run's settings are in the FITS
    header (CONFIG table).  Never fails the computation."""
    try:
        base = cfg
        p = os.path.join(out_dir, 'index.json')
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                base = _config_from(json.load(f).get('config')) or cfg
        vb.write_index(out_dir, [rec], variant, base)
    except Exception:
        pass


def _run_voronoi(stacks, z, cfg, key, path, meta, variant, progress=None):
    """Worker: vb.run_galaxy for one galaxy, write its product (+ index entry) and read it back.
    Returns (MapProduct, loaded data, index record)."""
    prod = vb.run_galaxy(stacks, z, cfg, name=key, progress=progress)
    vb.write_product(prod, path, meta=meta)
    rec = vb.record_from_header(path)
    rec['message'] = (f'EMPOWER Explorer {__version__}: max {cfg.max_components} components, '
                      f'target S/N {cfg.target_sn:g}')
    _update_index(os.path.dirname(path), rec, variant, cfg)
    mp = _product_from_file(path)
    return mp, _load_product_data(mp), rec


def _compute_bin(res, req):
    """Worker: spectrum of one Voronoi bin (or spaxel) in the band of one line group, made exactly as
    the pipeline made it (vb.region_spectrum with the product's noise model of that band: background-
    subtracted sum of the spaxels, error sigma_1 sqrt(n) c(N) x err_scale), and the fitted model
    rebuilt from the bin's BINS row (vb.model_curve) - the same pixels and errors as the fit, so the
    residuals reproduce its chi2_nu (up to clipped pixels).  No refit."""
    st, cfg, g, z = req['stack'], req['cfg'], res.group, req['z_ref']
    nz = req['noise']
    noise = vb.noise_from_product(nz)
    with np.errstate(all='ignore'):
        spec, err = vb.region_spectrum(st.cube, req['sel'], noise, cfg.edge_margin)
    wave = np.asarray(st.wave, float)
    gd = vb.GroupData(g, wave, spec, err, z, res.band, cfg)
    lines = GROUPS[g]['lines']
    if gd.ok:
        flo, fhi = float(gd.x.min()), float(gd.x.max())
    else:
        flo = (min(vb.REST[k] for k in lines) - cfg.win_pad) * (1 + z)
        fhi = (max(vb.REST[k] for k in lines) + cfg.win_pad) * (1 + z)
    pad = max(0.3 * (fhi - flo), 0.002 * (1 + z))
    lo, hi = max(flo - pad, st.wave_min), min(fhi + pad, st.wave_max)
    win = (wave >= lo) & (wave <= hi)
    res.wave, res.spec, res.err = wave[win], spec[win], err[win]
    res.lo, res.hi, res.fit_lo, res.fit_hi = lo, hi, flo, fhi
    res.covered = list(getattr(gd, 'keys', []))
    res.n_spax = int(np.count_nonzero(req['sel']))
    res.has_data = bool(np.isfinite(res.spec).any())
    row, G = res.row, _gtag(g)
    if row is not None and gd.ok and str(row.get(f'{G}_BAND', '') or '').strip() == res.band:
        kw = dict(rest=req.get('rest'), resolving_power=nz.get('resolving_power'))
        model = vb.model_curve(row, g, res.wave, z, **kw)
        if model is not None:
            mx = np.linspace(flo, fhi, max(400, 8 * gd.x.size))
            total, parts = vb.model_curve(row, g, mx, z, components=True, **kw)
            res.mx, res.mtotal, res.mparts = mx, total, parts
            res.mcont = row[f'{G}_CONT_A'] + row[f'{G}_CONT_B'] * (mx - row[f'{G}_X0'])
            with np.errstate(invalid='ignore', divide='ignore'):
                inside = (res.wave >= flo) & (res.wave <= fhi) & (res.err > 0)
                res.chi = np.where(inside, (res.spec - model) / res.err, np.nan)
            res.used = np.isin(res.wave, gd.x) & np.isfinite(res.chi)
            clipped = req.get('clipped')
            if clipped is not None and len(clipped):       # pixels the pipeline clipped before its fit
                res.used &= ~np.isin(res.wave, clipped)
    return res


def _section(text):
    lab = QLabel(text.upper())
    lab.setProperty('role', 'section')
    return lab


def _dim(text=''):
    lab = QLabel(text)
    lab.setProperty('role', 'dim')
    return lab


# ---------------------------------------------------------------------------- data containers
@dataclass
class _MapSet:
    """The maps of one source, all on the pixel grid of `band`."""
    kind: str                       # 'product:<kind>', 'cube', 'live' (remembered across galaxies)
    title: str
    band: str | None                # band whose stack grid the maps are on
    order: list                     # map names in chooser order
    labels: dict                    # name -> pretty label with units
    maps: dict = field(default_factory=dict)     # name -> 2-D float array (empty: not computed)
    tips: dict = field(default_factory=dict)     # name -> tooltip
    defaults: list = field(default_factory=list)
    grid_ok: bool = True            # maps on the stack grid -> spectra available
    note: str = ''                  # hint line
    warn: bool = False              # note is a warning
    empty_msg: str = ''             # shown in the panels when maps is empty
    line: str | None = None         # line of the whole set (live fits)
    binning: int = 1                # live fits: block size
    binmaps: dict = field(default_factory=dict)  # 'BINMAP'/'KINBINMAP' -> int array (products)
    bin_of: dict = field(default_factory=dict)   # map name -> binmap name its values belong to
    pipe: dict | None = None        # Voronoi fit: anchor, aband, rows, groups, z_ref, cfg, band_binmaps, ...
    empty_msgs: dict = field(default_factory=dict)   # map name -> why it has no value at all

    @property
    def shape(self):
        for a in self.maps.values():
            return a.shape
        return None

    def style(self, name):
        """'velocity', 'integer', 'flux' or 'other' -> colour map choice."""
        if self.kind == 'live':
            return {'dv': 'velocity', 'detected': 'integer', 'flux': 'flux',
                    'flux_err': 'flux'}.get(name, 'other')
        if is_integer_map(name):
            return 'integer'
        if is_velocity_map(name):
            return 'velocity'
        n = name.upper()
        if 'FLUX' in n or n in ('WHITE', 'CONT_HA', 'CONT_H') or n.startswith('LINE_'):
            return 'flux'
        return 'other'

    def snr_name(self, name):
        """Name of the S/N map used to mask `name` (its own line's S/N if there is one)."""
        if self.kind == 'live':
            return 'snr' if 'snr' in self.maps else None
        if self.kind == 'cube':
            return None
        if self.pipe is not None:       # '<LINE>_SNR' of the map's line, else of the anchor line
            if name in _SEP_SNR:        # separate Halpha / [NII] fit: its own S/N
                return _SEP_SNR[name] if _SEP_SNR[name] in self.maps else None
            line = _line_of_map(name)
            cand = f'{line.upper()}_SNR' if line else None
            if cand in self.maps:
                return cand
            return self.pipe['anchor_snr'] if self.pipe['anchor_snr'] in self.maps else None
        n = name.upper()
        prefix = next((p for p in ('SPAX_', 'BIN_') if n.startswith(p)), '')
        line = _line_of_map(name)
        if line:
            tok = 'HA' if line == 'Ha' else ('HB' if line == 'Hb' else line)
            cand = f'{prefix}{tok}_SNR'
            if cand in self.maps:
                return cand
        return next((c for c in ('HA_SNR', 'BIN_HA_SNR', 'SPAX_HA_SNR') if c in self.maps), None)

    def binmap_for(self, name):
        """(tag, integer bin map) whose bins the spectrum of `name` is summed over, or (None, None)
        for a single-spaxel spectrum.  Products: the bin map within whose bins the map is
        constant (BIN_*, Voronoi maps, KIN_* -> KINBINMAP; SPAX_*, CONT_H stay per spaxel)."""
        if self.kind == 'live' and self.binning > 1 and self.shape is not None:
            ny, nx = self.shape
            nbx = -(-nx // self.binning)
            yy, xx = np.indices((ny, nx))
            return f'block{self.binning}', (yy // self.binning) * nbx + xx // self.binning
        bn = self.bin_of.get(name) if self.kind.startswith('product') else None
        if bn is None or bn not in self.binmaps:
            return None, None
        return bn, self.binmaps[bn]


@dataclass
class _PixelResult:
    band: str | None
    x: int
    y: int
    line: str | None
    spec: object = None
    fits: dict = field(default_factory=dict)
    info: object = None
    what: str = ''                  # 'spaxel (9, 9)', 'Voronoi bin 21 · 3 spaxels', ...
    members: np.ndarray | None = None   # (ny, nx) bool on the map grid (bins / blocks)
    note: str = ''


@dataclass
class _BinResult:
    """Spectrum of one Voronoi bin in the band of one line group + the pipeline's model of it."""
    band: str | None
    x: int
    y: int
    group: str
    bin_id: int                     # -1: spaxel outside the bins (spectrum without fit)
    row: dict | None = None         # the bin's BINS row
    what: str = ''
    note: str = ''
    message: tuple | None = None    # (title, hint) when there is nothing to extract
    members: np.ndarray | None = None
    noise_key: tuple | None = None
    noise: tuple | None = None      # unused for products (their noise model is stored in the file)
    wave: np.ndarray | None = None  # display window
    spec: np.ndarray | None = None
    err: np.ndarray | None = None
    lo: float = np.nan
    hi: float = np.nan
    fit_lo: float = np.nan
    fit_hi: float = np.nan
    covered: list = field(default_factory=list)
    n_spax: int = 0
    has_data: bool = False
    mx: np.ndarray | None = None    # fine grid over the fit window
    mtotal: np.ndarray | None = None
    mparts: list = field(default_factory=list)   # continuum + component k
    mcont: np.ndarray | None = None
    chi: np.ndarray | None = None   # (data - model)/sigma on `wave` (NaN outside the fit window)
    used: np.ndarray | None = None  # pixels the fit used


# ---------------------------------------------------------------------------- one map panel
class _MapPanel(QWidget):
    """Map chooser above an ImagePanel; the value under the mouse is shown in the top-left
    corner of the image (x, y in spaxels, 0-based) and cleared when the mouse leaves."""
    mapChosen = pyqtSignal(int)
    mouseLeft = pyqtSignal()

    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.combo = QComboBox()
        self.combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setMinimumContentsLength(14)
        self.combo.setToolTip('Map shown in this panel')
        self.combo.currentIndexChanged.connect(lambda _: self.mapChosen.emit(self.index))
        self.image = ImagePanel()
        self.image.setMinimumSize(200, 200)
        self.image.setToolTip('Hover: value under the mouse · click: spectrum of that spaxel')
        vb = self.image.plot.getViewBox()
        self.readout = pg.TextItem('', color=C.TEXT, anchor=(0, 0),
                                   fill=pg.mkBrush(*pg.mkColor(C.BG_DARK).getRgb()[:3], 215))
        self.readout.setFont(QFont('Segoe UI', 9))
        self.readout.setParentItem(vb)          # view-box pixel coordinates: fixed corner
        self.readout.setPos(4, 4)
        self.readout.setZValue(1000)
        self.readout.hide()
        self.image.installEventFilter(self)
        lay.addWidget(self.combo)
        lay.addWidget(self.image, 1)

    def eventFilter(self, obj, ev):
        if obj is self.image and ev.type() == QEvent.Leave:
            self.mouseLeft.emit()
        return False

    def set_readout(self, text):
        self.readout.setText(text)
        self.readout.setVisible(bool(text))

    def name(self):
        return self.combo.currentData()

    def populate(self, ms, current):
        self.combo.blockSignals(True)
        self.combo.clear()
        if ms is not None:
            for n in ms.order:
                self.combo.addItem(ms.labels.get(n, n), n)
                self.combo.setItemData(self.combo.count() - 1, ms.tips.get(n, n), Qt.ToolTipRole)
        i = self.combo.findData(current) if current is not None else -1
        self.combo.setCurrentIndex(i)
        self.combo.setEnabled(self.combo.count() > 0)
        self.combo.blockSignals(False)


# ---------------------------------------------------------------------------- the tab
class MapsTab(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._dirty = True
        self._mapset = None
        self._shown = [None] * N_PANELS        # displayed (masked) arrays
        self._shape_drawn = [None] * N_PANELS
        self._source_pref = None               # map-set kind chosen by the user
        self._chosen = {}                      # kind -> [map name per panel] chosen by the user
        self._band_pref = None
        self._line_pref = None
        self._cube_cache = {}
        self._live_cache = {}
        self._jobs = {}                        # live cache key -> (Job, t0)
        self._auto_done = set()
        self._fit_cache = OrderedDict()
        self._sel = None                       # (x, y) on the grid of the current map set
        self._focus = 0                        # panel clicked last (sets the spectrum line)
        self._result = None
        self._pending = None                   # cache key of the pixel fit the plot waits for
        self._inflight = set()                 # cache keys of pixel fits running in the background
        self._line_labels = []                 # [(λ, InfiniteLine)] markers in the spectrum plot
        self._grid = None                      # (band, shape) of the last map set with maps
        self._prod_data = OrderedDict()        # product token -> data read by _load_product_data
        self._prod_jobs = {}                   # product token -> Job reading it
        self._noise_cache = OrderedDict()      # (product token, band, centre) -> (sigma_1, c(N))
        self._vjobs = {}                       # (variant, galaxy) -> running Voronoi fit (job, progress)
        self._spec_mode = 'lines'              # spectrum chooser lists 'lines' or line 'groups'
        self._spec_choice = {'lines': None, 'groups': None}
        self._synced_root = None               # engine root whose rest wavelengths vb uses
        self._build_ui()
        self._spec_timer = QTimer(self)
        self._spec_timer.setSingleShot(True)
        self._spec_timer.setInterval(0)
        self._spec_timer.timeout.connect(self._update_spectrum)
        # the velocity colour map comes from matplotlib: importing it takes ~0.3-0.5 s, so do that in
        # the background now instead of in the GUI thread when the first velocity map is drawn
        run_async(colormap, theme.CMAP_VELOCITY)

        state.galaxyChanged.connect(self._on_galaxy)
        state.sepFitsChanged.connect(self._on_sep_fits)
        state.loadFailed.connect(lambda k: self._show_empty(
            f'The cubes of {k} could not be loaded',
            'See the error message; select the galaxy again in the list to retry.'))
        state.stacksReady.connect(self._on_data_changed)
        state.redshiftChanged.connect(self._on_data_changed)
        state.catalogChanged.connect(self._on_catalog)
        self._show_empty('No galaxy selected', 'Pick a galaxy in the list on the left.')

    def heightForWidth(self, w):
        # word-wrapped labels make Qt reserve their one-word-per-line height: use the real minimum
        return self.minimumSizeHint().height()

    # ================================================================== UI construction
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 6, 4, 4)
        outer.setSpacing(6)
        self.pages = QStackedWidget()
        self.empty = EmptyState()
        self.pages.addWidget(self.empty)
        content = QWidget()
        self.pages.addWidget(content)
        outer.addWidget(self.pages)

        lay = QVBoxLayout(content)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addLayout(self._build_controls())
        self.hint = _dim('')
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)

        split = QSplitter(Qt.Horizontal)
        split.setHandleWidth(6)
        left = QSplitter(Qt.Vertical)
        left.setHandleWidth(6)
        left.addWidget(self._build_maps())
        left.addWidget(self._build_spectrum())
        left.setStretchFactor(0, 5)
        left.setStretchFactor(1, 4)
        left.setSizes([390, 320])
        split.addWidget(left)
        split.addWidget(self._build_side())
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([1000, 290])
        lay.addWidget(split, 1)

    def _build_controls(self):
        row = QHBoxLayout()
        row.setSpacing(8)
        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(250)
        self.source_combo.setToolTip('Which maps to show: products of the notebooks, instant maps '
                                     'from the cube, or a live per-spaxel line fit')
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.band_combo = QComboBox()
        self.band_combo.setToolTip('Band (stacked cube) whose pixel grid the maps are on')
        self.band_combo.currentIndexChanged.connect(self._band_changed)
        self.line_combo = QComboBox()
        self.line_combo.setToolTip('Line fitted in every spaxel')
        self.line_combo.currentIndexChanged.connect(self._live_option_changed)
        self.bin_combo = QComboBox()
        for b, lab in BINNINGS:
            self.bin_combo.addItem(lab, b)
        self.bin_combo.setToolTip('Fit blocks of b×b spaxels (higher S/N, coarser map)')
        self.bin_combo.currentIndexChanged.connect(self._live_option_changed)
        self.compute_btn = QPushButton('Compute')
        self.compute_btn.clicked.connect(self._compute)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(110)
        self.progress.setFixedHeight(8)
        self.progress.hide()
        self.auto_chk = QCheckBox('Auto')
        self.auto_chk.setToolTip('Compute the live fit automatically when the galaxy, band, line or '
                                 'binning changes (runs in the background, ~3 s)')
        self.auto_chk.toggled.connect(lambda on: on and self._maybe_auto_compute())
        self.snr_spin = QDoubleSpinBox()
        self.snr_spin.setRange(0, 100)
        self.snr_spin.setSingleStep(0.5)
        self.snr_spin.setDecimals(1)
        self.snr_spin.setSpecialValueText('off')
        self.snr_spin.setFixedWidth(84)
        self.snr_spin.setToolTip('Hide spaxels whose line S/N is below this value (uses the S/N map '
                                 'of the map\'s own line where one exists, else the Hα S/N map)')
        self.snr_spin.valueChanged.connect(lambda _: self._redraw_maps(keep_view=True))

        # Voronoi fit (empower_explorer/pipeline/voronoi_binning.py) of the current galaxy
        cfg0 = vb.Config() if vb is not None else None
        self.vcomp_label = _dim('Max comp.')
        self.vcomp_spin = QSpinBox()
        self.vcomp_spin.setRange(1, 3)
        self.vcomp_spin.setValue(cfg0.max_components if cfg0 else 3)
        self.vcomp_spin.setToolTip('Maximum number of Gaussian components per line group; per bin, one more '
                                   'component is kept only if BIC drops by ≥ 10 and it has S/N ≥ 3')
        self.vsn_label = _dim('Target S/N')
        self.vsn_spin = QDoubleSpinBox()
        self.vsn_spin.setRange(3, 100)
        self.vsn_spin.setSingleStep(1)
        self.vsn_spin.setDecimals(0)
        self.vsn_spin.setValue(cfg0.target_sn if cfg0 else 15)
        self.vsn_spin.setFixedWidth(64)
        self.vsn_spin.setToolTip('Anchor-line S/N per Voronoi bin (lower targets 10, 8, 6 are used if fewer '
                                 'than 25 bins result)')
        self.vfit_btn = QPushButton('Compute Voronoi fit')
        self.vfit_btn.clicked.connect(self._compute_voronoi)
        self.vprogress = QProgressBar()
        self.vprogress.setFixedWidth(110)
        self.vprogress.setFixedHeight(8)
        self.vprogress.setTextVisible(False)
        self.vprogress.hide()
        self.sep_btn = QPushButton('Hα / [NII] separately')
        self.sep_btn.clicked.connect(self._compute_separate)

        self.band_label = _dim('Band')
        self.line_label = _dim('Line')
        self.bin_label = _dim('Binning')
        self.snr_label = _dim('Mask S/N ≥')
        for w in (_dim('Source'), self.source_combo, self.band_label, self.band_combo, self.line_label,
                  self.line_combo, self.bin_label, self.bin_combo, self.compute_btn, self.progress,
                  self.auto_chk, self.vcomp_label, self.vcomp_spin, self.vsn_label, self.vsn_spin,
                  self.vfit_btn, self.vprogress, self.sep_btn):
            row.addWidget(w)
        row.addStretch(1)
        row.addWidget(self.snr_label)
        row.addWidget(self.snr_spin)
        return row

    def _build_maps(self):
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self.panels = []
        for i in range(N_PANELS):
            p = _MapPanel(i)
            p.mapChosen.connect(self._map_chosen)
            p.image.pixelClicked.connect(lambda x, y, i=i: self._clicked(i, x, y))
            p.image.pixelHovered.connect(lambda x, y, v: self._hovered(x, y))
            p.mouseLeft.connect(self._clear_readouts)
            row.addWidget(p, 1)
            self.panels.append(p)
        return w

    def _build_spectrum(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(4)
        head = QHBoxLayout()
        head.addWidget(_section('Spectrum'))
        self.spec_title = QLabel('')
        self.spec_title.setStyleSheet(f'color: {C.TEXT}; font-weight: bold;')
        head.addWidget(self.spec_title, 1)
        self.spec_line_label = _dim('Line')
        head.addWidget(self.spec_line_label)
        self.spec_line_combo = QComboBox()
        self.spec_line_combo.setToolTip('Line whose fit is shown; "auto" follows the map you clicked')
        self.spec_line_combo.currentIndexChanged.connect(self._spec_choice_changed)
        head.addWidget(self.spec_line_combo)
        lay.addLayout(head)

        self.spec_pages = QStackedWidget()
        self.spec_empty = EmptyState('Click a spaxel in any map')
        self.spec_pages.addWidget(self.spec_empty)
        body = QWidget()
        bl = QHBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(8)
        self.spec_glw = pg.GraphicsLayoutWidget()
        self.spec_glw.setBackground(C.BG_MEDIUM)
        self.spec_glw.ci.setContentsMargins(4, 4, 8, 4)
        self.spec_glw.ci.setSpacing(2)
        self.spec_plot = self.spec_glw.addPlot(row=0, col=0)
        self.resid_plot = self.spec_glw.addPlot(row=1, col=0)
        from empower_explorer.widgets.plots import style_plot_item
        style_plot_item(self.spec_plot, '', '', 'Flux')
        style_plot_item(self.resid_plot, '', 'Observed wavelength [µm]', 'resid [σ]')
        self.resid_plot.setXLink(self.spec_plot)
        self.spec_plot.getAxis('bottom').setStyle(showValues=False)
        self.spec_glw.ci.layout.setRowStretchFactor(0, 3)
        self.spec_glw.ci.layout.setRowStretchFactor(1, 1)
        for p in (self.spec_plot, self.resid_plot):
            p.getAxis('left').setWidth(58)
            p.hideButtons()
        vb = self.spec_plot.getViewBox()
        vb.sigResized.connect(self._place_line_labels)
        vb.sigXRangeChanged.connect(self._place_line_labels)
        bl.addWidget(self.spec_glw, 1)

        card = QFrame()
        card.setProperty('role', 'card')
        card.setFixedWidth(262)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(12, 10, 6, 10)
        self.info = QLabel('')
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.RichText)
        self.info.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.info.setContentsMargins(0, 0, 6, 0)
        info_scroll = QScrollArea()             # long results (3 components + map values) scroll
        info_scroll.setWidget(self.info)
        info_scroll.setWidgetResizable(True)
        info_scroll.setFrameShape(QFrame.NoFrame)
        info_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        info_scroll.setStyleSheet('QScrollArea { background: transparent; }')
        info_scroll.viewport().setAutoFillBackground(False)
        self.info.setAutoFillBackground(False)
        cl.addWidget(info_scroll, 1)
        bl.addWidget(card)
        self.spec_pages.addWidget(body)
        lay.addWidget(self.spec_pages, 1)
        return w

    def _build_side(self):
        w = QWidget()
        w.setMinimumWidth(230)
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        head = QHBoxLayout()
        head.addWidget(_section('Distribution · map 1'), 1)
        self.log_chk = QCheckBox('log')
        self.log_chk.setToolTip('Logarithmic count axis')
        self.log_chk.toggled.connect(lambda _: self._update_histogram())
        head.addWidget(self.log_chk)
        lay.addLayout(head)
        self.hist_plot = make_plot('', '', 'Spaxels')
        self.hist_plot.getPlotItem().hideButtons()
        self.hist_plot.setMouseEnabled(False, False)
        lay.addWidget(self.hist_plot, 1)
        self.hist_label = _dim('')
        self.hist_label.setWordWrap(True)
        lay.addWidget(self.hist_label)
        lay.addSpacing(6)
        lay.addWidget(_section('Fit residuals'))
        self.rhist_plot = make_plot('', '(data−model)/σ', 'Pixels')
        self.rhist_plot.getPlotItem().hideButtons()
        self.rhist_plot.setMouseEnabled(False, False)
        lay.addWidget(self.rhist_plot, 1)
        self.rhist_label = _dim('')
        self.rhist_label.setWordWrap(True)
        lay.addWidget(self.rhist_label)
        return w

    # ================================================================== state signals
    def _on_galaxy(self, key):
        self._sel = None
        self._grid = None
        self._result = None
        self._pending = None
        self._show_empty(f'Loading {key} …', 'Reading and stacking the cubes of this galaxy.')

    def _on_data_changed(self, *_):
        if self.isVisible():
            self._rebuild()
        else:
            self._dirty = True

    def _on_catalog(self):
        self._cube_cache.clear()
        self._live_cache.clear()
        self._fit_cache.clear()
        self._prod_data.clear()
        self._noise_cache.clear()
        self._dirty = True

    def showEvent(self, ev):
        super().showEvent(ev)
        if self._dirty:
            QTimer.singleShot(0, self._rebuild)

    def _show_empty(self, text, hint=''):
        self.empty.set_text(text, hint)
        self.pages.setCurrentIndex(0)

    # ================================================================== rebuild / sources
    def _rebuild(self):
        """New galaxy or redshift: list the sources and show the preferred one."""
        self._dirty = False
        st = self.state
        if vb is not None and engine.REST_UM and self._synced_root != engine.ROOT:
            vb.sync_rest(engine.REST_UM, engine.BAND_RESOLUTION)     # the notebooks' line table
            self._synced_root = engine.ROOT
        if not st.stacks:
            if st.key and not st.is_loading:
                self._show_empty(f'The cubes of {st.key} could not be loaded',
                                 'See the error message; select the galaxy again in the list to retry.')
            elif st.key:
                self._show_empty(f'Loading {st.key} …')
            else:
                self._show_empty('No galaxy selected', 'Pick a galaxy in the list on the left.')
            return
        self.pages.setCurrentIndex(1)
        self._populate_sources()
        self._apply_source()

    def _populate_sources(self):
        """Voronoi fit first (or a 'not computed yet' entry with the Compute button), then the
        notebook products, the cube maps and the live fit."""
        prods = self.state.products()
        cb = self.source_combo
        cb.blockSignals(True)
        cb.clear()
        pipe = [i for i, p in enumerate(prods) if p.is_pipeline]
        if not pipe:
            ok, why = self._vfit_possible()
            cb.addItem('Voronoi fit  ·  not computed yet', 'vnew')
            cb.setItemData(cb.count() - 1, 'No Voronoi-pipeline product for this galaxy yet: choose '
                           'this and press "Compute Voronoi fit"' if ok else f'Voronoi fit: {why}', Qt.ToolTipRole)
            cb.model().item(cb.count() - 1).setEnabled(ok)
        for i in pipe:
            label, tip = self._pipe_label(prods[i])
            cb.addItem(label, f'product:{i}')
            cb.setItemData(cb.count() - 1, tip, Qt.ToolTipRole)
        for i, p in enumerate(prods):
            if p.is_pipeline:
                continue
            band = p.band or f'{self._halpha_band()}, Hα band'
            cb.addItem(f'{p.kind} (notebook)  ·  {band}', f'product:{i}')
            cb.setItemData(cb.count() - 1, f'Notebook product {os.path.basename(p.path)}\n'
                           f'{len(p.names)} maps', Qt.ToolTipRole)
        cb.addItem('Cube: line maps', 'cube')
        cb.setItemData(cb.count() - 1, 'White light, continuum S/N and continuum-subtracted line maps '
                       'computed from the stacked cube (instant)', Qt.ToolTipRole)
        cb.addItem('Live fit: per spaxel', 'live')
        live_ok = self.state.z is not None
        cb.setItemData(cb.count() - 1, 'Fit one line in every spaxel (flux, S/N, Δv, σ_int, W80)'
                       if live_ok else 'Needs a redshift', Qt.ToolTipRole)
        cb.model().item(cb.count() - 1).setEnabled(live_ok)
        # restore the user's choice of source type, else the Voronoi fit, else the first notebook
        # product, else the cube maps
        kinds = [self._kind_of(cb.itemData(i)) for i in range(cb.count())]
        enabled = [cb.model().item(i).isEnabled() for i in range(cb.count())]
        want = None
        if self._source_pref in kinds and enabled[kinds.index(self._source_pref)]:
            want = kinds.index(self._source_pref)
        elif pipe:
            want = cb.findData(f'product:{pipe[0]}')
        elif prods:
            want = next(j for j in range(cb.count()) if str(cb.itemData(j)).startswith('product:'))
        else:
            want = cb.findData('cube')
        cb.setCurrentIndex(want)
        cb.blockSignals(False)

    @staticmethod
    def _pipe_label(p):
        """('Voronoi fit · 38 bins · Hα (IZ) · max 3 comp.', tooltip) of a pipeline product."""
        m = p.meta
        n = int(_num(m.get('NBINS'), 0))
        anchor = str(m.get('ANCHOR') or '').strip()
        alab = LINE_LABELS.get(GROUPS.get(anchor, {}).get('main'), anchor or '?')
        ts = _num(m.get('TARGETSN'))
        label = (f'Voronoi fit  ·  {n} bin{"s" if n != 1 else ""}  ·  {alab} ({m.get("ANCHBAND", "?")})  ·  '
                 f'max {m.get("MAXCOMP", "?")} comp.')
        tip = (f'Voronoi-pipeline product {os.path.basename(p.path)}\n'
               f'{m.get("SCHEMA", "")}, pipeline {m.get("PIPEVER", "?")}, {m.get("DATE", "")}\n'
               f'bins: {m.get("BINMODE", "?")}' + (f', target S/N {ts:g}' if ts > 0 else '')
               + f' on {alab}; z_ref = {_num(m.get("Z_REF")):.5f}\n'
               f'bands {m.get("BANDS", "?")}; up to {m.get("MAXCOMP", "?")} Gaussian components per line group')
        if str(m.get('STATUS', '')).strip() == 'no_detection':
            label += '  ·  not detected'
            tip += '\nanchor line not detected: one central aperture, fluxes forced to the reference kinematics'
        return label, tip

    def _kind_of(self, sid):
        if sid == 'vnew':
            return 'product:Voronoi fit'
        if sid and sid.startswith('product:'):
            prods = self.state.products()
            i = int(sid.split(':')[1])
            return f'product:{prods[i].kind}' if i < len(prods) else 'product'
        return sid

    def _is_pipe_sid(self, sid):
        """True for a Voronoi-fit source (a product or the 'not computed yet' entry)."""
        if sid == 'vnew':
            return True
        if sid and sid.startswith('product:'):
            prods = self.state.products()
            i = int(sid.split(':')[1])
            return i < len(prods) and prods[i].is_pipeline
        return False

    def _current_product(self):
        sid = self.source_combo.currentData()
        if sid and sid.startswith('product:'):
            prods = self.state.products()
            i = int(sid.split(':')[1])
            return prods[i] if i < len(prods) else None
        return None

    def _populate_spec_lines(self, ms):
        """Spectrum chooser: the line groups of a Voronoi fit, else the single lines ('auto' first)."""
        mode = 'groups' if (ms is not None and ms.pipe is not None) else 'lines'
        cb = self.spec_line_combo
        cb.blockSignals(True)
        cb.clear()
        cb.addItem('auto', None)
        if mode == 'groups':
            for g in GROUP_KEYS:
                if g in ms.pipe['groups']:
                    cb.addItem(GROUPS[g]['label'], f'grp:{g}')
            enabled = cb.count() > 1
            self.spec_line_label.setText('Lines')
            cb.setToolTip('Line group whose spectrum and fitted model are shown; "auto": the group of the '
                          f'map you clicked, else the anchor group ({GROUPS[ms.pipe["anchor"]]["label"]})')
        else:
            z = self.state.z
            if z:
                for k in FIT_LINES:
                    if analysis.pick_band(self.state.stacks, REST[k] * (1 + z)) is not None:
                        cb.addItem(LINE_LABELS[k], k)
            enabled = bool(z)
            self.spec_line_label.setText('Line')
            cb.setToolTip('Line whose fit is shown; "auto" follows the map you clicked')
        i = cb.findData(self._spec_choice.get(mode)) if self._spec_choice.get(mode) else 0
        cb.setCurrentIndex(max(i, 0))
        cb.setEnabled(enabled)
        cb.blockSignals(False)
        self._spec_mode = mode

    def _spec_choice_changed(self, _):
        self._spec_choice[self._spec_mode] = self.spec_line_combo.currentData()
        self._schedule_spectrum()

    def _source_changed(self, _):
        self._source_pref = self._kind_of(self.source_combo.currentData())
        self._apply_source()

    def _band_changed(self, _):
        b = self.band_combo.currentData()
        if b and self.band_combo.isEnabled():
            self._band_pref = b
        self._apply_source(controls=False)

    def _live_option_changed(self, _):
        k = self.line_combo.currentData()
        if k:
            self._line_pref = k
        self._apply_source(controls=False)

    def _apply_source(self, controls=True):
        """(Re)build the map set of the chosen source and show it."""
        sid = self.source_combo.currentData()
        if sid is None:
            return
        if controls:
            self._update_controls(sid)
        if sid == 'vnew':
            ms = self._vnew_set()
        elif sid.startswith('product:'):
            prods = self.state.products()
            i = int(sid.split(':')[1])
            ms = self._product_set(prods[i]) if i < len(prods) else None
        elif sid == 'cube':
            ms = self._cube_set(self.band_combo.currentData())
        else:
            ms = self._live_set(self.band_combo.currentData(), self.line_combo.currentData(),
                                self.bin_combo.currentData())
        self._update_live_buttons()
        self._update_vfit_controls()
        self._set_mapset(ms)
        if sid == 'live':
            self._maybe_auto_compute()

    def _update_controls(self, sid):
        """Show band/line/binning controls that apply to the source and fill them."""
        live, cube = sid == 'live', sid == 'cube'
        pipe = self._is_pipe_sid(sid)
        stacks, z = self.state.stacks, self.state.z
        for w in (self.line_label, self.line_combo, self.bin_label, self.bin_combo, self.compute_btn,
                  self.auto_chk):
            w.setVisible(live)
        for w in (self.vcomp_label, self.vcomp_spin, self.vsn_label, self.vsn_spin, self.vfit_btn, self.sep_btn):
            w.setVisible(pipe)
        self.progress.setVisible(live and self._current_live_key() in self._jobs)
        bc = self.band_combo
        bc.blockSignals(True)
        bc.clear()
        if sid.startswith('product:') or sid == 'vnew':
            ms_band = (self._predicted_anchor()[1] if sid == 'vnew'
                       else self._product_band(self.state.products()[int(sid.split(':')[1])]))
            bc.addItem(ms_band or '?', ms_band)
            bc.setEnabled(False)
            bc.setToolTip('Voronoi fits are on the grid of the band of their anchor line (Hα, else [OIII], '
                          'else Hβ); the bins are carried to the other bands' if pipe else
                          'Notebook products are on the grid of the stack of their band')
        else:
            bands = list(stacks)
            if live:
                bands = [b for b in bands if z and analysis.covered_lines(stacks[b], z, LIVE_LINES)]
            for b in bands:
                bc.addItem(stacks[b].describe(), b)
            want = self._band_pref if self._band_pref in bands else (
                self._halpha_band() if self._halpha_band() in bands else (bands[0] if bands else None))
            bc.setCurrentIndex(max(bc.findData(want), 0))
            bc.setEnabled(len(bands) > 1)
            bc.setToolTip('Band (stacked cube) whose pixel grid the maps are on')
        bc.blockSignals(False)
        if live:
            self._fill_live_lines()

    def _fill_live_lines(self):
        st = self.state.stacks.get(self.band_combo.currentData())
        z = self.state.z
        keys = analysis.covered_lines(st, z, LIVE_LINES) if (st is not None and z) else []
        lc = self.line_combo
        lc.blockSignals(True)
        lc.clear()
        for k in keys:
            lc.addItem(LINE_LABELS[k], k)
        want = self._line_pref if self._line_pref in keys else (keys[0] if keys else None)
        lc.setCurrentIndex(max(lc.findData(want), 0))
        lc.blockSignals(False)

    def _halpha_band(self):
        """Band with Hα (deepest that covers it) - the grid of products without BAND."""
        z = self.state.z
        if z:
            b = analysis.pick_band(self.state.stacks, REST['Ha'] * (1 + z))
            if b:
                return b
        return self.state.ref_band

    def _product_band(self, prod):
        return prod.band or self._halpha_band()

    # ================================================================== map sets
    @staticmethod
    def _prod_token(prod):
        """Identity of a product file version (a recomputed file gets a new DATE)."""
        return os.path.normcase(os.path.abspath(prod.path)), str(prod.meta.get('DATE', ''))

    def _request_product(self, prod, token):
        """Read a product in a worker; the map set is rebuilt when it arrives (if still shown)."""
        if token in self._prod_jobs:
            return

        def finish(data, token=token):
            self._prod_jobs.pop(token, None)
            self._prod_data[token] = data
            while len(self._prod_data) > 24:
                self._prod_data.popitem(last=False)
            cur = self._current_product()
            if cur is not None and self._prod_token(cur) == token and self.pages.currentIndex() == 1:
                self._apply_source(controls=False)

        self._prod_jobs[token] = run_async(_load_product_data, prod, on_done=finish,
                                           on_error=lambda msg, token=token: finish(
                                               {'error': msg.splitlines()[0]}, token))

    def _product_set(self, prod):
        base = os.path.basename(prod.path)
        token = self._prod_token(prod)
        data = self._prod_data.get(token)
        if data is None:            # first use: read in the background (~0.2 s), shown when ready
            self._request_product(prod, token)
            return _MapSet(kind=f'product:{prod.kind}', title=prod.kind, band=self._product_band(prod),
                           order=[], labels={}, empty_msg='reading …', note=f'Reading {base} …')
        self._prod_data.move_to_end(token)
        if 'error' in data:         # unreadable file -> message instead of a traceback
            return _MapSet(kind=f'product:{prod.kind}', title=prod.kind, band=None, order=[], labels={},
                           empty_msg='could not read the file', note=f'Could not read {base}: {data["error"]}',
                           warn=True)
        if prod.is_pipeline:
            return self._pipeline_set(prod, data)
        raw = data['maps']
        maps, binmaps = {}, {}
        for n, a in raw.items():
            a = np.asarray(a)
            if n in ('BINMAP', 'KINBINMAP'):
                binmaps[n] = a.astype(np.int64)
                a = np.where(a < 0, np.nan, a.astype(float))      # -1 = spaxel in no bin
            maps[n] = np.asarray(a, dtype=float)
        band = self._product_band(prod)
        ms = _MapSet(kind=f'product:{prod.kind}', title=prod.kind, band=band, order=list(maps),
                     labels={n: pretty_map_name(n, prod.bunit) for n in maps}, maps=maps, binmaps=binmaps,
                     tips={n: f'{n}  ({base})' for n in maps})
        ms.defaults = self._product_defaults(list(maps))
        ms.bin_of = {bn: bn for bn in binmaps}
        for n, a in maps.items():
            if n in binmaps:
                continue
            prefs = ('KINBINMAP', 'BINMAP') if n.upper().startswith('KIN_') else ('BINMAP', 'KINBINMAP')
            cand = next((c for c in prefs if c in binmaps and _constant_in_bins(a, binmaps[c])), None)
            if cand:
                ms.bin_of[n] = cand
        for n, bn in ms.bin_of.items():
            what = 'kinematic bin' if bn == 'KINBINMAP' else 'bin'
            ms.tips[n] += f'\none value per {what} ({bn}): a click shows the spectrum of the whole {what}'
        st = self.state.stacks.get(band)
        ny, nx = ms.shape if ms.shape else (0, 0)
        where = f'{band} grid' + ('' if prod.band else ' (band with Hα; the file has no BAND)')
        if st is None:
            ms.grid_ok, ms.warn = False, True
            ms.note = f'{prod.kind} · {base} · spectra unavailable: band {band} is not loaded in this reduction'
        elif st.cube.shape[1:] != (ny, nx):
            ms.grid_ok, ms.warn = False, True
            sy, sx = st.cube.shape[1:]
            ms.note = (f'{prod.kind} · {base} · spectra unavailable (grid differs: maps {nx}×{ny}, '
                       f'{band} stack {sx}×{sy} spaxels)')
        else:
            ms.note = (f'{prod.kind} of the notebooks ({base}) · {where}, {nx}×{ny} spaxels of '
                       f'{st.pixscale:.2f}″ · click a spaxel for its spectrum')
        return ms

    def _pipeline_set(self, prod, data):
        """Map set of a Voronoi-pipeline product: maps on the anchor-band grid; the bins on
        the grids of the other bands (BINMAP_<band>, not listed), the BINS rows and the fit config
        go to ms.pipe."""
        meta, base = data.get('header') or prod.meta, os.path.basename(prod.path)
        anchor = str(meta.get('ANCHOR') or 'ha').strip()
        anchor = anchor if anchor in GROUPS else 'ha'
        aband = self._product_band(prod)
        maps, binmaps, band_bms = {}, {}, {}
        for n, a in data['maps'].items():
            a = np.asarray(a)
            if n.startswith('BINMAP_'):
                band_bms[n[len('BINMAP_'):]] = a.astype(np.int64)
                continue
            if n.startswith('RELNOISE_'):                     # part of the noise model, not a map
                continue
            if n == 'BINMAP':
                binmaps[n] = a.astype(np.int64)
                a = np.where(a < 0, np.nan, a.astype(float))      # -1 = spaxel in no bin
            maps[n] = np.asarray(a, dtype=float)
        unit = _flux_base_unit(prod)
        rows_all = data.get('rows') or {}
        bm = binmaps.get('BINMAP')
        alab = GROUPS[anchor]['label']
        extra_tips, empty = {}, {}
        for g in GROUP_KEYS:          # every fitted group lists its v50 and W80; forced bins stay blank
            G, lab = _gtag(g), GROUPS[g]['label']
            states = [_num(r.get(f'{G}_FORCED')) for r in rows_all.values()]
            fitted = [x for x in states if x in (0, 1)]
            if not fitted or bm is None:
                continue
            n_forced = sum(1 for x in fitted if x == 1)
            undetected = str(meta.get('STATUS') or '').strip() == 'no_detection'
            for q in ('V50', 'W80'):
                name = f'{G}_{q}'
                if name not in maps:
                    maps[name] = _paint_bins(bm, {b: _num(r.get(name)) for b, r in rows_all.items()})
                if g == anchor:            # FORCED for the anchor: the anchor line itself was not detected there
                    if n_forced:
                        extra_tips[name] = (f'\n{lab} (the anchor) not detected in {n_forced} of {len(fitted)} bins '
                                            '(velocity forced to the template of the detected bins / the centre) - '
                                            'blank there')
                    if n_forced == len(fitted):
                        empty[name] = ('anchor line not detected in this galaxy:\nno velocity' if undetected else
                                       f'{lab} (the anchor) not detected in any bin:\nno {q.lower()} of its own')
                    continue
                if n_forced:
                    extra_tips[name] = (f'\n{n_forced} of {len(fitted)} bins: {lab} forced to the {alab} kinematics '
                                        '(no velocity of its own) - blank there')
                if n_forced == len(fitted):
                    empty[name] = (f'all {len(fitted)} bins forced to the {alab} kinematics:\n'
                                   f'{lab} has no {q.lower()} of its own here')
        sep = self.state.sep_fits.get(self._prod_token(prod))
        if sep and bm is not None:    # Halpha and [NII] with separate kinematics (computed on request)
            for name, (key, _lab, _tip) in _SEP_MAPS.items():
                maps[name] = _paint_bins(bm, {b: v.get(key, np.nan) for b, v in sep.items()})
                if not np.isfinite(maps[name]).any():
                    empty[name] = f'no bin reaches S/N {vb.SEP_SNR_MIN:g} in this line'
        tips = {n: f'{n}  ({base})' + ('' if n == 'SN_LINE' else
                                       '\none value per Voronoi bin: a click shows the spectrum of the bin '
                                       'and its fitted model') for n in maps}
        tips['SN_LINE'] = (f'SN_LINE  ({base})\nS/N of the anchor line per spaxel - the input of the '
                           'binning (a click shows the bin containing the spaxel)')
        for n, t in extra_tips.items():
            tips[n] += t
        for n, (_key, _lab, tip) in _SEP_MAPS.items():
            if n in maps:
                tips[n] = (f'{n}\n{tip}\n(one Gaussian per line, own velocity and width; velocity and width '
                           f'shown from S/N {vb.SEP_SNR_MIN:g} on)')
        names = list(maps)
        ms = _MapSet(kind=f'product:{prod.kind}', title=prod.kind, band=aband, order=_pipe_order(names, anchor),
                     labels={n: _SEP_MAPS[n][1] if n in _SEP_MAPS else pretty_map_name(n, unit) for n in maps},
                     maps=maps, binmaps=binmaps, tips={n: tips[n] for n in maps}, empty_msgs=empty)
        ms.defaults = _pipe_defaults(names, anchor)
        ms.bin_of = {n: 'BINMAP' for n in maps if n != 'SN_LINE'} if binmaps else {}
        rows = data.get('rows') or {}
        main = GROUPS[anchor]['main']
        ms.pipe = dict(token=self._prod_token(prod), anchor=anchor, aband=aband, rows=rows,
                       groups=[g for g in GROUP_KEYS
                               if any(str(r.get(f'{_gtag(g)}_BAND') or '').strip() for r in rows.values())],
                       z_ref=_num(meta.get('Z_REF', meta.get('Z'))), cfg=data.get('cfg'), band_binmaps=band_bms,
                       err_scale={k[len('ERRSC_'):]: _num(v, 1.0) for k, v in meta.items() if k.startswith('ERRSC_')},
                       anchor_snr=f'{_ltag(main)}_SNR', noise=data.get('noise') or {}, rest=data.get('rest'),
                       total=data.get('total'), status=str(meta.get('STATUS') or '').strip(),
                       clipped=data.get('clipped') or {},
                       centre=(int(_num(meta.get('CENTY'), -1)), int(_num(meta.get('CENTX'), -1))))
        st = self.state.stacks.get(aband)
        ny, nx = ms.shape if ms.shape else (0, 0)
        if st is None:
            ms.grid_ok, ms.warn = False, True
            ms.note = f'Voronoi fit · {base} · spectra unavailable: band {aband} is not loaded in this reduction'
        elif st.cube.shape[1:] != (ny, nx):
            ms.grid_ok, ms.warn = False, True
            sy, sx = st.cube.shape[1:]
            ms.note = (f'Voronoi fit · {base} · spectra unavailable (grid differs: maps {nx}×{ny}, '
                       f'{aband} stack {sx}×{sy} spaxels)')
        else:
            nb, ts = int(_num(meta.get('NBINS'), 0)), _num(meta.get('TARGETSN'))
            mode = str(meta.get('BINMODE') or '').strip()
            how = {'voronoi': f'{nb} Voronoi bins at S/N {ts:g}',
                   'single': 'one bin (smaller bins do not reach S/N 6)',
                   'spaxel': f'{nb} single spaxels'}.get(mode, f'{nb} bins ({mode})')
            if ms.pipe['status'] == 'no_detection':
                how = 'anchor line NOT detected: one central aperture, fluxes forced (upper limits)'
                ms.warn = True
            ms.note = (f'Voronoi fit ({base}, pipeline {meta.get("PIPEVER", "?")}, '
                       f'{str(meta.get("DATE", ""))[:10]}) · {how} on {LINE_LABELS.get(main, main)} · '
                       f'{aband} grid, {nx}×{ny} spaxels of {st.pixscale:.2f}″ · bands '
                       f'{str(meta.get("BANDS") or aband).replace("+", ", ")} · velocities w.r.t. z_ref = '
                       f'{ms.pipe["z_ref"]:.5f} · click a bin for its spectrum and fit')
            if vb is None:
                ms.warn = True
                ms.note += f' · spectra unavailable: {_VB_NAME} not importable ({_VB_ERROR})'
            elif ms.pipe['cfg'] is None:
                ms.warn = True
                ms.note += (f' · made by an older pipeline ({prod.schema or "unknown schema"}): press '
                            f'"Compute Voronoi fit" to update it (spectra and fit details need {vb.SCHEMA})')
        return ms

    def _vnew_set(self):
        """Placeholder of a galaxy without Voronoi fit: explains the Compute button."""
        st = self.state
        g, band = self._predicted_anchor()
        ms = _MapSet(kind='product:Voronoi fit', title='Voronoi fit', band=band, order=[], labels={})
        job = self._vjobs.get(self._vjob_key())
        ok, why = self._vfit_possible()
        if job is not None:
            ms.empty_msg = 'fitting …'
            ms.note = (f'Voronoi fit of {st.key} running in the background '
                       f'(bin {job["done"]}/{job["total"] or "?"}) …')
        elif not ok:
            ms.empty_msg, ms.note, ms.warn = 'not available', f'Voronoi fit not possible: {why}', True
        else:
            ms.empty_msg = 'press Compute Voronoi fit'
            ms.note = (f'No Voronoi fit of {st.key} yet: press "Compute Voronoi fit" - Voronoi bins on the '
                       f'{LINE_LABELS.get(GROUPS[g]["main"], g)} S/N ({band} grid), up to {self.vcomp_spin.value()} '
                       'Gaussian components per line group, line ratios. Runs in the background; the product '
                       f'is saved to {PIPELINE_DIR}/{st.catalog.variant}/.')
        return ms

    # ================================================================== Voronoi fit worker
    def _vjob_key(self):
        st = self.state
        return (st.catalog.variant if st.catalog is not None else None, st.key)

    def _predicted_anchor(self):
        """(group, band) the pipeline anchors on: first of Hα, [OIII], Hβ covered (deepest band)."""
        z, stacks = self.state.z, self.state.stacks
        if not z or not stacks:
            return None, None
        for g in (vb.ANCHORS if vb is not None else ('ha', 'oiii', 'hb')):
            b = analysis.pick_band(stacks, REST[GROUPS[g]['main']] * (1 + z))
            if b:
                return g, b
        return None, None

    def _vfit_possible(self):
        """(ok, reason) - can the Voronoi fit of the current galaxy be computed?"""
        st = self.state
        if vb is None:
            return False, f'{_VB_NAME} could not be imported ({_VB_ERROR})'
        if st.root is None or st.catalog is None or not st.key:
            return False, 'no galaxy loaded'
        if not st.stacks:
            return False, 'the cubes are not loaded'
        if st.z is None:
            return False, 'needs a redshift (none known for this galaxy; enter one in the header)'
        if self._predicted_anchor()[0] is None:
            return False, 'none of Hα, [OIII]5007, Hβ is covered by the loaded bands'
        return True, ''

    def _vfit_path(self):
        cat = self.state.catalog          # its own folder: state.root may already be a newly chosen one
        return os.path.join(pipeline_dir(cat.root, cat.variant), vb.product_name(self.state.key))

    def _sep_possible(self):
        """(ok, reason, product) - can Halpha and [NII] be fitted separately for the shown Voronoi fit?"""
        prod = self._current_product()
        if vb is None or prod is None or not prod.is_pipeline:
            return False, 'needs a Voronoi fit as the map source', None
        data = self._prod_data.get(self._prod_token(prod))
        if not data:
            return False, 'the Voronoi fit is still being read', prod
        if 'error' in data:
            return False, f'the Voronoi fit could not be read ({data["error"]})', prod
        if data.get('cfg') is None:
            return False, 'made by an older pipeline: compute the Voronoi fit again first', prod
        rows = data.get('rows') or {}
        bands = {str(r.get('HANII_BAND') or '').strip() for r in rows.values()} - {''}
        if not bands:
            return False, 'Hα+[NII] was not fitted in this Voronoi fit', prod
        if not bands <= set(self.state.stacks):
            return False, f'band {"/".join(sorted(bands))} of Hα+[NII] is not loaded', prod
        return True, '', prod

    def _update_sep_button(self):
        ok, why, prod = self._sep_possible()
        token = self._prod_token(prod) if prod is not None else None
        running, done = token in self.state.sep_running, token in self.state.sep_fits
        b = self.sep_btn
        b.setText('Fitting Hα / [NII] …' if running else ('Hα / [NII] separately ✓' if done else 'Hα / [NII] separately'))
        b.setEnabled(ok and not running and not done)
        b.setToolTip('Fit Hα and the [NII] doublet of every bin with separate velocity and width (the Voronoi fit '
                     'ties them).\nAdds maps of Hα and [NII] v50, W80, S/N and Δv([NII] − Hα); a few seconds.'
                     if ok and not done else ('Done: see the maps “… separate fit” in the map choosers'
                                              if done else f'Not possible: {why}'))

    def _compute_separate(self):
        ok, why, prod = self._sep_possible()
        if not ok:
            self.state.status.emit(f'Hα / [NII] separately: {why}', 6000)
            return
        token = self._prod_token(prod)
        self.state.start_separate_fit(token, self.state.key, _separate_job, prod.path, dict(self.state.stacks),
                                      token[1])

    def _on_sep_fits(self, token):
        """A separate fit started or finished (here or in the Radial profiles tab)."""
        cur = self._current_product()
        if cur is not None and self._prod_token(cur) == token and token in self.state.sep_fits:
            if self.isVisible():
                self._apply_source(controls=False)
            else:
                self._dirty = True
        self._update_sep_button()

    def _update_vfit_controls(self):
        self._update_sep_button()
        ok, why = self._vfit_possible()
        job = self._vjobs.get(self._vjob_key())
        pipe = self._is_pipe_sid(self.source_combo.currentData())
        btn = self.vfit_btn
        btn.setText('Compute Voronoi fit' if job is None else
                    f'Computing… {job["done"]}/{job["total"]}' if job['total'] else 'Computing…')
        btn.setEnabled(ok and job is None)
        if ok:
            path = self._vfit_path()
            btn.setToolTip('Run the Voronoi pipeline for this galaxy: Voronoi bins on the anchor-line S/N, '
                           '1 to "Max comp." Gaussian components per line group (Hβ, [OIII], Hα+[NII], [SII]), '
                           'line ratios. Runs in the background (seconds to a few minutes).\nWrites '
                           f'{path}' + (' (replaces the existing file)' if os.path.exists(path) else ''))
        else:
            btn.setToolTip(f'Not possible: {why}')
        for w in (self.vcomp_spin, self.vsn_spin):
            w.setEnabled(job is None)
        self.vprogress.setVisible(pipe and job is not None)
        if job is not None:
            self.vprogress.setRange(0, max(job['total'], 1))
            self.vprogress.setValue(job['done'])

    def _compute_voronoi(self):
        """vb.run_galaxy for the current galaxy in a worker; write and show the product."""
        ok, why = self._vfit_possible()
        jk = self._vjob_key()
        if not ok:
            self.state.status.emit(f'Voronoi fit: {why}', 6000)
            return
        if jk in self._vjobs:
            return
        st = self.state
        key, z, variant = st.key, float(st.z), st.catalog.variant
        target = float(self.vsn_spin.value())
        cfg = vb.Config(max_components=int(self.vcomp_spin.value()), target_sn=target,
                        target_fallback=tuple(t for t in vb.Config().target_fallback if t < target))
        vb.sync_rest(engine.REST_UM, engine.BAND_RESOLUTION)
        vb.sync_sky()                            # OH lines for the sky-residual guard
        self._synced_root = engine.ROOT
        path = self._vfit_path()
        zr = st.redshift
        meta = dict(VARIANT=variant, ZTYPE=(zr.z_type if zr is not None else '') or '',
                    CREATOR=f'EMPOWER Explorer {__version__}')
        info = dict(job=None, t0=time.time(), done=0, total=0)

        def done(result, jk=jk):
            self._vjobs.pop(jk, None)
            mp, data, rec = result
            self._prod_data[self._prod_token(mp)] = data
            cat = self.state.catalog
            if cat is not None and cat.variant == jk[0]:        # register it in the catalogue
                lst = cat.products.setdefault(jk[1], [])
                same = os.path.normcase(os.path.abspath(mp.path))
                lst[:] = [mp] + [p for p in lst if os.path.normcase(os.path.abspath(p.path)) != same]
                if 'has_maps' in cat.galaxies.columns and jk[1] in cat.galaxies.index:
                    cat.galaxies.loc[jk[1], 'has_maps'] = True
                self.state.productsChanged.emit(jk[1])
            self.state.status.emit(f'Voronoi fit of {jk[1]}: {rec["nbins"]} bin(s) ({rec["status"]}) in '
                                   f'{time.time() - info["t0"]:.0f} s → {os.path.basename(mp.path)}', 8000)
            if jk == self._vjob_key() and self.state.stacks:        # still this galaxy: show it
                self._source_pref = 'product:Voronoi fit'
                if self.isVisible():
                    self._rebuild()
                else:
                    self._dirty = True
            else:                                                   # galaxy changed: not shown
                self._update_vfit_controls()

        def fail(msg, jk=jk):
            self._vjobs.pop(jk, None)
            first = msg.splitlines()[0] if msg else 'unknown error'
            self.state.status.emit(f'Voronoi fit of {jk[1]} failed: {first}', 10000)
            self.state.error.emit(f'Voronoi fit of {jk[1]} failed', msg)
            if jk == self._vjob_key():
                self._update_vfit_controls()
                if self.source_combo.currentData() == 'vnew':
                    self._apply_source(controls=False)

        def progress(d, t, jk=jk):
            info['done'], info['total'] = d, t
            if jk == self._vjob_key():
                self.vprogress.setRange(0, max(t, 1))
                self.vprogress.setValue(d)
                self.vfit_btn.setText(f'Computing… {d}/{t}')
                if self.source_combo.currentData() == 'vnew':
                    self.hint.setText(f'Voronoi fit of {jk[1]} running in the background (bin {d}/{t}) …')

        info['job'] = run_async(_run_voronoi, dict(st.stacks), z, cfg, key, path, meta, variant,
                                on_done=done, on_error=fail, on_progress=progress)
        self._vjobs[jk] = info
        self.state.status.emit(f'Voronoi fit of {key} started (max {cfg.max_components} components, '
                               f'target S/N {target:g}) …', 5000)
        self._update_vfit_controls()
        if self.source_combo.currentData() == 'vnew':
            self._apply_source(controls=False)      # panels say 'fitting …'

    @staticmethod
    def _product_defaults(names):
        picks = [('HA_FLUX', 'SPAX_HA_FLUX', 'BIN_HA_FLUX', 'OIII5007_FLUX', 'HB_FLUX'),
                 ('HA_DV_KMS', 'V', 'KIN_V_HA', 'BIN_HA_DV', 'OIII5007_DV_KMS'),
                 ('HA_SIGMA_INT_KMS', 'SIGMA_CORR', 'KIN_SIG_CORR', 'SIGMA', 'BIN_HA_SIG',
                  'OIII5007_SIGMA_INT_KMS')]
        out = []
        for cands in picks:
            n = next((c for c in cands if c in names and c not in out), None)
            if n is None:
                n = next((c for c in names if c not in out and c not in ('BINMAP', 'KINBINMAP')), None)
            out.append(n)
        return out

    def _cube_set(self, band):
        st = self.state.stacks.get(band)
        z = self.state.z
        if st is None:
            return None
        key = (self.state.variant, self.state.key, band, round(z, 6) if z else None)
        if key not in self._cube_cache:
            maps = {}
            line_keys = analysis.covered_lines(st, z) if z else []
            for k in line_keys:
                m = analysis.line_map(st, z, k)
                if m is not None:
                    maps[f'LINE_{k}'] = np.asarray(m, float)
            maps['WHITE'] = np.asarray(st.white, float)
            snr = analysis.snr_map(st)
            if snr is not None:
                maps['SNR'] = np.asarray(snr, float)
            if len(self._cube_cache) > 20:
                self._cube_cache.clear()
            self._cube_cache[key] = maps
        maps = self._cube_cache[key]
        u = _bunit(st)
        labels, tips = {}, {}
        for n in maps:
            if n.startswith('LINE_'):
                k = n[5:]
                labels[n] = f'{LINE_LABELS[k]} line map [{u}]'
                tips[n] = (f'{LINE_LABELS[k]}: sum of 7 channels at the line minus the local continuum '
                           f'(median 40–120 Å away), {u}')
        labels['WHITE'], tips['WHITE'] = f'White light [{u}]', f'Median over all wavelengths, {u}'
        labels['SNR'], tips['SNR'] = 'Continuum S/N', 'Median DATA/STAT over the band (per channel)'
        anchors = [f'LINE_{k}' for k in ('Ha', 'OIII5007', 'Hb') if f'LINE_{k}' in maps]
        others = [n for n in maps if n.startswith('LINE_') and n not in anchors]
        defaults = (anchors[:1] + ['WHITE', 'SNR'] + anchors[1:] + others)
        ms = _MapSet(kind='cube', title='Cube: line maps', band=band,
                     order=[n for n in maps if n.startswith('LINE_')] + [n for n in ('WHITE', 'SNR') if n in maps],
                     labels=labels, maps=maps, tips=tips,
                     defaults=[n for n in defaults if n in maps][:N_PANELS])
        ny, nx = st.cube.shape[1:]
        ms.note = (f'Maps from the {band} stack ({nx}×{ny} spaxels of {st.pixscale:.2f}″) · '
                   'click a spaxel for its spectrum')
        if not z:
            ms.note = f'No redshift known: only white light and S/N of the {band} stack (line maps need z)'
            ms.warn = True
            ms.empty_msg = 'no line maps without z'
        return ms

    def _current_live_key(self):
        band, line, b = (self.band_combo.currentData(), self.line_combo.currentData(),
                         self.bin_combo.currentData())
        z = self.state.z
        if not (band and line and z) or self.source_combo.currentData() != 'live':
            return None
        return (self.state.variant, self.state.key, band, line, int(b or 1), round(z, 6))

    def _live_set(self, band, line, binning):
        st = self.state.stacks.get(band)
        z = self.state.z
        binning = int(binning or 1)
        lab = LINE_LABELS.get(line, '')
        u = _bunit(st) if st is not None else 'ADU/s'
        labels = {'flux': f'{lab} flux [{u}·µm]', 'flux_err': f'{lab} flux error [{u}·µm]',
                  'snr': f'{lab} searched S/N', 'dv': f'{lab} Δv [km/s]', 'sigma_int': f'{lab} σ_int [km/s]',
                  'w80': f'{lab} W80 [km/s]', 'detected': f'{lab} detected (1 = yes)'}
        tips = {'dv': 'Velocity offset w.r.t. the redshift in use',
                'sigma_int': 'Intrinsic dispersion (instrumental LSF removed)',
                'w80': 'Width containing 80 % of the flux = 2.563 σ_int (single Gaussian)',
                'detected': 'Reliable fit and min(fit S/N, aperture S/N) ≥ 5 - the calibrated anchor '
                            'rule of line_snr.measure_lines (~2 % false positives)',
                'snr': 'Gaussian-fit flux / its error of a FREE velocity/width search per spaxel. Not '
                       'calibrated: on empty sky it gives S/N ≥ 3 in ~20 % of the spaxels (line_snr null '
                       'tests). Use "detected" or S/N ≥ 5 for a reliable mask.'}
        ms = _MapSet(kind='live', title='Live fit', band=band, order=list(LIVE_ORDER), labels=labels,
                     tips=tips, defaults=['flux', 'dv', 'w80'], line=line, binning=binning)
        if z is None:
            ms.empty_msg = 'needs a redshift'
            ms.note, ms.warn = 'No redshift known: live fits need z', True
            return ms
        if st is None or line is None:
            ms.empty_msg = 'no fit line in this band'
            ms.note, ms.warn = 'None of Hα, [OIII]5007, Hβ, [SII]6716, [NII]6583 lies in this band', True
            return ms
        key = self._current_live_key()
        ny, nx = st.cube.shape[1:]
        if key in self._live_cache:
            ms.maps = self._live_cache[key]
            ms.note = (f'Live {lab} fit in every {binning}×{binning} block of the {band} stack '
                       f'({nx}×{ny} spaxels) · click a spaxel for its spectrum')
        else:
            running = key in self._jobs
            ms.empty_msg = 'fitting …' if running else 'press Compute'
            ms.note = (f'Fit {lab} in every {binning}×{binning} block of the {band} stack: press Compute '
                       '(about 3 s, runs in the background)')
        return ms

    # ================================================================== live fit worker
    def _update_live_buttons(self):
        key = self._current_live_key()
        running = key in self._jobs
        cached = key in self._live_cache
        self.compute_btn.setEnabled(key is not None and not running and not cached)
        self.compute_btn.setText('Computing…' if running else ('Done' if cached else 'Compute'))
        lab = LINE_LABELS.get(self.line_combo.currentData(), 'the line')
        self.compute_btn.setToolTip(f'Fit {lab} in every spaxel block of the chosen band '
                                    '(background job, about 3 s; cached per galaxy, band, line, binning)')
        self.progress.setVisible(self.source_combo.currentData() == 'live' and running)

    def _maybe_auto_compute(self):
        key = self._current_live_key()
        if self.auto_chk.isChecked() and key and key not in self._live_cache and key not in self._jobs:
            self._compute()

    def _compute(self):
        key = self._current_live_key()
        if key is None or key in self._jobs or key in self._live_cache:
            return
        _, gkey, band, line, b, z = key
        st = self.state.stacks.get(band)
        if st is None:
            return

        def done(maps, key=key):
            job_t0 = self._jobs.pop(key, (None, time.time()))[1]
            self._live_cache[key] = maps
            n = int(np.isfinite(maps['snr']).sum())
            self.state.status.emit(f'Live {LINE_LABELS[key[3]]} fit of {key[1]} ({key[2]}, {key[4]}×{key[4]}): '
                                   f'{n} spaxels in {time.time() - job_t0:.1f} s', 5000)
            if key == self._current_live_key():
                self._apply_source(controls=False)
            else:
                self._update_live_buttons()

        def fail(msg, key=key):
            self._jobs.pop(key, None)
            self._update_live_buttons()
            self.state.status.emit(f'Live fit failed: {msg.splitlines()[0]}', 8000)
            if key == self._current_live_key():
                for p in self.panels:
                    p.image.clear_image('fit failed')
                    if p.image.bar is not None:
                        p.image.bar.setVisible(False)

        def progress(d, t, key=key):
            if key == self._current_live_key():
                self.progress.setRange(0, max(t, 1))
                self.progress.setValue(d)

        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        job = run_async(analysis.spaxel_fit_maps, st, z, line, binning=b,
                        on_done=done, on_error=fail, on_progress=progress)
        self._jobs[key] = (job, time.time())
        self._apply_source(controls=False)      # panels say 'fitting …', button + progress update

    # ================================================================== showing a map set
    def _set_mapset(self, ms):
        self._mapset = ms
        self._populate_spec_lines(ms)
        self._clear_readouts()
        self.hint.setText(ms.note if ms else '')
        self.hint.setProperty('role', 'warn' if (ms and ms.warn) else 'dim')
        self.hint.style().unpolish(self.hint)
        self.hint.style().polish(self.hint)
        chosen = self._chosen.get(ms.kind, []) if ms else []
        for i, p in enumerate(self.panels):
            want = None
            if ms is not None:
                if i < len(chosen) and chosen[i] in ms.order:
                    want = chosen[i]
                elif i < len(ms.defaults):
                    want = ms.defaults[i]
            p.populate(ms, want)
        has_snr = bool(ms and ms.maps and any(ms.snr_name(n) for n in ms.order[:1] + ms.defaults if n))
        self.snr_spin.setEnabled(has_snr)
        self.snr_label.setEnabled(has_snr)
        # keep the clicked pixel (and the zoom) while the maps stay on the same grid; a set without
        # maps yet ('reading …', 'press Compute') keeps it for the set that follows
        prev = self._grid
        grid = (ms.band, ms.shape) if (ms is not None and ms.maps) else None
        if grid is not None:
            if grid != prev or self._sel is None:
                self._sel = self._default_pixel(ms)
            self._grid = grid
        elif ms is None or prev is None or ms.band != prev[0]:
            self._sel = self._default_pixel(ms)
        self._redraw_maps(keep_view=grid is not None and grid == prev)
        self._schedule_spectrum()

    def _default_pixel(self, ms):
        """The shared aperture centre (Hα peak) on the grid of the map set (Voronoi fits: their
        centre, the anchor-line peak)."""
        st = self.state
        if ms is None or not ms.grid_ok or ms.band not in st.stacks or st.ap_x is None:
            return None
        if ms.pipe is not None and ms.shape is not None:
            px = self._pipe_default_pixel(ms)
            if px is not None:
                return px
        try:
            x, y, _ = analysis.transfer(st.stacks[st.ref_band], st.stacks[ms.band], st.ap_x, st.ap_y)
        except Exception:
            return None
        ny, nx = st.stacks[ms.band].cube.shape[1:]
        ix, iy = int(round(x)), int(round(y))
        return (ix, iy) if (0 <= ix < nx and 0 <= iy < ny) else None

    @staticmethod
    def _pipe_default_pixel(ms):
        """Voronoi fit: the galaxy centre of the product if it lies in a bin, else the bin with the
        highest anchor-line S/N (its spaxel nearest the bin centroid)."""
        P, (ny, nx) = ms.pipe, ms.shape
        bm = ms.binmaps.get('BINMAP')
        cy, cx = P['centre']
        if bm is None or bm.shape != (ny, nx):
            return (cx, cy) if (0 <= cy < ny and 0 <= cx < nx) else None
        if 0 <= cy < ny and 0 <= cx < nx and bm[cy, cx] >= 0:
            return cx, cy
        main = _ltag(GROUPS[P['anchor']]['main'])
        snr = {b: _num(r.get(f'{main}_SNR')) for b, r in P['rows'].items()}
        snr = {b: s for b, s in snr.items() if np.isfinite(s) and (bm == b).any()}
        if snr:
            b = max(snr, key=snr.get)
            ys, xs = np.nonzero(bm == b)
            r = P['rows'][b]
            yc, xc = _num(r.get('YBAR'), ys.mean()), _num(r.get('XBAR'), xs.mean())
        else:
            ys, xs = np.nonzero(bm >= 0)
            yc, xc = cy, cx
        if xs.size == 0:
            return None
        k = int(np.argmin(np.hypot(ys - yc, xs - xc)))
        return int(xs[k]), int(ys[k])

    def _map_chosen(self, i):
        ms = self._mapset
        if ms is None:
            return
        self._chosen[ms.kind] = [p.name() for p in self.panels]
        self._clear_readouts()
        self._draw_panel(i, keep_view=True)
        if i == 0:
            self._update_histogram()
        if i == self._focus:
            self._schedule_spectrum()

    def _masked(self, name):
        """Map `name` with spaxels below the S/N threshold set to NaN."""
        ms = self._mapset
        arr = ms.maps[name]
        thr = self.snr_spin.value()
        sn = ms.snr_name(name)
        if thr <= 0 or sn is None or ms.style(name) == 'integer' or ms.maps[sn].shape != arr.shape:
            return arr
        with np.errstate(invalid='ignore'):
            return np.where(ms.maps[sn] >= thr, arr, np.nan)

    def _redraw_maps(self, keep_view=False):
        for i in range(N_PANELS):
            self._draw_panel(i, keep_view)
        self._update_histogram()

    def _draw_panel(self, i, keep_view=False):
        panel, ms = self.panels[i].image, self._mapset
        name = self.panels[i].name()
        panel.clear_overlays()
        if ms is None or not ms.maps or name not in ms.maps:
            msg = (ms.empty_msg if ms is not None and not ms.maps else
                   ('no further maps' if ms is not None and ms.maps else 'nothing to show'))
            if ms is not None and not ms.maps and i > 0:      # one message is enough
                msg = ''
            panel.clear_image(msg)
            panel.set_marker()
            if panel.bar is not None:
                panel.bar.setVisible(False)     # no stale colour scale next to an empty panel
            self._shown[i] = None
            self._shape_drawn[i] = None
            return
        arr = self._masked(name)
        style = ms.style(name)
        autorange = not (keep_view and self._shape_drawn[i] == arr.shape)
        if panel.bar is not None:
            panel.bar.getAxis('right').setTicks(None)       # categorical maps set integer ticks
        if style == 'integer':
            self._draw_categorical(panel, arr, autorange)
        elif style == 'velocity':
            panel.set_image(arr, cmap=theme.CMAP_VELOCITY, symmetric=True, autorange=autorange)
        else:
            panel.set_image(arr, cmap=theme.CMAP_IMAGE if style == 'flux' else theme.CMAP_SEQUENTIAL,
                            autorange=autorange)
        if panel.data is None:
            thr = self.snr_spin.value()
            panel.clear_image(f'all spaxels below S/N {thr:g}' if (thr > 0 and np.isfinite(ms.maps[name]).any())
                              else ms.empty_msgs.get(name, 'no data'))
        if panel.bar is not None:
            panel.bar.setVisible(panel.data is not None)
        self._shown[i] = arr
        self._shape_drawn[i] = arr.shape if panel.data is not None else None   # empty: next map autoranges
        if self._sel is not None and panel.data is not None:
            panel.set_marker(*self._sel)
        else:
            panel.set_marker()
        if self._result is not None and self._result.members is not None and panel.data is not None:
            self._overlay_members(panel, self._result.members)

    @staticmethod
    def _draw_categorical(panel, arr, autorange):
        v = arr[np.isfinite(arr)]
        if v.size == 0:
            panel.clear_image('no data')
            return
        lo, hi = int(np.floor(v.min())), int(np.ceil(v.max()))
        n = hi - lo + 1
        if n > MAX_CATEGORIES:
            panel.set_image(arr, cmap=theme.CMAP_SEQUENTIAL, autorange=autorange)
            return
        binary = lo >= 0 and hi <= 1
        if binary:
            lo, hi, n = 0, 1, 2
        cm = _categorical_cmap(n, binary)
        panel.set_image(arr, cmap=theme.CMAP_SEQUENTIAL, levels=(lo - 0.5, hi + 0.5), autorange=autorange)
        panel.img.setColorMap(cm)
        if panel.bar is not None:
            panel.bar.setColorMap(cm)
            panel.bar.setLevels((lo - 0.5, hi + 0.5))
            step = max(1, int(np.ceil(n / 12)))
            panel.bar.getAxis('right').setTicks([[(float(k), str(k)) for k in range(lo, hi + 1, step)], []])

    @staticmethod
    def _overlay_members(panel, members):
        ys, xs = np.nonzero(members)
        if xs.size == 0:
            return
        sc = pg.ScatterPlotItem(x=xs.astype(float), y=ys.astype(float), symbol='s', size=1.0, pxMode=False,
                                pen=pg.mkPen(255, 255, 255, 170, width=1), brush=None)
        sc.setZValue(22)
        panel.add_overlay(sc)

    # ================================================================== mouse
    def _hovered(self, x, y):
        ms = self._mapset
        if ms is None or not ms.maps:
            return
        where = f'({x}, {y})'
        bm = ms.binmaps.get('BINMAP') if ms.pipe is not None else None
        if bm is not None and 0 <= y < bm.shape[0] and 0 <= x < bm.shape[1] and bm[y, x] >= 0:
            where += f' bin {bm[y, x]}'
        for i, p in enumerate(self.panels):
            n = p.name()
            a = ms.maps.get(n)
            if a is None or p.image.data is None or not (0 <= y < a.shape[0] and 0 <= x < a.shape[1]):
                p.set_readout('')
                continue
            v = a[y, x]
            txt = _fmt(v, '')
            if ms.style(n) == 'integer' and np.isfinite(v):
                txt = f'{int(round(v))}'
            shown = self._shown[i]
            if (np.isfinite(v) and shown is not None and shown.shape == a.shape
                    and not np.isfinite(shown[y, x])):
                txt += '  (below S/N cut)'
            p.set_readout(f'{where}  {txt}')

    def _clear_readouts(self):
        for p in self.panels:
            p.set_readout('')

    def _clicked(self, i, x, y):
        self._focus = i
        self._sel = (x, y)
        for p in self.panels:
            if p.image.data is not None:
                p.image.set_marker(x, y)
        self._schedule_spectrum()

    # ================================================================== spectrum of a pixel
    def _schedule_spectrum(self):
        self._spec_timer.start()

    def _spectrum_line(self, ms):
        """Line whose fit to show: header choice, else the clicked map's line, else the set's."""
        z = self.state.z
        if not z:
            return None
        forced = self.spec_line_combo.currentData()
        if forced:
            return forced
        names = [self.panels[self._focus].name()] + [p.name() for p in self.panels]
        for n in names:
            k = ms.line if ms.kind == 'live' else (_line_of_map(n) if n else None)
            if k and analysis.pick_band(self.state.stacks, REST[k] * (1 + z)):
                return k
        for k in ('Ha', 'OIII5007', 'Hb', 'NII6583', 'SII6716'):
            if analysis.pick_band(self.state.stacks, REST[k] * (1 + z)):
                return k
        return None

    def _pixel_request(self, ms, x, y):
        """What to extract at grid pixel (x, y) - cheap, GUI thread.  Returns (cache key,
        _PixelResult without spectrum, worker arguments (stack, binmap, bin_id, ix, iy, z))."""
        stacks, z = self.state.stacks, self.state.z
        g = stacks[ms.band]
        line = self._spectrum_line(ms)
        band = ms.band
        if line and not g.covers(REST[line] * (1 + z)):
            band = analysis.pick_band(stacks, REST[line] * (1 + z)) or ms.band
        st = stacks[band]
        if band == ms.band:
            xb, yb = float(x), float(y)
        else:
            xb, yb, _ = analysis.transfer(g, st, x, y)
        ix, iy = int(round(xb)), int(round(yb))
        aligned = band == ms.band or (abs(xb - x) < 0.3 and abs(yb - y) < 0.3
                                      and st.cube.shape[1:] == g.cube.shape[1:])
        focus_name = self.panels[self._focus].name() or ''
        btag, binmap = ms.binmap_for(focus_name)
        res = _PixelResult(band=band, x=x, y=y, line=line)
        bin_id = None
        if binmap is not None and binmap.shape == g.cube.shape[1:]:
            bin_id = int(binmap[y, x])
            if bin_id < 0:
                res.note = 'This spaxel is in no bin: single-spaxel spectrum.'
                bin_id = None
            elif not aligned:
                res.note = f'Bins are on the {ms.band} grid: single-spaxel spectrum in {band}.'
                bin_id = None
        ckey = (self.state.variant, self.state.key, round(z, 6) if z else None, band, ix, iy,
                None if bin_id is None else (ms.kind, btag, bin_id), line)
        if bin_id is not None:
            res.members = binmap == bin_id
            n = int(res.members.sum())
            what = 'kinematic bin' if btag == 'KINBINMAP' else 'bin'
            res.what = (f'{ms.binning}×{ms.binning} block at ({x}, {y})' if ms.kind == 'live'
                        else f'{what} {bin_id} ({n} spaxel{"s" if n != 1 else ""}) at ({x}, {y})')
        else:
            res.what = f'spaxel ({ix}, {iy})' + (f' [map ({x}, {y})]' if (ix, iy) != (x, y) else '')
        return ckey, res, (st, binmap if bin_id is not None else None, bin_id, ix, iy, z)

    @staticmethod
    def _compute_pixel(res, st, binmap, bin_id, ix, iy, z):
        """Worker thread: spectrum of the spaxel / bin / block and its line fit (~0.2 s)."""
        if bin_id is not None:
            res.spec = analysis.bin_spectrum(st, binmap, bin_id)
        else:
            ny, nx = st.cube.shape[1:]
            if 0 <= ix < nx and 0 <= iy < ny:
                res.spec = analysis.extract_spectrum(st, ix, iy, 0.5)
        if res.spec is not None and res.spec.n_pix > 0 and z:
            res.fits, res.info = analysis.fit_lines(res.spec, z)
        return res

    # ------------------------------------------------------------------ Voronoi fits: bin spectra
    def _spectrum_group(self, ms):
        """Line group of a Voronoi fit to show: header choice, else the clicked map's group, else
        the anchor group."""
        P = ms.pipe
        want = self.spec_line_combo.currentData()
        if isinstance(want, str) and want.startswith('grp:') and want[4:] in P['groups']:
            return want[4:]
        g = _group_of_map(self.panels[self._focus].name())
        if g in P['groups']:
            return g
        return P['anchor'] if (P['anchor'] in P['groups'] or not P['groups']) else P['groups'][0]

    def _bin_request(self, ms, x, y):
        """What to extract for a click on a Voronoi-fit map (GUI thread, cheap).  The spectrum of a
        group comes from the band that group was fitted in (row '<G>_BAND'): the bin on that band's
        grid is BINMAP (anchor band) or BINMAP_<band>.  Returns (cache key, _BinResult, (request,))
        or (None, _BinResult with .message, None) when there is nothing to extract."""
        P, stacks = ms.pipe, self.state.stacks
        g = self._spectrum_group(ms)
        G, glab, z = _gtag(g), GROUPS[g]['label'], P['z_ref']
        bm = ms.binmaps.get('BINMAP')
        bin_id = int(bm[y, x]) if bm is not None else -1
        row = P['rows'].get(bin_id) if bin_id >= 0 else None
        band = str(row.get(f'{G}_BAND') or '').strip() if row else ''
        if not band:          # not fitted in this bin (or no bin): the band the pipeline would use
            band = analysis.pick_band(stacks, REST[GROUPS[g]['main']] * (1 + z)) if np.isfinite(z) else None
        res = _BinResult(band=band, x=x, y=y, group=g, bin_id=bin_id, row=row)
        if bin_id >= 0:
            n = int((bm == bin_id).sum())
            res.what = f'bin {bin_id} ({n} spaxel{"s" if n != 1 else ""}) at ({x}, {y})'
            res.members = bm == bin_id
        else:
            res.what = f'spaxel ({x}, {y})'
        if band is None or band not in stacks:
            res.message = (f'{glab} is not covered', f'No loaded band covers {glab} at z_ref = {z:.4f}.')
            return None, res, None
        st = stacks[band]
        ny, nx = st.cube.shape[1:]
        bmb = bm if band == P['aband'] else P['band_binmaps'].get(band)
        if bmb is not None and bmb.shape != (ny, nx):
            bmb = None
        if bin_id >= 0:
            if bmb is None:
                res.message = (f'The bins are not available in {band}',
                               f'The product has no BINMAP_{band} on the grid of the {band} stack.')
                return None, res, None
            sel = bmb == bin_id
            if not sel.any():
                res.message = (f'Bin {bin_id} lies outside the {band} field',
                               f'None of its spaxels is covered by the {band} cube.')
                return None, res, None
        else:                 # outside the bins: the spaxel's spectrum in that band, no fit
            if band == P['aband']:
                ix, iy = x, y
            else:
                xb, yb, _ = analysis.transfer(stacks[P['aband']], st, x, y)
                ix, iy = int(round(xb)), int(round(yb))
            if not (0 <= ix < nx and 0 <= iy < ny):
                res.message = (f'Outside the {band} field', f'Spaxel ({x}, {y}) is not covered by the {band} cube.')
                return None, res, None
            sel = np.zeros((ny, nx), bool)
            sel[iy, ix] = True
            res.note = ('This spaxel is in no Voronoi bin (anchor-line S/N too low or outside the binned '
                        'region): its spectrum, without fit.')
        # the noise model the pipeline used in that band (stored in the product)
        nz = P['noise'].get(band)
        if nz is None or len(nz['sd1']) != st.wave.size:
            res.message = (f'No noise model for {band}',
                           f'The product has no NOISE_{band} table matching the {band} stack.')
            return None, res, None
        req = dict(stack=st, cfg=P['cfg'], sel=sel, z_ref=z, noise=nz, rest=P['rest'],
                   clipped=P['clipped'].get((bin_id, G)) if bin_id >= 0 else None)
        ckey = ('bin', P['token'], bin_id if bin_id >= 0 else ('spaxel', x, y), g, band)
        return ckey, res, (req,)

    def _start_pixel_fit(self, ckey, res, args, fn=None):
        """Fit (or, for Voronoi fits, extract + rebuild the model) in the background; only the
        latest request is drawn (older ones are cached)."""
        self._pending = ckey
        if ckey in self._inflight:          # same spaxel requested again while it is being fitted
            return
        self._inflight.add(ckey)

        def done(r, ckey=ckey):
            self._inflight.discard(ckey)
            if getattr(r, 'noise', None) is not None and r.noise_key is not None:
                self._noise_cache[r.noise_key] = r.noise       # noise model of that band, reused
                while len(self._noise_cache) > 16:
                    self._noise_cache.popitem(last=False)
                r.noise = None
            if len(self._fit_cache) > 200:
                self._fit_cache.popitem(last=False)
            self._fit_cache[ckey] = r
            if ckey == self._pending:
                self._pending = None
                self._update_spectrum()

        def fail(msg, ckey=ckey):
            self._inflight.discard(ckey)
            if ckey == self._pending:
                self._pending = None
                self._spec_message('Could not fit this spectrum', msg.splitlines()[0])

        run_async(fn or self._compute_pixel, res, *args, on_done=done, on_error=fail)

    @staticmethod
    def _result_label(res):
        if isinstance(res, _BinResult):
            return GROUPS[res.group]['label']
        return LINE_LABELS.get(res.line, '')

    def _update_spectrum(self):
        ms = self._mapset
        prev_members = self._result.members if self._result is not None else None
        self._result = None
        self._pending = None
        busy = False
        if ms is None or not ms.maps:
            self._clear_fit_plots()
            text, hint = 'No maps yet', ''
            if ms is not None:
                if ms.empty_msg == 'reading …':
                    text = 'Reading the maps …'
                elif ms.empty_msg == 'fitting …':
                    text = 'Fitting every spaxel …' if ms.kind == 'live' else 'Computing the Voronoi fit …'
                elif ms.empty_msg == 'press Compute':
                    hint = 'Press Compute to fit every spaxel, then click one.'
                elif ms.empty_msg == 'press Compute Voronoi fit':
                    hint = 'Press "Compute Voronoi fit", then click a bin.'
            self._spec_message(text, hint)
        elif ms.pipe is not None and vb is None:
            self._spec_message('Spectra unavailable', f'{_VB_NAME} could not be imported '
                                                      f'({_VB_ERROR}): the fitted models can not be rebuilt.')
        elif ms.pipe is not None and ms.pipe['cfg'] is None:
            self._spec_message('Spectra unavailable', 'This Voronoi fit was made by an older pipeline version: '
                                                      'press "Compute Voronoi fit" to update it.')
        elif not ms.grid_ok:
            self._spec_message('Spectra unavailable (grid differs)',
                               'These maps are not on the pixel grid of a loaded stack, so a map pixel '
                               'can not be matched to a cube spaxel.')
        elif ms.band not in self.state.stacks:
            self._spec_message('Spectra unavailable', f'Band {ms.band} is not loaded.')
        elif self._sel is None:
            self._spec_message('Click a spaxel in any map', 'Its spectrum and line fit appear here.')
        else:
            x, y = self._sel
            ny, nx = self.state.stacks[ms.band].cube.shape[1:]
            if not (0 <= x < nx and 0 <= y < ny):
                self._spec_message('Click a spaxel inside the map')
            else:
                if ms.pipe is not None:
                    ckey, res, args = self._bin_request(ms, x, y)
                    fn = _compute_bin
                else:
                    ckey, res, args = self._pixel_request(ms, x, y)
                    fn = None
                if ckey is None:                    # nothing to extract: say why
                    self._result = res
                    self._spec_message(*res.message)
                    self.spec_title.setText(f'  {res.what} · {self._result_label(res)}')
                elif ckey in self._fit_cache:
                    cached = self._fit_cache[ckey]
                    self._fit_cache.move_to_end(ckey)
                    cached.x, cached.y, cached.note, cached.what = x, y, res.note, res.what
                    self._result = cached
                    self._clear_fit_plots()
                    self._draw_result(ms, cached)
                else:
                    # keep the previous plot until the new fit arrives (no flicker), say what is coming
                    self._result = res
                    busy = True
                    lab = self._result_label(res)
                    self.spec_title.setText(f'  {res.what} · {res.band}' + (f' · {lab}' if lab else '')
                                            + ('  ·  reading …' if fn is not None else '  ·  fitting …'))
                    if self.spec_pages.currentIndex() == 0:
                        self.spec_empty.set_text('Reading …' if fn is not None else 'Fitting …', res.what)
                    self._start_pixel_fit(ckey, res, args, fn)
        if not busy and self.spec_pages.currentIndex() == 0:
            self._clear_fit_plots()
        members = self._result.members if self._result is not None else None
        if members is not None or prev_members is not None:
            for p in self.panels:
                p.image.clear_overlays()
                if members is not None and p.image.data is not None:
                    self._overlay_members(p.image, members)

    def _spec_message(self, text, hint=''):
        self.spec_title.setText('')
        self.spec_empty.set_text(text, hint)
        self.spec_pages.setCurrentIndex(0)
        self._update_resid_hist(None)

    def _clear_fit_plots(self):
        self._line_labels = []
        self.spec_plot.clear()
        self.resid_plot.clear()

    def _place_line_labels(self, *_):
        """Put each line label right (else left) of its marker, in the highest row where it
        overlaps no other label and stays inside the plot ([NII]6548 | Hα | [NII]6583 are only
        ~25 Å apart).  Re-run when the plot is resized or the x range changes."""
        items = [(lam, it.label) for lam, it in self._line_labels if getattr(it, 'label', None) is not None]
        if not items:
            return 0, 0.0
        vb = self.spec_plot.getViewBox()
        (x0, x1), _ = vb.viewRange()
        wpx, hpx = vb.width(), vb.height()
        if not (wpx > 20 and hpx > 20 and x1 > x0):
            return 0, 0.0
        upp = (x1 - x0) / wpx                                   # data units per screen pixel
        text_h = max(lab.textItem.boundingRect().height() for _, lab in items)
        dy = min(0.3, (text_h - 2) / hpx)
        n_rows = max(1, min(4, int(0.6 / dy) if dy > 0 else 1))
        rows = [[] for _ in range(n_rows)]
        marks = [lam for lam, _ in items]
        for lam, lab in sorted(items, key=lambda t: t[0]):
            w = (lab.textItem.boundingRect().width() + 2) * upp
            others = [m for m in marks if m != lam]
            choice = None
            # first pass: the label must not cross another line marker; second pass: it may
            for strict in (True, False):
                for r in range(n_rows):
                    for side in ('right', 'left'):
                        a, b = (lam, lam + w) if side == 'right' else (lam - w, lam)
                        if a < x0 or b > x1:
                            continue
                        if strict and any(a < m < b for m in others):
                            continue
                        if all(b <= c or a >= d for c, d in rows[r]):
                            choice = (r, side, a, b)
                            break
                    if choice:
                        break
                if choice:
                    break
            if choice is None:                                  # crowded: least-used row, right side
                r = min(range(n_rows), key=lambda i: len(rows[i]))
                side = 'right' if lam + w <= x1 else 'left'
                choice = (r, side, *((lam, lam + w) if side == 'right' else (lam - w, lam)))
            r, side, a, b = choice
            rows[r].append((a, b))
            anchor = (0, 0) if side == 'right' else (1, 0)
            lab.anchors = [anchor, anchor]
            lab.setPosition(0.97 - r * dy)
        return sum(1 for row in rows if row), dy

    def _draw_result(self, ms, res):
        if isinstance(res, _BinResult):
            self._draw_bin_result(ms, res)
            return
        z = self.state.z
        sp = res.spec
        lab = LINE_LABELS.get(res.line, '')
        self.spec_title.setText(f'  {res.what} · {res.band}' + (f' · {lab}' if lab else ''))
        if sp is None or sp.n_pix == 0 or not np.isfinite(sp.flux).any():
            self._spec_message('No data at this spaxel', 'Fewer than half of the wavelengths are valid here '
                               '(edge of the field or outside this band).')
            self.spec_title.setText(f'  {res.what} · {res.band}')
            return
        self.spec_pages.setCurrentIndex(1)
        st = self.state.stacks[res.band]
        lf = res.fits.get(res.line) if res.line else None
        unit = _bunit(st)
        self.spec_plot.setLabel('left', f'Flux [{unit}]', color=C.TEXT_DIM)

        # x window: the fit window, else the zoom panel of the line, else the whole band
        if lf is not None and lf.wave is not None and len(lf.wave) > 1:
            lo, hi = float(np.nanmin(lf.wave)), float(np.nanmax(lf.wave))
        elif res.line and z:
            panel = next((p for p in PANELS if res.line in p[2]), None)
            _, lo, hi = panel_window(panel, z) if panel else (0, st.wave_min, st.wave_max)
        else:
            lo, hi = st.wave_min, st.wave_max
        inwin = (sp.wave >= lo) & (sp.wave <= hi)
        add_spans(self.spec_plot, [(a, b) for a, b in analysis.telluric_spans(st) if b > lo and a < hi])

        # data with ±1σ band (fit error where there is a fit)
        if lf is not None and lf.err is not None and lf.resid is not None:
            data_w = lf.resid + lf.cont
            _fill_band(self.spec_plot, lf.wave, data_w - lf.err, data_w + lf.err, C.BLUE, 55)
        else:
            e = sp.err if sp.err is not None else sp.noise
            if e is not None:
                _fill_band(self.spec_plot, sp.wave[inwin], (sp.flux - e)[inwin], (sp.flux + e)[inwin], C.BLUE, 55)
        self.spec_plot.plot(sp.wave[inwin], sp.flux[inwin], pen=pg.mkPen(C.BLUE, width=1.3), connect='finite')
        ys = [sp.flux[inwin]]
        if lf is not None and lf.model is not None:
            col = LINE_COLORS.get(res.line, C.ACCENT)
            self.spec_plot.plot(lf.wave, lf.cont, pen=pg.mkPen(C.TEXT_DIM, width=1, style=Qt.DashLine))
            self.spec_plot.plot(lf.wave, lf.model, pen=pg.mkPen(col, width=2.2))
            ys.append(lf.model)
        self._add_line_markers(z, lo, hi, (res.line,))
        self._set_spec_view(lo, hi, ys, lf.model if (lf is not None and lf.model is not None) else None)

        chi = self._draw_residuals(lf, res.line)
        self._update_resid_hist(chi)
        self.info.setText(self._info_html(ms, res, lf, unit))

    def _add_line_markers(self, z, lo, hi, bold=()):
        """Labelled markers of the fit lines within [lo, hi] (the lines in `bold` highlighted)."""
        if not z or not np.isfinite(z):
            return
        for k in FIT_LINES:
            lam = REST[k] * (1 + z)
            if lo <= lam <= hi:
                it = add_vline(self.spec_plot, lam, LINE_LABELS[k], LINE_COLORS.get(k, C.TEXT_DIM),
                               bold=(k in bold))
                it.setZValue(5)
                if getattr(it, 'label', None) is not None:
                    it.label.fill = pg.mkBrush(*pg.mkColor(C.BG_DARK).getRgb()[:3], 200)
                self._line_labels.append((lam, it))

    def _set_spec_view(self, lo, hi, ys, model=None):
        """x range [lo, hi]; y range from the data (and the model peak), the top kept free for the
        line labels so they never sit on the line peak."""
        self.spec_plot.setXRange(lo, hi, padding=0.01)
        n_rows, dy = self._place_line_labels()
        yv = np.concatenate([np.asarray(a, float)[np.isfinite(a)] for a in ys]) if ys else np.array([])
        if yv.size:
            a, b = np.percentile(yv, [0.5, 99.8])
            if model is not None and np.isfinite(model).any():
                b = max(b, float(np.nanmax(model)))
            span = b - a if b > a else abs(b) or 1.0
            f = min(0.45, 0.06 + n_rows * dy) if n_rows else 0.08
            lo_y = a - 0.08 * span
            hi_y = b + (b - lo_y) * f / (1 - f)
            self.spec_plot.setYRange(lo_y, hi_y, padding=0)

    def _draw_residuals(self, lf, line):
        """(data - model)/σ of the local fit region of a LineFit (see _plot_residuals)."""
        if lf is None or lf.err is None or lf.resid is None or lf.model is None or lf.wave is None:
            return self._plot_residuals(None, None, None)
        with np.errstate(invalid='ignore', divide='ignore'):
            ok_err = np.isfinite(lf.err) & (lf.err > 0)
            chi = (lf.resid + lf.cont - lf.model) / np.where(ok_err, lf.err, np.nan)
        used = _fit_pixels(lf, line, self.state.z) & np.isfinite(chi)
        return self._plot_residuals(lf.wave, chi, used)

    def _plot_residuals(self, wave, chi, used):
        """Residual panel: pixels used by the fit bright with dots, the rest of the region
        (continuum side bands, masked pixels) dim.  Returns the χ of the fitted pixels only (what
        the residual histogram and its statistics describe), None without a model."""
        p = self.resid_plot
        band = pg.LinearRegionItem(values=(-1, 1), orientation='horizontal', movable=False,
                                   brush=pg.mkBrush(*pg.mkColor(C.ACCENT).getRgb()[:3], 38))
        for ln in band.lines:
            ln.setPen(pg.mkPen(None))
        band.setZValue(-10)
        p.addItem(band, ignoreBounds=True)
        p.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(C.TEXT_DIM, width=1)), ignoreBounds=True)
        if wave is None or chi is None or used is None:
            p.setYRange(-3, 3, padding=0)
            return None
        # whole local region dim (no line across pixels without a valid error)
        p.plot(wave, chi, pen=pg.mkPen(C.TEXT_DIM, width=1), connect='finite')
        chi_fit = np.where(used, chi, np.nan)
        p.plot(wave, chi_fit, pen=pg.mkPen(C.TEXT, width=1.2), connect='finite',
               symbol='o', symbolSize=3, symbolPen=None, symbolBrush=pg.mkBrush(C.TEXT))
        fin = chi[used]
        m = max(3.5, float(np.percentile(np.abs(fin), 99)) * 1.15) if fin.size else 3.5
        p.setYRange(-min(m, 10), min(m, 10), padding=0)
        return fin

    # ------------------------------------------------------------------ Voronoi fits: drawing
    def _draw_bin_result(self, ms, res):
        """Spectrum of the bin in the group's band (±1σ of the fit noise), the pipeline model (total
        solid, each component dashed in its colour, continuum dim), residuals and the bin's results."""
        P, g = ms.pipe, res.group
        glab = GROUPS[g]['label']
        self.spec_title.setText(f'  {res.what} · {res.band} · {glab}')
        if res.wave is None or not res.has_data:
            self._spec_message('No data here', f'The {res.band} cube has no valid pixels around {glab} '
                                               'for these spaxels (edge of the field or of the band).')
            self.spec_title.setText(f'  {res.what} · {res.band} · {glab}')
            return
        self.spec_pages.setCurrentIndex(1)
        st = self.state.stacks[res.band]
        unit = _bunit(st)
        self.spec_plot.setLabel('left', f'Flux [{unit}]', color=C.TEXT_DIM)
        lo, hi = res.lo, res.hi
        add_spans(self.spec_plot, [(a, b) for a, b in analysis.telluric_spans(st) if b > lo and a < hi])
        w, f, e = res.wave, res.spec, res.err
        _fill_band(self.spec_plot, w, f - e, f + e, C.BLUE, 55)
        self.spec_plot.plot(w, f, pen=pg.mkPen(C.BLUE, width=1.3), connect='finite')
        ys = [f]
        if res.mtotal is not None:
            self.spec_plot.plot(res.mx, res.mcont, pen=pg.mkPen(C.TEXT_DIM, width=1, style=Qt.DashLine))
            if len(res.mparts) > 1:
                for k, part in enumerate(res.mparts):
                    self.spec_plot.plot(res.mx, part, pen=pg.mkPen(COMP_COLORS[k % len(COMP_COLORS)], width=1.6,
                                                                   style=Qt.DashLine))
            col = LINE_COLORS.get(GROUPS[g]['main'], C.ACCENT)
            self.spec_plot.plot(res.mx, res.mtotal, pen=pg.mkPen(col, width=2.2))
            ys.append(res.mtotal)
        self._add_line_markers(P['z_ref'], lo, hi, tuple(GROUPS[g]['lines']))
        self._set_spec_view(lo, hi, ys, res.mtotal)
        chi = self._plot_residuals(res.wave, res.chi, res.used)
        self._update_resid_hist(chi, 'no fit in this bin' if res.bin_id >= 0 else 'spaxel outside the bins: no fit')
        self.info.setText(self._bin_info_html(ms, res, unit))

    def _bin_info_html(self, ms, res, unit):
        """The bin's results for the group: components (v, σ_int ± errors, flux of the main line),
        forced / free, ΔBIC, v50, W80, χ²ν and the S/N of every line of the group."""
        P, g = ms.pipe, res.group
        G, grp = _gtag(g), GROUPS[g]
        dim = f'color:{C.TEXT_DIM}'
        main = grp['main']
        M = _ltag(main)
        col = LINE_COLORS.get(main, C.TEXT)
        alab = GROUPS[P['anchor']]['label']
        cfg = P['cfg']
        rows = [f'<span style="font-size:15px; font-weight:bold; color:{col}">{grp["label"]}</span>'
                f' <span style="{dim}">· {res.band} · {res.what.split(" at ")[0]}</span>']
        r = res.row
        fitted = (r is not None and str(r.get(f'{G}_BAND') or '').strip() == res.band
                  and np.isfinite(_num(r.get(f'{G}_X0'))))
        if r is None:
            if res.bin_id >= 0:
                rows.append(f'<span style="{dim}">The product has no fit results for bin {res.bin_id}.</span>')
        elif not fitted:
            rows.append(f'<span style="{dim}">{grp["label"]} was not fitted in this bin (too few valid '
                        'pixels around the lines).</span>')
        else:
            n = _num(r.get(f'{G}_NCOMP'))
            ncomp = int(n) if np.isfinite(n) else 0
            forced = _num(r.get(f'{G}_FORCED')) == 1
            dbic = _num(r.get(f'{G}_DBIC_NCOMP' if g == P['anchor'] else f'{G}_DBIC_FREE'))
            fb = cfg.force_below if cfg is not None else 5.0
            detected_bins = any(_num(x.get(f'{G}_FORCED')) == 0 for x in P['rows'].values())
            if g == P['anchor']:
                if not forced:
                    kin = '<b>free</b> fit (anchor line: sets the kinematics of the other groups)'
                elif detected_bins:
                    kin = (f'<b>forced</b>: {alab} too weak here (S/N &lt; {fb:g}), velocity interpolated from '
                           'the detected bins')
                else:
                    kin = (f'<b>forced</b>: {alab} not detected (S/N &lt; {fb:g}), reference kinematics '
                           '(catalogue redshift) - the fluxes are upper limits')
            else:
                kin = (f'<b>own kinematics</b> (ΔBIC {dbic:.1f} vs tied to {alab})' if not forced else
                       f'<b>forced</b> to the {alab} kinematics of this bin (own ones not favoured by BIC)')
            rows.append(f'<span style="{dim}">Kinematics:</span> {kin}')
            comp = f'<b>{ncomp}</b> Gaussian component{"s" if ncomp != 1 else ""}'
            if g == P['anchor'] and not forced and ncomp > 1 and np.isfinite(dbic):
                comp += f' <span style="{dim}">(ΔBIC {dbic:.1f} vs 1)</span>'
            rows.append(comp)
            for k in range(1, ncomp + 1):
                ccol = COMP_COLORS[(k - 1) % len(COMP_COLORS)] if ncomp > 1 else col
                frac = _num(r.get(f'{G}_C{k}_MFRAC' if forced else f'{G}_C{k}_FRAC'))
                if forced:      # template (model) kinematics: not a measurement of this group
                    kv = (f'template v {_fkms(r.get(f"{G}_C{k}_MV"), signed=True)} · '
                          f'σ<sub>int</sub> {_fkms(r.get(f"{G}_C{k}_MSIG"))} km/s')
                else:
                    kv = (f'v {_fkms(r.get(f"{G}_C{k}_V"), r.get(f"{G}_C{k}_VERR"), True)} · '
                          f'σ<sub>int</sub> {_fkms(r.get(f"{G}_C{k}_SIG"), r.get(f"{G}_C{k}_SIGERR"))} km/s')
                    if _num(r.get(f'{G}_C{k}_UNRES')) == 1:
                        kv += f' <span style="{dim}">(unresolved)</span>'
                rows.append(f'<span style="color:{ccol}; font-weight:bold">C{k}</span> {kv}')
                rows.append(f'&nbsp;&nbsp;&nbsp;&nbsp;<span style="{dim}">{LINE_LABELS.get(main, main)}</span> '
                            f'{_fmt(r.get(f"{M}_C{k}_FLUX"))} ± {_fmt(r.get(f"{M}_C{k}_FLUX_ERR"))}'
                            + (f' <span style="{dim}">({100 * frac:.0f} %)</span>' if np.isfinite(frac) else ''))
            if forced:
                rows.append(f'<span style="{dim}">v50, W80: not measured (fixed kinematics)</span>')
            elif not np.isfinite(_num(r.get(f'{G}_V50'))):
                rows.append(f'<span style="{dim}">v50, W80: not published (S/N below '
                            f'{cfg.kin_snr_min if cfg is not None else 5:g})</span>')
            else:
                rows.append(f'<span style="{dim}">v50</span> {_fkms(r.get(f"{G}_V50"), signed=True)} &nbsp; '
                            f'<span style="{dim}">W80</span> {_fkms(r.get(f"{G}_W80"))} km/s')
            cgs = _num(r.get(f'{M}_FLUX_CGS'))
            rows.append(f'<span style="{dim}">{LINE_LABELS.get(main, main)} flux</span> '
                        f'{_fmt(r.get(f"{M}_FLUX"))} ± {_fmt(r.get(f"{M}_FLUX_ERR"))} '
                        f'<span style="{dim}">{unit}·µm</span>'
                        + (f' <span style="{dim}">≈ {_fmt(cgs)} × 10⁻¹⁷ erg/s/cm²</span>' if np.isfinite(cgs) else ''))
            sn = [f'<span style="color:{LINE_COLORS.get(k, C.TEXT)}">{LINE_LABELS.get(k, k)}</span> '
                  f'{_fmt(r.get(f"{_ltag(k)}_SNR"))}' for k in grp['lines']
                  if np.isfinite(_num(r.get(f'{_ltag(k)}_SNR')))]
            if sn:
                rows.append(f'<span style="{dim}">S/N</span> ' + ' · '.join(sn))
            npix = int(np.count_nonzero(res.used)) if res.used is not None else 0
            rows.append(f'<span style="{dim}">χ²ν</span> {_fmt(r.get(f"{G}_CHI2NU"))}'
                        + (f' <span style="{dim}">({npix} pixels)</span>' if npix else ''))
            q = int(_num(r.get(f'{G}_QFLAG'), 0))
            flags = [txt for bit, txt in _QFLAGS if q & bit]
            if flags:
                nclip = int(_num(r.get(f'{G}_NCLIP'), 0))
                flags = [f'{t} ({nclip})' if t == 'pixels clipped' else t for t in flags]
                rows.append(f'<span style="color:{C.AMBER}">flags: {", ".join(flags)}</span>')
            es = P['err_scale'].get(res.band, 1.0)
            if abs(es - 1.0) > 0.001:
                rows.append(f'<span style="{dim}">errors × {es:.2f} in {res.band} (null tests)</span>')
        sep = (self.state.sep_fits.get(P.get('token')) or {}).get(res.bin_id) if g == 'ha' else None
        if sep:                                # Halpha and [NII] with separate kinematics
            def kin(tag):
                return (f'v50 {_fmt(sep.get(f"{tag}_V50"))} ± {_fmt(sep.get(f"{tag}_VERR"))}, '
                        f'σ {_fmt(sep.get(f"{tag}_SIG"))} km/s, S/N {_fmt(sep.get(f"{tag}_SNR"))}')
            rows.append(f'<span style="{dim}">separate fit:</span> '
                        f'<span style="color:{LINE_COLORS.get("Ha", C.TEXT)}">Hα</span> {kin("HA")}')
            rows.append(f'<span style="color:{LINE_COLORS.get("NII6583", C.TEXT)}">[NII]</span> {kin("NII")}'
                        + (f' · <b>Δv {_fmt(sep.get("DV_NII_HA"))} ± {_fmt(sep.get("DV_NII_HA_ERR"))} km/s</b>'
                           if np.isfinite(_num(sep.get('DV_NII_HA'))) else ''))
        if res.note:
            rows.append(f'<span style="color:{C.AMBER}">{res.note}</span>')
        rows += self._map_values_html(ms, res.x, res.y)
        return '<br>'.join(rows)

    def _map_values_html(self, ms, x, y):
        """Values of the shown maps at (x, y) for the info card."""
        dim = f'color:{C.TEXT_DIM}'
        vals = []
        for p in self.panels:
            n = p.name()
            if n and n in ms.maps and 0 <= y < ms.maps[n].shape[0] and 0 <= x < ms.maps[n].shape[1]:
                vals.append(f'<span style="{dim}">{ms.labels.get(n, n)}:</span> {_fmt(ms.maps[n][y, x])}')
        if not vals:
            return []
        return [f'<br><span style="{dim}; font-size:11px; font-weight:bold">MAPS AT ({x}, {y})</span>'] + vals

    def _info_html(self, ms, res, lf, unit):
        z = self.state.z
        dim = f'color:{C.TEXT_DIM}'
        rows = []
        title = LINE_LABELS.get(res.line, 'No line') if res.line else 'No line fit'
        col = LINE_COLORS.get(res.line, C.TEXT)
        rows.append(f'<span style="font-size:15px; font-weight:bold; color:{col}">{title}</span>'
                    f' <span style="{dim}">· {res.band}</span>')
        if z is None:
            rows.append(f'<span style="color:{C.PINK}">No redshift known – no line fit.</span>')
        elif lf is None:
            rows.append(f'<span style="{dim}">Line not covered in this spectrum.</span>')
        else:
            snr_col = C.EMERALD if lf.detected else C.TEXT
            rows.append(f'<span style="{dim}">S/N</span> <b style="color:{snr_col}; font-size:15px">'
                        f'{_fmt(lf.snr)}</b>' + ('' if lf.detected else f' <span style="{dim}">(not detected)</span>'))
            rows.append(f'<span style="{dim}">Δv</span> {_fmt(lf.dv_kms, "km/s")} &nbsp; '
                        f'<span style="{dim}">σ_int</span> {_fmt(lf.sigma_int_kms, "km/s")}')
            rows.append(f'<span style="{dim}">W80</span> {_fmt(lf.w80_kms, "km/s")}')
            rows.append(f'<span style="{dim}">Flux</span> {_fmt(lf.flux)} ± {_fmt(lf.flux_err)} '
                        f'<span style="{dim}">{unit}·µm</span>')
            info = res.info
            if info is not None and info.anchor == res.line:
                kin = 'free fit (anchor line)'
            elif info is not None and info.anchor:
                kin = f'velocity and width tied to {LINE_LABELS.get(info.anchor, info.anchor)}'
            else:
                kin = 'no anchor line: fixed at catalogue z, σ_int = 80 km/s'
            rows.append(f'<span style="{dim}">Kinematics:</span> {kin}')
            flags = ', '.join(lf.flags) if lf.flags else 'none'
            fcol = C.AMBER if lf.flags else C.TEXT_DIM
            rows.append(f'<span style="{dim}">Flags:</span> <span style="color:{fcol}">{flags}</span>')
        if res.note:
            rows.append(f'<span style="color:{C.AMBER}">{res.note}</span>')
        rows += self._map_values_html(ms, res.x, res.y)
        return '<br>'.join(rows)

    # ================================================================== side panel
    def _update_histogram(self):
        p = self.hist_plot
        p.clear()
        ms = self._mapset
        arr = self._shown[0]
        name = self.panels[0].name()
        if ms is None or arr is None or name is None:
            p.setLabel('bottom', '')
            self.hist_label.setText('')
            return
        v = arr[np.isfinite(arr)]
        if v.size == 0:
            p.setLabel('bottom', ms.labels.get(name, name), color=C.TEXT_DIM)
            self.hist_label.setText('no finite values')
            return
        if ms.style(name) == 'integer':
            edges = np.arange(np.floor(v.min()) - 0.5, np.ceil(v.max()) + 1.5)
            if edges.size > MAX_CATEGORIES + 1:
                edges = np.linspace(v.min(), v.max(), 30)
        else:
            lo, hi = robust_levels(v, 0.5, 99.5)
            edges = np.linspace(lo, hi, 26)
            v = np.clip(v, lo, hi)
        counts, edges = np.histogram(v, edges)
        log = self.log_chk.isChecked()
        h = np.log10(1 + counts) if log else counts.astype(float)
        col = pg.mkColor(theme.CATEGORY_COLORS[0])
        bars = pg.BarGraphItem(x0=edges[:-1], x1=edges[1:], height=h,
                               brush=pg.mkBrush(*col.getRgb()[:3], 150), pen=pg.mkPen(C.BG_DARK, width=1))
        p.addItem(bars)
        left = p.getAxis('left')
        if log:
            top = max(int(counts.max()), 1)
            ticks = [(np.log10(1 + n), str(n)) for n in (0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
                     if n <= top]
            left.setTicks([ticks])
        else:
            left.setTicks(None)
        p.setLabel('bottom', ms.labels.get(name, name), color=C.TEXT_DIM, **{'font-size': '9pt'})
        p.getPlotItem().enableAutoRange()
        med = float(np.median(arr[np.isfinite(arr)]))
        self.hist_label.setText(f'{int(np.isfinite(arr).sum())} spaxels · median {_fmt(med)}'
                                + (f' · S/N ≥ {self.snr_spin.value():g}' if self.snr_spin.value() > 0
                                   and ms.snr_name(name) and ms.style(name) != 'integer' else ''))

    def _update_resid_hist(self, chi, none_msg='no line fit at this spaxel'):
        p = self.rhist_plot
        p.clear()
        if chi is None or chi.size < 3:
            self.rhist_label.setText(none_msg if chi is None else 'too few fitted pixels')
            self.rhist_label.setToolTip('')
            return
        lim = 5.0
        edges = np.arange(-lim, lim + 0.001, 0.5)
        counts, _ = np.histogram(np.clip(chi, -lim + 1e-3, lim - 1e-3), edges)   # outliers -> edge bins
        col = pg.mkColor(C.PURPLE)
        p.addItem(pg.BarGraphItem(x0=edges[:-1], x1=edges[1:], height=counts,
                                  brush=pg.mkBrush(*col.getRgb()[:3], 150), pen=pg.mkPen(C.BG_DARK, width=1)))
        xs = np.linspace(-lim, lim, 200)
        gauss = chi.size * 0.5 * np.exp(-0.5 * xs ** 2) / np.sqrt(2 * np.pi)
        p.plot(xs, gauss, pen=pg.mkPen(C.AMBER, width=2))
        p.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(C.TEXT_DIM, width=1, style=Qt.DotLine)))
        p.setXRange(-lim, lim, padding=0.03)
        p.getPlotItem().enableAutoRange(axis='y')
        # statistics of exactly what the bars show: outliers beyond ±5 sit in the edge bins and are
        # counted separately, so one bad pixel can not turn the std into nonsense
        inside = np.abs(chi) < lim
        ci = chi[inside]
        n_out = int((~inside).sum())
        txt = (f'{chi.size} fitted pixels · mean {ci.mean():+.2f} · std {ci.std():.2f}'
               if ci.size else f'{chi.size} fitted pixels, all beyond ±{lim:g}')
        note = 'amber: unit Gaussian (ideal fit and noise)'
        if n_out:
            note = f'{n_out} beyond ±{lim:g} (in the edge bins) · amber: unit Gaussian'
        self.rhist_label.setText(f'{txt}\n{note}')
        self.rhist_label.setToolTip('(data − model)/σ of the pixels used by the Gaussian fit (within '
                                    '2 × the search window of the line and its tied companions, 1-px '
                                    'spikes excluded).  Mean ≈ 0 and std ≈ 1 mean that model and noise '
                                    'estimate describe the data; std > 1: noise underestimated or the line '
                                    'is not a single Gaussian.')
