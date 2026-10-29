"""Radial profiles tab - line fluxes, kinematics and ratios against the distance from the centre.

Two sources on one radius axis (px, arcsec, kpc or R_e - as the Spectrum tab's aperture):
  * Voronoi bins of the galaxy's pipeline product: every bin at the distance of its centroid (XBAR,
    YBAR), with a running median - local values: surface brightness, S/N, v50, W80, sigma, line ratios,
    EW, n_e, metallicity, and Halpha / [NII] fitted separately once computed (here or in the Maps tab)
  * rings fitted from the cubes (analysis.ring_profile): each ring fitted freely (its own kinematics)
    or, without an anchor line of its own, forced to the central aperture (fluxes only); optionally
    velocity-registered with the Voronoi velocity field, so ring W80 / sigma are dispersions without the
    rotation (as the annular stacks of kmos_best3_single_gauss.ipynb)
The centre is the aperture centre of the Spectrum tab ('Centre on Hα' there); grey: radii inside the
half width of the seeing (image quality of the cube), dotted: 1, 2, 3 R_e.
"""
import csv
import os
import time

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QLocale, Qt, QTimer
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel,
                             QProgressBar, QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from empower_explorer import analysis
from empower_explorer.lines import LINE_LABELS, REST
from empower_explorer.theme import C
from empower_explorer.views.spectrum_tab import RADIUS_UNITS, Segmented, _dim, _flat_button, _section
from empower_explorer.widgets.plots import EmptyState, make_plot
from empower_explorer.workers import run_async

try:
    from empower_explorer.pipeline import voronoi_binning as vb
except Exception:                       # pragma: no cover - the tab then shows rings only
    vb = None

N_PANELS = 6
MAX_RINGS = 30
UNIT_NAME = {'px': 'px', 'arcsec': '″', 'kpc': 'kpc', 're': 'Rₑ'}
# default ring width and outer radius per unit
UNIT_DEFAULTS = {'px': (1.0, 8.0), 'arcsec': (0.2, 1.6), 'kpc': (1.0, 10.0), 're': (0.5, 3.0)}
BIN_COLOR, MEDIAN_COLOR, RING_COLOR = C.TEXT_DIM, C.ACCENT, C.AMBER


def _f(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return np.nan
    return v


def _sb(flux_key, err_key=None):
    """Surface brightness [flux / arcsec²] as log10 (value, error) of a bin (row, area) or a ring."""
    def bins(row, ctx):
        a = _f(row.get('NPIX')) * ctx['pix_a'] ** 2
        f = sum(_f(row.get(k)) for k in flux_key)
        e = np.sqrt(sum(_f(row.get(k)) ** 2 for k in err_key)) if err_key else np.nan
        return _log(f / a, e / a)

    def rings(rec, ctx):
        keys = ctx['lines'][flux_key]
        band = rec.get(f'{keys[0]}_band', '')
        n, pix = (rec.get('npix_b') or {}).get(band, 0), (rec.get('pix_b') or {}).get(band, np.nan)
        area = (pix ** 2 if rec.get('method') == 'mean' else n * pix ** 2) if n else np.nan   # 'mean': per spaxel
        f = sum(_f(rec.get(f'{k}_flux')) for k in keys)
        e = np.sqrt(sum(_f(rec.get(f'{k}_err')) ** 2 for k in keys))
        return _log(f / area, e / area) if np.isfinite(area) and area > 0 else (np.nan, np.nan)
    return bins, rings


def _log(v, e):
    if not (np.isfinite(v) and v > 0):
        return np.nan, np.nan
    return float(np.log10(v)), float(0.4343 * e / v) if np.isfinite(e) else np.nan


def _col(name, err=None, absolute=False):
    def bins(row, ctx):
        v = _f(row.get(name.replace('{G}', ctx['G'])))
        e = _f(row.get(err.replace('{G}', ctx['G']))) if err else np.nan
        return (abs(v) if absolute else v), e
    return bins


def _ring(key, err=None):
    def rings(rec, ctx):
        return _f(rec.get(key)), (_f(rec.get(err)) if err else np.nan)
    return rings


def _sep(key, err=None):
    def bins(row, ctx):
        s = (ctx['sep'] or {}).get(int(row['BIN'])) or {}
        return _f(s.get(key)), (_f(s.get(err)) if err else np.nan)

    def rings(rec, ctx):
        s = rec.get('sep') or {}
        return _f(s.get(key)), (_f(s.get(err)) if err else np.nan)
    return bins, rings


def _oh_ring(rec, ctx):
    n2 = _f(rec.get('n2'))
    return (vb.oh_m13(n2) if vb is not None else np.nan), 0.462 * _f(rec.get('n2_err'))


# quantity key -> (label, unit text, bins(row, ctx) or None, rings(rec, ctx) or None, tooltip)
_SB_HA = _sb(('HA_FLUX',), ('HA_FLUX_ERR',))
_SB_NII = _sb(('NII6583_FLUX',), ('NII6583_FLUX_ERR',))
_SB_OIII = _sb(('OIII5007_FLUX',), ('OIII5007_FLUX_ERR',))
_SB_HB = _sb(('HB_FLUX',), ('HB_FLUX_ERR',))
_SEP_DV = _sep('DV_NII_HA', 'DV_NII_HA_ERR')
_SEP_W80_HA = _sep('HA_W80')
_SEP_W80_NII = _sep('NII_W80')
QUANTITIES = {
    'sb_ha': ('log Hα surface brightness', 'flux/″²', _SB_HA[0], _SB_HA[1], 'Hα flux per arcsec² (log10)'),
    'sb_nii': ('log [NII]6583 surface brightness', 'flux/″²', _SB_NII[0], _SB_NII[1], '[NII]6583 flux per arcsec² (log10)'),
    'sb_oiii': ('log [OIII]5007 surface brightness', 'flux/″²', _SB_OIII[0], _SB_OIII[1],
                '[OIII]5007 flux per arcsec² (log10)'),
    'sb_hb': ('log Hβ surface brightness', 'flux/″²', _SB_HB[0], _SB_HB[1], 'Hβ flux per arcsec² (log10)'),
    'snr_ha': ('Hα S/N', '', _col('HA_SNR'), _ring('Ha_snr'), 'S/N of Hα in the bin / ring'),
    'v50': ('v50 (anchor group)', 'km/s', _col('{G}_V50'), None,
            'Median velocity of the anchor group (Hα+[NII]) per bin, w.r.t. Z_REF; rings: not meaningful '
            '(a ring averages over all azimuths)'),
    'abs_v50': ('|v50| (anchor group)', 'km/s', _col('{G}_V50', absolute=True), None,
                'Absolute velocity per bin: a rotation curve without deprojection'),
    'w80': ('W80 (anchor group)', 'km/s', _col('{G}_W80'), _ring('w80'),
            'Line width containing 80 % of the flux; rings include the rotation unless velocity-registered'),
    'sigma': ('σ_int (= W80 / 2.563)', 'km/s', lambda row, ctx: (_f(row.get(f'{ctx["G"]}_W80')) / 2.5631, np.nan),
              lambda rec, ctx: (_f(rec.get('w80')) / 2.5631, np.nan),
              'Intrinsic dispersion of the whole line profile (LSF removed), W80 / 2.563 - for one Gaussian its σ; '
              'bins with 2-3 components are not reduced to their first (narrow) component'),
    'n2': ('log [NII]/Hα', '', _col('LOG_N2', 'LOG_N2_ERR'), _ring('n2', 'n2_err'), 'N2 index (both lines S/N ≥ 3)'),
    's2': ('log [SII]/Hα', '', _col('LOG_S2', 'LOG_S2_ERR'), _ring('s2', 's2_err'), '([SII]6716 + 6731) / Hα'),
    'o3': ('log [OIII]/Hβ', '', _col('LOG_O3', 'LOG_O3_ERR'), _ring('o3', 'o3_err'), '[OIII]5007 / Hβ'),
    'oh': ('12 + log(O/H) (N2, M13)', '', _col('OH_N2_M13'), _oh_ring, 'Marino+13 N2 calibration'),
    'ew': ('EW(Hα)', 'Å', _col('EW_HA', 'EW_HA_ERR'), _ring('ew', 'ew_err'),
           'Rest-frame equivalent width of Hα (line and continuum S/N ≥ 3)'),
    'ne': ('n_e [SII]', 'cm⁻³', _col('NE_SII'), None, 'Electron density from [SII]6716/6731 (Sanders+16)'),
    'sep_dv': ('Δv [NII] − Hα (separate fit)', 'km/s', _SEP_DV[0], _SEP_DV[1],
               'Velocity of [NII] minus Hα from the fit with separate kinematics'),
    'sep_w80_ha': ('Hα W80 (separate fit)', 'km/s', _SEP_W80_HA[0], _SEP_W80_HA[1], 'Hα W80 with its own kinematics'),
    'sep_w80_nii': ('[NII] W80 (separate fit)', 'km/s', _SEP_W80_NII[0], _SEP_W80_NII[1],
                    '[NII] W80 with its own kinematics'),
}
DEFAULTS = ['sb_ha', 'w80', 'sigma', 'n2', 'ew', 'sep_dv']
RING_LINES = {('HA_FLUX',): ['Ha'], ('NII6583_FLUX',): ['NII6583'], ('OIII5007_FLUX',): ['OIII5007'],
              ('HB_FLUX',): ['Hb']}


def _product_job(path, bands):
    """Worker: the Voronoi product and its velocity field on the grid of every band (registration)."""
    p = vb.read_product(path)
    return dict(p=p, vmaps={b: vb.velocity_field(p, b) for b in bands})


class _Panel(QWidget):
    """Quantity chooser above a profile plot."""

    def __init__(self, key, changed, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.combo = QComboBox()
        for k, q in QUANTITIES.items():
            self.combo.addItem(q[0], k)
            self.combo.setItemData(self.combo.count() - 1, q[4], Qt.ToolTipRole)
        self.combo.setCurrentIndex(max(self.combo.findData(key), 0))
        self.combo.currentIndexChanged.connect(lambda _i: changed())
        lay.addWidget(self.combo)
        self.pw = make_plot('', 'radius', '')
        self.pw.setMinimumHeight(170)
        self.plot = self.pw.getPlotItem()
        self.plot.hideButtons()
        lay.addWidget(self.pw, 1)

    def key(self):
        return self.combo.currentData()


class RadialTab(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._unit_choice = 're'
        self._width = dict((u, d[0]) for u, d in UNIT_DEFAULTS.items())
        self._rmax = dict((u, d[1]) for u, d in UNIT_DEFAULTS.items())
        self._prod = {}                 # product token -> {'p', 'vmaps'} or {'error'}
        self._prod_jobs = set()
        self._rings = {}                # ring key -> (rings, centre FitInfo)
        self._ring_job = None           # key of the running ring job
        self._ring_wanted = None
        self._ring_error = None         # (key, message) of the last failed ring job
        self._dirty = True
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)    # an aperture drag in the Spectrum tab: compute once it stops
        self._timer.timeout.connect(self._refresh)
        for sig in (state.stacksReady, state.redshiftChanged):     # (a new reduction ends with stacksReady)
            sig.connect(lambda *_: self._changed())
        state.apertureChanged.connect(lambda: self._changed(delay=True))
        state.sepFitsChanged.connect(lambda _t: self._changed())
        state.productsChanged.connect(lambda k: self._changed() if k == state.key else None)
        state.loadFailed.connect(lambda k: self._show_empty(f'The cubes of {k} could not be loaded',
                                                            'See the error message; select the galaxy again to retry.'))
        state.optionsChanged.connect(lambda n: self._changed() if n in ('method', 'subtract_bg') else None)
        state.galaxyChanged.connect(lambda _k: self._show_empty(f'Loading {_k} …'))

    # ================================================================== UI
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        self.pages = QStackedWidget()
        self.empty = EmptyState('No galaxy loaded', 'Select a galaxy in the list.')
        self.pages.addWidget(self.empty)
        body = QWidget()
        lay = QVBoxLayout(body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(_section('Radial profiles'))
        self.unit_seg = Segmented(RADIUS_UNITS)
        self.unit_seg.changed.connect(self._unit_selected)
        row.addWidget(self.unit_seg)
        row.addWidget(_dim('Ring width'))
        self.width_spin = self._spin('Width of the rings fitted from the cubes', self._ring_params)
        row.addWidget(self.width_spin)
        row.addWidget(_dim('out to'))
        self.rmax_spin = self._spin('Outer radius of the last ring', self._ring_params)
        row.addWidget(self.rmax_spin)
        self.reg_chk = QCheckBox('Velocity-registered')
        self.reg_chk.setChecked(True)
        self.reg_chk.setToolTip('Shift every spaxel to rest with the Voronoi velocity field before the rings are '
                                'summed:\nthe rotation drops out, so ring W80 / σ are dispersions (needs a Voronoi fit)')
        self.reg_chk.toggled.connect(lambda _on: self._changed())
        row.addWidget(self.reg_chk)
        self.ring_progress = QProgressBar()
        self.ring_progress.setFixedWidth(90)
        self.ring_progress.setFixedHeight(8)
        self.ring_progress.setTextVisible(False)
        self.ring_progress.hide()
        row.addWidget(self.ring_progress)
        row.addStretch(1)
        self.show_bins = QCheckBox('Bins')
        self.show_rings = QCheckBox('Rings')
        self.show_median = QCheckBox('Median of bins')
        for cb, tip in ((self.show_bins, 'Voronoi bins of the pipeline product (grey points)'),
                        (self.show_rings, 'Rings fitted from the cubes (amber; open: forced or < 60 % on the field)'),
                        (self.show_median, 'Running median of the bins in the rings\' radius steps (cyan)')):
            cb.setChecked(True)
            cb.setToolTip(tip)
            cb.toggled.connect(lambda _on: self._draw())
            row.addWidget(cb)
        self.sep_btn = QPushButton('Hα / [NII] separately')
        self.sep_btn.setToolTip('Fit Hα and [NII] of every Voronoi bin with separate kinematics (also offered in '
                                'the Maps tab); the rings always get it')
        self.sep_btn.clicked.connect(self._compute_separate)
        row.addWidget(self.sep_btn)
        self.export_btn = _flat_button('Export CSV…', 'Save the bins and rings of the shown galaxy as a table')
        self.export_btn.clicked.connect(self._export)
        row.addWidget(self.export_btn)
        lay.addLayout(row)
        self.info = _dim('', shrink=True)
        self.info.setWordWrap(False)
        lay.addWidget(self.info)
        grid = QGridLayout()
        grid.setSpacing(10)
        self.panels = []
        for i in range(N_PANELS):
            p = _Panel(DEFAULTS[i], self._draw)
            grid.addWidget(p, i // 3, i % 3)
            self.panels.append(p)
        lay.addLayout(grid, 1)
        self.pages.addWidget(body)
        outer.addWidget(self.pages)
        self.unit_seg.set_current(self._unit_choice)

    def _spin(self, tip, slot):
        sp = QDoubleSpinBox()
        sp.setLocale(QLocale.c())
        sp.setDecimals(2)
        sp.setKeyboardTracking(False)
        sp.setRange(0.01, 999)
        sp.setToolTip(tip)
        sp.setMinimumWidth(78)
        sp.valueChanged.connect(slot)
        return sp

    def _show_empty(self, text, hint=''):
        self.empty.set_text(text, hint)
        self.pages.setCurrentIndex(0)

    # ================================================================== units
    def _pix(self):
        s = self.state
        return s.stacks[s.ref_band].pixscale if s.ref_band in s.stacks else 0.2

    def _px_per_unit(self, unit):
        s = self.state
        if unit == 'arcsec':
            return 1.0 / self._pix()
        if unit == 'kpc':
            k = analysis.kpc_per_arcsec(s.z)
            return 1.0 / (self._pix() * k) if np.isfinite(k) and k > 0 else np.nan
        if unit == 're':
            re, _ = s.effective_radius() if s.stacks else (np.nan, '')
            return re / self._pix() if np.isfinite(re) and re > 0 else np.nan
        return 1.0

    def _unit(self):
        u = self._unit_choice
        return u if np.isfinite(self._px_per_unit(u)) else ('arcsec' if u != 'px' else 'px')

    def _unit_selected(self, key):
        self._unit_choice = key
        self._changed()

    def _ring_params(self, _v=None):
        if getattr(self, '_syncing', False):
            return
        u = self._unit()
        self._width[u], self._rmax[u] = self.width_spin.value(), self.rmax_spin.value()
        self._changed()

    def _sync_controls(self):
        s = self.state
        kpc_ok = np.isfinite(self._px_per_unit('kpc'))
        re_ok = np.isfinite(self._px_per_unit('re'))
        tips = {k: t for k, _l, t in RADIUS_UNITS}
        self.unit_seg.set_item_enabled('kpc', bool(kpc_ok), tips['kpc'] if kpc_ok else 'kpc: needs a redshift')
        self.unit_seg.set_item_enabled('re', bool(re_ok), tips['re'] if re_ok else 'Rₑ: not known for this galaxy')
        u = self._unit()
        self.unit_seg.set_current(u)
        self._syncing = True
        try:
            for sp, val in ((self.width_spin, self._width[u]), (self.rmax_spin, self._rmax[u])):
                sp.setSuffix(f' {UNIT_NAME[u]}' if u != 'arcsec' else '″')
                sp.setSingleStep({'px': 0.5, 'arcsec': 0.1, 'kpc': 0.5, 're': 0.25}[u])
                sp.setValue(val)
        finally:
            self._syncing = False
        ok, why = self._registration_possible()
        self.reg_chk.setEnabled(ok)
        self.reg_chk.setToolTip(self.reg_chk.toolTip().split('\n(')[0] + ('' if ok else f'\n({why})'))
        self._update_sep_button()

    def _registration_possible(self):
        """(ok, reason): the rings can be velocity-registered with the Voronoi velocity field."""
        s = self.state
        _prod, data = self._prod_data()
        if data is None:
            return False, 'no Voronoi fit read: rings are not registered'
        if not any(v is not None and np.any(np.isfinite(v) & (v != 0)) for v in data['vmaps'].values()):
            return False, 'the Voronoi fit has no velocity field (anchor line not detected): rings are not registered'
        if s.redshift is not None and s.redshift.override:
            return False, 'the redshift was entered by hand: rings are fitted at it, not registered to Z_REF'
        return True, ''

    # ================================================================== data
    def _product(self):
        """The pipeline product (Voronoi fit) of the current galaxy and reduction, or None."""
        for p in self.state.products():
            if p.is_pipeline:
                return p
        return None

    @staticmethod
    def _token(prod):
        return os.path.normcase(os.path.abspath(prod.path)), str(prod.meta.get('DATE', ''))

    def _changed(self, delay=False):
        if not self.isVisible():
            self._dirty = True
            return
        if delay:
            self._timer.start()
        else:
            self._refresh()

    def showEvent(self, ev):
        super().showEvent(ev)
        if self._dirty:
            self._refresh()

    def _refresh(self):
        self._dirty = False
        s = self.state
        if not s.stacks or s.ref_band not in s.stacks or s.ap_x is None:
            if s.key is None:
                self._show_empty('No galaxy loaded', 'Select a galaxy in the list.')
            return
        self.pages.setCurrentIndex(1)
        self._sync_controls()
        prod = self._product()
        if prod is not None and vb is not None:
            tok = self._token(prod)
            old = self._prod.get(tok)
            if old is not None and 'error' in old and time.time() - old.get('t', 0) > 30:
                self._prod.pop(tok)                       # a failed read (e.g. a OneDrive lock): try again
            if tok not in self._prod and tok not in self._prod_jobs:
                self._prod_jobs.add(tok)
                run_async(_product_job, prod.path, list(s.stacks),
                          on_done=lambda res, t=tok: self._product_done(t, res),
                          on_error=lambda msg, t=tok: self._product_done(
                              t, {'error': (msg.strip().splitlines() or ['?'])[-1], 't': time.time()}))
        self._request_rings()
        self._draw()

    def _product_done(self, tok, res):
        self._prod_jobs.discard(tok)
        self._prod[tok] = res
        while len(self._prod) > 12:
            self._prod.pop(next(iter(self._prod)))
        self._changed()

    def _prod_data(self):
        prod = self._product()
        if prod is None:
            return None, None
        d = self._prod.get(self._token(prod))
        return prod, (d if d and 'error' not in d else None)

    def _edges(self):
        u = self._unit()
        k = self._px_per_unit(u)
        w, r = self._width[u] * k, self._rmax[u] * k
        if not (np.isfinite(w) and w > 0 and np.isfinite(r) and r > 0):
            return None
        n = int(min(np.ceil(r / w - 1e-9), MAX_RINGS))
        return tuple(round(float(i * w), 6) for i in range(n + 1))

    def _anchor_line(self, data):
        """The line whose free fit gives a ring its kinematics: the Voronoi fit's anchor (as the bins), else
        the first of Halpha, [OIII]5007, Hbeta in the loaded bands."""
        s = self.state
        if data is not None and vb is not None:
            g = str(data['p']['header'].get('ANCHOR') or 'ha').strip()
            if g in vb.GROUPS:
                return vb.GROUPS[g]['main']
        for line in ('Ha', 'OIII5007', 'Hb'):
            if s.z and analysis.pick_band(s.stacks, REST[line] * (1 + s.z)) is not None:
                return line
        return 'Ha'

    def _ring_key(self):
        """Everything the rings depend on; [5] registered, [8] fitting z (Z_REF of the Voronoi fit when the
        spaxels are shifted to it), [11] anchor line."""
        s = self.state
        edges = self._edges()
        prod, data = self._prod_data()
        reg = bool(self.reg_chk.isChecked() and self._registration_possible()[0])
        z_fit = s.z
        if reg:
            zr = _f(data['p']['header'].get('Z_REF'))
            z_fit = zr if np.isfinite(zr) else s.z
        fallback = None               # no anchor in the centre: force to the Voronoi fit's reference, not z_cat
        if data is not None and z_fit is not None:
            h = data['p']['header']
            zr, sr = _f(h.get('Z_REF')), _f(h.get('SIG_REF'))
            if np.isfinite(zr) and np.isfinite(sr):
                fallback = (round(((1 + zr) / (1 + z_fit) - 1) * 299792.458, 3), round(sr, 3),
                            'Voronoi reference (Z_REF, SIG_REF)')
        return (s.key, s.ref_band, round(s.ap_x, 2), round(s.ap_y, 2), edges, reg, s.options.get('method'),
                bool(s.options.get('subtract_bg')), None if z_fit is None else round(z_fit, 6),
                self._token(prod) if prod is not None else None, None if s.z is None else round(s.z, 6),
                self._anchor_line(data), fallback, s.variant, id(s.stacks))

    def _request_rings(self):
        s = self.state
        if s.z is None or self._edges() is None or not s.stacks or s.ref_band not in s.stacks or s.ap_x is None:
            return
        key = self._ring_key()
        if key in self._rings:
            return
        self._ring_wanted = key
        if self._ring_job is not None:
            return                                # started when the running job finishes
        _prod, data = self._prod_data()
        vmaps = data['vmaps'] if key[5] and data is not None else None
        z = key[8]
        sep = ((lambda w, f, e, b, start, z=z: vb.fit_hanii_separate(w, f, e, z, b, start=start))
               if vb is not None else None)
        self._ring_job = key
        self.ring_progress.setVisible(True)
        self.ring_progress.setRange(0, 0)
        run_async(analysis.ring_profile, dict(s.stacks), s.ref_band, s.ap_x, s.ap_y, list(key[4]), z,
                  s.options.get('method', 'sum'), bool(s.options.get('subtract_bg')), vmaps, sep, key[11], key[12],
                  on_done=lambda res, k=key: self._rings_done(k, res, None),
                  on_error=lambda msg, k=key: self._rings_done(k, None, msg))

    def _rings_done(self, key, res, err):
        self._ring_job = None
        self.ring_progress.hide()
        if err:                       # not cached: the next change tries again
            print(f'Radial profiles: rings failed:\n{err}')
            self.state.status.emit('Radial profiles: the rings could not be fitted (see console)', 6000)
            self._ring_error = (key, (err.strip().splitlines() or ['?'])[-1])
        else:
            self._rings[key] = res
            while len(self._rings) > 20:
                self._rings.pop(next(iter(self._rings)))
        if self._ring_wanted is not None and self._ring_wanted != key and self._ring_wanted not in self._rings:
            self._request_rings()
        if self.isVisible():
            self._draw()
        else:
            self._dirty = True        # drawn when the tab is shown again

    # ================================================================== separate Halpha / [NII]
    def _update_sep_button(self):
        prod, data = self._prod_data()
        tok = self._token(prod) if prod is not None else None
        done = tok in self.state.sep_fits
        running = tok in self.state.sep_running
        ok, why = self._sep_possible(data)
        self.sep_btn.setEnabled(ok and not done and not running)
        self.sep_btn.setText('Fitting Hα / [NII] …' if running else
                             ('Hα / [NII] separately ✓' if done else 'Hα / [NII] separately'))
        self.sep_btn.setToolTip('Fit Hα and [NII] of every Voronoi bin with separate kinematics (also in the Maps '
                                'tab); the rings always get it' if ok else f'Not possible: {why}')

    def _sep_possible(self, data):
        if vb is None or data is None:
            return False, 'needs a Voronoi fit of this galaxy'
        bins = data['p']['bins']
        bands = set() if bins is None or 'HANII_BAND' not in bins else \
            {str(b).strip() for b in bins['HANII_BAND']} - {'', 'nan', 'None'}
        if not bands:
            return False, 'Hα+[NII] was not fitted in this Voronoi fit'
        if not bands <= set(self.state.stacks):
            return False, f'band {"/".join(sorted(bands))} of Hα+[NII] is not loaded'
        return True, ''

    def _compute_separate(self):
        prod, data = self._prod_data()
        if data is None:
            return
        if not self._sep_possible(data)[0]:
            return
        self.state.start_separate_fit(self._token(prod), self.state.key, vb.hanii_separate, data['p'],
                                      dict(self.state.stacks))

    # ================================================================== drawing
    def _bins_xy(self, data):
        """(radius [px of the reference band], rows) of the Voronoi bins around the aperture centre."""
        s = self.state
        p = data['p']
        aband = str(p['header'].get('ANCHBAND', '')).strip()
        if aband not in s.stacks or p['bins'] is None or p['bins'].empty:
            return None, []
        ast = s.stacks[aband]
        cx, cy, _ = analysis.transfer(s.stacks[s.ref_band], ast, s.ap_x, s.ap_y)
        rows = p['bins'].to_dict('records')
        r = np.array([np.hypot(_f(row.get('XBAR')) - cx, _f(row.get('YBAR')) - cy) for row in rows])
        return r * ast.pixscale / self._pix(), rows

    def _psf_hwhm(self):
        """Half width of the seeing [arcsec] from the image quality keywords of the reference stack."""
        st = self.state.stacks.get(self.state.ref_band)
        h = getattr(st, 'header0', None) or {}
        vals = [_f(h.get(k)) for k in ('HIERARCH ESO TEL IA FWHM', 'ESO TEL IA FWHM', 'HIERARCH ESO TEL AMBI FWHM END',
                                       'ESO TEL AMBI FWHM END')]
        vals = [v for v in vals if np.isfinite(v) and v > 0]
        if not vals:
            return np.nan
        lam = float(np.nanmedian(st.wave)) if st is not None else 1.0      # micron
        return vals[0] * (lam / 0.5) ** -0.2 / 2           # an estimate: one OB's optical seeing, not the cube PSF

    def _draw(self):
        s = self.state
        if not self.isVisible() or self.pages.currentIndex() != 1 or not s.stacks or s.ref_band not in s.stacks:
            return
        u = self._unit()
        k = self._px_per_unit(u)
        prod, data = self._prod_data()
        rings, centre, bg_r = self._rings.get(self._ring_key(), (None, None, {}))
        G = 'HANII'
        if data is not None and vb is not None:
            G = vb.group_tag(str(data['p']['header'].get('ANCHOR') or 'ha').strip())
        ctx = dict(G=G, pix_a=self._pix(), lines=RING_LINES,
                   sep=self.state.sep_fits.get(self._token(prod)) if prod is not None else None)
        if data is not None:
            aband = str(data['p']['header'].get('ANCHBAND', '')).strip()
            if aband in s.stacks:
                ctx['pix_a'] = s.stacks[aband].pixscale
        r_bins, rows = self._bins_xy(data) if data is not None else (None, [])
        re, re_src = s.effective_radius()
        hwhm = self._psf_hwhm()
        edges = self._edges() or ()
        for p in self.panels:
            self._draw_panel(p, u, k, rows, r_bins, rings, ctx, re, hwhm, edges)
        self._update_info(u, prod, data, rings, centre, re, re_src, hwhm, bg_r)

    def _draw_panel(self, panel, u, k, rows, r_bins, rings, ctx, re, hwhm, edges):
        key = panel.key()
        label, unit, fb, fr, _tip = QUANTITIES[key]
        plot = panel.plot
        plot.clear()
        unit_txt = '″' if u == 'arcsec' else UNIT_NAME[u]
        plot.setLabel('bottom', f'radius [{unit_txt}]', color=C.TEXT_DIM)
        plot.setLabel('left', f'{label}' + (f' [{unit}]' if unit else ''), color=C.TEXT_DIM)
        pix = self._pix()
        if np.isfinite(hwhm):
            reg = pg.LinearRegionItem((0, hwhm / pix / k), movable=False, brush=pg.mkBrush(148, 163, 184, 35),
                                      pen=pg.mkPen(None))
            reg.setZValue(-10)
            plot.addItem(reg)
        if np.isfinite(re):
            for m in (1, 2, 3):
                x = m * re / pix / k
                line = pg.InfiniteLine(pos=x, angle=90, pen=pg.mkPen(C.TEXT_DIM, width=1, style=Qt.DotLine),
                                       label=f'{m} Rₑ', labelOpts=dict(position=0.95, color=C.TEXT_DIM))
                plot.addItem(line, ignoreBounds=True)
        anything = False
        if fb is not None and rows and r_bins is not None:
            vals = np.array([fb(row, ctx) for row in rows], float).reshape(-1, 2)
            x = r_bins / k
            ok = np.isfinite(vals[:, 0]) & np.isfinite(x)
            if self.show_bins.isChecked() and ok.any():
                tips = [f'bin {int(row["BIN"])}' for row in rows]
                sc = pg.ScatterPlotItem(x[ok], vals[ok, 0], size=6, pen=pg.mkPen(None),
                                        brush=pg.mkBrush(148, 163, 184, 150), hoverable=True,
                                        data=[t for t, o in zip(tips, ok) if o],
                                        tip=lambda x, y, data: f'{data}\nr = {x:.2f} {unit_txt}\n{label} = {y:.3g}')
                plot.addItem(sc)
                anything = True
            if self.show_median.isChecked() and ok.sum() >= 3 and len(edges) >= 2:
                xe = np.array(edges) / k
                med_x, med_y = [], []
                for a, b in zip(xe[:-1], xe[1:]):
                    sel = ok & ((x > a) | ((a <= 0) & (x >= 0))) & (x <= b)
                    if sel.sum() >= 2:
                        med_x.append(0.5 * (a + b))
                        med_y.append(float(np.median(vals[sel, 0])))
                if med_x:
                    plot.addItem(pg.PlotDataItem(med_x, med_y, pen=pg.mkPen(MEDIAN_COLOR, width=2)))
                    anything = True
        if fr is not None and rings and self.show_rings.isChecked():
            xr = np.array([0.5 * (r['r_in'] + r['r_out']) for r in rings]) / k
            vr = np.array([fr(r, ctx) for r in rings], float).reshape(-1, 2)
            good = np.array([r.get('kin') == 'free' and r.get('coverage', 0) >= 0.6 for r in rings])
            ok = np.isfinite(vr[:, 0])
            if ok.any():
                plot.addItem(pg.PlotDataItem(xr[ok], vr[ok, 0], pen=pg.mkPen(RING_COLOR, width=1.5)))
                e = np.where(np.isfinite(vr[:, 1]), vr[:, 1], 0.0)
                plot.addItem(pg.ErrorBarItem(x=xr[ok], y=vr[ok, 0], height=2 * e[ok], beam=0.0,
                                             pen=pg.mkPen(RING_COLOR, width=1)))
                for sel, brush in ((ok & good, pg.mkBrush(RING_COLOR)), (ok & ~good, pg.mkBrush(None))):
                    if sel.any():
                        plot.addItem(pg.ScatterPlotItem(
                            xr[sel], vr[sel, 0], size=8, symbol='s', pen=pg.mkPen(RING_COLOR, width=1.5),
                            brush=brush, hoverable=True,
                            data=[f'ring {r["r_in"] / k:.2f}–{r["r_out"] / k:.2f} {unit_txt} · {r["npix"]} spaxels · '
                                  + (f'{LINE_LABELS.get(r.get("anchor"), r.get("anchor"))} free'
                                     if r.get('kin') == 'free' else f'forced to the {r.get("kin")}')
                                  + ('' if r.get('coverage', 1) >= 0.6 else ' · < 60 % on the field')
                                  for r, o in zip(rings, sel) if o],
                            tip=lambda x, y, data: f'{data}\n{label} = {y:.3g}'))
                anything = True
        if not anything:
            txt = pg.TextItem('no values', color=C.TEXT_DIM, anchor=(0.5, 0.5))
            plot.addItem(txt)
            txt.setPos(0.5, 0.5)
            plot.setXRange(0, 1, padding=0)
            plot.setYRange(0, 1, padding=0)
        else:
            plot.enableAutoRange()

    def _update_info(self, u, prod, data, rings, centre, re, re_src, hwhm, bg_r=None):
        s = self.state
        parts = [f'centre x = {s.ap_x:.1f}, y = {s.ap_y:.1f} px ({s.ref_band}) = the aperture centre of the Spectrum tab']
        if np.isfinite(re):
            parts.append(f'Rₑ = {re:.2f}″ ({re_src})')
        kpc = analysis.kpc_per_arcsec(s.z)
        if np.isfinite(kpc):
            parts.append(f'1″ = {kpc:.2f} kpc')
        if np.isfinite(hwhm):
            parts.append(f'PSF HWHM ≈ {hwhm:.2f}″ (grey: guider seeing scaled to the band)')
        if prod is None:
            parts.append('no Voronoi fit: bins and registration unavailable (compute one in the Maps tab)')
        elif data is None:
            err = (self._prod.get(self._token(prod)) or {}).get('error')
            parts.append(f'the Voronoi fit could not be read ({err}) - tried again on the next change' if err
                         else 'reading the Voronoi fit …')
        if s.z is None:
            parts.append('no redshift: rings can not be fitted')
        elif rings is None:
            key = self._ring_key()
            parts.append(f'the rings could not be fitted ({self._ring_error[1]})'
                         if self._ring_error and self._ring_error[0] == key else 'fitting the rings …')
        elif rings:
            key = self._ring_key()
            reg = 'velocity-registered, fitted at Z_REF' if key[5] else 'not registered'
            names = {'free': f'with free {LINE_LABELS.get(key[11], key[11])}',
                     'centre': 'forced to the central aperture', 'reference': 'forced to the Voronoi reference '
                     '(Z_REF, SIG_REF)', 'catalogue z': 'forced to the catalogue z', '': 'without data'}
            counts = {}
            for r in rings:
                counts[r.get('kin', '')] = counts.get(r.get('kin', ''), 0) + 1
            parts.append(f'{len(rings)} rings ({reg}): ' + ', '.join(f'{n} {names.get(k_, k_)}'
                                                                    for k_, n in counts.items()))
            ok_reg, why = self._registration_possible()
            if not ok_reg and self.reg_chk.isChecked():
                parts.append(why)
            u_ = self._unit()
            want = int(np.ceil(self._rmax[u_] / self._width[u_] - 1e-9))
            if want > MAX_RINGS:
                parts.append(f'limited to {MAX_RINGS} rings: out to {MAX_RINGS * self._width[u_]:.2f} '
                             f'{UNIT_NAME[u_]} (widen the rings to reach further)')
            if s.options.get('subtract_bg') and bg_r:
                missed = [b for b, r in bg_r.items() if not np.isfinite(r)]
                done = ', '.join(f'{b} ≥ {r + 2:.1f} px' if r + 2 > 6 else f'{b} ≥ 6 px'
                                 for b, r in bg_r.items() if np.isfinite(r))
                parts.append('residual sky: one pedestal per band for all rings' + (f' ({done})' if done else '')
                             + (f'; not subtracted in {"/".join(missed)} (field too small)' if missed else ''))
        self.info.setText(' · '.join(parts))
        self.info.setToolTip('\n'.join(parts))

    # ================================================================== export
    def _export(self):
        s = self.state
        path, _ = QFileDialog.getSaveFileName(self, 'Export radial profiles',
                                              f'{(s.key or "galaxy").replace(" ", "_")}_radial_profiles.csv',
                                              'CSV (*.csv)')
        if path:
            try:
                self.write_csv(path)
            except OSError as e:          # e.g. the file is open in Excel
                self.state.error.emit('Export failed', f'{path}\n{e}')

    def write_csv(self, path):
        """Bins and rings of the shown galaxy with the radius in every available unit and every quantity."""
        s = self.state
        prod, data = self._prod_data()
        rings = self._rings.get(self._ring_key(), (None, None, {}))[0]
        G = vb.group_tag(str(data['p']['header'].get('ANCHOR') or 'ha').strip()) if data is not None and vb else 'HANII'
        ctx = dict(G=G, pix_a=self._pix(), lines=RING_LINES,
                   sep=s.sep_fits.get(self._token(prod)) if prod is not None else None)
        if data is not None:
            aband = str(data['p']['header'].get('ANCHBAND', '')).strip()
            if aband in s.stacks:
                ctx['pix_a'] = s.stacks[aband].pixscale
        units = [u for u in ('px', 'arcsec', 'kpc', 're') if np.isfinite(self._px_per_unit(u))]
        head = (['source', 'id', 'kinematics'] + [f'r_{u}' for u in units] + ['r_in_px', 'r_out_px', 'npix']
                + [f'{q}{suf}' for q in QUANTITIES for suf in ('', '_err')])
        out = []
        r_bins, rows = self._bins_xy(data) if data is not None else (None, [])
        for r, row in zip(r_bins if r_bins is not None else [], rows):
            vals = []
            for q, (_l, _u, fb, _fr, _t) in QUANTITIES.items():
                vals += list(fb(row, ctx)) if fb is not None else [np.nan, np.nan]
            out.append(['bin', int(row['BIN']), 'free' if _f(row.get(f'{G}_FORCED')) == 0 else 'forced']
                       + [r / self._px_per_unit(u) for u in units] + ['', '', int(_f(row.get('NPIX')))] + vals)
        for i, rec in enumerate(rings or []):
            rm = 0.5 * (rec['r_in'] + rec['r_out'])
            vals = []
            for q, (_l, _u, _fb, fr, _t) in QUANTITIES.items():
                vals += list(fr(rec, ctx)) if fr is not None else [np.nan, np.nan]
            out.append(['ring', i, rec.get('kin')] + [rm / self._px_per_unit(u) for u in units]
                       + [rec['r_in'], rec['r_out'], rec['npix']] + vals)
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow([f'# EMPOWER Explorer radial profiles · {s.key} · centre x={s.ap_x:.2f} y={s.ap_y:.2f} px on '
                        f'{s.ref_band} · rings {"velocity-registered" if self._ring_key()[5] else "not registered"}'])
            w.writerow(head)
            for r in out:
                w.writerow(['' if (isinstance(v, float) and not np.isfinite(v)) else v for v in r])
        self.state.status.emit(f'Saved {len(out)} rows to {path}', 5000)
