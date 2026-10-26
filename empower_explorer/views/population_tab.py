"""Population tab - the whole sample in one diagram (successor of 'Correlations' + WHAN).

Layout
    [Main sequence | BPT [NII] | BPT [SII] | WHAN | Redshift | S/N vs time | Custom]   count
    X [column] [log]   Y [column] [log]   Colour [column]        (Redshift: Group by, Stacked, Bin)
    +-------------------------------------------------+  +-----------------+
    |  scatter plot or histogram (+ colour bar)        |  | selected galaxy |
    |  literature lines, region labels, legend         |  | values + pct    |
    +-------------------------------------------------+  | about, export   |

Points: filled circles are measured values, open triangles are limits and point to where the
true value lies (the <col>_kind columns: 'val' / 'upper' / 'lower').  Hover shows a tooltip,
a click selects that galaxy (state.select_galaxy) and the current galaxy carries a white ring.
The presets add the literature lines (Speagle+14 / Popesso+23, Kauffmann+03, Kewley+01/06,
Cid Fernandes+11).  Changing X, Y or a log toggle turns a preset into 'Custom'; the colour can
be changed in every preset.

Data: state.catalog.galaxies only (one row per galaxy of the loaded reduction).  Nothing comes
from the cubes, so a redraw is a few array operations on ~200 rows; the per-column arrays are
cached per catalogue and the plot items are created once and reused.
"""
import math

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtCore import QLocale, QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QStandardItemModel
from PyQt5.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
                             QHBoxLayout, QLabel, QPushButton, QSizePolicy, QStackedWidget, QVBoxLayout,
                             QWidget)

from empower_explorer import lines, theme
from empower_explorer.data import tables
from empower_explorer.theme import C
from empower_explorer.widgets.plots import EmptyState, colormap, robust_levels, style_plot_item

# ---------------------------------------------------------------------------------------------- presets
# key -> label, tooltip, columns, log axes, default colour, reference box (data range always shown,
# in plotted coordinates, i.e. log10 for log axes) and legend corner.
PRESETS = {
    'ms': dict(label='Main sequence', x='logM', y='logSFR', color='z', box=((8.4, 12.0), (-2.3, 3.0)),
               legend='tl', tip='Stellar mass vs SFR with the Speagle+14 main sequence at the median z'),
    'bpt_n2': dict(label='BPT [NII]', x='N2', y='O3', color='field', box=((-1.6, 0.8), (-1.25, 1.5)),
                   legend='bl', tip='[NII]/Hα vs [OIII]/Hβ with the Kauffmann+03 and Kewley+01 demarcations'),
    'bpt_s2': dict(label='BPT [SII]', x='S2', y='O3', color='field', box=((-1.3, 0.7), (-1.25, 1.5)),
                   legend='bl', tip='[SII]/Hα vs [OIII]/Hβ with the Kewley+01 and Kewley+06 lines'),
    'whan': dict(label='WHAN', x='N2', y='EW_ha', logy=True, color='whan',
                 box=((-1.6, 0.8), (math.log10(0.25), math.log10(700))), legend='br',
                 tip='[NII]/Hα vs EW(Hα) with the Cid Fernandes+11 classes'),
    'hist': dict(label='Redshift', x='z', color='field', legend='tr',
                 tip='Redshift distribution of the sample per field'),
    'snr': dict(label='S/N vs time', x='texp_ha_h', y='snr_ha', logy=True, color='field', legend='tl',
                tip='Integrated Hα S/N vs the on-source time of the band it was measured in'),
    'custom': dict(label='Custom', legend='tr', tip='Any two columns, any colour'),
}
PRESET_ORDER = ['ms', 'bpt_n2', 'bpt_s2', 'whan', 'hist', 'snr', 'custom']
CUSTOM_DEFAULT = dict(x='logM', y='OH_N2', logx=False, logy=False)

ABOUT = {
    'ms': 'Star-forming main sequence. Solid line: Speagle+14 at the median redshift of the plotted '
          'galaxies (shaded ±0.3 dex); dashed: Popesso+23. Galaxies with log SFR < −2 (quiescent '
          'by SED) sit at the lower edge as open triangles. Masses and SFRs from the SED catalogue.',
    'bpt_n2': 'Below the dashed Kauffmann+03 curve the lines come from star formation, above the '
              'solid Kewley+01 curve from an AGN; in between: composite. Open triangles are 2σ limits '
              'and point to where the true ratio lies.',
    'bpt_s2': 'Kewley+01 (solid) separates star formation from AGN; Kewley+06 (dashed) splits the AGN '
              'into Seyferts (above) and LINERs (below). Open triangles are 2σ limits.',
    'whan': 'WHAN diagram (Cid Fernandes+11): EW(Hα) against [NII]/Hα separates star formation, strong '
            'and weak AGN, and retired galaxies (EW < 3 Å) whose lines come from old stars.',
    'hist': 'Catalogue redshifts of the galaxies in this reduction. Hover a bar for the counts; '
            'the cyan line marks the selected galaxy.',
    'snr': 'Integrated Hα S/N against the on-source time of the band Hα was measured in, both from '
           'kmos_galaxy_properties.csv (catalogue-statistics notebook: 5×5 box, its own reduction). '
           'Dashed: S/N ∝ √t through the median galaxy; dotted: S/N = 3. S/N ≤ 0 is not shown (log axis).',
    'custom': 'Pick any two columns. Colour by a number (colour bar) or a category (legend). '
              'Log axes leave out values ≤ 0.',
}

HINT_SCATTER = ('Click a point to select the galaxy · drag to pan · wheel to zoom · drag the legend aside · '
                'right-click to export as image')
HINT_HIST = 'Hover a bar for the counts · drag to pan · wheel to zoom · right-click to export as image'

# limit flags of a column ('val' / 'upper' / 'lower')
KIND_COLUMNS = {'N2': 'N2_kind', 'O3': 'O3_kind', 'S2': 'S2_kind', 'OH_N2': 'OH_N2_kind',
                'sigma_ha': 'sigma_kind'}
SFR_FLOOR = -2.0        # main sequence: log SFR below this is drawn at the floor (as the notebook)
COSMO_H0, COSMO_OM = 70.0, 0.3     # FlatLambdaCDM of COSMOS2020 / the notebooks

# colours of known classes (BPT '?' = a limit was involved: darker shade of the class colour)
_CC = theme.CATEGORY_COLORS
CLASS_COLORS = {'SF': _CC[0], 'comp': _CC[4], 'AGN': _CC[7], 'sAGN': _CC[7], 'wAGN': _CC[5],
                'retired': _CC[3], 'passive': C.TEXT_DIM}
CLASS_ORDER = ['SF', 'SF?', 'comp', 'comp?', 'AGN', 'AGN?', 'sAGN', 'wAGN', 'retired', 'passive']

SIZE_VALUE, SIZE_LIMIT = 10, 13


# ---------------------------------------------------------------------------------------------- science
def age_gyr(z):
    """Age of the universe [Gyr] for flat ΛCDM (H0 = 70, Ωm = 0.3, no radiation) - astropy's value."""
    ol = 1.0 - COSMO_OM
    hubble_time = 977.792 / COSMO_H0
    return 2.0 / (3.0 * hubble_time ** -1 * math.sqrt(ol)) * math.asinh(math.sqrt(ol / COSMO_OM) * (1 + z) ** -1.5)


def ms_speagle14(logm, z):
    """Star-forming main sequence, Speagle et al. 2014 (eq. 28); scatter ~0.3 dex."""
    t = age_gyr(z)
    return (0.84 - 0.026 * t) * logm - (6.51 - 0.11 * t)


def ms_popesso23(logm, z):
    """Main sequence with the high-mass bending, Popesso et al. 2023 (eq. 14)."""
    t = age_gyr(z)
    return (-0.034 * t + 4.722) * logm - 0.1925 * logm ** 2 + (0.20 * t - 26.134)


# ---------------------------------------------------------------------------------------------- helpers
def _qcolor(c, alpha=None):
    q = QColor(c)
    if alpha is not None:
        q.setAlpha(alpha)
    return q


def _pen(color, width=1.5, style=Qt.SolidLine):
    p = pg.mkPen(color, width=width)
    p.setStyle(style)
    p.setCosmetic(True)
    return p


def _fmt(v, col=''):
    if v is None or not np.isfinite(v):
        return '—'
    if col in ('ra', 'dec'):
        return f'{v:.5f}'
    a = abs(v)
    if a >= 1e4 or 0 < a < 1e-3:
        return f'{v:.3g}'
    return f'{v:.4g}' if a >= 10 else f'{v:.3g}'


def _ordinal(n):
    n = int(n)
    suffix = 'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f'{n}{suffix}'


def _unit(col):
    return tables.COLUMNS[col][1] if col in tables.COLUMNS else ''


def _short(col):
    """Label without unit (tooltips, side panel)."""
    return tables.COLUMNS[col][0] if col in tables.COLUMNS else tables.CATEGORICAL.get(col, col)


_TRIANGLES = {}


def _triangle(dx, dy):
    """Triangle symbol whose tip points along (dx, dy) in data orientation (y up)."""
    if (dx, dy) not in _TRIANGLES:
        ux, uy = dx, -dy                  # symbol coordinates have y pointing down
        n = math.hypot(ux, uy)
        ux, uy = ux / n, uy / n
        px, py = -uy, ux
        path = QPainterPath()
        path.moveTo(QPointF(0.55 * ux, 0.55 * uy))
        path.lineTo(QPointF(-0.45 * ux + 0.5 * px, -0.45 * uy + 0.5 * py))
        path.lineTo(QPointF(-0.45 * ux - 0.5 * px, -0.45 * uy - 0.5 * py))
        path.closeSubpath()
        _TRIANGLES[(dx, dy)] = path
    return _TRIANGLES[(dx, dy)]


def _sequential_cmap():
    """theme.CMAP_SEQUENTIAL without its darkest part (invisible on the dark background)."""
    base = colormap(theme.CMAP_SEQUENTIAL)
    return pg.ColorMap(np.linspace(0, 1, 24), base.map(np.linspace(0.18, 1.0, 24), mode='byte'))


_SUPERSCRIPT = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')


def _log_tick(v):
    """0.001 … 5000 as plain numbers, beyond that 10⁴ / 3×10⁻⁵."""
    if not np.isfinite(v) or v <= 0:
        return ''
    if 1e-3 <= v < 9.9e3:
        return f'{v:g}'
    e = math.floor(math.log10(v) + 1e-9)
    m = v / 10.0 ** e
    sup = str(e).translate(_SUPERSCRIPT)
    return f'10{sup}' if abs(m - 1.0) < 1e-3 else f'{m:.0f}×10{sup}'


class _Axis(pg.AxisItem):
    """Log axes labelled 0.1, 1, 10, 100 instead of 10⁻¹, 10⁰, ... (10⁶ for very wide ranges)."""

    def logTickStrings(self, values, scale, spacing):
        vals = 10 ** np.asarray(values, float) * scale
        if spacing is not None:                 # decade level
            return [_log_tick(v) for v in vals]
        # minor level (1..9 × 10ⁿ): label only as many multiples as fit without overlapping
        lo, hi = self.range
        length = self.width() if self.orientation in ('top', 'bottom') else self.height()
        ppd = length / max(hi - lo, 1e-9)       # pixels per decade
        keep = (set(range(1, 10)) if ppd > 700 else {1, 2, 5} if ppd > 200 else {1, 3} if ppd > 110 else {1})
        out = []
        for v in vals:
            m = int(round(v / 10.0 ** math.floor(math.log10(v) + 1e-9))) if v > 0 else 0
            out.append(_log_tick(v) if m in keep else '')
        return out


class _Columns:
    """Cached per-column arrays of the galaxies table of one catalogue (row order = df order)."""

    def __init__(self, df):
        self.df = df
        self.keys = np.asarray(df.index, dtype=object)
        self.row = {k: i for i, k in enumerate(self.keys)}
        self.n = len(df)
        self.jitter = np.random.default_rng(7).uniform(-0.28, 0.28, self.n)   # categorical axes
        self._num, self._cat, self._kind = {}, {}, {}

    def invalidate(self, col):
        self._num.pop(col, None)
        self._cat.pop(col, None)
        self._kind.pop(col, None)

    def num(self, col):
        if col not in self._num:
            self._num[col] = tables.numeric(self.df, col)
        return self._num[col]

    def cat(self, col):
        """Column as text, '' for missing (also '—' and 'nan')."""
        if col not in self._cat:
            if col in self.df.columns:
                a = np.array(['' if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v).strip()
                              for v in self.df[col]], dtype=object)
                a[(a == '—') | (a == 'nan') | (a == 'None')] = ''
            else:
                a = np.full(self.n, '', dtype=object)
            self._cat[col] = a
        return self._cat[col]

    def kind(self, col):
        """Limit direction per galaxy: 0 value, -1 upper limit, +1 lower limit."""
        if col not in self._kind:
            kc = KIND_COLUMNS.get(col, f'{col}_kind')
            k = np.zeros(self.n, int)
            if kc in self.df.columns:
                s = self.cat(kc)
                k[s == 'upper'] = -1
                k[s == 'lower'] = 1
            self._kind[col] = k
        return self._kind[col]

    def has_data(self, col):
        if col in tables.CATEGORICAL:
            return bool((self.cat(col) != '').any())
        return bool(np.isfinite(self.num(col)).any())


def _is_cat(col):
    return col in tables.CATEGORICAL


class _PercentBar(QWidget):
    """Thin bar showing a percentile (None = empty)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(6)
        self._p = None

    def set_value(self, p):
        self._p = p
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        r = QRectF(self.rect())
        p.setBrush(QColor(C.BG_LIGHT))
        p.drawRoundedRect(r, 3, 3)
        if self._p is not None:
            p.setBrush(QColor(C.ACCENT))
            p.drawRoundedRect(QRectF(0, 0, max(6.0, r.width() * self._p / 100.0), r.height()), 3, 3)


class _ValueRow(QWidget):
    """Side panel block: column label, value, percentile text and bar."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 6)
        lay.setSpacing(2)
        self.name = QLabel()
        self.name.setProperty('role', 'dim')
        row = QHBoxLayout()
        row.setSpacing(6)
        self.value = QLabel()
        self.value.setStyleSheet('font-size: 16px; font-weight: bold;')
        self.pct = QLabel()
        self.pct.setProperty('role', 'dim')
        self.pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.value)
        row.addStretch()
        row.addWidget(self.pct)
        self.bar = _PercentBar()
        lay.addWidget(self.name)
        lay.addLayout(row)
        lay.addWidget(self.bar)

    def set(self, name, value, pct_text='', pct=None, tip=''):
        self.name.setText(name)
        self.value.setText(value)
        self.pct.setText(pct_text)
        self.bar.set_value(pct)
        self.bar.setVisible(pct is not None)
        self.setToolTip(tip)


# ---------------------------------------------------------------------------------------------- the view
class PopulationTab(QWidget):
    """Galaxy properties of the whole sample: presets (MS, BPT, WHAN, z, S/N) and a custom scatter."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.table = None                    # _Columns of the current catalogue
        self._preset = 'ms'
        self._custom = dict(CUSTOM_DEFAULT)
        self._colours = {k: p.get('color', 'field') for k, p in PRESETS.items()}
        self._syncing = False
        self._overlays, self._overlay_sig = [], None
        self._overlay_curves, self._overlay_labels = [], []   # line vertices / label centres (legend placement)
        self._legend_ctx = None              # (preferred corner, point x, point y) of the last draw
        self._hist_items = []                # [(BarGraphItem, PlotCurveItem)] reused per group
        self._hist = None                    # edges + counts per group for the hover text
        self._plotted = np.array([], int)    # table rows shown in the current plot
        self._view_x = self._view_y = None   # plotted coordinates per table row (NaN = not shown)
        self._spec = {}

        self._build_ui()
        state.catalogLoading.connect(self._on_catalog_loading)
        state.catalogChanged.connect(self._on_catalog)
        state.galaxyChanged.connect(lambda _k: self._update_selection())
        state.redshiftChanged.connect(self._on_redshift)
        if state.catalog is not None:
            self._on_catalog()
        else:
            self.empty.set_text('No reduction loaded yet', 'The population plots appear once the cubes are scanned.')
            self.stack.setCurrentIndex(0)

    # ================================================================================= UI construction
    def heightForWidth(self, w):
        # word-wrapped labels make Qt reserve their one-word-per-line height: use the real minimum
        return self.minimumSizeHint().height()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 8, 0, 0)
        self.stack = QStackedWidget()
        outer.addWidget(self.stack)
        self.empty = EmptyState()
        self.stack.addWidget(self.empty)
        page = QWidget()
        self.stack.addWidget(page)
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lay.addLayout(self._build_preset_bar())
        lay.addLayout(self._build_controls())
        body = QHBoxLayout()
        body.setSpacing(10)
        body.addWidget(self._build_plot(), 1)
        body.addWidget(self._build_side())
        lay.addLayout(body, 1)

    def _build_preset_bar(self):
        row = QHBoxLayout()
        row.setSpacing(12)
        seg = QFrame()
        seg.setObjectName('seg')
        seg.setStyleSheet(
            f'QFrame#seg {{ background: {C.BG_MEDIUM}; border: 1px solid {C.BG_LIGHT}; border-radius: 9px; }}'
            f'QFrame#seg QPushButton {{ background: transparent; color: {C.TEXT_DIM}; border: none;'
            f' padding: 6px 14px; border-radius: 6px; font-weight: bold; }}'
            f'QFrame#seg QPushButton:hover {{ background: {C.BG_LIGHT}; color: {C.TEXT}; }}'
            f'QFrame#seg QPushButton:checked {{ background: {C.ACCENT}; color: white; }}')
        sl = QHBoxLayout(seg)
        sl.setContentsMargins(3, 3, 3, 3)
        sl.setSpacing(2)
        self.preset_group = QButtonGroup(self)
        self.preset_group.setExclusive(True)
        self.preset_buttons = {}
        for key in PRESET_ORDER:
            b = QPushButton(PRESETS[key]['label'])
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(PRESETS[key]['tip'])
            b.clicked.connect(lambda _c, k=key: self.set_preset(k))
            self.preset_group.addButton(b)
            self.preset_buttons[key] = b
            sl.addWidget(b)
        self.preset_buttons[self._preset].setChecked(True)
        row.addWidget(seg)
        row.addStretch()
        self.count_label = QLabel('')
        self.count_label.setProperty('role', 'dim')
        self.count_label.setWordWrap(True)          # wraps instead of widening the window
        self.count_label.setMinimumWidth(120)
        self.count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.count_label, 1)
        return row

    def _combo(self, tip):
        cb = QComboBox()
        cb.setToolTip(tip)
        cb.setMaxVisibleItems(30)
        # fixed, moderate width (long labels are elided in the box, shown in full in the list)
        cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        cb.setMinimumContentsLength(15)
        cb.view().setMinimumWidth(260)
        return cb

    def _dim(self, text):
        lab = QLabel(text)
        lab.setProperty('role', 'dim')
        return lab

    def _build_controls(self):
        row = QHBoxLayout()
        row.setSpacing(6)
        # scatter controls
        self.scatter_controls = QWidget()
        sc = QHBoxLayout(self.scatter_controls)
        sc.setContentsMargins(0, 0, 0, 0)
        sc.setSpacing(6)
        self.x_combo = self._combo('Column on the x axis (changing it switches to Custom)')
        self.y_combo = self._combo('Column on the y axis (changing it switches to Custom)')
        self.logx = QCheckBox('log')
        self.logx.setToolTip('Logarithmic x axis (values ≤ 0 are left out)')
        self.logy = QCheckBox('log')
        self.logy.setToolTip('Logarithmic y axis (values ≤ 0 are left out)')
        self.c_combo = self._combo('Colour of the points: a number (colour bar) or a category (legend)')
        for w in (self._dim('X'), self.x_combo, self.logx, self._spacer(14), self._dim('Y'), self.y_combo,
                  self.logy, self._spacer(14), self._dim('Colour'), self.c_combo):
            sc.addWidget(w)
        row.addWidget(self.scatter_controls)
        # histogram controls
        self.hist_controls = QWidget()
        hc = QHBoxLayout(self.hist_controls)
        hc.setContentsMargins(0, 0, 0, 0)
        hc.setSpacing(6)
        self.g_combo = self._combo('Split the histogram by a category')
        self.stacked = QCheckBox('Stacked')
        self.stacked.setChecked(True)
        self.stacked.setToolTip('Stack the groups on top of each other (off: overlaid outlines)')
        self.bin_spin = QDoubleSpinBox()
        self.bin_spin.setLocale(QLocale.c())     # English UI: 0.025, not 0,025
        self.bin_spin.setDecimals(3)
        self.bin_spin.setRange(0.005, 0.5)
        self.bin_spin.setSingleStep(0.005)
        self.bin_spin.setValue(0.025)
        self.bin_spin.setToolTip('Width of a histogram bin in redshift')
        for w in (self._dim('Group by'), self.g_combo, self._spacer(14), self.stacked, self._spacer(14),
                  self._dim('Bin width Δz'), self.bin_spin):
            hc.addWidget(w)
        row.addWidget(self.hist_controls)
        row.addStretch()

        self.x_combo.currentIndexChanged.connect(self._axes_edited)
        self.y_combo.currentIndexChanged.connect(self._axes_edited)
        self.logx.toggled.connect(self._axes_edited)
        self.logy.toggled.connect(self._axes_edited)
        self.c_combo.currentIndexChanged.connect(self._colour_edited)
        self.g_combo.currentIndexChanged.connect(self._colour_edited)
        self.stacked.toggled.connect(lambda _v: self.redraw())
        self.bin_spin.valueChanged.connect(lambda _v: self.redraw())
        return row

    def _spacer(self, w):
        s = QWidget()
        s.setFixedWidth(w)
        return s

    def _build_plot(self):
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setBackground(C.BG_MEDIUM)
        self.glw.ci.setContentsMargins(6, 8, 6, 6)
        self.plot = self.glw.addPlot(row=0, col=0, axisItems={'bottom': _Axis('bottom'), 'left': _Axis('left')})
        style_plot_item(self.plot)
        self.plot.setMenuEnabled(False, enableViewBoxMenu=True)   # no 'Plot options' (log mode etc.)
        for name in ('left', 'bottom'):
            self.plot.getAxis(name).setStyle(tickFont=QFont(theme.FONT_FAMILY, 10))
        self.vb = self.plot.getViewBox()

        self.cbar = pg.ColorBarItem(values=(0, 1), colorMap=_sequential_cmap(), width=12, interactive=False)
        ax = self.cbar.getAxis('right')
        ax.setWidth(None)            # room for tick labels AND the rotated title (no overlap)
        ax.enableAutoSIPrefix(False)         # plain values (z = 0.64, not 640 x0.001)
        ax.setTextPen(pg.mkPen(C.TEXT_DIM))
        ax.setPen(pg.mkPen(C.BG_LIGHT))
        ax.setStyle(tickFont=QFont(theme.FONT_FAMILY, 9))
        self._cbar_shown = False

        self.scatter = pg.ScatterPlotItem(hoverable=True, tip=self._tip, hoverSize=SIZE_LIMIT + 3,
                                          hoverPen=_pen('#FFFFFF', 1.6))
        self.scatter.setZValue(5)
        self.scatter.sigClicked.connect(self._point_clicked)
        self.scatter.sigHovered.connect(self._hovered)
        self.plot.addItem(self.scatter)

        self.ring = pg.ScatterPlotItem(pxMode=True)
        self.ring.setZValue(10)
        self.ring.setAcceptedMouseButtons(Qt.NoButton)
        self.ring.setAcceptHoverEvents(False)
        self.plot.addItem(self.ring, ignoreBounds=True)
        self.ring_label = pg.TextItem('', color=C.TEXT, anchor=(-0.12, 1.25), fill=pg.mkBrush(_qcolor(C.BG_DARK, 200)))
        self.ring_label.setFont(QFont(theme.FONT_FAMILY, 9, QFont.Bold))
        self.ring_label.setZValue(11)
        self.plot.addItem(self.ring_label, ignoreBounds=True)

        self.zline = pg.InfiniteLine(angle=90, movable=False, pen=_pen(C.ACCENT, 2))
        self.zline.label = pg.InfLineLabel(self.zline, text='', position=0.93, anchors=[(0, 0), (0, 0)],
                                           color=C.ACCENT)
        self.zline.label.setFont(QFont(theme.FONT_FAMILY, 9, QFont.Bold))
        self.zline.setZValue(10)
        self.plot.addItem(self.zline, ignoreBounds=True)
        self.zline.hide()

        self.message = pg.TextItem('', color=C.TEXT_DIM, anchor=(0.5, 0.5))
        self.message.setFont(QFont(theme.FONT_FAMILY, 12, italic=True))
        self.plot.addItem(self.message, ignoreBounds=True)

        self.legend = pg.LegendItem(offset=(10, 10), labelTextColor=C.TEXT, labelTextSize='9pt',
                                    brush=pg.mkBrush(_qcolor(C.BG_MEDIUM, 225)), pen=pg.mkPen(C.BG_LIGHT),
                                    verSpacing=-3)
        self.legend.setParentItem(self.vb)
        self.glw.scene().sigMouseMoved.connect(self._mouse_moved)
        return self.glw

    def _build_side(self):
        card = QFrame()
        card.setProperty('role', 'card')
        card.setFixedWidth(260)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(6)
        sec = QLabel('SELECTED GALAXY')
        sec.setProperty('role', 'section')
        lay.addWidget(sec)
        self.sel_key = QLabel('—')
        self.sel_key.setStyleSheet('font-size: 17px; font-weight: bold;')
        self.sel_name = QLabel('')
        self.sel_name.setProperty('role', 'dim')
        self.sel_name.setWordWrap(True)
        self.sel_note = QLabel('')
        self.sel_note.setProperty('role', 'warn')
        self.sel_note.setWordWrap(True)
        for w in (self.sel_key, self.sel_name, self.sel_note):
            lay.addWidget(w)
        lay.addSpacing(6)
        sec = QLabel('VALUES · PERCENTILE IN SAMPLE')
        sec.setProperty('role', 'section')
        sec.setToolTip('Percentile among all galaxies of this reduction that have a value')
        lay.addWidget(sec)
        self.value_rows = [_ValueRow() for _ in range(3)]
        for r in self.value_rows:
            lay.addWidget(r)
        lay.addSpacing(6)
        sec = QLabel('ABOUT THIS PLOT')
        sec.setProperty('role', 'section')
        lay.addWidget(sec)
        self.about = QLabel('')
        self.about.setWordWrap(True)
        self.about.setStyleSheet(f'color: {C.TEXT_DIM}; font-size: 12px;')
        self.about.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        lay.addWidget(self.about)
        self.hint = QLabel(HINT_SCATTER)
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(f'color: {C.TEXT_DIM}; font-size: 11px; font-style: italic;')
        lay.addStretch()
        lay.addWidget(self.hint)
        self.export_btn = QPushButton('Export CSV…')
        self.export_btn.setToolTip('Save the plotted galaxies (key + plotted columns) as a CSV table')
        self.export_btn.clicked.connect(self._export_clicked)
        lay.addWidget(self.export_btn)
        card.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        return card

    # ================================================================================= state signals
    def _on_catalog_loading(self, variant):
        self.empty.set_text(f'Scanning {variant} …', 'The population plots appear when the scan is done.')
        self.stack.setCurrentIndex(0)

    def _on_catalog(self):
        cat = self.state.catalog
        if cat is None or cat.galaxies is None or cat.galaxies.empty:
            self.table = None
            self.empty.set_text('No galaxies in this reduction', 'Choose another reduction in the toolbar.')
            self.stack.setCurrentIndex(0)
            return
        self.table = _Columns(cat.galaxies)
        self._fill_combos()
        self._overlay_sig = None
        self.stack.setCurrentIndex(1)
        self.set_preset(self._preset, force=True)

    def _on_redshift(self):
        if self.table is not None:
            self.table.invalidate('z')
            self._overlay_sig = None
            self.redraw()

    # ================================================================================= combos / presets
    def _fill_combo(self, cb, numeric=True, categorical=True, none_label=None):
        current = cb.currentData()
        cb.blockSignals(True)
        cb.clear()
        if none_label:
            cb.addItem(none_label, '')
        groups = ([list(tables.COLUMNS)] if numeric else []) + ([list(tables.CATEGORICAL)] if categorical else [])
        for gi, cols in enumerate(groups):
            if gi and cb.count():
                cb.insertSeparator(cb.count())
            for col in cols:
                has = self.table.has_data(col)
                cb.addItem(tables.label(col) + ('' if has else '  — no data'), col)
                i = cb.count() - 1
                hint = tables.COLUMNS[col][2] if col in tables.COLUMNS else 'category'
                cb.setItemData(i, hint or tables.label(col), Qt.ToolTipRole)
                if not has and isinstance(cb.model(), QStandardItemModel):
                    cb.model().item(i).setEnabled(False)
        idx = cb.findData(current)
        cb.setCurrentIndex(max(idx, 0))
        cb.blockSignals(False)

    def _fill_combos(self):
        self._fill_combo(self.x_combo)
        self._fill_combo(self.y_combo)
        self._fill_combo(self.c_combo, none_label='Single colour')
        self._fill_combo(self.g_combo, numeric=False, none_label='No grouping')

    @staticmethod
    def _set_combo(cb, col):
        i = cb.findData(col or '')
        if i >= 0:
            cb.setCurrentIndex(i)

    def set_preset(self, key, force=False):
        """Show a preset ('ms', 'bpt_n2', 'bpt_s2', 'whan', 'hist', 'snr', 'custom')."""
        if key not in PRESETS or (key == self._preset and not force and self.table is not None
                                  and self._spec.get('preset') == key):
            return
        self._preset = key
        self.preset_buttons[key].setChecked(True)
        p = PRESETS[key]
        spec = self._custom if key == 'custom' else dict(x=p.get('x'), y=p.get('y'), logx=p.get('logx', False),
                                                          logy=p.get('logy', False))
        self._syncing = True
        try:
            if key == 'hist':
                self._set_combo(self.g_combo, self._colours[key])
            else:
                self._set_combo(self.x_combo, spec['x'])
                self._set_combo(self.y_combo, spec['y'])
                self.logx.setChecked(bool(spec['logx']))
                self.logy.setChecked(bool(spec['logy']))
                self._set_combo(self.c_combo, self._colours[key])
        finally:
            self._syncing = False
        self.scatter_controls.setVisible(key != 'hist')
        self.hist_controls.setVisible(key == 'hist')
        self.about.setText(ABOUT[key])
        self.hint.setText(HINT_HIST if key == 'hist' else HINT_SCATTER)
        self.redraw()

    def _axes_edited(self, *_):
        if self._syncing or self.table is None:
            return
        self._custom = dict(x=self.x_combo.currentData(), y=self.y_combo.currentData(),
                            logx=self.logx.isChecked(), logy=self.logy.isChecked())
        if self._preset != 'custom':
            self._colours['custom'] = self.c_combo.currentData()
            self._preset = 'custom'
            self.preset_buttons['custom'].setChecked(True)
            self.about.setText(ABOUT['custom'])
        self.redraw()

    def _colour_edited(self, *_):
        if self._syncing or self.table is None:
            return
        cb = self.g_combo if self._preset == 'hist' else self.c_combo
        self._colours[self._preset] = cb.currentData()
        self.redraw()

    # ================================================================================= drawing
    def redraw(self):
        if self.table is None:
            return
        self.legend.clear()
        self.message.setText('')
        self.vb.setToolTip('')
        if self._preset == 'hist':
            self._draw_hist()
        else:
            self._draw_scatter()
        self._update_selection()

    # ---------------------------------------------------------------- scatter
    def _axis_values(self, col, log):
        """(plotted values, raw values, limit directions, n_nonpositive) of one axis column."""
        t = self.table
        if not col:
            nan = np.full(t.n, np.nan)
            return nan, nan, np.zeros(t.n, int), 0
        if _is_cat(col):
            cats = t.cat(col)
            names = sorted({c for c in cats if c}, key=self._cat_sort_key(col))
            code = {c: i for i, c in enumerate(names)}
            v = np.array([code.get(c, np.nan) for c in cats], float) + t.jitter
            return v, v, np.zeros(t.n, int), 0
        raw = t.num(col)
        kind = t.kind(col)
        if not log:
            return raw, raw, kind, 0
        pos = raw > 0
        with np.errstate(divide='ignore', invalid='ignore'):
            v = np.where(pos, np.log10(np.where(pos, raw, 1.0)), np.nan)
        return v, raw, kind, int((np.isfinite(raw) & ~pos).sum())

    def _draw_scatter(self):
        t = self.table
        p = PRESETS[self._preset]
        xcol, ycol = self.x_combo.currentData(), self.y_combo.currentData()
        logx = self.logx.isChecked() and not _is_cat(xcol)
        logy = self.logy.isChecked() and not _is_cat(ycol)
        ccol = self.c_combo.currentData()
        self.logx.setEnabled(not _is_cat(xcol))
        self.logy.setEnabled(not _is_cat(ycol))
        self._spec = dict(preset=self._preset, x=xcol, y=ycol, logx=logx, logy=logy, color=ccol)

        x, xraw, kx, nonpos_x = self._axis_values(xcol, logx)
        y, yraw, ky, nonpos_y = self._axis_values(ycol, logy)
        x, y, kx, ky = x.copy(), y.copy(), kx.copy(), ky.copy()
        if self._preset == 'ms':            # quiescent by SED: draw at the floor as an upper limit
            low = np.isfinite(y) & (y < SFR_FLOOR)
            y[low] = SFR_FLOOR
            ky[low] = -1
        ok = np.isfinite(x) & np.isfinite(y)
        idx = np.flatnonzero(ok)
        is_lim = (kx[idx] != 0) | (ky[idx] != 0)
        idx = np.concatenate([idx[is_lim], idx[~is_lim]])     # values drawn on top of limits
        self._plotted = idx
        self._view_x = np.where(ok, x, np.nan)
        self._view_y = np.where(ok, y, np.nan)
        self._raw = (xraw, yraw)
        self._kinds = (kx, ky)

        colours, cat_entries = self._colours_for(ccol, idx)
        self._style_points(idx, x, y, kx, ky, colours)
        self._set_axes(xcol, ycol, logx, logy)
        overlay_entries = self._set_overlays(idx, x, y)
        self._set_range(idx, x, y, p.get('box') if self._preset != 'custom' else None)
        self._set_legend(overlay_entries, cat_entries, n_limits=int(is_lim.sum()))
        self._place_legend(p.get('legend', 'tr'), x[idx], y[idx])

        n_missing = t.n - len(idx)
        nonpos = nonpos_x + nonpos_y
        text = f'{len(idx)} galaxies with both values ({n_missing} missing)'
        if is_lim.any():
            text += f' · {int(is_lim.sum())} limits'
        self.count_label.setText(text)
        self.count_label.setToolTip(f'{t.n} galaxies in this reduction. Missing = no value for '
                                    f'{_short(xcol)} or {_short(ycol)}'
                                    + (f'; {nonpos} values ≤ 0 cannot be shown on a log axis' if nonpos else ''))
        if not len(idx):
            self._show_message(f'No galaxy has both {_short(xcol)} and {_short(ycol)}')
        for bar, curve in self._hist_items:
            bar.hide()
            curve.hide()
        self.zline.hide()
        self.scatter.show()

    def _style_points(self, idx, x, y, kx, ky, colours):
        symbols, pens, brushes, sizes = [], [], [], []
        outline = _pen(C.BG_DARK, 1.0)
        for i, col in zip(idx, colours):
            dx, dy = int(kx[i]), int(ky[i])
            if dx == 0 and dy == 0:
                symbols.append('o')
                pens.append(outline)
                brushes.append(pg.mkBrush(col))
                sizes.append(SIZE_VALUE)
            else:
                symbols.append(_triangle(dx, dy))
                pens.append(_pen(col, 1.7))
                brushes.append(pg.mkBrush(_qcolor(col, 45)))
                sizes.append(SIZE_LIMIT)
        self.scatter.setData(x=x[idx], y=y[idx], data=list(idx), symbol=symbols, pen=pens, brush=brushes,
                             size=sizes)

    def _cat_sort_key(self, col):
        if col in ('class_NII_BPT', 'class_SII_BPT', 'whan'):
            return lambda c: (CLASS_ORDER.index(c) if c in CLASS_ORDER else len(CLASS_ORDER), c)
        if col in ('bands', 'band_ha'):
            from empower_explorer.data.catalog import band_rank
            return lambda c: (band_rank(c.split('+')[0]), len(c), c)
        return lambda c: c

    def _category_colour(self, col, name, i):
        base = name.rstrip('?')
        if col in ('class_NII_BPT', 'class_SII_BPT', 'whan') and base in CLASS_COLORS:
            c = QColor(CLASS_COLORS[base])
            return c.darker(150) if name.endswith('?') else c
        if col in ('bands', 'band_ha') and name in theme.BAND_COLORS:
            return QColor(theme.band_color(name))
        return QColor(_CC[i % len(_CC)])

    def _colours_for(self, ccol, idx):
        """Point colours for the rows idx and the legend entries [(QColor, label)]; colour bar for numbers."""
        grey = _qcolor(C.TEXT_DIM, 150)
        if not ccol:
            self._show_cbar(False)
            return [QColor(C.ACCENT)] * len(idx), []
        if _is_cat(ccol):
            self._show_cbar(False)
            cats = self.table.cat(ccol)
            names = sorted({c for c in cats if c}, key=self._cat_sort_key(ccol))
            cmap = {n: self._category_colour(ccol, n, i) for i, n in enumerate(names)}
            sub = cats[idx]
            entries = [(cmap[n], f'{n} ({int((sub == n).sum())})') for n in names if (sub == n).any()]
            n_unknown = int((sub == '').sum())
            if n_unknown:
                entries.append((grey, f'no {tables.CATEGORICAL[ccol].split(" (")[0]} ({n_unknown})'))
            return [cmap.get(c, grey) for c in sub], entries
        v = self.table.num(ccol)[idx]
        fin = np.isfinite(v)
        if not fin.any():
            self._show_cbar(False)
            return [grey] * len(idx), [(grey, f'no {_short(ccol)} ({len(idx)})')]
        lo, hi = robust_levels(v[fin], 2, 98)
        cm = _sequential_cmap()
        norm = np.clip((np.where(fin, v, lo) - lo) / (hi - lo), 0, 1)
        mapped = cm.map(norm, mode='qcolor')
        cols = [mapped[i] if fin[i] else grey for i in range(len(idx))]
        self.cbar.setColorMap(cm)
        self.cbar.setLevels((lo, hi))
        self.cbar.getAxis('right').setLabel(tables.label(ccol), color=C.TEXT_DIM)
        self._show_cbar(True)
        n_nan = int((~fin).sum())
        return cols, ([(grey, f'no {_short(ccol)} ({n_nan})')] if n_nan else [])

    def _show_cbar(self, show):
        if show and not self._cbar_shown:
            self.glw.ci.addItem(self.cbar, row=0, col=1)
        elif not show and self._cbar_shown:
            self.glw.ci.removeItem(self.cbar)
        self._cbar_shown = show

    def _set_axes(self, xcol, ycol, logx, logy):
        for name, col, log in (('bottom', xcol, logx), ('left', ycol, logy)):
            ax = self.plot.getAxis(name)
            ax.setLogMode(bool(log))
            lab = tables.label(col) if col else ''
            ax.setLabel(lab + ('  (log)' if log else ''), color=C.TEXT_DIM, **{'font-size': '11pt'})
            if col and _is_cat(col):
                names = sorted({c for c in self.table.cat(col) if c}, key=self._cat_sort_key(col))
                ax.setTicks([[(i, n) for i, n in enumerate(names)], []])
            else:
                ax.setTicks(None)

    def _set_range(self, idx, x, y, box):
        def span(v, box_range):
            v = v[np.isfinite(v)]
            if v.size:
                lo, hi = float(v.min()), float(v.max())
                pad = 0.07 * (hi - lo) if hi > lo else 0.5
                lo, hi = lo - pad, hi + pad
                if box_range:
                    lo, hi = min(lo, box_range[0]), max(hi, box_range[1])
                return lo, hi
            return box_range or (0.0, 1.0)

        bx, by = (box if box else (None, None))
        self.vb.setRange(xRange=span(x[idx], bx), yRange=span(y[idx], by), padding=0)

    def _show_message(self, text):
        self.message.setText(text)
        (x0, x1), (y0, y1) = self.vb.viewRange()
        self.message.setPos((x0 + x1) / 2, (y0 + y1) / 2)

    # ---------------------------------------------------------------- overlays (literature lines)
    def _set_overlays(self, idx, x, y):
        """Literature lines and region labels of the preset; returns legend entries [(item, label)]."""
        sig, builder = (self._preset,), None
        if self._preset == 'ms':
            z = self.table.num('z')[idx]
            zmed = float(np.median(z[np.isfinite(z)])) if np.isfinite(z).any() else 1.0
            sig, builder = ('ms', round(zmed, 3)), lambda: self._overlay_ms(zmed)
        elif self._preset == 'bpt_n2':
            builder = self._overlay_bpt_n2
        elif self._preset == 'bpt_s2':
            builder = self._overlay_bpt_s2
        elif self._preset == 'whan':
            builder = self._overlay_whan
        elif self._preset == 'snr':
            t, s = self.table.num('texp_ha_h')[idx], self._raw[1][idx]
            good = (t > 0) & (s > 0)
            amp = float(np.median(s[good] / np.sqrt(t[good]))) if good.any() else np.nan
            tmax = float(np.nanmax(t)) if t.size else 1.0
            sig, builder = ('snr', round(amp, 4), round(tmax, 2)), lambda: self._overlay_snr(amp, tmax)
        if sig == self._overlay_sig:
            return self._overlay_entries
        for it in self._overlays:
            self.plot.removeItem(it)
        self._overlays, self._overlay_entries = [], []
        self._overlay_curves, self._overlay_labels = [], []
        if builder is not None:
            builder()
        self._overlay_sig = sig
        return self._overlay_entries

    def _add_curve(self, x, y, color=C.TEXT, width=1.8, style=Qt.SolidLine, name=None):
        x, y = np.asarray(x, float), np.asarray(y, float)
        c = pg.PlotCurveItem(x, y, pen=_pen(color, width, style), connect='finite')
        c.setZValue(-5)
        self.plot.addItem(c, ignoreBounds=True)
        self._overlays.append(c)
        self._overlay_curves.append((x, y))
        if name:
            self._overlay_entries.append((c, name))
        return c

    def _add_region(self, text, x, y, anchor=(0.5, 0.5)):
        t = pg.TextItem(text, color=_qcolor(C.TEXT_DIM, 200), anchor=anchor)
        f = QFont(theme.FONT_FAMILY, 11, QFont.Bold)
        f.setLetterSpacing(QFont.PercentageSpacing, 108)
        t.setFont(f)
        t.setPos(x, y)
        t.setZValue(-4)
        self.plot.addItem(t, ignoreBounds=True)
        self._overlays.append(t)
        self._overlay_labels.append((x, y))

    def _overlay_ms(self, zmed):
        lm = np.linspace(8.0, 12.5, 160)
        ms = ms_speagle14(lm, zmed)
        lo = pg.PlotCurveItem(lm, ms - 0.3, pen=pg.mkPen(None))
        hi = pg.PlotCurveItem(lm, ms + 0.3, pen=pg.mkPen(None))
        band = pg.FillBetweenItem(lo, hi, brush=pg.mkBrush(_qcolor(C.TEXT_DIM, 38)))
        for it in (lo, hi, band):
            it.setZValue(-6)
            self.plot.addItem(it, ignoreBounds=True)
            self._overlays.append(it)
        self._add_curve(lm, ms, C.TEXT, 2.0, name=f'Speagle+14 · z = {zmed:.2f} (±0.3 dex)')
        self._add_curve(lm, ms_popesso23(lm, zmed), C.TEXT_DIM, 1.6, Qt.DashLine, name='Popesso+23')

    def _overlay_bpt_n2(self):
        n2 = np.linspace(-2.5, 0.04, 300)
        k03 = lines.kauffmann03(n2)
        self._add_curve(n2, np.where(k03 > -3, k03, np.nan), C.TEXT_DIM, 1.8, Qt.DashLine, name='Kauffmann+03')
        n2 = np.linspace(-2.5, 0.46, 300)
        k01 = lines.kewley01_n2(n2)
        self._add_curve(n2, np.where(k01 > -3, k01, np.nan), C.TEXT, 2.0, name='Kewley+01')
        self._add_region('Star-forming', -1.25, 0.2)
        self._add_region('Composite', 0.0, -1.08)       # between the curves (-0.21 .. 0.20 at this height)
        self._add_region('AGN', 0.35, 1.2)

    def _overlay_bpt_s2(self):
        s2 = np.linspace(-2.5, 0.31, 300)
        k01 = lines.kewley01_s2(s2)
        self._add_curve(s2, np.where(k01 > -3, k01, np.nan), C.TEXT, 2.0, name='Kewley+01')
        # Kewley+06 Seyfert / LINER line starts where it crosses the Kewley+01 curve
        grid = np.linspace(-1.0, 0.3, 2000)
        cross = grid[np.argmin(np.abs(lines.kewley06_s2_seyfert_liner(grid) - lines.kewley01_s2(grid)))]
        s2 = np.linspace(cross, 1.5, 50)
        self._add_curve(s2, lines.kewley06_s2_seyfert_liner(s2), C.TEXT_DIM, 1.8, Qt.DashLine, name='Kewley+06')
        self._add_region('Star-forming', -1.0, 0.2)
        self._add_region('Seyfert', -0.45, 1.25)
        self._add_region('LINER', 0.42, -0.55)

    def _overlay_whan(self):
        l3, l6, l05 = math.log10(3.0), math.log10(6.0), math.log10(0.5)
        nan = np.nan
        self._add_curve([-0.4, -0.4, nan, -0.4, 5, nan, -5, 5], [l3, 5, nan, l6, l6, nan, l3, l3], C.TEXT, 2.0,
                        name='Cid Fernandes+11')
        self._add_curve([-5, 5], [l05, l05], C.TEXT_DIM, 1.5, Qt.DashLine)
        self._add_region('SF', -1.0, math.log10(250))
        self._add_region('sAGN', 0.35, math.log10(250))
        self._add_region('wAGN', 0.35, math.log10(4.2))
        self._add_region('Retired', -1.0, math.log10(1.25))
        self._add_region('Passive', -1.0, math.log10(0.33))

    def _overlay_snr(self, amp, tmax):
        l3 = math.log10(3.0)
        self._add_curve([1e-3, 1e3], [l3, l3], C.TEXT_DIM, 1.4, Qt.DotLine, name='S/N = 3')
        if np.isfinite(amp) and amp > 0:
            t = np.linspace(0.05, max(tmax, 1.0) * 1.2, 200)
            self._add_curve(t, np.log10(amp * np.sqrt(t)), C.TEXT, 1.8, Qt.DashLine,
                            name='S/N ∝ √t through the median galaxy')

    # ---------------------------------------------------------------- legend
    def _set_legend(self, overlay_entries, cat_entries, n_limits=0):
        for item, name in overlay_entries:
            self.legend.addItem(item, name)
        for col, name in cat_entries:
            self.legend.addItem(pg.ScatterPlotItem(symbol='o', size=SIZE_VALUE, pen=_pen(C.BG_DARK, 1.0),
                                                   brush=pg.mkBrush(col)), name)
        if n_limits:
            self.legend.addItem(pg.ScatterPlotItem(symbol=_triangle(1, 0), size=SIZE_LIMIT,
                                                   pen=_pen(C.TEXT_DIM, 1.7), brush=pg.mkBrush(None)),
                                'limit (points to the true value)')
        self.legend.setVisible(bool(self.legend.items))

    def _place_legend(self, default=None, px=None, py=None):
        """Anchor the legend in the corner that hides the fewest points, lines and region labels.

        The preset's own corner wins ties.  Called after every draw and when the tab is shown
        again (the size of the view in pixels is only reliable then)."""
        if default is not None:
            self._legend_ctx = (default, px, py)
        if self._legend_ctx is None or not self.legend.items:
            return
        default, px, py = self._legend_ctx
        corner = default
        vw, vh = self.vb.width(), self.vb.height()
        lw, lh = self.legend.width(), self.legend.height()
        if vw > 150 and vh > 150 and lw > 0 and lh > 0:
            (x0, x1), (y0, y1) = self.vb.viewRange()
            sx, sy = (x1 - x0) / vw, (y1 - y0) / vh            # data units per pixel

            def rect(c, mx=7, my=7):
                if c[1] == 'l':
                    a, b = x0 + (12 - mx) * sx, x0 + (12 + lw + mx) * sx
                else:
                    a, b = x1 - (12 + lw + mx) * sx, x1 - (12 - mx) * sx
                if c[0] == 't':
                    lo, hi = y1 - (12 + lh + my) * sy, y1 - (12 - my) * sy
                else:
                    lo, hi = y0 + (12 - my) * sy, y0 + (12 + lh + my) * sy
                return a, b, lo, hi

            def inside(r, x, y):
                with np.errstate(invalid='ignore'):
                    return (x >= r[0]) & (x <= r[1]) & (y >= r[2]) & (y <= r[3])

            curves = list(self._curve_samples(sx, sy))
            best = None
            for c in [default] + [c for c in ('tl', 'tr', 'br', 'bl') if c != default]:
                r = rect(c)
                score = int(inside(r, px, py).sum()) if px is not None and len(px) else 0
                score += 3 * sum(bool(inside(r, cx, cy).any()) for cx, cy in curves)
                rl = rect(c, 55, 14)                           # label centre within half a label
                score += 3 * sum(bool(inside(rl, lx, ly)) for lx, ly in self._overlay_labels)
                if best is None or score < best[0]:
                    best = (score, c)
                if score == 0:
                    break
            corner = best[1]
        ix = 0 if corner[1] == 'l' else 1
        iy = 0 if corner[0] == 't' else 1
        self.legend.anchor(itemPos=(ix, iy), parentPos=(ix, iy), offset=(12 if ix == 0 else -12, 12 if iy == 0 else -12))

    def _curve_samples(self, sx, sy):
        """Overlay lines as point samples; short polylines are filled in every ~3 px."""
        for cx, cy in self._overlay_curves:
            if len(cx) > 20:
                yield cx, cy
                continue
            xs, ys = [], []
            for i in range(len(cx) - 1):
                xa, xb, ya, yb = cx[i], cx[i + 1], cy[i], cy[i + 1]
                if not (np.isfinite(xa) and np.isfinite(xb) and np.isfinite(ya) and np.isfinite(yb)):
                    continue
                n = int(min(math.hypot((xb - xa) / sx, (yb - ya) / sy) / 3.0, 3000)) + 2
                xs.append(np.linspace(xa, xb, n))
                ys.append(np.linspace(ya, yb, n))
            if xs:
                yield np.concatenate(xs), np.concatenate(ys)

    def showEvent(self, ev):
        super().showEvent(ev)
        self._place_legend()

    # ---------------------------------------------------------------- histogram
    def _draw_hist(self):
        t = self.table
        col = PRESETS['hist']['x']
        gcol = self.g_combo.currentData()
        self._spec = dict(preset='hist', x=col, y='', color=gcol)
        self._show_cbar(False)
        self.scatter.hide()
        self.ring.setData([], [])
        self.ring_label.setText('')
        self._set_overlays(np.array([], int), None, None)
        self._set_axes(col, '', False, False)
        self.plot.getAxis('left').setLabel('Number of galaxies', color=C.TEXT_DIM, **{'font-size': '11pt'})

        v = t.num(col)
        ok = np.isfinite(v)
        self._plotted = np.flatnonzero(ok)
        self._view_x = np.where(ok, v, np.nan)
        self._view_y = np.full(t.n, np.nan)
        n_missing = int((~ok).sum())
        self.count_label.setText(f'{int(ok.sum())} galaxies with {_short(col)} ({n_missing} missing)')
        self.count_label.setToolTip(f'{t.n} galaxies in this reduction; missing = no catalogue redshift')
        if not ok.any():
            for bar, curve in self._hist_items:
                bar.hide()
                curve.hide()
            self._hist = None
            self.vb.setRange(xRange=(0, 2), yRange=(0, 1), padding=0)
            self._show_message(f'No galaxy has a {_short(col).lower()}')
            return

        width = self.bin_spin.value()
        lo = math.floor(v[ok].min() / width) * width
        hi = math.ceil(v[ok].max() / width + 1e-9) * width
        edges = np.arange(lo, hi + width * 0.5, width)
        if len(edges) < 2:
            edges = np.array([lo, lo + width])
        if gcol:
            cats = t.cat(gcol)
            names = sorted({c for c in cats[ok] if c}, key=self._cat_sort_key(gcol))
            groups = [(n, self._category_colour(gcol, n, i), ok & (cats == n)) for i, n in enumerate(names)]
            if (ok & (cats == '')).any():
                groups.append((f'no {tables.CATEGORICAL[gcol].split(" (")[0]}', _qcolor(C.TEXT_DIM, 160),
                               ok & (cats == '')))
        else:
            groups = [('all', QColor(C.ACCENT), ok)]
        counts = [np.histogram(v[m], bins=edges)[0] for _, _, m in groups]
        self._hist = dict(edges=edges, groups=[g[0] for g in groups], counts=counts)

        while len(self._hist_items) < len(groups):
            bar = pg.BarGraphItem(x0=[0], x1=[1], height=[0])
            curve = pg.PlotCurveItem()
            for it in (bar, curve):
                self.plot.addItem(it)
            self._hist_items.append((bar, curve))
        stacked = self.stacked.isChecked()
        base = np.zeros(len(edges) - 1)
        entries = []
        for i, (bar, curve) in enumerate(self._hist_items):
            if i >= len(groups):
                bar.hide()
                curve.hide()
                continue
            name, colour, _m = groups[i]
            c = counts[i]
            if stacked:
                bar.setOpts(x0=edges[:-1], x1=edges[1:], y0=base.copy(), height=c, brush=pg.mkBrush(colour),
                            pen=_pen(C.BG_DARK, 1.0))
                bar.setZValue(1)
                bar.show()
                curve.hide()
                base = base + c
            else:
                curve.setData(edges, c, stepMode='center', fillLevel=0, pen=_pen(colour, 2.0),
                              brush=pg.mkBrush(_qcolor(colour, 55)))
                curve.setZValue(1 + i)
                curve.show()
                bar.hide()
            entries.append((colour, f'{name} ({int(c.sum())})'))
        ymax = float(base.max()) if stacked else float(max(c.max() for c in counts))
        self.vb.setRange(xRange=(edges[0] - width, edges[-1] + width), yRange=(0, max(1.0, ymax) * 1.12), padding=0)
        if gcol:
            for colour, name in entries:
                self.legend.addItem(pg.ScatterPlotItem(symbol='s', size=11, pen=pg.mkPen(None),
                                                       brush=pg.mkBrush(colour)), name)
        self.legend.setVisible(bool(self.legend.items))
        # the bars as point samples (3 across, 6 up each bar) so the legend avoids tall bars
        tops = base if stacked else np.max(counts, axis=0)
        filled = np.flatnonzero(tops > 0)
        if filled.size:
            fx = edges[filled][:, None] + width * np.array([0.1, 0.5, 0.9])[None, :]
            fy = tops[filled][:, None] * np.linspace(0, 1, 6)[None, :]
            px = np.repeat(fx, 6, axis=1).ravel()
            py = np.tile(fy, (1, 3)).ravel()
        else:
            px = py = np.array([])
        self._place_legend('tr', px, py)

    def _hist_tip(self, xv):
        h = self._hist
        if h is None:
            return ''
        e = h['edges']
        i = int(np.searchsorted(e, xv, side='right')) - 1
        if not 0 <= i < len(e) - 1:
            return ''
        rows = [f'{_short(PRESETS["hist"]["x"])} {e[i]:.3f} – {e[i + 1]:.3f}']
        total = 0
        for name, c in zip(h['groups'], h['counts']):
            if c[i]:
                rows.append(f'{name}: {int(c[i])}')
                total += int(c[i])
        if len(h['groups']) > 1:
            rows.append(f'total: {total}')
        return '\n'.join(rows) if total else ''

    # ================================================================================= interaction
    def _mouse_moved(self, scene_pos):
        if self._preset != 'hist' or self.table is None or not self.vb.sceneBoundingRect().contains(scene_pos):
            return
        p = self.vb.mapSceneToView(scene_pos)
        self.vb.setToolTip(self._hist_tip(p.x()))

    def _hovered(self, _item, points, _ev):
        self.glw.viewport().setCursor(Qt.PointingHandCursor if len(points) else Qt.ArrowCursor)

    def _tip(self, x, y, data):
        t = self.table
        if t is None or data is None or not 0 <= int(data) < t.n:
            return ''
        i = int(data)
        key = t.keys[i]
        name = t.cat('name')[i]
        rows = [key + (f'  ·  {name}' if name else '')]
        s = self._spec
        for col, raw, kind in ((s.get('x'), self._raw[0], self._kinds[0]), (s.get('y'), self._raw[1], self._kinds[1])):
            if col:
                rows.append(f'{_short(col)} = {self._value_text(col, i, raw, kind)}')
        c = s.get('color')
        if c and c not in (s.get('x'), s.get('y')):
            rows.append(f'{_short(c)} = {self._value_text(c, i)}')
        rows.append('click to select')
        return '\n'.join(rows)

    def _value_text(self, col, i, raw=None, kind=None):
        t = self.table
        if _is_cat(col):
            return t.cat(col)[i] or '—'
        raw = t.num(col) if raw is None else raw
        kind = t.kind(col) if kind is None else kind
        sign = {-1: '< ', 1: '> '}.get(int(kind[i]), '')
        if self._preset == 'ms' and col == 'logSFR' and raw[i] < SFR_FLOOR:
            sign = ''
        unit = _unit(col)
        v = _fmt(raw[i], col)
        return f'{sign}{v}' + (f' {unit}' if unit and v != '—' else '')

    def _point_clicked(self, _item, points, _ev):
        if not len(points) or self.table is None:
            return
        i = points[0].data()
        if i is not None and 0 <= int(i) < self.table.n:
            self.state.select_galaxy(self.table.keys[int(i)])

    # ================================================================================= selection + side panel
    def _update_selection(self):
        key = self.state.key
        t = self.table
        i = t.row.get(key) if (t is not None and key is not None) else None
        self.ring.setData([], [])
        self.ring_label.setText('')
        self.zline.hide()
        if i is not None and self._view_x is not None:
            xv, yv = self._view_x[i], self._view_y[i]
            if self._preset == 'hist':
                if np.isfinite(xv):
                    self.zline.setPos(xv)
                    self.zline.label.setText(f'{key}  z = {xv:.3f}')
                    self.zline.show()
            elif np.isfinite(xv) and np.isfinite(yv):
                self.ring.setData([xv, xv], [yv, yv], size=[30, 20], symbol='o', brush=[None, None],
                                  pen=[_pen(_qcolor(C.ACCENT, 130), 4.0), _pen('#FFFFFF', 2.2)])
                self.ring_label.setText(key)
                self.ring_label.setPos(xv, yv)
                (x0, x1), (y0, y1) = self.vb.viewRange()   # keep the name inside the frame
                right = xv > x0 + 0.72 * (x1 - x0)
                top = yv > y0 + 0.9 * (y1 - y0)
                self.ring_label.setAnchor((1.12 if right else -0.12, -0.25 if top else 1.25))
        self._update_side(i)

    def _percentile(self, col, i):
        v = self.table.num(col)
        fin = v[np.isfinite(v)]
        if not np.isfinite(v[i]) or fin.size < 2:
            return None, fin.size
        p = 100.0 * ((fin < v[i]).sum() + 0.5 * (fin == v[i]).sum()) / fin.size
        return p, fin.size

    def _update_side(self, i):
        t = self.table
        s = self._spec
        cols = [c for c in (s.get('x'), s.get('y'), s.get('color')) if c]
        cols = list(dict.fromkeys(cols))[:3]
        if t is None or i is None:
            self.sel_key.setText(self.state.key or '—')
            self.sel_name.setText('')
            self.sel_note.setText('No galaxy selected' if self.state.key is None else 'Not in this reduction')
            for r in self.value_rows:
                r.hide()
            self.export_btn.setEnabled(t is not None and len(self._plotted) > 0)
            return
        key = t.keys[i]
        self.sel_key.setText(key)
        name = t.cat('name')[i]
        z = t.num('z')[i]
        self.sel_name.setText(' · '.join(x for x in (name, f'z = {z:.3f}' if np.isfinite(z) else 'redshift unknown') if x))
        self.sel_note.setText(self._not_shown_reason(i))
        for r, col in zip(self.value_rows, cols + [None] * 3):
            if col is None:
                r.hide()
                continue
            r.show()
            if _is_cat(col):
                cat = t.cat(col)[i]
                n_same = int((t.cat(col) == cat).sum()) if cat else 0
                r.set(tables.CATEGORICAL[col], cat or '—', f'{n_same} of {t.n} galaxies' if cat else '')
                continue
            p, n = self._percentile(col, i)
            pct = f'{_ordinal(round(p))} pct · of {n}' if p is not None else (f'no value · {n} have one' if n else '')
            hint = tables.COLUMNS[col][2] if col in tables.COLUMNS else ''
            r.set(tables.label(col), self._value_text(col, i), pct, p, hint)
        self.export_btn.setEnabled(len(self._plotted) > 0)

    def _not_shown_reason(self, i):
        s = self._spec
        if self._view_x is None:
            return ''
        if self._preset == 'hist':
            return '' if np.isfinite(self._view_x[i]) else 'Not in this plot: redshift unknown'
        if np.isfinite(self._view_x[i]) and np.isfinite(self._view_y[i]):
            return ''
        why = []
        for col, log in ((s.get('x'), s.get('logx')), (s.get('y'), s.get('logy'))):
            if not col:
                continue
            if _is_cat(col):
                if not self.table.cat(col)[i]:
                    why.append(f'no {_short(col)}')
                continue
            v = self.table.num(col)[i]
            if not np.isfinite(v):
                why.append(f'no {_short(col)}')
            elif log and v <= 0:
                why.append(f'{_short(col)} ≤ 0 (log axis)')
        return 'Not in this plot: ' + ', '.join(why) if why else ''

    # ================================================================================= export
    def plotted_table(self):
        """DataFrame of the plotted galaxies: key (index), name, field and the plotted columns (+ limit flags)."""
        t = self.table
        if t is None or not len(self._plotted):
            return pd.DataFrame()
        s = self._spec
        cols = ['name', 'field']
        for c in (s.get('x'), s.get('y'), s.get('color')):
            if c and c not in cols:
                cols.append(c)
                kc = KIND_COLUMNS.get(c, f'{c}_kind')
                if kc in t.df.columns:
                    cols.append(kc)
        cols = [c for c in cols if c in t.df.columns]
        rows = np.sort(self._plotted)
        out = t.df.iloc[rows][cols].copy()
        out.index.name = 'key'
        return out

    def export_csv(self, path):
        df = self.plotted_table()
        df.to_csv(path, encoding='utf-8')
        return len(df)

    def _export_clicked(self):
        s = self._spec
        stem = '_'.join(c for c in (s.get('preset'), s.get('x'), s.get('y')) if c)
        path, _ = QFileDialog.getSaveFileName(self, 'Export plotted galaxies', f'empower_population_{stem}.csv',
                                              'CSV table (*.csv)')
        if not path:
            return
        try:
            n = self.export_csv(path)
        except OSError as e:
            self.state.error.emit('Export failed', str(e))
            return
        self.state.status.emit(f'Saved {n} galaxies to {path}', 6000)
