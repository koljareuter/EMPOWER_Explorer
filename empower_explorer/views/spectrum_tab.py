"""Spectrum tab - the main 'Explorer' view (successor of the KMOS viewer).

Layout
    left    image of the galaxy (white light, median S/N or a line map) with the draggable aperture,
            band chooser, aperture radius, 'Centre on' buttons and the shared display options
    right   2 x 2 zoom panels (Hβ | [OIII] / Hα+[NII] | [SII]): a header row per panel with the
            title (left) and the S/N badge (right) - both shorten themselves so they never overlap -
            and inside the plot only data, fit and marker labels (labels in a top strip that the y
            range keeps free; a label that would touch another goes one row lower).  The axis units
            are named once in the card header.  Below: the full spectrum of all bands on one axis and
            the table of line measurements.

Data flow
    stacksReady / redshiftChanged / some options -> _rebuild(): static plot items (windows, markers,
        telluric bands, sky lines), image, controls; then spectra and fits are redrawn.  'smooth',
        'mode', 'method' and 'subtract_bg' only redraw the curves / fetch new spectra.
    apertureChanged -> aperture circle and labels at once; spectra after SPEC_DEBOUNCE_MS and fits
        after FIT_DEBOUNCE_MS, i.e. when the drag pauses.  Extraction (~100-200 ms) and fitting
        (~70 ms) run in a worker (_JobQueue: one job at a time, only the newest request waits) and
        are cached here by aperture + options; the GUI thread only draws (< ~100 ms).  Both are
        skipped when nothing changed (draw keys).  While the fit is pending the old model is drawn
        dashed, so it is clear that it belongs to the previous aperture.
    While the tab is hidden nothing is drawn; it catches up in showEvent.

All x values are drawn in the display frame: observed micron, or rest-frame micron (divide by 1+z)
when the 'rest_frame' option is on and z is known.  Telluric bands and sky lines are converted too.
"""
import csv
import io
import traceback
from collections import OrderedDict

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QLocale, QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QTextDocument
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QFileDialog,
                             QDoubleSpinBox, QFrame, QGraphicsTextItem, QGridLayout, QHBoxLayout, QHeaderView,
                             QLabel, QPushButton,
                             QSizePolicy, QSlider, QSpinBox, QSplitter, QStackedWidget, QTableWidget,
                             QTableWidgetItem, QToolButton, QVBoxLayout, QWidget)

from empower_explorer import analysis, lines
from empower_explorer.data.catalog import band_rank
from empower_explorer.lines import FIT_LINES, FULL_MARKERS, LINE_COLORS, LINE_LABELS, PANELS, REST
from empower_explorer.theme import C, band_color
from empower_explorer.widgets.plots import (EmptyState, ImagePanel, SpectrumViewBox, add_vline, boxcar,
                                            make_plot, style_plot_item)
from empower_explorer.workers import run_async

SPEC_DEBOUNCE_MS = 40       # aperture moved -> new spectra
FIT_DEBOUNCE_MS = 250       # aperture moved -> new line fits (after the drag pauses)
R_MIN, R_MAX, R_STEP = 0.5, 12.0, 0.5

# aperture radius units: (key, button, tooltip); spin-box step per unit
RADIUS_UNITS = [('px', 'px', 'Radii in pixels of the reference band'),
                ('arcsec', '″', 'Radii in arcsec'),
                ('kpc', 'kpc', 'Radii in proper kpc at the galaxy redshift (flat ΛCDM, H₀ = 70, Ωm = 0.3)'),
                ('re', 'Rₑ', 'Radii in units of the effective radius: the half-light radius of the anchor line\n'
                             '(Hα, else [OIII]) of kmos_snr_distribution.csv, else measured here the same way')]
UNIT_STEP = {'px': 0.5, 'arcsec': 0.1, 'kpc': 0.5, 're': 0.25}
UNIT_TEXT = {'px': ' px', 'arcsec': '″', 'kpc': ' kpc', 're': ' Rₑ'}

# image chooser: (key, button label, line keys summed into the map, tooltip)
IMAGE_KINDS = [
    ('white', 'White light', (), 'Median over all wavelengths of the selected band'),
    ('snr', 'S/N', (), 'Median S/N per spaxel over the selected band (DATA / STAT)'),
    ('Ha', 'Hα', ('Ha',), 'Continuum-subtracted Hα narrow-band map\n(in the deepest band that covers Hα)'),
    ('OIII', '[OIII]', ('OIII5007',), 'Continuum-subtracted [OIII]5007 narrow-band map'),
    ('Hb', 'Hβ', ('Hb',), 'Continuum-subtracted Hβ narrow-band map'),
    ('SII', '[SII]', ('SII6716', 'SII6731'), 'Continuum-subtracted [SII]6716 + 6731 narrow-band map'),
]

# short marker labels inside the zoom panels (the panel title names the lines)
MARKER_LABELS = {'Hb': 'Hβ', 'OIII4959': '4959', 'OIII5007': '5007', 'NII6548': '[NII]', 'Ha': 'Hα',
                 'NII6583': '[NII]', 'SII6716': '6716', 'SII6731': '6731'}

PEAK_TOL_UM = 0.0025        # data this close to a line (observed) always stays inside the y range
MODEL_COLOR = C.AMBER


# ---------------------------------------------------------------------------------- helpers
def _pen(color, width=1.3, style=Qt.SolidLine, alpha=255):
    c = QColor(color)
    c.setAlpha(alpha)
    p = pg.mkPen(c, width=width)
    p.setStyle(style)
    p.setCosmetic(True)
    return p


def _brush(color, alpha):
    c = QColor(color)
    c.setAlpha(alpha)
    return pg.mkBrush(c)


def _fmt(v, spec='.1f', dash='—'):
    try:
        return format(v, spec) if np.isfinite(v) else dash
    except (TypeError, ValueError):
        return dash


def _unit(stack):
    u = (stack.bunit or '').replace('ADU/sec', 'ADU/s').strip()
    return u or 'arb.'


def _robust_ylim(x, y, lo, hi, line_x=(), tol=0.0, extra=(), q=(2, 98), head=0.3, foot=0.08):
    """y range of the data inside [lo, hi]: percentiles (one cosmic or sky residual must not flatten
    the plot), but data near the given lines and the `extra` values always stay visible."""
    if x is None or y is None:
        return None
    sel = (x >= lo) & (x <= hi) & np.isfinite(y)
    if sel.sum() < 3:
        return None
    a, b = np.percentile(y[sel], q)
    for lx in line_x:
        near = sel & (np.abs(x - lx) < tol)
        if near.any():
            b = max(b, float(np.max(y[near])))
    for e in extra:
        if e is not None and np.isfinite(e):
            b = max(b, float(e))
    span = b - a if b > a else (abs(b) or 1.0)
    return a - foot * span, b + head * span


def _sample(xp, fp, x):
    """fp at x: exact lookup when x lies on the grid xp (a fit window is a slice of its spectrum;
    np.interp would turn the pixel next to a NaN into NaN), else linear interpolation, NaN outside."""
    xp, fp, x = np.asarray(xp, float), np.asarray(fp, float), np.asarray(x, float)
    if len(xp) == 0:
        return np.full(x.shape, np.nan)
    idx = np.clip(np.searchsorted(xp, x), 0, len(xp) - 1)
    if np.allclose(xp[idx], x, rtol=0, atol=1e-9):
        return fp[idx]
    return np.interp(x, xp, fp, left=np.nan, right=np.nan)


def _merge_spans(spans):
    out = []
    for lo, hi in sorted(spans):
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [tuple(s) for s in out]


def _section(text):
    lab = QLabel(text.upper())
    lab.setProperty('role', 'section')
    return lab


def _dim(text='', shrink=False):
    """Dim label; shrink=True lets it get narrower than its text (long status lines must not
    set the minimum width of the whole tab)."""
    lab = QLabel(text)
    lab.setProperty('role', 'dim')
    if shrink:
        lab.setMinimumWidth(1)
        lab.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    return lab


def _card(widget, margins=(8, 8, 8, 8)):
    frame = QFrame()
    frame.setProperty('role', 'card')
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(*margins)
    lay.addWidget(widget)
    return frame


_FLAT_DISABLED = f'QPushButton:disabled {{ color: {C.BG_HOVER}; background-color: {C.BG_MEDIUM}; }}'


def _flat_button(text, tip, css=''):
    """Secondary button; disabled ones look disabled (the theme's flat rule would hide that)."""
    b = QPushButton(text)
    b.setProperty('flat', 'true')
    b.setToolTip(tip)
    b.setCursor(Qt.PointingHandCursor)
    b.setStyleSheet((f'QPushButton {{ {css} }} ' if css else '') + _FLAT_DISABLED)
    return b


# ---------------------------------------------------------------------------------- background work
def _extract_all(stacks, ref_band, x, y, r, method, subtract_bg, r_in=0.0):
    """Worker: the aperture (or annulus) spectra of all bands (same sky position) - AppState.spectra() logic."""
    ref = stacks[ref_band]
    out = {}
    for b, st in stacks.items():
        bx, by, br = analysis.transfer(ref, st, x, y, r)
        sp = analysis.extract_spectrum(st, bx, by, br, method=method, subtract_bg=subtract_bg,
                                       r_in=r_in * br / r if r > 0 else 0.0)
        if sp.n_pix > 0:
            out[b] = sp
    return out


class _JobQueue:
    """At most one background job at a time; while it runs only the newest request waits (a drag
    produces many requests, only the last one matters).  done(key, result, error) is called in the
    GUI thread for every finished job - the receiver decides whether the result is still wanted."""

    def __init__(self, done):
        self._done = done
        self._running = None
        self._pending = None

    def request(self, key, fn, *args):
        if self._running is not None:
            if key != self._running:
                self._pending = (key, fn, args)
            return
        self._start(key, fn, args)

    def busy(self, key):
        return key == self._running or (self._pending is not None and self._pending[0] == key)

    def drop_pending(self):
        self._pending = None

    def _start(self, key, fn, args):
        self._running = key
        run_async(fn, *args, on_done=lambda res, k=key: self._finish(k, res, None),
                  on_error=lambda msg, k=key: self._finish(k, None, msg))

    def _finish(self, key, res, err):
        self._running = None
        nxt, self._pending = self._pending, None
        if nxt is not None:
            self._start(*nxt)
        self._done(key, res, err)


class _VariantLabel(QLabel):
    """Rich-text label with long and short variants: shows the longest one that fits and never
    sets the minimum width of its layout (it may get narrower than any variant)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._variants = ['']
        self.setMinimumWidth(1)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._doc = QTextDocument()
        self._doc.setDocumentMargin(0)

    def set_variants(self, variants):
        self._variants = [v for v in variants if v is not None] or ['']
        self._pick()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._pick()

    def _pick(self):
        self._doc.setDefaultFont(self.font())
        pick = self._variants[-1]
        for v in self._variants:
            self._doc.setHtml(v)
            if self._doc.idealWidth() <= self.width() - 2:
                pick = v
                break
        if pick != self.text():
            super().setText(pick)


# ---------------------------------------------------------------------------------- small widgets
class Segmented(QWidget):
    """A row of exclusive toggle buttons (segmented control).  `changed(key)` on user clicks only."""
    changed = pyqtSignal(str)

    _CSS = f"""
    QToolButton {{ background-color: {C.BG_LIGHT}; color: {C.TEXT_DIM}; border: none; border-radius: 0px;
                   padding: 4px 8px; font-weight: 600; font-size: 12px; }}
    QToolButton:hover {{ background-color: {C.BG_HOVER}; color: {C.TEXT}; }}
    QToolButton:checked {{ background-color: {C.ACCENT}; color: white; }}
    QToolButton:disabled {{ background-color: {C.BG_MEDIUM}; color: {C.BG_HOVER}; }}
    QToolButton[seg="first"] {{ border-top-left-radius: 6px; border-bottom-left-radius: 6px; }}
    QToolButton[seg="last"] {{ border-top-right-radius: 6px; border-bottom-right-radius: 6px; }}
    QToolButton[seg="only"] {{ border-radius: 6px; }}
    """

    def __init__(self, items=(), parent=None):
        super().__init__(parent)
        self.setStyleSheet(self._CSS)
        self._lay = QHBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(1)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons = {}
        self.set_items(items)

    def set_items(self, items):
        """items: [(key, label, tooltip)]"""
        for b in self._buttons.values():      # remove at once (deleteLater alone can leave them visible)
            self._group.removeButton(b)
            self._lay.removeWidget(b)
            b.hide()
            b.setParent(None)
            b.deleteLater()
        self._buttons = {}
        n = len(items)
        for i, (key, label, tip) in enumerate(items):
            b = QToolButton()
            b.setText(label)
            b.setToolTip(tip)
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setProperty('seg', 'only' if n == 1 else 'first' if i == 0 else 'last' if i == n - 1 else 'mid')
            b.clicked.connect(lambda _=False, k=key: self.changed.emit(k))
            self._group.addButton(b)
            self._lay.addWidget(b)
            self._buttons[key] = b

    def keys(self):
        return list(self._buttons)

    def set_current(self, key):
        b = self._buttons.get(key)
        if b is not None and not b.isChecked():
            b.setChecked(True)

    def current(self):
        for k, b in self._buttons.items():
            if b.isChecked():
                return k
        return None

    def set_item_enabled(self, key, enabled, tooltip=None):
        b = self._buttons.get(key)
        if b is None:
            return
        b.setEnabled(enabled)
        if tooltip is not None:
            b.setToolTip(tooltip)


class _Corner(pg.TextItem):
    """Text pinned to a corner ('tl', 'tr') or the centre ('c') of a ViewBox, independent of the
    data range (parented to the ViewBox itself, not to its scaled child group)."""

    def __init__(self, vb, where='tl', margin=(8, 6)):
        super().__init__('', anchor={'tl': (0, 0), 'tr': (1, 0), 'c': (0.5, 0.5)}[where])
        self._where, self._margin = where, margin
        self.setParentItem(vb)
        self.setZValue(50)
        vb.sigResized.connect(self._place)
        self._place()

    def _place(self, *_):
        r = self.parentItem().boundingRect()
        mx, my = self._margin
        if self._where == 'tl':
            self.setPos(r.left() + mx, r.top() + my)
        elif self._where == 'tr':
            self.setPos(r.right() - mx, r.top() + my)
        else:
            self.setPos(r.center())

    def show_html(self, html, border=None, fill=None):
        self.border = pg.mkPen(border) if border is not None else pg.mkPen(None)
        self.fill = fill if fill is not None else pg.mkBrush(None)
        self.setHtml(html)
        self.setVisible(bool(html))
        self._place()


class _VLines(pg.PlotCurveItem):
    """Many vertical lines over the full view height, drawn as ONE path (hundreds of sky lines would
    be far too slow as InfiniteLines).  Follows the y range of its ViewBox."""

    def __init__(self, xs, pen):
        super().__init__(pen=pen, connect='pairs')
        self._xs = np.repeat(np.asarray(xs, float), 2)
        self._vb = None

    def attach(self, plot):
        plot.addItem(self, ignoreBounds=True)
        self._vb = plot.getViewBox()
        self._vb.sigYRangeChanged.connect(self._follow)
        self._follow(self._vb, self._vb.viewRange()[1])
        return self

    def detach(self, plot):
        if self._vb is not None:
            try:
                self._vb.sigYRangeChanged.disconnect(self._follow)
            except TypeError:
                pass
        plot.removeItem(self)

    def _follow(self, _vb, rng):
        lo, hi = rng
        self.setData(self._xs, np.tile([lo, hi], len(self._xs) // 2))


class _Spans(pg.GraphicsObject):
    """Shaded x ranges (telluric bands) over the full view height as ONE item: dozens of
    LinearRegionItems (two InfiniteLines each) made every rebuild ~60 ms slower."""

    def __init__(self, spans, rgba=C.TELLURIC):
        super().__init__()
        self._spans = [(float(a), float(b)) for a, b in spans if b > a]
        self._brush = pg.mkBrush(*rgba)
        self._y = (0.0, 1.0)
        self._vb = None
        self.setZValue(-10)

    def attach(self, plot):
        plot.addItem(self, ignoreBounds=True)
        self._vb = plot.getViewBox()
        self._vb.sigYRangeChanged.connect(self._follow)
        self._follow(self._vb, self._vb.viewRange()[1])
        return self

    def detach(self, plot):
        if self._vb is not None:
            try:
                self._vb.sigYRangeChanged.disconnect(self._follow)
            except TypeError:
                pass
        plot.removeItem(self)

    def _follow(self, _vb, rng):
        self.prepareGeometryChange()
        self._y = (float(rng[0]), float(rng[1]))
        self.update()

    def boundingRect(self):
        if not self._spans:
            return QRectF()
        lo, hi = self._spans[0][0], max(b for _, b in self._spans)
        y0, y1 = self._y
        return QRectF(lo, y0, hi - lo, y1 - y0)

    def paint(self, p, *args):
        p.setPen(pg.mkPen(None))
        p.setBrush(self._brush)
        y0, y1 = self._y
        for a, b in self._spans:
            p.drawRect(QRectF(a, y0, b - a, y1 - y0))


def _clear_items(plot, items):
    for it in items:
        if hasattr(it, 'detach'):
            it.detach(plot)
        else:
            plot.removeItem(it)


# ---------------------------------------------------------------------------------- line panel
class _PanelHeader(pg.GraphicsWidget):
    """Header row above a zoom panel: title (left, aligned with the plot area) and S/N badge (right).

    Title and badge come in long and short variants; the longest pair that fits next to each other
    is shown, so they never overlap whatever the window size (the full text is in the tooltip).
    Nothing of this lies inside the plot area, which therefore only holds data and line labels."""
    HEIGHT = 24
    GAP = 12

    def __init__(self, plot):
        super().__init__()
        self._plot = plot
        self.title = self._text_item()
        self.badge = self._text_item()
        self._titles, self._title_w = [''], [0.0]
        self._badges, self._badge_w = [''], [0.0]
        self._badge_pen = self._badge_brush = None
        self.setFlag(self.ItemClipsChildrenToShape, True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumSize(10, self.HEIGHT)
        self.setPreferredSize(10, self.HEIGHT)
        self.setMaximumHeight(self.HEIGHT)
        plot.getViewBox().sigResized.connect(self._layout)

    def _text_item(self):
        it = QGraphicsTextItem(self)
        it.setFont(QFont('Segoe UI', 10))
        it.document().setDocumentMargin(3)
        return it

    @staticmethod
    def _measure(item, variants):
        widths = []
        for html in variants:
            item.setHtml(html)
            widths.append(item.boundingRect().width() if html else 0.0)
        return widths

    def set_title(self, variants):
        """variants: HTML strings, longest first."""
        self._titles = list(variants) or ['']
        self._title_w = self._measure(self.title, self._titles)
        self._layout()

    def set_badge(self, variants, border=None, fill=None, tip=''):
        self._badges = list(variants) or ['']
        self._badge_w = self._measure(self.badge, self._badges)
        self._badge_pen = pg.mkPen(border, width=1) if border else None
        self._badge_brush = fill
        self.setToolTip(tip)
        self._layout()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._layout()

    def _layout(self, *_):
        w, h = self.width(), self.height()
        vb = self._plot.getViewBox()
        r = vb.mapRectToItem(self, vb.boundingRect())
        left, right = (max(0.0, r.left()), min(w, r.right())) if r.width() > 20 else (0.0, w)
        avail = right - left
        nt, nb = len(self._titles), len(self._badges)
        # preference: full pair; title without 'λλ...'; badge without flags, without 'forced'; title
        # without band.  If even the shortest pair does not fit, the badge is left out (tooltip).
        combos = []
        for t, b in ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)):
            c = (min(t, nt - 1), min(b, nb - 1))
            if c not in combos:
                combos.append(c)
        ti, bi = nt - 1, None
        for t, b in combos:
            tw, bw = self._title_w[t], self._badge_w[b]
            if tw + bw + (self.GAP if tw and bw else 0) <= avail:
                ti, bi = t, b
                break
        self.title.setHtml(self._titles[ti])
        self.title.setVisible(bool(self._titles[ti]))
        self.badge.setVisible(bi is not None and bool(self._badges[bi]))
        bi = nb - 1 if bi is None else bi
        self.badge.setHtml(self._badges[bi])
        th, bh = self.title.boundingRect().height(), self.badge.boundingRect().height()
        self.title.setPos(left - 3, (h - th) / 2)
        self.badge.setPos(right - self._badge_w[bi], (h - bh) / 2)
        self.update()

    def paint(self, p, *args):
        if not self.badge.isVisible() or self._badge_pen is None and self._badge_brush is None:
            return
        rect = self.badge.mapRectToParent(self.badge.boundingRect()).adjusted(0.5, 1.5, -0.5, -1.5)
        p.setRenderHint(p.Antialiasing, True)
        p.setPen(self._badge_pen if self._badge_pen is not None else pg.mkPen(None))
        p.setBrush(self._badge_brush if self._badge_brush is not None else pg.mkBrush(None))
        p.drawRoundedRect(rect, 4, 4)


class _LinePanel:
    """One zoom panel: header (title + S/N badge) above a PlotItem with data, fit model, line
    markers and a centred placeholder message.  The marker labels sit in the top rows of the plot,
    side by side or (when they would touch) one row lower; the y range keeps that strip free."""

    LABEL_TOP = 2          # px between the top of the plot and the first label row

    def __init__(self, cell, spec):
        self.spec = spec
        self.name, self.title, self.line_keys = spec[0], spec[1], spec[2]
        self.short_title = self.title.split(' λ')[0]
        self.main_keys = list(spec[3]) if isinstance(spec[3], tuple) else [spec[3]]
        cell.setContentsMargins(0, 0, 0, 0)
        cell.setSpacing(0)
        plot = cell.addPlot(row=1, col=0, viewBox=SpectrumViewBox())
        self.plot = plot
        self.header = _PanelHeader(plot)
        cell.addItem(self.header, row=0, col=0)
        style_plot_item(plot)
        plot.hideButtons()
        plot.getAxis('bottom').setStyle(tickTextOffset=3)
        vb = plot.getViewBox()                    # wheel: wavelength; Shift+wheel or y axis: y
        vb.sigXRangeChanged.connect(lambda *_: self.autoscale())
        vb.sigResized.connect(lambda *_: self.autoscale())
        vb.sigYAuto.connect(self.autoscale)       # double-click: y follows the data again
        self.curve = pg.PlotDataItem(pen=_pen(C.ACCENT, 1.0))
        self.curve.setClipToView(True)          # only the zoom window of the band is drawn
        self.curve.setZValue(5)
        self.model = pg.PlotDataItem(pen=_pen(MODEL_COLOR, 1.8))
        self.model.setZValue(6)
        plot.addItem(self.curve)
        plot.addItem(self.model)
        self.msg = _Corner(vb, 'c', margin=(0, 0))
        self.static = []
        self.marks = []            # [(InfiniteLine, label left of the line, main line)]
        self._label_rows = 1
        self.band = None
        self.window = None
        self.x = self.y = None
        self.line_x, self.tol, self.model_peak = [], PEAK_TOL_UM, np.nan

    # -- static content
    def clear_static(self):
        _clear_items(self.plot, self.static)
        self.static = []
        self.marks = []

    def set_title(self, band=None):
        def html(title):
            out = f'<span style="color:{C.TEXT}; font-size:10pt; font-weight:600;">{title}</span>'
            if band:
                out += (f'<span style="color:{C.TEXT_DIM}; font-size:10pt;"> · </span>'
                        f'<span style="color:{band_color(band)}; font-size:10pt; font-weight:600;">{band}</span>')
            return out
        variants = [html(self.title), html(self.short_title)]      # same slots for every panel
        if band:                                 # narrowest: line name only (band = curve colour)
            variants.append(f'<span style="color:{C.TEXT}; font-size:10pt; font-weight:600;">'
                            f'{self.short_title.replace(" + ", "+")}</span>')
        self.header.set_title(variants)

    def placeholder(self, text):
        """No data for this panel at all (no z, not covered)."""
        self.clear_static()
        self.band, self.x, self.y = None, None, None
        self.set_title(None)
        self.curve.setData([], [])
        self.model.setData([], [])
        self.set_badge(None)
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')
        self.plot.showGrid(x=False, y=False)
        self._message(text)

    def configure(self, band, window, markers, spans, sky, keep_x=False, keep_data=False):
        """New band / window: markers [(x, label, colour, main line, label left of the line)].
        The axis units are named once in the header of the panel card (saves height)."""
        self.clear_static()
        if not keep_data or band != self.band:
            self.x = self.y = None
            self.curve.setData([], [])
        self.model.setData([], [])
        self.model_peak = np.nan
        self.band = band
        self.set_title(band)
        self.set_badge(None)
        self.plot.showAxis('left')
        self.plot.showAxis('bottom')
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.curve.setPen(_pen(band_color(band), 1.0))   # >1 px antialiased: ~5x slower repaint
        self.curve.setFillLevel(0)
        self.curve.setFillBrush(_brush(band_color(band), 38))
        self.line_x = [m[0] for m in markers]
        if spans:
            self.static.append(_Spans(sorted(spans)).attach(self.plot))
        if len(sky):
            self.static.append(_VLines(sky, _pen(C.SKY, 1.0, Qt.DotLine, 200)).attach(self.plot))
        for x, label, color, bold, left in markers:
            item = add_vline(self.plot, x, label, color, dashed=True, width=1.3 if bold else 1.0,
                             label_pos=0.95, anchor_left=left, bold=bold)
            self.static.append(item)
            self.marks.append((item, left, bold))
        self._message('')
        self.window = window
        if not keep_x:
            self.plot.getViewBox().y_manual = False
            self.plot.setXRange(*window, padding=0)
        self._layout_labels()

    def _layout_labels(self):
        """Marker labels in the top strip of the plot: main lines first, a label that would touch
        an already placed one goes one row lower.  Pixel based, so it adapts to size and zoom."""
        vb = self.plot.getViewBox()
        h, w = vb.height(), vb.width()
        labs = [(it, left, bold) for it, left, bold in self.marks if getattr(it, 'label', None) is not None]
        if h < 10 or w < 10 or not labs:
            self._label_rows = 1 if labs else 0
            return
        x0, x1 = vb.viewRange()[0]
        if not x1 > x0:
            return
        px = w / (x1 - x0)
        rows = [[], []]
        used = 1
        step = 14
        for it, left, bold in sorted(labs, key=lambda m: not m[2]):
            br = it.label.boundingRect()
            lw = br.width() - 4                  # the text item has a few px margin
            step = max(br.height() - 5, 10)
            xp = (it.value() - x0) * px
            if left and xp - lw < 0:             # would be cut at the plot edge: other side
                left = False
            elif not left and xp + lw > w and xp - lw >= 0:
                left = True
            anchor = (1, 0) if left else (0, 0)
            if it.label.anchors[0] != anchor:
                it.label.anchors = [anchor, anchor]
            span = (xp - lw - 2, xp + 2) if left else (xp - 2, xp + lw + 2)
            row = 0
            if any(span[0] < b and span[1] > a for a, b in rows[0]):
                row = 1
            rows[row].append(span)
            used = max(used, row + 1)
            it.label.setPosition(1.0 - (self.LABEL_TOP + row * step) / h)
        self._label_rows = used
        self._label_h = step

    # -- dynamic content
    def set_curve(self, x, y):
        self.x, self.y = x, y
        self._message('')
        self.curve.setData(x, y)
        self.autoscale()

    def set_model(self, x, y, peak=np.nan):
        self.model_peak = peak
        if x is None:
            self.model.setData([], [])
        else:
            self.model.setData(x, y, connect='finite')
            self.model.setPen(_pen(MODEL_COLOR, 1.8))
        self.autoscale()

    def fade_model(self):
        self.model.setPen(_pen(MODEL_COLOR, 1.4, Qt.DashLine, 110))

    def set_badge(self, variants, detected=False, tip=''):
        if not variants:
            self.header.set_badge([''], tip='')
            return
        self.header.set_badge(variants, border=C.EMERALD if detected else C.BG_HOVER,
                              fill=_brush(C.EMERALD if detected else C.BG_LIGHT, 45 if detected else 150), tip=tip)

    def outside(self, text):
        """Panel configured, but the current aperture has no data in its band."""
        self.x = self.y = None
        self.curve.setData([], [])
        self.model.setData([], [])
        self.set_badge(None)
        self._message(text)

    def autoscale(self):
        if self.band is None:
            return
        self._layout_labels()
        if self.x is None:
            return
        vb = self.plot.getViewBox()
        lo, hi = vb.viewRange()[0]
        # keep the label strip at the top free of data: head = fraction of the data span above it
        h = max(vb.height(), 1.0)
        strip = self.LABEL_TOP + 4 + self._label_rows * getattr(self, '_label_h', 14)
        frac = min(strip / h, 0.5)
        head = max(0.12, frac * 1.08 / (1.0 - frac))
        lim = _robust_ylim(self.x, self.y, lo, hi, self.line_x, self.tol, extra=[self.model_peak], head=head)
        if lim is None:
            self._message('no valid data in this window')
            return
        self._message('')
        if not vb.y_manual:                       # zoomed or panned in y by hand: keep that range
            self.plot.setYRange(*lim, padding=0)

    def _message(self, text):
        self.msg.show_html(f'<span style="color:{C.TEXT_DIM}; font-size:10pt; font-style:italic;">'
                           f'{text.replace(chr(10), "<br>")}</span>' if text else '')


# ---------------------------------------------------------------------------------- full spectrum
class _FullSpectrum(QWidget):
    """All bands on one axis, with band legend, telluric bands, sky lines, line markers and a
    hover readout.  y follows the data inside the visible wavelength range."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        head = QHBoxLayout()
        head.setContentsMargins(4, 0, 4, 0)
        head.addWidget(_section('Full spectrum'))
        head.addSpacing(10)
        self.legend = QLabel('')
        self.legend.setMinimumWidth(1)           # long legends are cut, they never widen the tab
        self.legend.setToolTip('Bands with their on-source time (all cubes of the band stacked)')
        head.addWidget(self.legend)
        head.addSpacing(12)
        self.readout = _dim('', shrink=True)
        self.readout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.readout.setToolTip('Value under the mouse cursor')
        head.addWidget(self.readout, 1)
        lay.addLayout(head)
        self.pw = make_plot('', 'Observed wavelength [µm]', 'S/N', viewBox=SpectrumViewBox())
        self.pw.setToolTip('Scroll: zoom the wavelength axis · Shift + scroll or scroll on the y axis: zoom y\n'
                           'Drag: pan · drag the y axis: pan y · double-click: y follows the data again\n'
                           'Right-click → View All / Export…')
        self.plot = self.pw.getPlotItem()
        self.plot.hideButtons()
        vb = self.plot.getViewBox()
        vb.sigXRangeChanged.connect(lambda *_: self.autoscale())
        vb.sigYAuto.connect(self.autoscale)
        lay.addWidget(self.pw, 1)
        self.cross = pg.InfiniteLine(angle=90, movable=False, pen=_pen(C.TEXT_DIM, 1.0, Qt.DotLine))
        self.cross.setZValue(40)
        self.cross.hide()
        self.plot.addItem(self.cross, ignoreBounds=True)
        self.msg = _Corner(vb, 'c', margin=(0, 0))
        self.curves = {}
        self.static = []
        self.xy = {}
        self.peaks_x = []
        self.ylabel = 'S/N'
        self.rest = None          # (z) when the x axis is rest frame
        self.pw.scene().sigMouseMoved.connect(self._hover)

    def clear(self):
        _clear_items(self.plot, self.static)
        self.static = []
        for c in self.curves.values():
            self.plot.removeItem(c)
        self.curves, self.xy = {}, {}

    def configure(self, bands, xrange, markers, spans, sky, xlabel, keep_x=False):
        self.clear()
        for b in bands:
            # 1 px pen + peak downsampling: a full band (~2000 px) repaints in a few ms
            c = pg.PlotDataItem(pen=_pen(band_color(b), 1.0), fillLevel=0, fillBrush=_brush(band_color(b), 30))
            c.setClipToView(True)
            c.setDownsampling(auto=True, method='peak')
            c.setZValue(5)
            self.plot.addItem(c)
            self.curves[b] = c
        if spans:
            self.static.append(_Spans(sorted(spans)).attach(self.plot))
        if len(sky):
            self.static.append(_VLines(sky, _pen(C.SKY, 1.0, Qt.SolidLine, 90)).attach(self.plot))
        self.peaks_x = [m[0] for m in markers]
        for x, label, color, pos in markers:
            self.static.append(add_vline(self.plot, x, label, color, dashed=True, width=1.0, label_pos=pos))
        self.plot.setLabel('bottom', xlabel, color=C.TEXT_DIM)
        self.msg.show_html('')
        if not keep_x and xrange is not None:
            self.plot.getViewBox().y_manual = False
            self.plot.setXRange(*xrange, padding=0.01)

    def set_legend(self, entries, note=''):
        """entries: [(band, text, active)]; note: e.g. the flux scaling of the bands."""
        parts = []
        for b, text, active in entries:
            col = band_color(b) if active else C.BG_HOVER
            tcol = C.TEXT if active else C.TEXT_DIM
            tip = '' if active else ' (no data at the aperture)'
            parts.append(f'<span style="color:{col}; font-size:14px;">●</span>'
                         f'<span style="color:{tcol};"> {text}{tip}</span>')
        html = '&nbsp;&nbsp;&nbsp;'.join(parts)
        if note:
            html += f'&nbsp;&nbsp;&nbsp;<span style="color:{C.TEXT_DIM};">flux: {note}</span>'
        self.legend.setText(html)

    def set_curves(self, xy, ylabel, message=''):
        self.xy = xy
        for b, c in self.curves.items():
            if b in xy:
                c.setData(*xy[b])
            else:
                c.setData([], [])
        if ylabel != self.ylabel:
            self.ylabel = ylabel
        self.plot.setLabel('left', ylabel, color=C.TEXT_DIM)
        self.msg.show_html(f'<span style="color:{C.TEXT_DIM}; font-size:10pt; font-style:italic;">{message}</span>'
                           if message else '')
        self.autoscale()

    def autoscale(self):
        if not self.xy or self.plot.getViewBox().y_manual:
            return
        lo, hi = self.plot.getViewBox().viewRange()[0]
        lims = [_robust_ylim(x, y, lo, hi, self.peaks_x, PEAK_TOL_UM * 1.2, q=(0.5, 99.5), head=0.18, foot=0.05)
                for x, y in self.xy.values()]
        lims = [lm for lm in lims if lm is not None]
        if lims:
            self.plot.setYRange(min(lm[0] for lm in lims), max(lm[1] for lm in lims), padding=0)

    def _hover(self, pos):
        vb = self.plot.getViewBox()
        if not self.xy or not vb.sceneBoundingRect().contains(pos):
            self.cross.hide()
            return
        x = vb.mapSceneToView(pos).x()
        self.cross.setPos(x)
        self.cross.show()
        vals = []
        for b, (xs, ys) in self.xy.items():
            if len(xs) < 2 or not (xs[0] <= x <= xs[-1]):
                continue
            i = int(np.clip(np.searchsorted(xs, x), 1, len(xs) - 1))
            i = i if abs(xs[i] - x) < abs(xs[i - 1] - x) else i - 1
            vals.append(f'{_fmt(ys[i], ".1f" if self.ylabel.startswith("S/N") else ".3g")} '
                        f'<span style="color:{band_color(b)};">{b}</span>')
        name = self.ylabel.split(' [')[0].split(' (')[0]
        lam = (f'λ<sub>rest</sub> = {x:.4f} µm (obs {x * (1 + self.rest):.4f})' if self.rest
               else f'λ = {x:.4f} µm')
        self.readout.setText(f'{lam} · {name} = ' + ' · '.join(vals) if vals else lam)


# ---------------------------------------------------------------------------------- measurements
class _Measurements(QWidget):
    """Table of the line fits of the current aperture (state.fits()) + kinematics line + export."""

    COLS = [('Line', 'Emission line (vacuum rest wavelength)'),
            ('Band', 'Band the line is measured in: the deepest band that covers it'),
            ('S/N', 'Gaussian-fit flux / its error (scripts/line_snr.py, calibrated)'),
            ('Flux', 'Fitted line flux'),
            ('Δv [km/s]', 'Velocity offset from the catalogue redshift'),
            ('σ_int [km/s]', 'Intrinsic velocity dispersion (instrumental LSF removed)'),
            ('W80 [km/s]', 'Width containing 80 % of the flux (2.563 σ_int for one Gaussian)'),
            ('Status', 'detected: S/N ≥ 3 (or the free anchor line)\n'
                       'free: velocity and width fitted · forced: fixed to the anchor kinematics')]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.meta = lambda: []          # set by the tab: comment lines for the export
        self._rows = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        head = QHBoxLayout()
        head.setContentsMargins(4, 0, 0, 0)
        head.addWidget(_section('Line measurements'))
        head.addSpacing(10)
        self.kin = _VariantLabel()
        self.kin.setToolTip('The anchor line is fitted with free velocity and width; all other lines are\n'
                            'fitted with velocity and width fixed to it (forced photometry, line_snr.py)')
        head.addWidget(self.kin, 1)
        css = 'padding: 3px 10px; font-size: 12px;'
        self.copy_btn = _flat_button('Copy', 'Copy the table to the clipboard (tab-separated, for Excel)', css)
        self.copy_btn.clicked.connect(self.copy)
        self.export_btn = _flat_button('Export CSV…', 'Save the table with the aperture and redshift as a CSV file', css)
        self.export_btn.clicked.connect(self.export)
        for b in (self.copy_btn, self.export_btn):
            head.addWidget(b)
        lay.addLayout(head)
        self.table = QTableWidget(len(FIT_LINES), len(self.COLS))
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(21)
        self.table.setShowGrid(False)
        self.table.setFocusPolicy(Qt.NoFocus)
        self.table.setStyleSheet('QTableWidget { font-size: 12px; }')
        for i, (name, tip) in enumerate(self.COLS):
            it = QTableWidgetItem(name)
            it.setToolTip(tip)
            self.table.setHorizontalHeaderItem(i, it)
        hh = self.table.horizontalHeader()
        hh.setHighlightSections(False)
        hh.setMinimumSectionSize(30)
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(QHeaderView.Fixed)     # widths set by _fit_columns
        self.table.setMinimumWidth(280)          # narrow window: horizontal scroll bar instead
        self.table.viewport().installEventFilter(self)
        lay.addWidget(self.table, 1)
        self.flux_unit = ''
        self.clear('')

    def eventFilter(self, obj, ev):
        if obj is self.table.viewport() and ev.type() == ev.Resize:
            self._fit_columns()
        return False

    def _fit_columns(self):
        """Every column at least as wide as its content; spare width is shared by the number
        columns.  When the table is too narrow it scrolls sideways - nothing is cut."""
        hh = self.table.horizontalHeader()
        n = len(self.COLS)
        need = [max(self.table.sizeHintForColumn(c), hh.sectionSizeHint(c)) + 8 for c in range(n)]
        spare = self.table.viewport().width() - sum(need)
        if spare > 0:
            mid = list(range(1, n - 1))
            for i, c in enumerate(mid):
                need[c] += spare // len(mid) + (1 if i < spare % len(mid) else 0)
        for c, w in enumerate(need):
            if hh.sectionSize(c) != w:
                hh.resizeSection(c, w)

    def set_flux_unit(self, unit):
        if unit == self.flux_unit:
            return
        self.flux_unit = unit
        self.table.horizontalHeaderItem(3).setText(f'Flux [{unit}·µm]' if unit else 'Flux')
        self._fit_columns()

    def clear(self, note):
        self._rows = []
        self.kin.set_variants([f'<span style="color:{C.TEXT_DIM};">{note}</span>'])
        for r, k in enumerate(FIT_LINES):
            self._set_row(r, k, None)
        self._enable_export(False)
        self._fit_columns()

    def set_fits(self, fits, info):
        a = info.anchor
        dim = f'color:{C.TEXT_DIM};'
        if info.forced_from:                 # annulus: every line forced to the centre's kinematics
            if a in LINE_LABELS:
                line = f'<b style="color:{LINE_COLORS.get(a, C.TEXT)};">{LINE_LABELS[a]}</b>'
                vals = f'Δv = {info.dv:+.0f} km/s, σ = {_fmt(info.sigma_int, ".0f")} km/s'
                self.kin.set_variants([
                    f'<span style="{dim}">ring: all lines forced to</span> {line}<span style="{dim}"> '
                    f'({info.anchor_band}) of the {info.forced_from}:</span> {vals}',
                    f'<span style="{dim}">forced to</span> {line}<span style="{dim}"> (centre):</span> {vals}',
                    f'{line} (centre): Δv {info.dv:+.0f}, σ {_fmt(info.sigma_int, ".0f")}'])
            else:
                self.kin.set_variants([f'<span style="{dim}">ring: the {info.forced_from} has no line strong '
                                       'enough for kinematics - all lines forced to the catalogue z</span>',
                                       f'<span style="{dim}">all lines forced to the catalogue z</span>'])
        elif a in LINE_LABELS:
            line = f'<b style="color:{LINE_COLORS.get(a, C.TEXT)};">{LINE_LABELS[a]}</b>'
            vals = f'Δv = {info.dv:+.0f} km/s, σ = {_fmt(info.sigma_int, ".0f")} km/s'
            self.kin.set_variants([
                f'<span style="{dim}">kinematics from</span> {line}<span style="{dim}"> ({info.anchor_band}):</span> {vals}',
                f'{line}<span style="{dim}"> ({info.anchor_band}):</span> {vals}',
                f'{line}: Δv {info.dv:+.0f}, σ {_fmt(info.sigma_int, ".0f")} km/s'])
        elif fits:
            zname = 'the entered z' if getattr(self, 'z_entered', False) else 'the catalogue z'
            self.kin.set_variants([f'<span style="{dim}">no line strong enough for free kinematics '
                                   f'- all lines forced to {zname}</span>',
                                   f'<span style="{dim}">all lines forced to {zname}</span>'])
        else:
            self.kin.set_variants([f'<span style="{dim}">no line of the fit list is covered</span>'])
        self._rows = []
        for r, k in enumerate(FIT_LINES):
            lf = fits.get(k)
            self._set_row(r, k, lf, 'not covered')
            if lf is not None:
                self._rows.append(lf)
        self._enable_export(bool(self._rows))
        self._fit_columns()

    def _set_row(self, r, key, lf, missing=''):
        def item(text, color=C.TEXT, tip='', bold=False, align=Qt.AlignRight | Qt.AlignVCenter):
            it = QTableWidgetItem(text)
            it.setForeground(QColor(color))
            it.setTextAlignment(align)
            if tip:
                it.setToolTip(tip)
            if bold:
                f = it.font()
                f.setBold(True)
                it.setFont(f)
            return it

        left = Qt.AlignLeft | Qt.AlignVCenter
        self.table.setItem(r, 0, item(f'●  {LINE_LABELS[key]}', LINE_COLORS.get(key, C.TEXT),
                                      f'rest {REST[key]:.6f} µm (vacuum)', align=left))
        if lf is None or not lf.covered:
            for c in range(1, 7):
                self.table.setItem(r, c, item('—', C.TEXT_DIM, align=Qt.AlignCenter if c == 1 else Qt.AlignRight | Qt.AlignVCenter))
            why = missing if lf is None else 'not covered'
            tip = (f'{LINE_LABELS[key]} lies outside the loaded bands (or too close to a band edge)'
                   if why else '')
            if lf is not None:
                tip = f'flags: {", ".join(lf.flags)}'
            self.table.setItem(r, 7, item(why, C.TEXT_DIM, tip, align=left))
            return
        bright = C.TEXT if lf.detected else C.TEXT_DIM
        flags = ', '.join(lf.flags) if lf.flags else 'none'
        self.table.setItem(r, 1, item(lf.band, band_color(lf.band), align=Qt.AlignCenter))
        self.table.setItem(r, 2, item(_fmt(lf.snr), C.EMERALD if lf.detected else C.TEXT_DIM,
                                      f'flags: {flags}', bold=lf.detected))
        self.table.setItem(r, 3, item(_fmt(lf.flux, '.2e'), bright,
                                      f'{_fmt(lf.flux, ".3e")} ± {_fmt(lf.flux_err, ".2e")}'))
        self.table.setItem(r, 4, item(_fmt(lf.dv_kms, '+.0f'), bright))
        self.table.setItem(r, 5, item(_fmt(lf.sigma_int_kms, '.0f'), bright,
                                      f'observed σ = {_fmt(lf.sigma_obs_kms, ".0f")} km/s'))
        self.table.setItem(r, 6, item(_fmt(lf.w80_kms, '.0f'), bright))
        status = ('detected' if lf.detected else 'not detected') + (' · forced' if lf.forced else ' · free')
        kin = f'kinematics: {lf.kinematics}\n' if lf.kinematics else ''
        self.table.setItem(r, 7, item(status, C.EMERALD if lf.detected else C.TEXT_DIM, f'{kin}flags: {flags}',
                                      align=left))

    def _enable_export(self, on):
        self.copy_btn.setEnabled(on)
        self.export_btn.setEnabled(on)

    # -- export
    def _records(self):
        head = ['line', 'band', 'snr', 'flux', 'flux_err', 'dv_kms', 'sigma_int_kms', 'w80_kms',
                'center_um', 'detected', 'forced', 'kinematics', 'flags']
        rows = [[lf.key, lf.band, lf.snr, lf.flux, lf.flux_err, lf.dv_kms, lf.sigma_int_kms, lf.w80_kms,
                 lf.center, int(lf.detected), int(lf.forced), lf.kinematics, ';'.join(lf.flags)] for lf in self._rows]
        return head, rows

    def _text(self, delimiter):
        head, rows = self._records()
        buf = io.StringIO()
        for m in self.meta():
            buf.write(f'# {m}\n')
        w = csv.writer(buf, delimiter=delimiter, lineterminator='\n')
        w.writerow(head)
        for r in rows:
            w.writerow(['' if isinstance(v, float) and not np.isfinite(v) else
                        (f'{v:.6g}' if isinstance(v, float) else v) for v in r])
        return buf.getvalue()

    def copy(self):
        QApplication.clipboard().setText(self._text('\t'))

    def export(self):
        name = getattr(self, 'default_name', 'lines.csv')
        path, _ = QFileDialog.getSaveFileName(self, 'Export line measurements', name, 'CSV files (*.csv)')
        if path:
            with open(path, 'w', encoding='utf-8', newline='') as f:
                f.write(self._text(','))


# ---------------------------------------------------------------------------------- the tab
class SpectrumTab(QWidget):
    """Image + aperture | line panels, full spectrum and line measurements of the current galaxy."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._gen = 0                 # incremented by every rebuild (part of the draw keys)
        self._dirty = True            # rebuild needed (done when the tab becomes visible)
        self._stale = False           # aperture changed while hidden
        self._spec_key = self._fit_key = None
        self._z = None                # redshift of the shown galaxy (state.z is a table lookup)
        self._data_gen = 0            # incremented for every new set of stacks (part of the cache keys)
        self._shape = 'circle'        # aperture shape chosen ('annulus' also with r_in = 0)
        self._unit_choice = 'px'      # radius unit chosen (shown in px while unavailable for a galaxy)
        self._unit_values = ('px', 0.0, 2.5)   # (unit, r_in, r) last set by the user: kept for the next galaxy
        self._spec_cache = OrderedDict()   # cache key -> {band: Spectrum}       (computed in a worker)
        self._fit_cache = OrderedDict()    # cache key + z -> (fits, FitInfo)     (computed in a worker)
        self._fit_wanted = False      # fits requested while the spectra were still being extracted
        self._jobs = _JobQueue(self._job_done)
        self._img_kind = 'white'
        self._img_band = None
        self._img_unit = ''
        self._img_cache = {}
        self._shown_ref = None
        self._build_ui()
        self._spec_timer = QTimer(self, singleShot=True, interval=SPEC_DEBOUNCE_MS, timeout=self._update_spectra)
        self._fit_timer = QTimer(self, singleShot=True, interval=FIT_DEBOUNCE_MS, timeout=self._update_fits)
        state.catalogLoading.connect(self._on_catalog_loading)
        state.galaxyChanged.connect(self._on_galaxy)
        state.loadFailed.connect(self._on_load_failed)
        state.stacksReady.connect(self._on_stacks)
        state.redshiftChanged.connect(self._on_redshift)
        state.optionsChanged.connect(self._on_option)
        state.apertureChanged.connect(self._on_aperture)
        if state.options.get('mode') != 'flux':
            state.set_option('mode', 'flux')
        self._sync_option_widgets()
        if state.stacks and state.ref_band in state.stacks:
            self._rebuild()
        else:
            self._show_empty('No galaxy loaded', 'Choose a galaxy in the list on the left.')

    # ================================================================== UI
    def _build_ui(self):
        self.pages = QStackedWidget()
        self.empty = EmptyState('No galaxy loaded')
        self.pages.addWidget(self.empty)
        split = QSplitter(Qt.Horizontal)
        split.setHandleWidth(6)
        split.setChildrenCollapsible(False)
        split.addWidget(self._build_left())
        split.addWidget(self._build_right())
        split.setStretchFactor(0, 35)
        split.setStretchFactor(1, 65)
        split.setSizes([440, 860])
        self.pages.addWidget(split)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 6, 0, 0)
        lay.addWidget(self.pages)

    def _build_left(self):
        card = QFrame()
        card.setProperty('role', 'card')
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 12)
        lay.setSpacing(7)

        self.img_seg = Segmented([(k, lab, tip) for k, lab, _, tip in IMAGE_KINDS])
        self.img_seg.changed.connect(self._image_kind_chosen)
        self.img_seg.set_current('white')
        lay.addWidget(self.img_seg, 0, Qt.AlignLeft)

        self.band_row = QWidget()
        br = QHBoxLayout(self.band_row)
        br.setContentsMargins(0, 0, 0, 0)
        br.setSpacing(8)
        lab = _dim('Band')
        lab.setToolTip('Reference band: the white-light / S/N image and the pixel grid of the aperture')
        br.addWidget(lab)
        self.band_seg = Segmented()
        self.band_seg.changed.connect(self.state.set_ref_band)
        br.addWidget(self.band_seg)
        br.addSpacing(6)
        self.grid_label = _dim('', shrink=True)
        self.grid_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        br.addWidget(self.grid_label, 1)
        lay.addWidget(self.band_row)

        self.image = ImagePanel()
        self.image.setMinimumHeight(200)
        self.image.setToolTip('Drag the circle (or click anywhere) to move the aperture; drag its handle to resize\n'
                              '(the outer radius; an annulus keeps its inner radius, dashed)')
        self.image.apertureMoved.connect(self._roi_moved)
        self.image.apertureResized.connect(self._roi_resized)
        self.image.pixelHovered.connect(self._image_hovered)
        lay.addWidget(self.image, 1)
        self.hover_label = _dim(' ', shrink=True)
        lay.addWidget(self.hover_label)

        lay.addSpacing(2)
        lay.addWidget(_section('Aperture'))
        row = QHBoxLayout()
        row.setSpacing(6)
        self.shape_seg = Segmented([('circle', 'Circle', 'Circular aperture of radius r'),
                                    ('annulus', 'Annulus', 'Ring r_in < d ≤ r_out around the aperture centre\n'
                                                           '(the spaxels of the circle inside are left out)')])
        self.shape_seg.changed.connect(self._shape_selected)
        row.addWidget(self.shape_seg)
        row.addStretch()
        self.unit_seg = Segmented(RADIUS_UNITS)
        self.unit_seg.set_current('px')
        self.unit_seg.changed.connect(self._unit_selected)
        row.addWidget(self.unit_seg)
        lay.addLayout(row)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.rin_label = _dim('r_in')
        row.addWidget(self.rin_label)
        self.rin_spin = self._radius_spin('Inner radius of the annulus (0: full circle)')
        row.addWidget(self.rin_spin)
        self.rout_label = _dim('r')
        row.addWidget(self.rout_label)
        self.rout_spin = self._radius_spin('Aperture radius (outer radius of an annulus)')
        row.addWidget(self.rout_spin)
        self.r_slider = QSlider(Qt.Horizontal)
        self.r_slider.setRange(int(R_MIN / R_STEP), int(R_MAX / R_STEP))
        self.r_slider.setPageStep(2)
        self.r_slider.setToolTip(f'Aperture (outer) radius {R_MIN:g}-{R_MAX:g} px in {R_STEP:g} px steps')
        self.r_slider.valueChanged.connect(self._slider_moved)
        row.addWidget(self.r_slider, 1)
        lay.addLayout(row)
        self.r_label = _dim('', shrink=True)
        self.r_label.setToolTip('Aperture radius in every unit')
        lay.addWidget(self.r_label)
        self.pos_label = _dim('', shrink=True)
        lay.addWidget(self.pos_label)
        row = QHBoxLayout()
        row.setSpacing(6)
        self.btn_ha = _flat_button('Centre on Hα', 'Move the aperture to the Hα peak (brightest Hα spaxel)')
        self.btn_ha.clicked.connect(lambda: self.state.center_on_line('Ha'))
        self.btn_oiii = _flat_button('Centre on [OIII]', 'Move the aperture to the [OIII]5007 peak')
        self.btn_oiii.clicked.connect(lambda: self.state.center_on_line('OIII5007'))
        row.addWidget(self.btn_ha)
        row.addWidget(self.btn_oiii)
        row.addStretch()
        self.ring_prev = _flat_button('◀ Ring', 'Move the annulus inwards by its own width')
        self.ring_prev.clicked.connect(lambda: self._step_ring(-1))
        self.ring_next = _flat_button('Ring ▶', 'Move the annulus outwards by its own width')
        self.ring_next.clicked.connect(lambda: self._step_ring(+1))
        row.addWidget(self.ring_prev)
        row.addWidget(self.ring_next)
        lay.addLayout(row)

        lay.addSpacing(2)
        lay.addWidget(_section('Display'))
        row = QHBoxLayout()
        self.mode_seg = Segmented([('snr', 'S/N', 'y axis: signal-to-noise per pixel (flux / error)'),
                                   ('flux', 'Flux', 'y axis: aperture flux (bands matched in their overlap\n'
                                                    'in the full spectrum)')])
        self.mode_seg.changed.connect(lambda k: self.state.set_option('mode', k))
        row.addWidget(self.mode_seg)
        row.addStretch()
        lab = _dim('Smooth')
        row.addWidget(lab)
        self.smooth_spin = QSpinBox()
        self.smooth_spin.setRange(0, 5)
        self.smooth_spin.setSuffix(' px')
        self.smooth_spin.setSpecialValueText('off')
        self.smooth_spin.setToolTip('Boxcar smoothing of the displayed spectra (the fits use the raw spectrum)')
        self.smooth_spin.valueChanged.connect(lambda v: self.state.set_option('smooth', int(v)))
        row.addWidget(self.smooth_spin)
        lay.addLayout(row)
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)
        self.checks = {}
        for i, (name, text, tip) in enumerate([
                ('subtract_bg', 'Subtract residual sky',
                 'Subtract, per wavelength, the median of the spaxels ≥ 6 px from the aperture\n'
                 '(SUBTRACT_BG of the notebooks; removes a residual OH pedestal)'),
                ('telluric', 'Telluric bands', 'Shade wavelengths where the atmosphere transmits < 50 %'),
                ('sky_lines', 'Sky lines', 'Mark the bright OH sky lines (UVES/Oliva atlas, vacuum)'),
                ('rest_frame', 'Rest frame', 'Wavelength axis in the rest frame of the galaxy (λ / (1+z))')]):
            cb = QCheckBox(text)
            cb.setToolTip(tip)
            cb.toggled.connect(lambda on, n=name: self.state.set_option(n, bool(on)))
            grid.addWidget(cb, i // 2, i % 2)
            self.checks[name] = cb
        lay.addLayout(grid)
        return card

    def _build_right(self):
        split = QSplitter(Qt.Vertical)
        split.setHandleWidth(6)
        split.setChildrenCollapsible(False)

        box = QWidget()
        bl = QVBoxLayout(box)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(2)
        head = QHBoxLayout()
        head.setContentsMargins(4, 0, 4, 0)
        head.addWidget(_section('Emission lines'))
        head.addSpacing(10)
        self.panels_info = _VariantLabel()
        self.panels_info.setToolTip('Scroll to zoom the wavelength axis, drag to pan; right-click → View All.\n'
                                    'Amber: line fit (continuum + Gaussians) of the current aperture,\n'
                                    'dashed while the fit of a new aperture is being computed.\n'
                                    'Badge: S/N of the main line (green: detected, S/N ≥ 3).')
        head.addWidget(self.panels_info, 1)
        bl.addLayout(head)
        self.panels_view = pg.GraphicsLayoutWidget()
        self.panels_view.setBackground(C.BG_MEDIUM)
        self.panels_view.ci.setContentsMargins(0, 0, 0, 0)
        self.panels_view.ci.setSpacing(4)
        self.panels_view.ci.layout.setHorizontalSpacing(14)
        self.panels = []
        for i, spec in enumerate(PANELS):
            cell = self.panels_view.addLayout(row=i // 2, col=i % 2)
            self.panels.append(_LinePanel(cell, spec))
        bl.addWidget(self.panels_view, 1)
        split.addWidget(_card(box, (8, 8, 8, 6)))

        self.full = _FullSpectrum()
        split.addWidget(_card(self.full, (8, 8, 8, 6)))

        self.table = _Measurements()
        self.table.meta = self._export_meta
        split.addWidget(_card(self.table, (8, 8, 8, 8)))
        split.setStretchFactor(0, 50)
        split.setStretchFactor(1, 28)
        split.setStretchFactor(2, 22)
        split.setSizes([390, 220, 220])
        return split

    def _show_empty(self, text, hint=''):
        self.empty.set_text(text, hint)
        self.pages.setCurrentIndex(0)

    # ================================================================== state -> view
    def _on_catalog_loading(self, variant):
        if self.pages.currentIndex() == 0:
            self._show_empty(f'Scanning {variant} …', 'Reading the cube headers (cached after the first time).')

    def _on_galaxy(self, key):
        self._spec_timer.stop()
        self._fit_timer.stop()
        self._jobs.drop_pending()
        self.table._enable_export(False)     # the table still shows the previous galaxy
        # never leave the previous galaxy on screen under the new galaxy's card
        self._show_empty(f'Loading {key} …')

    def _on_load_failed(self, key):
        self._show_empty(f'The cubes of {key} could not be loaded',
                         'See the error message; select the galaxy again in the list to retry.')

    def _on_stacks(self, _key):
        self._apply_unit_radii()          # e.g. 0.5-1 Rₑ stays 0.5-1 Rₑ of the new galaxy
        self._sync_telluric_check()
        if len(self._img_cache) > 60:
            self._img_cache.clear()
        self._data_gen += 1
        self._rebuild()

    def _on_redshift(self):
        self._apply_unit_radii()          # kpc (and the Rₑ measured here) depend on z
        self.table._enable_export(False)
        self._rebuild()

    def _on_option(self, name):
        self._sync_option_widgets()
        shown = self.isVisible() and not self._dirty and self.pages.currentIndex() == 1 and self._ready()
        if name == 'smooth' and shown:            # display only: redraw the curves
            self._spec_key = None
            self._update_spectra()
        elif name == 'mode' and shown:            # S/N <-> flux: curves, fits and axis names
            self._spec_key = self._fit_key = None
            for vb in [p.plot.getViewBox() for p in self.panels] + [self.full.plot.getViewBox()]:
                vb.y_manual = False               # a y range in the other quantity means nothing
            self._update_panels_info()
            self._update_spectra()
            self._update_fits()
        elif name in ('method', 'subtract_bg') and shown:     # new spectra, same plots
            self._update_spectra()
            self._update_fits()
        elif name in ('mode', 'method', 'subtract_bg', 'telluric', 'sky_lines', 'rest_frame', 'smooth'):
            self._rebuild(keep_x=name != 'rest_frame')

    def _on_aperture(self):
        s = self.state
        if not self._ready():
            return
        if not self.isVisible():
            self._stale = True
            return
        if s.ref_band != self._shown_ref:
            self._sync_bands()
            self._update_image()
        else:
            self._place_roi()
        self._sync_radius()
        self._spec_timer.start()
        self._fit_timer.start(FIT_DEBOUNCE_MS)

    def showEvent(self, ev):
        super().showEvent(ev)
        if self._dirty:
            self._rebuild()
        elif self._stale:
            self._stale = False
            self._on_aperture()

    def _ready(self):
        s = self.state
        return bool(s.stacks) and s.ref_band in s.stacks and s.ap_x is not None

    # ================================================================== rebuild
    def _rebuild(self, keep_x=False):
        if not self.isVisible():
            self._dirty = True
            return
        if not self._ready():
            if self.state.key is not None and not self.state.stacks:
                self._show_empty(f'Loading {self.state.key} …')
            return
        self._dirty = self._stale = False
        self._spec_timer.stop()
        self._fit_timer.stop()
        same = self.pages.currentIndex() == 1 and self._shown_key == self.state.key
        keep_x = keep_x and same
        self.pages.setCurrentIndex(1)
        self._gen += 1
        self._shown_key = self.state.key
        self._z = self.state.z
        try:
            self._sync_bands()
            self._sync_image_kinds()
            self._sync_buttons()
            self._sync_radius()
            self._update_image()
            self._setup_panels(keep_x, keep_data=keep_x)
            self._setup_full(keep_x, keep_data=keep_x)
            self._update_panels_info()
            self._update_spectra()      # from the cache at once, else extracted in the background
            self._update_fits()
        except Exception:          # never leave a half-drawn tab with a traceback in a Qt slot
            traceback.print_exc()
            self.state.status.emit('Spectrum view: could not draw this galaxy (see console)', 6000)

    _shown_key = None

    def _frame(self):
        """(divisor for x values, z of the rest frame or None)."""
        z = self._z
        if self.state.options.get('rest_frame') and z:
            return 1.0 + z, z
        return 1.0, None

    def _xlabel(self):
        _, rz = self._frame()
        if rz:
            return 'Rest-frame wavelength [µm]'
        if self.state.options.get('rest_frame') and not self._z:
            return 'Observed wavelength [µm]  (no z: rest frame unavailable)'
        return 'Observed wavelength [µm]'

    def _setup_panels(self, keep_x, keep_data=False):
        s = self.state
        z = self._z
        f, _ = self._frame()
        sky = analysis.sky_lines() if s.options.get('sky_lines') else np.array([])
        for p in self.panels:
            if z is None:
                p.placeholder('no redshift')
                continue
            c, lo, hi = lines.panel_window(p.spec, z)
            band = analysis.pick_band(s.stacks, c)
            if band is None:
                p.placeholder(f'not covered by the loaded bands\n(λ = {c:.3f} µm)')
                continue
            st = s.stacks[band]
            centre_rest = c / (1 + z)
            markers = []
            for k in p.line_keys:
                bold = k in p.main_keys
                left = REST[k] < centre_rest - 1e-6
                markers.append((REST[k] * (1 + z) / f, MARKER_LABELS.get(k, LINE_LABELS[k]),
                                LINE_COLORS.get(k, C.TEXT_DIM), bold, left))
            margin = hi - lo
            spans = []
            if s.options.get('telluric'):
                spans = [(max(a, lo - margin) / f, min(b, hi + margin) / f)
                         for a, b in analysis.telluric_spans(st) if b > lo - margin and a < hi + margin]
            sk = sky[(sky > lo - margin) & (sky < hi + margin)] / f if len(sky) else sky
            p.tol = PEAK_TOL_UM / f
            p.configure(band, (lo / f, hi / f), markers, spans, sk, keep_x=keep_x and p.band == band,
                        keep_data=keep_data)
        self.table.set_flux_unit(_unit(s.stacks[s.ref_band]))

    def _update_panels_info(self):
        """Header of the line panels: axis units (the panels have no axis labels) and the legend."""
        s = self.state
        dim = f'color:{C.TEXT_DIM};'
        _, rz = self._frame()
        xl = 'rest-frame λ [µm]' if rz else 'observed λ [µm]'
        flux = s.options.get('mode') == 'flux'
        yl = f'flux [{_unit(s.stacks[s.ref_band])}]' if flux else 'S/N per pixel'
        fit = f'<span style="color:{MODEL_COLOR};">━</span><span style="{dim}"> line fit</span>'
        if self._z is None:
            self.panels_info.set_variants([f'<span style="{dim}">no redshift - enter one in the galaxy card '
                                           'to see the line panels</span>',
                                           f'<span style="{dim}">no redshift</span>'])
            return
        self.panels_info.set_variants([
            f'<span style="{dim}">x: {xl} · y: {yl} · </span>{fit}'
            f'<span style="{dim}"> · scroll: zoom λ · Shift+scroll: zoom y · double-click: auto y</span>',
            f'<span style="{dim}">x: {xl} · y: {yl} · </span>{fit}',
            f'<span style="{dim}">{xl} · {yl} · </span>{fit}',
            f'<span style="{dim}">{xl} · {yl}</span>'])

    def _setup_full(self, keep_x, keep_data=False):
        s = self.state
        z = self._z
        f, rz = self._frame()
        bands = sorted(s.stacks, key=band_rank)
        ranges = [(s.stacks[b].wave_min, s.stacks[b].wave_max) for b in bands
                  if np.isfinite(s.stacks[b].wave_min)]
        xrange = (min(r[0] for r in ranges) / f, max(r[1] for r in ranges) / f) if ranges else None
        spans = []
        if s.options.get('telluric'):
            spans = _merge_spans([sp for b in bands for sp in analysis.telluric_spans(s.stacks[b])])
            spans = [(a / f, b / f) for a, b in spans]
        sky = np.array([])
        if s.options.get('sky_lines') and ranges:
            sl = analysis.sky_lines()
            sky = sl[np.any([(sl >= a) & (sl <= b) for a, b in ranges], axis=0)] / f if len(sl) else sl
        markers = []
        if z:
            last_x, level = -np.inf, 0
            width = (xrange[1] - xrange[0]) if xrange else 1.0
            for label, k in FULL_MARKERS:
                obs = REST[k] * (1 + z)
                if not any(a <= obs <= b for a, b in ranges):
                    continue
                x = obs / f
                if label:
                    level = (level + 1) % 2 if x - last_x < 0.05 * width else 0
                    last_x = x
                markers.append((x, label, LINE_COLORS.get(k, C.TEXT_DIM), 0.95 - 0.09 * level))
        self.full.rest = rz
        old = dict(self.full.xy) if keep_data else {}
        self.full.configure(bands, xrange, markers, spans, sky, self._xlabel(), keep_x=keep_x)
        if old:            # keep showing the previous spectra until the new ones are extracted
            self.full.set_curves({b: xy for b, xy in old.items() if b in self.full.curves}, self.full.ylabel)

    # ================================================================== controls
    def _sync_bands(self):
        s = self.state
        bands = sorted(s.stacks, key=band_rank)
        if self.band_seg.keys() != bands:
            self.band_seg.set_items([(b, b, f'{s.stacks[b].describe()} · {s.stacks[b].wave_min:.3f}-'
                                             f'{s.stacks[b].wave_max:.3f} µm') for b in bands])
        self.band_seg.set_current(s.ref_band)
        self._shown_ref = s.ref_band

    def _sync_image_kinds(self):
        s = self.state
        z = self._z
        ref = s.stacks[s.ref_band]
        for key, label, keys, tip in IMAGE_KINDS:
            if key == 'white':
                ok, why = True, tip
            elif key == 'snr':
                ok = ref.stat is not None
                why = tip if ok else 'No STAT (error) extension in this band'
            elif z is None:
                ok, why = False, f'{label} map: no redshift'
            else:
                band = analysis.pick_band(s.stacks, REST[keys[0]] * (1 + z))
                ok = band is not None
                why = f'{tip}\nfrom the {band} band' if ok else \
                    f'{label} is not covered by the loaded bands (λ = {REST[keys[0]] * (1 + z):.3f} µm)'
            self.img_seg.set_item_enabled(key, ok, why)
            if not ok and self._img_kind == key:
                self._img_kind = 'white'
        self.img_seg.set_current(self._img_kind)

    def _sync_buttons(self):
        s = self.state
        z = self._z
        for btn, key in ((self.btn_ha, 'Ha'), (self.btn_oiii, 'OIII5007')):
            covered = bool(z) and analysis.pick_band(s.stacks, REST[key] * (1 + z)) is not None
            btn.setEnabled(covered)
            label = LINE_LABELS[key]
            btn.setToolTip(f'Move the aperture to the {label} peak (brightest line spaxel)' if covered else
                           (f'{label}: no redshift' if not z else f'{label} is not covered by the loaded bands'))

    # ---------------------------------------------------------------- aperture radius and units
    def _radius_spin(self, tip):
        sp = QDoubleSpinBox()
        sp.setLocale(QLocale.c())               # decimal point, as everywhere else in the app
        sp.setDecimals(2)
        sp.setKeyboardTracking(False)           # typed values apply on Enter / focus change
        sp.setToolTip(tip)
        sp.setMinimumWidth(72)
        sp.valueChanged.connect(self._spin_changed)
        return sp

    def _px_per_unit(self, unit):
        """Pixels of the reference band per radius unit; NaN if the unit is not available (no z, no Rₑ)."""
        s = self.state
        pix = s.stacks[s.ref_band].pixscale if s.ref_band in s.stacks else 0.2
        if unit == 'arcsec':
            return 1.0 / pix
        if unit == 'kpc':
            k = analysis.kpc_per_arcsec(s.z)
            return 1.0 / (pix * k) if np.isfinite(k) and k > 0 else np.nan
        if unit == 're':
            re, _ = s.effective_radius() if s.stacks else (np.nan, '')
            return re / pix if np.isfinite(re) and re > 0 else np.nan
        return 1.0

    def _unit(self):
        """The unit the radii are shown in: the chosen one, or px while it is not available for this galaxy
        (the choice itself is kept for the next galaxy)."""
        u = self._unit_choice
        return u if np.isfinite(self._px_per_unit(u)) else 'px'

    def _remember_radii(self):
        """After a radius change by the user: keep it in the chosen unit (re-applied to the next galaxy).
        While that unit is not available, the previous request stays."""
        k = self._px_per_unit(self._unit_choice)
        if np.isfinite(k):
            s = self.state
            self._unit_values = (self._unit_choice, s.ap_rin / k, s.ap_r / k)

    def _shape_selected(self, key):
        s = self.state
        self._shape = key
        if key == 'annulus' and s.ap_rin <= 0:
            r_in = max(R_STEP, round(s.ap_r / 2 / R_STEP) * R_STEP)
            s.set_aperture(r=max(s.ap_r, r_in + R_STEP), r_in=r_in)
        elif key == 'circle' and s.ap_rin > 0:
            s.set_aperture(r_in=0.0)
        self._remember_radii()
        self._sync_radius()

    def _unit_selected(self, key):
        self._unit_choice = key
        self._remember_radii()
        self._sync_radius()

    def _spin_changed(self, _v=None):
        k = self._px_per_unit(self._unit())
        r_out = float(np.clip(self.rout_spin.value() * k, R_MIN, R_MAX))
        r_in = float(np.clip(self.rin_spin.value() * k, 0.0, R_MAX - R_STEP)) if self._shape == 'annulus' else 0.0
        self.state.set_aperture(r=r_out, r_in=r_in)
        self._remember_radii()

    def _slider_moved(self, v):
        self.state.set_aperture(r=v * R_STEP)
        self._remember_radii()

    def _step_ring(self, direction):
        s = self.state
        width = s.ap_r - s.ap_rin
        if width <= 0:
            return
        if direction > 0:
            if s.ap_r + width > R_MAX + 1e-6:
                return
            s.set_aperture(r=s.ap_r + width, r_in=s.ap_r)
        else:
            if s.ap_rin <= 0:
                return
            s.set_aperture(r_in=max(0.0, s.ap_rin - width), r=s.ap_rin)
        self._remember_radii()

    def _apply_unit_radii(self):
        """A new galaxy (or z): radii chosen in arcsec, kpc or Rₑ keep their value in that unit.  The request
        itself is not changed (a radius clipped to R_MIN/R_MAX here is not clipped for the next galaxy)."""
        u, rin_u, rout_u = self._unit_values
        if u == 'px' or not self.state.stacks:
            return
        k = self._px_per_unit(u)
        if not np.isfinite(k):
            return                                 # unit not available here: keep the pixel radii
        r_out = float(np.clip(rout_u * k, R_MIN, R_MAX))
        r_in = float(np.clip(rin_u * k, 0.0, r_out - R_STEP)) if self._shape == 'annulus' else 0.0
        self.state.set_aperture(r=r_out, r_in=r_in)

    def _sync_radius(self):
        """Show the state's radii in the chosen unit (widgets set without signals: no feedback)."""
        s = self.state
        r, r_in = s.ap_r, s.ap_rin
        if r_in > 0:
            self._shape = 'annulus'
        ring = self._shape == 'annulus'
        pix = s.stacks[s.ref_band].pixscale if s.ref_band in s.stacks else 0.2
        kpc = analysis.kpc_per_arcsec(s.z)
        re, re_src = s.effective_radius() if s.stacks else (np.nan, '')
        self.unit_seg.set_item_enabled('kpc', bool(np.isfinite(kpc)),
                                       RADIUS_UNITS[2][2] if np.isfinite(kpc) else 'kpc: needs a redshift')
        self.unit_seg.set_item_enabled('re', bool(np.isfinite(re)),
                                       f'{RADIUS_UNITS[3][2]}\nRₑ = {re:.2f}″ ({re_src})' if np.isfinite(re)
                                       else 'Rₑ: not known for this galaxy (no line half-light radius)')
        u = self._unit()
        self.unit_seg.set_current(u)
        self.shape_seg.set_current('annulus' if ring else 'circle')
        k = self._px_per_unit(u)
        widgets = (self.r_slider, self.rin_spin, self.rout_spin)
        for w in widgets:
            w.blockSignals(True)
        try:
            self.r_slider.setValue(int(round(np.clip(r, R_MIN, R_MAX) / R_STEP)))
            for sp, lo, hi, val in ((self.rin_spin, 0.0, R_MAX - R_STEP, r_in), (self.rout_spin, R_MIN, R_MAX, r)):
                sp.setRange(lo / k, hi / k)
                sp.setSingleStep(UNIT_STEP[u])
                sp.setSuffix(UNIT_TEXT[u])
                sp.setValue(val / k)
        finally:
            for w in widgets:
                w.blockSignals(False)
        for w in (self.rin_label, self.rin_spin, self.ring_prev, self.ring_next):
            w.setVisible(ring)
        self.rout_label.setText('r_out' if ring else 'r')
        if ring:
            self.ring_prev.setEnabled(r_in > 0)
            self.ring_next.setEnabled(r + (r - r_in) <= R_MAX + 1e-6)

        def fmt(v, unit_px):
            return f'{v / unit_px:.2f}' if np.isfinite(unit_px) else '?'
        parts = []
        for name, upx in ((' px', 1.0), ('″', 1.0 / pix), (' kpc', 1.0 / (pix * kpc) if np.isfinite(kpc) else np.nan),
                          (' Rₑ', re / pix if np.isfinite(re) else np.nan)):
            if np.isfinite(upx):
                parts.append(f'{fmt(r_in, upx)}–{fmt(r, upx)}{name}' if ring else f'{fmt(r, upx)}{name}')
        text = ('annulus ' if ring else 'r = ') + ' · '.join(parts)
        if np.isfinite(re):
            short = {'measured in the notebooks': 'measured'}.get(re_src, re_src)
            text += f'  (Rₑ = {re:.2f}″, {short})'
        self.r_label.setText(text)
        tip = 'Aperture radius in every unit'
        if np.isfinite(re):
            tip += f'\nRₑ = {re:.2f}″' + (f' = {re * kpc:.2f} kpc' if np.isfinite(kpc) else '') + f' ({re_src})'
        if np.isfinite(kpc):
            tip += f'\n1″ = {kpc:.2f} kpc at z = {s.z:.4f} (flat ΛCDM, H₀ = 70, Ωm = 0.3)'
        if self._unit_choice != u:
            tip += f'\n{dict((k_, l_) for k_, l_, _ in RADIUS_UNITS)[self._unit_choice]} is not available for this ' \
                   'galaxy: shown in px; your radii in that unit come back with the next galaxy that has it'
        self.r_label.setToolTip(tip)

    def _sync_option_widgets(self):
        o = self.state.options
        self.mode_seg.set_current(o.get('mode', 'flux'))
        for name, cb in self.checks.items():
            cb.blockSignals(True)
            cb.setChecked(bool(o.get(name)))
            cb.blockSignals(False)
        self.smooth_spin.blockSignals(True)
        self.smooth_spin.setValue(int(o.get('smooth') or 0))
        self.smooth_spin.blockSignals(False)
        self._sync_telluric_check()

    def _sync_telluric_check(self):
        """'Telluric bands' only makes sense when the stacks carry a transmission (TELLEFF files)."""
        cb = self.checks.get('telluric')
        if cb is None or not self.state.stacks:
            return
        has = any(st.atm is not None for st in self.state.stacks.values())
        cb.setEnabled(has)
        cb.setToolTip('Shade wavelengths where the atmosphere transmits < 50 %' if has else
                      'No TELLEFF transmission files in this reduction - telluric bands cannot be shaded')

    # ================================================================== image + aperture
    def _image_kind_chosen(self, key):
        self._img_kind = key
        if self._ready():
            self._update_image()

    def _image_content(self):
        """(band, image, title, colour map, unit) for the chosen image kind."""
        s = self.state
        kind = self._img_kind
        ref = s.ref_band
        entry = next(e for e in IMAGE_KINDS if e[0] == kind)
        if kind == 'white':
            st = s.stacks[ref]
            return ref, st.white, f'White light · {ref}', 'magma', _unit(st)
        if kind == 'snr':
            ck = (s.key, ref, 'snr')
            if ck not in self._img_cache:
                self._img_cache[ck] = analysis.snr_map(s.stacks[ref])
            return ref, self._img_cache[ck], f'Median S/N · {ref}', 'viridis', ''
        z = self._z
        keys = entry[2]
        band = analysis.pick_band(s.stacks, REST[keys[0]] * (1 + z)) if z else None
        if band is None:
            st = s.stacks[ref]
            return ref, st.white, f'White light · {ref}', 'magma', _unit(st)
        ck = (s.key, band, kind, round(z, 6))
        if ck not in self._img_cache:
            maps = [analysis.line_map(s.stacks[band], z, k) for k in keys]
            maps = [m for m in maps if m is not None]
            self._img_cache[ck] = np.sum(maps, axis=0) if maps else None
        return band, self._img_cache[ck], f'{entry[1]} map · {band}', 'magma', _unit(s.stacks[band])

    def _update_image(self):
        s = self.state
        try:
            band, img, title, cmap, unit = self._image_content()
        except Exception:
            traceback.print_exc()
            band, img, title, cmap, unit = s.ref_band, None, 'Image', 'magma', ''
        self._img_band, self._img_unit = band, unit
        self.image.set_image(img, cmap=cmap, title=title)
        st = s.stacks[band]
        ny, nx = st.cube.shape[1:]
        note = '' if band == s.ref_band else f' · aperture from {s.ref_band}'
        self.grid_label.setText(f'{nx} × {ny} px · {st.pixscale:.2f}″/px{note}')
        self.grid_label.setToolTip(f'Image grid: {band} stack ({st.describe()})')
        self.img_seg.set_current(self._img_kind)
        self.hover_label.setText(' ')
        self._place_roi()

    def _place_roi(self):
        s = self.state
        band = self._img_band
        if band not in s.stacks or self.image.data is None:
            self.image.roi.hide()
            return
        x, y, r, r_in = (s.ap_x, s.ap_y, s.ap_r, s.ap_rin) if band == s.ref_band else s.annulus_in(band)
        self.image.set_aperture(x, y, r, r_in)

    def _roi_moved(self, x, y):
        s = self.state
        if not self._ready() or self._img_band not in s.stacks:
            return
        if self._img_band != s.ref_band:
            x, y, _ = analysis.transfer(s.stacks[self._img_band], s.stacks[s.ref_band], x, y)
        if abs(x - s.ap_x) > 1e-3 or abs(y - s.ap_y) > 1e-3:
            s.set_aperture(x, y)

    def _roi_resized(self, r):
        s = self.state
        if not self._ready() or self._img_band not in s.stacks:
            return
        r *= s.stacks[self._img_band].pixscale / s.stacks[s.ref_band].pixscale
        r = float(np.clip(r, R_MIN, R_MAX))
        if abs(r - s.ap_r) > 1e-3:
            s.set_aperture(r=r)
            self._remember_radii()

    def _image_hovered(self, ix, iy, v):
        if self._img_kind == 'snr':
            val = f'S/N = {_fmt(v, ".2f")}'
        else:
            val = f'{_fmt(v, ".3g")} {self._img_unit}'
        self.hover_label.setText(f'pixel x = {ix}, y = {iy} ({self._img_band}) · {val}')

    # ================================================================== spectra
    def _ap_key(self):
        s = self.state
        return (s.key, s.ref_band, round(s.ap_x, 2), round(s.ap_y, 2), round(s.ap_r, 6), round(s.ap_rin, 6),
                s.options.get('method'), s.options.get('subtract_bg'), self._z)

    def _cache_key(self):
        """Key of the spectra of the current aperture (data generation + aperture + options)."""
        s = self.state
        return (self._data_gen, s.key, s.ref_band, round(s.ap_x, 2), round(s.ap_y, 2), round(s.ap_r, 6),
                round(s.ap_rin, 6), s.options.get('method'), bool(s.options.get('subtract_bg')))

    def _fit_cache_key(self):
        return self._cache_key() + (round(self._z, 6),)

    @staticmethod
    def _remember(cache, key, value, limit=40):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)

    def _request_spectra(self, ck):
        s = self.state
        self._jobs.request(('spec', ck), _extract_all, s.stacks, s.ref_band, s.ap_x, s.ap_y, s.ap_r,
                           s.options.get('method', 'sum'), bool(s.options.get('subtract_bg')), s.ap_rin)

    def _job_done(self, jkey, res, err):
        """A background extraction or fit finished (GUI thread)."""
        kind, ck = jkey
        if err:
            print(f'Spectrum view: background {kind} failed:\n{err}')
            self.state.status.emit(f'Spectrum view: the {"spectra" if kind == "spec" else "line fits"} '
                                   'could not be computed (see console)', 6000)
            res = {} if kind == 'spec' else ({}, analysis.FitInfo())
        self._remember(self._spec_cache if kind == 'spec' else self._fit_cache, ck, res)
        if not self._ready() or self._dirty or not self.isVisible():
            return
        if kind == 'spec' and ck == self._cache_key():
            self._update_spectra()
            if self._fit_wanted and not self._fit_timer.isActive():
                self._update_fits()
        elif kind == 'fit' and self._z is not None and ck == self._fit_cache_key():
            self._update_fits()

    def _update_spectra(self):
        if not self._ready() or self._dirty or not self.isVisible():
            return
        key = (self._gen, self._ap_key())
        if key == self._spec_key:
            return
        s = self.state
        ck = self._cache_key()
        spectra = self._spec_cache.get(ck)
        if spectra is None:                 # extract in the background; drawn by _job_done
            self._request_spectra(ck)
            for p in self.panels:           # the drawn fit belongs to the previous aperture
                p.fade_model()
            if not self.full.xy:
                self.full.msg.show_html(f'<span style="color:{C.TEXT_DIM}; font-size:10pt; font-style:italic;">'
                                        'extracting the aperture spectra …</span>')
            return
        self._spec_key = key
        snr = s.options.get('mode') != 'flux'
        sm = int(s.options.get('smooth') or 0)
        f, _ = self._frame()
        disp = {b: (sp.wave / f, boxcar(sp.snr if snr else sp.flux, sm)) for b, sp in spectra.items()}

        for p in self.panels:
            if p.band is None:
                continue
            if p.band in disp:
                p.set_curve(*disp[p.band])
            else:
                p.outside(self._empty_text(p.band))

        ylabel = 'S/N' if snr else f'Flux [{_unit(s.stacks[s.ref_band])}]'
        note = ''
        if not snr and len(spectra) > 1:
            ref = s.ref_band if s.ref_band in spectra else next(iter(spectra))
            scales = analysis.match_band_scales(spectra, ref)
            moved = {b: v for b, v in scales.items() if abs(v - 1.0) > 1e-6}
            if moved:
                disp = {b: (x, y * scales[b]) for b, (x, y) in disp.items()}
                note = ', '.join(f'{b} ×{v:.2f}' for b, v in moved.items()) + f' (matched to {ref})'
        msg = '' if spectra else self._empty_text()
        self.full.set_curves(disp, ylabel, msg)
        self.full.set_legend([(b, s.stacks[b].describe(), b in spectra) for b in sorted(s.stacks, key=band_rank)],
                             note)
        self._update_pos_label(spectra)
        if self._fit_key != key:
            for p in self.panels:
                p.fade_model()

    def _ring_empty(self, band=None):
        """True if the annulus has no spaxel centre although its outer circle lies on the data."""
        s = self.state
        band = band or s.ref_band
        if s.ap_rin <= 0 or band not in s.stacks:
            return False
        st = s.stacks[band]
        x, y, r, _ = s.annulus_in(band)
        return bool((analysis.aperture_mask(st.cube.shape[1:], x, y, r) & (st.coverage > 0.5)).any())

    def _empty_text(self, band=None):
        if self._ring_empty(band):
            return 'no spaxel centre lies in the ring - widen it'
        return f'aperture outside the {band} field' if band else 'the aperture lies outside the data of all bands'

    def _sky_note(self, spectra):
        """Bands where 'Subtract residual sky' is on but could not be done (too few spaxels outside the aperture)."""
        if not self.state.options.get('subtract_bg'):
            return []
        return [b for b, sp in spectra.items() if not sp.bg_subtracted]

    def _update_pos_label(self, spectra):
        s = self.state
        n = spectra[s.ref_band].n_pix if s.ref_band in spectra else 0
        extra = '' if n else (' · no spaxel centre in the ring' if self._ring_empty() else ' · outside the data')
        missed = self._sky_note(spectra)
        if missed:
            extra += f' · residual sky not subtracted in {"/".join(missed)} (too few spaxels ≥ r + 2 px away)'
        self.pos_label.setText(f'centre x = {s.ap_x:.1f}, y = {s.ap_y:.1f} px ({s.ref_band}) · '
                               f'{n} spaxels{extra}')

    # ================================================================== fits
    def _update_fits(self):
        if not self._ready() or self._dirty or not self.isVisible():
            return
        key = (self._gen, self._ap_key())
        if key == self._fit_key:
            return
        s = self.state
        if self._z is None:
            self._fit_key = key
            self._fit_wanted = False
            for p in self.panels:
                p.set_model(None, None)
                p.set_badge(None)
            self.table.clear('no redshift: the lines can not be fitted')
            return
        spectra = self._spec_cache.get(self._cache_key())
        if spectra is None:                 # spectra still being extracted: fit when they arrive
            self._fit_wanted = True
            self._request_spectra(self._cache_key())
            return
        self._fit_wanted = False
        fck = self._fit_cache_key()
        res = self._fit_cache.get(fck)
        if res is None:
            if s.ap_rin > 0:                # annulus: lines forced to the central aperture's kinematics
                self._jobs.request(('fit', fck), analysis.fit_annulus, spectra, s.stacks, self._z, s.ref_band,
                                   s.ap_x, s.ap_y, s.options.get('method', 'sum'), bool(s.options.get('subtract_bg')))
            else:
                self._jobs.request(('fit', fck), analysis.fit_galaxy, spectra, s.stacks, self._z)
            return
        if self._spec_key != key:          # make sure the drawn spectra belong to this aperture
            self._update_spectra()
        self._fit_key = key
        fits, info = res
        snr = s.options.get('mode') != 'flux'
        f, _ = self._frame()
        for p in self.panels:
            if p.band is None or p.band not in spectra:
                continue
            try:
                self._draw_fit(p, fits, spectra[p.band], snr, f)
            except Exception:
                traceback.print_exc()
                p.set_model(None, None)
        snap = self._export_meta()          # metadata of exactly the result drawn now
        self.table.meta = lambda: snap
        self.table.z_entered = bool(s.redshift is not None and s.redshift.override)
        if spectra:
            self.table.set_fits(fits, info)
        else:
            self.table.clear(self._empty_text())
        self.table.default_name = f'{(s.key or "galaxy").replace(" ", "_")}_lines.csv'

    def _draw_fit(self, p, fits, sp, snr, f):
        """Model of the panel's main line (it contains its companions) + the S/N badge."""
        lf = next((fits[k] for k in p.main_keys + p.line_keys
                   if k in fits and fits[k].band == p.band and fits[k].wave is not None), None)
        y = None
        if lf is not None:
            # the model only where the band has data (not over NaN gaps / beyond the band edge)
            data = _sample(sp.wave, sp.flux, lf.wave)
            y = np.where(np.isfinite(data), lf.model, np.nan)
            if snr:
                e = sp.err if sp.err is not None else sp.noise
                if e is None:
                    y = None
                else:
                    ei = _sample(sp.wave, e, lf.wave)
                    ok = np.isfinite(ei) & (ei > 0)
                    if ok.any():          # edge pixels with a tiny error would make the model shoot up
                        ok &= ei > 0.5 * np.median(ei[ok])
                    with np.errstate(divide='ignore', invalid='ignore'):
                        y = np.where(ok, y / np.where(ok, ei, 1.0), np.nan)
        if y is None or not np.isfinite(y).any():
            p.set_model(None, None)
        else:
            # the y range keeps the model's line peak in view (not its continuum at a band edge)
            centres = [REST[k] * (1 + self._z) for k in p.line_keys]
            near = np.any([np.abs(lf.wave - c) < PEAK_TOL_UM for c in centres], axis=0) & np.isfinite(y)
            peak = float(np.max(y[near])) if near.any() else np.nan
            p.set_model(lf.wave / f, y, peak)
        mains = [fits[k] for k in p.main_keys if k in fits and fits[k].covered]
        if not mains:
            p.set_badge(None)
            return
        detected = any(m.detected for m in mains)
        col = C.TEXT if detected else C.TEXT_DIM
        vals = ' · '.join(_fmt(m.snr) for m in mains)
        core = f'<span style="color:{col}; font-size:10pt; font-weight:700;">S/N {vals}</span>'
        forced = all(m.forced for m in mains)
        f_html = f'<span style="color:{C.TEXT_DIM}; font-size:9pt;"> forced</span>' if forced else ''
        flags = sorted({fl for m in mains for fl in m.flags})
        fl_html = (f'<span style="color:{C.AMBER}; font-size:9pt;"> · {", ".join(fl.replace("_", " ") for fl in flags)}'
                   '</span>') if flags else ''
        variants = [core + f_html + fl_html, core + f_html, core]     # slots: all, no flags, S/N only
        tip = ['S/N (Gaussian-fit flux / error): ' + ', '.join(f'{LINE_LABELS[m.key]} {_fmt(m.snr)}' for m in mains),
               'detected (S/N ≥ 3)' if detected else 'not detected (S/N < 3)']
        if forced:
            tip.append('forced: velocity and width fixed to the anchor line')
        if flags:
            tip.append('flags: ' + ', '.join(flags))
        p.set_badge(variants, detected, '\n'.join(tip))

    # ================================================================== export
    def _sky_done_text(self):
        """'True' / 'False' / 'not in IZ' - what was really done (the option can not always be applied)."""
        if not self.state.options.get('subtract_bg'):
            return 'False'
        missed = self._sky_note(self._spec_cache.get(self._cache_key()) or {})
        return f'True except {"/".join(missed)} (too few spaxels outside the aperture)' if missed else 'True'

    def _export_meta(self):
        s = self.state
        z = s.z
        r = s.redshift
        variant = s.catalog.variant if s.catalog is not None else s.variant
        return [f'EMPOWER Explorer · {s.key} · reduction {variant}',
                f'z = {z:.5f} ({r.z_type or "?"}{", manual" if r.override else ""})' if z else 'z unknown',
                (f'annulus x = {s.ap_x:.2f}, y = {s.ap_y:.2f}, {s.ap_rin:.2f} < r <= {s.ap_r:.2f} px'
                 if s.ap_rin > 0 else f'aperture x = {s.ap_x:.2f}, y = {s.ap_y:.2f}, r = {s.ap_r:.2f} px') +
                f' on the {s.ref_band} grid; '
                f'method {s.options.get("method")}; residual sky subtracted: {self._sky_done_text()}',
                f'flux unit: {self.table.flux_unit} x micron; velocities in km/s']
