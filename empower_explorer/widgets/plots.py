"""Plot building blocks shared by all tabs (pyqtgraph, dark theme).

    plot = make_plot('Title', 'Wavelength [µm]', 'S/N')     # pg.PlotWidget, styled
    plot = make_plot(..., viewBox=SpectrumViewBox())          # wheel: x; Shift+wheel or y axis: y
    add_vline(plot, x, 'Hα', color, dashed=True)              # line marker with label
    add_spans(plot, [(lo, hi), ...])                           # grey shading (telluric bands)
    panel = ImagePanel(title='White light')                    # image + colour bar + aperture circle
    panel.set_image(img, cmap='magma')
    panel.set_aperture(x, y, r, r_in);  panel.apertureMoved.connect(...)   # drag or click on the image
    panel.pixelClicked.connect(...)  /  panel.pixelHovered.connect(...)
    empty = EmptyState('No maps for this galaxy')

pyqtgraph's right-click menu already offers 'Export…' (PNG, SVG, CSV) on every plot.
Images use row-major arrays img[y, x]; pixel centres are at integer data coordinates
(x = column, y = row), origin at the bottom left like matplotlib origin='lower'.
"""
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import QGraphicsEllipseItem, QLabel, QVBoxLayout, QWidget

from empower_explorer.theme import C


def _pen(color, width=1.5, style=Qt.SolidLine):
    p = pg.mkPen(color, width=width)
    p.setStyle(style)
    p.setCosmetic(True)
    return p


def style_plot_item(pi, title='', xlabel='', ylabel=''):
    """Apply the dark theme to a pg.PlotItem."""
    for name in ('left', 'bottom', 'right', 'top'):
        ax = pi.getAxis(name)
        ax.setPen(pg.mkPen(C.BG_LIGHT))
        ax.setTextPen(pg.mkPen(C.TEXT_DIM))
        ax.setStyle(tickFont=QFont('Segoe UI', 9))
        ax.enableAutoSIPrefix(False)     # never '(x0.001)': axes show µm, km/s, z as they are
    pi.showGrid(x=True, y=True, alpha=0.15)
    if title:
        pi.setTitle(title, color=C.TEXT, size='11pt', bold=True)
    if xlabel:
        pi.setLabel('bottom', xlabel, color=C.TEXT_DIM)
    if ylabel:
        pi.setLabel('left', ylabel, color=C.TEXT_DIM)
    pi.getViewBox().setBackgroundColor(C.BG_DARK)
    return pi


class SpectrumViewBox(pg.ViewBox):
    """ViewBox of a spectrum: the wheel zooms the wavelength axis; Shift + wheel, or the wheel over the y
    axis, zooms y (not Ctrl: a touchpad pinch arrives as Ctrl + wheel and stays on the wavelength); dragging
    the y axis pans y.  While y_manual is False the owner scales y to the data; a hand-made y range (also
    typed in the context menu) sets it - then a drag in the plot also pans y - until a double-click or the
    menu's y 'Auto' hands y back (sigYAuto: the owner re-scales)."""
    sigYAuto = pyqtSignal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.y_manual = False
        self.setMouseEnabled(x=True, y=True)
        try:                                  # the context menu's y controls (pyqtgraph ViewBoxMenu)
            ctrl = self.menu.ctrl[1]
            for w in (ctrl.minText, ctrl.maxText):
                w.editingFinished.connect(self._set_manual)
            ctrl.manualRadio.clicked.connect(self._set_manual)
            ctrl.autoRadio.clicked.connect(self._set_auto)
        except Exception:
            pass

    def _set_manual(self):
        self.y_manual = True

    def _set_auto(self):
        self.y_manual = False
        self.sigYAuto.emit()

    def _y_enabled(self):
        return bool(self.state['mouseEnabled'][1])

    def wheelEvent(self, ev, axis=None):
        if axis is None:
            axis = 1 if ev.modifiers() & Qt.ShiftModifier else 0
        if axis == 1 and self._y_enabled():
            self.y_manual = True
        super().wheelEvent(ev, axis=axis)

    def mouseDragEvent(self, ev, axis=None):
        if axis is None and not self.y_manual and self.state['mouseMode'] != pg.ViewBox.RectMode:
            axis = 0                          # y follows the data: a drag in the plot pans the wavelength
        if axis == 1 and self._y_enabled():
            self.y_manual = True
        super().mouseDragEvent(ev, axis=axis)

    def mouseClickEvent(self, ev):
        if ev.double() and ev.button() == Qt.LeftButton:
            ev.accept()
            self.y_manual = False
            self.sigYAuto.emit()
            return
        super().mouseClickEvent(ev)


def make_plot(title='', xlabel='', ylabel='', parent=None, viewBox=None):
    w = pg.PlotWidget(parent=parent, background=C.BG_MEDIUM, viewBox=viewBox)
    style_plot_item(w.getPlotItem(), title, xlabel, ylabel)
    w.setMinimumSize(120, 90)
    return w


def add_vline(plot, x, label=None, color=C.TEXT_DIM, dashed=True, width=1.3, label_pos=0.95,
              anchor_left=False, bold=False):
    """Vertical marker line (not movable) with an optional label near the top."""
    item = pg.InfiniteLine(pos=x, angle=90, movable=False,
                           pen=_pen(color, width, Qt.DashLine if dashed else Qt.DotLine))
    if label:
        item.label = pg.InfLineLabel(item, text=label, position=label_pos,
                                     anchors=[(1, 0), (1, 0)] if anchor_left else [(0, 0), (0, 0)],
                                     color=color)
        f = QFont('Segoe UI', 9)
        f.setBold(bold)
        item.label.setFont(f)
    plot.addItem(item, ignoreBounds=True)
    return item


class SpanItem(pg.GraphicsObject):
    """Shaded x ranges over the full view height as ONE item (dozens of LinearRegionItems made
    every rebuild ~60 ms slower).  Follows the y range of its view box."""

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

    def _follow(self, vb, rng):
        if self.scene() is None:          # removed from the plot (plot.clear()): stop following
            try:
                vb.sigYRangeChanged.disconnect(self._follow)
            except (TypeError, RuntimeError):
                pass
            return
        self.prepareGeometryChange()
        self._y = (float(rng[0]), float(rng[1]))
        self.update()

    def boundingRect(self):
        if not self._spans:
            return QRectF()
        lo, hi = min(a for a, _ in self._spans), max(b for _, b in self._spans)
        y0, y1 = self._y
        return QRectF(lo, y0, hi - lo, y1 - y0)

    def paint(self, p, *args):
        p.setPen(pg.mkPen(None))
        p.setBrush(self._brush)
        y0, y1 = self._y
        for a, b in self._spans:
            p.drawRect(QRectF(a, y0, b - a, y1 - y0))


def add_spans(plot, spans, rgba=C.TELLURIC):
    """Shaded x ranges (e.g. low atmospheric transmission) as one item. Returns [item]."""
    spans = list(spans)
    if not spans:
        return []
    return [SpanItem(spans, rgba).attach(plot)]


def colormap(name):
    try:
        return pg.colormap.get(name)
    except Exception:
        try:
            return pg.colormap.get(name, source='matplotlib')
        except Exception:
            return pg.colormap.get('viridis')


def robust_levels(img, lo=1.0, hi=99.5, symmetric=False):
    v = np.asarray(img, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    if symmetric:
        m = float(np.percentile(np.abs(v), hi))
        return (-m, m) if m > 0 else (-1.0, 1.0)
    a, b = np.percentile(v, [lo, hi])
    if not b > a:
        b = a + (abs(a) if a else 1.0)
    return float(a), float(b)


class ImagePanel(pg.GraphicsLayoutWidget):
    """A 2-D map with colour bar, optional draggable aperture circle (with a dashed inner circle for an
    annulus) and crosshair marker.

    Signals: apertureMoved(x, y) (drag or click while the aperture is shown),
             apertureResized(r), pixelClicked(x, y) (integer pixel), pixelHovered(x, y, value)."""
    apertureMoved = pyqtSignal(float, float)
    apertureResized = pyqtSignal(float)
    pixelClicked = pyqtSignal(int, int)
    pixelHovered = pyqtSignal(int, int, float)

    def __init__(self, title='', parent=None, colorbar=True):
        super().__init__(parent=parent)
        self.setBackground(C.BG_MEDIUM)
        self.ci.setContentsMargins(4, 4, 4, 4)
        self.plot = self.addPlot()
        style_plot_item(self.plot, title)
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=False, y=False)
        self.plot.hideButtons()
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        self.bar = None
        if colorbar:
            self.bar = pg.ColorBarItem(values=(0, 1), colorMap=colormap('magma'), width=12, interactive=False)
            self.bar.setImageItem(self.img, insert_in=self.plot)
            self.bar.getAxis('right').setTextPen(pg.mkPen(C.TEXT_DIM))
            self.bar.getAxis('right').setStyle(tickFont=QFont('Segoe UI', 8))
            self.bar.getAxis('right').enableAutoSIPrefix(False)
        self._data = None
        self._message = pg.TextItem('', color=C.TEXT_DIM, anchor=(0.5, 0.5))
        self._message.setFont(QFont('Segoe UI', 10, italic=True))
        self.plot.addItem(self._message, ignoreBounds=True)
        self.roi = pg.CircleROI((0, 0), (5, 5), pen=_pen(C.ACCENT, 2), movable=True, rotatable=False,
                                resizable=True, removable=False)
        self.roi.setZValue(20)
        self.roi.hide()
        self.plot.addItem(self.roi)
        self.roi.sigRegionChanged.connect(self._roi_changed)
        self._roi_guard = False
        self.roi_inner = QGraphicsEllipseItem()          # inner edge of an annulus (follows the circle)
        self.roi_inner.setPen(_pen(C.ACCENT, 1.5, Qt.DashLine))
        self.roi_inner.setZValue(19)
        self.roi_inner.hide()
        self.plot.addItem(self.roi_inner)
        self._r_in_frac = 0.0
        self.marker = pg.ScatterPlotItem(size=14, symbol='+', pen=_pen('#FFFFFF', 2), brush=None)
        self.marker.setZValue(25)
        self.plot.addItem(self.marker)
        self.overlays = []
        self.scene().sigMouseClicked.connect(self._clicked)
        self.scene().sigMouseMoved.connect(self._moved)

    # -- content
    def set_image(self, img, cmap='magma', levels=None, symmetric=False, title=None, autorange=True):
        if title is not None:
            self.plot.setTitle(title, color=C.TEXT, size='11pt', bold=True)
        if img is None or not np.isfinite(np.asarray(img, float)).any():
            self.clear_image('no data')
            return
        img = np.asarray(img, float)
        self._data = img
        self._message.setText('')
        lv = levels if levels is not None else robust_levels(img, symmetric=symmetric)
        cm = colormap(cmap)
        self.img.setImage(np.where(np.isfinite(img), img, np.nan), levels=lv)
        self.img.setColorMap(cm)
        ny, nx = img.shape
        self.img.setRect(QRectF(-0.5, -0.5, nx, ny))
        if self.bar is not None:
            self.bar.setColorMap(cm)
            self.bar.setLevels(lv)
        if autorange:
            self.plot.getViewBox().setRange(QRectF(-0.5, -0.5, nx, ny), padding=0.02)

    def clear_image(self, message=''):
        self._data = None
        self.img.clear()
        self._message.setText(message)
        self._message.setPos(0, 0)
        self.plot.getViewBox().setRange(QRectF(-5, -5, 10, 10), padding=0)

    @property
    def data(self):
        return self._data

    # -- aperture
    def set_aperture(self, x, y, r, r_in=0.0, visible=True):
        self._roi_guard = True
        self.roi.setPos(QPointF(x - r, y - r))
        self.roi.setSize((2 * r, 2 * r))
        self.roi.setVisible(visible)
        self._r_in_frac = r_in / r if r > 0 and r_in > 0 else 0.0
        self._place_inner(x, y, r)
        self._roi_guard = False

    def _place_inner(self, x, y, r):
        ri = self._r_in_frac * r
        self.roi_inner.setRect(QRectF(x - ri, y - ri, 2 * ri, 2 * ri))
        self.roi_inner.setVisible(ri > 0 and self.roi.isVisible())

    def _roi_changed(self):
        pos, size = self.roi.pos(), self.roi.size()
        r = size[0] / 2.0
        self._place_inner(pos.x() + r, pos.y() + r, r)     # the inner circle moves with a drag at once
        if self._roi_guard:
            return
        self.apertureMoved.emit(pos.x() + r, pos.y() + r)
        self.apertureResized.emit(r)

    def set_marker(self, x=None, y=None):
        if x is None:
            self.marker.setData([], [])
        else:
            self.marker.setData([x], [y])

    def add_overlay(self, item):
        self.plot.addItem(item)
        self.overlays.append(item)
        return item

    def clear_overlays(self):
        for it in self.overlays:
            self.plot.removeItem(it)
        self.overlays = []

    # -- mouse
    def _data_pos(self, scene_pos):
        vb = self.plot.getViewBox()
        if not self.plot.sceneBoundingRect().contains(scene_pos):
            return None
        p = vb.mapSceneToView(scene_pos)
        return p.x(), p.y()

    def _clicked(self, ev):
        if ev.button() != Qt.LeftButton or self._data is None:
            return
        pos = self._data_pos(ev.scenePos())
        if pos is None:
            return
        x, y = pos
        ny, nx = self._data.shape
        if not (-0.5 <= x < nx - 0.5 and -0.5 <= y < ny - 0.5):
            return
        if self.roi.isVisible() and not self.roi.contains(self.roi.mapFromScene(ev.scenePos())):
            self.apertureMoved.emit(x, y)
        self.pixelClicked.emit(int(round(x)), int(round(y)))

    def _moved(self, scene_pos):
        if self._data is None:
            return
        pos = self._data_pos(scene_pos)
        if pos is None:
            return
        ix, iy = int(round(pos[0])), int(round(pos[1]))
        ny, nx = self._data.shape
        if 0 <= ix < nx and 0 <= iy < ny:
            self.pixelHovered.emit(ix, iy, float(self._data[iy, ix]))


class EmptyState(QWidget):
    """Centred, dim message for 'nothing to show here' (with an optional hint line)."""

    def __init__(self, text='', hint='', parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.addStretch()
        self.label = QLabel(text)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(f'color: {C.TEXT_DIM}; font-size: 15px;')
        self.hint = QLabel(hint)
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet(f'color: {C.TEXT_DIM}; font-size: 12px;')
        lay.addWidget(self.label)
        lay.addWidget(self.hint)
        lay.addStretch()

    def set_text(self, text, hint=''):
        self.label.setText(text)
        self.hint.setText(hint)


def boxcar(y, n):
    """Boxcar smoothing that keeps NaN gaps (n <= 1: unchanged)."""
    n = int(n or 0)
    if n <= 1:
        return y
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    k = np.ones(n)
    num = np.convolve(np.where(ok, y, 0.0), k, mode='same')
    den = np.convolve(ok.astype(float), k, mode='same')
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / den
    out[~ok] = np.nan
    return out
