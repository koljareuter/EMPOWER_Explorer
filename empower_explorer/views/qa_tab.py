"""Data QA tab - data quality and provenance of the current galaxy.

Five pages, chosen with the segmented control at the top:

  Cubes       every combined cube of the galaxy in the loaded reduction (header scan infos):
              band, NCOMBINE, on-source time, telluric, DATE-OBS, seeing, IQ, airmass, file and run
              folder.  Double-click a row to open its folder in Windows Explorer.
  Reductions  which reductions contain the galaxy, with bands and on-source time (the other
              reductions are header-scanned once in a worker; data.scan caches them on disk), and
              'Compare spectra': one band from two reductions, extracted at the sky position of the
              current aperture, overlaid, with the S/N of every covered line in both.
  Redshift    catalogue, public and KMOS-line redshifts and the velocity offset of the current fit;
              pink warnings when |Δv| > 200 km/s or the catalogues disagree.
  Census      the galaxy's rows of halpha_census.csv, kmos_sample_overview.csv and
              kmos_snr_distribution.csv.
  Header      any FITS header of the galaxy's cubes, with a filter box.

Pages are rebuilt lazily: a state change only marks pages stale and the visible page is rebuilt
(coalesced by a short timer), so the tab costs nothing while hidden.  Everything that reads FITS
files or fits lines runs in a worker; a result that arrives for an older request is dropped.
"""
import copy
import functools
import os
import sys
import threading
import traceback
from collections import OrderedDict

import numpy as np
import pyqtgraph as pg
from astropy.io import fits
from PyQt5.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import (QBrush, QColor, QDesktopServices, QFont, QGuiApplication, QSyntaxHighlighter,
                         QTextCharFormat)
from PyQt5.QtWidgets import (QAbstractItemView, QAbstractScrollArea, QButtonGroup, QComboBox, QFrame, QGridLayout, QHBoxLayout,
                             QHeaderView, QLabel, QLineEdit, QMenu, QPlainTextEdit, QProgressBar, QPushButton,
                             QScrollArea, QSizePolicy, QSplitter, QStackedWidget, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from empower_explorer import analysis
from empower_explorer.data.catalog import band_rank
from empower_explorer.data.scan import scan_variant
from empower_explorer.data.stack import load_band_stack
from empower_explorer.lines import C_KMS, FIT_LINES, FULL_MARKERS, LINE_COLORS, LINE_LABELS, PANELS, REST, \
    panel_window
from empower_explorer.theme import C, band_color
from empower_explorer.widgets.plots import EmptyState, add_spans, add_vline, boxcar, make_plot
from empower_explorer.workers import run_async

DV_WARN_KMS = 200.0          # |Δv| above this is flagged
SEP_WARN_ARCSEC = 1.0        # catalogue source farther than this from the KMOS target is flagged
DASH = '—'
NUM = Qt.AlignRight | Qt.AlignVCenter
LEFT = Qt.AlignLeft | Qt.AlignVCenter
CENTER = Qt.AlignCenter
COLOR_A, COLOR_B = C.ACCENT, C.AMBER      # reductions A and B in 'Compare spectra'
PEAK_TOL_UM = 0.0025         # data this close to a line stay inside the y range of 'Compare spectra'


# ====================================================================== small helpers
def _isnum(v):
    if v is None or isinstance(v, (str, bool, np.bool_)):
        return False
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def _f(v, fmt='{:.2f}', unit=''):
    """Number -> text, '—' for None / NaN / text."""
    if not _isnum(v):
        return DASH
    s = fmt.format(float(v))
    return f'{s} {unit}' if unit else s


def _s(v):
    """Any table value -> text, '—' when empty."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return DASH
    s = str(v).strip()
    return s if s and s.lower() not in ('nan', 'none') else DASH


def _bool(v):
    """True / False / None from bools, numpy bools and 'True'/'False' strings."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in ('true', 'false'):
        return v.strip().lower() == 'true'
    if _isnum(v):
        return bool(float(v))
    return None


def _positive(v):
    """float(v) if it is a positive number (ESO headers use -1 for 'not measured'), else None."""
    return float(v) if _isnum(v) and float(v) > 0 else None


def _hours(seconds):
    return _f(seconds / 3600.0 if _isnum(seconds) else None, '{:.2f}', 'h')


def _dv(z_ref, z):
    """Velocity of z relative to z_ref [km/s]."""
    if not (_isnum(z_ref) and _isnum(z)):
        return np.nan
    return C_KMS * (float(z) - float(z_ref)) / (1.0 + float(z_ref))


def _signed(v, unit='km/s'):
    return _f(v, '{:+.0f}', unit)


def _rgba(color, alpha):
    c = QColor(color)
    return f'rgba({c.red()}, {c.green()}, {c.blue()}, {alpha})'


def _clear_layout(lay):
    while lay.count():
        it = lay.takeAt(0)
        w = it.widget()
        if w is not None:
            w.hide()               # deleteLater alone leaves the old label painted until the deferred delete
            w.deleteLater()
        elif it.layout() is not None:
            _clear_layout(it.layout())


def _label(text='', role=None, tip=None, selectable=False, wrap=False):
    lab = QLabel(text)
    if role:
        lab.setProperty('role', role)
    if tip:
        lab.setToolTip(tip)
    if selectable:
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
    lab.setWordWrap(wrap)
    return lab


def _flat_button(text, tip=''):
    b = QPushButton(text)
    b.setProperty('flat', 'true')
    b.setToolTip(tip)
    b.setCursor(Qt.PointingHandCursor)
    return b


def _make_table(headers, tips=None, stretch=None):
    """Read-only, row-selecting table in the app style (columns sized to their contents)."""
    t = QTableWidget(0, len(headers))
    set_headers(t, headers, tips, stretch)
    t.verticalHeader().hide()
    t.verticalHeader().setDefaultSectionSize(27)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setSelectionMode(QAbstractItemView.SingleSelection)
    t.setAlternatingRowColors(True)
    t.setShowGrid(False)
    t.setWordWrap(False)
    t.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
    t.setStyleSheet('QTableView::item { padding-left: 5px; padding-right: 5px; }')   # air between columns
    hh = t.horizontalHeader()
    hh.setHighlightSections(False)
    hh.setDefaultAlignment(LEFT)
    hh.setMinimumSectionSize(40)
    return t


def set_headers(t, headers, tips=None, stretch=None):
    t.setColumnCount(len(headers))
    t.setHorizontalHeaderLabels(headers)
    for i, tip in enumerate(tips or []):
        if tip and t.horizontalHeaderItem(i) is not None:
            t.horizontalHeaderItem(i).setToolTip(tip)
    hh = t.horizontalHeader()
    for i in range(len(headers)):
        hh.setSectionResizeMode(i, QHeaderView.ResizeToContents)
    if stretch is not None:
        hh.setSectionResizeMode(stretch, QHeaderView.Stretch)


def _align_headers(t, aligns):
    """aligns: {column: Qt alignment} - header text over right-aligned numbers / centred checks."""
    for c, a in aligns.items():
        it = t.horizontalHeaderItem(c)
        if it is not None:
            it.setTextAlignment(int(a))


def _item(text, tip=None, color=None, bold=False, align=LEFT, data=None):
    it = QTableWidgetItem(text)
    it.setTextAlignment(int(align))
    if tip:
        it.setToolTip(tip)
    if color:
        it.setForeground(QBrush(QColor(color)))
    if bold:
        f = it.font()
        f.setBold(True)
        it.setFont(f)
    if data is not None:
        it.setData(Qt.UserRole, data)
    return it


def _check_item(v, tip=None):
    b = _bool(v)
    if b is None:
        return _item(DASH, tip, C.TEXT_DIM, align=CENTER)
    return _item('✓' if b else '✗', tip, C.EMERALD if b else C.TEXT_DIM, align=CENTER)


def _fit_height(t, max_rows=None):
    """Fix a table's height to show all (or max_rows) rows without a scroll bar."""
    n = t.rowCount() if max_rows is None else min(t.rowCount(), max_rows)
    h = t.horizontalHeader().sizeHint().height() + t.verticalHeader().defaultSectionSize() * max(n, 1) + 6
    if t.horizontalScrollBar().isVisible() or t.horizontalHeader().length() > t.viewport().width() > 50:
        h += 10
    t.setFixedHeight(h)


def _snr_color(v, thr=3.0):
    return C.EMERALD if _isnum(v) and float(v) >= thr else C.TEXT_DIM


def _guard(method):
    """Slot wrapper: an unexpected exception is shown in the page instead of killing the app."""
    @functools.wraps(method)
    def wrapper(self, *args):
        try:
            return method(self, *args)
        except Exception:
            tb = traceback.format_exc()
            print(tb, file=sys.stderr, flush=True)
            page = self if isinstance(self, _Page) else getattr(self, 'page', None)
            if page is not None:
                page.show_error(tb)
    return wrapper


def _galaxy_infos(state):
    """Scan infos of all cubes of the current galaxy in the loaded reduction, band order."""
    if state.catalog is None or state.key is None:
        return []
    bands = state.catalog.cubes.get(state.key, {})
    return [i for b in sorted(bands, key=band_rank) for i in sorted(bands[b], key=lambda i: i['path'])]


def _table_rows(state, name):
    """Rows of the CSV table `name` for the current galaxy; None if the table does not exist."""
    df = state.catalog.tables.get(name) if state.catalog is not None else None
    if df is None or 'key' not in df.columns:
        return None
    return df[df['key'] == state.key]


# ====================================================================== shared widgets
class _Card(QFrame):
    """Rounded panel with an upper-case section title, a dim subtitle and a body layout."""

    def __init__(self, title='', subtitle='', parent=None):
        super().__init__(parent)
        self.setProperty('role', 'card')
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(8)
        self.head = QHBoxLayout()
        self.head.setSpacing(10)
        self.title = _label(title.upper(), 'section')
        self.subtitle = _label(subtitle, 'dim')
        self.head.addWidget(self.title)
        self.head.addWidget(self.subtitle)
        self.head.addStretch()
        lay.addLayout(self.head)
        self.body = lay


class _Chips(QWidget):
    """A row of small rounded labels."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lay = QHBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lay.setSpacing(6)

    def set_chips(self, chips):
        """chips: [(text, tooltip, colour or None)]."""
        _clear_layout(self.lay)
        for text, tip, color in chips:
            lab = _label(text, 'chip', tip)
            if color:
                lab.setStyleSheet(f'color: {color};')
            self.lay.addWidget(lab)
        self.lay.addStretch()


class _Banner(QFrame):
    """Tinted message box (pink warnings, amber notes); hidden when empty."""

    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.setObjectName('qaBanner')
        self.setStyleSheet(f'QFrame#qaBanner {{ background-color: {_rgba(color, 34)}; '
                           f'border: 1px solid {color}; border-radius: 8px; }}')
        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.RichText)
        self.label.setStyleSheet(f'color: {C.TEXT}; font-size: 13px;')
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 7, 12, 7)
        lay.addWidget(self.label)
        self.hide()

    def set_messages(self, msgs):
        self.label.setText('<br>'.join(f'⚠&nbsp; {m}' for m in msgs))
        self.setVisible(bool(msgs))


class _KVCard(_Card):
    """Card with one large value, a dim note and label / value rows."""

    def __init__(self, title, subtitle='', parent=None):
        super().__init__(title, subtitle, parent)
        self.big = _label('', selectable=True)
        self.big.setStyleSheet(f'font-size: 26px; font-weight: 600; color: {C.TEXT};')
        self.note = _label('', 'dim', wrap=True)
        top = QHBoxLayout()
        top.setSpacing(12)
        top.addWidget(self.big)
        top.addWidget(self.note, 1)
        self.body.addLayout(top)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(18)
        self.grid.setVerticalSpacing(5)
        self.grid.setColumnStretch(1, 1)
        self.body.addLayout(self.grid)
        self.body.addStretch()

    def set_content(self, big, note='', rows=(), big_color=None):
        """rows: [(label, value text, tooltip, colour or None)]."""
        self.big.setText(big)
        self.big.setStyleSheet(f'font-size: 26px; font-weight: 600; color: {big_color or C.TEXT};')
        self.note.setText(note)
        _clear_layout(self.grid)
        for r, (lab, val, tip, color) in enumerate(rows):
            k = _label(lab, 'dim', tip)
            v = _label(val, selectable=True, tip=tip)
            if color:
                v.setStyleSheet(f'color: {color};')
            self.grid.addWidget(k, r, 0, Qt.AlignLeft | Qt.AlignTop)
            self.grid.addWidget(v, r, 1, Qt.AlignLeft | Qt.AlignTop)


class _Page(QWidget):
    """Base of the five pages: a body and an empty / error state; refresh() rebuilds the body.
    TRIGGERS: which state changes make the page stale (see QATab._invalidate)."""
    TRIGGERS = frozenset()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.stale = True
        self._stack = QStackedWidget()
        self.body = QWidget()
        self.empty = EmptyState()
        self._stack.addWidget(self.body)
        self._stack.addWidget(self.empty)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._stack)

    def show_empty(self, text, hint=''):
        self.empty.set_text(text, hint)
        self._stack.setCurrentWidget(self.empty)

    def show_body(self):
        self._stack.setCurrentWidget(self.body)

    def show_error(self, tb):
        last = tb.strip().splitlines()[-1] if tb.strip() else 'unknown error'
        self.show_empty('Something went wrong on this page', last)

    def has_galaxy(self):
        """Show the right empty state when no reduction / galaxy is loaded; True if there is one."""
        st = self.state
        if st.catalog is None:
            self.show_empty('No reduction loaded yet', 'Choose a reduction in the toolbar.')
            return False
        if st.key is None or st.key not in st.catalog.galaxies.index:
            self.show_empty('No galaxy selected', 'Pick a galaxy in the list on the left.')
            return False
        return True

    def refresh(self):
        raise NotImplementedError


# ====================================================================== 1. Cubes
class CubesPage(_Page):
    """All combined cubes of the galaxy in the loaded reduction."""
    TRIGGERS = frozenset({'catalog', 'galaxy'})
    showHeader = pyqtSignal(str)          # path of a cube whose header should be shown

    COLUMNS = [
        ('Band', 'KMOS grating'),
        ('NCOMBINE', 'Number of exposures combined into the cube (NCOMBINE)'),
        ('On-source', 'On-source time NCOMBINE × DIT × NDIT (kmos_sample.onsource_seconds)'),
        ('Telluric', 'A TELLEFF_ telluric file exists next to the cube'),
        ('DATE-OBS', 'Start of the first combined exposure (UT)'),
        ('Seeing', 'DIMM seeing at the start → end of the first exposure [arcsec] (TEL AMBI FWHM)'),
        ('IQ', 'Image quality at the telescope, first exposure [arcsec] (TEL IA FWHM)'),
        ('Airmass', 'Airmass at the start of the first exposure (TEL AIRM START)'),
        ('Size', 'Cube dimensions nλ × ny × nx [px]'),
        ('File', 'DATACUBE_COMBINED file'),
        ('Run folder', 'Folder relative to the reduction — double-click a row to open it'),
    ]

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        card = _Card('Cubes in this reduction')
        self.chips = _Chips()
        card.head.insertWidget(2, self.chips)
        self.open_btn = _flat_button('Open folder', 'Open the folder of the selected cube in Explorer')
        self.copy_btn = _flat_button('Copy path', 'Copy the full path of the selected cube')
        self.header_btn = _flat_button('Show header', 'Show the FITS header of the selected cube')
        for b in (self.open_btn, self.copy_btn, self.header_btn):
            card.head.addWidget(b)
        self.open_btn.clicked.connect(lambda: self._open_folder(self._selected_path()))
        self.copy_btn.clicked.connect(lambda: self._copy_path(self._selected_path()))
        self.header_btn.clicked.connect(lambda: self._show_header(self._selected_path()))
        self.table = _make_table([c for c, _ in self.COLUMNS], [t for _, t in self.COLUMNS], stretch=10)
        _align_headers(self.table, {1: NUM, 2: NUM, 3: CENTER, 5: NUM, 6: NUM, 7: NUM, 8: NUM})
        self.table.cellDoubleClicked.connect(lambda r, c: self._open_folder(self._path_of_row(r)))
        self.table.itemSelectionChanged.connect(self._update_buttons)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        card.body.addWidget(self.table, 1)
        self.note = _label('Double-click a row to open its folder · right-click for more.  Seeing, IQ and '
                           'airmass come from the primary header of the combined cube (first exposure).', 'dim',
                           wrap=True)
        card.body.addWidget(self.note)
        lay.addWidget(card, 1)
        self._update_buttons()

    def refresh(self):
        if not self.has_galaxy():
            return
        infos = _galaxy_infos(self.state)
        if not infos:
            self.show_empty('No cubes for this galaxy', f'{self.state.key} has no DATACUBE_COMBINED files '
                                                        f'in {self.state.variant}.')
            return
        self.show_body()
        self._fill(infos)
        self._set_chips(infos)

    def _fill(self, infos):
        t = self.table
        t.setRowCount(len(infos))
        for r, i in enumerate(infos):
            path = i.get('path', '')
            frames = sorted(i.get('frames') or [])
            ftip = (f'{len(frames)} input frames:\n' + '\n'.join(frames[:12])
                    + ('\n…' if len(frames) > 12 else '')) if frames else None
            shape = i.get('shape')
            hdr_t = i.get('exptime_hdr')
            items = [
                _item(i.get('band', '?'), color=band_color(i.get('band')), bold=True, data=path),
                _item(_f(i.get('n_frames'), '{:.0f}'), ftip, align=NUM),
                _item(_hours(i.get('exptime_s')), f'EXPTIME keyword: {_hours(hdr_t)}' if _isnum(hdr_t) else None,
                      align=NUM),
                _check_item(i.get('telluric'), 'TELLEFF_ file found' if i.get('telluric') else 'no telluric file'),
                _item(self._date(i.get('date_obs')), _s(i.get('date_obs'))),
                _item(self._seeing(i), align=NUM),
                _item(_f(_positive(i.get('iq')), '{:.2f}″'), align=NUM),
                _item(_f(_positive(i.get('airmass_start')), '{:.2f}'), align=NUM),
                _item(' × '.join(str(s) for s in shape) if shape else DASH, align=NUM),
                _item(os.path.basename(path), path),
                _item(_s(i.get('folder')), os.path.dirname(path)),
            ]
            for c, it in enumerate(items):
                t.setItem(r, c, it)
        if infos:
            t.selectRow(0)

    def _set_chips(self, infos):
        bands = sorted({i['band'] for i in infos}, key=band_rank)
        total = np.nansum([i['exptime_s'] for i in infos if _isnum(i.get('exptime_s'))])
        n_tell = sum(bool(i.get('telluric')) for i in infos)
        seeing = [v for i in infos for v in (_positive(i.get('seeing_start')), _positive(i.get('seeing_end')))
                  if v is not None]
        chips = [(f'{len(infos)} cube{"s" if len(infos) != 1 else ""}', None, None),
                 (' + '.join(bands), 'bands', None),
                 (f'{total / 3600:.2f} h on source', 'summed over all cubes', None)]
        if n_tell == len(infos):
            chips.append(('telluric ✓', 'every cube has a TELLEFF_ file', C.EMERALD))
        else:
            chips.append((f'telluric {n_tell}/{len(infos)}', 'cubes with a TELLEFF_ file', C.AMBER))
        if seeing:
            chips.append((f'seeing {np.median(seeing):.2f}″', 'median DIMM seeing of the cube headers', None))
        self.chips.set_chips(chips)

    @staticmethod
    def _date(v):
        s = _s(v)
        return s if s == DASH else s.replace('T', ' ')[:16]

    @staticmethod
    def _seeing(info):
        a, b = _positive(info.get('seeing_start')), _positive(info.get('seeing_end'))
        if a is None and b is None:
            return DASH
        if a is not None and b is not None:
            return f'{a:.2f} → {b:.2f}″'
        return f'{a if a is not None else b:.2f}″'

    # -- actions
    def _path_of_row(self, row):
        it = self.table.item(row, 0)
        return it.data(Qt.UserRole) if it is not None else None

    def _selected_path(self):
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return self._path_of_row(rows[0].row()) if rows else None

    def _update_buttons(self):
        ok = self._selected_path() is not None
        for b in (self.open_btn, self.copy_btn, self.header_btn):
            b.setEnabled(ok)

    def _open_folder(self, path):
        if not path:
            return
        folder = os.path.dirname(path)
        if not os.path.isdir(folder):
            self.state.status.emit(f'Folder not found: {folder}', 6000)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
        self.state.status.emit(f'Opened {folder}', 4000)

    def _copy_path(self, path):
        if path:
            QGuiApplication.clipboard().setText(path)
            self.state.status.emit('Path copied to the clipboard', 3000)

    def _show_header(self, path):
        if path:
            self.showHeader.emit(path)

    def _context_menu(self, pos):
        row = self.table.rowAt(pos.y())
        path = self._path_of_row(row) if row >= 0 else None
        if not path:
            return
        m = QMenu(self)
        m.addAction('Open folder in Explorer', lambda: self._open_folder(path))
        m.addAction('Copy path', lambda: self._copy_path(path))
        m.addAction('Show FITS header', lambda: self._show_header(path))
        m.exec_(self.table.viewport().mapToGlobal(pos))


# ====================================================================== 2. Reductions
_SCANS = {}                 # reduction -> {galaxy key: {band: [scan infos]}}, whole session
_SCAN_ERRORS = {}           # reduction -> error text
_SCAN_ROOT = [None]         # EMPOWER folder the scans belong to (cleared when it changes)
_COMPARE_STACKS = OrderedDict()   # (reduction, key, band, paths) -> BandStack
_COMPARE_LOCK = threading.Lock()
MAX_COMPARE_STACKS = 4


def _group_by_galaxy(infos):
    out = {}
    for i in infos:
        out.setdefault(i['key'], {}).setdefault(i['band'], []).append(i)
    return out


def _compare_stack(variant, key, band, infos):
    """BandStack of one reduction for 'Compare spectra' (small LRU cache shared by the workers)."""
    ck = (variant, key, band, tuple(sorted(i['path'] for i in infos)))
    with _COMPARE_LOCK:
        st = _COMPARE_STACKS.get(ck)
        if st is not None:
            _COMPARE_STACKS.move_to_end(ck)
            return st
    st = load_band_stack(infos)
    with _COMPARE_LOCK:
        _COMPARE_STACKS[ck] = st
        while len(_COMPARE_STACKS) > MAX_COMPARE_STACKS:
            _COMPARE_STACKS.popitem(last=False)
    return st


def _sky_to_pixels(stack, sky, r_default):
    """(x, y, r) on the grid of `stack` for sky = (ra, dec, r_arcsec); the target centre without sky."""
    if sky is not None and stack.wcs is not None:
        try:
            px, py = stack.wcs.world_to_pixel_values(sky[0], sky[1])
            return float(px), float(py), float(sky[2] / stack.pixscale)
        except Exception:
            pass
    x, y = analysis.default_center(stack)
    return x, y, float(r_default)


def _compare_job(sources, sky, r_default, z, method, subtract_bg):
    """Worker: stack + extract + fit one band of two reductions.
    sources: [(reduction, key, band, BandStack or [scan infos])] -> [result dict]."""
    out = []
    for variant, key, band, src in sources:
        st = _compare_stack(variant, key, band, src) if isinstance(src, list) else src
        x, y, r = _sky_to_pixels(st, sky, r_default)
        spec = analysis.extract_spectrum(st, x, y, r, method=method, subtract_bg=subtract_bg)
        fits_ = {}
        if z is not None and spec.n_pix > 0:
            fits_, _ = analysis.fit_lines(spec, z)
        out.append(dict(variant=variant, spec=spec, fits=fits_, x=x, y=y, r=r, n_pix=spec.n_pix,
                        spans=analysis.telluric_spans(st), describe=st.describe(), bunit=st.bunit,
                        shape=st.cube.shape, pixscale=st.pixscale, telluric=st.atm is not None))
    return out


class ReductionsPage(_Page):
    """Reductions containing the galaxy + 'Compare spectra' of one band from two reductions."""
    TRIGGERS = frozenset({'catalog', 'galaxy', 'stacks', 'aperture', 'redshift', 'extract'})

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self._scan_job = None
        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Vertical)
        split.setHandleWidth(8)
        split.setChildrenCollapsible(False)

        card = _Card('Reductions containing this galaxy', 'on-source time per band')
        self.scan_label = _label('', 'dim')
        self.scan_bar = QProgressBar()
        self.scan_bar.setFixedWidth(140)
        self.scan_bar.setMaximumHeight(8)
        self.scan_bar.hide()
        self.rescan_btn = _flat_button('Rescan', 'Read the cube headers of all reductions again '
                                                 '(only new or changed files are opened)')
        self.rescan_btn.clicked.connect(self._rescan)
        for w in (self.scan_label, self.scan_bar, self.rescan_btn):
            card.head.addWidget(w)
        self.table = _make_table(['', 'Reduction'])
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setToolTip('Double-click a reduction to compare it with the loaded one')
        self.table.cellDoubleClicked.connect(self._row_double_clicked)
        card.body.addWidget(self.table, 1)
        split.addWidget(card)

        self.compare = _ComparePanel(self)
        split.addWidget(self.compare)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([215, 420])
        lay.addWidget(split)
        self._variants = []

    # -- header scans of the other reductions (chained jobs, one reduction at a time)
    @_guard
    def ensure_scans(self):
        st = self.state
        if st.catalog is None or st.root is None:
            return
        if _SCAN_ROOT[0] != st.root:          # another EMPOWER folder: the old scans do not apply
            _SCANS.clear()
            _SCAN_ERRORS.clear()
            _SCAN_ROOT[0] = st.root
            self._variants = []
        _SCANS[st.catalog.variant] = st.catalog.cubes
        if self._scan_job is not None:
            return
        named = self.variants()
        variants = [v for v, _ in named]
        todo = [v for v in variants if v not in _SCANS]
        if not todo:
            self.scan_bar.hide()
            n_err = sum(v in _SCAN_ERRORS for v in variants)
            self.scan_label.setText(f'{len(variants)} reductions scanned'
                                    + (f' · {n_err} could not be read' if n_err else ''))
            return
        v = todo[0]
        self.scan_label.setText(f'Scanning {dict(named).get(v, v)}  ({len(variants) - len(todo) + 1} of '
                                f'{len(variants)}) …')
        self.scan_label.setToolTip(f'Reading the cube headers of {v} in the background '
                                   f'(cached on disk, so this happens only once)')
        self.scan_bar.setRange(0, 0)
        self.scan_bar.show()
        root = st.root
        self._scan_job = run_async(scan_variant, root, v,
                                   on_done=lambda res, v=v, root=root: self._scan_done(v, root, res, ''),
                                   on_error=lambda msg, v=v, root=root: self._scan_done(v, root, None, msg),
                                   on_progress=self._scan_progress)

    def _scan_progress(self, done, total):
        self.scan_bar.setRange(0, max(total, 1))
        self.scan_bar.setValue(done)

    @_guard
    def _scan_done(self, variant, root, res, err):
        self._scan_job = None
        try:
            if root == self.state.root and root == _SCAN_ROOT[0]:
                if res is not None:
                    _SCANS[variant] = _group_by_galaxy(res[0])
                    _SCAN_ERRORS.pop(variant, None)
                else:
                    _SCANS[variant] = {}
                    _SCAN_ERRORS[variant] = err.strip().splitlines()[0] if err.strip() else 'unknown error'
                self._scan_changed()
        finally:
            self.ensure_scans()           # next reduction (or the first one of a new folder)

    @_guard
    def _scan_changed(self):
        if self.isVisible() and self.state.key is not None and self.state.catalog is not None:
            self._fill_table()
            self.compare.update_choices()
        else:
            self.stale = True             # rebuilt when the page is shown again

    def _rescan(self):
        _SCANS.clear()
        _SCAN_ERRORS.clear()
        self._variants = []               # also pick up reduction folders added meanwhile
        self.ensure_scans()
        self.refresh()

    def variants(self):
        """[(folder, label)] of the reductions, listed once per refresh (listing the OneDrive folder
        can take 100 ms while a scan reads files, so it is not repeated for every combo update)."""
        if not self._variants and self.state.root:
            self._variants = self.state.variants()
        return self._variants

    # -- page
    def refresh(self):
        if not self.has_galaxy():
            return
        self.show_body()
        self._variants = []
        self.ensure_scans()
        self._fill_table()
        self.compare.update_choices()

    def _fill_table(self):
        st = self.state
        key = st.key
        variants = self.variants()
        present = {v: _SCANS[v].get(key) for v, _ in variants if v in _SCANS}
        bands = sorted({b for cubes in present.values() if cubes for b in cubes}, key=band_rank)
        headers = ['', 'Reduction'] + bands + ['Total', 'Cubes', 'Telluric', 'Folder', 'Signal-to-noise Hα']
        tips = ['● = loaded reduction', 'Reduction (folder in the EMPOWER directory)'] + \
               [f'On-source time in {b} [h]' for b in bands] + \
               ['On-source time summed over the bands [h]', 'Number of combined cubes',
                'All cubes have a TELLEFF_ telluric file', 'Folder name', 'Signal-to-noise of the Hα line in the extracted spectrum']
        t = self.table
        t.clearSpans()
        t.setRowCount(len(variants))
        set_headers(t, headers, tips, stretch=len(headers) - 1)
        _align_headers(t, {**{c: NUM for c in range(2, 4 + len(bands))}, 0: CENTER,
                    4 + len(bands): CENTER, len(headers) - 1: NUM})
        n_mid = len(bands) + 3          # columns between 'Reduction' and 'Folder'
        for r, (v, label) in enumerate(variants):
            current = v == st.variant
            dim = None if (present.get(v) or current) else C.TEXT_DIM
            t.setItem(r, 0, _item('●' if current else '', 'loaded reduction' if current else None,
                                  C.ACCENT, align=CENTER))
            t.setItem(r, 1, _item(label, v, dim or (C.ACCENT if current else None), bold=current, data=v))
            t.setItem(r, len(headers) - 2, _item(v, v, C.TEXT_DIM))
            snr = self.compare.line_snr(v)
            t.setItem(r, len(headers) - 1,
                    _item(_f(snr, '{:.1f}'), 'Hα line S/N in the extracted spectrum',
                        _snr_color(snr), align=NUM))
            cubes = present.get(v)
            if v not in _SCANS:
                self._span(t, r, n_mid, 'scanning headers …', C.TEXT_DIM)
            elif v in _SCAN_ERRORS:
                self._span(t, r, n_mid, f'could not scan: {_SCAN_ERRORS[v]}', C.PINK)
            elif not cubes:
                self._span(t, r, n_mid, 'galaxy not in this reduction', C.TEXT_DIM)
            else:
                self._fill_present(t, r, bands, cubes)
        _fit_height(t, max_rows=8)

    def _update_table_snr(self):
        """Refresh the Hα S/N cells after the comparison worker returns."""
        col = self.table.columnCount() - 1
        for r, (variant, _) in enumerate(self._variants):
            snr = self.compare.line_snr(variant)
            self.table.setItem(r, col,
                               _item(_f(snr, '{:.1f}'), 'Hα line S/N in the extracted spectrum',
                                     _snr_color(snr), align=NUM))

    @staticmethod
    def _span(t, r, n, text, color):
        t.setSpan(r, 2, 1, n)
        t.setItem(r, 2, _item(text, color=color))
        for c in range(3, 2 + n):
            t.setItem(r, c, _item(''))

    @staticmethod
    def _fill_present(t, r, bands, cubes):
        total, n_cubes, tell = 0.0, 0, True
        for c, b in enumerate(bands):
            infos = cubes.get(b, [])
            if not infos:
                t.setItem(r, 2 + c, _item(DASH, f'no {b} cube', C.TEXT_DIM, align=NUM))
                continue
            sec = np.nansum([i['exptime_s'] for i in infos if _isnum(i.get('exptime_s'))])
            nfr = sum(int(i.get('n_frames') or 0) for i in infos)
            total += sec
            n_cubes += len(infos)
            tell &= all(bool(i.get('telluric')) for i in infos)
            tip = f'{len(infos)} cube(s) · {nfr} exposures\n' + '\n'.join(i.get('folder', '') for i in infos)
            t.setItem(r, 2 + c, _item(_hours(sec), tip, band_color(b), align=NUM))
        k = 2 + len(bands)
        t.setItem(r, k, _item(_hours(total), align=NUM, bold=True))
        t.setItem(r, k + 1, _item(str(n_cubes), align=NUM))
        t.setItem(r, k + 2, _check_item(tell))

    def _row_double_clicked(self, row, _col):
        if 0 <= row < len(self._variants):
            self.compare.compare_with(self._variants[row][0])


class _ComparePanel(QFrame):
    """One band of two reductions at the same sky position: overlaid spectra + line S/N table."""
    WINDOW_FULL = 'full'

    def __init__(self, page):
        super().__init__()
        self.page = page
        self.state = page.state
        self.setProperty('role', 'card')
        self._token = 0
        self._req = None                # request of the running or drawn job
        self._result = None             # [result A, result B] on screen
        self._result_req = None         # request that produced self._result
        self._snr_results = {}          # latest extracted result for every reduction in the band
        self._want = {'band': None, 'a': None, 'b': None, 'window': self.WINDOW_FULL}
        self._mode = self.state.options.get('mode', 'snr')
        self._timer = QTimer(self, singleShot=True, interval=120)
        self._timer.timeout.connect(self._start)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 12)
        lay.setSpacing(8)
        lay.addLayout(self._build_controls())
        body = QHBoxLayout()
        body.setSpacing(12)
        self.view = QStackedWidget()
        self.plot = make_plot('', 'Observed wavelength [µm]', 'S/N per pixel')
        # the plot title is the legend: it never covers the spectra and is part of an exported image
        self.empty = EmptyState()
        self.view.addWidget(self.plot)
        self.view.addWidget(self.empty)
        body.addWidget(self.view, 1)
        side = QVBoxLayout()
        side.setSpacing(6)
        side.addWidget(_label('LINE S/N', 'section',
                              'Calibrated line S/N (line_snr.measure_lines) of the spectrum of each reduction'))
        self.snr_table = _make_table(['Line', 'A', 'B', 'B / A'],
                                     ['Emission line', 'S/N in reduction A', 'S/N in reduction B',
                                      'S/N ratio B / A'])
        self.snr_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.snr_table.setFixedWidth(300)
        _align_headers(self.snr_table, {1: NUM, 2: NUM, 3: NUM})
        side.addWidget(self.snr_table, 1)
        self.side_note = _label('', 'dim', wrap=True)
        self.side_note.setFixedWidth(300)
        side.addWidget(self.side_note)
        body.addLayout(side)
        lay.addLayout(body, 1)

    def _build_controls(self):
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(_label('COMPARE SPECTRA', 'section'))
        row.addSpacing(6)
        self.band_combo = QComboBox()
        self.band_combo.setToolTip('Band to compare (only bands that exist in at least one reduction)')
        self.a_combo, self.b_combo = QComboBox(), QComboBox()
        self.window_combo = QComboBox()
        self.window_combo.setToolTip('Show the whole band or zoom on a line')
        # size hints independent of the longest entry, so long reduction names never widen the window
        for c, n in ((self.a_combo, 8), (self.b_combo, 8), (self.window_combo, 8)):
            c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(n)
        for c, tag in ((self.a_combo, 'A'), (self.b_combo, 'B')):
            c.setToolTip(f'Reduction {tag} (reductions that contain this galaxy in the chosen band)')
        self.band_combo.setMinimumWidth(72)
        row.addWidget(_label('Band', 'dim'))
        row.addWidget(self.band_combo)
        row.addWidget(self._swatch(COLOR_A, 'A'))
        row.addWidget(self.a_combo, 1)
        row.addWidget(self._swatch(COLOR_B, 'B'))
        row.addWidget(self.b_combo, 1)
        row.addWidget(_label('Window', 'dim'))
        row.addWidget(self.window_combo)
        self.mode_flux = _flat_button('Flux', 'Show the aperture flux')
        self.mode_snr = _flat_button('S/N', 'Show the S/N per pixel (flux / error)')
        grp = QButtonGroup(self)
        for b in (self.mode_flux, self.mode_snr):
            b.setCheckable(True)
            grp.addButton(b)
            row.addWidget(b)
        (self.mode_snr if self._mode == 'snr' else self.mode_flux).setChecked(True)
        self.mode_flux.clicked.connect(lambda: self._set_mode('flux'))
        self.mode_snr.clicked.connect(lambda: self._set_mode('snr'))
        row.addSpacing(4)
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setFixedWidth(90)
        self.busy.setMaximumHeight(6)
        self.busy.hide()
        self.status = _label('', 'dim')
        row.addWidget(self.busy)
        row.addWidget(self.status)
        self.band_combo.currentIndexChanged.connect(lambda _i: self._user_changed('band'))
        self.a_combo.currentIndexChanged.connect(lambda _i: self._user_changed('a'))
        self.b_combo.currentIndexChanged.connect(lambda _i: self._user_changed('b'))
        self.window_combo.currentIndexChanged.connect(lambda _i: self._window_changed())
        return row

    @staticmethod
    def _swatch(color, text):
        lab = QLabel(text)
        lab.setAlignment(Qt.AlignCenter)
        lab.setFixedSize(22, 22)
        lab.setStyleSheet(f'background-color: {color}; color: {C.BG_DARK}; border-radius: 11px; '
                          f'font-weight: bold; font-size: 12px;')
        return lab

    # -- choices
    def _variants_with(self, band):
        key = self.state.key
        return [(v, lab) for v, lab in self.page.variants()
                if v in _SCANS and band in (_SCANS[v].get(key) or {})]

    def update_choices(self):
        """Fill the band / reduction combos from the scans (keeping the user's choices)."""
        st = self.state
        key = st.key
        if self._result_req is not None and self._result_req[0] != key:
            # another galaxy: never show the previous galaxy's spectra while the new ones load
            self._show_message('Loading the cubes of this galaxy …', busy=True)
        counts = {}
        for v, _ in self.page.variants():
            for b in (_SCANS.get(v, {}).get(key) or {}):
                counts[b] = counts.get(b, 0) + 1
        bands = sorted(counts, key=band_rank)
        want = self._want['band']
        if want not in counts:
            multi = [b for b in bands if counts[b] >= 2]
            want = st.ref_band if st.ref_band in multi else (multi[0] if multi else (bands[0] if bands else None))
        self._fill_combo(self.band_combo, [(b, f'{b}', f'{counts[b]} reduction(s) contain {b}') for b in bands], want)
        self._fill_variant_combos()
        self.request()

    def _fill_variant_combos(self):
        band = self.band_combo.currentData()
        opts = self._variants_with(band) if band else []
        names = [v for v, _ in opts]
        a = self._want['a'] if self._want['a'] in names else (self.state.variant if self.state.variant in names
                                                               else (names[0] if names else None))
        b = self._want['b'] if self._want['b'] in names and self._want['b'] != a else \
            next((v for v in names if v != a), a)
        items = [(v, lab, v) for v, lab in opts]
        self._fill_combo(self.a_combo, items, a)
        self._fill_combo(self.b_combo, items, b)

    @staticmethod
    def _fill_combo(combo, items, select):
        """items: [(data, text, tooltip)]."""
        combo.blockSignals(True)
        combo.clear()
        for data, text, tip in items:
            combo.addItem(text, data)
            combo.setItemData(combo.count() - 1, tip, Qt.ToolTipRole)
        idx = combo.findData(select)
        combo.setCurrentIndex(idx if idx >= 0 else (0 if combo.count() else -1))
        combo.blockSignals(False)

    def _user_changed(self, which):
        combo = {'band': self.band_combo, 'a': self.a_combo, 'b': self.b_combo}[which]
        self._want[which] = combo.currentData()
        if which == 'band':
            self._fill_variant_combos()
        self.request()

    def compare_with(self, variant):
        """Double-click in the reductions table: loaded reduction (A) vs `variant` (B)."""
        if self.band_combo.currentData() is None:
            return
        names = [v for v, _ in self._variants_with(self.band_combo.currentData())]
        if variant not in names:
            self.state.status.emit(f'{variant} has no {self.band_combo.currentData()} cube of this galaxy', 4000)
            return
        self._want['a'] = self.state.variant if self.state.variant in names else self._want['a']
        self._want['b'] = variant
        self._fill_variant_combos()
        self.request()

    def _set_mode(self, mode):
        self._mode = mode
        self._redraw()

    def sync_mode(self):
        """Follow the app-wide 'mode' option (Spectrum tab)."""
        self._mode = self.state.options.get('mode', 'snr')
        (self.mode_snr if self._mode == 'snr' else self.mode_flux).setChecked(True)
        self._redraw()

    def _window_changed(self):
        self._want['window'] = self.window_combo.currentData()
        self._redraw()

    # -- computation
    def request(self):
        """Start (debounced) the comparison for the current choices; a burst of requests (scan
        results, stacksReady + apertureChanged) gives one start, never an endless postponement."""
        if not self._timer.isActive():
            self._timer.start()

    def _show_message(self, text, hint='', busy=False):
        """Replace plot and table by a message; forgets the drawn result (a later redraw, e.g. a
        Flux / S/N toggle, must not bring back the spectra of another galaxy)."""
        self.empty.set_text(text, hint)
        self.view.setCurrentWidget(self.empty)
        self.snr_table.clearSpans()
        self.snr_table.setRowCount(0)
        self.side_note.setText('')
        self.busy.setVisible(busy)
        self.status.setText('')
        self._result = None
        self._result_req = None
        self._snr_results = {}
        self._req = None

    def _sky(self):
        """(RA, Dec, r [arcsec]) of the current aperture, or None (no stacks / no WCS)."""
        st = self.state
        ref = st.stacks.get(st.ref_band) if st.stacks else None
        if ref is None or ref.wcs is None or st.ap_x is None:
            return None
        try:
            ra, dec = ref.wcs.pixel_to_world_values(st.ap_x, st.ap_y)
            return float(ra), float(dec), float(st.ap_r * ref.pixscale)
        except Exception:
            return None

    def _source(self, variant, band):
        st = self.state
        if variant == st.variant and band in st.stacks:
            return st.stacks[band]          # already loaded by the app: no second read
        return list(_SCANS[variant][st.key][band])

    @_guard
    def _start(self):
        st = self.state
        if st.catalog is None or st.key is None:
            return
        band, va, vb = self.band_combo.currentData(), self.a_combo.currentData(), self.b_combo.currentData()
        if band is None:
            self._show_message('No cubes to compare')
            return
        n_red = len(self._variants_with(band))
        if n_red < 2:
            scanning = self.page._scan_job is not None
            self._show_message(f'Only one reduction contains {band} of this galaxy' if not scanning
                               else 'Scanning the other reductions …',
                               'Comparisons need the band in two reductions.' if not scanning
                               else 'The comparison starts as soon as a second reduction with this band is found.')
            return
        if not st.stacks or st.ref_band not in st.stacks:
            self._show_message('Loading the cubes of this galaxy …')
            return
        sky = self._sky()
        z = st.z
        method, sub = st.options['method'], st.options['subtract_bg']
        sky_key = None if sky is None else tuple(round(v, 7) for v in sky)
        req = (st.key, band, va, vb, sky_key, round(st.ap_r, 3), method, sub, None if z is None else round(z, 6))
        if req == self._req:
            if self._result is not None:
                self._redraw()
            return                          # same request already drawn or still running
        n_load = len({va, vb})
        if self._result_req is None or self._result_req[:4] != req[:4]:
            # other galaxy / band / reductions: no stale spectra on screen while loading
            self._show_message(f'Extracting {band} from {n_load} reduction{"s" if n_load > 1 else ""} …',
                               'Cubes that are not loaded yet are read and stacked in the background.',
                               busy=True)
        self._req = req
        self._token += 1
        token = self._token
        # Extract every reduction in the selected band so the QA table can show
        # Hα S/N for all rows, while the plot still displays only A and B.
        sources = [(v, st.key, band, self._source(v, band))
               for v, _ in self.page.variants() if band in (_SCANS.get(v, {}).get(st.key) or {})]
        self.busy.show()
        self.busy.setToolTip(f'Extracting {band} from {n_load} reduction{"s" if n_load > 1 else ""} '
                             f'and fitting its lines')
        self.status.setText('Loading …')
        run_async(_compare_job, sources, sky, st.ap_r, z, method, sub,
                  on_done=lambda res, t=token, r=req: self._done(t, r, res),
                  on_error=lambda msg, t=token: self._failed(t, msg))

    def _current_choice(self):
        return (self.state.key, self.band_combo.currentData(), self.a_combo.currentData(),
                self.b_combo.currentData())

    def line_snr(self, variant):
        """Return the fitted Hα S/N for a reduction currently in the comparison."""
        result = self._snr_results.get(variant)
        if result is None:
            result = next((r for r in (self._result or []) if r['variant'] == variant), None)
        if result is None:
            return np.nan
        fit = next((f for key, f in result['fits'].items()
                    if str(key).lower() in ('ha', 'halpha', 'hα')), None)
        return fit.snr if fit is not None and fit.covered else np.nan

    @_guard
    def _done(self, token, req, res):
        # drop results of an older request, and results that arrive after a galaxy / band switch
        # but before the (debounced) new request started
        if token != self._token:
            return
        if req[:4] != self._current_choice():
            self._req = None        # nothing runs any more: the next start must not take it as pending
            return
        self._snr_results = {r['variant']: r for r in res}
        by_variant = {r['variant']: r for r in res}
        self._result = [by_variant[req[2]], by_variant[req[3]]]
        self._result_req = req
        self.busy.hide()
        self.status.setText('')
        self._fill_windows()
        self._redraw()

    @_guard
    def _failed(self, token, msg):
        if token != self._token:
            return
        first = msg.strip().splitlines()[0] if msg.strip() else 'unknown error'
        self._show_message('Could not compare the spectra', first)

    @property
    def _z(self):
        """Redshift the drawn result was fitted with (not a newer override still being fitted)."""
        return self._result_req[8] if self._result_req is not None else None

    # -- drawing
    def _fill_windows(self):
        z = self._z
        items = [(self.WINDOW_FULL, 'Full band', 'whole wavelength range')]
        if z is not None and self._result:
            lo, hi = self._wave_range()
            for p in PANELS:
                c, wlo, whi = panel_window(p, z)
                if lo <= c <= hi:
                    items.append((p[0], p[1], f'{wlo:.4f} – {whi:.4f} µm'))
        self._fill_combo(self.window_combo, items, self._want['window'])

    def _wave_range(self):
        los, his = [], []
        for r in self._result or []:
            w = r['spec'].wave[np.isfinite(r['spec'].flux)]
            if w.size:
                los.append(w.min())
                his.append(w.max())
        return (min(los), max(his)) if los else (np.nan, np.nan)

    def _y(self, res):
        sp = res['spec']
        y = sp.snr if self._mode == 'snr' else sp.flux
        return boxcar(y, self.state.options.get('smooth', 0))

    @_guard
    def _redraw(self):
        if not self._result:
            return
        a, b = self._result
        if a['n_pix'] == 0 and b['n_pix'] == 0:
            self._show_message('The aperture lies outside both cubes', 'Move the aperture onto the galaxy.')
            return
        self.view.setCurrentWidget(self.plot)
        labels = dict(self.page.variants())
        p = self.plot
        p.clear()
        add_spans(p, a['spans'] or b['spans'])
        lo, hi = self._wave_range()
        z = self._z
        if z is not None:
            for text, k in FULL_MARKERS:
                lam = REST[k] * (1 + z)
                if lo <= lam <= hi:      # labels near the red end point left, so they stay visible
                    add_vline(p, lam, text, LINE_COLORS.get(k, C.TEXT_DIM),
                              anchor_left=lam > lo + 0.88 * (hi - lo))
        curves, title = [], []
        for res, color, tag in ((a, COLOR_A, 'A'), (b, COLOR_B, 'B')):
            name = labels.get(res['variant'], res['variant'])
            if res['n_pix'] == 0:
                title.append(f'<span style="color:{C.TEXT_DIM}">{tag} · {name}: aperture outside the cube</span>')
                continue
            title.append(f'<span style="color:{color}">━━ {tag} · {name}</span>')
            y = self._y(res)
            # width 1: antialiased Qt pens wider than 1 px take ~1 s per repaint of a noisy spectrum
            p.plot(res['spec'].wave, y, pen=pg.mkPen(color, width=1), connect='finite')
            curves.append((res['spec'].wave, y))
        p.setTitle('&nbsp;&nbsp;&nbsp;&nbsp;'.join(title), size='10pt')
        unit = a['bunit'] or b['bunit'] or 'ADU/s'
        p.setLabel('left', 'S/N per pixel' if self._mode == 'snr' else f'Flux [{unit}]', color=C.TEXT_DIM)
        self._set_ranges(curves, lo, hi)
        self._fill_snr(a, b)
        self._set_notes(a, b)
        self.page._update_table_snr()

    def _set_ranges(self, curves, lo, hi):
        """x: the band or the chosen line window; y: robust range of the visible data (no spikes)."""
        win = self.window_combo.currentData() or self.WINDOW_FULL
        z = self._z
        if win != self.WINDOW_FULL and z is not None:
            panel = next((pp for pp in PANELS if pp[0] == win), None)
            if panel is not None:
                _, lo, hi = panel_window(panel, z)
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            return
        vals = np.concatenate([y[(w >= lo) & (w <= hi) & np.isfinite(y)] for w, y in curves]) if curves else []
        vb = self.plot.getPlotItem().getViewBox()
        vb.enableAutoRange(False)
        self.plot.setXRange(lo, hi, padding=0.01)
        if len(vals):
            q = (0.5, 99.5) if win == self.WINDOW_FULL else (0.0, 100.0)
            ylo, yhi = np.percentile(vals, q)
            if z is not None:            # emission-line peaks always stay inside the range
                near = [y[(np.abs(w - REST[k] * (1 + z)) < PEAK_TOL_UM) & (w >= lo) & (w <= hi) & np.isfinite(y)]
                        for w, y in curves for k in FIT_LINES]
                near = np.concatenate(near) if near else np.array([])
                if near.size:
                    yhi = max(yhi, float(near.max()))
            pad = 0.08 * (yhi - ylo if yhi > ylo else abs(yhi) or 1.0)
            self.plot.setYRange(ylo - pad, yhi + pad * 2.2, padding=0)

    def _fill_snr(self, a, b):
        t = self.snr_table
        z = self._z
        t.clearSpans()
        if z is None:
            t.setRowCount(1)
            t.setSpan(0, 0, 1, 4)
            t.setItem(0, 0, _item('Redshift unknown — no line S/N', color=C.PINK))
            return
        rows = []
        for k in FIT_LINES:
            fa, fb = a['fits'].get(k), b['fits'].get(k)
            if (fa is None or not fa.covered) and (fb is None or not fb.covered):
                continue
            rows.append((k, fa, fb))
        t.setRowCount(max(len(rows), 1))
        if not rows:
            t.setSpan(0, 0, 1, 4)
            t.setItem(0, 0, _item('No fitted line in this band', color=C.TEXT_DIM))
        for r, (k, fa, fb) in enumerate(rows):
            sa = fa.snr if fa is not None and fa.covered else np.nan
            sb = fb.snr if fb is not None and fb.covered else np.nan
            t.setItem(r, 0, _item(LINE_LABELS.get(k, k), color=LINE_COLORS.get(k), bold=True))
            t.setItem(r, 1, _item(_f(sa, '{:.1f}'), self._fit_tip(fa), _snr_color(sa), align=NUM))
            t.setItem(r, 2, _item(_f(sb, '{:.1f}'), self._fit_tip(fb), _snr_color(sb), align=NUM))
            ratio = sb / sa if _isnum(sa) and _isnum(sb) and abs(sa) > 1e-9 else np.nan
            color = None if not _isnum(ratio) else (C.EMERALD if ratio >= 1.1 else C.PINK if ratio <= 0.9 else None)
            t.setItem(r, 3, _item(_f(ratio, '{:.2f}×'), 'B better' if _isnum(ratio) and ratio > 1 else None,
                                  color, align=NUM))
        t.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)

    @staticmethod
    def _fit_tip(lf):
        if lf is None or not lf.covered:
            return 'not covered'
        kind = 'free (anchor)' if not lf.forced else 'forced to the anchor velocity and width'
        tip = f'S/N {lf.snr:.2f} · {kind}\nflux {lf.flux:.3g} ± {lf.flux_err:.3g}'
        if lf.flags:
            tip += '\nflags: ' + ', '.join(lf.flags)
        return tip

    def _set_notes(self, a, b):
        st = self.state
        labels = dict(self.page.variants())
        lines_ = []
        for res, tag in ((a, 'A'), (b, 'B')):
            pos = (f'aperture ({res["x"]:.1f}, {res["y"]:.1f}) r = {res["r"]:.1f} px'
                   if res['n_pix'] else 'aperture outside this cube')
            lines_.append(f'<b>{tag}</b> {labels.get(res["variant"], res["variant"])}: {res["describe"]}, '
                          f'{"telluric" if res["telluric"] else "no telluric"} · {pos}')
        where = 'same sky position as the current aperture' if self._sky() is not None else 'target centre'
        method = self._result_req[6] if self._result_req is not None else st.options['method']
        how = 'summed over' if method == 'sum' else 'weighted mean over'
        lines_.append(f'Both spectra at the {where}, {how} the aperture.')
        self.side_note.setText('<br>'.join(lines_))
        self.side_note.setTextFormat(Qt.RichText)
        same = a['variant'] == b['variant']
        self.status.setText('A = B' if same else '')
        self.status.setToolTip('A and B are the same reduction' if same else '')


# ====================================================================== 3. Redshift
def _zcsv_row(cat, key):
    """The galaxy's row of target_redshifts.csv (all columns), or None."""
    z = getattr(cat, 'zcsv', None)
    if z is None or z.empty or 'region' not in z.columns or key not in cat.galaxies.index:
        return None
    g = cat.galaxies.loc[key]
    sel = z[(z['region'] == g['field']) & (z['arm_target_id'].astype(str).str.strip() == str(g['tid']))]
    return None if sel.empty else sel.iloc[0]


def _catalogue_redshift(cat, key):
    """Catalog.redshift(key) without a hand-entered override (the catalogue still shows its own
    value while the app uses the typed one).  A shallow copy: the shared catalogue is not touched."""
    if key not in cat.overrides:
        return cat.redshift(key)
    plain = copy.copy(cat)
    plain.overrides = {}
    return plain.redshift(key)


def _fit_job(stacks, apertures, z, method, subtract_bg):
    """Worker: the same spectra and fit as AppState.spectra() / fits() of the current aperture."""
    spectra = {}
    for b, st in stacks.items():
        x, y, r = apertures[b]
        sp = analysis.extract_spectrum(st, x, y, r, method=method, subtract_bg=subtract_bg)
        if sp.n_pix > 0:
            spectra[b] = sp
    return analysis.fit_galaxy(spectra, stacks, z)


class RedshiftPage(_Page):
    """Catalogue / public / KMOS-line redshifts and the velocity offset of the current fit."""
    TRIGGERS = frozenset({'catalog', 'galaxy', 'stacks', 'aperture', 'redshift', 'extract'})

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self._token = 0
        self._fit_req = None
        self._fit = None
        self._warnings = []
        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.warn = _Banner(C.PINK)
        self.note = _Banner(C.AMBER)
        lay.addWidget(self.warn)
        lay.addWidget(self.note)
        grid = QGridLayout()
        grid.setSpacing(10)
        self.cat_card = _KVCard('Catalogue redshift', 'target_redshifts.csv — used by the app')
        self.pub_card = _KVCard('Public redshift', 'target_redshifts.csv')
        self.kmos_card = _KVCard('KMOS emission lines', 'kmos_galaxy_properties.csv')
        self.fit_card = _KVCard('Current fit', 'this aperture, all bands')
        for i, c in enumerate((self.cat_card, self.pub_card, self.kmos_card, self.fit_card)):
            grid.addWidget(c, i // 2, i % 2)
        lay.addLayout(grid, 1)

    def refresh(self):
        if not self.has_galaxy():
            return
        self.show_body()
        info = self._catalogue_info()
        row = self.state.galaxy_row()
        self._warnings = []
        notes = []
        self._fill_catalogue(info, notes)
        self._fill_public(info)
        self._fill_kmos(info, row)
        self._start_fit()
        self.note.set_messages(notes)
        self._show_warnings()

    def _catalogue_info(self):
        """Catalogue + public redshift details (catalogue values even when z is overridden)."""
        st = self.state
        r = st.redshift
        rc = _catalogue_redshift(st.catalog, st.key)
        zr = _zcsv_row(st.catalog, st.key)
        g = (lambda c: zr.get(c) if zr is not None else None)
        num = (lambda v: float(v) if _isnum(v) else np.nan)
        info = dict(z=rc.z if rc.known else None, type=rc.z_type, quality=rc.quality, source=rc.source,
                    target_list=rc.target_list, sep=rc.separation, pub_z=rc.public_z, pub_type=rc.public_type,
                    pub_source=rc.public_source,
                    override=r.z if r is not None and r.override and r.known else None)
        info.update(source_id=_s(g('source_id')), pub_quality=g('public_quality'),
                    pub_sep=num(g('public_separation_arcsec')), ra=num(g('ra_deg')), dec=num(g('dec_deg')))
        return info

    def _fill_catalogue(self, info, notes):
        z = info['z']
        if info['override'] is not None:
            dv = _dv(z, info['override'])
            notes.append(f'z = {info["override"]:.5f} was entered by hand in the app and is used instead of the '
                         f'catalogue value' + (f' (Δv = {dv:+.0f} km/s).' if _isnum(dv) else '.'))
            if _isnum(dv) and abs(dv) > DV_WARN_KMS:
                self._warnings.append(f'The redshift in use (entered by hand) is offset by Δv = {dv:+.0f} km/s '
                                      f'from the catalogue: lines are searched at the entered redshift.')
        if z is None:
            self._warnings.append('No catalogue redshift for this target: emission lines cannot be located, '
                                  'so line fits and S/N are disabled.')
            self.cat_card.set_content('unknown', 'target_redshifts.csv has no redshift for this target.',
                                      big_color=C.PINK)
            return
        if _isnum(info['sep']) and info['sep'] > SEP_WARN_ARCSEC:
            self._warnings.append(f'The catalogue source is {info["sep"]:.2f}″ from the KMOS target '
                                  f'(> {SEP_WARN_ARCSEC:.0f}″): possibly a different object.')
        rows = [('Type', _s(info['type']), 'spec = spectroscopic, phot = photometric', None),
                ('Quality', _f(info['quality'], '{:g}'), 'quality flag of the source catalogue', None),
                ('Source catalogue', _s(info['source']), None, None),
                ('Source ID', info['source_id'], None, None),
                ('Target list', _s(info['target_list']), 'KMOS target list the galaxy was found under', None),
                ('Separation', _f(info['sep'], '{:.2f}', '″'), 'KMOS target – catalogue source',
                 C.PINK if _isnum(info['sep']) and info['sep'] > SEP_WARN_ARCSEC else None)]
        self.cat_card.set_content(f'z = {z:.5f}', f'{_s(info["type"])} · quality {_f(info["quality"], "{:g}")}', rows)

    def _fill_public(self, info):
        pz = info['pub_z']
        if not _isnum(pz):
            self.pub_card.set_content(DASH, 'No public redshift for this target.', big_color=C.TEXT_DIM)
            return
        dv = _dv(info['z'], pz)
        bad = _isnum(dv) and abs(dv) > DV_WARN_KMS
        if bad:
            self._warnings.append(f'Catalogue and public redshift disagree: z = {info["z"]:.5f} vs {pz:.5f} '
                                  f'(Δv = {dv:+.0f} km/s).')
        rows = [('Type', _s(info['pub_type']), None, None),
                ('Source', _s(info['pub_source']), None, None),
                ('Quality', _s(info['pub_quality']), 'quality flag of the public catalogue', None),
                ('Separation', _f(info['pub_sep'], '{:.2f}', '″'), 'KMOS target – public source', None),
                ('Δv public − catalogue', _signed(dv), 'c · Δz / (1 + z_cat)', C.PINK if bad else None)]
        self.pub_card.set_content(f'z = {pz:.5f}', f'{_s(info["pub_type"])} · {_s(info["pub_source"])}', rows,
                                  big_color=C.PINK if bad else None)

    def _fill_kmos(self, info, row):
        get = (lambda c: row.get(c) if row is not None else None)
        zk, dvk = get('z_kmos'), get('dv_kmos')
        if not _isnum(zk):
            self.kmos_card.set_content(DASH, 'No KMOS-line redshift in the properties table '
                                             '(no line with enough S/N, or not measured).', big_color=C.TEXT_DIM)
            return
        bad = _isnum(dvk) and abs(float(dvk)) > DV_WARN_KMS
        if bad:
            self._warnings.append(f'The KMOS-line redshift of the properties table is offset by '
                                  f'Δv = {float(dvk):+.0f} km/s from the catalogue.')
        rows = [('Δv KMOS − catalogue', _signed(dvk), 'velocity offset of the KMOS lines (notebook value)',
                 C.PINK if bad else None),
                ('Line', _s(get('zline')), 'line that set the KMOS redshift', None),
                ('Survey (properties)', _s(get('cat_survey')), None, None),
                ('Quality (properties)', _f(get('cat_qual'), '{:g}'), None, None)]
        self.kmos_card.set_content(f'z = {float(zk):.5f}', f'from {_s(get("zline"))}', rows,
                                   big_color=C.PINK if bad else None)

    # -- fit of the current aperture (worker)
    def _start_fit(self):
        st = self.state
        z = st.z
        if z is None:
            self.fit_card.set_content(DASH, 'No fit: the redshift is unknown.', big_color=C.TEXT_DIM)
            return
        if not st.stacks or st.ref_band not in st.stacks or st.ap_x is None:
            self.fit_card.set_content('…', 'Waiting for the cubes of this galaxy.', big_color=C.TEXT_DIM)
            return
        method, sub = st.options['method'], st.options['subtract_bg']
        req = (st.key, st.ref_band, round(st.ap_x, 2), round(st.ap_y, 2), round(st.ap_r, 2), method, sub, round(z, 6))
        if req == self._fit_req and self._fit is not None:
            self._show_fit(*self._fit)
            return
        self._fit_req = req
        self._fit = None
        self._token += 1
        token = self._token
        stacks = dict(st.stacks)
        apertures = {b: st.aperture_in(b) for b in stacks}
        self.fit_card.set_content('…', 'Fitting the lines of the current aperture …', big_color=C.TEXT_DIM)
        run_async(_fit_job, stacks, apertures, z, method, sub,
                  on_done=lambda res, t=token, zz=z: self._fit_done(t, zz, res),
                  on_error=lambda msg, t=token: self._fit_failed(t, msg))

    @_guard
    def _fit_done(self, token, z, res):
        if token != self._token or self._fit_req is None or self._fit_req[0] != self.state.key:
            return                  # an older request, or the galaxy changed before the next refresh
        self._fit = (z, res[0], res[1])
        self._show_fit(*self._fit)

    @_guard
    def _fit_failed(self, token, msg):
        if token == self._token:
            self._fit_req = None
            self.fit_card.set_content('failed', msg.strip().splitlines()[0] if msg.strip() else '', big_color=C.PINK)

    def _show_fit(self, z, fits_, info):
        st = self.state
        self._warnings = [w for w in self._warnings if not w.startswith('The lines of the current aperture')]
        x, y, r = st.ap_x, st.ap_y, st.ap_r
        ap = f'({x:.1f}, {y:.1f}) r = {r:.1f} px on {st.ref_band}' if x is not None else DASH
        if info.anchor in (None, 'external'):
            rows = [('Anchor', 'none (no line ≥ 5σ)', 'measure_lines needs an anchor line with S/N ≥ 5', None),
                    ('Aperture', ap, None, None)]
            where = 'the redshift in use' if st.redshift is not None and st.redshift.override else                 'the catalogue redshift'
            self.fit_card.set_content('forced', f'All lines forced at {where} (σ = 80 km/s).', rows,
                                      big_color=C.TEXT_DIM)
            self._show_warnings()
            return
        dv = info.dv
        bad = _isnum(dv) and abs(dv) > DV_WARN_KMS
        if bad:
            self._warnings.append(f'The lines of the current aperture are offset by Δv = {dv:+.0f} km/s from the '
                                  f'catalogue redshift (anchor {LINE_LABELS.get(info.anchor, info.anchor)} in '
                                  f'{info.anchor_band}).')
        anchor_fit = fits_.get(info.anchor)
        rows = [('Anchor line', LINE_LABELS.get(info.anchor, info.anchor), 'line fitted freely; all others are forced '
                 'to its velocity and width', LINE_COLORS.get(info.anchor)),
                ('Band', info.anchor_band, None, band_color(info.anchor_band)),
                ('Anchor S/N', _f(anchor_fit.snr if anchor_fit else None, '{:.1f}'), None, None),
                ('σ intrinsic', _f(info.sigma_int, '{:.0f}', 'km/s'), 'LSF removed', None),
                ('Implied z', _f(z + dv / C_KMS * (1 + z), '{:.5f}'), 'z_cat + Δv/c · (1 + z_cat)', None),
                ('Aperture', ap, None, None)]
        self.fit_card.set_content(f'Δv = {dv:+.0f} km/s', 'line velocity relative to the redshift in use', rows,
                                  big_color=C.PINK if bad else C.EMERALD)
        self._show_warnings()

    def _show_warnings(self):
        self.warn.set_messages(self._warnings)


# ====================================================================== 4. Census
SNR_LINES = [('hb', 'Hβ'), ('oiii', '[OIII]5007'), ('ha', 'Hα'), ('nii', '[NII]6583'), ('sii', '[SII]')]
SNR_LINE_COLOR = {'hb': LINE_COLORS['Hb'], 'oiii': LINE_COLORS['OIII5007'], 'ha': LINE_COLORS['Ha'],
                  'nii': LINE_COLORS['NII6583'], 'sii': LINE_COLORS['SII6716']}
SNR_REGIONS = [('int', 'Integrated', 'integrated 1″ × 1″ box (5 × 5 spaxels) on the galaxy'),
               ('r0', '0–1 Rₑ', 'annulus 0–1.0 Rₑ (all lines forced to the integrated kinematics)'),
               ('r1', '1–1.5 Rₑ', 'annulus 1.0–1.5 Rₑ'),
               ('r2', '1.5–2 Rₑ', 'annulus 1.5–2.0 Rₑ')]
OVERVIEW_LINES = [('ha', 'Hα'), ('oiii', '[OIII]5007'), ('hb', 'Hβ'), ('sii', '[SII]')]


class CensusPage(_Page):
    """The galaxy's rows of the census, sample overview and S/N distribution tables."""
    TRIGGERS = frozenset({'catalog', 'galaxy'})

    CENSUS_COLUMNS = [
        ('band', 'Band', 'KMOS band'),
        ('n_obs', 'Observed', 'object exposures in the ESO archive on an active arm'),
        ('t_obs_h', 't obs', 'observed on-source time [h]'),
        ('n_reduced', 'Reduced', 'reconstructed frame present and the IFU holds data'),
        ('n_arm_empty', 'Arm empty', 'frame reduced but this IFU holds no data'),
        ('n_not_reduced', 'Not reduced', 'observed frame missing from the reduced-frame index'),
        ('n_combined', 'Combined', 'exposures in the combined cube (PROV keywords)'),
        ('t_combined_h', 't combined', 'on-source time in the combined cube [h]'),
        (None, 'Kept', 't combined / t obs'),
        ('name_mismatch', 'Name mismatch', 'reduced frames whose arm is assigned to another target id'),
        ('ha_in_band', 'Hα in band', 'Hα (vacuum) × (1 + z) lies inside the band'),
        ('target_lists', 'Target lists', 'KMOS target lists of the observations'),
        ('not_reduced_frames', 'Not-reduced frames', 'archive names of the frames that were not reduced'),
    ]

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        outer = QVBoxLayout(self.body)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.setSpacing(10)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        self.census_card = _Card('Observations and reduction', 'halpha_census.csv · one row per band')
        self.census_chips = _Chips()
        self.census_table = _make_table([h for _, h, _ in self.CENSUS_COLUMNS],
                                        [t for _, _, t in self.CENSUS_COLUMNS], stretch=len(self.CENSUS_COLUMNS) - 1)
        _align_headers(self.census_table, {**{c: NUM for c in range(1, 10)}, 10: CENTER})
        self.census_empty = _label('', 'dim')
        for w in (self.census_chips, self.census_table, self.census_empty):
            self.census_card.body.addWidget(w)
        lay.addWidget(self.census_card)

        self.overview_card = _Card('Sample overview', 'kmos_sample_overview.csv')
        self.overview_chips = _Chips()
        self.exp_table = _make_table(['Band', 'Observed', 'In cube', 'Team catalogue'],
                                     ['KMOS band', 'observed on-source time [h]',
                                      'on-source time in the combined cube [h]',
                                      'on-source time listed in the team catalogue [h]'])
        self.lines_table = _make_table(['Line', 'λ obs', 'In band window', 'In cube', 'Band', 'Team window'],
                                       ['emission line', 'observed wavelength [µm]',
                                        'inside the wavelength window of an observed band',
                                        'inside the data range of a combined cube', 'band(s) covering the line',
                                        'inside a band window according to the team catalogue'])
        _align_headers(self.exp_table, {1: NUM, 2: NUM, 3: NUM})
        _align_headers(self.lines_table, {1: NUM, 2: CENTER, 3: CENTER, 5: CENTER})
        tables = QHBoxLayout()
        tables.setSpacing(12)
        for t in (self.exp_table, self.lines_table):      # compact, side by side, as wide as their columns
            t.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
            t.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            tables.addWidget(t, 0, Qt.AlignTop)
        tables.addStretch()
        self.overview_empty = _label('', 'dim')
        self.overview_card.body.addWidget(self.overview_chips)
        self.overview_card.body.addLayout(tables)
        self.overview_card.body.addWidget(self.overview_empty)
        lay.addWidget(self.overview_card)

        self.snr_card = _Card('S/N distribution', 'kmos_snr_distribution.csv · deepest band per line')
        self.snr_chips = _Chips()
        self.snr_table = _make_table(['Line', 'Band', 't'] + [h for _, h, _ in SNR_REGIONS] + ['Anchor'],
                                     ['emission line', 'band the line was measured in', 'on-source time [h]']
                                     + [t for _, _, t in SNR_REGIONS] + ['anchor line of the galaxy'])
        self.snr_table.horizontalHeader().setSectionResizeMode(3 + len(SNR_REGIONS), QHeaderView.Stretch)
        _align_headers(self.snr_table, {c: NUM for c in range(2, 3 + len(SNR_REGIONS))})
        self.snr_empty = _label('', 'dim')
        for w in (self.snr_chips, self.snr_table, self.snr_empty):
            self.snr_card.body.addWidget(w)
        lay.addWidget(self.snr_card)
        lay.addStretch()
        for t in (self.census_table, self.exp_table, self.lines_table, self.snr_table):
            t.setSelectionMode(QAbstractItemView.NoSelection)

    def refresh(self):
        if not self.has_galaxy():
            return
        self.show_body()
        self._fill_census()
        self._fill_overview()
        self._fill_snr()

    @staticmethod
    def _missing(widgets, label, rows, name, fname):
        """Hide `widgets` and explain in `label` when the table or the galaxy's rows are missing."""
        ok = rows is not None and not rows.empty
        for w in widgets:
            w.setVisible(ok)
        label.setVisible(not ok)
        if rows is None:
            label.setText(f'{fname} not found in the EMPOWER folder.')
        elif rows.empty:
            label.setText(f'This galaxy is not in {fname}.')
        return not ok

    def _fill_census(self):
        rows = _table_rows(self.state, 'census')
        if self._missing((self.census_table, self.census_chips), self.census_empty, rows, 'census',
                         'halpha_census.csv'):
            return
        rows = rows.sort_values('band', key=lambda s: s.map(band_rank))
        t = self.census_table
        t.setRowCount(len(rows))
        for r, (_, row) in enumerate(rows.iterrows()):
            for c, (col, _, _) in enumerate(self.CENSUS_COLUMNS):
                t.setItem(r, c, self._census_item(col, row))
        _fit_height(t)
        t_obs = np.nansum(rows['t_obs_h'].astype(float)) if 't_obs_h' in rows else np.nan
        t_comb = np.nansum(rows['t_combined_h'].astype(float)) if 't_combined_h' in rows else np.nan
        n_lost = int(np.nansum(rows['n_not_reduced'].astype(float))) if 'n_not_reduced' in rows else 0
        n_mis = int(np.nansum(rows['name_mismatch'].astype(float))) if 'name_mismatch' in rows else 0
        first = rows.iloc[0]
        chips = [(f'observed {t_obs:.2f} h', 'summed over the bands', None),
                 (f'combined {t_comb:.2f} h' + (f' ({100 * t_comb / t_obs:.0f} %)' if t_obs > 0 else ''),
                  'on-source time that reached the combined cubes', None)]
        if n_lost:
            chips.append((f'{n_lost} exposure(s) not reduced', 'see the column Not-reduced frames', C.PINK))
        if n_mis:
            chips.append((f'{n_mis} name mismatch(es)', 'arm assigned to another target in reduced frames', C.PINK))
        if _s(first.get('ha_bands')) != DASH:
            chips.append((f'Hα at {_f(first.get("lam_ha_um"), "{:.4f}")} µm · inside {first.get("ha_bands")}',
                          'observed Hα wavelength and the KMOS band windows that contain it', None))
        self.census_chips.set_chips(chips)

    @staticmethod
    def _census_item(col, row):
        if col is None:      # derived: kept fraction of the observed time
            to, tc = row.get('t_obs_h'), row.get('t_combined_h')
            frac = float(tc) / float(to) if _isnum(to) and _isnum(tc) and float(to) > 0 else np.nan
            return _item(_f(frac * 100, '{:.0f} %'), align=NUM,
                         color=C.PINK if _isnum(frac) and frac < 0.8 else None)
        v = row.get(col)
        if col == 'band':
            return _item(_s(v), color=band_color(_s(v)), bold=True)
        if col == 'ha_in_band':
            return _check_item(v)
        if col in ('t_obs_h', 't_combined_h'):
            return _item(_f(v, '{:.2f}', 'h'), align=NUM)
        if col in ('target_lists', 'not_reduced_frames'):
            text = _s(v).replace(';', ', ')
            return _item(text, text if text != DASH else None)
        bad = col in ('n_not_reduced', 'name_mismatch') and _isnum(v) and float(v) > 0
        return _item(_f(v, '{:.0f}'), align=NUM, color=C.PINK if bad else None)

    def _fill_overview(self):
        rows = _table_rows(self.state, 'overview')
        if self._missing((self.overview_chips, self.exp_table, self.lines_table), self.overview_empty, rows,
                         'overview', 'kmos_sample_overview.csv'):
            return
        row = rows.iloc[0]
        in_team = _bool(row.get('in_team'))
        chips = [(f'set {_s(row.get("set"))}', 'observing set / mask', C.PINK if row.get('set') == 'bad_conditions'
                  else None),
                 (f'observed {_s(row.get("bands"))}', 'bands with observations', None),
                 (f'cubes {_s(row.get("bands_cube"))}', 'bands with a combined cube', None)]
        if in_team:
            chips.append((f'team catalogue: z = {_f(row.get("z_team"), "{:.5f}")}, q = {_f(row.get("q_team"), "{:g}")}',
                          'redshift and quality in the team catalogue', None))
        else:
            chips.append(('not in the team catalogue', None, C.TEXT_DIM))
        self.overview_chips.set_chips(chips)
        bands = [b for b in ('IZ', 'YJ', 'H', 'HK', 'K')
                 if any(_isnum(row.get(f'{p}_{b}')) and float(row.get(f'{p}_{b}')) > 0
                        for p in ('t_obs', 't_cube', 't_team'))]
        t = self.exp_table
        t.setRowCount(max(len(bands), 1))
        if not bands:
            t.setSpan(0, 0, 1, 4)
            t.setItem(0, 0, _item('no exposure times listed', color=C.TEXT_DIM))
        for r, b in enumerate(bands):
            t.setItem(r, 0, _item(b, color=band_color(b), bold=True))
            for c, p in enumerate(('t_obs', 't_cube', 't_team')):
                t.setItem(r, c + 1, _item(_f(row.get(f'{p}_{b}'), '{:.2f}', 'h'), align=NUM))
        t = self.lines_table
        t.setRowCount(len(OVERVIEW_LINES))
        for r, (k, lab) in enumerate(OVERVIEW_LINES):
            color = SNR_LINE_COLOR.get(k)
            t.setItem(r, 0, _item(lab, color=color, bold=True))
            t.setItem(r, 1, _item(_f(row.get(f'lam_{k}'), '{:.4f}', 'µm'), align=NUM))
            t.setItem(r, 2, _check_item(row.get(f'win_{k}')))
            t.setItem(r, 3, _check_item(row.get(f'cube_{k}')))
            t.setItem(r, 4, _item(_s(row.get(f'band_{k}'))))
            t.setItem(r, 5, _check_item(row.get(f'team_win_{k}')))
        _fit_height(self.exp_table, max_rows=max(len(bands), len(OVERVIEW_LINES)))
        _fit_height(self.lines_table)

    def _fill_snr(self):
        rows = _table_rows(self.state, 'snr_distribution')
        if self._missing((self.snr_chips, self.snr_table), self.snr_empty, rows, 'snr_distribution',
                         'kmos_snr_distribution.csv'):
            return
        t = self.snr_table
        present = [(k, lab) for k, lab in SNR_LINES if (rows['line'] == k).any()]
        t.setRowCount(len(present))
        for r, (k, lab) in enumerate(present):
            sub = rows[rows['line'] == k]
            first = sub.iloc[0]
            t.setItem(r, 0, _item(lab, color=SNR_LINE_COLOR.get(k), bold=True))
            t.setItem(r, 1, _item(_s(first.get('band')), color=band_color(_s(first.get('band')))))
            t.setItem(r, 2, _item(_f(first.get('texp_h'), '{:.2f}', 'h'), align=NUM))
            for c, (reg, _, tip) in enumerate(SNR_REGIONS):
                hit = sub[sub['region'] == reg]
                v = hit.iloc[0].get('snr') if not hit.empty else None
                det = _bool(hit.iloc[0].get('detected')) if not hit.empty else None
                cover = hit.iloc[0].get('ring_cover') if not hit.empty else None
                tt = f'{tip}\n' + ('detected' if det else 'not detected') + \
                     (f' · ring coverage {float(cover):.0%}' if _isnum(cover) and reg != 'int' else '')
                t.setItem(r, 3 + c, _item(_f(v, '{:.1f}'), tt, C.EMERALD if det else C.TEXT_DIM, bold=bool(det),
                                          align=NUM))
            anchor = _s(first.get('anchor'))
            t.setItem(r, 3 + len(SNR_REGIONS), _item(dict(SNR_LINES).get(anchor, anchor),
                                                     color=SNR_LINE_COLOR.get(anchor)))
        _fit_height(t)
        first = rows.iloc[0]
        re_px = first.get('re_px')
        chips = [(f'Rₑ = {_f(re_px, "{:.2f}")} px ({_f(float(re_px) * 0.2 if _isnum(re_px) else None, "{:.2f}")}″)',
                  'effective radius of the anchor-line map (0.2″ spaxels)', None),
                 (_s(first.get('re_src')), 'measured for this galaxy, or the sample median', None),
                 ('green = detected (S/N ≥ 3)', None, C.EMERALD)]
        self.snr_chips.set_chips(chips)


# ====================================================================== 5. Header
_HEADER_CACHE = OrderedDict()     # path -> [(HDU label, header text)], GUI thread only
MAX_HEADER_CACHE = 16


def _read_headers(path):
    """Worker: [(HDU label, header text)] of every HDU of a FITS file, one 80-character card per line."""
    out = []
    with fits.open(path, memmap=True, lazy_load_hdus=True) as hl:
        for i, hdu in enumerate(hl):
            name = hdu.name or ('PRIMARY' if i == 0 else f'HDU {i}')
            text = hdu.header.tostring(sep='\n', endcard=False, padding=False)
            out.append((f'{i} · {name}', '\n'.join(line.rstrip() for line in text.split('\n'))))
    return out


class _FitsHighlighter(QSyntaxHighlighter):
    """Keywords in cyan, comments dim, the filter text on amber."""

    def __init__(self, doc):
        super().__init__(doc)
        self.key_fmt = QTextCharFormat()
        self.key_fmt.setForeground(QColor(C.ACCENT))
        self.comment_fmt = QTextCharFormat()
        self.comment_fmt.setForeground(QColor(C.TEXT_DIM))
        self.match_fmt = QTextCharFormat()
        self.match_fmt.setBackground(QColor(C.AMBER))
        self.match_fmt.setForeground(QColor(C.BG_DARK))
        self.term = ''

    def highlightBlock(self, text):
        if text.startswith(('COMMENT', 'HISTORY')):
            self.setFormat(0, len(text), self.comment_fmt)
        else:
            eq = text.find('=')
            if eq > 0:
                self.setFormat(0, eq, self.key_fmt)
                c = self._comment_start(text, eq + 1)
                if c >= 0:
                    self.setFormat(c, len(text) - c, self.comment_fmt)
        if self.term:
            low, n, i = text.lower(), len(self.term), 0
            while True:
                i = low.find(self.term, i)
                if i < 0:
                    break
                self.setFormat(i, n, self.match_fmt)
                i += n

    @staticmethod
    def _comment_start(text, i):
        """Index of the ' / comment' after the value (string values may contain '/')."""
        rest = text[i:].lstrip()
        start = len(text) - len(rest)
        if rest.startswith("'"):
            j = start + 1
            while j < len(text):
                if text[j] == "'" and not text.startswith("''", j):
                    break
                j += 2 if text.startswith("''", j) else 1
            k = text.find('/', j + 1)
        else:
            k = text.find('/', start)
        return k


class HeaderPage(_Page):
    """FITS header of one of the galaxy's cubes, with a filter box."""
    TRIGGERS = frozenset({'catalog', 'galaxy'})

    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self._token = 0
        self._want_path = None
        self._hdus = []
        lay = QVBoxLayout(self.body)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.cube_combo = QComboBox()
        self.cube_combo.setMinimumWidth(320)
        self.cube_combo.setToolTip('Cube of this galaxy in the loaded reduction')
        self.hdu_combo = QComboBox()
        self.hdu_combo.setToolTip('Header unit: 0 = primary header (observation, instrument, pipeline)')
        self.hdu_combo.setMinimumWidth(130)
        self.hdu_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText('Filter cards, e.g.  AMBI FWHM   ·   AIRM   ·   PRO REC1')
        self.filter.setClearButtonEnabled(True)
        self.filter.setToolTip('Show only the cards that contain this text (case-insensitive)')
        self.count = _label('', 'dim')
        self.copy_btn = _flat_button('Copy', 'Copy the shown cards to the clipboard')
        row.addWidget(_label('Cube', 'dim'))
        row.addWidget(self.cube_combo)
        row.addWidget(_label('HDU', 'dim'))
        row.addWidget(self.hdu_combo)
        row.addWidget(self.filter, 1)
        row.addWidget(self.count)
        row.addWidget(self.copy_btn)
        lay.addLayout(row)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QPlainTextEdit.NoWrap)
        font = QFont('Consolas', 10)
        font.setStyleHint(QFont.Monospace)
        self.text.setFont(font)
        self.highlighter = _FitsHighlighter(self.text.document())
        lay.addWidget(self.text, 1)
        self.cube_combo.currentIndexChanged.connect(lambda _i: self._load())
        self.hdu_combo.currentIndexChanged.connect(lambda _i: self._apply_filter())
        # typing re-filters after a short pause (a primary header has ~2500 cards to re-highlight)
        self._filter_timer = QTimer(self, singleShot=True, interval=140)
        self._filter_timer.timeout.connect(self._apply_filter)
        self.filter.textChanged.connect(lambda _t: self._filter_timer.start())
        self.filter.returnPressed.connect(self._apply_filter)
        self.copy_btn.clicked.connect(lambda: QGuiApplication.clipboard().setText(self.text.toPlainText()))

    def select_path(self, path):
        """Show the header of `path` (from the Cubes page)."""
        self._want_path = path
        i = self.cube_combo.findData(path)
        if i >= 0 and i != self.cube_combo.currentIndex():
            self.cube_combo.setCurrentIndex(i)

    def refresh(self):
        if not self.has_galaxy():
            return
        infos = _galaxy_infos(self.state)
        if not infos:
            self.show_empty('No cubes for this galaxy')
            return
        self.show_body()
        keep = self._want_path or self.cube_combo.currentData()
        self.cube_combo.blockSignals(True)
        self.cube_combo.clear()
        for i in infos:
            self.cube_combo.addItem(f'{i["band"]}  ·  {os.path.basename(i["path"])}', i['path'])
            self.cube_combo.setItemData(self.cube_combo.count() - 1, i['path'], Qt.ToolTipRole)
        idx = self.cube_combo.findData(keep)
        self.cube_combo.setCurrentIndex(max(idx, 0))
        self.cube_combo.blockSignals(False)
        self._want_path = None
        self._load()

    @_guard
    def _load(self):
        path = self.cube_combo.currentData()
        if not path:
            return
        if path in _HEADER_CACHE:
            _HEADER_CACHE.move_to_end(path)
            self._show(path, _HEADER_CACHE[path])
            return
        self._token += 1
        token = self._token
        self.text.setPlainText('Reading header …')
        self.count.setText('')
        run_async(_read_headers, path, on_done=lambda res, t=token, p=path: self._loaded(t, p, res),
                  on_error=lambda msg, t=token: self._failed(t, msg))

    @_guard
    def _loaded(self, token, path, hdus):
        _HEADER_CACHE[path] = hdus
        while len(_HEADER_CACHE) > MAX_HEADER_CACHE:
            _HEADER_CACHE.popitem(last=False)
        if token == self._token:
            self._show(path, hdus)

    @_guard
    def _failed(self, token, msg):
        if token == self._token:
            self._hdus = []
            self.text.setPlainText('Could not read the header:\n\n' + msg)
            self.count.setText('')

    def _show(self, path, hdus):
        self._hdus = hdus
        keep = self.hdu_combo.currentIndex()
        self.hdu_combo.blockSignals(True)
        self.hdu_combo.clear()
        for label, _ in hdus:
            self.hdu_combo.addItem(label)
        self.hdu_combo.setCurrentIndex(keep if 0 <= keep < len(hdus) else 0)
        self.hdu_combo.blockSignals(False)
        self._apply_filter()

    def _apply_filter(self):
        i = self.hdu_combo.currentIndex()
        if not (0 <= i < len(self._hdus)):
            return
        lines_ = self._hdus[i][1].split('\n')
        term = self.filter.text().strip().lower()
        shown = [ln for ln in lines_ if term in ln.lower()] if term else lines_
        self.highlighter.term = term
        self.text.setPlainText('\n'.join(shown) if shown else '(no card contains this text)')
        self.count.setText(f'{len(shown)} of {len(lines_)} cards' if term else f'{len(lines_)} cards')


# ====================================================================== the tab
PAGES = [
    ('cubes', 'Cubes', CubesPage, 'Combined cubes of this galaxy: exposures, on-source time, conditions, files'),
    ('reductions', 'Reductions', ReductionsPage, 'Which reductions contain this galaxy; compare their spectra'),
    ('redshift', 'Redshift', RedshiftPage, 'Catalogue, public and KMOS-line redshifts; velocity offsets'),
    ('census', 'Census', CensusPage, 'Observed vs reduced vs combined exposures, line windows, S/N table'),
    ('header', 'Header', HeaderPage, 'FITS headers of the cubes'),
]

NAV_QSS = f"""
QFrame#qaNav {{ background-color: {C.BG_MEDIUM}; border: 1px solid {C.BG_LIGHT}; border-radius: 9px; }}
QFrame#qaNav QPushButton {{ background: transparent; color: {C.TEXT_DIM}; border: none; border-radius: 6px;
    padding: 6px 16px; font-weight: bold; }}
QFrame#qaNav QPushButton:hover {{ color: {C.TEXT}; background-color: {C.BG_LIGHT}; }}
QFrame#qaNav QPushButton:checked {{ color: white; background-color: {C.ACCENT}; }}
"""


class QATab(QWidget):
    """Data quality and provenance of the current galaxy (five pages, see the module docstring)."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 10, 0, 0)
        lay.setSpacing(10)
        lay.addLayout(self._build_nav())
        self.stack = QStackedWidget()
        self.pages = {}
        for name, _, cls, _ in PAGES:
            self.pages[name] = cls(state)
            self.stack.addWidget(self.pages[name])
        lay.addWidget(self.stack, 1)
        self.stack.currentChanged.connect(self._fit_stack_to_page)
        self._fit_stack_to_page(self.stack.currentIndex())
        self.pages['cubes'].showHeader.connect(self._show_header)

        self._timer = QTimer(self, singleShot=True)
        self._timer.timeout.connect(self._refresh_current)
        state.catalogChanged.connect(lambda: self._invalidate('catalog'))
        state.galaxyChanged.connect(lambda _k: self._invalidate('galaxy'))
        state.stacksReady.connect(lambda _k: self._invalidate('stacks'))
        state.apertureChanged.connect(lambda: self._invalidate('aperture', delay=250))
        state.redshiftChanged.connect(lambda: self._invalidate('redshift'))
        state.optionsChanged.connect(self._option_changed)
        self.nav.button(0).setChecked(True)
        self._update_context()

    def _fit_stack_to_page(self, index):
        """A QStackedWidget is as wide as its widest page: let only the visible page count."""
        for i in range(self.stack.count()):
            pol = QSizePolicy.Preferred if i == index else QSizePolicy.Ignored
            self.stack.widget(i).setSizePolicy(pol, pol)
        self.stack.adjustSize()

    def _build_nav(self):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        seg = QFrame()
        seg.setObjectName('qaNav')
        seg.setStyleSheet(NAV_QSS)
        sl = QHBoxLayout(seg)
        sl.setContentsMargins(3, 3, 3, 3)
        sl.setSpacing(2)
        self.nav = QButtonGroup(self)
        self.nav.setExclusive(True)
        for i, (_, title, _, tip) in enumerate(PAGES):
            b = QPushButton(title)
            b.setCheckable(True)
            b.setToolTip(tip)
            b.setCursor(Qt.PointingHandCursor)
            self.nav.addButton(b, i)
            sl.addWidget(b)
        self.nav.idClicked.connect(self.show_page)
        row.addWidget(seg)
        row.addStretch()
        self.context = _label('', 'dim')
        row.addWidget(self.context)
        return row

    def show_page(self, index):
        if isinstance(index, str):
            index = [p[0] for p in PAGES].index(index)
        self.nav.button(index).setChecked(True)
        self.stack.setCurrentIndex(index)
        self._refresh_current()

    def _show_header(self, path):
        self.pages['header'].select_path(path)
        self.show_page('header')

    def _update_context(self):
        st = self.state
        if st.catalog is None:
            self.context.setText('')
            return
        label = dict(st.variants()).get(st.variant, st.variant)
        self.context.setText(f'Reduction: {label}  ·  {st.variant}')
        self.context.setToolTip(os.path.join(st.root or '', st.variant or ''))

    def _invalidate(self, reason, delay=30):
        """Mark the pages that depend on `reason` stale; the visible one is rebuilt after `delay` ms
        (several signals in a row, e.g. stacksReady + apertureChanged, give one rebuild)."""
        if reason == 'catalog':
            self._update_context()
        for p in self.pages.values():
            if reason in p.TRIGGERS:
                p.stale = True
        if not self._timer.isActive() or delay > self._timer.remainingTime():
            self._timer.start(delay)

    def _option_changed(self, name):
        if name in ('method', 'subtract_bg'):
            self._invalidate('extract')
        elif name in ('mode', 'smooth'):
            cmp_ = self.pages['reductions'].compare
            if name == 'mode':
                cmp_.sync_mode()
            else:
                cmp_._redraw()

    def _refresh_current(self):
        if not self.isVisible():
            return
        page = self.stack.currentWidget()
        if page is None or not page.stale:
            return
        page.stale = False
        try:
            page.refresh()
        except Exception:
            tb = traceback.format_exc()
            print(tb, file=sys.stderr, flush=True)
            page.show_error(tb)

    def showEvent(self, ev):
        super().showEvent(ev)
        # start the header scans of the other reductions as soon as the tab is first looked at
        if self.state.catalog is not None:
            self.pages['reductions'].ensure_scans()
        self._timer.start(0)
