"""Left sidebar: the searchable, sortable list of all galaxies of the loaded reduction.

    search box     key, catalogue name, field or id; several words must all match ('cos 92')
    filter row     field, and three quick toggles: Hα S/N >= 5, more than one band, resolved maps
    table          Galaxy (◆ = resolved maps) | z | Bands | t [h] | S/N (Hα) | BPT   (click a header
                   to sort; the sorted column has a small accent triangle above its title)
    footer         'N of M galaxies' and a button that clears all filters

The galaxy key is the main identifier, so the Galaxy column gets all the room the other columns
leave; those are sized from their actual contents after every catalogue load.

Clicking a row or moving with Up/Down (also from inside the search box) selects the galaxy for the
whole app via state.select_galaxy; keyboard moves are debounced so holding a key does not start a
cube load for every row passed.  A galaxy selected elsewhere (e.g. in the Population tab) is
highlighted here without echoing back.

The table is a QAbstractTableModel over catalog.galaxies behind a QSortFilterProxyModel; numeric
columns sort numerically and unknown values (NaN) always stay at the bottom, in both directions.
"""
from dataclasses import dataclass, field

import numpy as np
from PyQt5.QtCore import (QAbstractTableModel, QModelIndex, QPointF, QRect, QSize, QSortFilterProxyModel,
                          Qt, QTimer)
from PyQt5.QtGui import QBrush, QColor, QFont, QFontMetrics, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QFrame, QHBoxLayout,
                             QHeaderView, QLabel, QLineEdit, QProgressBar, QPushButton, QSizePolicy,
                             QStackedWidget, QStyle, QStyledItemDelegate, QStyleOptionViewItem,
                             QTableView, QToolButton, QVBoxLayout, QWidget)

from empower_explorer.data.catalog import band_rank
from empower_explorer.theme import C, band_color

KEY_ROLE = Qt.UserRole + 1          # galaxy key of a row
SEGMENTS_ROLE = Qt.UserRole + 2     # [(text, colour, bold)] for the multi-colour cells
MAPS_ROLE = Qt.UserRole + 3         # True if the galaxy has resolved map products (◆ in the key cell)

SNR_GOOD, SNR_OK = 5.0, 3.0
FIELD_ORDER = ['COSMOS', 'CDFS', 'SPT2040', 'XMM2215']
KEYBOARD_DEBOUNCE_MS = 140
SELECTION_BG = '#164E63'            # muted accent: keeps the coloured cell text readable
MAP_MARK = '◆'
CELL_PAD = 2                        # px left/right inside a cell (stylesheet padding) ...
TEXT_MARGIN = 3                     # ... plus the style's own text margin (PM_FocusFrameHMargin + 1)
KEY_INSET = 10                      # px before the key (room for the accent bar of the selected row)
MARK_W = 15                         # px reserved at the right of the key cell for ◆

# (id, header, header tooltip, alignment, first click sorts descending)
COLUMNS = [
    ('key', 'Galaxy', 'Galaxy key: field and KMOS target id\n'
     f'{MAP_MARK} = resolved map products of the notebooks exist (Maps tab)', Qt.AlignLeft, False),
    ('z', 'z', 'Redshift used by the app (catalogue, or entered in the app, shown in cyan)\n'
     '— = unknown', Qt.AlignRight, False),
    ('bands', 'Bands', 'KMOS bands with combined cubes in this reduction', Qt.AlignLeft, False),
    ('texp', 't [h]', 'On-source time summed over all bands [hours]', Qt.AlignRight, True),
    ('snr', 'S/N', 'Integrated Hα S/N of the notebooks (line_snr)\n'
     'green ≥ 5 · amber 3–5 · grey < 3 · — = not measured', Qt.AlignRight, True),
    ('bpt', 'BPT', '[NII]-BPT class of the notebooks: SF, comp (composite), AGN\n'
     '"?" = class from an upper/lower limit', Qt.AlignLeft, False),
]
COL = {c[0]: i for i, c in enumerate(COLUMNS)}
# texts that must fit even before a catalogue is loaded (columns are re-fitted to the data later)
MIN_SAMPLES = {'z': ['0.000'], 'bands': ['IZ+YJ'], 'texp': ['18.8'], 'snr': ['103.4'], 'bpt': ['comp?']}


# ---------------------------------------------------------------------------------- row records
def _num(row, col):
    try:
        v = float(row.get(col, np.nan))
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def _text(row, col):
    v = row.get(col, '')
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ''
    s = str(v).strip()
    return '' if s.lower() in ('nan', 'none', '—', '-') else s


def _natural(field_, tid):
    s = str(tid)
    return (field_, 0, int(s), '') if s.isdigit() else (field_, 1, 0, s)


def class_color(cls):
    """Colour of a BPT / WHAN class ('SF?', 'comp', 'AGN', 'sAGN', 'retired', ...)."""
    c = (cls or '').lower()
    if c.startswith('sf'):
        return C.EMERALD
    if c.startswith('comp') or c.startswith('wagn'):
        return C.AMBER
    if 'agn' in c or c.startswith('sey') or c.startswith('liner'):
        return C.PINK
    return C.TEXT_DIM


def snr_color(snr):
    if not np.isfinite(snr):
        return C.TEXT_DIM
    return C.EMERALD if snr >= SNR_GOOD else (C.AMBER if snr >= SNR_OK else C.TEXT_DIM)


def fmt_hours(h):
    if not np.isfinite(h):
        return '—'
    return f'{h:.1f} h' if h >= 1 else f'{h * 60:.0f} min'


@dataclass
class _Row:
    key: str
    field: str
    tid: str
    name: str = ''
    z: float = np.nan
    z_type: str = ''
    z_quality: float = np.nan
    z_source: str = ''
    override: bool = False
    bands: list = field(default_factory=list)
    texp: float = np.nan
    texp_band: dict = field(default_factory=dict)
    cubes_band: dict = field(default_factory=dict)
    snr: float = np.nan
    bpt: str = ''
    whan: str = ''
    maps: bool = False
    haystack: str = ''
    natural: tuple = ()
    tooltip: str = ''


def _make_row(cat, key, row):
    fld, tid = _text(row, 'field'), _text(row, 'tid')
    if not fld:
        fld, _, tid = str(key).partition(' ')
    bands = sorted(cat.cubes.get(key, {}), key=band_rank) or [b for b in _text(row, 'bands').split('+') if b]
    # the galaxy table already holds the redshift (set_override keeps its 'z' column up to date), so no
    # per-galaxy catalogue look-up is needed - building 170 rows stays well below 50 ms
    override = key in getattr(cat, 'overrides', {})
    z = _num(row, 'z')
    r = _Row(key=key, field=fld, tid=tid, name=_text(row, 'name'),
             z=z if z > 0 else np.nan, z_type='manual' if override else _text(row, 'z_type'),
             z_quality=np.nan if override else _num(row, 'z_quality'),
             z_source='' if override else _text(row, 'z_source'), override=override, bands=bands,
             texp=_num(row, 'texp_h'),
             texp_band={b: _num(row, f'texp_{b}_h') for b in bands},
             cubes_band={b: len(cat.cubes.get(key, {}).get(b, [])) for b in bands},
             snr=_num(row, 'snr_ha'), bpt=_text(row, 'class_NII_BPT'), whan=_text(row, 'whan'),
             maps=bool(row.get('has_maps', False)) or key in cat.products)
    r.haystack = ' '.join([key, r.name, fld, tid]).lower()
    r.natural = _natural(fld, tid)
    r.tooltip = _tooltip(r)
    return r


def _tooltip(r):
    dim = f'color:{C.TEXT_DIM}'
    out = [f'<b style="font-size:14px">{r.key}</b>']
    if r.name:
        out.append(f'<span style="{dim}">{r.name}</span>')
    if np.isfinite(r.z):
        if r.override:
            how = 'entered in the app'
        else:
            bits = [b for b in (r.z_type, f'Q {r.z_quality:.0f}' if np.isfinite(r.z_quality) else '',
                                r.z_source) if b]
            how = ' · '.join(bits) or 'catalogue'
        out.append(f'z = {r.z:.5f} <span style="{dim}">({how})</span>')
    else:
        out.append(f'<span style="color:{C.PINK}">no redshift</span>')
    bands = []
    for b in r.bands:
        n = r.cubes_band.get(b, 0)
        cubes = f' ({n} cubes)' if n > 1 else ''
        bands.append(f'<span style="color:{band_color(b)}">{b}</span> {fmt_hours(r.texp_band.get(b, np.nan))}{cubes}')
    if bands:
        out.append(' · '.join(bands) + f' <span style="{dim}">= {fmt_hours(r.texp)} on source</span>')
    facts = []
    if np.isfinite(r.snr):
        facts.append(f'S/N Hα <span style="color:{snr_color(r.snr)}">{r.snr:.1f}</span>')
    if r.bpt:
        facts.append(f'BPT <span style="color:{class_color(r.bpt)}">{r.bpt}</span>')
    if r.whan:
        facts.append(f'WHAN <span style="color:{class_color(r.whan)}">{r.whan}</span>')
    if facts:
        out.append(' · '.join(facts))
    if r.maps:
        out.append(f'<span style="color:{C.PURPLE}">{MAP_MARK} resolved map products</span>')
    return '<br>'.join(out)


# ---------------------------------------------------------------------------------- model
class GalaxyModel(QAbstractTableModel):
    """One row per galaxy of state.catalog (records precomputed: painting and sorting stay cheap)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows = []
        self._index = {}

    def set_catalog(self, cat):
        self.beginResetModel()
        g = cat.galaxies if cat is not None else None
        self._rows = [] if g is None or g.empty else [_make_row(cat, k, row) for k, row in g.iterrows()]
        self._index = {r.key: i for i, r in enumerate(self._rows)}
        self.endResetModel()

    def refresh_key(self, cat, key):
        """Re-read one galaxy (e.g. its redshift was overridden)."""
        i = self._index.get(key)
        if i is None or cat is None or key not in cat.galaxies.index:
            return
        self._rows[i] = _make_row(cat, key, cat.galaxies.loc[key])
        self.dataChanged.emit(self.index(i, 0), self.index(i, len(COLUMNS) - 1))

    def record(self, i):
        return self._rows[i]

    def source_row(self, key):
        return self._index.get(key)

    def fields(self):
        """[(field, number of galaxies)] in the usual order."""
        counts = {}
        for r in self._rows:
            counts[r.field] = counts.get(r.field, 0) + 1
        order = [f for f in FIELD_ORDER if f in counts] + sorted(f for f in counts if f not in FIELD_ORDER)
        return [(f, counts[f]) for f in order]

    # Qt interface
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation != Qt.Horizontal or not 0 <= section < len(COLUMNS):
            return None
        cid, title, tip, align, desc = COLUMNS[section]
        if role == Qt.DisplayRole:
            return title
        if role == Qt.ToolTipRole:
            return tip + '\n\nClick to sort (again: reverse); unknown values always stay at the bottom.'
        if role == Qt.TextAlignmentRole:
            return int(align | Qt.AlignVCenter)
        if role == Qt.InitialSortOrderRole:
            return Qt.DescendingOrder if desc else Qt.AscendingOrder
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        r = self._rows[index.row()]
        cid = COLUMNS[index.column()][0]
        if role == Qt.DisplayRole:
            return self._display(r, cid)
        if role == Qt.ForegroundRole:
            c = self._color(r, cid)
            return QBrush(QColor(c)) if c else None
        if role == Qt.TextAlignmentRole:
            return int(COLUMNS[index.column()][3] | Qt.AlignVCenter)
        if role == Qt.ToolTipRole:
            return r.tooltip
        if role == KEY_ROLE:
            return r.key
        if role == SEGMENTS_ROLE:
            return self._segments(r, cid)
        if role == MAPS_ROLE:
            return r.maps if cid == 'key' else False
        return None

    @staticmethod
    def _display(r, cid):
        if cid == 'key':
            return r.key
        if cid == 'z':
            return f'{r.z:.3f}' if np.isfinite(r.z) else '—'
        if cid == 'bands':
            return '+'.join(r.bands)
        if cid == 'texp':
            return f'{r.texp:.1f}' if np.isfinite(r.texp) else '—'
        if cid == 'snr':
            return f'{r.snr:.1f}' if np.isfinite(r.snr) else '—'
        if cid == 'bpt':
            return r.bpt
        return ''

    @staticmethod
    def _color(r, cid):
        if cid == 'z':
            return C.TEXT_DIM if not np.isfinite(r.z) else (C.ACCENT if r.override else None)
        if cid == 'snr':
            return snr_color(r.snr)
        if cid == 'bpt':
            return class_color(r.bpt)
        if cid == 'texp':
            return None if np.isfinite(r.texp) else C.TEXT_DIM
        return None

    @staticmethod
    def _segments(r, cid):
        if cid == 'key':
            return [(r.field + ' ', C.TEXT_DIM, False), (r.tid, C.TEXT, True)]
        if cid == 'bands':
            segs = []
            for i, b in enumerate(r.bands):
                if i:
                    segs.append(('+', C.TEXT_DIM, False))
                segs.append((b, band_color(b), False))
            return segs
        return None

    @staticmethod
    def sort_value(r, cid):
        """Comparable value of a cell, or None if unknown (sorted last)."""
        if cid == 'key':
            return r.natural
        if cid in ('z', 'texp', 'snr'):
            v = getattr(r, cid)
            return v if np.isfinite(v) else None
        if cid == 'bands':
            return (len(r.bands), tuple(band_rank(b) for b in r.bands)) if r.bands else None
        if cid == 'bpt':
            return r.bpt.lower() or None
        return None


class GalaxyFilter(QSortFilterProxyModel):
    """Text / field / toggle filters, and sorting with unknown values always at the bottom."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tokens = []
        self.field = None
        self.need_snr = False
        self.need_multiband = False
        self.need_maps = False

    def active(self):
        return bool(self.tokens or self.field or self.need_snr or self.need_multiband or self.need_maps)

    def filterAcceptsRow(self, src_row, parent):
        r = self.sourceModel().record(src_row)
        if self.field and r.field != self.field:
            return False
        if self.need_snr and not (np.isfinite(r.snr) and r.snr >= SNR_GOOD):
            return False
        if self.need_multiband and len(r.bands) < 2:
            return False
        if self.need_maps and not r.maps:
            return False
        return all(t in r.haystack for t in self.tokens)

    def lessThan(self, left, right):
        src = self.sourceModel()
        a_row, b_row = src.record(left.row()), src.record(right.row())
        cid = COLUMNS[left.column()][0]
        a, b = src.sort_value(a_row, cid), src.sort_value(b_row, cid)
        # Qt sorts descending with lessThan(right, left), so everything that must not reverse
        # (unknown values at the bottom, ties in natural key order) flips with the order
        ascending = self.sortOrder() == Qt.AscendingOrder
        if a is None or b is None or a == b:
            if (a is None) != (b is None):
                return (b is None) if ascending else (a is None)
            return (a_row.natural < b_row.natural) if ascending else (a_row.natural > b_row.natural)
        return a < b


# ---------------------------------------------------------------------------------- painting
class _RowDelegate(QStyledItemDelegate):
    """Multi-colour text for the Galaxy and Bands cells, the ◆ maps mark at the right of the Galaxy
    cell, and an accent bar on the selected row."""

    def paint(self, painter, option, index):
        segs = index.data(SEGMENTS_ROLE)
        if not segs:
            super().paint(painter, option, index)
        else:
            opt = QStyleOptionViewItem(option)
            self.initStyleOption(opt, index)
            opt.text = ''
            style = opt.widget.style() if opt.widget is not None else QApplication.style()
            style.drawControl(QStyle.CE_ItemViewItem, opt, painter, opt.widget)
            key_cell = index.column() == 0
            self._draw_segments(painter, opt, segs, inset=KEY_INSET if key_cell else CELL_PAD + TEXT_MARGIN,
                                right=MARK_W if key_cell else CELL_PAD)
            if key_cell and index.data(MAPS_ROLE):
                painter.save()
                painter.setPen(QPen(QColor('#A78BFA' if option.state & QStyle.State_Selected else C.PURPLE)))
                painter.drawText(option.rect.adjusted(0, 0, -(CELL_PAD + 1), 0), int(Qt.AlignRight | Qt.AlignVCenter),
                                 MAP_MARK)
                painter.restore()
        if index.column() == 0 and option.state & QStyle.State_Selected:
            painter.fillRect(QRect(option.rect.left(), option.rect.top() + 3, 3, option.rect.height() - 6),
                             QColor(C.ACCENT))

    @staticmethod
    def _draw_segments(painter, opt, segs, inset, right=3):
        painter.save()
        rect = opt.rect.adjusted(inset, 0, -right, 0)
        x = rect.left()
        for text, color, bold in segs:
            f = QFont(opt.font)
            if bold:
                f.setWeight(QFont.DemiBold)
            fm = QFontMetrics(f)
            room = rect.right() - x
            if room <= 4:
                break
            shown = fm.elidedText(text, Qt.ElideRight, room)
            painter.setFont(f)
            painter.setPen(QPen(QColor(color)))
            painter.drawText(QRect(x, rect.top(), room, rect.height()), int(Qt.AlignLeft | Qt.AlignVCenter), shown)
            x += fm.horizontalAdvance(shown)
        painter.restore()


class _SortHeader(QHeaderView):
    """Horizontal header that marks the sorted column with a small accent triangle centred above
    its title (as in the Windows Explorer).  Qt's own indicator sits at the right edge and collides
    with right-aligned titles in narrow columns, so it is switched off and sorting is driven here."""

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.sort_col, self.sort_order = -1, Qt.AscendingOrder
        self.setSectionsClickable(True)
        self.setSortIndicatorShown(False)
        self.setHighlightSections(False)
        self.setSectionsMovable(False)
        self.setCursor(Qt.PointingHandCursor)

    def set_sort(self, col, order):
        self.sort_col, self.sort_order = col, order
        self.viewport().update()

    def paintSection(self, painter, rect, logical):
        painter.save()
        super().paintSection(painter, rect, logical)
        painter.restore()
        if logical != self.sort_col or not rect.isValid():
            return
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(C.ACCENT))
        cx, top = rect.center().x() + 0.5, rect.top() + 2.5
        if self.sort_order == Qt.DescendingOrder:        # ▼ largest first
            pts = [QPointF(cx - 4, top), QPointF(cx + 4, top), QPointF(cx, top + 4)]
        else:                                            # ▲ smallest first
            pts = [QPointF(cx - 4, top + 4), QPointF(cx + 4, top + 4), QPointF(cx, top)]
        painter.drawPolygon(QPolygonF(pts))
        painter.restore()


def _search_icon():
    pm = QPixmap(32, 32)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(QColor(C.TEXT_DIM), 3.2))
    p.drawEllipse(5, 5, 15, 15)
    p.drawLine(18, 18, 27, 27)
    p.end()
    return QIcon(pm)


class _Message(QWidget):
    """Centred message shown instead of the table (scanning, no data, no match)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.addStretch(2)
        self.title = QLabel('')
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setWordWrap(True)
        self.title.setStyleSheet(f'color: {C.TEXT}; font-size: 14px; font-weight: bold;')
        self.detail = QLabel('')
        self.detail.setAlignment(Qt.AlignCenter)
        self.detail.setWordWrap(True)
        self.detail.setProperty('role', 'dim')
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.setFixedHeight(6)
        self.bar.setTextVisible(False)
        self.button = QPushButton('Clear filters')
        self.button.setProperty('flat', 'true')
        self.button.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.title)
        lay.addWidget(self.detail)
        lay.addSpacing(6)
        lay.addWidget(self.bar)
        lay.addWidget(self.button, 0, Qt.AlignHCenter)
        lay.addStretch(3)

    def show_message(self, title, detail='', busy=False, button=False):
        self.title.setText(title)
        self.detail.setText(detail)
        self.detail.setVisible(bool(detail))
        self.bar.setVisible(busy)
        self.button.setVisible(button)


# ---------------------------------------------------------------------------------- the view
class GalaxyList(QWidget):
    """Sidebar with search, filters and the galaxy table; drives state.select_galaxy."""

    WIDTH = 376          # fits 'COSMOS recenter7' + ◆ next to the five data columns
    MAX_WIDTH = 560

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._syncing = False          # True while the selection is changed programmatically
        self._scanning = False
        self._sorted_once = False
        self._pending_key = None
        self._batch = False            # True while clear_filters resets several widgets

        self.setMinimumWidth(self.WIDTH)
        self.setMaximumWidth(self.MAX_WIDTH)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self._build_ui()

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(lambda: self._select(self._pending_key))

        state.catalogLoading.connect(self._on_catalog_loading)
        state.catalogChanged.connect(self._on_catalog)
        state.galaxyChanged.connect(self._sync_to)
        state.redshiftChanged.connect(self._on_redshift)
        state.error.connect(self._on_error)
        if state.catalog is not None:
            self._on_catalog()
        else:
            self.message.show_message('No galaxies yet', 'Waiting for the EMPOWER data folder …')
            self.stack.setCurrentWidget(self.message)

    def sizeHint(self):
        return QSize(self.WIDTH, 800)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 2, 4)
        card = QFrame()
        card.setProperty('role', 'card')
        outer.addWidget(card)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(10, 12, 10, 8)
        lay.setSpacing(8)

        head = QHBoxLayout()
        title = QLabel('GALAXIES')
        title.setProperty('role', 'section')
        head.addWidget(title)
        head.addStretch(1)
        hint = QLabel('Ctrl+F')
        hint.setProperty('role', 'dim')
        hint.setStyleSheet(f'color: {C.BG_HOVER}; font-size: 11px;')
        hint.setToolTip('Ctrl+F jumps to the search box; ↑/↓ change the galaxy')
        head.addWidget(hint)
        lay.addLayout(head)

        self.search = QLineEdit()
        self.search.setPlaceholderText('Search galaxy, field, id…')
        self.search.setClearButtonEnabled(True)
        self.search.addAction(_search_icon(), QLineEdit.LeadingPosition)
        self.search.setToolTip('Filters instantly on key, catalogue name, field and target id.\n'
                               'Several words must all match, e.g. "cos 92".  ↑/↓ change the galaxy, Esc clears.')
        self.search.textChanged.connect(self._apply_filters)
        self.search.installEventFilter(self)
        lay.addWidget(self.search)

        row = QHBoxLayout()
        row.setSpacing(5)
        self.field_combo = QComboBox()
        self.field_combo.setToolTip('Show only the galaxies of one field')
        self.field_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.field_combo.currentIndexChanged.connect(self._apply_filters)
        row.addWidget(self.field_combo, 1)
        self.btn_snr = self._toggle('Hα ≥ 5σ', 'Only galaxies with an integrated Hα S/N of at least 5')
        self.btn_multi = self._toggle('Multi-band', 'Only galaxies observed in more than one band')
        self.btn_maps = self._toggle('Maps', 'Only galaxies with resolved map products (Maps tab)')
        for b in (self.btn_snr, self.btn_multi, self.btn_maps):
            row.addWidget(b)
        lay.addLayout(row)

        self.model = GalaxyModel(self)
        self.proxy = GalaxyFilter(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setDynamicSortFilter(True)
        self.table = self._make_table()
        self._fit_columns()
        self.message = _Message()
        self.message.button.clicked.connect(self.clear_filters)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.table)
        self.stack.addWidget(self.message)
        lay.addWidget(self.stack, 1)

        foot = QHBoxLayout()
        self.count_label = QLabel('')
        self.count_label.setProperty('role', 'dim')
        foot.addWidget(self.count_label)
        foot.addStretch(1)
        self.clear_btn = QPushButton('Clear filters')
        self.clear_btn.setProperty('flat', 'true')
        self.clear_btn.setCursor(Qt.PointingHandCursor)
        self.clear_btn.setStyleSheet('padding: 2px 8px; font-size: 12px;')
        self.clear_btn.setToolTip('Show all galaxies again')
        self.clear_btn.clicked.connect(self.clear_filters)
        self.clear_btn.hide()
        foot.addWidget(self.clear_btn)
        lay.addLayout(foot)

    def _toggle(self, text, tip):
        b = QToolButton()
        b.setText(text)
        b.setCheckable(True)
        b.setToolTip(tip)
        b.setCursor(Qt.PointingHandCursor)
        b.setStyleSheet('QToolButton { padding: 5px 7px; font-size: 12px; }')
        b.toggled.connect(self._apply_filters)
        return b

    def _make_table(self):
        t = QTableView()
        t.setModel(self.proxy)
        t.setItemDelegate(_RowDelegate(t))
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setSelectionMode(QAbstractItemView.SingleSelection)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setShowGrid(False)
        t.setAlternatingRowColors(True)
        t.setWordWrap(False)
        t.setCornerButtonEnabled(False)
        t.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        t.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        t.setFrameShape(QFrame.NoFrame)
        t.setStyleSheet(
            f'QTableView {{ background-color: {C.BG_MEDIUM}; border: none; font-size: 12px; '
            f'selection-background-color: {SELECTION_BG}; selection-color: {C.TEXT}; outline: none; }}'
            f'QTableView::item {{ padding: 0px {CELL_PAD}px; border: none; }}'
            f'QHeaderView::section {{ background-color: {C.BG_MEDIUM}; color: {C.TEXT_DIM}; border: none;'
            f' border-bottom: 1px solid {C.BG_LIGHT}; padding: 8px {CELL_PAD + TEXT_MARGIN}px 4px;'
            f' font-size: 11px; font-weight: bold; }}'
            f'QHeaderView::section:first {{ padding-left: {KEY_INSET}px; }}'   # aligned with the keys
            f'QHeaderView::section:hover {{ color: {C.TEXT}; }}')
        vh = t.verticalHeader()
        vh.hide()
        vh.setDefaultSectionSize(27)
        vh.setSectionResizeMode(QHeaderView.Fixed)
        self.header = _SortHeader(t)
        t.setHorizontalHeader(self.header)
        self.header.setMinimumSectionSize(20)
        self.header.sectionClicked.connect(self._on_header_clicked)
        for i in range(len(COLUMNS)):
            self.header.setSectionResizeMode(i, QHeaderView.Stretch if i == COL['key'] else QHeaderView.Fixed)
        t.setSortingEnabled(False)          # sorting is driven by _SortHeader / _sort
        t.selectionModel().currentRowChanged.connect(self._on_current_row)
        return t

    def _fit_columns(self):
        """Data columns just as wide as their widest text (or title); the Galaxy column gets the rest."""
        f = QFont(self.table.font())
        f.setPixelSize(12)
        fm = QFontMetrics(f)
        hf = QFont(f)
        hf.setPixelSize(11)
        hf.setBold(True)
        hm = QFontMetrics(hf)
        margin = max(TEXT_MARGIN, self.table.style().pixelMetric(QStyle.PM_FocusFrameHMargin, None, self.table) + 1)
        n = self.model.rowCount()
        for i, (cid, title, _, _, _) in enumerate(COLUMNS):
            if cid == 'key':
                continue
            texts = list(MIN_SAMPLES.get(cid, []))
            texts += {self.model.data(self.model.index(r, i)) or '' for r in range(n)}
            w = max([fm.horizontalAdvance(s) for s in texts] + [hm.horizontalAdvance(title)])
            self.header.resizeSection(i, w + 2 * (CELL_PAD + margin) + 2)

    # ------------------------------------------------------------------ public
    def focus_search(self):
        self.search.setFocus(Qt.ShortcutFocusReason)
        self.search.selectAll()

    def clear_filters(self):
        # no blockSignals here: the search box hides its clear button through its own textChanged
        self._batch = True
        try:
            self.search.clear()
            self.field_combo.setCurrentIndex(0)
            for b in (self.btn_snr, self.btn_multi, self.btn_maps):
                b.setChecked(False)
        finally:
            self._batch = False
        self._apply_filters()

    # ------------------------------------------------------------------ state -> list
    def _on_catalog_loading(self, variant):
        self._scanning = True
        self.count_label.setText('Scanning …')
        self.clear_btn.hide()
        self.message.show_message('Scanning the cubes …', variant, busy=True)
        self.stack.setCurrentWidget(self.message)

    def _on_catalog(self):
        self._scanning = False
        self._syncing = True
        try:
            self.model.set_catalog(self.state.catalog)
            self._fill_fields()
            self._fit_columns()
            if not self._sorted_once and self.model.rowCount():
                self._sort(COL['snr'], Qt.DescendingOrder)
                self._sorted_once = True
        finally:
            self._syncing = False
        self._after_filter_change()
        self._sync_to(self.state.key)

    # ------------------------------------------------------------------ sorting
    def _on_header_clicked(self, col):
        if col == self.header.sort_col:
            order = Qt.AscendingOrder if self.header.sort_order == Qt.DescendingOrder else Qt.DescendingOrder
        else:
            order = Qt.DescendingOrder if COLUMNS[col][4] else Qt.AscendingOrder
        self._sort(col, order)

    def _sort(self, col, order):
        """Sort by column `col`; the selected galaxy stays selected and in view."""
        self.header.set_sort(col, order)
        was = self._syncing
        self._syncing = True
        try:
            self.proxy.sort(col, order)
        finally:
            self._syncing = was
        if not was:
            self._sync_to(self.state.key)

    def _on_redshift(self):
        if self.state.key is not None:
            self.model.refresh_key(self.state.catalog, self.state.key)

    def _on_error(self, title, _details):
        if self._scanning and title.startswith('Could not scan'):
            self._scanning = False
            self._after_filter_change()
            if not self.model.rowCount():
                self.message.show_message('No galaxies', 'Scanning the reduction failed - see the message.')

    def _sync_to(self, key):
        """Highlight `key` without selecting it again (it was selected elsewhere)."""
        if key != self._pending_key:
            self._debounce.stop()
        src = self.model.source_row(key) if key else None
        idx = self.proxy.mapFromSource(self.model.index(src, 0)) if src is not None else QModelIndex()
        sel = self.table.selectionModel()
        self._syncing = True
        try:
            if idx.isValid():
                cur = self.table.currentIndex()
                if not (cur.isValid() and cur.row() == idx.row() and sel.isRowSelected(idx.row(), QModelIndex())):
                    sel.setCurrentIndex(idx, sel.ClearAndSelect | sel.Rows)
                self.table.scrollTo(idx, QAbstractItemView.EnsureVisible)
            else:
                sel.clearSelection()
                sel.setCurrentIndex(QModelIndex(), sel.NoUpdate)
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ list -> state
    def _on_current_row(self, cur, _prev):
        if self._syncing or not cur.isValid():
            return
        key = cur.data(KEY_ROLE)
        if QApplication.mouseButtons() != Qt.NoButton:
            self._debounce.stop()
            self._select(key)
        else:                           # keyboard: wait until the key is released / stops repeating
            self._pending_key = key
            self._debounce.start(KEYBOARD_DEBOUNCE_MS)

    def _select(self, key):
        self._pending_key = None
        # the current galaxy again: retry only if its cubes failed to load (no stacks, no load running)
        retry = key == self.state.key and not self.state.stacks and not self.state.is_loading
        if key and (key != self.state.key or retry):
            self.state.select_galaxy(key, force=retry)

    # ------------------------------------------------------------------ filtering
    def _fill_fields(self):
        prev = self.field_combo.currentData()
        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        self.field_combo.addItem('All fields', None)
        for f, n in self.model.fields():
            self.field_combo.addItem(f'{f}  ·  {n}', f)
        i = self.field_combo.findData(prev) if prev else 0
        self.field_combo.setCurrentIndex(max(i, 0))
        self.field_combo.blockSignals(False)
        self.proxy.field = self.field_combo.currentData()

    def _apply_filters(self, *_):
        if self._batch:
            return
        p = self.proxy
        p.tokens = self.search.text().lower().split()
        p.field = self.field_combo.currentData()
        p.need_snr = self.btn_snr.isChecked()
        p.need_multiband = self.btn_multi.isChecked()
        p.need_maps = self.btn_maps.isChecked()
        self._syncing = True            # rows vanishing must not select a neighbour galaxy
        try:
            p.invalidateFilter()
        finally:
            self._syncing = False
        self._after_filter_change()
        self._sync_to(self.state.key)

    def _after_filter_change(self):
        n, m = self.proxy.rowCount(), self.model.rowCount()
        if self._scanning:
            return
        self.count_label.setText(f'{n} of {m} galaxies')
        self.clear_btn.setVisible(self.proxy.active())
        if n:
            self.stack.setCurrentWidget(self.table)
        elif m:
            self.message.show_message('No galaxy matches', 'Try another search or fewer filters.', button=True)
            self.stack.setCurrentWidget(self.message)
        else:
            self.message.show_message('No galaxies', 'This reduction has no combined cubes.')
            self.stack.setCurrentWidget(self.message)

    # ------------------------------------------------------------------ keyboard in the search box
    def eventFilter(self, obj, ev):
        if obj is self.search and ev.type() == ev.KeyPress:
            k = ev.key()
            if k in (Qt.Key_Down, Qt.Key_Up, Qt.Key_PageDown, Qt.Key_PageUp):
                self._step(k)
                return True
            if k in (Qt.Key_Return, Qt.Key_Enter):
                self._step(None)
                return True
            if k == Qt.Key_Escape and self.search.text():
                self.search.clear()
                return True
        return super().eventFilter(obj, ev)

    def _step(self, key):
        """Move the current row from the search box (None = take the first row if none is current)."""
        n = self.proxy.rowCount()
        if not n:
            return
        cur = self.table.currentIndex()
        row = cur.row() if cur.isValid() else -1
        page = max(1, self.table.viewport().height() // max(1, self.table.verticalHeader().defaultSectionSize()) - 1)
        step = {Qt.Key_Down: 1, Qt.Key_Up: -1, Qt.Key_PageDown: page, Qt.Key_PageUp: -page}.get(key, 0)
        if key is None:
            new = row if row >= 0 else 0
        else:
            new = min(max(row + step, 0), n - 1) if row >= 0 else 0
        idx = self.proxy.index(new, 0)
        if key is None:                    # Enter: select immediately
            self.table.selectionModel().setCurrentIndex(idx, self.table.selectionModel().ClearAndSelect
                                                        | self.table.selectionModel().Rows)
            self._debounce.stop()
            self._select(idx.data(KEY_ROLE))
        else:
            self.table.selectionModel().setCurrentIndex(idx, self.table.selectionModel().ClearAndSelect
                                                        | self.table.selectionModel().Rows)
        self.table.scrollTo(idx)
