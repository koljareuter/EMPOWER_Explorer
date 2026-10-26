"""Header card above the tabs: which galaxy is selected and what is known about it.

    row 1   key (big) · catalogue name · coordinates · SIMBAD / NED / ESO archive links
            right: redshift editor (Enter applies, Reset restores the catalogue value)
    row 2   chips: z with type/quality · each band with its on-source time · S/N Hα · BPT · WHAN ·
            log M* · log SFR · number of cubes · resolved maps · 'loading cubes…'
            right: where the redshift comes from (catalogue / entered in the app / unknown)

The chips stay on one line (the card keeps its height): when the line is too short the chips with
the lowest priority are left out and a '+N' chip lists them in its tooltip.

The redshift editor accepts '.' and ',' as decimal separator (German keyboards); the value goes to
state.set_redshift and is used by every tab.  The links open cone searches (3 arcsec) around the
catalogue RA/Dec in the browser; they are disabled when the coordinates are unknown.
"""
from urllib.parse import quote, urlencode

import numpy as np
from PyQt5.QtCore import QRegularExpression, QSize, Qt, QUrl
from PyQt5.QtGui import QColor, QDesktopServices, QRegularExpressionValidator
from PyQt5.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QSizePolicy, QVBoxLayout, QWidget)

from empower_explorer.data.catalog import band_rank
from empower_explorer.theme import C, band_color

CONE_ARCSEC = 3.0
Z_MAX = 15.0
CHIP_SPACING = 6
# the global [flat="true"] rule outranks QPushButton:disabled, so disabled flat buttons need their own look
FLAT_DISABLED = 'QPushButton:disabled { background-color: rgba(51, 65, 85, 110); color: #64748B; }'


# ---------------------------------------------------------------------------------- helpers
def parse_redshift(text):
    """'0.8456', '0,8456', ' 1,5 ' -> float; None if empty, not a number or outside 0 < z < Z_MAX."""
    s = (text or '').strip().replace(' ', '').replace(',', '.')
    if not s:
        return None
    try:
        z = float(s)
    except ValueError:
        return None
    return z if np.isfinite(z) and 0 < z < Z_MAX else None


def format_redshift(z):
    """Editor text: up to 5 decimals, trailing zeros dropped (1.476, 0.62633)."""
    return f'{z:.5f}'.rstrip('0').rstrip('.') if z is not None else ''


def _finite(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _text(row, col):
    if row is None:
        return ''
    v = row.get(col, '')
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ''
    s = str(v).strip()
    return '' if s.lower() in ('nan', 'none', '—', '-') else s


def sexagesimal(ra, dec):
    """'10:00:38.54  +02:06:03.6' (integer arithmetic, so 59.999 s never shows as 60.00)."""
    cs = int(round((ra % 360.0) / 15.0 * 3600 * 100))
    ra_s = f'{cs // 360000 % 24:02d}:{cs // 6000 % 60:02d}:{cs % 6000 / 100:05.2f}'
    ds = int(round(abs(dec) * 3600 * 10))
    dec_s = f'{"-" if dec < 0 else "+"}{ds // 36000:02d}:{ds // 600 % 60:02d}:{ds % 600 / 10:04.1f}'
    return ra_s, dec_s


def _url(base, params):
    return QUrl.fromEncoded((base + '?' + urlencode(params, quote_via=quote, safe=',')).encode('ascii'))


def simbad_url(ra, dec, radius_arcsec=CONE_ARCSEC):
    return _url('https://simbad.cds.unistra.fr/simbad/sim-coo', [
        ('Coord', f'{ra:.6f} {dec:+.6f}'), ('CooFrame', 'FK5'), ('CooEpoch', '2000'), ('CooEqui', '2000'),
        ('Radius', f'{radius_arcsec:g}'), ('Radius.unit', 'arcsec'), ('submit', 'submit query')])


def ned_url(ra, dec, radius_arcsec=CONE_ARCSEC):
    return _url('https://ned.ipac.caltech.edu/conesearch', [
        ('search_type', 'Near Position Search'), ('in_csys', 'Equatorial'), ('in_equinox', 'J2000'),
        ('ra', f'{ra:.6f}'), ('dec', f'{dec:+.6f}'), ('radius', f'{radius_arcsec / 60:.4f}'),   # arcmin
        ('Z_CONSTRAINT', 'Unconstrained'), ('out_csys', 'Equatorial'), ('out_equinox', 'J2000'),
        ('obj_sort', 'Distance to search center')])


def eso_url(ra, dec, radius_arcsec=CONE_ARCSEC):
    return _url('https://archive.eso.org/scienceportal/home', [
        ('pos', f'{ra:.6f},{dec:+.6f}'), ('r', f'{radius_arcsec / 3600:.6f}')])     # degrees


def _rgba(color, alpha):
    c = QColor(color)
    return f'rgba({c.red()}, {c.green()}, {c.blue()}, {alpha})'


def class_color(cls):
    c = (cls or '').lower()
    if c.startswith('sf'):
        return C.EMERALD
    if c.startswith('comp') or c.startswith('wagn'):
        return C.AMBER
    if 'agn' in c or c.startswith('sey') or c.startswith('liner'):
        return C.PINK
    return C.TEXT_DIM


def fmt_hours(h):
    if h is None:
        return '—'
    return f'{h:.1f} h' if h >= 1 else f'{h * 60:.0f} min'


def make_chip(value, label='', color=None, tip=''):
    """Rounded pill: dim label + value; tinted with `color` if given (else neutral grey)."""
    text = (f'<span style="color:{C.TEXT_DIM}">{label}</span>&nbsp;{value}' if label else value)
    lab = QLabel(text)
    lab.setTextFormat(Qt.RichText)
    if color:
        bg, fg, border = _rgba(color, 34), color, _rgba(color, 90)
    else:
        bg, fg, border = C.BG_LIGHT, C.TEXT, C.BG_LIGHT
    lab.setStyleSheet(f'QLabel {{ background-color: {bg}; color: {fg}; border: 1px solid {border};'
                      f' border-radius: 10px; padding: 2px 9px; font-size: 12px; }}')
    if tip:
        lab.setToolTip(tip)
    return lab


def chip(value, label='', color=None, tip='', prio=5):
    """(QLabel, priority, plain text) for ChipRow; priority 0 = always shown, higher = dropped first."""
    return make_chip(value, label, color, tip), prio, (f'{label} {value}' if label else value)


class ChipRow(QWidget):
    """Chips on one line in their given order.  If the line is too short, the lowest-priority chips
    are left out (priority 0 never) and a '+N' chip names them in its tooltip.  Positions come from
    the chips' own size hints whenever the row is resized - nothing wraps, nothing overlaps."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)   # never widens the window
        self._chips = []                      # [(QLabel, priority, plain text)]
        self.more = make_chip('+0')
        self.more.setParent(self)
        self.more.hide()
        self.hidden_texts = []

    def set_chips(self, chips):
        for lab, _, _ in self._chips:
            lab.hide()                        # at once: deleteLater only runs in the event loop
            lab.deleteLater()
        self._chips = list(chips)
        for lab, _, _ in self._chips:
            lab.setParent(self)
        self._relayout()
        self.updateGeometry()

    def visible_texts(self):
        return [t for lab, _, t in self._chips if not lab.isHidden()]

    def _height(self):
        return max([lab.sizeHint().height() for lab, _, _ in self._chips] + [self.more.sizeHint().height()])

    def sizeHint(self):
        w = sum(lab.sizeHint().width() for lab, _, _ in self._chips) + CHIP_SPACING * max(0, len(self._chips) - 1)
        return QSize(w, self._height())

    def minimumSizeHint(self):
        return QSize(0, self._height())

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._relayout()

    def _relayout(self):
        n = len(self._chips)
        widths = [lab.sizeHint().width() for lab, _, _ in self._chips]
        avail = self.width()
        if sum(widths) + CHIP_SPACING * max(0, n - 1) <= avail:
            shown = set(range(n))
        else:
            self.more.setText(f'+{n}')                      # widest possible '+N' text
            budget = avail - self.more.sizeHint().width() - CHIP_SPACING
            shown, used = set(), 0
            for i in sorted(range(n), key=lambda i: (self._chips[i][1], i)):
                need = widths[i] + (CHIP_SPACING if shown else 0)
                if self._chips[i][1] <= 0 or used + need <= budget:
                    shown.add(i)
                    used += need
        h = self.height()
        x = 0
        for i, (lab, _, _) in enumerate(self._chips):
            if i in shown:
                lh = lab.sizeHint().height()
                lab.setGeometry(x, max(0, (h - lh) // 2), widths[i], lh)
                lab.show()
                x += widths[i] + CHIP_SPACING
            else:
                lab.hide()
        self.hidden_texts = [t for i, (_, _, t) in enumerate(self._chips) if i not in shown]
        if self.hidden_texts:
            self.more.setText(f'+{len(self.hidden_texts)}')
            self.more.setToolTip('Not shown for lack of space (make the window wider):\n'
                                 + '\n'.join(self.hidden_texts))
            mh = self.more.sizeHint().height()
            self.more.setGeometry(x, max(0, (h - mh) // 2), self.more.sizeHint().width(), mh)
            self.more.show()
            self.more.raise_()
        else:
            self.more.hide()


class ElidedLabel(QLabel):
    """Plain-text label that shortens its text with '…' instead of forcing its full width."""

    def __init__(self, mode=Qt.ElideRight, min_width=40, parent=None):
        super().__init__(parent)
        self._full = ''
        self._tip = ''
        self._mode = mode
        self._min = min_width
        self.setTextFormat(Qt.PlainText)

    def setText(self, text):
        self._full = text or ''
        self._elide()
        self.updateGeometry()

    def text(self):
        return self._full

    def sizeHint(self):
        self.ensurePolished()
        fm = self.fontMetrics()
        m = self.contentsMargins()
        return QSize(fm.horizontalAdvance(self._full) + m.left() + m.right() + 2, fm.height() + m.top() + m.bottom())

    def minimumSizeHint(self):
        sh = self.sizeHint()
        return QSize(min(sh.width(), self._min), sh.height())

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._elide()

    def _elide(self):
        self.ensurePolished()
        shown = self.fontMetrics().elidedText(self._full, self._mode, max(0, self.contentsRect().width()))
        if shown != super().text():
            super().setText(shown)
        if shown != self._full:
            super().setToolTip(self._full + (f'\n\n{self._tip}' if self._tip else ''))
        else:
            super().setToolTip(self._tip)

    def setToolTip(self, tip):
        """Shown on hover; the full text is added when it is elided."""
        self._tip = tip or ''
        self._elide()


# ---------------------------------------------------------------------------------- the card
class GalaxyCard(QFrame):
    """Compact header: identity, key numbers as chips, redshift editor and catalogue links."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._failed_key = None
        self._radec = None
        self.setProperty('role', 'card')
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._build_ui()
        state.catalogLoading.connect(self._on_catalog_loading)
        state.catalogChanged.connect(self.refresh)
        state.galaxyChanged.connect(self._on_galaxy)
        state.stacksReady.connect(self._on_stacks)
        state.redshiftChanged.connect(self.refresh)
        state.error.connect(self._on_error)
        self.refresh()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 10, 16, 10)
        lay.setSpacing(18)

        left = QVBoxLayout()
        left.setSpacing(6)
        top = QHBoxLayout()
        top.setSpacing(12)
        self.title = QLabel('')
        self.title.setProperty('role', 'title')
        self.title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.name = ElidedLabel(Qt.ElideRight, 60)          # catalogue name: shortened first
        self.name.setStyleSheet(f'color: {C.TEXT_DIM}; font-size: 13px;')
        self.coords = QLabel('')
        self.coords.setStyleSheet(f'color: {C.TEXT_DIM}; font-size: 12px; font-family: Consolas, monospace;')
        self.coords.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(self.title, 0, Qt.AlignVCenter)
        top.addWidget(self.name, 0, Qt.AlignVCenter)
        top.addWidget(self.coords, 0, Qt.AlignVCenter)
        self.links = {}
        for name, tip in (('SIMBAD', 'SIMBAD'), ('NED', 'NED'), ('ESO archive', 'the ESO Science Archive')):
            b = QPushButton(f'{name} ↗')
            b.setProperty('flat', 'true')
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet('QPushButton { padding: 3px 9px; font-size: 12px; } ' + FLAT_DISABLED)
            b.setToolTip(f'Open a {CONE_ARCSEC:g}″ cone search in {tip} around the galaxy (browser)')
            b.clicked.connect(lambda _=False, n=name: self._open_link(n))
            self.links[name] = b
            top.addWidget(b, 0, Qt.AlignVCenter)
        top.addStretch(1)
        left.addLayout(top)

        self.chips = ChipRow()
        left.addWidget(self.chips)
        lay.addLayout(left, 1)

        lay.addLayout(self._build_redshift())

    def _build_redshift(self):
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        lab = QLabel('Redshift')
        lab.setProperty('role', 'dim')
        self.z_edit = QLineEdit()
        self.z_edit.setFixedWidth(110)
        self.z_edit.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.z_edit.setPlaceholderText('e.g. 0,8456')
        self.z_edit.setValidator(QRegularExpressionValidator(QRegularExpression(r'\s*\d{0,2}([.,]\d{0,8})?\s*')))
        self.z_edit.setToolTip('Redshift used by all tabs (line windows, fits, maps).\n'
                               'Type it with "." or "," (0.8456 or 0,8456) and press Enter; Esc reverts.')
        self.z_edit.returnPressed.connect(self._apply_redshift)
        self.z_edit.textEdited.connect(self._on_z_edited)
        self.z_edit.installEventFilter(self)
        self.reset_btn = QPushButton('Reset')
        self.reset_btn.setProperty('flat', 'true')
        self.reset_btn.setStyleSheet(FLAT_DISABLED)
        self.reset_btn.setCursor(Qt.PointingHandCursor)
        self.reset_btn.setToolTip('Go back to the catalogue redshift (target_redshifts.csv)')
        self.reset_btn.clicked.connect(self._reset_redshift)
        self.source = ElidedLabel(Qt.ElideMiddle, 0)       # as wide as the editor row, never wider
        self.source.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.source.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        grid.addWidget(lab, 0, 0, Qt.AlignRight)
        grid.addWidget(self.z_edit, 0, 1)
        grid.addWidget(self.reset_btn, 0, 2)
        grid.addWidget(self.source, 1, 0, 1, 3)
        return grid

    # ------------------------------------------------------------------ state -> card
    def _on_catalog_loading(self, variant):
        self._set_identity('Scanning …', f'reading the headers of {variant}', None)
        self.chips.set_chips([chip('scanning cubes…', color=C.ACCENT, prio=0)])
        self._set_redshift_enabled(False)
        self._set_source('', C.TEXT_DIM)

    def _on_galaxy(self, _key):
        self._failed_key = None
        self.refresh()

    def _on_stacks(self, key):
        row = self._row()
        if row is not None and key == self.state.key:      # only the 'loading' chip changes
            self.chips.set_chips(self._chips_for(key, row))

    def _on_error(self, title, _details):
        if self.state.key and not self.state.stacks and title.startswith('Could not load'):
            self._failed_key = self.state.key
            self.refresh()

    def _row(self):
        """Catalogue row of the current galaxy, None if none (or it is not in a just-loaded catalogue)."""
        cat, key = self.state.catalog, self.state.key
        if cat is None or key is None or key not in cat.galaxies.index:
            return None
        return self.state.galaxy_row()

    def refresh(self):
        row = self._row()
        if row is None:
            self._show_empty()
            return
        key = self.state.key
        ra, dec = _finite(row.get('ra')), _finite(row.get('dec'))
        self._set_identity(key, _text(row, 'name'), (ra, dec) if ra is not None and dec is not None else None)
        self.chips.set_chips(self._chips_for(key, row))
        self._set_redshift_enabled(True)
        self._show_redshift()

    def _show_empty(self):
        self._set_identity('No galaxy selected',
                           'choose a galaxy in the list' if self.state.catalog is not None else 'no data loaded yet',
                           None)
        self.chips.set_chips([])
        self._set_redshift_enabled(False)
        self._set_source('', C.TEXT_DIM)

    def _set_identity(self, title, name, radec):
        self.title.setText(title)
        self.name.setText(name)
        self.name.setVisible(bool(name))
        if radec is not None:
            ra, dec = radec
            ra_s, dec_s = sexagesimal(ra, dec)
            self.coords.setText(f'{ra_s}  {dec_s}')
            self.coords.setToolTip(f'RA, Dec (J2000)\n{ra:.6f}°  {dec:+.6f}°')
        else:
            self.coords.setText('no coordinates' if self.state.key and title == self.state.key else '')
            self.coords.setToolTip('RA/Dec unknown (not in target_redshifts.csv nor in the property table)')
        self._radec = radec
        for b in self.links.values():
            b.setEnabled(radec is not None)

    # ------------------------------------------------------------------ chips
    # priority: 0 always shown · 1 bands · 2 S/N Hα · 3 BPT · 4 maps · 5 WHAN · 6 log M* · 7 log SFR · 8 cubes
    def _chips_for(self, key, row):
        chips = [self._z_chip()]
        cat = self.state.catalog
        cubes = cat.cubes.get(key, {})
        for b in sorted(cubes, key=band_rank):
            h = _finite(row.get(f'texp_{b}_h'))
            n = len(cubes[b])
            folders = sorted({i.get('folder', '') for i in cubes[b]} - {''})
            tip = (f'{b} band: {fmt_hours(h)} on source in {n} combined cube{"s" if n != 1 else ""}'
                   + (f'\n{", ".join(folders)}' if folders else ''))
            chips.append(chip(fmt_hours(h), b, band_color(b), tip, prio=1))
        snr = _finite(row.get('snr_ha'))
        if snr is not None:
            col = C.EMERALD if snr >= 5 else (C.AMBER if snr >= 3 else C.TEXT_DIM)
            band = _text(row, 'band_ha')
            chips.append(chip(f'{snr:.1f}', 'S/N Hα', col,
                              'Integrated Hα S/N of the notebooks (line_snr)'
                              + (f', measured in {band}' if band else ''), prio=2))
        bpt = _text(row, 'class_NII_BPT')
        if bpt:
            s2 = _text(row, 'class_SII_BPT')
            chips.append(chip(bpt, 'BPT', class_color(bpt),
                              '[NII]-BPT class (SF, comp = composite, AGN; "?" = from a limit)'
                              + (f'\n[SII]-BPT class: {s2}' if s2 else ''), prio=3))
        whan = _text(row, 'whan')
        if whan:
            chips.append(chip(whan, 'WHAN', class_color(whan),
                              'WHAN class (Cid Fernandes+11) from log [NII]/Hα and EW(Hα)', prio=5))
        logm, logsfr = _finite(row.get('logM')), _finite(row.get('logSFR'))
        if logm is not None:
            chips.append(chip(f'{logm:.2f}', 'log M*', tip='Stellar mass, log M* [M☉] (LePhare, COSMOS2020)', prio=6))
        if logsfr is not None:
            chips.append(chip(f'{logsfr:.2f}', 'log SFR', tip='Star-formation rate, log SFR [M☉/yr]', prio=7))
        n_cubes = sum(len(v) for v in cubes.values())
        if n_cubes:
            chips.append(chip(f'{n_cubes} cube{"s" if n_cubes != 1 else ""}',
                              tip='Combined cubes (DATACUBE_COMBINED) of this galaxy in the loaded reduction', prio=8))
        prods = cat.products.get(key, [])
        if prods:
            kinds = ', '.join(sorted({p.kind for p in prods}))
            chips.append(chip('resolved maps', color=C.PURPLE,
                              tip=f'Map products of the notebooks: {kinds}\nshown in the Maps tab', prio=4))
        if self._failed_key == key:
            chips.append(chip('cubes could not be loaded', color=C.PINK, tip='See the error message', prio=0))
        elif not self.state.stacks:
            chips.append(chip('loading cubes…', color=C.ACCENT, tip='The band stacks are being read', prio=0))
        return chips

    def _z_chip(self):
        r = self.state.redshift
        if r is None or not r.known:
            return chip('unknown', 'z', C.PINK, 'No redshift in target_redshifts.csv - type one in on the right\n'
                                                '(line windows, fits and maps need it)', prio=0)
        if r.override:
            return chip(f'{r.z:.4f} · entered', 'z', C.ACCENT, 'Redshift typed in the app (session only)', prio=0)
        bits = [f'{r.z:.4f}']
        if r.z_type:
            bits.append(r.z_type)
        q = _finite(r.quality)
        if q is not None:
            bits.append(f'Q{q:.0f}')
        return chip(' · '.join(bits), 'z', tip=self._provenance(r), prio=0)

    @staticmethod
    def _provenance(r):
        lines = [f'z = {r.z:.5f} ({r.z_type or "type unknown"})']
        q = _finite(r.quality)
        if q is not None:
            lines.append(f'quality flag {q:.0f} (of the source catalogue)')
        if r.source:
            lines.append(f'source: {r.source}')
        if r.target_list:
            lines.append(f'target list: {r.target_list}')
        pz = _finite(r.public_z)
        if pz is not None:
            lines.append(f'public z = {pz:.5f} ({r.public_type or "?"}, {r.public_source or "?"})')
        sep = _finite(r.separation)
        if sep is not None:
            lines.append(f'separation KMOS target - catalogue source: {sep:.2f}″')
        return '\n'.join(lines)

    # ------------------------------------------------------------------ redshift editor
    def _show_redshift(self):
        r = self.state.redshift
        known = r is not None and r.known
        self.z_edit.setText(format_redshift(r.z) if known else '')
        self.reset_btn.setEnabled(bool(r is not None and r.override))
        if not known:
            self._edit_style(_rgba(C.PINK, 150))             # invite an entry
            self._set_source('no redshift - type one in, then Enter', C.PINK,
                             'Line windows, fits and maps need a redshift')
        elif r.override:
            self._edit_style(None)
            cat = self._catalogue_z()
            self._set_source('entered in the app', C.ACCENT,
                             'Typed in this session; Reset returns to the catalogue value'
                             + (f' (z = {cat:.5f})' if cat is not None else ' (none)'))
        else:
            self._edit_style(None)
            self._set_source(f'catalogue · {r.source}' if r.source else 'catalogue (target_redshifts.csv)',
                             C.TEXT_DIM, self._provenance(r))

    def _catalogue_z(self):
        cat, key = self.state.catalog, self.state.key
        if cat is None or key is None:
            return None
        saved = cat.overrides.pop(key, None)       # read the catalogue value behind the override
        try:
            r = cat.redshift(key)
        finally:
            if saved is not None:
                cat.overrides[key] = saved
        return r.z if r.known else None

    def _set_source(self, text, color, tip=''):
        self.source.setStyleSheet(f'color: {color}; font-size: 12px;')
        self.source.setText(text)
        self.source.setToolTip(tip)

    def _set_redshift_enabled(self, on):
        self.z_edit.setEnabled(on)
        if not on:
            self.z_edit.clear()
            self.reset_btn.setEnabled(False)

    def _edit_style(self, color):
        self.z_edit.setStyleSheet(f'QLineEdit {{ border: 2px solid {color}; }}' if color else '')

    def _on_z_edited(self, text):
        z = parse_redshift(text)
        if not text.strip():
            self._edit_style(None)
            self._set_source('Enter applies · Reset = catalogue value', C.TEXT_DIM)
        elif z is None:
            self._edit_style(C.PINK)
            self._set_source(f'not a redshift (0 < z < {Z_MAX:g})', C.PINK)
        else:
            self._edit_style(C.AMBER)
            self._set_source('press Enter to apply', C.AMBER)

    def _apply_redshift(self):
        if self.state.key is None:
            return
        z = parse_redshift(self.z_edit.text())
        if z is None:
            self._edit_style(C.PINK)
            self._set_source(f'not a redshift (0 < z < {Z_MAX:g})', C.PINK)
            return
        current = self.state.z
        if current is not None and abs(current - z) < 5e-7:
            self._show_redshift()              # nothing changed: just clear the 'pending' look
            return
        self.state.set_redshift(z)
        self.state.status.emit(f'{self.state.key}: z = {z:.5f} (entered in the app)', 5000)

    def _reset_redshift(self):
        if self.state.key is not None:
            self.state.set_redshift(None)
            self.state.status.emit(f'{self.state.key}: catalogue redshift restored', 4000)

    def eventFilter(self, obj, ev):
        if obj is self.z_edit and ev.type() == ev.KeyPress and ev.key() == Qt.Key_Escape:
            self._show_redshift()
            return True
        return super().eventFilter(obj, ev)

    # ------------------------------------------------------------------ links
    def _open_link(self, name):
        if self._radec is None:
            return
        ra, dec = self._radec
        url = {'SIMBAD': simbad_url, 'NED': ned_url, 'ESO archive': eso_url}[name](ra, dec)
        if not QDesktopServices.openUrl(url):
            self.state.status.emit(f'Could not open the browser for {name}', 5000)
