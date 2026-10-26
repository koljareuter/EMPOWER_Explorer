"""MainWindow and main().

Layout:  toolbar (reduction, settings) | left: galaxy list | right: galaxy header card + tabs
         (Spectrum, Maps, Population, Data QA) | status bar with progress.
Each tab is a QWidget class taking the shared AppState; a tab that fails to import is replaced by
a placeholder showing the error, so one broken view never takes the whole app down.
"""
import importlib
import os
import sys
import traceback

from PyQt5.QtCore import Qt, QThreadPool, QTimer
from PyQt5.QtGui import QKeySequence, QPixmap
from PyQt5.QtWidgets import (QApplication, QComboBox, QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel,
                             QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                             QScrollArea, QShortcut, QSizePolicy, QSplitter, QTabWidget, QToolBar,
                             QVBoxLayout, QWidget)

from empower_explorer import APP_NAME, ORG_NAME, __version__, branding, config, engine, theme
from empower_explorer.state import AppState
from empower_explorer.theme import C

TABS = [   # (module, class, title, tooltip)
    ('empower_explorer.views.spectrum_tab', 'SpectrumTab', 'Spectrum',
     'Image, aperture, line panels (Hβ, [OIII], Hα, [SII]) and the full spectrum of all bands'),
    ('empower_explorer.views.maps_tab', 'MapsTab', 'Maps',
     'Resolved maps: notebook products and per-spaxel fits; click a spaxel for its spectrum'),
    ('empower_explorer.views.population_tab', 'PopulationTab', 'Population',
     'Galaxy properties of the whole sample: main sequence, BPT, WHAN, ...'),
    ('empower_explorer.views.qa_tab', 'QATab', 'Data QA',
     'Cubes, exposure times, reductions, redshift provenance and FITS headers'),
]


def _load_class(module, cls):
    mod = importlib.import_module(module)
    return getattr(mod, cls)


def _scroll(widget, vertical=True):
    """Frameless scroll area around a view: on small screens (laptops) the window can be narrower
    than a view's natural minimum width; scroll bars appear only then."""
    sa = QScrollArea()
    sa.setWidget(widget)
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.NoFrame)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    sa.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded if vertical else Qt.ScrollBarAlwaysOff)
    sa.setMinimumWidth(200)
    if not vertical:
        sa.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sa.setFixedHeight(widget.sizeHint().height() + 12)
    sa.view = widget
    return sa


class _Tabs(QTabWidget):
    """QTabWidget whose widget(i) returns the view itself, not the scroll area around it."""

    def widget(self, i):
        w = super().widget(i)
        return getattr(w, 'view', w)

    def currentWidget(self):
        w = super().currentWidget()
        return getattr(w, 'view', w)


class _Broken(QWidget):
    def __init__(self, title, err, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lab = QLabel(f'The {title} view could not be loaded:')
        lab.setProperty('role', 'warn')
        txt = QPlainTextEdit(err)
        txt.setReadOnly(True)
        lay.addWidget(lab)
        lay.addWidget(txt)


class MainWindow(QMainWindow):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.setWindowTitle(f'{APP_NAME} {__version__}')
        self.resize(1680, 1000)

        # toolbar
        tb = QToolBar('Main')
        tb.setMovable(False)
        self.addToolBar(tb)
        logo = branding.toolbar_logo(30)
        if logo.isNull():
            title = QLabel(f'  {APP_NAME}  ')
            title.setStyleSheet(f'color: {C.ACCENT}; font-size: 18px; font-weight: bold;')
            tb.addWidget(title)
        else:
            badge = QLabel()
            badge.setPixmap(logo)
            badge.setToolTip(branding.TAGLINE)
            badge.setCursor(Qt.PointingHandCursor)
            badge.mousePressEvent = lambda _e: self.about()
            tb.addWidget(badge)
            title = QLabel(' Explorer  ')
            title.setStyleSheet(f'color: {C.ACCENT}; font-size: 17px; font-weight: bold;')
            tb.addWidget(title)
        tb.addSeparator()
        lab = QLabel(' Reduction ')
        lab.setProperty('role', 'dim')
        tb.addWidget(lab)
        self.variant_combo = QComboBox()
        self.variant_combo.setToolTip('Which combined cubes to show (folder in the EMPOWER directory)')
        self.variant_combo.setMinimumWidth(230)
        self.variant_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.variant_combo.currentIndexChanged.connect(self._variant_selected)
        tb.addWidget(self.variant_combo)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)
        self.root_label = QLabel('')
        self.root_label.setProperty('role', 'dim')
        tb.addWidget(self.root_label)
        act = tb.addAction('Data folder…')
        act.setToolTip('Choose the EMPOWER_Analysis_and_presentation folder')
        act.triggered.connect(self.choose_root)
        act = tb.addAction('About')
        act.triggered.connect(self.about)

        # central: galaxy list | (card + tabs)
        split = QSplitter(Qt.Horizontal)
        split.setHandleWidth(6)
        self.galaxy_list = self._make('empower_explorer.views.galaxy_list', 'GalaxyList', 'galaxy list')
        split.addWidget(self.galaxy_list)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 8, 8, 4)
        rl.setSpacing(8)
        self.card = self._make('empower_explorer.views.galaxy_card', 'GalaxyCard', 'galaxy card')
        rl.addWidget(_scroll(self.card, vertical=False))
        self.tabs = _Tabs()
        self.tabs.setDocumentMode(True)
        for module, cls, ttl, tip in TABS:
            w = self._make(module, cls, ttl)
            i = self.tabs.addTab(_scroll(w), ttl)
            self.tabs.setTabToolTip(i, tip)
        rl.addWidget(self.tabs, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([360, 1320])
        self.setCentralWidget(split)

        # status bar
        sb = self.statusBar()
        self.busy_label = QLabel('')
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setMaximumHeight(10)
        self.progress.hide()
        sb.addPermanentWidget(self.busy_label)
        sb.addPermanentWidget(self.progress)

        state.status.connect(lambda t, ms: sb.showMessage(t, ms))
        state.busy.connect(self._busy)
        state.error.connect(self._error)
        state.galaxyChanged.connect(lambda k: self.setWindowTitle(f'{k} — {APP_NAME}'))
        state.catalogChanged.connect(self._variant_loaded)
        state.catalogFailed.connect(lambda _v: self._select_variant_in_combo(state.variant))

        for i in range(len(TABS)):
            QShortcut(QKeySequence(f'Ctrl+{i + 1}'), self, activated=lambda i=i: self.tabs.setCurrentIndex(i))
        QShortcut(QKeySequence('Ctrl+F'), self, activated=self._focus_search)

    def _make(self, module, cls, title):
        try:
            return _load_class(module, cls)(self.state)
        except Exception:
            return _Broken(title, traceback.format_exc())

    def _focus_search(self):
        f = getattr(self.galaxy_list, 'focus_search', None)
        if callable(f):
            f()

    # ------------------------------------------------------------------ data folder / reduction
    def start(self):
        root = config.empower_root()
        if root is None:
            self.close_splash()
            QMessageBox.information(self, APP_NAME, 'Please choose the EMPOWER_Analysis_and_presentation folder\n'
                                                    '(it contains scripts/kmos_sample.py and the cube folders).')
            if not self.choose_root():
                return
        else:
            self.use_root(root)

    def choose_root(self):
        d = QFileDialog.getExistingDirectory(self, 'EMPOWER_Analysis_and_presentation folder',
                                             self.state.root or os.path.expanduser('~'))
        if not d:
            return False
        if not config.is_empower_root(d):
            QMessageBox.warning(self, APP_NAME, 'This folder has no scripts/kmos_sample.py and scripts/line_snr.py.')
            return False
        if not self.use_root(d):
            return False
        config.set_empower_root(d)          # remembered only if it actually works
        return True

    def use_root(self, root):
        """Import the folder's EMPOWER scripts and scan its default reduction; False on failure."""
        self.statusBar().showMessage('Loading the EMPOWER scripts (line_snr.py, kmos_sample.py) …')
        self._busy('Starting …', 0, 1)
        QApplication.processEvents()        # paint the window before the ~1 s import
        try:
            self.state.set_root(root)
        except engine.EngineError as e:
            self._busy('', 0, 0)
            self._error('EMPOWER scripts', str(e))
            return False
        self.root_label.setText(f'{root}  ')
        self.root_label.setToolTip(root)
        variants = self.state.variants()
        self.variant_combo.blockSignals(True)
        self.variant_combo.clear()
        for name, label in variants:
            self.variant_combo.addItem(label, name)
            self.variant_combo.setItemData(self.variant_combo.count() - 1, f'folder: {name}', Qt.ToolTipRole)
        stored = config.settings().value('variant', config.DEFAULT_VARIANT, type=str)
        names = [n for n, _ in variants]
        idx = names.index(stored) if stored in names else (names.index(config.DEFAULT_VARIANT)
                                                             if config.DEFAULT_VARIANT in names else 0)
        self.variant_combo.setCurrentIndex(idx)
        self.variant_combo.blockSignals(False)
        if variants:
            self.state.load_variant(names[idx])
        else:
            self._busy('', 0, 0)
            self._error('No combined cubes', f'No kmos_comb* folders with DATACUBE_COMBINED files in\n{root}')
        return True

    def _variant_selected(self, i):
        name = self.variant_combo.itemData(i)
        if name and name != self.state.variant and name != self.state.loading_variant:
            self.state.load_variant(name)

    def _variant_loaded(self):
        """Remember the reduction once it really loaded; keep the combo in sync."""
        if self.state.variant:
            config.settings().setValue('variant', self.state.variant)
        self._select_variant_in_combo(self.state.variant)

    def _select_variant_in_combo(self, name):
        i = self.variant_combo.findData(name)
        if i >= 0 and i != self.variant_combo.currentIndex():
            self.variant_combo.blockSignals(True)
            self.variant_combo.setCurrentIndex(i)
            self.variant_combo.blockSignals(False)

    # ------------------------------------------------------------------ feedback
    def _busy(self, text, done, total):
        if total <= 0:
            self.progress.hide()
            self.busy_label.setText('')
            QApplication.restoreOverrideCursor() if QApplication.overrideCursor() else None
            return
        self.busy_label.setText(text)
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.progress.show()

    def close_splash(self):
        """Replaced by main() when a start-up splash is shown (it must not cover dialogs)."""

    def _error(self, title, details):
        self.close_splash()
        self.statusBar().showMessage(f'⚠ {title}', 10000)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle(title)
        first = details.strip().split('\n\n')[0]
        box.setText(first[:600])
        if len(details) > len(first):
            box.setDetailedText(details)
        box.exec_()

    def about(self):
        self.close_splash()
        box = QDialog(self)
        box.setWindowTitle(APP_NAME)

        layout = QVBoxLayout(box)

        image = QPixmap(os.path.join(os.path.dirname(__file__), 'image.png'))
        if not image.isNull():
            image_label = QLabel()
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setPixmap(image.scaled(
                800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            layout.addWidget(image_label)

        text_label = QLabel(
            f'<b>{APP_NAME} {__version__}</b><br><br>'
            'Explorer for the EMPOWER KMOS combined cubes.<br>'
            'Line S/N, stacking and redshifts use the notebooks\' own code '
            '(scripts/line_snr.py, scripts/kmos_sample.py).<br><br>'
            'Studentship @ ESO · Kolja Reuter<br>'
            "EMAIL: <a href='mailto:kolja@reuter-post.de'>kolja@reuter-post.de</a>"
        )
        text_label.setOpenExternalLinks(True)
        text_label.setWordWrap(True)
        layout.addWidget(text_label)

        close_button = QPushButton('Close')
        close_button.clicked.connect(box.accept)
        layout.addWidget(close_button)

        box.exec_()

    def closeEvent(self, ev):
        config.settings().setValue('geometry', self.saveGeometry())
        pool = QThreadPool.globalInstance()
        pool.clear()                 # drop queued jobs, let running ones finish briefly
        pool.waitForDone(3000)
        super().closeEvent(ev)


def main():
    branding.set_windows_app_id()         # own taskbar button with the EMPOWER icon
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setOrganizationName(ORG_NAME)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(branding.app_icon())
    theme.setup_pyqtgraph()
    theme.apply_palette(app)

    splash = branding.splash_screen()     # the logo while the window is built and the data scanned
    if splash is not None:
        splash.show()
        branding.splash_message(splash, 'Building the window …')

    state = AppState()

    def close_splash(*_):
        if splash is not None and splash.isVisible():
            splash.close()
    if splash is not None:            # connected before the window, so it runs before any error box
        state.busy.connect(lambda text, d, t: branding.splash_message(splash, text)
                           if t > 0 and splash.isVisible() else None)
        state.catalogChanged.connect(close_splash)
        state.catalogFailed.connect(close_splash)
        state.error.connect(close_splash)
        splash.mousePressEvent = lambda _e: close_splash()
        QTimer.singleShot(20000, close_splash)

    win = MainWindow(state)
    win.close_splash = close_splash
    geo = config.settings().value('geometry')
    if geo is not None:
        win.restoreGeometry(geo)
    win.show()
    if splash is not None:
        splash.raise_()
    QTimer.singleShot(50, win.start)      # paint the window first, then import and scan
    code = app.exec_()
    sys.exit(code)
