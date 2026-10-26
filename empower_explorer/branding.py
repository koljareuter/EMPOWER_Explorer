"""EMPOWER logo and the app icon (the VLT 'M', assets/make_icon.py).

    branding.set_windows_app_id()      # before QApplication: own taskbar button + icon on Windows
    app.setWindowIcon(branding.app_icon())
    splash = branding.splash_screen()  # big logo while the scripts load
    label.setPixmap(branding.toolbar_logo(30))
"""
import os
import sys

from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPixmap
from PyQt5.QtWidgets import QApplication, QSplashScreen

from empower_explorer.theme import C

HERE = os.path.dirname(os.path.abspath(__file__))
LOGO = os.path.join(HERE, 'image.png')                       # the EMPOWER banner (1620 x 325)
ICON_ICO = os.path.join(HERE, 'assets', 'empower_icon.ico')
ICON_PNG = os.path.join(HERE, 'assets', 'empower_icon.png')
APP_ID = 'ESO.EMPOWER.Explorer'                              # Windows AppUserModelID (taskbar grouping)
TAGLINE = 'Emission line Mapping of the galaxy POpulation in the cosmic WEb enviRonments'
LETTERS_ROWS = (48, 240)       # rows of the big letters in the banner (without the tagline)


def set_windows_app_id():
    """Give the process its own taskbar identity, so Windows shows the EMPOWER icon instead of
    Python's and groups the window with a pinned 'EMPOWER Explorer' shortcut (same ID)."""
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:
        pass


def app_icon():
    icon = QIcon()
    for path in (ICON_ICO, ICON_PNG):
        if os.path.exists(path):
            icon.addFile(path)
    return icon


def _rounded(pix, radius):
    out = QPixmap(pix.size())
    out.setDevicePixelRatio(pix.devicePixelRatio())
    out.fill(Qt.transparent)
    p = QPainter(out)
    p.setRenderHint(QPainter.Antialiasing)
    path = QPainterPath()
    w, h = pix.width() / pix.devicePixelRatio(), pix.height() / pix.devicePixelRatio()
    path.addRoundedRect(QRectF(0, 0, w, h), radius, radius)
    p.setClipPath(path)
    p.drawPixmap(0, 0, pix)
    p.end()
    return out


def _dpr():
    app = QApplication.instance()
    return app.devicePixelRatio() if app is not None else 1.0


def toolbar_logo(height=30):
    """The big EMPOWER letters (no tagline) as a rounded badge `height` logical px high."""
    src = QPixmap(LOGO)
    if src.isNull():
        return QPixmap()
    r0, r1 = LETTERS_ROWS
    letters = src.copy(0, r0, src.width(), r1 - r0)
    dpr = _dpr()
    pix = letters.scaledToHeight(int(round(height * dpr)), Qt.SmoothTransformation)
    pix.setDevicePixelRatio(dpr)
    return _rounded(pix, 6)


def splash_screen(width=860):
    """Start-up splash: the full banner with the tagline and room for a status line below."""
    src = QPixmap(LOGO)
    if src.isNull():
        return None
    dpr = _dpr()
    logo = src.scaledToWidth(int(round(width * dpr)), Qt.SmoothTransformation)
    lh = logo.height() / dpr
    canvas = QPixmap(int(round(width * dpr)), int(round((lh + 44) * dpr)))
    canvas.setDevicePixelRatio(dpr)
    canvas.fill(QColor(C.BG_DARK))
    p = QPainter(canvas)
    p.setRenderHint(QPainter.Antialiasing)
    logo.setDevicePixelRatio(dpr)
    p.drawPixmap(0, 0, logo)
    p.setPen(QColor(C.ACCENT))
    p.setFont(QFont('Segoe UI', 13, QFont.Bold))
    p.drawText(QRectF(18, lh, width - 36, 44), Qt.AlignLeft | Qt.AlignVCenter, 'Explorer')
    p.end()
    splash = QSplashScreen(_rounded(canvas, 12), Qt.WindowStaysOnTopHint)
    splash.setAttribute(Qt.WA_TranslucentBackground)
    splash.setFont(QFont('Segoe UI', 10))
    return splash


def splash_message(splash, text):
    if splash is not None:
        splash.showMessage(text + '   ', Qt.AlignRight | Qt.AlignBottom, QColor(C.TEXT_DIM))
        QApplication.processEvents()
