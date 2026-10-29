"""EMPOWER logo and the app icon (the VLT 'M', assets/make_icon.py).

    branding.set_windows_app_id()      # before QApplication: own taskbar button + icon on Windows
    app.setWindowIcon(branding.app_icon())
    title_bars = branding.TitleBars(app)   # every window's Windows title bar in the app's blue
    splash = branding.splash_screen()  # big logo while the app starts
    label.setPixmap(branding.toolbar_logo(30))
"""
import os
import sys

from PyQt5.QtCore import QEvent, QObject, QRectF, Qt
from PyQt5.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPixmap
from PyQt5.QtWidgets import QApplication, QSplashScreen, QWidget

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


def _colorref(hexcolor):
    """'#RRGGBB' -> Windows COLORREF (0x00BBGGRR)."""
    r, g, b = (int(hexcolor[i:i + 2], 16) for i in (1, 3, 5))
    return r | (g << 8) | (b << 16)


def color_title_bar(widget, caption=C.BG_MEDIUM, text=C.TEXT):
    """The Windows title bar of a top-level widget in the app's colours (the toolbar's blue) instead of
    the system's light grey: DWM caption, text and border colour on Windows 11, the dark title bar on
    Windows 10.  Returns the DWM results {attribute: HRESULT}; no-op elsewhere or on failure."""
    if sys.platform != 'win32':
        return {}
    out = {}
    try:
        import ctypes
        from ctypes import wintypes
        hwnd = wintypes.HWND(int(widget.winId()))
        for attr, value in ((20, 1),                       # DWMWA_USE_IMMERSIVE_DARK_MODE
                            (35, _colorref(caption)),      # DWMWA_CAPTION_COLOR (Windows 11)
                            (36, _colorref(text)),         # DWMWA_TEXT_COLOR
                            (34, _colorref(caption))):     # DWMWA_BORDER_COLOR
            v = ctypes.c_int(value)
            out[attr] = ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(v), ctypes.sizeof(v))
    except Exception:
        pass
    return out


class TitleBars(QObject):
    """Colours the title bar of every top-level window (main window, dialogs, message boxes) just before
    it is first shown.  Keep a reference: TitleBars(app)."""

    def __init__(self, app):
        super().__init__(app)
        app.installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Show and isinstance(obj, QWidget) and obj.isWindow() \
                and not obj.property('empower_title_bar'):
            obj.setProperty('empower_title_bar', True)
            color_title_bar(obj)
        return False


def launch_command():
    """How Windows starts this copy of the app: (program, arguments, working folder, icon file).
    The .exe starts itself and carries its icon (the .ico of a one-file .exe lives in a temporary folder);
    a source checkout starts pythonw.exe with EMPOWER_Explorer.py."""
    if getattr(sys, 'frozen', False):
        exe = os.path.abspath(sys.executable)
        return exe, '', os.path.dirname(exe), exe
    repo = os.path.dirname(HERE)
    python = os.path.abspath(sys.executable)
    pythonw = os.path.join(os.path.dirname(python), 'pythonw.exe')
    script = os.path.join(repo, 'EMPOWER_Explorer.py')
    return (pythonw if os.path.exists(pythonw) else python), f'"{script}"', repo, ICON_ICO


def set_window_relaunch(widget):
    """Tell the Windows taskbar how to start the app again from a button pinned while it runs, and with
    which icon (AppUserModel relaunch properties of the window).  No-op elsewhere or on failure."""
    if sys.platform != 'win32':
        return
    try:
        program, args, _workdir, icon = launch_command()
        _set_window_properties(int(widget.winId()), [
            (2, f'"{program}" {args}'.strip()),       # System.AppUserModel.RelaunchCommand
            (3, f'{icon},0'),                         # .RelaunchIconResource
            (4, 'EMPOWER Explorer'),                  # .RelaunchDisplayNameResource
            (5, APP_ID),                              # .ID, set last
        ])
    except Exception:
        pass


def _set_window_properties(hwnd, values):
    """Set System.AppUserModel.<pid> string properties of a window (IPropertyStore via ctypes)."""
    import ctypes
    import uuid
    from ctypes import wintypes

    class GUID(ctypes.Structure):
        _fields_ = [('d1', wintypes.DWORD), ('d2', wintypes.WORD), ('d3', wintypes.WORD), ('d4', ctypes.c_ubyte * 8)]

        def __init__(self, text):
            u = uuid.UUID(text)
            super().__init__(u.time_low, u.time_mid, u.time_hi_version, (ctypes.c_ubyte * 8)(*u.bytes[8:]))

    class PROPERTYKEY(ctypes.Structure):
        _fields_ = [('fmtid', GUID), ('pid', wintypes.DWORD)]

    class PROPVARIANT(ctypes.Structure):
        _fields_ = [('vt', wintypes.WORD), ('r1', wintypes.WORD), ('r2', wintypes.WORD), ('r3', wintypes.WORD),
                    ('p', ctypes.c_void_p), ('p2', ctypes.c_void_p)]

    store = ctypes.c_void_p()
    iid = GUID('886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99')                      # IID_IPropertyStore
    hr = ctypes.windll.shell32.SHGetPropertyStoreForWindow(wintypes.HWND(hwnd), ctypes.byref(iid),
                                                           ctypes.byref(store))
    if hr != 0 or not store:
        return
    vtbl = ctypes.cast(ctypes.cast(store, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
    set_value = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, ctypes.POINTER(PROPERTYKEY),
                                   ctypes.POINTER(PROPVARIANT))(vtbl[6])
    commit = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p)(vtbl[7])
    release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtbl[2])
    fmtid = GUID('9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3')                    # System.AppUserModel
    try:
        for pid, text in values:
            buf = ctypes.create_unicode_buffer(text)                         # the store keeps its own copy
            value = PROPVARIANT(vt=31, p=ctypes.cast(buf, ctypes.c_void_p))  # VT_LPWSTR
            set_value(store, ctypes.byref(PROPERTYKEY(fmtid, pid)), ctypes.byref(value))
        commit(store)
    finally:
        release(store)


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
