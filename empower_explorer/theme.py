"""Colours, the Qt stylesheet and pyqtgraph defaults (dark theme of the KMOS Viewer)."""
import pyqtgraph as pg
from PyQt5.QtGui import QColor, QFont, QPalette


class C:
    BG_DARK = '#0F172A'
    BG_MEDIUM = '#1E293B'
    BG_LIGHT = '#334155'
    BG_HOVER = '#475569'
    ACCENT = '#06B6D4'          # cyan: buttons, selection, aperture
    ACCENT_HOVER = '#0891B2'
    ACCENT_PRESSED = '#0E7490'
    EMERALD = '#10B981'         # S/N spectra
    BLUE = '#3B82F6'            # flux spectra
    PURPLE = '#8B5CF6'
    PINK = '#EC4899'            # warnings
    AMBER = '#F59E0B'
    TEXT = '#F1F5F9'
    TEXT_DIM = '#94A3B8'
    GRID = '#334155'
    TELLURIC = (148, 163, 184, 45)   # shading of low atmospheric transmission (RGBA)
    SKY = '#64748B'


# Spectrum colour per band (several bands of one galaxy on one axis)
BAND_COLORS = {'IZ': '#38BDF8', 'YJ': '#34D399', 'H': '#A3E635', 'HK': '#CBD5E1', 'K': '#818CF8'}

# Colour maps for images (pyqtgraph names, matplotlib maps are available in pyqtgraph >= 0.13)
CMAP_IMAGE = 'magma'
# Spectral_r colours with extra transitions around zero velocity.
CMAP_VELOCITY = 'rainbow_r'
CMAP_SEQUENTIAL = 'viridis'

CATEGORY_COLORS = ['#38BDF8', '#F472B6', '#A3E635', '#FACC15', '#A78BFA', '#FB923C',
                   '#34D399', '#F87171', '#60A5FA', '#E879F9']

FONT_FAMILY = 'Segoe UI'


def band_color(band):
    return BAND_COLORS.get(band, C.ACCENT)


def setup_pyqtgraph():
    pg.setConfigOptions(antialias=True, background=C.BG_DARK, foreground=C.TEXT_DIM,
                        imageAxisOrder='row-major', leftButtonPan=True)


def apply_palette(app):
    app.setStyle('Fusion')
    app.setFont(QFont(FONT_FAMILY, 10))
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(C.BG_DARK))
    pal.setColor(QPalette.WindowText, QColor(C.TEXT))
    pal.setColor(QPalette.Base, QColor(C.BG_MEDIUM))
    pal.setColor(QPalette.AlternateBase, QColor('#172033'))
    pal.setColor(QPalette.ToolTipBase, QColor(C.BG_MEDIUM))
    pal.setColor(QPalette.ToolTipText, QColor(C.TEXT))
    pal.setColor(QPalette.Text, QColor(C.TEXT))
    pal.setColor(QPalette.Button, QColor(C.BG_LIGHT))
    pal.setColor(QPalette.ButtonText, QColor(C.TEXT))
    pal.setColor(QPalette.Highlight, QColor(C.ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor(C.BG_DARK))
    pal.setColor(QPalette.PlaceholderText, QColor(C.TEXT_DIM))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor('#64748B'))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor('#64748B'))
    app.setPalette(pal)
    app.setStyleSheet(STYLESHEET)


STYLESHEET = f"""
QMainWindow, QDialog {{ background-color: {C.BG_DARK}; }}
QWidget {{ color: {C.TEXT}; font-size: 13px; }}
QLabel {{ background: transparent; }}
QLabel[role="dim"] {{ color: {C.TEXT_DIM}; font-size: 12px; }}
QLabel[role="title"] {{ color: {C.TEXT}; font-size: 20px; font-weight: bold; }}
QLabel[role="section"] {{ color: {C.TEXT_DIM}; font-size: 11px; font-weight: bold; letter-spacing: 1px; }}
QLabel[role="chip"] {{ background-color: {C.BG_LIGHT}; color: {C.TEXT}; border-radius: 9px;
                       padding: 2px 9px; font-size: 12px; }}
QLabel[role="warn"] {{ color: {C.PINK}; font-size: 12px; }}
QFrame[role="card"] {{ background-color: {C.BG_MEDIUM}; border: 1px solid {C.BG_LIGHT}; border-radius: 10px; }}
QPushButton {{ background-color: {C.ACCENT}; color: white; border: none; padding: 7px 14px;
               border-radius: 6px; font-weight: bold; }}
QPushButton:hover {{ background-color: {C.ACCENT_HOVER}; }}
QPushButton:pressed {{ background-color: {C.ACCENT_PRESSED}; }}
QPushButton:disabled {{ background-color: {C.BG_LIGHT}; color: {C.TEXT_DIM}; }}
QPushButton[flat="true"], QToolButton {{ background-color: {C.BG_LIGHT}; color: {C.TEXT};
               border: none; padding: 6px 10px; border-radius: 6px; font-weight: normal; }}
QPushButton[flat="true"]:hover, QToolButton:hover {{ background-color: {C.BG_HOVER}; }}
QPushButton:checked, QToolButton:checked {{ background-color: {C.ACCENT}; color: white; }}
QPushButton[flat="true"]:disabled, QToolButton:disabled {{ background-color: {C.BG_MEDIUM}; color: #64748B; }}
QComboBox, QLineEdit, QDoubleSpinBox, QSpinBox {{ background-color: {C.BG_LIGHT}; color: {C.TEXT};
               border: 2px solid {C.BG_LIGHT}; padding: 5px 8px; border-radius: 6px; }}
QComboBox:hover, QLineEdit:hover, QDoubleSpinBox:hover, QSpinBox:hover,
QLineEdit:focus, QDoubleSpinBox:focus {{ border: 2px solid {C.ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{ background-color: {C.BG_LIGHT}; color: {C.TEXT};
               selection-background-color: {C.ACCENT}; border: none; outline: none; }}
QCheckBox, QRadioButton {{ spacing: 7px; background: transparent; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 2px solid {C.BG_HOVER};
               border-radius: 4px; background-color: {C.BG_MEDIUM}; }}
QCheckBox::indicator:checked {{ background-color: {C.ACCENT}; border: 2px solid {C.ACCENT}; }}
QSlider::groove:horizontal {{ height: 6px; background: {C.BG_LIGHT}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: {C.ACCENT}; width: 16px; margin: -6px 0; border-radius: 8px; }}
QTabWidget::pane {{ border: none; background: {C.BG_DARK}; }}
QTabBar::tab {{ background: transparent; color: {C.TEXT_DIM}; padding: 9px 18px; margin-right: 2px;
               border-bottom: 3px solid transparent; font-weight: bold; font-size: 13px; min-width: 80px; }}
QTabBar::tab:selected {{ color: {C.TEXT}; border-bottom: 3px solid {C.ACCENT}; }}
QTabBar::tab:hover {{ color: {C.TEXT}; }}
QTableView, QTreeView, QListView, QPlainTextEdit, QTextEdit {{ background-color: {C.BG_MEDIUM};
               alternate-background-color: #172033; border: 1px solid {C.BG_LIGHT}; border-radius: 8px;
               gridline-color: {C.BG_LIGHT}; selection-background-color: {C.ACCENT};
               selection-color: white; }}
QHeaderView::section {{ background-color: {C.BG_DARK}; color: {C.TEXT_DIM}; border: none;
               border-bottom: 1px solid {C.BG_LIGHT}; padding: 5px 6px; font-weight: bold; font-size: 12px; }}
QTableCornerButton::section {{ background-color: {C.BG_DARK}; border: none; }}
QToolBar {{ background-color: {C.BG_MEDIUM}; border: none; border-bottom: 1px solid {C.BG_LIGHT};
               spacing: 8px; padding: 6px 10px; }}
QStatusBar {{ background-color: {C.BG_MEDIUM}; color: {C.TEXT_DIM}; border-top: 1px solid {C.BG_LIGHT}; }}
QStatusBar QLabel {{ color: {C.TEXT_DIM}; font-size: 12px; }}
QSplitter::handle {{ background-color: {C.BG_DARK}; }}
QSplitter::handle:hover {{ background-color: {C.BG_LIGHT}; }}
QScrollArea {{ background: transparent; border: none; }}
QScrollBar:vertical {{ background: {C.BG_DARK}; width: 9px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {C.BG_LIGHT}; border-radius: 4px; min-height: 30px; }}
QScrollBar:horizontal {{ background: {C.BG_DARK}; height: 9px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {C.BG_LIGHT}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0px; height: 0px; }}
QProgressBar {{ background-color: {C.BG_LIGHT}; border: none; border-radius: 4px; height: 8px;
               text-align: center; color: transparent; }}
QProgressBar::chunk {{ background-color: {C.ACCENT}; border-radius: 4px; }}
QToolTip {{ background-color: {C.BG_MEDIUM}; color: {C.TEXT}; border: 1px solid {C.BG_LIGHT};
               padding: 5px; border-radius: 4px; }}
QMenu {{ background-color: {C.BG_MEDIUM}; border: 1px solid {C.BG_LIGHT}; }}
QMenu::item:selected {{ background-color: {C.ACCENT}; }}
QGroupBox {{ border: 1px solid {C.BG_LIGHT}; border-radius: 8px; margin-top: 14px; padding-top: 8px;
               font-weight: bold; color: {C.TEXT_DIM}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 5px; }}
"""
