"""Emission lines, zoom panels and diagnostic-diagram demarcations.

Rest wavelengths are VACUUM micron (KMOS cubes have CTYPE3 = 'WAVE').  REST starts with the
values of scripts/line_snr.py REST_UM and is refreshed from it by sync() once the engine is
loaded, so the app always uses the notebooks' table.
"""
import numpy as np

C_KMS = 299792.458

REST = {
    'OII3726': 0.372709, 'OII3729': 0.372988, 'Hg': 0.434169, 'Hb': 0.486268,
    'OIII4959': 0.496030, 'OIII5007': 0.500824, 'HeI5876': 0.587725, 'OI6300': 0.630204,
    'NII6548': 0.654986, 'Ha': 0.656461, 'NII6583': 0.658527, 'SII6716': 0.671829,
    'SII6731': 0.673267,
}

# Display colours of the lines (also used for map colour bars of that line)
LINE_COLORS = {
    'Hb': '#FACC15', 'OIII4959': '#C4B5FD', 'OIII5007': '#A78BFA',
    'NII6548': '#FB923C', 'Ha': '#EF4444', 'NII6583': '#F97316',
    'SII6716': '#F472B6', 'SII6731': '#F9A8D4', 'OII3726': '#94A3B8', 'OII3729': '#94A3B8',
    'Hg': '#FDE68A', 'HeI5876': '#94A3B8', 'OI6300': '#94A3B8',
}

LINE_LABELS = {
    'Hb': 'Hβ', 'OIII4959': '[OIII]4959', 'OIII5007': '[OIII]5007', 'NII6548': '[NII]6548',
    'Ha': 'Hα', 'NII6583': '[NII]6583', 'SII6716': '[SII]6716', 'SII6731': '[SII]6731',
    'OII3726': '[OII]3727', 'OII3729': '[OII]3729', 'Hg': 'Hγ', 'HeI5876': 'HeI', 'OI6300': '[OI]6300',
}

# The lines measured for every galaxy (names = keys of REST; same set as the notebooks)
FIT_LINES = ['Hb', 'OIII4959', 'OIII5007', 'NII6548', 'Ha', 'NII6583', 'SII6716', 'SII6731']

# Zoom panels in wavelength order: (key, title, lines drawn, line the window is centred on,
# minimum half window [micron, observed]).  Top row Hβ / [OIII], bottom row Hα / [SII].
PANELS = [
    ('hb', 'Hβ', ['Hb'], 'Hb', 0.012),
    ('oiii', '[OIII] λλ4959,5007', ['OIII5007', 'OIII4959'], 'OIII5007', 0.015),
    ('ha', 'Hα + [NII]', ['Ha', 'NII6548', 'NII6583'], 'Ha', 0.015),
    ('sii', '[SII] λλ6716,6731', ['SII6716', 'SII6731'], ('SII6716', 'SII6731'), 0.015),
]

# Line markers in the full spectrum: (label or None = marker only, key)
FULL_MARKERS = [('[OII]', 'OII3726'), ('Hγ', 'Hg'), ('Hβ', 'Hb'), (None, 'OIII4959'),
                ('[OIII]', 'OIII5007'), ('HeI', 'HeI5876'), ('[OI]', 'OI6300'), (None, 'NII6548'),
                ('Hα', 'Ha'), (None, 'NII6583'), (None, 'SII6716'), ('[SII]', 'SII6731')]


def sync(rest_um):
    """Take the rest wavelengths of the engine (scripts/line_snr.py REST_UM)."""
    REST.update({k: float(v) for k, v in rest_um.items()})


def rest(key):
    return REST[key]


def panel_window(panel, z):
    """(centre, lo, hi) of a zoom panel in observed micron."""
    _, _, keys, centre_key, min_half = panel
    if isinstance(centre_key, tuple):
        centre_rest = float(np.mean([REST[k] for k in centre_key]))
    else:
        centre_rest = REST[centre_key]
    spread = max(abs(REST[k] - centre_rest) for k in keys)
    half = max(min_half, spread * (1 + z) + 0.003)
    c = centre_rest * (1 + z)
    return c, c - half, c + half


# ---------------------------------------------------------------- diagnostic diagrams
def kauffmann03(n2):
    """[NII]-BPT star-forming / composite demarcation (Kauffmann+03), log([OIII]/Hβ)."""
    n2 = np.asarray(n2, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        y = 0.61 / (n2 - 0.05) + 1.30
    return np.where(n2 < 0.05, y, np.nan)


def kewley01_n2(n2):
    """[NII]-BPT maximum starburst line (Kewley+01)."""
    n2 = np.asarray(n2, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        y = 0.61 / (n2 - 0.47) + 1.19
    return np.where(n2 < 0.47, y, np.nan)


def kewley01_s2(s2):
    """[SII]-BPT maximum starburst line (Kewley+01)."""
    s2 = np.asarray(s2, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        y = 0.72 / (s2 - 0.32) + 1.30
    return np.where(s2 < 0.32, y, np.nan)


def kewley06_s2_seyfert_liner(s2):
    """[SII]-BPT Seyfert / LINER line (Kewley+06)."""
    return 1.89 * np.asarray(s2, float) + 0.76


def whan_class(n2, ew_ha):
    """WHAN class (Cid Fernandes+11) from log([NII]/Hα) and EW(Hα) [Å]."""
    if not (np.isfinite(n2) and np.isfinite(ew_ha)):
        return ''
    if ew_ha < 0.5:
        return 'passive'
    if ew_ha < 3:
        return 'retired'
    if n2 < -0.4:
        return 'SF'
    return 'sAGN' if ew_ha > 6 else 'wAGN'
