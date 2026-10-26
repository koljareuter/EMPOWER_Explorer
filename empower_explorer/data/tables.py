"""The CSV result tables of the EMPOWER notebooks, joined per galaxy.

Galaxy key everywhere: 'FIELD tid' (kmos_sample.galaxy_name), e.g. 'COSMOS 101639'.
COLUMNS describes the columns the app offers in plots and tables (label, unit, hint).
"""
import os

import numpy as np
import pandas as pd

from empower_explorer import config

# column -> (label, unit, short description). Only columns listed here appear in the axis pickers.
COLUMNS = {
    'z': ('Redshift', '', 'catalogue redshift (target_redshifts.csv)'),
    'z_kmos': ('z (KMOS lines)', '', 'redshift measured from the KMOS emission lines'),
    'dv_kmos': ('Δv KMOS − catalogue', 'km/s', 'velocity offset of the KMOS lines from the catalogue z'),
    'texp_h': ('On-source time', 'h', 'summed over all bands of the loaded reduction'),
    'texp_ha_h': ('On-source time (Hα band)', 'h', 'band the Hα S/N was measured in (properties table)'),
    'n_bands': ('Number of bands', '', ''),
    'snr_ha': ('S/N Hα', '', 'calibrated integrated S/N (line_snr.measure_lines)'),
    'snr_hb': ('S/N Hβ', '', ''),
    'snr_oiii': ('S/N [OIII]5007', '', ''),
    'snr_nii': ('S/N [NII]6583', '', ''),
    'snr_sii1': ('S/N [SII]6716', '', ''),
    'flux_ha': ('Hα flux (normalised stack)', 'norm. µm',
                'per-cube median-normalised stacks of the catalogue notebook - use ratios only'),
    'EW_ha': ('EW(Hα)', 'Å', 'rest-frame equivalent width'),
    'sigma_ha': ('σ(Hα)', 'km/s', 'intrinsic velocity dispersion (integrated)'),
    'N2': ('log [NII]/Hα', '', ''),
    'O3': ('log [OIII]/Hβ', '', ''),
    'S2': ('log [SII]/Hα', '', ''),
    'OH_N2': ('12+log(O/H) (N2)', '', ''),
    'OH_O3N2': ('12+log(O/H) (O3N2)', '', ''),
    'logM': ('log M*', 'M☉', 'stellar mass (LePhare, COSMOS2020)'),
    'logSFR': ('log SFR', 'M☉/yr', ''),
    'sSFR': ('log sSFR', '1/yr', ''),
    'dMS': ('Δ main sequence', 'dex', 'offset from the star-forming main sequence'),
    'av': ('A_V (Balmer)', 'mag', 'from the Hβ detection notebook'),
    'mag_H': ('H magnitude', 'mag', ''),
    'mag_Ks': ('Ks magnitude', 'mag', ''),
    'ra': ('RA', 'deg', ''),
    'dec': ('Dec', 'deg', ''),
}

CATEGORICAL = {
    'field': 'Field',
    'class_NII_BPT': 'BPT class ([NII])',
    'class_SII_BPT': 'BPT class ([SII])',
    'whan': 'WHAN class',
    'z_type': 'Redshift type',
    'bands': 'Bands',
    'band_ha': 'Band with Hα',
}


def label(col):
    if col in COLUMNS:
        lab, unit, _ = COLUMNS[col]
        return f'{lab} [{unit}]' if unit else lab
    return CATEGORICAL.get(col, col)


def _key(df, field_col, tid_col='tid'):
    return df[field_col].astype(str).str.strip() + ' ' + df[tid_col].astype(str).str.strip()


def load_tables(root):
    """{short name: DataFrame with a 'key' column} for every CSV of config.TABLES that exists."""
    out = {}
    for fname, short in config.TABLES.items():
        path = os.path.join(root, fname)
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        field_col = 'region' if 'region' in df.columns and 'field' not in df.columns else 'field'
        if field_col in df.columns and 'tid' in df.columns:
            df['key'] = _key(df, field_col)
        out[short] = df
    return out


def galaxy_table(tables):
    """One row per galaxy: properties + BPT classes + Hβ/A_V + WHAN class (index = key)."""
    frames = []
    if 'properties' in tables:
        frames.append(tables['properties'].drop_duplicates('key').set_index('key'))
    if 'bpt' in tables:
        frames.append(tables['bpt'].drop_duplicates('key').set_index('key')[['class_NII_BPT', 'class_SII_BPT']])
    if 'hbeta' in tables:
        h = tables['hbeta'].drop_duplicates('key').set_index('key')
        frames.append(h[[c for c in ('av', 'av_lo', 'av_hi', 'ebv_lp') if c in h.columns]])
    if not frames:
        return pd.DataFrame()
    df = frames[0]
    for f in frames[1:]:
        df = df.join(f[[c for c in f.columns if c not in df.columns]], how='outer')
    if 'N2' in df.columns and 'EW_ha' in df.columns:
        df['whan'] = [_whan(n, e) for n, e in zip(pd.to_numeric(df['N2'], errors='coerce'),
                                                   pd.to_numeric(df['EW_ha'], errors='coerce'))]
    return df


def _whan(n2, ew):
    from empower_explorer.lines import whan_class
    return whan_class(n2, ew)


def numeric(df, col):
    """Column as float array (NaN for text / missing)."""
    if col not in df.columns:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
