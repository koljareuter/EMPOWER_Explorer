"""Where the EMPOWER data live, and which combined-cube reductions exist.

The EMPOWER data folder is found in this order:
  1. the folder chosen in the app (Settings), stored with QSettings
  2. the environment variable EMPOWER_ROOT
  3. a folder EMPOWER_data or EMPOWER_Analysis_and_presentation next to the app (the repository or
     the .exe) or above it; for the .exe also the folder the .exe is in
It must contain at least one reduction folder (kmos_comb_* / kmos_proc_and_comb_*) with combined cubes
and must not be a reduction or product folder itself.
The analysis scripts and reference tables ship with the app (empower_explorer/scripts).
"""
import os
import sys
from pathlib import Path

from empower_explorer import APP_NAME, ORG_NAME

# Combined-cube reductions, the default first: (folder name, label shown in the app)
KNOWN_VARIANTS = [
    ('kmos_comb_hyb_tell', 'Hybrid + telluric'),
    ('kmos_comb_ohlin_tell_v2', 'OH-linear + telluric v2'),
    ('kmos_comb_ohlin_tell', 'OH-linear + telluric'),
    ('kmos_comb_ohlin', 'OH-linear'),
    ('kmos_comb_hyb', 'Hybrid'),
    ('kmos_proc_and_comb_server_tell', 'Server + telluric'),
    ('kmos_comb_det3fix', 'det3 fix'),
    ('kmos_comb_zapfix', 'ZAP fix'),
    ('kmos_proc_and_comb_server', 'Server (old)'),
]
DEFAULT_VARIANT = 'kmos_comb_hyb_tell'     # the reduction the maps are made from (2026-10-27)

# CSV result tables of the EMPOWER notebooks (file name -> short name used in the app)
TABLES = {
    'kmos_galaxy_properties.csv': 'properties',
    'kmos_bpt_table.csv': 'bpt',
    'kmos_hbeta_detection.csv': 'hbeta',
    'kmos_sample_overview.csv': 'overview',
    'halpha_census.csv': 'census',
    'kmos_snr_distribution.csv': 'snr_distribution',
}


def settings():
    from PyQt5.QtCore import QSettings      # lazy: the data layer and the pipeline run without Qt
    return QSettings(ORG_NAME, APP_NAME)


def _stored_root():
    try:
        return settings().value('empower_root', '', type=str)
    except Exception:          # no Qt (server) -> environment / default location only
        return ''


DATA_FOLDER_NAMES = ('EMPOWER_data', 'EMPOWER_Analysis_and_presentation')
SCRIPTS_DIR = Path(__file__).resolve().parent / 'scripts'     # analysis scripts + reference tables


def bundled(name):
    """Path of a file shipped with the app in empower_explorer/scripts."""
    return str(SCRIPTS_DIR / name)


def redshift_table(root):
    """target_redshifts.csv: the data folder's own copy if it has one (<root>/target_redshifts.csv or
    <root>/scripts/target_redshifts.csv, where the notebooks keep it), else the one shipped with the app."""
    for p in ([Path(root) / 'target_redshifts.csv', Path(root) / 'scripts' / 'target_redshifts.csv'] if root else []):
        if p.is_file():
            return str(p)
    return bundled('target_redshifts.csv')


def _is_reduction_name(name):
    return name.startswith('kmos_comb') or name.startswith('kmos_proc_and_comb')


def is_empower_root(path):
    """True for a data folder: it holds a reduction folder (kmos_comb_* / kmos_proc_and_comb_*) with combined
    cubes and is neither a reduction folder (some hold per-band reductions) nor the product folder itself."""
    p = Path(path) if path else None
    if not (p and p.is_dir()) or _is_reduction_name(p.name) or p.name == 'kmos_voronoi_maps':
        return False
    try:
        return any(c.is_dir() and _is_reduction_name(c.name) and _has_cubes(c) for c in p.iterdir())
    except OSError:
        return False


def default_root_candidates():
    """DATA_FOLDER_NAMES next to the app (repository or .exe) or up to six levels above it;
    for the .exe first the folder the .exe is in."""
    frozen = getattr(sys, 'frozen', False)
    here = Path(sys.executable if frozen else __file__).resolve().parent
    out = [here] if frozen else []
    for parent in [here, *here.parents][:7]:
        out += [parent / name for name in DATA_FOLDER_NAMES]
    return out


def empower_root():
    """The EMPOWER data folder, or None if it can not be found (the app then asks for it)."""
    stored = _stored_root()
    for cand in [stored, os.environ.get('EMPOWER_ROOT', '')] + [str(c) for c in default_root_candidates()]:
        if cand and is_empower_root(cand):
            return str(Path(cand).resolve())
    return None


def set_empower_root(path):
    settings().setValue('empower_root', str(path))


def _has_cubes(folder):
    """True as soon as one DATACUBE_COMBINED_*.fits is found below folder (stops early)."""
    for _dirpath, _dirs, files in os.walk(folder):
        if any(f.startswith('DATACUBE_COMBINED_') and f.endswith('.fits') for f in files):
            return True
    return False


def available_variants(root):
    """[(folder, label)] of the reductions under root that contain combined cubes, newest first;
    any other kmos_comb* / kmos_proc_and_comb* folder is listed at the end (copies skipped)."""
    if not root:
        return []
    root = Path(root)
    out = [(name, label) for name, label in KNOWN_VARIANTS if (root / name).is_dir()]
    known = {name for name, _ in out}
    for p in sorted(root.iterdir()):
        if (p.is_dir() and p.name not in known and 'Kopie' not in p.name and 'backup' not in p.name.lower()
                and _is_reduction_name(p.name)):
            out.append((p.name, p.name.replace('kmos_', '').replace('_', ' ')))
    return [(name, label) for name, label in out if _has_cubes(root / name)]


def cache_dir():
    """Per-user cache folder (header scans) - never inside the data folders."""
    base = os.environ.get('LOCALAPPDATA') or os.path.join(Path.home(), '.cache')
    d = Path(base) / 'EMPOWER_Explorer'
    d.mkdir(parents=True, exist_ok=True)
    return d
