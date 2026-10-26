"""Where the EMPOWER data live, and which combined-cube reductions exist.

The EMPOWER folder (EMPOWER_Analysis_and_presentation) is found in this order:
  1. the folder chosen in the app (Settings), stored with QSettings
  2. the environment variable EMPOWER_ROOT
  3. next to this repository: <ESO_rep>/EMPOWER_Analysis_and_presentation
It must contain scripts/kmos_sample.py, scripts/line_snr.py and scripts/target_redshifts.csv.
"""
import os
from pathlib import Path

from empower_explorer import APP_NAME, ORG_NAME

# Combined-cube reductions, newest first: (folder name, label shown in the app)
KNOWN_VARIANTS = [
    ('kmos_comb_ohlin_tell_v2', 'OH-linear + telluric v2'),
    ('kmos_comb_hyb_tell', 'Hybrid + telluric'),
    ('kmos_comb_ohlin_tell', 'OH-linear + telluric'),
    ('kmos_comb_ohlin', 'OH-linear'),
    ('kmos_comb_hyb', 'Hybrid'),
    ('kmos_proc_and_comb_server_tell', 'Server + telluric'),
    ('kmos_comb_det3fix', 'det3 fix'),
    ('kmos_comb_zapfix', 'ZAP fix'),
    ('kmos_proc_and_comb_server', 'Server (old)'),
]
DEFAULT_VARIANT = 'kmos_comb_ohlin_tell_v2'     # the reduction the notebooks use (2026-10-25)

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


def is_empower_root(path):
    p = Path(path) if path else None
    return bool(p and (p / 'scripts' / 'kmos_sample.py').is_file()
                and (p / 'scripts' / 'line_snr.py').is_file())


def default_root_candidates():
    here = Path(__file__).resolve()
    out = []
    for parent in here.parents:
        out.append(parent / 'EMPOWER_Analysis_and_presentation')
        if len(out) > 6:
            break
    return out


def empower_root():
    """The EMPOWER folder, or None if it can not be found (the app then asks for it)."""
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
                and (p.name.startswith('kmos_comb') or p.name.startswith('kmos_proc_and_comb'))):
            out.append((p.name, p.name.replace('kmos_', '').replace('_', ' ')))
    return [(name, label) for name, label in out if _has_cubes(root / name)]


def cache_dir():
    """Per-user cache folder (header scans) - never inside the data folders."""
    base = os.environ.get('LOCALAPPDATA') or os.path.join(Path.home(), '.cache')
    d = Path(base) / 'EMPOWER_Explorer'
    d.mkdir(parents=True, exist_ok=True)
    return d
