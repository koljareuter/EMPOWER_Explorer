"""The EMPOWER notebook code, used as-is so the app gives the same numbers as the notebooks.

    from empower_explorer import engine
    engine.load(root)            # once, after the EMPOWER folder is known
    engine.ks.cube_info(...)     # scripts/kmos_sample.py
    engine.lsnr.measure_lines(...)  # scripts/line_snr.py
    engine.REST_UM['Ha']         # vacuum rest wavelengths [micron]
"""
import importlib
import sys
from pathlib import Path

ks = None          # scripts/kmos_sample.py
lsnr = None        # scripts/line_snr.py
REST_UM = {}
BAND_RESOLUTION = {}
ROOT = None


class EngineError(RuntimeError):
    pass


_MODULES = ('line_snr', 'kmos_sample', 'telluric_combined')


def load(root):
    """Import kmos_sample and line_snr from <root>/scripts (re-import if the root changed).
    All or nothing: if an import fails, sys.path, sys.modules and the loaded engine stay as before."""
    global ks, lsnr, REST_UM, BAND_RESOLUTION, ROOT
    scripts = str((Path(root) / 'scripts').resolve())
    if ROOT == str(Path(root).resolve()) and ks is not None:
        return
    saved_path = list(sys.path)
    saved_mods = {n: sys.modules[n] for n in _MODULES if n in sys.modules}
    try:
        for p in list(sys.path):
            if p.endswith('scripts') and p != scripts and (Path(p) / 'kmos_sample.py').exists():
                sys.path.remove(p)
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        for name in _MODULES:
            sys.modules.pop(name, None)
        new_lsnr = importlib.import_module('line_snr')
        new_ks = importlib.import_module('kmos_sample')
        rest, res = dict(new_lsnr.REST_UM), dict(new_lsnr.BAND_RESOLUTION)
    except Exception as e:      # pragma: no cover - shown to the user
        sys.path[:] = saved_path
        for name in _MODULES:
            sys.modules.pop(name, None)
        sys.modules.update(saved_mods)
        raise EngineError(f'Could not import the EMPOWER scripts from {scripts}:\n{e}') from e
    lsnr, ks, REST_UM, BAND_RESOLUTION = new_lsnr, new_ks, rest, res
    ROOT = str(Path(root).resolve())


def require():
    if ks is None or lsnr is None:
        raise EngineError('EMPOWER scripts not loaded - call engine.load(root) first')
