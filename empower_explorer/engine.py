"""The EMPOWER notebook code, used as-is so the app gives the same numbers as the notebooks.

The scripts ship with the app (empower_explorer/scripts); only the data folder is external.

    from empower_explorer import engine
    engine.load(root)               # once, after the EMPOWER data folder is known
    engine.ks.cube_info(...)        # scripts/kmos_sample.py
    engine.lsnr.measure_lines(...)  # scripts/line_snr.py
    engine.REST_UM['Ha']            # vacuum rest wavelengths [micron]
"""
from pathlib import Path

from empower_explorer.scripts import kmos_sample as ks, line_snr as lsnr

REST_UM = dict(lsnr.REST_UM)
BAND_RESOLUTION = dict(lsnr.BAND_RESOLUTION)
ROOT = None        # the EMPOWER data folder in use


class EngineError(RuntimeError):
    pass


def load(root):
    """Use the EMPOWER data folder `root`."""
    global ROOT
    if not root or not Path(root).is_dir():
        raise EngineError(f'EMPOWER data folder not found: {root}')
    ROOT = str(Path(root).resolve())
