"""The EMPOWER analysis scripts, shipped with the app so it needs nothing but the data folder.

Unchanged copies of EMPOWER_Analysis_and_presentation/scripts (tools/sync_scripts.py updates them),
so the app and the pipeline give the same numbers as the notebooks:

kmos_sample.py               galaxy identity, redshifts, exposure times, band stacks
line_snr.py                  line S/N, noise, rest wavelengths, band resolutions
telluric_combined.py         telluric envelope (used by kmos_sample.atmospheric_transmission)
target_redshifts.csv         redshift catalogue (a copy in the data folder takes precedence)
sky_lines_uves_oliva.csv     OH sky lines (Hanuschik 2003, Oliva+15)

The scripts import each other by their top-level names (from telluric_combined import envelope),
as in the notebooks, so they are registered under those names as well.
"""
import importlib
import sys

for _name in ('telluric_combined', 'line_snr', 'kmos_sample'):
    sys.modules.setdefault(_name, importlib.import_module(f'{__name__}.{_name}'))
