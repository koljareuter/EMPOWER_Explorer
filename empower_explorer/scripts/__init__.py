"""The EMPOWER analysis scripts, shipped with the app so it needs nothing but the data folder.

Unchanged copies of EMPOWER_Analysis_and_presentation/scripts (tools/sync_scripts.py updates them),
so the app and the pipeline give the same numbers as the notebooks:

kmos_sample.py               galaxy identity, redshifts, exposure times, band stacks
line_snr.py                  line S/N, noise, rest wavelengths, band resolutions
telluric_combined.py         telluric envelope (used by kmos_sample.atmospheric_transmission)
target_redshifts.csv         redshift catalogue (a copy in the data folder takes precedence)
sky_lines_uves_oliva.csv     OH sky lines (Hanuschik 2003, Oliva+15)

The scripts import each other by their top-level names (from telluric_combined import envelope), as
in the notebooks.  A finder at the end of sys.meta_path answers those imports with the copies here,
but only if nothing else provides the name: a notebook's own scripts folder on sys.path still wins.
"""
import importlib
import importlib.abc
import importlib.util
import sys

_NAMES = ('kmos_sample', 'line_snr', 'telluric_combined')


class _ScriptsFallback(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Top-level `import <script>` -> empower_explorer.scripts.<script> (the same module object)."""

    def find_spec(self, name, path=None, target=None):
        if path is None and name in _NAMES:
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        module = importlib.import_module(f'{__name__}.{spec.name}')
        _OWN_SPEC.setdefault(module.__name__, module.__spec__)
        return module

    def exec_module(self, module):
        module.__spec__ = _OWN_SPEC[module.__name__]       # already executed; keep its own spec


_OWN_SPEC = {}


if not any(type(f).__name__ == '_ScriptsFallback' for f in sys.meta_path):
    sys.meta_path.append(_ScriptsFallback())
