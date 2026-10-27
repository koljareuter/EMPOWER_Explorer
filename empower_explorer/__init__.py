"""EMPOWER Explorer - a fast desktop app for the EMPOWER KMOS combined cubes.

Start with:  python EMPOWER_Explorer.py   (or  python -m empower_explorer)

Package layout
--------------
config.py        where the EMPOWER data live (QSettings + defaults), reduction variants
scripts/         the EMPOWER notebook code and reference tables, shipped with the app
engine.py        the notebook code the app uses (scripts/kmos_sample.py, scripts/line_snr.py)
lines.py         emission lines (vacuum), zoom panels, BPT / WHAN demarcations
theme.py         colours, Qt stylesheet, pyqtgraph defaults
workers.py       background jobs (QThreadPool) so the window never freezes
state.py         AppState: the one shared selection (variant, galaxy, bands, aperture, z)
data/            scanning cubes, galaxy catalogue, band stacks, CSV tables, resolved map products
analysis.py      spectra, line fits, line maps, centring, per-spaxel fits (pure numpy, no Qt)
widgets/         reusable plot widgets (spectrum plot, image panel with aperture)
views/           the tabs: Spectrum, Maps, Population, Data QA + galaxy list and header card
pipeline/        Voronoi binning + multi-Gaussian line maps (no Qt; also a command-line tool)
app.py           MainWindow and main()
"""

__version__ = '1.1.0'
APP_NAME = 'EMPOWER Explorer'
ORG_NAME = 'EMPOWER'
