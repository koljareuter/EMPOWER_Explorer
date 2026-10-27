# EMPOWER Explorer

A galaxy viewer built for the EMPOWER KMOS sample: combined data cubes, spectra and line fits,
resolved line maps (Voronoi binning, multi-Gaussian fits) and the properties of the whole sample.

To run it you need two things: **the app** (the Windows .exe or this repository with Python) and
**the EMPOWER data folder**, which is not part of this repository. Everything else ships with
the app, including the analysis scripts of the EMPOWER notebooks, so the app gives the notebooks'
numbers.

## Getting started

### Windows: the .exe

1. Download `EMPOWER_Explorer.exe` (version 1.1.0 or newer) from the
   [releases](https://github.com/koljareuter/EMPOWER_Explorer/releases). Older releases still need
   the analysis scripts in the data folder.
2. Start it and choose the EMPOWER data folder when it asks. The app remembers the folder.

### Any system: Python 3.10 or newer

```bash
git clone https://github.com/koljareuter/EMPOWER_Explorer.git
cd EMPOWER_Explorer
pip install -r requirements.txt
python EMPOWER_Explorer.py
```

`python -m empower_explorer` starts the app as well. Tested on Windows 11 with Python 3.14.

## The data folder

The folder name does not matter. The app finds it automatically if it is called `EMPOWER_data` or
`EMPOWER_Analysis_and_presentation` and lies next to the repository or the .exe (or in a folder
above them), or if the environment variable `EMPOWER_ROOT` points to it. Otherwise the app asks
for it once.

```
EMPOWER_data/
├── kmos_comb_hyb_tell/                    a reduction: DATACUBE_COMBINED_*.fits and TELLEFF_*.fits
├── kmos_voronoi_maps/kmos_comb_hyb_tell/  line-map products of that reduction (optional)
├── kmos_galaxy_properties.csv             notebook tables (optional)
├── ...
└── target_redshifts.csv                   redshift catalogue (optional)
```

| In the data folder | Needed | Used for |
|---|---|---|
| `kmos_comb_hyb_tell/` (Hybrid + telluric, about 1.5 GB): copy the whole folder, the cubes and the `TELLEFF_*.fits` next to them, in any subfolders | yes | everything; the default reduction |
| other `kmos_comb_*` / `kmos_proc_and_comb_*` folders | no | other reductions, chosen in the toolbar |
| `kmos_voronoi_maps/<reduction>/` | no | line maps; otherwise made in the app or with the pipeline |
| `kmos_galaxy_properties.csv`, `kmos_bpt_table.csv`, `kmos_hbeta_detection.csv`, `kmos_sample_overview.csv`, `halpha_census.csv`, `kmos_snr_distribution.csv` | no | Population and Data QA tabs (empty without them) |
| `kmos_*maps*.fits`, `kmos_*voronoi*.fits` (the notebooks' map products) | no | further map sources in the Maps tab |
| `target_redshifts.csv` (or `scripts/target_redshifts.csv`) | no | replaces the redshift catalogue shipped with the app |

Choosing a reduction folder or `kmos_voronoi_maps` by mistake is fine: the app then uses the folder
above it.

## Line maps

In the app: **Maps** tab → **Compute Voronoi fit** for the selected galaxy. For all galaxies of a
reduction, use the pipeline from the command line in the repository folder (no Qt needed, also for
a server):

```bash
python -m empower_explorer.pipeline.voronoi_binning --all
```

Useful options: `--root <data folder>` (default: the folder chosen in the app, then `EMPOWER_ROOT`,
then automatic), `--variant kmos_comb_hyb_tell`, `--galaxy "COSMOS 92754"` (repeatable),
`--max-components 3`, `--target-sn 15`, `--overwrite`, `--workers 4`, `--out <folder>`. Galaxies that
already have a product of the same pipeline version are skipped unless `--overwrite` is given.

The pipeline fits Hβ, [OIII] 4959/5007, Hα, [NII] 6548/6583 and [SII] 6716/6731 with up to three
Gaussian components per line group, on Voronoi bins of the Hα S/N (else [OIII], else Hβ) that are
shared by all bands. It writes one FITS file per galaxy, `kmos_voronoi_maps/<reduction>/<FIELD>_<id>_voronoi.fits`,
plus `index.csv` / `index.json`. Each file holds the maps (fluxes, velocities, dispersions, line
ratios, n_e, metallicity, EW), the table of all bins and of their sum, the lines, the configuration
and the noise model, with FITS units and version stamps. Galaxies without a detected line get one
central bin with upper limits. Read a product in Python with:

```python
from empower_explorer.pipeline import voronoi_binning as vb
p = vb.read_product('COSMOS_92754_voronoi.fits')   # p['maps'], p['bins'], p['total'], p['header'], ...
```

## Repository layout

```
EMPOWER_Explorer.py      start the app
EMPOWER_Explorer.spec    build the Windows .exe (PyInstaller)
empower_explorer/
  app.py, state.py, views/, widgets/   the app (PyQt5, pyqtgraph)
  data/                  scanning cubes, galaxy catalogue, stacks, tables, map products
  analysis.py            spectra, line fits, line maps (no Qt)
  pipeline/              Voronoi binning + multi-Gaussian line maps (no Qt)
  scripts/               analysis scripts and reference tables of the EMPOWER notebooks
tools/sync_scripts.py    update empower_explorer/scripts from the notebooks' scripts folder
```

`empower_explorer/scripts` holds unchanged copies of `kmos_sample.py`, `line_snr.py`,
`telluric_combined.py`, `target_redshifts.csv` and `sky_lines_uves_oliva.csv` from
`EMPOWER_Analysis_and_presentation/scripts`. After changing them there, copy them over with:

```bash
python tools/sync_scripts.py "<path to EMPOWER_Analysis_and_presentation>"
```

## Building the .exe

```bash
pip install pyinstaller
pyinstaller EMPOWER_Explorer.spec
```

The result is `dist/EMPOWER_Explorer.exe`. It is not committed (GitHub accepts no files above
100 MB); upload it to a GitHub release instead.

## Citation, licence and contact

For questions or contributions, feel free to contact me anytime. Feel also free to use this code
for anything you want (MIT licence, see `LICENSE`), but if results should land in a paper I am
happy to be cited (see `CITATION.cff`, or "Cite this repository" on GitHub)!

© 2026 Kolja Reuter · kolja@reuter-post.de
