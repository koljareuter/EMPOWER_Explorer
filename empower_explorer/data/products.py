"""Resolved map products of the EMPOWER notebooks (spaxel fits, Voronoi maps).

Found automatically in the EMPOWER folder:
  kmos_best3_maps_<FIELD>_<tid>.fits     per-spaxel single-Gaussian fits (<LINE>_FLUX/_FLUX_ERR/_SNR/
                                         _DV_KMS/_SIGMA_INT_KMS/_FORCED, CONT_HA), header BAND, SPXBIN
  kmos_best3_voronoi_<FIELD>_<tid>.fits  Voronoi bins: BINMAP + HA_FLUX, V, SIGMA, SIGMA_CORR, EW_HA,
                                         LOG_N2, OH_M13, LOG_S2HA, NE_SII, LOG_O3HA, ...
  kmos_<name>_maps.fits                  other resolved products (e.g. kmos_xmm2215_31179_maps.fits)
  kmos_voronoi_maps/<reduction>/<FIELD>_<tid>_voronoi.fits
                                         empower_explorer.pipeline.voronoi_binning products (schema
                                         EMPOWER-VORONOI-2) of the loaded reduction, for every galaxy:
                                         Voronoi bins, 1-3 Gaussian components per line group, ratios;
                                         tables BINS / TOTAL with all parameters, the noise model, config
The galaxy is read from the OBJECT keyword ('CDFS 23420').  The maps are on the pixel grid of the
stacked cube of their band in kmos_comb_ohlin_tell (same stacking as data/stack.py), scaled by SPXBIN.
"""
import glob
import os
import re
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits

from empower_explorer.lines import LINE_LABELS

_LINE_TOKENS = {'HA': 'Ha', 'HB': 'Hb', 'OIII5007': 'OIII5007', 'OIII4959': 'OIII4959',
                'NII6583': 'NII6583', 'NII6548': 'NII6548', 'SII6716': 'SII6716', 'SII6731': 'SII6731',
                'O3': 'OIII5007'}


@dataclass
class MapProduct:
    key: str                    # galaxy key, 'FIELD tid'
    kind: str                   # 'Spaxel fits', 'Voronoi', 'Resolved maps'
    path: str
    names: list                 # extension names with 2-D maps
    band: str = ''
    spxbin: int = 1
    meta: dict = field(default_factory=dict)
    _maps: dict = field(default_factory=dict, repr=False)

    @property
    def label(self):
        return f'{self.kind} ({os.path.basename(self.path)})'

    @property
    def is_pipeline(self):
        """Made by the Voronoi pipeline (fluxes already in BUNIT x micron, table BINS)."""
        return str(self.meta.get('SCHEMA', '')).startswith('EMPOWER-VORONOI')

    @property
    def schema(self):
        return str(self.meta.get('SCHEMA', '') or '').strip()

    def bins_table(self):
        """pandas DataFrame of the BINS table (pipeline products), else None."""
        if not self.is_pipeline:
            return None
        if '_bins' not in self.__dict__:
            from astropy.table import Table
            try:
                with fits.open(self.path) as h:
                    t = Table.read(h['BINS']).to_pandas() if 'BINS' in h else None
                if t is not None:                   # FITS strings may come back as bytes
                    for c in t.columns:
                        if t[c].dtype == object:
                            t[c] = [v.decode().strip() if isinstance(v, bytes) else v for v in t[c]]
                self.__dict__['_bins'] = t
            except Exception:
                self.__dict__['_bins'] = None
        return self.__dict__['_bins']

    def load(self):
        """{name: 2-D float array}; BINMAP / KINBINMAP kept as int arrays.  Line fluxes are converted
        to BUNIT x micron like every other flux in the app (the notebooks fit on an Angstrom grid,
        so their *_FLUX / *_FLUX_ERR maps are in BUNIT x Angstrom = 1e4 x larger)."""
        if not self._maps:
            with fits.open(self.path) as h:
                for e in h[1:]:
                    if e.data is not None and np.ndim(e.data) == 2:
                        a = np.array(e.data)
                        if is_line_flux_map(e.name) and not self.is_pipeline:
                            a = a.astype(float) / 1e4
                        self._maps[e.name] = a
        return self._maps

    @property
    def bunit(self):
        return str(self.meta.get('BUNIT', '') or 'ADU/s').strip()

    def map(self, name):
        return self.load()[name]

    @property
    def binmap(self):
        m = self.load()
        return m.get('BINMAP')


PIPELINE_DIR = 'kmos_voronoi_maps'


def pipeline_dir(root, variant):
    """Folder of the Voronoi-pipeline products of one reduction."""
    return os.path.join(root, PIPELINE_DIR, variant)


def discover_products(root, variant=None):
    """{galaxy key: [MapProduct]} for all map products in the EMPOWER folder (headers only):
    the notebook products and, for `variant`, the pipeline products (listed first)."""
    out = {}
    pats = ['kmos_*maps*.fits', 'kmos_*voronoi*.fits']
    paths = sorted({p for pat in pats for p in glob.glob(os.path.join(root, pat))})
    if variant:
        paths = sorted(glob.glob(os.path.join(pipeline_dir(root, variant), '*_voronoi.fits'))) + paths
    for p in paths:
        try:
            with fits.open(p) as h:
                h0 = h[0].header
                names = [e.name for e in h[1:] if e.header.get('NAXIS', 0) == 2]
                meta = {k: h0[k] for k in h0 if k and isinstance(h0[k], (str, int, float, bool))}
        except Exception:
            continue
        if not names:
            continue
        key = str(h0.get('OBJECT') or '').strip()
        if not key:
            m = re.search(r'_(COSMOS|CDFS|SPT2040|XMM2215)_(\w+)\.fits$', os.path.basename(p), re.I)
            if not m:
                continue
            key = f'{m.group(1).upper()} {m.group(2)}'
        base = os.path.basename(p).lower()
        if str(h0.get('SCHEMA', '')).startswith('EMPOWER-VORONOI'):
            kind, band = 'Voronoi fit', str(h0.get('ANCHBAND') or '').strip()
        else:
            kind = 'Voronoi' if 'voronoi' in base else ('Spaxel fits' if 'best3_maps' in base else 'Resolved maps')
            band = str(h0.get('BAND') or '').strip()
        out.setdefault(key, []).append(MapProduct(
            key=key, kind=kind, path=p, names=names, band=band,
            spxbin=int(h0.get('SPXBIN', 1) or 1), meta=meta))
    return out


def is_line_flux_map(name):
    """*_FLUX / *_FLUX_ERR maps of a line (not continuum, not EW)."""
    n = name.upper()
    return bool(re.search(r'(^|_)FLUX(_ERR)?$', n)) and not n.startswith('CONT')


def pretty_map_name(name, bunit=None):
    """'HA_SIGMA_INT_KMS' -> 'Hα σ_int [km/s]', 'LOG_N2' -> 'log [NII]/Hα', ... (best effort);
    line-flux maps get the unit '[bunit·µm]' when bunit is given (see MapProduct.load)."""
    label = _pretty(name)
    if label.endswith(']'):                 # unit already in the label
        return label
    if bunit and is_line_flux_map(name):
        label += f' [{bunit}·µm]'
    return label


# Voronoi-pipeline names (schema EMPOWER-VORONOI-2): group tags and line tags
_GROUP_LABELS = {'HANII': 'Hα+[NII]', 'HBETA': 'Hβ', 'OIII': '[OIII]', 'SII': '[SII]'}
_PIPE_LINES = {'HA': 'Ha', 'HB': 'Hb', 'OIII5007': 'OIII5007', 'OIII4959': 'OIII4959', 'NII6583': 'NII6583',
               'NII6548': 'NII6548', 'SII6716': 'SII6716', 'SII6731': 'SII6731'}
_GROUP_RE = '|'.join(_GROUP_LABELS)
_LINE_RE = '|'.join(sorted(_PIPE_LINES, key=len, reverse=True))


def _pretty_pipeline(n):
    """Names of Voronoi-pipeline products, None if `n` is not one of them."""
    special = {'SN_LINE': 'Anchor-line S/N per spaxel (binning input)', 'LOG_N2': 'log [NII]/Hα',
               'LOG_O3': 'log [OIII]/Hβ', 'LOG_S2': 'log [SII]/Hα', 'LOG_O3N2': 'O3N2',
               'SII_RATIO': '[SII] 6716/6731', 'NE_SII': 'n_e([SII]) [cm⁻³]', 'OH_N2_M13': '12+log(O/H) N2 (M13)',
               'EW_HA': 'EW(Hα) rest [Å]'}
    if n in special:
        return special[n]
    m = re.match(rf'^({_GROUP_RE})_(NCOMP|V50|W80|V10|V90|CHI2NU|FORCED|QFLAG|C(\d)_(V|SIG|FRAC))$', n)
    if m:
        g = _GROUP_LABELS[m.group(1)]
        what = {'NCOMP': 'number of components', 'V50': 'v50 (median velocity) [km/s]',
                'W80': 'W80 [km/s]', 'V10': 'v10 [km/s]', 'V90': 'v90 [km/s]', 'CHI2NU': 'reduced χ²',
                'FORCED': 'forced to the anchor kinematics (1 = yes)', 'QFLAG': 'quality flags (bits)'}
        if m.group(3):
            return f'{g} component {m.group(3)} ' + {'V': 'velocity [km/s]', 'SIG': 'σ_int [km/s]',
                                                     'FRAC': 'flux fraction'}[m.group(4)]
        return f'{g} {what[m.group(2)]}'
    m = re.match(rf'^({_LINE_RE})_C(\d)_FLUX$', n)
    if m:
        return f'{LINE_LABELS.get(_PIPE_LINES[m.group(1)], m.group(1))} component {m.group(2)} flux'
    m = re.match(rf'^({_LINE_RE})_FLUX_CGS$', n)
    if m:
        return f'{LINE_LABELS.get(_PIPE_LINES[m.group(1)], m.group(1))} flux [10⁻¹⁷ erg/s/cm²]'
    return None


def _pretty(name):
    n = name.upper()
    pipe = _pretty_pipeline(n)
    if pipe is not None:
        return pipe
    special = {
        'BINMAP': 'Voronoi bin number', 'KINBINMAP': 'Kinematic bins', 'V': 'Velocity [km/s]',
        'V_ERR': 'Velocity error [km/s]', 'V_MODEL': 'Velocity model [km/s]', 'SIGMA': 'σ [km/s]',
        'SIGMA_CORR': 'σ beam-smearing corrected [km/s]', 'EW_HA': 'EW(Hα) [Å]', 'LOG_N2': 'log [NII]/Hα',
        'OH_M13': '12+log(O/H) (M13)', 'LOG_S2HA': 'log [SII]/Hα', 'NE_SII': 'n_e([SII]) [cm⁻³]',
        'LOG_O3HA': 'log [OIII]/Hα', 'CONT_HA': 'Continuum near Hα', 'CONT_H': 'Continuum',
        'BIN_N2': 'log [NII]/Hα', 'BIN_O3': 'log [OIII]/Hβ', 'BIN_S2': 'log [SII]/Hα',
        'BIN_O3N2': 'O3N2', 'BIN_OH_N2_M13': '12+log(O/H) N2 (M13)', 'BIN_OH_O3N2_M13': '12+log(O/H) O3N2 (M13)',
        'BIN_HA_DV': 'Hα Δv [km/s]', 'BIN_HA_SIG': 'Hα σ [km/s]', 'BIN_O3_DV': '[OIII] Δv [km/s]',
        'BIN_O3_SIG': '[OIII] σ [km/s]', 'BIN_CLASS_N2': 'BPT class ([NII])', 'KIN_V_HA': 'Velocity Hα [km/s]',
        'KIN_V_MODEL': 'Velocity model [km/s]', 'KIN_V_RESID': 'Velocity residual [km/s]',
        'KIN_SIG_HA': 'σ Hα [km/s]', 'KIN_SIG_BEAM': 'σ beam smearing [km/s]',
        'KIN_SIG_CORR': 'σ corrected [km/s]', 'KIN_EW_HA': 'EW(Hα) [Å]',
    }
    if n in special:
        return special[n]
    prefix = ''
    for p in ('SPAX_', 'BIN_'):
        if n.startswith(p):
            prefix, n = ('spaxel ' if p == 'SPAX_' else 'bin '), n[len(p):]
    for tok, key in sorted(_LINE_TOKENS.items(), key=lambda kv: -len(kv[0])):
        if n.startswith(tok + '_'):
            rest = n[len(tok) + 1:]
            what = {'FLUX': 'flux', 'FLUX_ERR': 'flux error', 'SNR': 'S/N', 'DV_KMS': 'Δv [km/s]',
                    'SIGMA_INT_KMS': 'σ_int [km/s]', 'FORCED': 'forced fit (1 = yes)', 'SIG': 'σ [km/s]',
                    'DV': 'Δv [km/s]'}.get(rest, rest.lower())
            return f'{prefix}{LINE_LABELS.get(key, key)} {what}'.strip()
    return (prefix + name).strip()


def is_velocity_map(name):
    n = name.upper()
    return (n in ('V', 'V_MODEL', 'KIN_V_HA', 'KIN_V_MODEL', 'KIN_V_RESID', 'DV_NII_HA') or '_DV' in n
            or n.endswith('_SEP_V50')
            or bool(re.match(rf'^({_GROUP_RE})_(V50|V10|V90|C\d_V)$', n)))


def is_integer_map(name):
    n = name.upper()
    return (n in ('BINMAP', 'KINBINMAP', 'BIN_CLASS_N2') or n.endswith('_FORCED') or n.endswith('_NCOMP')
            or n.endswith('_QFLAG'))
