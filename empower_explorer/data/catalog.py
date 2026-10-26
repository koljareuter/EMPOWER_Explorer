"""The galaxy catalogue of one reduction: which galaxies, which bands, redshifts, tables, maps.

    cat = Catalog.build(root, variant, progress=cb)     # slow part (header scan) - run in a worker
    cat.galaxies           DataFrame, index = galaxy key 'FIELD tid', one row per galaxy (see below)
    cat.cubes[key][band]   scan infos of the cubes of that galaxy and band
    cat.redshift(key)      Redshift(z, z_type, quality, source, ...) or Redshift(z=None)
    cat.products[key]      [MapProduct] resolved maps of the notebooks (may be missing)
    cat.tables[name]       raw CSV tables ('properties', 'bpt', 'census', ...)

galaxies columns: field, tid, name, ra, dec, z, z_type, z_quality, z_source, bands ('IZ+YJ'),
n_bands, texp_h, texp_<BAND>_h, n_cubes, telluric, has_maps, + the per-galaxy table columns
(snr_ha, logM, logSFR, N2, O3, S2, EW_ha, sigma_ha, class_NII_BPT, whan, av, ...).
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from empower_explorer import engine
from empower_explorer.data.products import discover_products
from empower_explorer.data.scan import scan_variant
from empower_explorer.data.tables import galaxy_table, load_tables

BANDS_ORDER = ['IZ', 'YJ', 'H', 'HK', 'K']


def band_rank(b):
    return BANDS_ORDER.index(b) if b in BANDS_ORDER else len(BANDS_ORDER)


@dataclass
class Redshift:
    z: float | None
    z_type: str = ''
    quality: float = np.nan
    source: str = ''              # catalogue the z comes from
    target_list: str = ''         # target list it was found under
    public_z: float = np.nan
    public_type: str = ''
    public_source: str = ''
    separation: float = np.nan    # arcsec, KMOS target - catalogue source
    override: bool = False        # entered by hand in the app

    @property
    def known(self):
        return self.z is not None and np.isfinite(self.z) and self.z > 0


class Catalog:
    def __init__(self, root, variant, infos, problems):
        engine.require()
        self.root, self.variant = root, variant
        self.infos, self.problems = infos, problems
        self.cubes = {}
        for i in infos:
            self.cubes.setdefault(i['key'], {}).setdefault(i['band'], []).append(i)
        zpath = os.path.join(root, 'scripts', 'target_redshifts.csv')
        self.zcat = engine.ks.RedshiftCatalog(zpath)
        try:
            self.zcsv = pd.read_csv(zpath, dtype={'arm_target_id': str})
            self.zcsv['region'] = [engine.ks.canonical_field(f) for f in self.zcsv['field'].astype(str)]
        except Exception:
            self.zcsv = pd.DataFrame()
        self.tables = load_tables(root)
        self.products = discover_products(root, variant)
        self.overrides = {}          # key -> z typed in by hand (session only)
        self.galaxies = self._build_galaxies()

    @classmethod
    def build(cls, root, variant, progress=None):
        infos, problems = scan_variant(root, variant, progress=progress)
        return cls(root, variant, infos, problems)

    # ------------------------------------------------------------------ redshifts
    def _zrow(self, field, tid):
        if self.zcsv.empty:
            return None
        sel = self.zcsv[(self.zcsv['region'] == field) & (self.zcsv['arm_target_id'].astype(str).str.strip() == str(tid))]
        return None if sel.empty else sel.iloc[0]

    def redshift(self, key):
        """Redshift of a galaxy: hand-entered override, else target_redshifts.csv (as kmos_sample)."""
        if key in self.overrides:
            z = self.overrides[key]
            return Redshift(z=z, z_type='manual', source='entered in the app', override=True)
        ents = [i for band in self.cubes.get(key, {}).values() for i in band]
        for e in ents:
            hit = self.zcat.lookup(e['target_list'], e['tid'])
            if hit is not None:
                z, ztype, q = hit
                r = Redshift(z=float(z), z_type=ztype, quality=q, target_list=e['target_list'])
                row = self._zrow(e['field'], e['tid'])
                if row is not None:
                    r.source = str(row.get('source_catalog', '') or '')
                    r.public_z = pd.to_numeric(row.get('public_redshift'), errors='coerce')
                    r.public_type = str(row.get('public_redshift_type', '') or '')
                    r.public_source = str(row.get('public_source', '') or '')
                    r.separation = pd.to_numeric(row.get('separation_arcsec'), errors='coerce')
                return r
        return Redshift(z=None)

    def set_override(self, key, z):
        """z typed in by hand (None or <= 0 removes the override and restores the catalogue)."""
        if z is None or not np.isfinite(z) or z <= 0:
            self.overrides.pop(key, None)
        else:
            self.overrides[key] = float(z)
        if key in self.galaxies.index:
            self.galaxies.loc[key, 'z'] = self.redshift(key).z if self.redshift(key).known else np.nan

    # ------------------------------------------------------------------ galaxy table
    def _build_galaxies(self):
        rows = []
        for key, bands in self.cubes.items():
            ents = [i for b in bands.values() for i in b]
            field, tid = ents[0]['field'], ents[0]['tid']
            zr = self.redshift(key)
            row = self._zrow(field, tid)
            r = dict(key=key, field=field, tid=tid,
                     ra=pd.to_numeric(row.get('ra_deg'), errors='coerce') if row is not None else np.nan,
                     dec=pd.to_numeric(row.get('dec_deg'), errors='coerce') if row is not None else np.nan,
                     z=zr.z if zr.known else np.nan, z_type=zr.z_type, z_quality=zr.quality, z_source=zr.source,
                     bands='+'.join(sorted(bands, key=band_rank)), n_bands=len(bands),
                     n_cubes=len(ents), telluric=all(e.get('telluric') for e in ents),
                     has_maps=key in self.products,
                     texp_h=np.nansum([e['exptime_s'] for e in ents]) / 3600.0)
            for b, es in bands.items():
                r[f'texp_{b}_h'] = np.nansum([e['exptime_s'] for e in es]) / 3600.0
            rows.append(r)
        if not rows:            # no readable cube: an empty table with the expected columns
            return pd.DataFrame(columns=['field', 'tid', 'name', 'ra', 'dec', 'z', 'z_type', 'z_quality',
                                         'z_source', 'bands', 'n_bands', 'n_cubes', 'telluric', 'has_maps',
                                         'texp_h']).rename_axis('key')
        df = pd.DataFrame(rows).set_index('key')
        props = galaxy_table(self.tables)
        if not df.empty and not props.empty:
            extra = props[[c for c in props.columns if c not in df.columns]]
            df = df.join(extra, how='left')
            if 'ra' in props.columns:      # coordinates of galaxies missing in target_redshifts.csv
                df['ra'] = df['ra'].fillna(pd.to_numeric(props['ra'], errors='coerce').reindex(df.index))
                df['dec'] = df['dec'].fillna(pd.to_numeric(props['dec'], errors='coerce').reindex(df.index))
        if 'name' not in df.columns:
            df['name'] = ''
        df['name'] = df['name'].fillna('')
        return df.sort_values(['field', 'tid'], key=lambda s: s.map(_natural) if s.name == 'tid' else s)

    def keys(self):
        return list(self.galaxies.index)

    def bands_of(self, key):
        return sorted(self.cubes.get(key, {}), key=band_rank)


def _natural(v):
    s = str(v)
    return (0, int(s), '') if s.isdigit() else (1, 0, s)
