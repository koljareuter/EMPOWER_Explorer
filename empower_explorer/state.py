"""AppState - the one shared selection every view observes (Qt signals).

Selection chain:  reduction variant -> galaxy (key 'FIELD tid') -> band stacks -> aperture.

Signals
    catalogLoading(str variant)     a new reduction is being scanned
    catalogChanged()                state.catalog is ready (galaxy list, tables, products)
    catalogFailed(str variant)      that scan failed; state.variant/catalog are unchanged (views that
                                    showed 'Scanning …' get catalogChanged again for the old catalog)
    galaxyChanged(str key)          a galaxy was selected; its stacks may still be loading
    stacksReady(str key)            state.stacks (band -> BandStack) of that galaxy are loaded
    loadFailed(str key)             the cubes of that galaxy could not be loaded (state.stacks empty)
    apertureChanged()               aperture position/radius or the reference band changed
    redshiftChanged()               the redshift of the current galaxy changed (override)
    optionsChanged(str name)        a shared display option changed (see OPTIONS)
    status(str text, int ms)        message for the status bar
    busy(str text, int done, int total)   progress; total == 0 means idle
    error(str title, str details)

Typical use in a view
    state.stacksReady.connect(self.rebuild)      # new galaxy -> rebuild plots
    state.apertureChanged.connect(self.refresh)  # aperture moved -> new spectra
    spectra = state.spectra()                     # {band: analysis.Spectrum}, cached
    fits, info = state.fits()                     # {line: analysis.LineFit}, cached
"""
from collections import OrderedDict

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from empower_explorer import analysis, config, engine, lines
from empower_explorer.data.catalog import Catalog, band_rank
from empower_explorer.data.stack import load_band_stack
from empower_explorer.workers import run_async

OPTIONS = {
    'mode': 'snr',            # 'snr' or 'flux' for the spectrum y axis
    'method': 'sum',          # aperture spectrum: 'sum' (total) or 'mean' (inverse-variance weighted)
    'subtract_bg': True,      # subtract the residual sky pedestal (median of outer spaxels), as the notebooks
    'telluric': True,         # shade low atmospheric transmission
    'sky_lines': False,       # mark bright OH sky lines
    'rest_frame': False,      # spectrum x axis in rest-frame wavelength
    'smooth': 0,              # boxcar smoothing of displayed spectra [px], 0 = off
}

MAX_CACHED_GALAXIES = 6


class AppState(QObject):
    catalogLoading = pyqtSignal(str)
    catalogChanged = pyqtSignal()
    catalogFailed = pyqtSignal(str)        # variant whose scan failed; state.catalog/variant unchanged
    galaxyChanged = pyqtSignal(str)
    stacksReady = pyqtSignal(str)
    loadFailed = pyqtSignal(str)           # galaxy whose cubes could not be loaded
    apertureChanged = pyqtSignal()
    redshiftChanged = pyqtSignal()
    optionsChanged = pyqtSignal(str)
    sepFitsChanged = pyqtSignal(object)    # product token whose separate Halpha/[NII] fit started / finished
    productsChanged = pyqtSignal(str)      # galaxy whose map products changed (a new Voronoi fit)
    status = pyqtSignal(str, int)
    busy = pyqtSignal(str, int, int)
    error = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.root = None
        self.variant = None
        self.catalog = None
        self.key = None
        self.stacks = {}
        self.ref_band = None
        self.ap_x = self.ap_y = None
        self.ap_r = 2.5
        self.ap_rin = 0.0                      # inner radius: > 0 makes the aperture the annulus ap_rin < d <= ap_r
        self.sep_fits = {}                     # product token -> {bin: vb.fit_hanii_separate result} (Maps, Radial)
        self.sep_running = set()               # product tokens whose separate fit runs now
        self._re_cache = {}
        self.options = dict(OPTIONS)
        self._stack_cache = OrderedDict()     # key -> {band: BandStack}
        self._spec_cache = {}
        self._fit_cache = {}
        self._pending_key = None
        self._job = None
        self._scan_job = None
        self._loading_key = None               # galaxy whose cubes are being read right now
        self.loading_variant = None
        self.z_overrides = {}                  # key -> z typed in by hand (kept across reductions)

    # ------------------------------------------------------------------ setup
    def set_root(self, root):
        """Use the EMPOWER data folder `root` - raises engine.EngineError."""
        engine.load(root)
        lines.sync(engine.REST_UM)
        self.root = root

    def variants(self):
        return config.available_variants(self.root)

    def load_variant(self, variant):
        """Scan a reduction in the background.  state.variant / state.catalog switch only when the scan
        succeeded (catalogChanged); on failure catalogFailed(variant) is emitted and nothing changes."""
        if self.root is None:
            return
        if self._scan_job is not None:          # a scan of another reduction is superseded
            self._scan_job.cancelled = True
        self.loading_variant = variant
        self.catalogLoading.emit(variant)
        self.busy.emit(f'Scanning {variant} …', 0, 1)
        keep_key = self.key

        def done(cat):
            if variant != self.loading_variant:
                return
            self.loading_variant = None
            self.busy.emit('', 0, 0)
            if cat.galaxies.empty:
                why = cat.problems[0] if cat.problems else 'no readable DATACUBE_COMBINED files'
                self.error.emit(f'No combined cubes in {variant}',
                                f'{len(cat.problems)} problem(s), e.g.: {why}')
                self._restore_after_failed_scan(variant)
                return
            for k, z in self.z_overrides.items():      # typed redshifts survive a reduction switch
                if k in cat.galaxies.index:
                    cat.set_override(k, z)
            self.variant = variant
            self.catalog = cat
            self._stack_cache.clear()
            self._spec_cache.clear()
            self._fit_cache.clear()
            n = len(cat.galaxies)
            self.status.emit(f'{variant}: {n} galaxies, {len(cat.infos)} cubes'
                             + (f' · {cat.problems[0]}' if cat.problems else ''), 8000)
            self.catalogChanged.emit()
            if keep_key in cat.galaxies.index:
                self.select_galaxy(keep_key, force=True)
            elif n:
                self.select_galaxy(self.default_galaxy(), force=True)

        def fail(msg):
            if variant != self.loading_variant:
                return
            self.loading_variant = None
            self.busy.emit('', 0, 0)
            self.error.emit(f'Could not scan {variant}', msg)
            self._restore_after_failed_scan(variant)

        def progress(d, t):
            if variant == self.loading_variant:
                self.busy.emit(f'Reading headers of {variant} …', d, t)

        self._scan_job = run_async(Catalog.build, self.root, variant, on_done=done, on_error=fail,
                                   on_progress=progress)

    def _restore_after_failed_scan(self, variant):
        """Views showed 'Scanning …' for `variant`: bring them back to the reduction still loaded."""
        self.catalogFailed.emit(variant)
        if self.catalog is None:
            return
        self.catalogChanged.emit()
        if self.key is not None and self.stacks:
            self.galaxyChanged.emit(self.key)
            self.stacksReady.emit(self.key)
            self.apertureChanged.emit()

    def default_galaxy(self):
        g = self.catalog.galaxies
        if 'snr_ha' in g.columns:
            s = np.nan_to_num(g['snr_ha'].astype(float).to_numpy(), nan=-1)
            return g.index[int(np.argmax(s))]
        return g.index[0]

    # ------------------------------------------------------------------ galaxy selection
    def select_galaxy(self, key, force=False):
        if self.catalog is None or key not in self.catalog.galaxies.index:
            return
        if key == self.key and not force and self.stacks:
            return
        self.key = key
        self._pending_key = key
        self.stacks = {}
        self.galaxyChanged.emit(key)
        if key in self._stack_cache:
            if self._job is not None:          # a load of another galaxy may still be running
                self._job.cancelled = True
            self._loading_key = None
            self.busy.emit('', 0, 0)
            self._stack_cache.move_to_end(key)
            self._activate(key, self._stack_cache[key])
            return
        bands = self.catalog.cubes.get(key, {})
        self.busy.emit(f'Loading {key} …', 0, len(bands))

        def load(progress=None):
            out = {}
            for i, b in enumerate(sorted(bands, key=band_rank)):
                out[b] = load_band_stack(bands[b])
                if progress:
                    progress(i + 1, len(bands))
            return key, out

        def done(res):
            k, stacks = res
            if k == self._loading_key:
                self._loading_key = None
            self._stack_cache[k] = stacks
            while len(self._stack_cache) > MAX_CACHED_GALAXIES:
                self._stack_cache.popitem(last=False)
            if k == self._pending_key:
                self.busy.emit('', 0, 0)
                self._activate(k, stacks)

        def fail(msg):
            if key == self._loading_key:
                self._loading_key = None
            if key != self._pending_key:        # the user already moved on
                return
            self.busy.emit('', 0, 0)
            self.error.emit(f'Could not load {key}', msg)
            self.loadFailed.emit(key)

        if self._job is not None:
            self._job.cancelled = True
        self._loading_key = key
        self._job = run_async(load, on_done=done, on_error=fail,
                              on_progress=lambda d, t: self.busy.emit(f'Loading {key} …', d, t))

    @property
    def is_loading(self):
        """True while the cubes of a galaxy are being read."""
        return self._loading_key is not None

    def _activate(self, key, stacks):
        self.stacks = stacks
        self._spec_cache.clear()
        self._fit_cache.clear()
        z = self.z
        ha = [b for b, s in stacks.items() if z and s.covers(lines.REST['Ha'] * (1 + z))]
        self.ref_band = max(ha or list(stacks), key=lambda b: stacks[b].exptime_s if np.isfinite(stacks[b].exptime_s) else 0)
        self.ap_x, self.ap_y = self._auto_center()
        self.stacksReady.emit(key)
        self.apertureChanged.emit()

    def _auto_center(self):
        st = self.stacks[self.ref_band]
        z = self.z
        for k in ('Ha', 'OIII5007', 'Hb'):
            if z and st.covers(lines.REST[k] * (1 + z)):
                x, y, found = analysis.line_peak_center(st, z, k)
                if found:
                    self.status.emit(f'Aperture centred on the {lines.LINE_LABELS[k]} peak', 4000)
                return x, y
        return analysis.default_center(st)

    def center_on_line(self, key='Ha'):
        """Move the aperture to the peak of a line (in the band that covers it)."""
        z = self.z
        if not self.stacks or not z:
            return False
        band = analysis.pick_band(self.stacks, lines.REST[key] * (1 + z))
        if band is None:
            self.status.emit(f'{lines.LINE_LABELS[key]} is not covered by the loaded bands', 4000)
            return False
        x, y, found = analysis.line_peak_center(self.stacks[band], z, key)
        if not found:
            self.status.emit(f'No clear {lines.LINE_LABELS[key]} peak - aperture left where it was', 4000)
            return False
        x, y, _ = analysis.transfer(self.stacks[band], self.stacks[self.ref_band], x, y)
        self.set_aperture(x, y)
        self.status.emit(f'Aperture centred on the {lines.LINE_LABELS[key]} peak ({band})', 4000)
        return True

    # ------------------------------------------------------------------ redshift
    @property
    def redshift(self):
        """analysis Redshift object of the current galaxy (z may be None)."""
        if self.catalog is None or self.key is None:
            return None
        return self.catalog.redshift(self.key)

    @property
    def z(self):
        r = self.redshift
        return r.z if (r is not None and r.known) else None

    def set_redshift(self, z):
        """Override the redshift of the current galaxy (None / 0 restores the catalogue value)."""
        if self.catalog is None or self.key is None:
            return
        if z is None or not np.isfinite(z) or z <= 0:
            self.z_overrides.pop(self.key, None)
        else:
            self.z_overrides[self.key] = float(z)
        self.catalog.set_override(self.key, z)
        self._fit_cache.clear()
        self.redshiftChanged.emit()

    # ------------------------------------------------------------------ aperture
    def set_aperture(self, x=None, y=None, r=None, r_in=None):
        """Move or resize the aperture (pixels of the reference band).  r_in > 0 makes it an annulus;
        the ring keeps at least 0.5 px width (a smaller outer radius pushes the inner one in)."""
        if x is not None:
            self.ap_x = float(x)
        if y is not None:
            self.ap_y = float(y)
        if r is not None:
            self.ap_r = float(max(0.5, r))
        if r_in is not None:
            self.ap_rin = float(max(0.0, r_in))
            if self.ap_rin > 0 and self.ap_r < self.ap_rin + 0.5:
                self.ap_r = self.ap_rin + 0.5
        elif self.ap_rin > 0 and self.ap_r < self.ap_rin + 0.5:
            self.ap_rin = max(0.0, self.ap_r - 0.5)
        self.apertureChanged.emit()

    def set_ref_band(self, band):
        """Band whose pixel grid the aperture lives on (shown in the image); keeps the sky position."""
        if band not in self.stacks or band == self.ref_band:
            return
        x, y, r = analysis.transfer(self.stacks[self.ref_band], self.stacks[band], self.ap_x, self.ap_y, self.ap_r)
        self.ap_rin *= r / self.ap_r
        self.ref_band = band
        self.ap_x, self.ap_y, self.ap_r = x, y, r
        self.apertureChanged.emit()

    def aperture_in(self, band):
        """(x, y, r) of the aperture on the grid of `band` (same sky position)."""
        return analysis.transfer(self.stacks[self.ref_band], self.stacks[band], self.ap_x, self.ap_y, self.ap_r)

    def annulus_in(self, band):
        """(x, y, r, r_in) of the aperture on the grid of `band`; r_in = 0 for a full circle."""
        x, y, r = self.aperture_in(band)
        return x, y, r, self.ap_rin * r / self.ap_r

    def effective_radius(self):
        """(R_e [arcsec], source) of the current galaxy - the anchor-line (Halpha, else [OIII]) half-light
        radius of the notebooks (kmos_snr_distribution.csv: re_px x 0.2", measured or the sample median),
        else measured here by the same curve of growth on the map of the first of Halpha, [OIII]5007, Hbeta
        that is covered, around its peak (>= 1 px); (nan, '') if neither works."""
        z = self.z
        ck = (self.variant, self.key, None if z is None else round(z, 6))
        if ck in self._re_cache:
            return self._re_cache[ck]
        out = (np.nan, '')
        tab = self.catalog.tables.get('snr_distribution') if self.catalog is not None else None
        if tab is not None and 'key' in tab.columns and 're_px' in tab.columns:
            rows = tab[tab['key'] == self.key]
            re_px = rows['re_px'].dropna()
            if len(re_px):
                src = str(rows['re_src'].dropna().iloc[0]) if 're_src' in rows and rows['re_src'].notna().any() else ''
                out = (float(re_px.iloc[0]) * 0.2, {'measured': 'measured in the notebooks',
                                                      'median': 'sample median of the notebooks'}.get(src, 'notebooks'))
        if not np.isfinite(out[0]) and self.stacks and z:
            for line in ('Ha', 'OIII5007', 'Hb'):
                band = analysis.pick_band(self.stacks, lines.REST[line] * (1 + z))
                if band is None:
                    continue
                st = self.stacks[band]
                x, y, found = analysis.line_peak_center(st, z, line)
                if found:
                    r = analysis.half_light_radius(analysis.line_map(st, z, line), x, y)
                    if np.isfinite(r) and r >= 1.0:
                        out = (r * st.pixscale, f'{lines.LINE_LABELS[line]} curve of growth')
                break                         # the anchor line decides, as in the notebooks
        self._re_cache[ck] = out
        return out

    # ------------------------------------------------------------------ Halpha and [NII] separately
    def start_separate_fit(self, token, label, job, *args):
        """Run job(*args) -> {bin: result} (vb.hanii_separate) once per product token, for the Maps and the
        Radial profiles tab alike; the result lands in sep_fits.  sepFitsChanged(token) when it starts and ends.
        An empty result (no bin with Halpha+[NII]) is reported, not stored, so it can be tried again."""
        if token in self.sep_running or token in self.sep_fits:
            return
        self.sep_running.add(token)
        self.sepFitsChanged.emit(token)

        def done(res, token=token):
            self.sep_running.discard(token)
            if res:
                self.sep_fits[token] = res
                n_n = sum(np.isfinite(v.get('NII_V50', np.nan)) for v in res.values())
                self.status.emit(f'{label}: Hα and [NII] fitted separately in {len(res)} bins '
                                 f'([NII] kinematics in {n_n})', 6000)
            else:
                self.status.emit(f'{label}: no bin with Hα+[NII] could be fitted separately', 6000)
            self.sepFitsChanged.emit(token)

        def fail(msg, token=token):
            self.sep_running.discard(token)
            self.error.emit('Hα / [NII] separately', msg)
            self.sepFitsChanged.emit(token)

        run_async(job, *args, on_done=done, on_error=fail)

    # ------------------------------------------------------------------ options
    def set_option(self, name, value):
        if self.options.get(name) == value:
            return
        self.options[name] = value
        if name in ('method', 'subtract_bg'):
            self._spec_cache.clear()
            self._fit_cache.clear()
        self.optionsChanged.emit(name)

    # ------------------------------------------------------------------ spectra and fits (cached)
    def _ap_key(self):
        return (self.key, self.ref_band, round(self.ap_x or 0, 2), round(self.ap_y or 0, 2), round(self.ap_r, 6),
                round(self.ap_rin, 6), self.options['method'], self.options['subtract_bg'])

    def spectra(self):
        """{band: analysis.Spectrum} for the current aperture (same sky position in every band)."""
        if not self.stacks or self.ap_x is None:
            return {}
        k = self._ap_key()
        if k not in self._spec_cache:
            out = {}
            for b, st in self.stacks.items():
                x, y, r, r_in = self.annulus_in(b)
                sp = analysis.extract_spectrum(st, x, y, r, method=self.options['method'],
                                               subtract_bg=self.options['subtract_bg'], r_in=r_in)
                if sp.n_pix > 0:
                    out[b] = sp
            if len(self._spec_cache) > 50:
                self._spec_cache.clear()
            self._spec_cache[k] = out
        return self._spec_cache[k]

    def fits(self):
        """({line key: LineFit}, FitInfo) of the current aperture (all lines, deepest band each)."""
        z = self.z
        if z is None:
            return {}, analysis.FitInfo()
        k = self._ap_key() + (round(z, 6),)
        if k not in self._fit_cache:
            if len(self._fit_cache) > 50:
                self._fit_cache.clear()
            if self.ap_rin > 0:               # annulus: forced to the kinematics of the central aperture
                self._fit_cache[k] = analysis.fit_annulus(self.spectra(), self.stacks, z, self.ref_band, self.ap_x,
                                                          self.ap_y, self.options['method'], self.options['subtract_bg'])
            else:
                self._fit_cache[k] = analysis.fit_galaxy(self.spectra(), self.stacks, z)
        return self._fit_cache[k]

    def products(self):
        """[MapProduct] of the current galaxy (resolved maps of the notebooks)."""
        if self.catalog is None or self.key is None:
            return []
        return self.catalog.products.get(self.key, [])

    def galaxy_row(self):
        if self.catalog is None or self.key is None:
            return None
        return self.catalog.galaxies.loc[self.key]
