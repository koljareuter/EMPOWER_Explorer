"""
Shared sample handling for the KMOS notebooks.

One place for the things every notebook must do identically:

* galaxy identity   — a galaxy is (survey region, target id).  The same galaxy
  appears in several target lists (OBS TARG NAME, e.g. 'cosmos' and
  'cosmos_target_legac.cat', or 'SPT2040_Halpha' and 'SPT2040_Hbeta_OIII') and
  in several bands; it must be counted once.
* redshifts         — looked up by (target list, id), else by (region, id), so a
  target-list name missing from the CSV (e.g. 'SPT2040_Halpha') still finds z.
* exposure time     — on-source time of a combined cube = NCOMBINE × DIT × NDIT.
  The EXPTIME keyword is 1.5× too short: the combined object frames already
  exclude the sky frames, but the IDP header step scales by 2/3 once more
  (checked frame by frame for all 260 cubes, 2026-10-07).
* stacking          — all cubes of one galaxy in one band are stacked, aligned
  on the sky (cubes from different target lists can be 14×14 vs 16×16 px),
  weighted by on-source time.
* duplicates        — re-running the combination writes the same cube again into
  a new run folder; cubes built from the identical set of input frames are
  counted once (newest run kept), otherwise exposure times add up 3x
  (found 2026-10-14: 93/282 stacks, e.g. CDFS 23420 IZ 32 h instead of 10.7 h).

Usage
-----
    import sys; sys.path.insert(0, "scripts")
    import kmos_sample as ks
    zcat   = ks.RedshiftCatalog("scripts/target_redshifts.csv")
    stacks = ks.load_galaxy_band_stacks("kmos_proc_and_comb_server", zcat)
    # → one dict per (galaxy, band): gid, field, tid, band, z, cube, wave,
    #   exptime_s, n_frames, n_cubes, target_lists, folders, paths,
    #   teff, atm (telluric transmission; None for uncorrected cubes)
"""
from __future__ import annotations

import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

__all__ = ["canonical_field", "RedshiftCatalog", "onsource_seconds",
           "wavelength_axis", "cube_info", "scan_combined_cubes", "stack_cubes",
           "load_galaxy_band_stacks", "load_stack_with_stat", "galaxy_name",
           "BANDS_ORDER", "band_rank",
           "pick_band", "telluric_of", "atmospheric_transmission", "low_transmission_spans",
           "subtract_residual_background"]

BANDS_ORDER = ["IZ", "YJ", "H", "HK", "K", "??"]


def band_rank(b):
    return BANDS_ORDER.index(b) if b in BANDS_ORDER else len(BANDS_ORDER)


# ── galaxy identity ───────────────────────────────────────────────────────────
def canonical_field(raw):
    """Map any KMOS target-list name (OBS TARG NAME) to its survey region."""
    r = (raw or "").lower()
    if "spt" in r:
        return "SPT2040"
    if "xmm" in r:
        return "XMM2215"
    if "cdfs" in r or "ecdfs" in r or "goods" in r:
        return "CDFS"
    if "cosmos" in r:
        return "COSMOS"
    return raw


def galaxy_name(field, tid):
    return f"{field} {tid}"


# ── redshifts ─────────────────────────────────────────────────────────────────
class RedshiftCatalog:
    """target_redshifts.csv, indexed by (target list, id) and by (region, id)."""

    def __init__(self, csv_path):
        self.table = {}          # (target list, tid) -> (z, z_type, quality)
        self.by_galaxy = {}      # (region, tid)      -> (z, z_type, quality)
        self.conflicts = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                zs = (row.get("redshift") or "").strip()
                if not zs:
                    continue
                q = (row.get("quality") or "").strip()
                try:
                    q = float(q)
                except ValueError:
                    q = np.nan
                val = (float(zs), (row.get("redshift_type") or "").strip(), q)
                key = (row["field"].strip(), row["arm_target_id"].strip())
                self.table[key] = val
                gk = (canonical_field(key[0]), key[1])
                if gk in self.by_galaxy and abs(self.by_galaxy[gk][0] - val[0]) > 1e-3:
                    self.conflicts.append((gk, self.by_galaxy[gk][0], val[0]))
                self.by_galaxy.setdefault(gk, val)

    def lookup(self, target_list, tid):
        """(z, z_type, quality) or None."""
        k = (target_list, str(tid))
        if k in self.table:
            return self.table[k]
        return self.by_galaxy.get((canonical_field(target_list), str(tid)))

    def __len__(self):
        return len(self.by_galaxy)


# ── headers ───────────────────────────────────────────────────────────────────
def onsource_seconds(hdr0):
    """On-source integration of a (combined or single) KMOS cube:
    NCOMBINE × DIT × NDIT  (NCOMBINE = 1 for a single reconstructed frame).
    Never use EXPTIME of DATACUBE_COMBINED files (1.5× too short)."""
    dit = hdr0.get("HIERARCH ESO DET SEQ1 DIT", hdr0.get("ESO DET SEQ1 DIT"))
    ndit = hdr0.get("HIERARCH ESO DET NDIT", hdr0.get("ESO DET NDIT", 1)) or 1
    ncomb = hdr0.get("NCOMBINE", 1) or 1
    if dit is None:
        return np.nan
    return float(ncomb) * float(dit) * float(ndit)


def wavelength_axis(hdr):
    cdelt = hdr.get("CDELT3", hdr.get("CD3_3", 1.0))
    return (np.arange(hdr["NAXIS3"]) + 1 - hdr["CRPIX3"]) * cdelt + hdr["CRVAL3"]


# ── residual background ───────────────────────────────────────────────────────
def subtract_residual_background(cube, centre, r_min=6.0, min_spaxels=20):
    """cube minus, at every wavelength, the median of the spaxels >= r_min px from
    `centre` (galaxy position) - removes the additive sky/background pedestal that
    some combined cubes carry (IZ of COSMOS 92754: 29 % of the central spaxel, which
    made Halpha look weaker after median normalisation; EW 88 -> 124 A, YJ 122 A).
    Returns (cleaned cube, background spectrum); unchanged if too few spaxels."""
    cube = np.asarray(cube, float)
    yy, xx = np.indices(cube.shape[1:])
    far = (np.hypot(yy - centre[0], xx - centre[1]) >= r_min) & (np.isfinite(cube).mean(0) > 0.5)
    if far.sum() < min_spaxels:
        return cube, np.zeros(cube.shape[0])
    with np.errstate(all="ignore"):
        bg = np.nanmedian(cube[:, far], axis=1)
    bg = np.where(np.isfinite(bg), bg, 0.0)
    return cube - bg[:, None, None], bg


# ── telluric transmission of corrected cubes ──────────────────────────────────
def telluric_of(path):
    """T_eff (instrument response x atmosphere) that scripts/telluric_combined.py
    divided out of a DATACUBE_COMBINED file, read from the TELLEFF_ file next to
    it; None for an uncorrected cube."""
    p = os.path.join(os.path.dirname(path),
                     "TELLEFF_" + os.path.basename(path)[len("DATACUBE_COMBINED_"):])
    return np.asarray(fits.getdata(p), float) if os.path.exists(p) else None


def telluric_atm_of(path):
    """Atmosphere-only transmission A (~1 clear, < 0.5 in H2O/O2/CO2 bands) of a cube corrected with
    telluric_combined.py method 'best' (ATM extension of the TELLEFF file); None for older corrections."""
    p = os.path.join(os.path.dirname(path),
                     "TELLEFF_" + os.path.basename(path)[len("DATACUBE_COMBINED_"):])
    if not os.path.exists(p):
        return None
    with fits.open(p) as h:
        return np.asarray(h["ATM"].data, float) if "ATM" in h else None


def _member_atm(path, teff, wave):
    a = telluric_atm_of(path)
    return a if a is not None else (atmospheric_transmission(teff, wave) if teff is not None else None)


def telluric_floor_ref_of(path):
    """The transmission a floor refers to: the plain-rv T_eff (T_RV extension of method 'best'), else the
    TELLEFF primary (version 1, where both are the same)."""
    p = os.path.join(os.path.dirname(path),
                     "TELLEFF_" + os.path.basename(path)[len("DATACUBE_COMBINED_"):])
    if not os.path.exists(p):
        return None
    with fits.open(p) as h:
        return np.asarray(h["T_RV"].data if "T_RV" in h else h[0].data, float)


def _stack_weights(texps, tells, atms, normalise, floor_refs=None):
    """Per-cube stacking weights.  Raw stacks (normalise 'after' / False): inverse variance after the telluric
    division, texp * A(lambda)^2.  Median-normalised stacks (normalise True): texp, but pixels with plain-rv
    T_eff <= 0.1 get a tiny weight (this reproduces the old 0.1 floor wherever another cube covers the pixel;
    per-wavelength weights on cubes with different medians would make the flux scale jump at telluric features)."""
    if any(t is None for t in tells):
        return list(texps)
    if normalise is True:
        refs = floor_refs if floor_refs is not None and all(r is not None for r in floor_refs) else tells
        return [tx * np.where(np.nan_to_num(t) > 0.1, 1.0, 1e-3)[:, None, None] for tx, t in zip(texps, refs)]
    return [tx * np.nan_to_num(a, nan=1.0)[:, None, None] ** 2 if a is not None else tx
            for tx, a in zip(texps, atms)]


def atmospheric_transmission(teff, wave):
    """Atmosphere-only part of T_eff: T_eff / smooth instrument response (upper
    envelope as in scripts/telluric_combined.py).  ~1 where the sky is clear,
    A < 0.5 in H2O/O2/CO2 bands, where the correction multiplies the noise by 1/A.
    (Cubes corrected with method 'best' carry the exact A: telluric_atm_of.)"""
    from telluric_combined import envelope
    ok = np.isfinite(teff)
    if ok.sum() < 100:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        return teff / envelope(np.where(ok, teff, np.nanmedian(teff)), wave)


def low_transmission_spans(wave, atm, thr=0.5, min_px=1):
    """[(lo, hi), ...] wavelength intervals where atm < thr or atm is NaN (masked),
    for shading telluric bands in plots."""
    if atm is None:
        return []
    bad = ~(atm >= thr)
    edges = np.flatnonzero(np.diff(np.r_[0, bad.astype(int), 0]))
    half = 0.5 * np.median(np.diff(wave))
    return [(wave[i0] - half, wave[i1 - 1] + half)
            for i0, i1 in zip(edges[::2], edges[1::2]) if i1 - i0 >= min_px]


_COMB_RE = re.compile(r"DATACUBE_COMBINED_(.+?)(?:_(IZ|YJ|HK|H|K))?\.fits$")


def cube_info(path, zcat=None):
    """Header-only description of one DATACUBE_COMBINED file."""
    h0 = fits.getheader(path, 0)
    targ = (h0.get("HIERARCH ESO OBS TARG NAME") or "").strip()
    tid = str(h0.get("HIERARCH ESO PRO REFLEX SUFFIX") or "").strip()
    band = (h0.get("HIERARCH ESO INS GRAT1 ID") or "").strip()
    m = _COMB_RE.match(os.path.basename(path))
    if m and (not targ or not tid):              # fall back to the file name
        stem = m.group(1)
        if zcat is not None:
            parts = stem.split("_")
            for i in range(len(parts) - 1, 0, -1):
                if ("_".join(parts[:i]), "_".join(parts[i:])) in zcat.table:
                    targ, tid = "_".join(parts[:i]), "_".join(parts[i:])
                    break
        if not targ or not tid:
            targ, _, tid = stem.rpartition("_")
    if not band and m and m.group(2):
        band = m.group(2)
    zi = zcat.lookup(targ, tid) if zcat is not None else None
    rel = os.path.relpath(path).replace("\\", "/").split("/")
    return dict(path=path, target_list=targ, tid=tid, field=canonical_field(targ),
                band=band or "??", exptime_s=onsource_seconds(h0),
                n_frames=int(h0.get("NCOMBINE", 1) or 1),
                z=zi[0] if zi else None, z_type=zi[1] if zi else "",
                z_quality=zi[2] if zi else np.nan,
                folder=rel[1] if len(rel) > 2 else rel[0],
                frames=input_frames(h0))


_FRAME_RE = re.compile(r"KMOS\.(\d{4}-\d\d-\d\dT[\d:.]+)")


def input_frames(hdr0):
    """frozenset of the raw exposures (KMOS.<date>T<time>) a combined cube was
    built from, read from its ASSON keywords; empty if none are listed."""
    return frozenset(m.group(1) for k in hdr0.keys() if k.startswith("ASSON")
                     for m in [_FRAME_RE.search(str(hdr0[k]))] if m)


def scan_combined_cubes(base, zcat):
    """All DATACUBE_COMBINED_*.fits under base → (infos with z, infos without z)."""
    paths = sorted(glob.glob(os.path.join(base, "**", "DATACUBE_COMBINED_*.fits"),
                             recursive=True))
    infos = []
    for p in paths:
        try:
            infos.append(cube_info(p, zcat))
        except OSError as err:                   # truncated / corrupt file on disk
            print(f"  WARNING: skipped unreadable cube {p}  ({err})")
    infos = drop_duplicate_cubes(infos)
    return [i for i in infos if i["z"] is not None], [i for i in infos if i["z"] is None]


def drop_duplicate_cubes(infos, verbose=True):
    """Keep one cube per (galaxy, band, set of input frames): re-runs of the same
    combination are the same data and must not add exposure time.  The newest run
    (latest path, run folders are timestamps) is kept."""
    best = {}
    for i in infos:
        key = (i["field"], i["tid"], i["band"], i["frames"]) if i["frames"] else i["path"]
        if key not in best or i["path"] > best[key]["path"]:
            best[key] = i
    keep = {id(i) for i in best.values()}
    out = [i for i in infos if id(i) in keep]
    if verbose and len(out) < len(infos):
        print(f"  skipped {len(infos) - len(out)} duplicate combined cubes "
              f"(same input frames as a newer run)")
    return out


# ── stacking ──────────────────────────────────────────────────────────────────
def stack_cubes(cubes, weights=None, wcss=None):
    """Weighted mean of cubes, aligned on the sky by integer pixel offsets of a
    common position (sub-pixel residual < 0.5 px = 0.1").  NaNs are ignored."""
    return _stack(cubes, weights, wcss)[0]


def _stack(cubes, weights=None, wcss=None, stats=None):
    """stack_cubes + (y, x) position of cube 0's pixel (0, 0) in the output grid.
    With stats (1-sigma cubes, e.g. the STAT extension) also returns the 1-sigma
    cube of the weighted mean: sqrt(sum w^2 s^2) / sum w."""
    if len(cubes) == 1:
        return (cubes[0], (0, 0)) if stats is None else (cubes[0], (0, 0), stats[0])
    weights = [1.0] * len(cubes) if weights is None else weights
    nz = cubes[0].shape[0]
    if wcss is None or any(w is None for w in wcss):
        offs = [(0, 0)] * len(cubes)
        if len({c.shape for c in cubes}) > 1:
            raise ValueError("cubes of different shape need WCS for alignment")
    else:
        ny0, nx0 = cubes[0].shape[1:]
        world = wcss[0].pixel_to_world_values((nx0 - 1) / 2, (ny0 - 1) / 2)
        offs = []
        for w in wcss:
            px, py = w.world_to_pixel_values(*world)
            offs.append((int(round(py - (ny0 - 1) / 2)), int(round(px - (nx0 - 1) / 2))))
    y0 = min(-oy for oy, ox in offs)
    x0 = min(-ox for oy, ox in offs)
    y1 = max(c.shape[1] - oy for c, (oy, ox) in zip(cubes, offs))
    x1 = max(c.shape[2] - ox for c, (oy, ox) in zip(cubes, offs))
    num = np.zeros((nz, y1 - y0, x1 - x0), np.float64)
    den = np.zeros_like(num)
    var = np.zeros_like(num) if stats is not None else None
    for k, (c, (oy, ox), wt) in enumerate(zip(cubes, offs, weights)):
        sy, sx = -oy - y0, -ox - x0
        fin = np.isfinite(c)
        if stats is not None:
            fin &= np.isfinite(stats[k])
            var[:, sy:sy + c.shape[1], sx:sx + c.shape[2]] += np.where(fin, stats[k], 0) ** 2 * wt ** 2
        num[:, sy:sy + c.shape[1], sx:sx + c.shape[2]] += np.where(fin, c, 0) * wt
        den[:, sy:sy + c.shape[1], sx:sx + c.shape[2]] += fin * wt
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(den > 0, num / den, np.nan).astype(np.float32)
        if stats is None:
            return mean, (-y0, -x0)
        return mean, (-y0, -x0), np.where(den > 0, np.sqrt(var) / den, np.nan).astype(np.float32)


def load_galaxy_band_stacks(base, zcat, normalise=True, verbose=True):
    """One stacked cube per (galaxy, band) from all combined cubes under base.

    normalise : True    divide every cube by its own |nanmedian| before stacking (as the
                        notebooks always did; costs S/N when a galaxy has cubes of very
                        different median, e.g. median 3 %, up to 34 % for the two-cube
                        COSMOS YJ galaxies)
                "after" stack the raw cubes, then divide the stack by its |nanmedian|
                        (identical to True for single-cube galaxies)
                False   no normalisation
    Returns a list of dicts sorted by (field, tid, band) with keys
      gid, name, field, tid, band, z, z_type, z_quality, wave, cube, wcs,
      exptime_s (on-source, summed), n_frames, n_cubes, target_lists,
      folders, paths,
      teff (telluric T_eff of the stack, response x atmosphere) and
      atm (atmosphere only, ~1 = clear) - both None unless every cube was
      corrected by scripts/telluric_combined.py
    gid is a stable integer per galaxy (90000 + rank of (field, tid))."""
    with_z, without_z = scan_combined_cubes(base, zcat)
    groups = defaultdict(list)
    for i in with_z:
        groups[(i["field"], i["tid"], i["band"])].append(i)
    gals = sorted({(f, t) for f, t, _ in groups})
    gid_of = {g: 90000 + n for n, g in enumerate(gals)}

    out = []
    for (field, tid, band), members in sorted(groups.items(),
                                              key=lambda kv: (kv[0][0], kv[0][1], band_rank(kv[0][2]))):
        cubes, wcss, wave = [], [], None
        for m in members:
            with fits.open(m["path"]) as hf:
                hdr = hf["DATA"].header
                cube = np.array(hf["DATA"].data, dtype=np.float32)
            w = wavelength_axis(hdr)
            if wave is None:
                wave = w
            elif not np.allclose(wave, w):
                raise ValueError(f"wavelength axes differ for {field} {tid} {band}")
            if normalise and not (isinstance(normalise, str) and normalise == "after"):
                ref = np.nanmedian(cube)
                if np.isfinite(ref) and ref != 0:
                    cube = cube / abs(ref)       # |median|: a negative median must not flip the cube
            cubes.append(cube)
            wcss.append(WCS(hdr).celestial)
        texps = [m["exptime_s"] if np.isfinite(m["exptime_s"]) and m["exptime_s"] > 0
                 else 1.0 for m in members]
        tells = [telluric_of(m["path"]) for m in members]
        if not all(t is not None and t.shape == wave.shape for t in tells):
            tells = [None] * len(members)
        atms = [_member_atm(m["path"], t, wave) if t is not None else None for m, t in zip(members, tells)]
        weights = _stack_weights(texps, tells, atms, normalise,
                                 [telluric_floor_ref_of(m["path"]) for m in members] if normalise is True else None)
        stacked, (sy0, sx0) = _stack(cubes, weights, wcss)
        if isinstance(normalise, str) and normalise == "after":
            ref = np.nanmedian(stacked)
            if np.isfinite(ref) and ref != 0:
                stacked = stacked / abs(ref)
        wcs_out = wcss[0].deepcopy()                 # celestial WCS of the stacked grid
        wcs_out.wcs.crpix = wcs_out.wcs.crpix + np.array([sx0, sy0])
        # telluric transmission (and atmosphere) of the stack: same weights as the cubes
        teff, atm = None, None
        if tells[0] is not None:
            if len(members) == 1:
                teff, atm = tells[0], atms[0]
            else:
                wl = [np.broadcast_to(np.asarray(w, float).reshape(-1), wave.shape) for w in weights]
                T = np.vstack(tells)
                wt = np.vstack(wl) * np.isfinite(T)
                with np.errstate(invalid="ignore", divide="ignore"):
                    teff = np.where(wt.sum(0) > 0, np.nansum(np.nan_to_num(T) * wt, 0) / wt.sum(0), np.nan)
                    if all(a is not None for a in atms):
                        Am = np.vstack(atms)
                        wa = np.vstack(wl) * np.isfinite(Am)
                        atm = np.where(wa.sum(0) > 0, np.nansum(np.nan_to_num(Am) * wa, 0) / wa.sum(0), np.nan)
        m0 = members[0]
        out.append(dict(
            gid=gid_of[(field, tid)], name=galaxy_name(field, tid),
            field=field, tid=tid, band=band,
            z=m0["z"], z_type=m0["z_type"], z_quality=m0["z_quality"],
            wave=wave, cube=stacked, wcs=wcs_out,
            exptime_s=float(np.nansum([m["exptime_s"] for m in members])),
            n_frames=sum(m["n_frames"] for m in members),
            n_cubes=len(members),
            target_lists=sorted({m["target_list"] for m in members}),
            folders=sorted({m["folder"] for m in members}),
            paths=[m["path"] for m in members],
            teff=teff,
            atm=atm if atm is not None else (atmospheric_transmission(teff, wave) if teff is not None else None),
        ))
    if verbose:
        nz = sorted({f"{i['target_list']}/{i['tid']}" for i in without_z})
        per_field = defaultdict(set)
        for e in out:
            per_field[e["field"]].add(e["tid"])
        print(f"{base}: {len(with_z) + len(without_z)} combined cubes, "
              f"{len(without_z)} without redshift" + (f" {nz}" if nz else ""))
        print(f"  → {len(gals)} galaxies, {len(out)} (galaxy, band) stacks, "
              f"{sum(e['n_cubes'] > 1 for e in out)} stacked from >1 cube   "
              + "  ".join(f"{f}: {len(t)}" for f, t in sorted(per_field.items())))
    return out


def load_stack_with_stat(stack):
    """DATA and STAT (1-sigma, RMSE) of one (galaxy, band) stack, both in raw
    units (ADU/s, not median-normalised), stacked like the DATA cube of
    load_galaxy_band_stacks(normalise=False) (weights texp * A^2 for telluric-corrected
    cubes).  Returns (cube, sigma_cube, wcs)."""
    cubes, stats, wcss, texps = [], [], [], []
    for p in stack["paths"]:
        with fits.open(p) as hf:
            cubes.append(np.array(hf["DATA"].data, dtype=np.float32))
            stats.append(np.array(hf["STAT"].data, dtype=np.float32))
            wcss.append(WCS(hf["DATA"].header).celestial)
            wave = wavelength_axis(hf["DATA"].header)
            t = onsource_seconds(hf[0].header)
            texps.append(t if np.isfinite(t) and t > 0 else 1.0)
    tells = [telluric_of(p) for p in stack["paths"]]
    if not all(t is not None and t.shape == wave.shape for t in tells):
        tells = [None] * len(tells)
    atms = [_member_atm(p, t, wave) if t is not None else None for p, t in zip(stack["paths"], tells)]
    cube, (sy0, sx0), sig = _stack(cubes, _stack_weights(texps, tells, atms, False), wcss, stats)
    w = wcss[0].deepcopy()
    w.wcs.crpix = w.wcs.crpix + np.array([sx0, sy0])
    return cube, sig, w


def pick_band(entries, covered, key="exptime_s"):
    """Among the (galaxy, band) entries of ONE galaxy, the entry used for a line:
    the deepest band that covers it (chosen by exposure, never by S/N)."""
    cands = [e for e in entries if covered(e)]
    return max(cands, key=lambda e: e[key]) if cands else None
