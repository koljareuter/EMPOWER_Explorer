"""Header scan of all DATACUBE_COMBINED_*.fits of one reduction, with a per-user disk cache.

Identity, exposure time and duplicate handling are exactly those of scripts/kmos_sample.py
(cube_info / drop_duplicate_cubes); one header read per cube, cached by (path, size, mtime), so
the second start of the app needs no FITS access at all.
"""
import glob
import hashlib
import json
import os
import re

import numpy as np
from astropy.io import fits

from empower_explorer import config, engine

CACHE_VERSION = 3

_EXTRA_KEYS = {          # primary-header keywords kept for the Data QA tab
    'object': 'OBJECT', 'ra': 'RA', 'dec': 'DEC', 'date_obs': 'DATE-OBS', 'exptime_hdr': 'EXPTIME',
    'seeing_start': 'HIERARCH ESO TEL AMBI FWHM START', 'seeing_end': 'HIERARCH ESO TEL AMBI FWHM END',
    'iq': 'HIERARCH ESO TEL IA FWHM', 'airmass_start': 'HIERARCH ESO TEL AIRM START',
    'pro_catg': 'HIERARCH ESO PRO CATG',
}


def _header_info(path, variant_dir):
    """Same fields as kmos_sample.cube_info (without z) + a few QA keywords."""
    ks = engine.ks
    with fits.open(path, memmap=True) as h:
        h0 = h[0].header
        shape = None
        for e in h[1:]:
            if str(e.header.get('EXTNAME', '')).upper() == 'DATA':
                shape = [int(e.header.get(f'NAXIS{i}', 0)) for i in (3, 2, 1)]
                break
    targ = (h0.get('HIERARCH ESO OBS TARG NAME') or '').strip()
    tid = str(h0.get('HIERARCH ESO PRO REFLEX SUFFIX') or '').strip()
    band = (h0.get('HIERARCH ESO INS GRAT1 ID') or '').strip()
    m = ks._COMB_RE.match(os.path.basename(path))
    if m and (not targ or not tid):
        targ, _, tid = m.group(1).rpartition('_')
    if not band and m and m.group(2):
        band = m.group(2)
    rel = os.path.relpath(path, variant_dir).replace('\\', '/').split('/')
    info = dict(path=path, target_list=targ, tid=tid, field=ks.canonical_field(targ),
                band=band or '??', exptime_s=float(ks.onsource_seconds(h0)),
                n_frames=int(h0.get('NCOMBINE', 1) or 1),
                folder='/'.join(rel[:-1]), frames=sorted(ks.input_frames(h0)), shape=shape,
                telluric=os.path.exists(os.path.join(
                    os.path.dirname(path), 'TELLEFF_' + os.path.basename(path)[len('DATACUBE_COMBINED_'):])))
    for k, key in _EXTRA_KEYS.items():
        v = h0.get(key)
        info[k] = v if isinstance(v, (int, float, str)) else None
    return info


def _cache_file(variant_dir):
    tag = hashlib.md5(os.path.abspath(variant_dir).lower().encode()).hexdigest()[:10]
    return config.cache_dir() / f'scan_{os.path.basename(variant_dir)}_{tag}.json'


def scan_variant(root, variant, progress=None):
    """All combined cubes of one reduction folder -> (infos, problems).

    infos: list of dicts (path, target_list, tid, field, band, exptime_s, n_frames, folder,
    frames, shape, telluric, object, ra, dec, ...), duplicates (same input frames as a newer
    run) removed as in kmos_sample.drop_duplicate_cubes; each info has 'key' = 'FIELD tid'.
    problems: list of 'file: error' strings (unreadable cubes are skipped)."""
    variant_dir = os.path.join(root, variant)
    paths = sorted(glob.glob(os.path.join(variant_dir, '**', 'DATACUBE_COMBINED_*.fits'), recursive=True))
    cache_path = _cache_file(variant_dir)
    try:
        cache = json.loads(cache_path.read_text(encoding='utf-8'))
        if cache.get('version') != CACHE_VERSION:
            cache = {}
    except (OSError, ValueError):
        cache = {}
    entries = cache.get('entries', {})
    infos, problems, fresh = [], [], {}
    for i, p in enumerate(paths):
        if progress is not None and (i % 10 == 0 or i == len(paths) - 1):
            progress(i + 1, len(paths))
        try:
            st = os.stat(p)
            sig = f'{st.st_size}:{int(st.st_mtime)}'
            hit = entries.get(p)
            if hit is not None and hit.get('_sig') == sig:
                info = dict(hit)
            else:
                info = _header_info(p, variant_dir)
                info['_sig'] = sig
            fresh[p] = info
            infos.append(dict(info))
        except Exception as e:          # truncated / corrupt / not yet synced file
            problems.append(f'{os.path.basename(p)}: {e}')
    changed = set(fresh) != set(entries) or any(entries[p].get('_sig') != v.get('_sig') for p, v in fresh.items())
    if changed:
        try:                 # atomic: several app instances may scan at the same time
            tmp = cache_path.with_suffix(f'.{os.getpid()}.tmp')
            tmp.write_text(json.dumps({'version': CACHE_VERSION, 'entries': fresh}), encoding='utf-8')
            os.replace(tmp, cache_path)
        except OSError:
            pass

    # kmos_sample.drop_duplicate_cubes: one cube per (galaxy, band, set of input frames), newest path
    for info in infos:
        info['frames'] = frozenset(info['frames'])
    best = {}
    for i in infos:
        key = (i['field'], i['tid'], i['band'], i['frames']) if i['frames'] else i['path']
        if key not in best or i['path'] > best[key]['path']:
            best[key] = i
    keep = {id(i) for i in best.values()}
    n_dup = len(infos) - len(keep)
    infos = [i for i in infos if id(i) in keep]
    for i in infos:
        i['key'] = engine.ks.galaxy_name(i['field'], i['tid'])
        i.pop('_sig', None)
    if n_dup:
        problems.insert(0, f'{n_dup} duplicate cube(s) skipped (same input frames as a newer run)')
    return infos, problems
