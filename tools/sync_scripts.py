"""Copy the EMPOWER analysis scripts from the analysis folder into the app (empower_explorer/scripts).

    python tools/sync_scripts.py <EMPOWER_Analysis_and_presentation>           # copy what changed
    python tools/sync_scripts.py <EMPOWER_Analysis_and_presentation> --check   # only list differences

Run it after changing kmos_sample.py, line_snr.py or telluric_combined.py for the notebooks, so the
app and the pipeline keep giving the notebooks' numbers (line endings are ignored in the comparison).
"""
import argparse
import shutil
import sys
from pathlib import Path

FILES = ('kmos_sample.py', 'line_snr.py', 'telluric_combined.py', 'target_redshifts.csv',
         'sky_lines_uves_oliva.csv')
DEST = Path(__file__).resolve().parents[1] / 'empower_explorer' / 'scripts'


def _same(a, b):
    return b.is_file() and a.read_bytes().replace(b'\r\n', b'\n') == b.read_bytes().replace(b'\r\n', b'\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description='Copy the EMPOWER analysis scripts into empower_explorer/scripts')
    ap.add_argument('analysis_folder', help='the folder with scripts/kmos_sample.py (EMPOWER_Analysis_and_presentation)')
    ap.add_argument('--check', action='store_true', help='list the differences, copy nothing')
    a = ap.parse_args(argv)
    src = Path(a.analysis_folder) / 'scripts'
    missing = [f for f in FILES if not (src / f).is_file()]
    if missing:
        ap.error(f'not found in {src}: {", ".join(missing)}')
    changed = [f for f in FILES if not _same(src / f, DEST / f)]
    for f in changed:
        print(('differs: ' if a.check else 'copied:  ') + f)
        if not a.check:
            shutil.copy2(src / f, DEST / f)
    if not changed:
        print('up to date')
    return 1 if a.check and changed else 0


if __name__ == '__main__':
    sys.exit(main())
