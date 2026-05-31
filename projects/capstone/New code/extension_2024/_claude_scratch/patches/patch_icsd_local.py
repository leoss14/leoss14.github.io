"""Patch the IMF ICSD cell to prefer local files, fall back to GitHub URLs.
Same pattern used for PWT and CEPII.

Idempotent: re-running on an already-patched notebook is a no-op.
"""
import json, sys

NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e1_data_pull.ipynb'

# These constants will be inserted at the very top of the cell, before any URL
# is used. We pick them up via two new variables so the resolution logic is
# explicit and the URL strings stay verbatim for diff visibility.
LOCAL_RAWDATA = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/rawdata'

ICSD_LOCAL_FILE = f"{LOCAL_RAWDATA}/dataset_2025-12-23T17_49_47.423426845Z_DEFAULT_INTEGRATION_IMF.FAD_ICSD_1.0.0.csv"
FAD_LOCAL_FILE  = f"{LOCAL_RAWDATA}/dataset_2026-01-05T21_29_20.743520144Z_DEFAULT_INTEGRATION_IMF.FAD_FM_5.0.0.csv"

ICSD_URL = "https://raw.githubusercontent.com/AyaanTigdikar/Capstone/main/rawdata/dataset_2025-12-23T17_49_47.423426845Z_DEFAULT_INTEGRATION_IMF.FAD_ICSD_1.0.0.csv"
FAD_URL  = "https://raw.githubusercontent.com/AyaanTigdikar/Capstone/refs/heads/main/rawdata/dataset_2026-01-05T21_29_20.743520144Z_DEFAULT_INTEGRATION_IMF.FAD_FM_5.0.0.csv"

# Strings to find in the current cell source
ICSD_READ_OLD = (
    '    imf_icsd_df = (\n'
    '        pd.read_csv(\n'
    '            "https://raw.githubusercontent.com/AyaanTigdikar/Capstone/main/rawdata/"\n'
    '            "dataset_2025-12-23T17_49_47.423426845Z_DEFAULT_INTEGRATION_IMF.FAD_ICSD_1.0.0.csv"\n'
    '        )'
)
ICSD_READ_NEW = (
    '    _icsd_local = "' + ICSD_LOCAL_FILE + '"\n'
    '    _icsd_url   = ("' + ICSD_URL + '")\n'
    '    _icsd_src   = _icsd_local if os.path.exists(_icsd_local) else _icsd_url\n'
    '    print(f"IMF ICSD source: {\'local\' if os.path.exists(_icsd_local) else \'GitHub URL\'}")\n'
    '    imf_icsd_df = (\n'
    '        pd.read_csv(_icsd_src)'
)

FAD_READ_OLD = (
    '    imf_icsd_df_2 = (\n'
    '        pd.read_csv(\n'
    '            "https://raw.githubusercontent.com/AyaanTigdikar/Capstone/refs/heads/main/rawdata/"\n'
    '            "dataset_2026-01-05T21_29_20.743520144Z_DEFAULT_INTEGRATION_IMF.FAD_FM_5.0.0.csv"\n'
    '        )'
)
FAD_READ_NEW = (
    '    _fad_local = "' + FAD_LOCAL_FILE + '"\n'
    '    _fad_url   = ("' + FAD_URL + '")\n'
    '    _fad_src   = _fad_local if os.path.exists(_fad_local) else _fad_url\n'
    '    print(f"IMF FAD_FM source: {\'local\' if os.path.exists(_fad_local) else \'GitHub URL\'}")\n'
    '    imf_icsd_df_2 = (\n'
    '        pd.read_csv(_fad_src)'
)


def patch():
    with open(NB) as f:
        nb = json.load(f)

    # Find the IMF ICSD cell
    target = None
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        src = ''.join(c.get('source', []))
        if 'imf_icsd_df' in src and 'imf_icsd_variables' in src and 'gfcf_codes' in src:
            target = i
            break

    if target is None:
        print('ERROR: could not find ICSD cell.')
        sys.exit(1)

    src = ''.join(nb['cells'][target]['source'])

    if '_icsd_local' in src and '_fad_local' in src:
        print(f'Cell {target}: already patched. No-op.')
        return

    applied = 0
    for old, new, label in [
        (ICSD_READ_OLD, ICSD_READ_NEW, 'ICSD'),
        (FAD_READ_OLD, FAD_READ_NEW, 'FAD'),
    ]:
        if old in src:
            src = src.replace(old, new)
            applied += 1
            print(f'  {label} read: patched.')
        else:
            print(f'  WARNING: {label} read pattern not found. Showing first chars of expected:')
            print(f'    {old[:120]!r}')

    if applied == 0:
        print('ERROR: no patches applied. Pattern mismatch.')
        sys.exit(1)

    nb['cells'][target]['source'] = src.splitlines(keepends=True)
    nb['cells'][target]['outputs'] = []
    nb['cells'][target]['execution_count'] = None

    with open(NB, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f'Cell {target}: applied {applied}/2 patches.')


if __name__ == '__main__':
    patch()
