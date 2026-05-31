"""Patch CEPII cell: prefer local file, fall back to GitHub URL.
Same pattern used for PWT, IMF ICSD/FAD.
"""
import json, sys

NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e1_data_pull.ipynb'

OLD = '''    url = "https://raw.githubusercontent.com/AyaanTigdikar/Capstone/main/rawdata/geo_cepii.xls"
    cepii_df = (
        pd.read_excel(url, sheet_name="geo_cepii")'''

NEW = '''    _cepii_local = "/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/rawdata/geo_cepii.xls"
    _cepii_url   = "https://raw.githubusercontent.com/AyaanTigdikar/Capstone/main/rawdata/geo_cepii.xls"
    _cepii_src   = _cepii_local if os.path.exists(_cepii_local) else _cepii_url
    print(f"CEPII source: {'local' if os.path.exists(_cepii_local) else 'GitHub URL'}")
    cepii_df = (
        pd.read_excel(_cepii_src, sheet_name="geo_cepii")'''

with open(NB) as f:
    nb = json.load(f)

target = None
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    s = ''.join(c.get('source', []))
    if 'cepii_df' in s and 'geo_cepii' in s and 'def ' not in s.split('cepii_df')[0][-100:]:
        target = i
        break

if target is None:
    print('ERROR: CEPII cell not found.')
    sys.exit(1)

src = ''.join(nb['cells'][target]['source'])

if '_cepii_local' in src:
    print(f'cell {target}: already patched.')
    sys.exit(0)

if OLD not in src:
    print(f'ERROR: expected pattern not found in cell {target}.')
    print('Looking for lines matching cepii_url:')
    for line in src.splitlines():
        if 'cepii' in line.lower() or 'url' in line.lower():
            print(f'  {line!r}')
    sys.exit(1)

src = src.replace(OLD, NEW)
nb['cells'][target]['source'] = src.splitlines(keepends=True)
nb['cells'][target]['outputs'] = []
nb['cells'][target]['execution_count'] = None

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'cell {target}: patched.')
