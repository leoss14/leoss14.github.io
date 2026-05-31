"""Remove the cfg.YEAR_MIN/MAX filter from the PinkSheet loader (e0 doesn't import cfg)."""
import json, sys

NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e0_NR_extraction.ipynb'

OLD = "df = df[(df['Year'] >= cfg.YEAR_MIN) & (df['Year'] <= cfg.YEAR_MAX)]"
NEW = "# Year filtering handled downstream (combine_and_clean / config)."

with open(NB) as f:
    nb = json.load(f)

target = None
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    s = ''.join(c.get('source', []))
    if 'def load_world_bank_pinksheet' in s and OLD in s:
        target = i
        nb['cells'][i]['source'] = s.replace(OLD, NEW).splitlines(keepends=True)
        nb['cells'][i]['outputs'] = []
        nb['cells'][i]['execution_count'] = None
        break

if target is None:
    # Already patched?
    for i, c in enumerate(nb['cells']):
        s = ''.join(c.get('source', []))
        if 'def load_world_bank_pinksheet' in s:
            if 'cfg.YEAR_MIN' not in s:
                print(f'cell {i}: already patched, no cfg reference.')
                sys.exit(0)
            else:
                print(f'cell {i}: cfg reference present but pattern did not match. Inspect manually.')
                sys.exit(1)
    print('ERROR: PinkSheet loader cell not found.')
    sys.exit(1)

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'cell {target}: removed cfg year filter.')
