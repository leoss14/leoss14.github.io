"""
Fix: cell 4 dropped the 'FEATURES = present' line that downstream cells
(8/14/21) all depend on. Without it, FEATURES is undefined or stale from
kernel memory, causing the ML to train on whatever was last loaded.

This patch restores 'FEATURES = present' at the end of cell 4 so it gets the
MODE-resolved feature list as intended.

Also patch the diagnostic print at the end of cell 4 to use FEATURES.
"""
import json
from pathlib import Path

NB = Path('/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e4_ml.ipynb')

with open(NB) as f:
    nb = json.load(f)

# Cell 4 current source ends with:
#   if missing:
#       print(f'  MISSING: {missing}')
# Add 'FEATURES = present' (and matching diagnostic) right after.

src = ''.join(nb['cells'][4]['source'])
old_tail = '''if missing:
    print(f'  MISSING: {missing}')
'''
new_tail = '''if missing:
    print(f'  MISSING: {missing}')

# FEATURES is the canonical feature list used by downstream cells (train_models
# default, SHAP labelling, structural-spec construction). Under MODE='net' this
# carries the *_net suffix on the resource share entries; under MODE='gross' it
# uses the plain *_share names.
FEATURES = present
'''

if old_tail in src:
    src = src.replace(old_tail, new_tail)
    nb['cells'][4]['source'] = src.splitlines(keepends=True)
    nb['cells'][4]['outputs'] = []
    nb['cells'][4]['execution_count'] = None
    with open(NB, 'w') as f:
        json.dump(nb, f, indent=1)
    print(f'Patched cell 4: appended FEATURES = present')
else:
    print('ERROR: cell 4 tail does not match expected pattern. Source ends:')
    print(src[-500:])
