import re, os
viz_dir = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code'
files = ['_config.py','_style.py','viz_1_descriptive.ipynb','viz_2_ml.ipynb','viz_3_regression.ipynb','viz_4_robustness.ipynb']

# Targeted patterns: the actual stale strings we know are problematic
patterns = [
    'Forestry Intensive',
    'Major Producers',
    'Diversified Producers',  # to see where new name is referenced
    '49-country',
    '49 country',
    'forest-adjusted panel',
    'silhouette',
]

for fn in files:
    p = os.path.join(viz_dir, fn)
    if not os.path.exists(p): continue
    with open(p) as fh:
        txt = fh.read()
    print(f'\n========== {fn} ==========')
    for pat in patterns:
        # Find each occurrence and report
        count = txt.count(pat)
        if count == 0: continue
        # Show first occurrence's context
        i = txt.find(pat)
        # Get the JSON cell source line
        line_start = txt.rfind('\\n', 0, i) + 2  # skip past previous newline marker
        ctx_start = max(0, i-40)
        ctx_end   = min(len(txt), i+80)
        ctx = txt[ctx_start:ctx_end].replace('\\n','/n/').replace('\\"','"')
        print(f'  "{pat}": {count} occurrence(s).  First: ...{ctx}...')
