import os, re
nb_dir = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code'

# Things to flag as stale references
stale_patterns = {
    'Major Producers':       'old cluster (removed)',
    'Forestry Intensive':    'old cluster (removed)',
    'Mixed Producers 1':     'current residual cluster (OK but ugly)',
    'Mongolia':              'check context',
    'Brunei':                'not in sample',
    'Norway':                'not in sample',
    'Australia':             'not in sample',
    '38 countries':          'old sample size',
    '49 countries':          'old sample size',
    '54 countries':          'old sample size',
    '38-country':            'old sample size',
    '49-country':            'old sample size',
    '54-country':            'old sample size',
    '5%':                    'old threshold',
}

for fn in ['viz_1_descriptive.ipynb','viz_2_ml.ipynb','viz_3_regression.ipynb','viz_4_robustness.ipynb']:
    p = os.path.join(nb_dir, fn)
    with open(p) as fh:
        txt = fh.read()
    print(f'=== {fn} ===')
    found_any = False
    for pat, label in stale_patterns.items():
        cnt = txt.count(pat)
        if cnt > 0:
            found_any = True
            # Show first instance with context
            i = txt.find(pat)
            ctx = txt[max(0,i-80):min(len(txt),i+80)].replace('\\n','/').replace('\\"','"').replace('    ','')
            print(f'  [{cnt}x] "{pat}" ({label})')
            print(f'         ...{ctx.strip()[:160]}...')
    if not found_any:
        print('  (no stale refs found)')
    print()
