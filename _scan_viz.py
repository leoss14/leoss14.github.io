import re, os
viz_dir = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code'
files = ['viz_1_descriptive.ipynb','viz_2_ml.ipynb','viz_3_regression.ipynb','viz_4_robustness.ipynb']

# Patterns that might indicate stale references
patterns = [
    (r'\b38\b', '38 (old sample size)'),
    (r'\b49\b', '49 (old sample size)'),
    (r'\b54\b', '54 (older sample size)'),
    (r'Forestry Intensive', 'stale cluster name'),
    (r'Major Producers', 'stale cluster name'),
    (r'1995 cross-section', 'stale clustering year ref'),
    (r'1995 snapshot', 'stale clustering year ref'),
    (r'silhouette', 'silhouette ref (may be stale value)'),
    (r'5%-of-GDP', 'stale threshold'),
    (r'5% of GDP', 'stale threshold'),
    (r'developing countr', 'developing-country ref'),
]

for fn in files:
    p = os.path.join(viz_dir, fn)
    if not os.path.exists(p): print(f'{fn}: NOT FOUND'); continue
    with open(p) as fh:
        txt = fh.read()
    found = []
    for pat, desc in patterns:
        for m in re.finditer(pat, txt):
            i = m.start()
            # Get surrounding context (one cell snippet)
            ctx = txt[max(0,i-60):min(len(txt),i+120)].replace('\\n','\n').replace('\\"','"')
            # Skip if it's in a base64 image, output cell stream, or execution_count field
            if '"execution_count"' in ctx[:80] or 'image/png' in ctx or 'base64,' in ctx:
                continue
            # Skip if it's plotly trace data (long sequence of numbers)
            line_start = txt.rfind('\n', 0, i) + 1
            line_end = txt.find('\n', i)
            line = txt[line_start:line_end] if line_end > 0 else txt[line_start:]
            # Crude check: if the line has lots of numbers separated by commas, it's data
            n_commas = line.count(',')
            n_digits = sum(c.isdigit() for c in line)
            if n_commas > 5 and n_digits > 30 and 'title' not in line.lower():
                continue
            found.append((pat, desc, ctx.strip()))
    if found:
        print(f'\n========== {fn} ==========')
        seen = set()
        for pat, desc, ctx in found:
            key = (pat, ctx[:50])
            if key in seen: continue
            seen.add(key)
            print(f'  [{desc}]')
            print(f'    ...{ctx[:200]}...')
