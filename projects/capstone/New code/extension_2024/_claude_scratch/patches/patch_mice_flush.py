"""Add flush=True to print statements in the run-MICE cell so Jupyter
shows iteration progress in real time instead of buffering."""
import json, sys

NB = '/Users/leoss/Desktop/GitHub/leoss14.github.io/projects/capstone/New code/extension_2024/e2_data_prep.ipynb'

with open(NB) as f:
    nb = json.load(f)

target = None
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    s = ''.join(c.get('source', []))
    if 'for it in range(1, ITERATIONS + 1)' in s:
        target = i
        break

if target is None:
    print('ERROR: progress loop cell not found.')
    sys.exit(1)

src = ''.join(nb['cells'][target]['source'])

# Add flush=True to each print() inside the iteration loop
patches_applied = 0
old1 = "    print(f'  iter {it}/{ITERATIONS}  ({iter_time:5.1f}s, '\n          f'elapsed {elapsed/60:4.1f} min, ETA {remaining_eta/60:4.1f} min)')"
new1 = "    print(f'  iter {it}/{ITERATIONS}  ({iter_time:5.1f}s, '\n          f'elapsed {elapsed/60:4.1f} min, ETA {remaining_eta/60:4.1f} min)', flush=True)"
if old1 in src:
    src = src.replace(old1, new1)
    patches_applied += 1

# Add flush=True to the "Starting miceforest" print
old2 = "      f'{len(mice_input):,} rows, {ITERATIONS} iterations')"
new2 = "      f'{len(mice_input):,} rows, {ITERATIONS} iterations', flush=True)"
if old2 in src:
    src = src.replace(old2, new2)
    patches_applied += 1

# Add sys import and immediate flush after kernel creation
if 'import sys' not in src.split('# Run MICE')[1][:200]:
    # Inject sys flush right after starting message
    inject_before = "kernel = mf.ImputationKernel("
    inject_text = "import sys; sys.stdout.flush()\n\n"
    if inject_before in src and inject_text not in src:
        src = src.replace(inject_before, inject_text + inject_before)
        patches_applied += 1

nb['cells'][target]['source'] = src.splitlines(keepends=True)
nb['cells'][target]['outputs'] = []
nb['cells'][target]['execution_count'] = None

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'cell {target}: applied {patches_applied} flush patches.')
