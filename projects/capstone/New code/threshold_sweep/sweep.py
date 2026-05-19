"""
Threshold sweep: re-runs NB1-4 with four different NR_THRESHOLD values
(0%, 1%, 2%, 3%) and saves each run's artifacts to its own subfolder.

Usage from inside the New code directory:
    python threshold_sweep/sweep.py

Each run takes a few minutes (regression + ML fits, no bootstrap).
Total runtime ~10-20 min on a typical laptop. The headline 1% artifacts
in projects/capstone/artifacts/ are left untouched.

Outputs:
    threshold_sweep/artifacts_t0/   <- threshold = 0 (developing only, no rent filter)
    threshold_sweep/artifacts_t1/   <- threshold = 1
    threshold_sweep/artifacts_t2/   <- threshold = 2
    threshold_sweep/artifacts_t3/   <- threshold = 3
    threshold_sweep/_executed/      <- temp copies of executed notebooks (logs only)
"""
import os, sys, subprocess, time

THRESHOLDS = [0, 1, 2, 3]
HERE       = os.path.dirname(os.path.abspath(__file__))
NB_DIR     = os.path.dirname(HERE)
PYTHON      = '/usr/local/bin/python3.10'   # interpreter
KERNEL_NAME = 'py310'                         # kernelspec registered for python3.10
NOTEBOOKS  = [
    '1_data_prep.ipynb',
    '2_clusters_and_correlations.ipynb',
    '3_ml_models.ipynb',
    '4_regressions.ipynb',
]

EXEC_DIR = os.path.join(HERE, '_executed')
os.makedirs(EXEC_DIR, exist_ok=True)


def run_one(threshold):
    art_dir = os.path.join(HERE, f'artifacts_t{threshold}')
    os.makedirs(art_dir, exist_ok=True)
    env = os.environ.copy()
    env['NR_THRESHOLD']        = str(threshold)
    env['CAPSTONE_ARTIFACTS']  = art_dir
    env['SKIP_BOOTSTRAP']      = '1'   # skip the 200-iter LASSO bootstrap to save time
    print(f'\n===== THRESHOLD = {threshold}% =====')
    print(f'  Artifacts -> {art_dir}')
    for nb in NOTEBOOKS:
        nb_path = os.path.join(NB_DIR, nb)
        out_path = os.path.join(EXEC_DIR, f't{threshold}_{nb}')
        t0 = time.time()
        print(f'  Running {nb} ...', end='', flush=True)
        try:
            subprocess.run(
                [PYTHON, '-m', 'jupyter', 'nbconvert',
                 '--to', 'notebook', '--execute',
                 nb_path, '--output', out_path,
                 f'--ExecutePreprocessor.kernel_name={KERNEL_NAME}',
                 '--ExecutePreprocessor.timeout=600'],
                env=env, check=True, capture_output=True, text=True,
            )
            print(f' done ({time.time()-t0:.0f}s)')
        except subprocess.CalledProcessError as e:
            print(f' FAILED after {time.time()-t0:.0f}s')
            log_path = os.path.join(EXEC_DIR, f't{threshold}_{nb}.stderr.log')
            with open(log_path, 'w') as f:
                f.write(e.stderr or '')
            print(f'  Full stderr -> {log_path}')
            print('  STDERR (last 2500 chars):')
            print('  ' + (e.stderr or '')[-2500:].replace('\n', '\n  '))
            raise


def main():
    t_total = time.time()
    for t in THRESHOLDS:
        run_one(t)
    print(f'\nAll runs complete in {(time.time()-t_total)/60:.1f} min')
    print(f'Build the comparison page with:')
    print(f'  python threshold_sweep/build_html.py')


if __name__ == '__main__':
    main()
