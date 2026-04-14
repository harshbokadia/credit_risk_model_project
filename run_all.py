"""
run_all.py — Master pipeline runner
─────────────────────────────────────
Executes all 6 notebooks in sequence from project root.

Usage:
    python run_all.py              # run all steps
    python run_all.py --skip 06   # skip enhancement step
"""
import subprocess, sys, time, argparse

STEPS = [
    ('01', 'Data Integrity & EDA',       'notebooks/01_eda.py'),
    ('02', 'Customer Segmentation',      'notebooks/02_segmentation.py'),
    ('03', 'Feature Engineering',        'notebooks/03_feature_engineering.py'),
    ('04', 'Model Training & Validation','notebooks/04_modeling.py'),
    ('05', 'Business Impact Sizing',     'notebooks/05_business_impact.py'),
    ('06', 'Model Enhancement (Optuna)', 'notebooks/06_model_enhancement.py'),
]

parser = argparse.ArgumentParser()
parser.add_argument('--skip', nargs='*', default=[],
                    help='Step numbers to skip, e.g. --skip 06')
args = parser.parse_args()

print('╔══════════════════════════════════════════════════════════╗')
print('║   Credit Risk Modelling — Full Pipeline Runner          ║')
print('╚══════════════════════════════════════════════════════════╝\n')

total_start = time.time()
passed, skipped = 0, 0

for num, name, script in STEPS:
    if num in args.skip:
        print(f'  ⏭  Step {num}: {name}  [skipped]')
        skipped += 1
        continue

    print(f'  ▶  Step {num}/06: {name}')
    t0     = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed= time.time() - t0

    if result.returncode == 0:
        print(f'     ✓ {elapsed:.1f}s\n')
        passed += 1
    else:
        print(f'     ✗ FAILED  (exit code {result.returncode})\n')
        sys.exit(1)

total = time.time() - total_start
print('╔══════════════════════════════════════════════════════════╗')
print(f'║   {passed} steps complete, {skipped} skipped — {total:.0f}s total          ')
print('║   Charts saved to outputs/                              ║')
print('╚══════════════════════════════════════════════════════════╝')
