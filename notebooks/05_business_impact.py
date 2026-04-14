"""
Notebook 05 — Business Impact Sizing
──────────────────────────────────────
Framework: Expected Loss = PD × EAD × LGD
  PD  = model-predicted default probability
  EAD = Exposure at Default  (~40% of credit limit)
  LGD = Loss Given Default   (65% — industry estimate)

Steps:
  1. Score the test set using the best model
  2. Simulate approval/decline at every cutoff threshold
  3. Compute losses prevented vs. revenue foregone
  4. Find the optimal cutoff that maximises net benefit
  5. Scale results to a 1-million customer portfolio
  6. Lift chart and cumulative gains analysis

Run from project root:
    python notebooks/05_business_impact.py
"""
import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from src.utils import *

print('=' * 60)
print('  05 — Business Impact Sizing')
print('=' * 60)

# ── Load ──────────────────────────────────────────────────────
df     = pd.read_csv('data/credit_card_engineered.csv')
meta   = joblib.load('outputs/model_metadata.pkl')
model  = joblib.load('outputs/best_model.pkl')
scaler = joblib.load('outputs/scaler.pkl')

FEATURE_COLS = meta['feature_cols']
best_name    = meta['best_model']
print(f'\n  Best model loaded : {best_name}')

X = df[FEATURE_COLS].fillna(0)
y = df[TARGET]
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

Xte = scaler.transform(X_test) if best_name in (
    'Logistic Regression','MLP Neural Net') else X_test
scores = model.predict_proba(Xte)[:, 1]

# ── Business assumptions ──────────────────────────────────────
section('1. Business assumptions')

AVG_CREDIT_LIMIT = 120_000    # NT$ average credit limit
LGD              = 0.65       # Loss Given Default
EAD_RATIO        = 0.40       # Exposure at Default as % of limit
REVENUE_PER_GOOD = 3_500      # NT$ annual revenue per non-defaulting account
LOSS_PER_DEFAULT = AVG_CREDIT_LIMIT * EAD_RATIO * LGD

print(f'  Avg credit limit      : NT$ {AVG_CREDIT_LIMIT:>10,}')
print(f'  EAD ratio             :      {EAD_RATIO:.0%}')
print(f'  Loss Given Default    :      {LGD:.0%}')
print(f'  Loss per default      : NT$ {LOSS_PER_DEFAULT:>10,.0f}')
print(f'  Revenue per good acct : NT$ {REVENUE_PER_GOOD:>10,}')

# ── Cutoff simulation ─────────────────────────────────────────
section('2. Cutoff simulation')

cutoffs = np.arange(0.05, 0.95, 0.025)
rows    = []
n       = len(y_test)

for c in cutoffs:
    approved = scores < c
    tp = ((scores >= c) & (y_test == 1)).sum()  # correctly declined defaults
    fp = ((scores >= c) & (y_test == 0)).sum()  # incorrectly declined good
    fn = ((scores <  c) & (y_test == 1)).sum()  # missed defaults

    lp  = tp * LOSS_PER_DEFAULT          # losses prevented
    rl  = fp * REVENUE_PER_GOOD          # revenue lost
    nb  = lp - rl                        # net benefit

    rows.append({
        'cutoff':              c,
        'approved_pct':        approved.sum() / n,
        'defaults_caught_pct': tp / max(y_test.sum(), 1),
        'good_declined_pct':   fp / max((y_test==0).sum(), 1),
        'losses_prevented':    lp,
        'revenue_lost':        rl,
        'net_benefit':         nb,
        'n_approved':          approved.sum(),
        'TP': tp, 'FP': fp, 'FN': fn,
    })

res = pd.DataFrame(rows)
best_row = res.loc[res['net_benefit'].idxmax()]

print(f'\n  Baseline (no model — approve all):')
print(f'    Defaults in test set  : {y_test.sum():,}')
print(f'    Total expected losses : NT$ {y_test.sum() * LOSS_PER_DEFAULT:,.0f}')
print(f'\n  Optimal cutoff ({best_row["cutoff"]:.3f}):')
print(f'    Defaults caught       : {best_row["defaults_caught_pct"]:.1%}')
print(f'    Good clients declined : {best_row["good_declined_pct"]:.1%}')
print(f'    Losses prevented      : NT$ {best_row["losses_prevented"]:,.0f}')
print(f'    Revenue foregone      : NT$ {best_row["revenue_lost"]:,.0f}')
print(f'    Net benefit           : NT$ {best_row["net_benefit"]:,.0f}')

scale          = 1_000_000 / n
scaled_benefit = best_row['net_benefit'] * scale
print(f'\n  Scaled to 1M customers:')
print(f'    Net annual benefit    : NT$ {scaled_benefit:,.0f}')
print(f'    Approx USD            : ${scaled_benefit/30:,.0f}')

res.to_csv('outputs/business_impact_simulation.csv', index=False)

# ── FIGURE 1: Net benefit curve ───────────────────────────────
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(res['cutoff'], res['net_benefit']/1e6,
         color=PALETTE['highlight'], linewidth=2.5)
ax1.axvline(best_row['cutoff'], color=PALETTE['default'],
            linestyle='--', linewidth=1.5,
            label=f'Optimal cutoff = {best_row["cutoff"]:.2f}')
ax1.fill_between(res['cutoff'], 0, res['net_benefit']/1e6,
                 alpha=0.1, color=PALETTE['highlight'])
ax1.axhline(0, color='gray', linewidth=0.8)
ax1.set_xlabel('Score cutoff'); ax1.set_ylabel('Net benefit (M NT$)')
ax1.set_title('Net benefit vs. approval cutoff', fontweight='bold')
ax1.legend()

ax2.plot(res['cutoff'], res['losses_prevented']/1e6,
         color=PALETTE['accent'], linewidth=2, label='Losses prevented')
ax2.plot(res['cutoff'], res['revenue_lost']/1e6,
         color=PALETTE['default'], linewidth=2, label='Revenue foregone')
ax2.axvline(best_row['cutoff'], color=PALETTE['highlight'],
            linestyle='--', linewidth=1.5)
ax2.set_xlabel('Score cutoff'); ax2.set_ylabel('NT$ (millions)')
ax2.set_title('Loss prevention vs. revenue impact', fontweight='bold')
ax2.legend()

fig1.suptitle('Business impact simulation — model vs. no-model',
              fontweight='bold')
plt.tight_layout()
save_fig(fig1, '05a_business_impact')

# ── FIGURE 2: Lift chart ──────────────────────────────────────
N_BUCKETS   = 10
sorted_idx  = np.argsort(-scores)
y_sorted    = y_test.values[sorted_idx]
bucket_size = len(y_sorted) // N_BUCKETS
baseline_dr = y_test.mean()
lifts, cum_dr = [], []
cum_def = 0

for i in range(N_BUCKETS):
    bucket   = y_sorted[i*bucket_size:(i+1)*bucket_size]
    cum_def += bucket.sum()
    lifts.append(bucket.mean() / baseline_dr)
    cum_dr.append(cum_def / y_test.sum())

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
b_labels   = [f'D{i+1}' for i in range(N_BUCKETS)]
bar_colors = [PALETTE['default'] if l > 1.5
              else PALETTE['highlight'] if l > 1.0
              else PALETTE['non_default'] for l in lifts]
bars = ax1.bar(b_labels, lifts, color=bar_colors, alpha=0.85)
ax1.axhline(1.0, color='gray', linestyle='--', linewidth=1,
            label='Baseline (no model)')
for bar, v in zip(bars, lifts):
    ax1.text(bar.get_x()+bar.get_width()/2,
             bar.get_height()+0.03,
             f'{v:.1f}x', ha='center', va='bottom', fontsize=8)
ax1.set_title('Lift by decile  (D1 = highest risk score)',
              fontweight='bold')
ax1.set_xlabel('Score decile')
ax1.set_ylabel('Lift vs. random')
ax1.legend()

ax2.plot(range(1, N_BUCKETS+1), [c*100 for c in cum_dr],
         'o-', color=PALETTE['highlight'], linewidth=2.5, label='Model')
ax2.plot(range(1, N_BUCKETS+1), [i*10 for i in range(1, N_BUCKETS+1)],
         '--', color='gray', linewidth=1.5, label='Random baseline')
ax2.fill_between(range(1, N_BUCKETS+1),
                 [c*100 for c in cum_dr],
                 [i*10 for i in range(1, N_BUCKETS+1)],
                 alpha=0.12, color=PALETTE['highlight'])
ax2.set_xlabel('Top N deciles reviewed')
ax2.set_ylabel('% defaults captured')
ax2.set_title('Cumulative gains chart', fontweight='bold')
ax2.legend()
fig2.suptitle('Model lift & cumulative gains', fontweight='bold')
plt.tight_layout()
save_fig(fig2, '05b_lift_gains')

# ── FIGURE 3: Impact summary ──────────────────────────────────
fig3, ax = plt.subplots(figsize=(9, 5))
baseline_loss = y_test.sum() * LOSS_PER_DEFAULT
cats   = ['Expected losses\n(no model)', 'Losses prevented\nby model',
          'Revenue foregone\n(good clients declined)', 'Net benefit\n(model)']
vals   = [baseline_loss/1e6,
          best_row['losses_prevented']/1e6,
          -best_row['revenue_lost']/1e6,
          best_row['net_benefit']/1e6]
cols   = [PALETTE['default'], PALETTE['accent'],
          PALETTE['neutral'], PALETTE['highlight']]
bars   = ax.bar(cats, vals, color=cols, alpha=0.85, width=0.55)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height() + 0.15,
            f'NT$ {abs(v):.1f}M', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
ax.axhline(0, color='gray', linewidth=0.8)
ax.set_ylabel('NT$ millions (test set)')
ax.set_title(f'Business impact summary — {best_name}  '
             f'(cutoff={best_row["cutoff"]:.2f})', fontweight='bold')
plt.tight_layout()
save_fig(fig3, '05c_impact_summary')

print('\n[✓] Business impact analysis complete')
