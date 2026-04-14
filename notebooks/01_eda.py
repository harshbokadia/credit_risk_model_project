"""
Notebook 01 — Data Integrity & Exploratory Data Analysis
─────────────────────────────────────────────────────────
Steps:
  1. Load data and inspect shape, dtypes, memory
  2. Missing value audit
  3. Duplicate row check
  4. Outlier detection via IQR method
  5. Class imbalance analysis
  6. Summary statistics
  7. Visualisations: distributions, correlations, trends

Run from project root:
    python notebooks/01_eda.py
"""
import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from src.utils import *

print('=' * 60)
print('  01 — Data Integrity & Exploratory Data Analysis')
print('=' * 60)

# ── 1. Load & inspect ─────────────────────────────────────────
df = load_data()
print(f'\n[1] Shape      : {df.shape[0]:,} rows × {df.shape[1]} columns')
print(f'    Memory     : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB')
print(f'    Dtypes     : {dict(df.dtypes.value_counts())}')

# ── 2. Missing values ─────────────────────────────────────────
missing = df.isnull().sum()
print(f'\n[2] Missing values : {missing.sum()} total')
if missing.sum() > 0:
    print(missing[missing > 0].to_string())
else:
    print('    None detected ✓')

# ── 3. Duplicates ─────────────────────────────────────────────
dupes = df.duplicated().sum()
print(f'\n[3] Duplicate rows : {dupes}')

# ── 4. Outlier detection (IQR) ────────────────────────────────
print('\n[4] Outlier detection — IQR method:')
num_cols = ['LIMIT_BAL', 'AGE'] + BILL_COLS + PAY_COLS
for col in num_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR    = Q3 - Q1
    n_out  = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    if n_out > 0:
        print(f'    {col:<15s}: {n_out:>4} outliers  ({n_out/len(df):.1%})')

# ── 5. Class imbalance ────────────────────────────────────────
dr = df[TARGET].mean()
print(f'\n[5] Class distribution:')
print(f'    Non-default : {(df[TARGET]==0).sum():>6,}  ({1-dr:.1%})')
print(f'    Default     : {(df[TARGET]==1).sum():>6,}  ({dr:.1%})')
print(f'    Imbalance   : {(1-dr)/dr:.1f}:1')

# ── 6. Summary statistics ─────────────────────────────────────
print(f'\n[6] Summary statistics:')
print(df[['LIMIT_BAL', 'AGE', 'BILL_AMT1', 'PAY_AMT1']].describe().round(0).to_string())

# ── FIGURE 1: Distributions & class breakdown ─────────────────
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.35)

# Class distribution
ax0   = fig.add_subplot(gs[0, 0])
cnts  = df[TARGET].value_counts()
bars  = ax0.bar(['Non-default', 'Default'], cnts.values,
                color=[PALETTE['non_default'], PALETTE['default']], width=0.5)
for bar, v in zip(bars, cnts.values):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 150,
             f'{v:,}\n({v/len(df):.1%})', ha='center', fontsize=9)
ax0.set_title('Class distribution', fontweight='bold')
ax0.set_ylabel('Count')

# Age distribution
ax1 = fig.add_subplot(gs[0, 1])
for lbl, col, nm in [(0, PALETTE['non_default'], 'Non-default'),
                      (1, PALETTE['default'],     'Default')]:
    ax1.hist(df[df[TARGET]==lbl]['AGE'], bins=30, alpha=0.6,
             color=col, label=nm, density=True)
ax1.set_title('Age by outcome', fontweight='bold')
ax1.set_xlabel('Age'); ax1.legend(fontsize=9)

# Credit limit
ax2 = fig.add_subplot(gs[0, 2])
for lbl, col, nm in [(0, PALETTE['non_default'], 'Non-default'),
                      (1, PALETTE['default'],     'Default')]:
    ax2.hist(df[df[TARGET]==lbl]['LIMIT_BAL']/1000, bins=30, alpha=0.6,
             color=col, label=nm, density=True)
ax2.set_title('Credit limit by outcome', fontweight='bold')
ax2.set_xlabel('Credit limit (K NT$)'); ax2.legend(fontsize=9)

# Default rate by education
ax3  = fig.add_subplot(gs[1, 0])
edu  = df.groupby('EDUCATION')[TARGET].mean().reindex([1,2,3,4])
bars3= ax3.bar(['Graduate','University','High School','Other'],
               edu.values * 100,
               color=PALETTE['highlight'], alpha=0.85)
for bar, v in zip(bars3, edu.values):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
             f'{v:.1%}', ha='center', fontsize=9)
ax3.set_title('Default rate by education', fontweight='bold')
ax3.set_ylabel('Default rate (%)'); ax3.tick_params(axis='x', rotation=15)

# Default rate by marriage
ax4  = fig.add_subplot(gs[1, 1])
mar  = df.groupby('MARRIAGE')[TARGET].mean().reindex([1,2,3])
bars4= ax4.bar(['Married','Single','Other'], mar.values * 100,
               color=PALETTE['accent'], alpha=0.85)
for bar, v in zip(bars4, mar.values):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
             f'{v:.1%}', ha='center', fontsize=9)
ax4.set_title('Default rate by marital status', fontweight='bold')
ax4.set_ylabel('Default rate (%)')

# Payment status distribution
ax5   = fig.add_subplot(gs[1, 2])
ps    = df['PAY_0'].value_counts().sort_index()
cols5 = [PALETTE['non_default'] if i <= 0 else PALETTE['default']
         for i in ps.index]
ax5.bar(ps.index, ps.values, color=cols5, alpha=0.85)
ax5.set_title('Payment status — most recent month', fontweight='bold')
ax5.set_xlabel('Status (-1=on time, 1-9=months late)')
ax5.set_ylabel('Count')

fig.suptitle('Exploratory Data Analysis', fontsize=14, fontweight='bold')
save_fig(fig, '01a_eda_distributions')

# ── FIGURE 2: Correlation heatmap ─────────────────────────────
fig2, ax = plt.subplots(figsize=(13, 9))
cols_h   = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2',
            'BILL_AMT1', 'PAY_AMT1', 'PAY_AMT2', TARGET]
corr     = df[cols_h].corr()
mask     = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, ax=ax,
            linewidths=0.5, annot_kws={'size': 9},
            vmin=-0.6, vmax=0.6)
ax.set_title('Correlation heatmap — key features', fontweight='bold', fontsize=13)
plt.tight_layout()
save_fig(fig2, '01b_correlation_heatmap')

# ── FIGURE 3: Monthly trends ──────────────────────────────────
months = ['M-6', 'M-5', 'M-4', 'M-3', 'M-2', 'M-1']
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
for lbl, col, nm in [(0, PALETTE['non_default'], 'Non-default'),
                      (1, PALETTE['default'],     'Default')]:
    sub = df[df[TARGET] == lbl]
    axes3[0].plot(months, sub[BILL_COLS].mean().values / 1000,
                  marker='o', color=col, label=nm, linewidth=2)
    axes3[1].plot(months, sub[PAY_COLS].mean().values / 1000,
                  marker='o', color=col, label=nm, linewidth=2)
axes3[0].set_title('Avg bill amount', fontweight='bold')
axes3[0].set_ylabel('NT$ (thousands)'); axes3[0].legend()
axes3[0].tick_params(axis='x', rotation=20)
axes3[1].set_title('Avg payment amount', fontweight='bold')
axes3[1].set_ylabel('NT$ (thousands)'); axes3[1].legend()
axes3[1].tick_params(axis='x', rotation=20)
fig3.suptitle('Payment behaviour — 6-month trend by outcome',
              fontweight='bold', fontsize=13)
plt.tight_layout()
save_fig(fig3, '01c_payment_trends')

print('\n[✓] EDA complete')
