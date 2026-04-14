"""
Notebook 03 — Feature Engineering
───────────────────────────────────
Steps:
  1. Ratio features  (utilisation, pay-to-bill, revolving balance)
  2. Delinquency features  (streak, max delay, trends)
  3. WoE encoding for categorical variables
  4. Feature importance preview (Random Forest)

Run from project root:
    python notebooks/03_feature_engineering.py
"""
import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from src.utils import *

print('=' * 60)
print('  03 — Feature Engineering')
print('=' * 60)

df = pd.read_csv('data/credit_card_with_clusters.csv')

# ── 1. Ratio features ─────────────────────────────────────────
section('1. Derived ratio features')

# Credit utilisation — how much of the limit is being used
df['util_ratio']         = (df['BILL_AMT1'] /
                             df['LIMIT_BAL'].replace(0, np.nan)).clip(0, 2).fillna(0)

# Average utilisation across all 6 months
df['avg_util_6m']        = (df[BILL_COLS].mean(axis=1) /
                             df['LIMIT_BAL'].replace(0, np.nan)).clip(0, 2).fillna(0)

# Payment coverage — fraction of bill that was paid last month
df['pay_to_bill_ratio']  = (df['PAY_AMT1'] /
                             df['BILL_AMT1'].replace(0, np.nan)).clip(0, 5).fillna(1)

# Average payment coverage across 6 months
ratio_matrix             = (df[PAY_COLS].values /
                             df[BILL_COLS].replace(0, np.nan).values)
df['avg_pay_to_bill_6m'] = pd.DataFrame(
    np.clip(ratio_matrix, 0, 5)).fillna(1).mean(axis=1)

# Revolving balance ratio — unpaid balance relative to limit
df['revolving_ratio']    = (
    (df['BILL_AMT1'] - df['PAY_AMT1']).clip(lower=0) /
    df['LIMIT_BAL'].replace(0, np.nan)
).clip(0, 2).fillna(0)

# ── 2. Delinquency features ───────────────────────────────────
section('2. Delinquency & trend features')

ps_df = df[PAY_STATUS].clip(lower=0)

# Count of months with any late payment
df['delinq_streak']     = (ps_df > 0).astype(int).sum(axis=1)

# Worst single month delay
df['max_delay_months']  = ps_df.max(axis=1)

# Bill trend: positive = bills increasing (more risk)
df['bill_trend']        = df['BILL_AMT1'] - df['BILL_AMT6']

# Payment trend: negative = paying less over time (more risk)
df['pay_trend']         = df['PAY_AMT1'] - df['PAY_AMT6']

# Average payment size over 6 months
df['avg_payment_6m']    = df[PAY_COLS].mean(axis=1)

# Total exposure over 6 months
df['total_bill_6m']     = df[BILL_COLS].sum(axis=1)

new_features = [
    'util_ratio', 'avg_util_6m', 'pay_to_bill_ratio',
    'avg_pay_to_bill_6m', 'revolving_ratio',
    'delinq_streak', 'max_delay_months',
    'bill_trend', 'pay_trend', 'avg_payment_6m', 'total_bill_6m',
]
print(f'  Created {len(new_features)} features')

# ── 3. Weight of Evidence encoding ───────────────────────────
section('3. Weight of Evidence (WoE) encoding')

def compute_woe(df, col, target):
    """
    WoE = ln(Distribution of Events / Distribution of Non-Events)
    IV  = sum((Events_dist - NonEvents_dist) * WoE)
    IV < 0.02 = weak predictor
    IV 0.02-0.1 = moderate
    IV > 0.3 = strong
    """
    total_ev  = df[target].sum()
    total_nev = len(df) - total_ev
    woe_map, iv = {}, 0
    for val in df[col].unique():
        sub    = df[df[col] == val]
        ev     = max(sub[target].sum(), 0.5)
        nev    = max(len(sub) - sub[target].sum(), 0.5)
        d_ev   = ev  / total_ev
        d_nev  = nev / total_nev
        woe    = np.log(d_ev / d_nev)
        iv    += (d_ev - d_nev) * woe
        woe_map[val] = round(woe, 4)
    return woe_map, round(iv, 4)

woe_results = {}
for col in CAT_COLS:
    woe_map, iv = compute_woe(df, col, TARGET)
    df[f'{col}_WoE']  = df[col].map(woe_map)
    woe_results[col]  = {'woe_map': woe_map, 'iv': iv}
    strength = ('strong' if iv > 0.3 else 'moderate' if iv > 0.1
                else 'weak' if iv > 0.02 else 'negligible')
    print(f'  {col:<12s}  IV={iv:.4f} ({strength})  WoE={woe_map}')

df['cluster_label'] = df['cluster'].astype(int)

# ── 4. Save engineered dataset ───────────────────────────────
all_features = (
    ['LIMIT_BAL', 'AGE'] + PAY_STATUS +
    ['BILL_AMT1','BILL_AMT2','BILL_AMT3',
     'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3'] +
    new_features +
    [f'{c}_WoE' for c in CAT_COLS] +
    ['cluster_label']
)
df.to_csv('data/credit_card_engineered.csv', index=False)
print(f'\n  Engineered dataset saved -> data/credit_card_engineered.csv')
print(f'  Total modelling features: {len(all_features)}')

# ── 5. Feature importance preview ────────────────────────────
section('4. Feature importance preview (Random Forest)')

feat_cols = [c for c in all_features if c in df.columns]
X = df[feat_cols].fillna(0)
y = df[TARGET]
rf = RandomForestClassifier(
    n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)
rf.fit(X, y)
imp = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)
print(f'\n  Top 10 features:')
print(imp.head(10).round(4).to_string())

# ── FIGURE 1: Feature importance ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
imp.head(15).sort_values().plot.barh(
    ax=ax, color=PALETTE['highlight'], alpha=0.85)
ax.set_title('Top 15 feature importances (Random Forest)',
             fontweight='bold', fontsize=13)
ax.set_xlabel('Mean decrease in impurity')
plt.tight_layout()
save_fig(fig, '03a_feature_importance')

# ── FIGURE 2: WoE bars ────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes2, CAT_COLS):
    woe_map = woe_results[col]['woe_map']
    iv      = woe_results[col]['iv']
    keys    = sorted(woe_map)
    labels  = [LABEL_MAPS[col].get(k, str(k)) for k in keys]
    vals    = [woe_map[k] for k in keys]
    colors  = [PALETTE['default'] if v < 0 else PALETTE['non_default']
               for v in vals]
    ax.bar(labels, vals, color=colors, alpha=0.85)
    ax.axhline(0, color='gray', linewidth=0.8)
    ax.set_title(f'{col}  (IV={iv:.3f})', fontweight='bold', fontsize=10)
    ax.tick_params(axis='x', rotation=15)
fig2.suptitle('Weight of Evidence by categorical feature', fontweight='bold')
plt.tight_layout()
save_fig(fig2, '03b_woe_encoding')

# ── FIGURE 3: Engineered feature distributions ───────────────
plot_feats = ['util_ratio', 'pay_to_bill_ratio', 'delinq_streak',
              'max_delay_months', 'bill_trend', 'revolving_ratio']
fig3, axes3 = plt.subplots(2, 3, figsize=(15, 8))
for ax, feat in zip(axes3.flatten(), plot_feats):
    for lbl, col, nm in [(0, PALETTE['non_default'], 'Non-default'),
                          (1, PALETTE['default'],     'Default')]:
        vals = df[df[TARGET]==lbl][feat].clip(
            df[feat].quantile(0.01), df[feat].quantile(0.99))
        ax.hist(vals, bins=40, alpha=0.55, color=col,
                label=nm, density=True)
    ax.set_title(feat, fontweight='bold', fontsize=10)
    ax.legend(fontsize=8)
fig3.suptitle('Engineered feature distributions by outcome',
              fontweight='bold', fontsize=13)
plt.tight_layout()
save_fig(fig3, '03c_engineered_features')

print('\n[✓] Feature engineering complete')
