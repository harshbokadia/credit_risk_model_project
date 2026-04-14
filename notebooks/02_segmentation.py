"""
Notebook 02 — Customer Segmentation (Unsupervised Learning)
────────────────────────────────────────────────────────────
Steps:
  1. Build clustering features (utilisation, payment behaviour)
  2. Elbow method to compare inertia across k values
  3. Silhouette score to select optimal k
  4. Fit final K-Means, label clusters as risk personas
  5. PCA 2D projection for visualisation
  6. Cluster profile comparison

Run from project root:
    python notebooks/02_segmentation.py
"""
import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from src.utils import *

print('=' * 60)
print('  02 — Customer Segmentation via K-Means Clustering')
print('=' * 60)

df = load_data()

# ── 1. Clustering features ────────────────────────────────────
section('1. Feature preparation')

X_c = df[['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3',
           'BILL_AMT1', 'BILL_AMT2', 'PAY_AMT1', 'PAY_AMT2']].copy()

X_c['utilization']   = (X_c['BILL_AMT1'] /
                         X_c['LIMIT_BAL'].replace(0, np.nan)).clip(0, 2).fillna(0)
X_c['pay_to_bill']   = (X_c['PAY_AMT1'] /
                         X_c['BILL_AMT1'].replace(0, np.nan)).clip(0, 3).fillna(1)
X_c['avg_delay']     = X_c[['PAY_0','PAY_2','PAY_3']].clip(lower=0).mean(axis=1)

CLUST_FEATS = ['LIMIT_BAL', 'AGE', 'utilization', 'pay_to_bill', 'avg_delay']
X_scaled    = StandardScaler().fit_transform(X_c[CLUST_FEATS])
print(f'  Clustering on {len(CLUST_FEATS)} features: {CLUST_FEATS}')

# ── 2+3. Elbow + Silhouette ───────────────────────────────────
section('2. Optimal k selection')

k_range, inertias, silhouettes = range(2, 10), [], []
for k in k_range:
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, lbl, sample_size=3000, random_state=42)
    silhouettes.append(sil)
    print(f'  k={k}  inertia={km.inertia_:>9,.0f}  silhouette={sil:.3f}')

best_k = list(k_range)[np.argmax(silhouettes)]
print(f'\n  Best k (max silhouette): {best_k}')

# ── 4. Final clustering & personas ───────────────────────────
section('3. Final clustering & persona labelling')

km_final     = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster']= km_final.fit_predict(X_scaled)
df['utilization'] = X_c['utilization']
df['pay_to_bill']  = X_c['pay_to_bill']
df['avg_delay']    = X_c['avg_delay']

profile_cols = ['LIMIT_BAL','AGE','utilization','pay_to_bill','avg_delay',TARGET]
profile      = df.groupby('cluster')[profile_cols].mean().round(3)
profile['size']     = df['cluster'].value_counts().sort_index()
profile['size_pct'] = (profile['size'] / len(df) * 100).round(1)

def assign_persona(row):
    if row['avg_delay'] < 0.2 and row['pay_to_bill'] > 0.8:
        return 'Transactor (Low Risk)'
    elif row['avg_delay'] > 1.0 or row['utilization'] > 0.7:
        return 'Delinquent (High Risk)'
    elif row['avg_delay'] < 0.5 and row['utilization'] < 0.5:
        return 'Revolver (Medium Risk)'
    else:
        return 'Dormant / Cautious'

profile['persona'] = profile.apply(assign_persona, axis=1)
df['persona']      = df['cluster'].map(profile['persona'])

print(f'\n  Cluster profiles:')
print(profile.to_string())

df.to_csv('data/credit_card_with_clusters.csv', index=False)
print('\n  Saved -> data/credit_card_with_clusters.csv')

# ── FIGURE 1: Elbow + Silhouette ─────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(list(k_range), inertias, 'o-',
         color=PALETTE['highlight'], linewidth=2)
ax1.set_title('Elbow method — inertia vs k', fontweight='bold')
ax1.set_xlabel('k'); ax1.set_ylabel('Inertia')

ax2.plot(list(k_range), silhouettes, 's-',
         color=PALETTE['accent'], linewidth=2)
ax2.axvline(best_k, color=PALETTE['default'], linestyle='--',
            label=f'Best k={best_k}')
ax2.set_title('Silhouette score vs k', fontweight='bold')
ax2.set_xlabel('k'); ax2.set_ylabel('Silhouette score')
ax2.legend()

fig.suptitle('Optimal cluster selection', fontweight='bold')
plt.tight_layout()
save_fig(fig, '02a_elbow_silhouette')

# ── FIGURE 2: PCA 2D projection ───────────────────────────────
pca    = PCA(n_components=2, random_state=42)
X_pca  = pca.fit_transform(X_scaled)
colors = ['#378ADD','#E24B4A','#1D9E75','#EF9F27','#534AB7']

fig2, ax = plt.subplots(figsize=(10, 7))
for i in range(best_k):
    mask    = df['cluster'] == i
    persona = profile.loc[i, 'persona']
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               s=8, alpha=0.35, color=colors[i % len(colors)],
               label=f'Cluster {i}: {persona}')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax.set_title('Customer segments — PCA 2D projection',
             fontweight='bold', fontsize=13)
ax.legend(fontsize=9, markerscale=3)
plt.tight_layout()
save_fig(fig2, '02b_cluster_pca')

# ── FIGURE 3: Cluster profile bars ───────────────────────────
metrics = ['utilization', 'pay_to_bill', 'avg_delay', TARGET]
labels  = ['Avg utilisation', 'Pay-to-bill ratio',
           'Avg payment delay', 'Default rate']
fig3, axes3 = plt.subplots(1, 4, figsize=(16, 5))
for ax, col, lbl in zip(axes3, metrics, labels):
    vals  = [profile.loc[i, col] for i in range(best_k)]
    names = [f'C{i}' for i in range(best_k)]
    bars  = ax.bar(names, vals,
                   color=[colors[i % len(colors)] for i in range(best_k)],
                   alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.005,
                f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_title(lbl, fontweight='bold', fontsize=10)

fig3.suptitle('Cluster profile comparison', fontweight='bold', fontsize=13)
plt.tight_layout()
save_fig(fig3, '02c_cluster_profiles')

print('\n[✓] Segmentation complete')
