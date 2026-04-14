"""
Notebook 04 — Model Training, Validation & SHAP Explainability
───────────────────────────────────────────────────────────────
Models trained:
  1. Logistic Regression   — classic credit scoring baseline
  2. Decision Tree         — interpretable rule-based model
  3. Random Forest         — bagging ensemble
  4. XGBoost               — gradient-boosted trees
  5. MLP Neural Network    — feedforward neural net

Techniques:
  - SMOTE to handle class imbalance
  - Stratified K-Fold cross-validation
  - Learning curves for overfitting diagnosis
  - SHAP TreeExplainer for XGBoost explainability
  - Calibration and confusion matrix analysis

Run from project root:
    python notebooks/04_modeling.py
"""
import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, learning_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, roc_curve,
                              precision_recall_curve,
                              average_precision_score,
                              brier_score_loss,
                              confusion_matrix)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
from src.utils import *

print('=' * 60)
print('  04 — Model Training, Validation & Explainability')
print('=' * 60)

# ── Load ──────────────────────────────────────────────────────
df = pd.read_csv('data/credit_card_engineered.csv')

FEATURE_COLS = (
    ['LIMIT_BAL', 'AGE'] + PAY_STATUS +
    ['BILL_AMT1','BILL_AMT2','BILL_AMT3',
     'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3'] +
    ['util_ratio','avg_util_6m','pay_to_bill_ratio','avg_pay_to_bill_6m',
     'delinq_streak','max_delay_months','bill_trend','pay_trend',
     'avg_payment_6m','revolving_ratio'] +
    [f'{c}_WoE' for c in CAT_COLS] +
    ['cluster_label']
)
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

X = df[FEATURE_COLS].fillna(0)
y = df[TARGET]

print(f'\n  Features  : {len(FEATURE_COLS)}')
print(f'  Samples   : {len(X):,}')
print(f'  Default % : {y.mean():.1%}')

# ── Split & SMOTE ─────────────────────────────────────────────
section('1. Train/test split & SMOTE')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

sm = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

print(f'  Train  : {len(X_train):,}  ->  {len(X_train_sm):,} after SMOTE')
print(f'  Test   : {len(X_test):,}   (real-world imbalance preserved)')

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_sm)
X_test_sc  = scaler.transform(X_test)
joblib.dump(scaler, 'outputs/scaler.pkl')

# ── Models ────────────────────────────────────────────────────
section('2. Training all models')

models = {
    'Logistic Regression': LogisticRegression(
        C=0.1, max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=50, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=20,
        random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
        eval_metric='logloss', verbosity=0, random_state=42),
    'MLP Neural Net': MLPClassifier(
        hidden_layer_sizes=(64, 32), activation='relu',
        alpha=0.01, max_iter=300, random_state=42,
        early_stopping=True),
}

# Models that need scaled input
NEEDS_SCALING = {'Logistic Regression', 'MLP Neural Net'}

def ks_stat(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return round(float(max(tpr - fpr)), 4)

results = {}
for name, model in models.items():
    Xtr = X_train_sc if name in NEEDS_SCALING else X_train_sm
    Xte = X_test_sc  if name in NEEDS_SCALING else X_test

    model.fit(Xtr, y_train_sm)
    prob = model.predict_proba(Xte)[:, 1]
    pred = (prob >= 0.5).astype(int)

    auc  = round(roc_auc_score(y_test, prob), 4)
    ks   = ks_stat(y_test, prob)
    gini = round(2 * auc - 1, 4)
    ap   = round(average_precision_score(y_test, prob), 4)
    bs   = round(brier_score_loss(y_test, prob), 4)
    cm   = confusion_matrix(y_test, pred)

    results[name] = {
        'AUC': auc, 'KS': ks, 'Gini': gini,
        'Avg Precision': ap, 'Brier Score': bs,
        'CM': cm, 'y_prob': prob,
    }
    print(f'  {name:<22s}  AUC={auc:.4f}  KS={ks:.4f}  Gini={gini:.4f}')

# ── Save best model ───────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['AUC'])
joblib.dump(models[best_name], 'outputs/best_model.pkl')
joblib.dump({
    'feature_cols': FEATURE_COLS,
    'best_model':   best_name,
    'metrics': {k: {m: v for m, v in v.items()
                    if m not in ('CM', 'y_prob')}
                for k, v in results.items()},
}, 'outputs/model_metadata.pkl')

res_df = pd.DataFrame({k: {m: v for m, v in v.items()
                              if m not in ('CM','y_prob')}
                         for k, v in results.items()}).T
res_df.to_csv('outputs/model_comparison.csv')
print(f'\n  Best model : {best_name}  (AUC={results[best_name]["AUC"]:.4f})')

# ── FIGURE 1: ROC + Precision-Recall ─────────────────────────
model_colors = ['#378ADD','#888780','#1D9E75','#534AB7','#E24B4A']
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for (name, res), col in zip(results.items(), model_colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax1.plot(fpr, tpr, color=col, linewidth=1.8,
             label=f'{name} ({res["AUC"]:.3f})')
    prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
    ax2.plot(rec, prec, color=col, linewidth=1.8,
             label=f'{name} (AP={res["Avg Precision"]:.3f})')

ax1.plot([0,1],[0,1], '--', color='gray', linewidth=1)
ax1.set_xlabel('False positive rate'); ax1.set_ylabel('True positive rate')
ax1.set_title('ROC curves', fontweight='bold'); ax1.legend(fontsize=8)

ax2.axhline(y_test.mean(), color='gray', linestyle='--', label='Baseline')
ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall curves', fontweight='bold')
ax2.legend(fontsize=8)

fig1.suptitle('Model evaluation curves', fontweight='bold')
plt.tight_layout()
save_fig(fig1, '04a_roc_pr_curves')

# ── FIGURE 2: Metric comparison bars ─────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
short = ['LR','DT','RF','XGB','MLP']
for ax, metric in zip(axes2, ['AUC','KS','Gini']):
    vals = [results[n][metric] for n in models]
    bars = ax.bar(short, vals, color=model_colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.003,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_title(metric, fontweight='bold')
    ax.set_ylim(0, max(vals)*1.15)
fig2.suptitle('Model performance comparison', fontweight='bold')
plt.tight_layout()
save_fig(fig2, '04b_model_comparison')

# ── FIGURE 3: Learning curves ─────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
for ax, (name, short_n) in zip(axes3, [('Random Forest','RF'),
                                         ('XGBoost','XGB')]):
    Xtr = X_train_sc if name in NEEDS_SCALING else X_train_sm.values
    tr_sz, tr_sc, val_sc = learning_curve(
        models[name], Xtr, y_train_sm,
        cv=3, scoring='roc_auc', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 8))
    ax.plot(tr_sz, tr_sc.mean(1), 'o-',
            color=PALETTE['non_default'], label='Train AUC', linewidth=2)
    ax.plot(tr_sz, val_sc.mean(1), 's-',
            color=PALETTE['default'],     label='Val AUC', linewidth=2)
    ax.fill_between(tr_sz,
                    tr_sc.mean(1)-tr_sc.std(1),
                    tr_sc.mean(1)+tr_sc.std(1),
                    alpha=0.12, color=PALETTE['non_default'])
    ax.fill_between(tr_sz,
                    val_sc.mean(1)-val_sc.std(1),
                    val_sc.mean(1)+val_sc.std(1),
                    alpha=0.12, color=PALETTE['default'])
    ax.set_title(f'Learning curve — {short_n}', fontweight='bold')
    ax.set_xlabel('Training samples'); ax.set_ylabel('ROC-AUC')
    ax.legend()
fig3.suptitle('Overfitting diagnosis via learning curves', fontweight='bold')
plt.tight_layout()
save_fig(fig3, '04c_learning_curves')

# ── FIGURE 4: SHAP explainability ────────────────────────────
section('3. SHAP explainability (XGBoost)')
xgb_model  = models['XGBoost']
explainer  = shap.TreeExplainer(xgb_model)
shap_samp  = X_test.sample(500, random_state=42)
shap_vals  = explainer.shap_values(shap_samp)

fig4, axes4 = plt.subplots(1, 2, figsize=(16, 6))

# Mean |SHAP|
shap_mean = np.abs(shap_vals).mean(axis=0)
shap_imp  = pd.Series(shap_mean, index=FEATURE_COLS).sort_values(ascending=True).tail(15)
shap_imp.plot.barh(ax=axes4[0], color=PALETTE['highlight'], alpha=0.85)
axes4[0].set_title('Mean |SHAP| — feature impact',
                    fontweight='bold', fontsize=11)
axes4[0].set_xlabel('Mean |SHAP value|')

# Beeswarm scatter
top10_idx   = np.argsort(np.abs(shap_vals).mean(0))[-10:]
top10_names = [FEATURE_COLS[i] for i in top10_idx]
for i, (idx, fname) in enumerate(zip(top10_idx, top10_names)):
    sv    = shap_vals[:, idx]
    fv    = shap_samp.iloc[:, idx].values
    fv_n  = (fv - fv.min()) / (fv.max() - fv.min() + 1e-8)
    axes4[1].scatter(sv, [i]*len(sv), c=fv_n,
                     cmap='RdBu_r', alpha=0.3, s=8)
axes4[1].set_yticks(range(10))
axes4[1].set_yticklabels(top10_names, fontsize=9)
axes4[1].axvline(0, color='gray', linewidth=0.8)
axes4[1].set_title('SHAP beeswarm — top 10 features',
                    fontweight='bold', fontsize=11)
axes4[1].set_xlabel('SHAP value  (+ = increases default probability)')

fig4.suptitle('XGBoost explainability — SHAP analysis', fontweight='bold')
plt.tight_layout()
save_fig(fig4, '04d_shap_explainability')

# ── FIGURE 5: Confusion matrices ─────────────────────────────
fig5, axes5 = plt.subplots(1, 3, figsize=(15, 4))
for ax, name in zip(axes5,
                    ['Logistic Regression','Random Forest','XGBoost']):
    sns.heatmap(results[name]['CM'], annot=True, fmt='d',
                cmap='Blues', ax=ax, cbar=False,
                xticklabels=['Pred 0','Pred 1'],
                yticklabels=['Actual 0','Actual 1'])
    ax.set_title(f'{name}\n(AUC={results[name]["AUC"]:.3f})',
                 fontweight='bold', fontsize=10)
fig5.suptitle('Confusion matrices at 0.5 threshold', fontweight='bold')
plt.tight_layout()
save_fig(fig5, '04e_confusion_matrices')

print('\n[✓] Modelling complete')
print(f'    Best model : {best_name}  AUC={results[best_name]["AUC"]:.4f}')
