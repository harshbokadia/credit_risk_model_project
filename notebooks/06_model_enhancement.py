"""
Notebook 06 — Model Enhancement
Three-stage AUC improvement pipeline:

  Stage 1 — Feature selection  (remove noise features)
  Stage 2 — Optuna tuning      (XGBoost + Random Forest, 50+30 trials)
  Stage 3 — Stacking ensemble  (meta-learner on tuned base models)

Run from project root:
    python notebooks/06_model_enhancement.py

Expected runtime on a modern laptop: 8-15 minutes
Expected AUC improvement: +0.02 to +0.06 over baseline
"""
import sys; sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib, warnings, time
warnings.filterwarnings('ignore')

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from src.utils import *
from matplotlib.patches import Patch

print("=" * 65)
print("  06 — Model Enhancement: Selection · Tuning · Stacking")
print("=" * 65)

# ── Load ──────────────────────────────────────────────────────
df           = pd.read_csv('data/credit_card_engineered.csv')
meta         = joblib.load('outputs/model_metadata.pkl')
FEATURE_COLS = meta['feature_cols']
baseline_auc = meta['metrics']['XGBoost']['AUC']

X = df[FEATURE_COLS].fillna(0)
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

sm = SMOTE(random_state=42, k_neighbors=5)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

print(f"\n  Baseline XGBoost AUC : {baseline_auc:.4f}")
print(f"  Features             : {len(FEATURE_COLS)}")
print(f"  Training samples     : {len(X_train_sm):,}  (after SMOTE)")

# ══════════════════════════════════════════════════════════════
# STAGE 1 — Feature selection
# ══════════════════════════════════════════════════════════════
print("\n" + "-" * 65)
print("  STAGE 1 — Feature Selection")
print("-" * 65)

rf_sel = RandomForestClassifier(
    n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
rf_sel.fit(X_train_sm, y_train_sm)

imp       = pd.Series(rf_sel.feature_importances_, index=FEATURE_COLS)
threshold = imp.mean() * 0.5
SELECTED  = imp[imp >= threshold].sort_values(ascending=False).index.tolist()

print(f"\n  Original : {len(FEATURE_COLS)}  ->  Selected : {len(SELECTED)}")
print(f"  Top 10: {SELECTED[:10]}")

X_tr = X_train_sm[SELECTED]
X_te = X_test[SELECTED]

check = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                            eval_metric='logloss', verbosity=0, random_state=42)
check.fit(X_tr, y_train_sm)
auc_sel = roc_auc_score(y_test, check.predict_proba(X_te)[:,1])
print(f"  AUC after selection : {auc_sel:.4f}  ({auc_sel-baseline_auc:+.4f})")

# ══════════════════════════════════════════════════════════════
# STAGE 2 — Optuna tuning
# ══════════════════════════════════════════════════════════════
print("\n" + "-" * 65)
print("  STAGE 2 — Optuna Hyperparameter Tuning")
print("-" * 65)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# XGBoost — 50 trials
print("\n  [XGBoost — 50 trials]...")
def xgb_obj(trial):
    p = dict(
        n_estimators     = trial.suggest_int('n_estimators', 200, 600),
        max_depth        = trial.suggest_int('max_depth', 3, 8),
        learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        subsample        = trial.suggest_float('subsample', 0.6, 1.0),
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0),
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10),
        reg_alpha        = trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
        reg_lambda       = trial.suggest_float('reg_lambda', 1e-4, 5.0, log=True),
        gamma            = trial.suggest_float('gamma', 0.0, 2.0),
        eval_metric='logloss', verbosity=0, random_state=42, n_jobs=-1
    )
    s = cross_val_score(xgb.XGBClassifier(**p), X_tr, y_train_sm,
                        cv=cv, scoring='roc_auc', n_jobs=-1)
    return s.mean()

t0 = time.time()
xgb_study = optuna.create_study(direction='maximize',
                                  sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(xgb_obj, n_trials=50)
print(f"  Best CV AUC : {xgb_study.best_value:.4f}  ({time.time()-t0:.0f}s)")
print(f"  Params : {xgb_study.best_params}")

best_xgb = xgb.XGBClassifier(**xgb_study.best_params,
                               eval_metric='logloss', verbosity=0, random_state=42)
best_xgb.fit(X_tr, y_train_sm)
auc_xgb = roc_auc_score(y_test, best_xgb.predict_proba(X_te)[:,1])
print(f"  Test AUC : {auc_xgb:.4f}  ({auc_xgb-baseline_auc:+.4f})")

# Random Forest — 30 trials
print("\n  [Random Forest — 30 trials]...")
def rf_obj(trial):
    p = dict(
        n_estimators     = trial.suggest_int('n_estimators', 200, 600),
        max_depth        = trial.suggest_int('max_depth', 5, 20),
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 5, 40),
        max_features     = trial.suggest_float('max_features', 0.3, 0.9),
        n_jobs=-1, random_state=42
    )
    s = cross_val_score(RandomForestClassifier(**p), X_tr, y_train_sm,
                        cv=cv, scoring='roc_auc', n_jobs=-1)
    return s.mean()

t0 = time.time()
rf_study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
rf_study.optimize(rf_obj, n_trials=30)
print(f"  Best CV AUC : {rf_study.best_value:.4f}  ({time.time()-t0:.0f}s)")

best_rf = RandomForestClassifier(**rf_study.best_params, random_state=42, n_jobs=-1)
best_rf.fit(X_tr, y_train_sm)
auc_rf = roc_auc_score(y_test, best_rf.predict_proba(X_te)[:,1])
print(f"  Test AUC : {auc_rf:.4f}  ({auc_rf-baseline_auc:+.4f})")

# LR base
scaler_sel = StandardScaler()
X_tr_sc    = scaler_sel.fit_transform(X_tr)
X_te_sc    = scaler_sel.transform(X_te)
best_lr    = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
best_lr.fit(X_tr_sc, y_train_sm)
auc_lr     = roc_auc_score(y_test, best_lr.predict_proba(X_te_sc)[:,1])
print(f"\n  LR AUC : {auc_lr:.4f}")

# ══════════════════════════════════════════════════════════════
# STAGE 3 — Stacking ensemble
# ══════════════════════════════════════════════════════════════
print("\n" + "-" * 65)
print("  STAGE 3 — Stacking Ensemble")
print("-" * 65)
print("  [XGBoost_prob, RF_prob, LR_prob] -> meta LogisticRegression")

t0 = time.time()
stack = StackingClassifier(
    estimators=[
        ('xgb', xgb.XGBClassifier(**xgb_study.best_params,
                                    eval_metric='logloss', verbosity=0, random_state=42)),
        ('rf',  RandomForestClassifier(**rf_study.best_params, random_state=42, n_jobs=-1)),
        ('lr',  LogisticRegression(C=0.5, max_iter=1000, random_state=42)),
    ],
    final_estimator = LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    cv=5, stack_method='predict_proba', passthrough=False, n_jobs=-1
)
stack.fit(X_tr, y_train_sm)
auc_stack = roc_auc_score(y_test, stack.predict_proba(X_te)[:,1])
print(f"  Stacking AUC : {auc_stack:.4f}  ({auc_stack-baseline_auc:+.4f})  ({time.time()-t0:.0f}s)")

# ── Results ───────────────────────────────────────────────────
results = {
    'Baseline XGBoost':         baseline_auc,
    'After feature selection':  auc_sel,
    'Tuned XGBoost (Optuna)':   auc_xgb,
    'Tuned Random Forest':      auc_rf,
    'Logistic Regression':      auc_lr,
    'Stacking Ensemble':        auc_stack,
}
best_k = max(results, key=results.get)
best_v = results[best_k]

print("\n" + "=" * 65)
print("  FINAL RESULTS")
print("=" * 65)
for k, v in results.items():
    tag   = "  <- BEST" if k == best_k else ""
    delta = f"  ({v-baseline_auc:+.4f})" if k != 'Baseline XGBoost' else ""
    print(f"  {k:<35s}  {v:.4f}{delta}{tag}")
print(f"\n  Net improvement : {baseline_auc:.4f} -> {best_v:.4f}"
      f"  ({best_v-baseline_auc:+.4f} / {(best_v-baseline_auc)/baseline_auc*100:+.1f}%)")

# Save
joblib.dump(stack,      'outputs/best_model_enhanced.pkl')
joblib.dump(scaler_sel, 'outputs/scaler_enhanced.pkl')
joblib.dump(SELECTED,   'outputs/selected_features.pkl')
joblib.dump({'results': results, 'best_model': best_k, 'best_auc': best_v,
             'xgb_best_params': xgb_study.best_params,
             'rf_best_params':  rf_study.best_params,
             'selected_features': SELECTED}, 'outputs/enhancement_meta.pkl')
print("\n  Saved -> outputs/best_model_enhanced.pkl")

# ── Charts ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
ax = axes[0]
short  = ['Baseline\nXGB','After\nSelection','Tuned\nXGB','Tuned\nRF','LR','Stacking']
colors = ['#888780','#378ADD','#534AB7','#1D9E75','#EF9F27','#E24B4A']
bars   = ax.bar(short, list(results.values()), color=colors, alpha=0.85, width=0.55)
for bar, v in zip(bars, results.values()):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
            f'{v:.4f}', ha='center', va='bottom', fontsize=9)
ax.axhline(baseline_auc, color='gray', linestyle='--', linewidth=1.2,
           label=f'Baseline ({baseline_auc:.4f})')
ax.set_ylim(min(results.values())*0.97, max(results.values())*1.025)
ax.set_ylabel('ROC-AUC')
ax.set_title('AUC at each enhancement stage', fontweight='bold')
ax.legend(fontsize=9)
ax.set_facecolor('#F8F8F6'); fig.patch.set_facecolor('#F8F8F6')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# ROC curves
ax2 = axes[1]
for mdl, Xin, name, col, lw in [
    (check,    X_te, 'Baseline XGBoost',    '#888780', 1.5),
    (best_xgb, X_te, 'Tuned XGBoost',       '#534AB7', 2.0),
    (best_rf,  X_te, 'Tuned Random Forest', '#1D9E75', 2.0),
    (stack,    X_te, 'Stacking Ensemble',   '#E24B4A', 2.8),
]:
    prob     = mdl.predict_proba(Xin)[:,1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax2.plot(fpr, tpr, color=col, linewidth=lw,
             label=f'{name} ({roc_auc_score(y_test,prob):.4f})')
ax2.plot([0,1],[0,1], '--', color='gray', linewidth=1)
ax2.set_xlabel('False positive rate'); ax2.set_ylabel('True positive rate')
ax2.set_title('ROC curves — baseline vs enhanced', fontweight='bold')
ax2.legend(fontsize=8.5)
ax2.set_facecolor('#F8F8F6'); fig.patch.set_facecolor('#F8F8F6')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
plt.suptitle('Model Enhancement Results', fontweight='bold', fontsize=13)
plt.tight_layout()
save_fig(fig, '06a_enhancement_results')

# Optuna history
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4))
for ax, study, label, col in zip(axes2,
    [xgb_study, rf_study], ['XGBoost (50 trials)','Random Forest (30 trials)'],
    [PALETTE['highlight'], PALETTE['accent']]):
    vals = [t.value for t in study.trials]
    ax.scatter(range(len(vals)), vals, s=18, alpha=0.45, color=PALETTE['neutral'])
    ax.plot(range(len(vals)), pd.Series(vals).cummax().values,
            color=col, linewidth=2.2, label='Best so far')
    ax.set_xlabel('Trial'); ax.set_ylabel('CV AUC')
    ax.set_title(f'Optuna search — {label}', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_facecolor('#F8F8F6'); fig2.patch.set_facecolor('#F8F8F6')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.suptitle('Optuna search history', fontweight='bold')
plt.tight_layout()
save_fig(fig2, '06b_optuna_trials')

# Feature selection
fig3, ax3 = plt.subplots(figsize=(10, 6))
imp_s  = imp.sort_values()
cols3  = ['#1D9E75' if f in SELECTED else '#888780' for f in imp_s.index]
imp_s.plot.barh(ax=ax3, color=cols3, alpha=0.85)
ax3.axvline(threshold, color=PALETTE['default'], linestyle='--', linewidth=1.3)
ax3.legend(handles=[
    Patch(facecolor='#1D9E75', label=f'Kept ({len(SELECTED)})'),
    Patch(facecolor='#888780', label=f'Dropped ({len(FEATURE_COLS)-len(SELECTED)})'),
], fontsize=9)
ax3.set_title('Feature importance — selected vs dropped', fontweight='bold')
ax3.set_xlabel('Mean decrease in impurity')
ax3.set_facecolor('#F8F8F6'); fig3.patch.set_facecolor('#F8F8F6')
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
plt.tight_layout()
save_fig(fig3, '06c_feature_selection')

print("\n[done] Charts -> outputs/06a, 06b, 06c")
