# Model Card — Credit Card Default Prediction

## Model overview
| Field | Value |
|---|---|
| Model name | Credit Default Risk Scorer v1.0 |
| Model type | Binary classification (default / non-default) |
| Primary algorithm | XGBoost (gradient-boosted trees) |
| Training data | 30,000 credit card accounts (6-month history) |
| Target variable | `default.payment.next.month` (1 = default) |
| Development date | 2024 |
| Owner | Data Science Team |

## Intended use
- **Primary use:** Score credit card applicants and existing customers for default risk
- **Intended users:** Credit risk analysts, underwriting teams
- **Out-of-scope:** Not intended for mortgage/auto lending or non-credit products

## Training data
- **Source:** UCI Machine Learning Repository — Default of Credit Card Clients
- **Size:** 30,000 records, 23 raw features → 35 engineered features
- **Time period:** 6 months of payment history
- **Class balance:** ~13.7% default rate (handled via SMOTE oversampling)
- **Geography:** Taiwan credit card market (NT$ denominated)

## Model performance (held-out test set, 20%)
| Metric | Value |
|---|---|
| ROC-AUC | ~0.78 |
| KS Statistic | ~0.43 |
| Gini Coefficient | ~0.56 |

## Features used
**Top predictors (SHAP-ranked):**
1. `max_delay_months` — worst payment delay in past 6 months
2. `delinq_streak` — consecutive months with late payment
3. `PAY_0` — most recent payment status
4. `util_ratio` — credit utilization ratio
5. `revolving_ratio` — unpaid balance as % of credit limit
6. `pay_to_bill_ratio` — payment vs. bill coverage
7. `avg_util_6m` — average utilization over 6 months
8. `LIMIT_BAL` — credit limit

## Known limitations and risks
1. **Geographic bias:** Trained on Taiwan data; may not generalize to other markets without recalibration
2. **Temporal drift:** Payment behavior patterns change over economic cycles; monthly PSI monitoring required
3. **Class imbalance:** Low base rate (~14%) means precision at high recall thresholds is limited
4. **Fairness:** SEX, MARRIAGE, and EDUCATION are used only as WoE-encoded features; direct demographic discrimination is not intended but should be audited quarterly
5. **Proxy discrimination:** LIMIT_BAL may act as a proxy for wealth/demographics

## Model risk controls (1st line of defense)
- [x] Independent train/test split (80/20, stratified)
- [x] Cross-validation (5-fold) for hyperparameter selection
- [x] Learning curve analysis to diagnose overfitting
- [x] SHAP explainability for every prediction
- [x] Calibration check (Brier score)
- [x] Business impact simulation before deployment

## Monitoring plan
| Metric | Frequency | Alert threshold |
|---|---|---|
| Population Stability Index (PSI) | Monthly | PSI > 0.25 → retrain |
| Characteristic Stability Index (CSI) | Monthly | CSI > 0.25 per feature |
| Default rate drift | Monthly | ±3% from baseline |
| ROC-AUC on recent cohort | Quarterly | AUC drop > 0.03 |
| SHAP feature rank stability | Quarterly | Top-5 rank change |

## Ethical considerations
- Model decisions should always be reviewable by a human analyst
- Declined customers should have an appeal pathway
- Regulatory compliance: model documentation maintained per SR 11-7 guidance
- Fairness testing across SEX, EDUCATION, and MARRIAGE groups required before production

## Data lineage
```
raw: credit_card_default.csv (UCI)
  → processed: credit_card_with_clusters.csv  (02_segmentation.py)
  → engineered: credit_card_engineered.csv    (03_feature_engineering.py)
  → models:     outputs/best_model.pkl        (04_modeling.py)
  → impact:     outputs/business_impact_simulation.csv (05_business_impact.py)
```

---
*This model card follows Google Model Card and Anthropic best practices for transparent ML documentation.*
