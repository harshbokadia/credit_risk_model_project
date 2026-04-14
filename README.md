# Credit Risk Modelling & Customer Intelligence System

An end-to-end machine learning pipeline that predicts credit card default risk, segments customers into behavioural risk personas, and quantifies the business dollar impact of model deployment.

**Tech stack:** Python · XGBoost · Scikit-learn · SHAP · Optuna · Streamlit · SQL

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (steps 01–05)
python run_all.py --skip 06

# 3. Run model enhancement separately (~10–15 min on a laptop)
python notebooks/06_model_enhancement.py

# 4. Launch the interactive demo app
streamlit run app.py
```

> All scripts must be run from the **project root** directory.

---

## Pipeline

| Step | Script | What it does |
|------|--------|--------------|
| 01 | `notebooks/01_eda.py` | Data integrity checks, outlier detection, EDA visualisations |
| 02 | `notebooks/02_segmentation.py` | K-Means customer segmentation, risk persona labelling |
| 03 | `notebooks/03_feature_engineering.py` | Ratio features, delinquency streaks, WoE encoding |
| 04 | `notebooks/04_modeling.py` | 5 models, SMOTE, SHAP explainability, validation metrics |
| 05 | `notebooks/05_business_impact.py` | Cutoff simulation, expected loss, NT$11M+ net benefit |
| 06 | `notebooks/06_model_enhancement.py` | Feature selection, Optuna tuning, stacking ensemble |

---

## Results

| Metric | Value |
|--------|-------|
| Best baseline model | Random Forest / XGBoost (~0.67 AUC) |
| After Optuna tuning | +0.03–0.06 improvement |
| Defaults caught at optimal cutoff | 88% |
| Net benefit (6,000-account test set) | NT$11.3M |
| Scaled to 1M customers | ~USD $62M annually |

---

## Project structure

```
credit-risk-model/
├── data/                          # raw + processed datasets
│   ├── credit_card_default.csv
│   ├── credit_card_with_clusters.csv
│   └── credit_card_engineered.csv
├── notebooks/
│   ├── 01_eda.py
│   ├── 02_segmentation.py
│   ├── 03_feature_engineering.py
│   ├── 04_modeling.py
│   ├── 05_business_impact.py
│   └── 06_model_enhancement.py    ← Optuna + stacking
├── outputs/                       # all charts + model artefacts
├── src/
│   └── utils.py                   # shared config, helpers, plot style
├── app.py                         # Streamlit interactive demo
├── model_card.md                  # governance doc
├── requirements.txt
├── run_all.py                     # one-command pipeline runner
└── README.md
```

---

## Tech stack

All free and open source:
`Python 3.10+` · `pandas` · `scikit-learn` · `XGBoost` · `SHAP` · `Optuna` · `imbalanced-learn` · `matplotlib` · `seaborn` · `Streamlit`
