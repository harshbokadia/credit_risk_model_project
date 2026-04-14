"""
Credit Risk Modelling — Interactive Demo App
Run: streamlit run app.py  (from project root)
"""
import sys; sys.path.insert(0, '.')
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib, os
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
# ── Dark chart defaults ───────────────────────────────────────
import matplotlib
matplotlib.rcParams.update({
    'text.color':        '#E2E8F0',
    'axes.labelcolor':   '#94A3B8',
    'xtick.color':       '#94A3B8',
    'ytick.color':       '#94A3B8',
    'axes.edgecolor':    '#334155',
    'legend.facecolor':  '#1E293B',
    'legend.edgecolor':  '#334155',
    'legend.labelcolor': '#E2E8F0',
    'axes.titlecolor':   '#93C5FD',
    'figure.facecolor':  '#1E293B',
    'axes.facecolor':    '#1E293B',
    'grid.color':        '#334155',
    'grid.alpha':         0.5,
})



st.set_page_config(
    page_title="Credit Risk Intelligence System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp, .main { background-color: #0F172A; }
    section[data-testid="stSidebar"] { background-color: #1E293B !important; }
    section[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
    html, body, [class*="css"], p, span, div, label { color: #E2E8F0; }
    h1, h2, h3 { color: #93C5FD !important; }
    [data-testid="stMetric"] {
        background: #1E293B; border: 1px solid #334155;
        border-radius: 10px; padding: 14px 16px;
    }
    [data-testid="stMetricLabel"] { color: #94A3B8 !important; font-size: 13px !important; }
    [data-testid="stMetricValue"] { color: #F1F5F9 !important; font-size: 22px !important; }
    [data-testid="stMetricDelta"] { color: #4ADE80 !important; }
    .dvn-scroller, .col_heading, .data { color: #E2E8F0 !important; background: #1E293B !important; }
    .stNumberInput input, .stTextInput input {
        background: #1E293B !important; color: #F1F5F9 !important;
        border: 1px solid #334155 !important; border-radius: 6px;
    }
    .stButton > button {
        background: #2563EB !important; color: white !important;
        border: none !important; border-radius: 8px !important; font-weight: 600 !important;
    }
    .stButton > button:hover { background: #1D4ED8 !important; }
    .risk-high   { background:#3B0F0F; border-left:4px solid #F87171; border-radius:8px; padding:16px; }
    .risk-low    { background:#052E16; border-left:4px solid #4ADE80; border-radius:8px; padding:16px; }
    .risk-medium { background:#2D1B00; border-left:4px solid #FBBF24; border-radius:8px; padding:16px; }
    .stCaption, small { color: #64748B !important; }
    hr { border-color: #334155 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load assets ───────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('data/credit_card_engineered.csv')
    return df

@st.cache_resource
def load_model():
    model  = joblib.load('outputs/best_model.pkl')
    scaler = joblib.load('outputs/scaler.pkl')
    meta   = joblib.load('outputs/model_metadata.pkl')
    return model, scaler, meta

@st.cache_data
def load_impact():
    return pd.read_csv('outputs/business_impact_simulation.csv')

TARGET = 'default.payment.next.month'
PAY_STATUS = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
BILL_COLS  = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
PAY_COLS   = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']

# Check files exist
if not os.path.exists('outputs/best_model.pkl'):
    st.error("⚠️ Model files not found. Please run `python run_all.py` first to generate them.")
    st.stop()

df    = load_data()
model, scaler, meta = load_model()
impact_df = load_impact()
FEATURE_COLS = meta['feature_cols']
best_name    = meta['best_model']

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Credit Risk System")
    st.markdown("---")
    page = st.radio("Navigate to", [
        "🏠  Overview",
        "🔍  Customer Risk Scorer",
        "📊  Model Performance",
        "💰  Business Impact",
        "👥  Customer Segments"
    ])
    st.markdown("---")
    st.markdown(f"**Best model:** {best_name}")
    st.markdown(f"**AUC:** {meta['metrics'][best_name]['AUC']:.4f}")
    st.markdown(f"**KS:** {meta['metrics'][best_name]['KS']:.4f}")
    st.markdown(f"**Dataset:** 30,000 accounts")
    st.markdown("---")
    st.caption("Built with Python · XGBoost · SHAP · Streamlit")

# ══════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════
if "Overview" in page:
    st.title("Credit Risk Modelling & Customer Intelligence")
    st.markdown("An end-to-end machine learning pipeline that predicts credit card default risk, "
                "segments customers by behavioural profile, and quantifies the business impact of model deployment.")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Accounts",   "30,000")
    c2.metric("Default Rate",     f"{df[TARGET].mean():.1%}")
    c3.metric("Best AUC",         f"{meta['metrics'][best_name]['AUC']:.4f}", f"Model: {best_name}")
    c4.metric("Defaults Caught",  "88%", "at optimal cutoff")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Pipeline")
        steps = [
            ("01", "Data Integrity & EDA",        "Missing values · outliers · distributions"),
            ("02", "Customer Segmentation",        "K-Means clustering · risk personas"),
            ("03", "Feature Engineering",          "35 features · WoE encoding · delinquency streaks"),
            ("04", "Model Training & Validation",  "5 models · SMOTE · SHAP · learning curves"),
            ("05", "Business Impact Sizing",       "Cutoff simulation · NT$11.3M net benefit"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style='display:flex;align-items:flex-start;gap:12px;margin-bottom:12px;'>
              <div style='background:#1B3A6B;color:white;border-radius:50%;width:28px;height:28px;
                          display:flex;align-items:center;justify-content:center;
                          font-size:12px;font-weight:bold;flex-shrink:0;'>{num}</div>
              <div>
                <div style='font-weight:600;font-size:14px;color:#1A1A2E;'>{title}</div>
                <div style='font-size:12px;color:#666;'>{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.subheader("Model comparison")
        metrics_df = pd.DataFrame(meta['metrics']).T[['AUC','KS','Gini']].round(4)
        metrics_df = metrics_df.sort_values('AUC', ascending=False)
        metrics_df.index.name = 'Model'
        st.dataframe(metrics_df, use_container_width=True)

        st.subheader("Class distribution")
        fig, ax = plt.subplots(figsize=(5, 2.5))
        counts = df[TARGET].value_counts()
        colors = ['#378ADD', '#E24B4A']
        ax.barh(['Non-default', 'Default'], counts.values, color=colors, height=0.5)
        for i, v in enumerate(counts.values):
            ax.text(v + 100, i, f'{v:,} ({v/len(df):.1%})', va='center', fontsize=10)
        ax.set_xlabel('Count')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#1E293B')
        fig.patch.set_facecolor('#1E293B')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════
# PAGE 2: CUSTOMER RISK SCORER
# ══════════════════════════════════════════════════════════════
elif "Scorer" in page:
    st.title("🔍 Customer Risk Scorer")
    st.markdown("Enter a customer's profile to get a real-time default probability prediction.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        limit_bal  = st.slider("Credit Limit (NT$)", 10000, 800000, 150000, step=10000,
                                format="NT$%d")
        age        = st.slider("Age", 21, 75, 35)
        sex        = st.selectbox("Sex", [1, 2], format_func=lambda x: "Male" if x==1 else "Female")
        education  = st.selectbox("Education", [1,2,3,4],
                                   format_func=lambda x: {1:"Graduate",2:"University",
                                                           3:"High School",4:"Other"}[x])
        marriage   = st.selectbox("Marital Status", [1,2,3],
                                   format_func=lambda x: {1:"Married",2:"Single",3:"Other"}[x])

    with col2:
        st.subheader("Payment History (1=on time, 2+=months late)")
        pay0 = st.slider("Month -1 (most recent)", -1, 6, 0)
        pay2 = st.slider("Month -2", -1, 6, 0)
        pay3 = st.slider("Month -3", -1, 6, 0)
        pay4 = st.slider("Month -4", -1, 6, 0)
        pay5 = st.slider("Month -5", -1, 6, 0)
        pay6 = st.slider("Month -6", -1, 6, 0)

    with col3:
        st.subheader("Balances & Payments")
        bill1 = st.number_input("Bill Month -1 (NT$)", 0, 500000, 60000, step=5000)
        bill2 = st.number_input("Bill Month -2 (NT$)", 0, 500000, 58000, step=5000)
        bill3 = st.number_input("Bill Month -3 (NT$)", 0, 500000, 55000, step=5000)
        pay_a1 = st.number_input("Payment Month -1 (NT$)", 0, 300000, 15000, step=1000)
        pay_a2 = st.number_input("Payment Month -2 (NT$)", 0, 300000, 14000, step=1000)
        pay_a3 = st.number_input("Payment Month -3 (NT$)", 0, 300000, 13000, step=1000)

    st.markdown("---")
    if st.button("🎯  Calculate Default Probability", type="primary", use_container_width=True):

        # Build feature row
        pay_statuses = [pay0, pay2, pay3, pay4, pay5, pay6]
        bills  = [bill1, bill2, bill3, bill1*0.98, bill1*0.96, bill1*0.94]
        pays   = [pay_a1, pay_a2, pay_a3, pay_a1*0.95, pay_a1*0.9, pay_a1*0.85]

        util_ratio        = bill1 / max(limit_bal, 1)
        avg_util_6m       = np.mean(bills) / max(limit_bal, 1)
        pay_to_bill       = pay_a1 / max(bill1, 1)
        avg_pay_to_bill   = np.mean([p/max(b,1) for p,b in zip(pays, bills)])
        delinq_streak     = sum(1 for p in pay_statuses if p > 0)
        max_delay         = max(0, max(pay_statuses))
        bill_trend        = bill1 - bills[-1]
        pay_trend         = pay_a1 - pays[-1]
        avg_payment_6m    = np.mean(pays)
        total_bill_6m     = sum(bills)
        revolving_ratio   = max(0, bill1 - pay_a1) / max(limit_bal, 1)

        # WoE maps (from training)
        woe_sex  = {1: 0.0275, 2: -0.0186}
        woe_edu  = {1: -0.2876, 2: 0.1499, 3: 0.0894, 4: 0.1341}
        woe_mar  = {1: 0.0188, 2: -0.0168, 3: 0.0172}

        row = {
            'LIMIT_BAL': limit_bal, 'AGE': age,
            'PAY_0': pay0, 'PAY_2': pay2, 'PAY_3': pay3,
            'PAY_4': pay4, 'PAY_5': pay5, 'PAY_6': pay6,
            'BILL_AMT1': bills[0], 'BILL_AMT2': bills[1], 'BILL_AMT3': bills[2],
            'PAY_AMT1': pays[0],  'PAY_AMT2': pays[1],  'PAY_AMT3': pays[2],
            'util_ratio': util_ratio, 'avg_util_6m': avg_util_6m,
            'pay_to_bill_ratio': pay_to_bill, 'avg_pay_to_bill_6m': avg_pay_to_bill,
            'delinq_streak': delinq_streak, 'max_delay_months': max_delay,
            'bill_trend': bill_trend, 'pay_trend': pay_trend,
            'avg_payment_6m': avg_payment_6m, 'revolving_ratio': revolving_ratio,
            'SEX_WoE': woe_sex.get(sex, 0), 'EDUCATION_WoE': woe_edu.get(education, 0),
            'MARRIAGE_WoE': woe_mar.get(marriage, 0), 'cluster_label': 0
        }

        X_row = pd.DataFrame([row])
        for col in FEATURE_COLS:
            if col not in X_row.columns:
                X_row[col] = 0
        X_row = X_row[FEATURE_COLS].fillna(0)

        if best_name in ('Logistic Regression', 'MLP Neural Net'):
            X_input = scaler.transform(X_row)
        else:
            X_input = X_row

        prob = model.predict_proba(X_input)[0][1]

        # Display result
        st.markdown("### Prediction Result")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Default Probability", f"{prob:.1%}")
        r2.metric("Delinquency Streak",  f"{delinq_streak} months")
        r3.metric("Max Payment Delay",   f"{max_delay} months")
        r4.metric("Credit Utilisation",  f"{util_ratio:.1%}")

        if prob >= 0.5:
            risk_label = "HIGH RISK"
            css_class  = "risk-high"
            icon = "🔴"
            action = "Recommend DECLINE. Default probability exceeds 50% — the customer shows significant delinquency or utilisation risk."
        elif prob >= 0.25:
            risk_label = "MEDIUM RISK"
            css_class  = "risk-medium"
            icon = "🟡"
            action = "Recommend REVIEW. Default probability is elevated — consider a lower credit limit or additional verification."
        else:
            risk_label = "LOW RISK"
            css_class  = "risk-low"
            icon = "🟢"
            action = "Recommend APPROVE. Default probability is low — customer profile is consistent with reliable repayment behaviour."

        st.markdown(f"""
        <div class='{css_class}' style='margin-top:16px;'>
          <div style='font-size:22px;font-weight:700;'>{icon} {risk_label} — {prob:.1%} default probability</div>
          <div style='margin-top:8px;font-size:14px;color:#444;'>{action}</div>
        </div>""", unsafe_allow_html=True)

        # Risk gauge
        fig, ax = plt.subplots(figsize=(8, 1.2))
        ax.barh([0], [1], color='#334155', height=0.4)
        color = '#E24B4A' if prob>=0.5 else '#EF9F27' if prob>=0.25 else '#3B6D11'
        ax.barh([0], [prob], color=color, height=0.4)
        ax.axvline(0.25, color='#EF9F27', linestyle='--', linewidth=1.2, alpha=0.7)
        ax.axvline(0.50, color='#E24B4A', linestyle='--', linewidth=1.2, alpha=0.7)
        ax.set_xlim(0, 1); ax.set_yticks([]); ax.set_xlabel("Default probability")
        ax.text(0.25, 0.6, 'Medium', transform=ax.transAxes, fontsize=8, color='#EF9F27')
        ax.text(0.6, 0.6, 'High', transform=ax.transAxes, fontsize=8, color='#E24B4A')
        ax.set_facecolor('#1E293B'); fig.patch.set_facecolor('#1E293B')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════
# PAGE 3: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif "Performance" in page:
    st.title("📊 Model Performance")
    st.markdown("---")

    # Metrics table
    st.subheader("Validation metrics — held-out test set (20%)")
    metrics_df = pd.DataFrame(meta['metrics']).T[['AUC','KS','Gini','Avg Precision','Brier Score']].round(4)
    metrics_df = metrics_df.sort_values('AUC', ascending=False)
    metrics_df.index.name = 'Model'
    st.dataframe(metrics_df, use_container_width=True)
    st.caption("AUC = Area under ROC curve · KS = Kolmogorov-Smirnov statistic · Gini = 2×AUC−1")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("AUC comparison")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        names   = list(meta['metrics'].keys())
        aucs    = [meta['metrics'][n]['AUC'] for n in names]
        short   = ['LR', 'DT', 'RF', 'XGB', 'MLP']
        colors  = ['#378ADD','#888780','#1D9E75','#534AB7','#E24B4A']
        bars    = ax.bar(short[:len(names)], aucs, color=colors[:len(names)], alpha=0.85, width=0.5)
        for bar, v in zip(bars, aucs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, max(aucs)*1.15)
        ax.set_ylabel('AUC'); ax.set_facecolor('#1E293B'); fig.patch.set_facecolor('#1E293B')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("KS & Gini")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        x       = np.arange(len(names))
        ks_vals  = [meta['metrics'][n]['KS']   for n in names]
        gi_vals  = [meta['metrics'][n]['Gini'] for n in names]
        ax.bar(x-0.2, ks_vals,  0.35, label='KS',   color='#534AB7', alpha=0.85)
        ax.bar(x+0.2, gi_vals,  0.35, label='Gini', color='#1D9E75', alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(short[:len(names)])
        ax.legend(); ax.set_facecolor('#1E293B'); fig.patch.set_facecolor('#1E293B')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Feature importance (top 15 by Random Forest)")
    if os.path.exists('outputs/03a_feature_importance.png'):
        st.image('outputs/03a_feature_importance.png', use_column_width=True)

    st.subheader("SHAP explainability — XGBoost")
    if os.path.exists('outputs/04d_shap_explainability.png'):
        st.image('outputs/04d_shap_explainability.png', use_column_width=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4: BUSINESS IMPACT
# ══════════════════════════════════════════════════════════════
elif "Business" in page:
    st.title("💰 Business Impact Simulator")
    st.markdown("Adjust the approval cutoff to see how the model's deployment affects losses and revenue in real time.")
    st.markdown("---")

    AVG_EXPOSURE           = 48000
    LOSS_GIVEN_DEFAULT     = 0.65
    REVENUE_PER_GOOD       = 3500
    EXPECTED_LOSS_DEFAULT  = AVG_EXPOSURE * LOSS_GIVEN_DEFAULT

    cutoff = st.slider("Score cutoff — decline if default probability exceeds:", 0.05, 0.90, 0.25, step=0.025,
                       format="%.3f")

    row = impact_df.iloc[(impact_df['cutoff'] - cutoff).abs().argsort()[:1]]
    row = row.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Defaults caught",         f"{row['defaults_caught_pct']:.1%}")
    c2.metric("Good customers declined", f"{row['good_declined_pct']:.1%}")
    c3.metric("Losses prevented",        f"NT${row['losses_prevented']/1e6:.1f}M")
    c4.metric("Net benefit",             f"NT${row['net_benefit']/1e6:.1f}M",
              delta=f"vs NT$0 (no model)")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Net benefit across all cutoffs")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(impact_df['cutoff'], impact_df['net_benefit']/1e6,
                color='#534AB7', linewidth=2.5)
        ax.axvline(cutoff, color='#E24B4A', linestyle='--', linewidth=1.5,
                   label=f'Selected: {cutoff:.3f}')
        ax.fill_between(impact_df['cutoff'], 0, impact_df['net_benefit']/1e6,
                        alpha=0.08, color='#534AB7')
        ax.axhline(0, color='gray', linewidth=0.8)
        ax.set_xlabel('Score cutoff'); ax.set_ylabel('Net benefit (M NT$)')
        ax.legend(); ax.set_facecolor('#1E293B'); fig.patch.set_facecolor('#1E293B')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Losses prevented vs. revenue foregone")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(impact_df['cutoff'], impact_df['losses_prevented']/1e6,
                color='#1D9E75', linewidth=2, label='Losses prevented')
        ax.plot(impact_df['cutoff'], impact_df['revenue_lost']/1e6,
                color='#E24B4A', linewidth=2, label='Revenue foregone')
        ax.axvline(cutoff, color='#534AB7', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Score cutoff'); ax.set_ylabel('NT$ millions')
        ax.legend(); ax.set_facecolor('#1E293B'); fig.patch.set_facecolor('#1E293B')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.subheader("Scale to your portfolio size")
    portfolio = st.number_input("Number of customers in your portfolio:", 10000, 5000000, 1000000, step=50000)
    n_test    = 6000
    scale     = portfolio / n_test
    scaled_benefit = row['net_benefit'] * scale
    scaled_losses  = row['losses_prevented'] * scale

    s1, s2, s3 = st.columns(3)
    s1.metric("Portfolio size",          f"{portfolio:,.0f} customers")
    s2.metric("Projected losses prevented", f"NT${scaled_losses/1e6:.0f}M")
    s3.metric("Projected net benefit",   f"NT${scaled_benefit/1e6:.0f}M",
              delta=f"≈ USD ${scaled_benefit/30/1e6:.0f}M")

# ══════════════════════════════════════════════════════════════
# PAGE 5: CUSTOMER SEGMENTS
# ══════════════════════════════════════════════════════════════
elif "Segments" in page:
    st.title("👥 Customer Segments")
    st.markdown("K-Means clustering identifies natural behavioural groups — each with a distinct risk profile.")
    st.markdown("---")

    if 'cluster' in df.columns and 'persona' in df.columns:
        cluster_stats = df.groupby(['cluster','persona']).agg(
            Count     = (TARGET, 'count'),
            DefaultRate = (TARGET, 'mean'),
            AvgLimit  = ('LIMIT_BAL', 'mean'),
            AvgAge    = ('AGE', 'mean'),
        ).reset_index()
        cluster_stats['DefaultRate'] = cluster_stats['DefaultRate'].map('{:.1%}'.format)
        cluster_stats['AvgLimit']    = cluster_stats['AvgLimit'].map('NT${:,.0f}'.format)
        cluster_stats['AvgAge']      = cluster_stats['AvgAge'].map('{:.0f} yrs'.format)
        cluster_stats['Share']       = (cluster_stats['Count'] / len(df)).map('{:.1%}'.format)
        cluster_stats.columns        = ['Cluster','Persona','Count','Default Rate','Avg Limit','Avg Age','Portfolio Share']
        st.dataframe(cluster_stats, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Cluster selection — elbow & silhouette")
        if os.path.exists('outputs/02a_elbow_silhouette.png'):
            st.image('outputs/02a_elbow_silhouette.png', use_column_width=True)
    with col2:
        st.subheader("Customer segments — PCA projection")
        if os.path.exists('outputs/02b_cluster_pca.png'):
            st.image('outputs/02b_cluster_pca.png', use_column_width=True)

    st.subheader("Cluster profile comparison")
    if os.path.exists('outputs/02c_cluster_profiles.png'):
        st.image('outputs/02c_cluster_profiles.png', use_column_width=True)
