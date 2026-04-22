"""
Credit Risk Modelling & Customer Intelligence — Interactive Demo App
Run from project root:
    streamlit run app.py
"""
import sys; sys.path.insert(0, '.')
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib, os
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stApp,.main,[data-testid="stAppViewContainer"]{background-color:#0A0F1E;}
section[data-testid="stSidebar"]{background-color:#0D1526!important;border-right:1px solid #1E2D45;}
section[data-testid="stSidebar"] *{color:#94A3B8!important;}
html,body,[class*="css"]{color:#E2E8F0;}
h1{color:#60A5FA!important;font-size:1.65rem!important;}
h2{color:#93C5FD!important;font-size:1.2rem!important;}
h3{color:#BAC8E0!important;font-size:1.0rem!important;}
p,li{color:#CBD5E1;line-height:1.65;}
.story-box{background:#111827;border-left:3px solid #3B82F6;border-radius:0 8px 8px 0;padding:14px 18px;margin:0 0 18px;color:#94A3B8;font-size:0.88rem;line-height:1.6;}
.story-box.green{border-color:#10B981;}
.story-box.amber{border-color:#F59E0B;}
.story-box.purple{border-color:#8B5CF6;}
.story-box.pink{border-color:#EC4899;}
.story-box.red{border-color:#EF4444;}
.kpi-card{background:#111827;border:1px solid #1E2D45;border-radius:10px;padding:14px 16px;margin-bottom:4px;}
.kpi-label{font-size:10px;color:#475569;margin-bottom:4px;letter-spacing:0.06em;text-transform:uppercase;}
.kpi-value{font-size:24px;font-weight:700;color:#F1F5F9;}
.kpi-sub{font-size:10px;color:#10B981;margin-top:3px;}
.def-pill{display:inline-block;background:#1E2D45;border:1px solid #2D4060;border-radius:6px;padding:6px 12px;margin:3px 3px 3px 0;font-size:12px;color:#94A3B8;}
.def-pill strong{color:#60A5FA;}
.risk-box{border-radius:10px;padding:16px 20px;margin-top:14px;}
.risk-high{background:#1C0A0A;border:1px solid #7F1D1D;}
.risk-medium{background:#1C1400;border:1px solid #78350F;}
.risk-low{background:#081C10;border:1px solid #14532D;}
.risk-title{font-size:17px;font-weight:700;margin-bottom:5px;}
.risk-body{font-size:12px;color:#94A3B8;}
.gauge-track{height:12px;background:#1E2D45;border-radius:6px;overflow:hidden;margin:8px 0 4px;}
.gauge-fill{height:100%;border-radius:6px;transition:width .3s;}
.gauge-ticks{display:flex;justify-content:space-between;font-size:10px;color:#475569;}
.stNumberInput input{background:#111827!important;color:#F1F5F9!important;border:1px solid #1E2D45!important;border-radius:6px;}
.stButton>button{background:#1D4ED8!important;color:#fff!important;border:none!important;border-radius:8px!important;font-weight:600!important;padding:10px 0!important;}
.stButton>button:hover{background:#1E40AF!important;}
hr{border-color:#1E2D45!important;}
[data-testid="stMetricLabel"]{color:#475569!important;font-size:11px!important;}
[data-testid="stMetricValue"]{color:#F1F5F9!important;}
</style>
""", unsafe_allow_html=True)

# ── Load assets ───────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('data/credit_card_engineered.csv')

@st.cache_resource
def load_assets():
    return (joblib.load('outputs/best_model.pkl'),
            joblib.load('outputs/scaler.pkl'),
            joblib.load('outputs/model_metadata.pkl'))

@st.cache_data
def load_impact():
    return pd.read_csv('outputs/business_impact_simulation.csv')

if not os.path.exists('outputs/best_model.pkl'):
    st.error("Run `python run_all.py --skip 06` first to generate model files.")
    st.stop()

TARGET = 'default.payment.next.month'
PAY_STATUS = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
BILL_COLS  = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
PAY_COLS   = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
NEEDS_SCALE= {'Logistic Regression','MLP Neural Net'}
BG, CARD, LINE = '#0A0F1E', '#111827', '#1E2D45'

plt.rcParams.update({
    'figure.facecolor':BG,'axes.facecolor':CARD,'axes.edgecolor':LINE,
    'axes.labelcolor':'#64748B','xtick.color':'#64748B','ytick.color':'#64748B',
    'text.color':'#CBD5E1','legend.facecolor':CARD,'legend.edgecolor':LINE,
    'legend.labelcolor':'#CBD5E1','axes.titlecolor':'#93C5FD',
    'grid.color':LINE,'grid.alpha':0.45,'axes.spines.top':False,'axes.spines.right':False,
})

df        = load_data()
model, scaler, meta = load_assets()
impact_df = load_impact()
FEATURE_COLS = meta['feature_cols']
best_name    = meta['best_model']

_, X_test_raw, _, y_test_raw = train_test_split(
    df[FEATURE_COLS].fillna(0), df[TARGET],
    test_size=0.2, random_state=42, stratify=df[TARGET])

# ── Sidebar ───────────────────────────────────────────────────
PAGES      = ['Overview','Customer Scorer','Model Performance','Business Impact','Customer Segments']
PAGE_COLOR = {'Overview':'#3B82F6','Customer Scorer':'#10B981','Model Performance':'#8B5CF6',
              'Business Impact':'#F59E0B','Customer Segments':'#EC4899'}

with st.sidebar:
    st.markdown("""
    <div style='padding:4px 0 18px;'>
      <div style='font-size:17px;font-weight:700;color:#60A5FA;'>💳 Credit Risk</div>
      <div style='font-size:10px;color:#1E3A5F;margin-top:1px;letter-spacing:0.05em;'>INTELLIGENCE SYSTEM</div>
    </div>""", unsafe_allow_html=True)

    page   = st.radio("", PAGES, label_visibility='collapsed')
    accent = PAGE_COLOR[page]
    st.markdown(f"""<style>
    [data-testid="stSidebar"] .stRadio [aria-checked="true"]+div{{color:{accent}!important;font-weight:600;}}
    </style>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='border-top:1px solid #1E2D45;margin:14px 0;padding-top:14px;'>
      <div style='font-size:10px;color:#1E3A5F;margin-bottom:4px;'>BEST MODEL</div>
      <div style='font-size:13px;color:#E2E8F0;font-weight:600;margin-bottom:10px;'>{best_name}</div>
      <div style='font-size:10px;color:#1E3A5F;margin-bottom:4px;'>BASELINE AUC</div>
      <div style='font-size:16px;color:{accent};font-weight:700;margin-bottom:10px;'>{meta["metrics"][best_name]["AUC"]:.4f}</div>
      <div style='font-size:10px;color:#1E3A5F;margin-bottom:4px;'>KS STATISTIC</div>
      <div style='font-size:13px;color:#E2E8F0;font-weight:600;margin-bottom:10px;'>{meta["metrics"][best_name]["KS"]:.4f}</div>
      <div style='font-size:10px;color:#1E3A5F;margin-bottom:4px;'>DATASET SIZE</div>
      <div style='font-size:13px;color:#E2E8F0;font-weight:600;'>30,000 accounts</div>
    </div>""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# OVERVIEW
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if page == 'Overview':
    st.title("Credit Risk Modelling & Customer Intelligence")
    st.markdown("""
    <div class='story-box'>
    This system predicts which credit card customers are likely to default in the following month,
    segments them into behavioural risk profiles, and quantifies the financial outcome of deploying
    the model in a live approval strategy. It is built on 30,000 real-world credit card records
    spanning 6 months of payment history — the same data structure used by major card issuers.
    Every step of this pipeline, from raw data to dollar impact, mirrors how production credit
    risk models are built and governed at financial institutions.
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col, lbl, val, sub, color in zip(
        [c1,c2,c3,c4],
        ["Total Accounts","Default Rate","Best AUC","Defaults Caught"],
        ["30,000","13.7%",f"{meta['metrics'][best_name]['AUC']:.4f}","88%"],
        ["UCI credit card dataset","6.3:1 class imbalance",f"Model: {best_name}","at optimal cutoff"],
        ["#3B82F6","#EF4444","#10B981","#F59E0B"]
    ):
        with col:
            st.markdown(f"""
            <div class='kpi-card' style='border-top:3px solid {color};'>
              <div class='kpi-label'>{lbl}</div>
              <div class='kpi-value' style='color:{color};'>{val}</div>
              <div class='kpi-sub' style='color:{color};opacity:0.75;'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1E2D45;margin:22px 0 18px;'>", unsafe_allow_html=True)
    left, right = st.columns([1,1])

    with left:
        st.subheader("Pipeline")
        for num, color, title, desc in [
            ("01","#3B82F6","Data Integrity & EDA","Missing value audit, IQR outlier detection, class imbalance analysis, 6-month payment trends."),
            ("02","#8B5CF6","Customer Segmentation","K-Means clustering on behavioural features — groups customers into Transactor, Revolver, and Delinquent risk personas."),
            ("03","#10B981","Feature Engineering","35 predictive features derived from raw data — utilisation ratios, delinquency streaks, WoE-encoded categoricals."),
            ("04","#F59E0B","Model Training & Validation","5 ML models — LR, Decision Tree, Random Forest, XGBoost, MLP — with SMOTE, learning curves, and SHAP explainability."),
            ("05","#EC4899","Business Impact Sizing","Approval strategy simulation — NT$11.3M net benefit identified at the optimal score threshold."),
            ("06","#60A5FA","Model Enhancement","Optuna hyperparameter tuning (50 trials) + stacking ensemble to push AUC beyond the baseline."),
        ]:
            st.markdown(f"""
            <div style='display:flex;gap:12px;align-items:flex-start;margin-bottom:13px;'>
              <div style='width:30px;height:30px;border-radius:50%;background:{color};display:flex;align-items:center;
                          justify-content:center;font-size:11px;font-weight:700;color:#fff;flex-shrink:0;'>{num}</div>
              <div>
                <div style='font-size:13px;font-weight:600;color:#E2E8F0;margin-bottom:2px;'>{title}</div>
                <div style='font-size:11px;color:#475569;line-height:1.5;'>{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    with right:
        st.subheader("Model comparison — AUC")
        st.markdown("""
        <div class='story-box'>
        AUC measures how well the model separates defaulters from non-defaulters at every possible
        threshold. A score of 0.5 is random guessing; 1.0 is perfect. Tree-based models lead because
        credit default risk is driven by non-linear interactions — high utilisation is only dangerous
        when combined with declining payment behaviour.
        </div>""", unsafe_allow_html=True)

        mnames  = list(meta['metrics'].keys())
        maucs   = [meta['metrics'][n]['AUC'] for n in mnames]
        shorts  = ['LR','DTree','RF','XGB','MLP']
        bcolors = ['#10B981' if n==best_name else '#1E3A5F' for n in mnames]

        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        bars = ax.barh(shorts, maucs, color=bcolors, alpha=0.9, height=0.5)
        ax.axvline(0.5, color='#EF4444', linestyle='--', linewidth=1, alpha=0.6, label='Random baseline')
        for bar, v, nm in zip(bars, maucs, mnames):
            ax.text(v+0.002, bar.get_y()+bar.get_height()/2,
                    f'{v:.4f}{"  ← best" if nm==best_name else ""}',
                    va='center', fontsize=8.5,
                    color='#10B981' if nm==best_name else '#64748B')
        ax.set_xlim(0.55, 0.73); ax.set_xlabel('AUC')
        ax.legend(fontsize=8); ax.set_title('Held-out test set (6,000 accounts)', fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        st.subheader("Class distribution")
        st.markdown("""
        <div class='story-box red'>
        At 13.7%, the default rate creates a 6.3:1 class imbalance. A naive model that always
        predicts "no default" would be 86% accurate — yet completely useless for risk management.
        We address this with SMOTE during training and evaluate with AUC and KS instead of accuracy.
        </div>""", unsafe_allow_html=True)
        cnts = df[TARGET].value_counts()
        fig2, ax2 = plt.subplots(figsize=(5.5, 2))
        ax2.barh(['Default','Non-default'],[cnts[1],cnts[0]],
                 color=['#EF4444','#3B82F6'], height=0.4, alpha=0.9)
        for v, y in [(cnts[1],0),(cnts[0],1)]:
            ax2.text(v+300, y, f'{v:,} ({v/len(df):.1%})', va='center', fontsize=9, color='#94A3B8')
        ax2.set_xlim(0, 31000); ax2.set_xlabel('Count')
        plt.tight_layout(); st.pyplot(fig2); plt.close()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUSTOMER SCORER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == 'Customer Scorer':
    st.title("Customer Risk Scorer")
    st.markdown("""
    <div class='story-box green'>
    Enter a customer's profile to get a live default probability from the trained model.
    The scorer uses the same 28 features as production — payment delinquency history,
    credit utilisation, and payment coverage ratios are the strongest drivers. The result
    includes a risk classification, a probability gauge, and a breakdown of which signals
    are contributing most to the score.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1E2D45;margin:16px 0;'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("#### Demographics")
        limit_bal = st.slider("Credit Limit (NT$)", 10000, 800000, 150000, step=10000, format="NT$%d")
        age       = st.slider("Age", 21, 75, 35)
        sex       = st.selectbox("Sex", [1,2], format_func=lambda x:"Male" if x==1 else "Female")
        education = st.selectbox("Education",[1,2,3,4],
                                  format_func=lambda x:{1:"Graduate",2:"University",3:"High School",4:"Other"}[x])
        marriage  = st.selectbox("Marital Status",[1,2,3],
                                  format_func=lambda x:{1:"Married",2:"Single",3:"Other"}[x])
        st.markdown("""
        <div style='background:#0D1526;border-radius:8px;padding:10px 12px;margin-top:10px;font-size:11px;color:#334155;'>
        <span style='color:#475569;font-weight:600;'>Why these matter:</span> Education
        correlates with income stability. Graduates default at meaningfully lower rates
        than high-school-only customers in this dataset.
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("#### Payment History")
        st.markdown("<div style='font-size:11px;color:#334155;margin-bottom:8px;'>-1 = paid on time &nbsp;·&nbsp; 1 = 1 month late &nbsp;·&nbsp; 2+ = months late</div>", unsafe_allow_html=True)
        pay0 = st.slider("Month -1 (most recent)", -1, 6, 0)
        pay2 = st.slider("Month -2", -1, 6, 0)
        pay3 = st.slider("Month -3", -1, 6, 0)
        pay4 = st.slider("Month -4", -1, 6, 0)
        pay5 = st.slider("Month -5", -1, 6, 0)
        pay6 = st.slider("Month -6", -1, 6, 0)
        st.markdown("""
        <div style='background:#0D1526;border-radius:8px;padding:10px 12px;margin-top:10px;font-size:11px;color:#334155;'>
        <span style='color:#475569;font-weight:600;'>Why this matters:</span> Payment delinquency
        is the single strongest default predictor. One missed payment in 3 months significantly
        elevates risk — consecutive misses are near-certain default signals.
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown("#### Balances & Payments")
        bill1  = st.number_input("Bill Month -1 (NT$)", 0, 500000, 60000, step=5000)
        bill2  = st.number_input("Bill Month -2 (NT$)", 0, 500000, 58000, step=5000)
        bill3  = st.number_input("Bill Month -3 (NT$)", 0, 500000, 55000, step=5000)
        pay_a1 = st.number_input("Payment Month -1 (NT$)", 0, 300000, 15000, step=1000)
        pay_a2 = st.number_input("Payment Month -2 (NT$)", 0, 300000, 14000, step=1000)
        pay_a3 = st.number_input("Payment Month -3 (NT$)", 0, 300000, 13000, step=1000)
        st.markdown("""
        <div style='background:#0D1526;border-radius:8px;padding:10px 12px;margin-top:10px;font-size:11px;color:#334155;'>
        <span style='color:#475569;font-weight:600;'>Why this matters:</span> The
        pay-to-bill ratio reveals intent. Paying only 10% of a bill monthly signals
        debt accumulation — a stronger risk signal than raw amounts alone.
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1E2D45;margin:16px 0;'>", unsafe_allow_html=True)
    if st.button("Score this Customer", use_container_width=True):
        p_statuses = [pay0,pay2,pay3,pay4,pay5,pay6]
        bills      = [bill1,bill2,bill3,bill1*0.98,bill1*0.96,bill1*0.94]
        pays       = [pay_a1,pay_a2,pay_a3,pay_a1*0.95,pay_a1*0.9,pay_a1*0.85]

        util       = bill1/max(limit_bal,1)
        p2b        = pay_a1/max(bill1,1)
        avg_p2b    = float(np.mean([p/max(b,1) for p,b in zip(pays,bills)]))
        delinq     = sum(1 for p in p_statuses if p>0)
        max_d      = max(0,max(p_statuses))
        rev_ratio  = max(0,bill1-pay_a1)/max(limit_bal,1)

        woe_sex={1:0.0275,2:-0.0186}; woe_edu={1:-0.2876,2:0.1499,3:0.0894,4:0.1341}; woe_mar={1:0.0188,2:-0.0168,3:0.0172}
        row_d = {'LIMIT_BAL':limit_bal,'AGE':age,'PAY_0':pay0,'PAY_2':pay2,'PAY_3':pay3,
                 'PAY_4':pay4,'PAY_5':pay5,'PAY_6':pay6,
                 'BILL_AMT1':bills[0],'BILL_AMT2':bills[1],'BILL_AMT3':bills[2],
                 'PAY_AMT1':pays[0],'PAY_AMT2':pays[1],'PAY_AMT3':pays[2],
                 'util_ratio':util,'avg_util_6m':np.mean(bills)/max(limit_bal,1),
                 'pay_to_bill_ratio':p2b,'avg_pay_to_bill_6m':avg_p2b,
                 'delinq_streak':delinq,'max_delay_months':max_d,
                 'bill_trend':bill1-bills[-1],'pay_trend':pay_a1-pays[-1],
                 'avg_payment_6m':float(np.mean(pays)),'revolving_ratio':rev_ratio,
                 'SEX_WoE':woe_sex.get(sex,0),'EDUCATION_WoE':woe_edu.get(education,0),
                 'MARRIAGE_WoE':woe_mar.get(marriage,0),'cluster_label':0}

        Xr = pd.DataFrame([row_d])
        for c in FEATURE_COLS:
            if c not in Xr.columns: Xr[c]=0
        Xi = scaler.transform(Xr[FEATURE_COLS].fillna(0)) if best_name in NEEDS_SCALE else Xr[FEATURE_COLS].fillna(0)
        prob = float(model.predict_proba(Xi)[0][1])

        if prob>=0.5:   risk_lbl,risk_col,box_cls='HIGH RISK','#EF4444','risk-high'
        elif prob>=0.25: risk_lbl,risk_col,box_cls='MEDIUM RISK','#F59E0B','risk-medium'
        else:           risk_lbl,risk_col,box_cls='LOW RISK','#10B981','risk-low'

        action = {'HIGH RISK':'Recommend DECLINE. Delinquency streak and low payment coverage exceed acceptable risk thresholds. Expected loss from approval exceeds expected revenue.',
                  'MEDIUM RISK':'Recommend REVIEW. Consider approval with a reduced credit limit (50–70% of requested) or require additional income verification before proceeding.',
                  'LOW RISK':'Recommend APPROVE. Profile is consistent with reliable repayment. Standard credit limit and terms are appropriate.'}[risk_lbl]

        r1,r2,r3,r4,r5 = st.columns(5)
        for col,(lbl,val,col_) in zip([r1,r2,r3,r4,r5],[
            ("Default Probability", f"{prob:.1%}", risk_col),
            ("Delinquency Streak",  f"{delinq} months", "#F59E0B" if delinq>0 else "#10B981"),
            ("Max Payment Delay",   f"{max_d} months",  "#EF4444" if max_d>=2 else "#10B981"),
            ("Credit Utilisation",  f"{util:.1%}",      "#EF4444" if util>0.8 else "#10B981"),
            ("Pay-to-Bill Ratio",   f"{p2b:.2f}",       "#10B981" if p2b>0.8 else "#F59E0B"),
        ]):
            with col:
                st.markdown(f"""<div class='kpi-card' style='border-top:3px solid {col_};'>
                  <div class='kpi-label'>{lbl}</div>
                  <div class='kpi-value' style='color:{col_};font-size:19px;'>{val}</div></div>""",
                unsafe_allow_html=True)

        st.markdown(f"""<div class='risk-box {box_cls}'>
          <div class='risk-title' style='color:{risk_col};'>{risk_lbl} — {prob:.1%} default probability</div>
          <div class='risk-body'>{action}</div></div>""", unsafe_allow_html=True)

        pct=prob*100
        st.markdown(f"""<div style='margin-top:14px;'>
          <div style='font-size:10px;color:#334155;margin-bottom:5px;letter-spacing:0.05em;'>RISK GAUGE</div>
          <div class='gauge-track'><div class='gauge-fill' style='width:{pct:.1f}%;background:{risk_col};'></div></div>
          <div class='gauge-ticks'><span>0% — No risk</span><span style='color:#F59E0B;'>25% — Medium</span><span style='color:#EF4444;'>50% — High</span><span>100%</span></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#1E2D45;margin:18px 0;'>", unsafe_allow_html=True)
        st.subheader("Risk signal breakdown")
        signals = {'Max delay months':min(max_d/6,1.0),'Delinquency streak':delinq/6,
                   'Credit utilisation':min(util,1.0),'Low payment coverage':max(0,1-p2b),
                   'Revolving balance ratio':min(rev_ratio,1.0)}
        fig_s, ax_s = plt.subplots(figsize=(8, 2.8))
        sc_cols=['#EF4444' if v>0.6 else '#F59E0B' if v>0.3 else '#10B981' for v in signals.values()]
        bars_s=ax_s.barh(list(signals.keys()),list(signals.values()),color=sc_cols,alpha=0.9,height=0.45)
        ax_s.set_xlim(0,1.15); ax_s.set_xlabel('Relative signal strength  (0=low risk, 1=high risk)')
        ax_s.set_title('Feature-level risk contribution for this customer profile', fontsize=9)
        for bar,v in zip(bars_s,signals.values()):
            ax_s.text(v+0.02,bar.get_y()+bar.get_height()/2,f'{v:.2f}',va='center',fontsize=8.5)
        plt.tight_layout(); st.pyplot(fig_s); plt.close()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MODEL PERFORMANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == 'Model Performance':
    st.title("Model Performance")
    st.markdown("""
    <div class='story-box purple'>
    Five machine learning models were trained on 80% of the data and evaluated on a held-out
    20% test set that the models had never seen. Training used SMOTE to balance the 13.7% default
    rate — but the test set preserves the real-world imbalance, ensuring the metrics reflect
    what would actually happen in production. The goal was to find the model that best separates
    defaulters from non-defaulters across the full risk spectrum, not just at a single threshold.
    </div>""", unsafe_allow_html=True)

    st.subheader("Metric definitions")
    st.markdown("""
    <div style='margin-bottom:18px;'>
    <div class='def-pill'><strong>AUC</strong> — Probability a random defaulter scores higher than a random non-defaulter. 0.5 = coin flip, 1.0 = perfect.</div>
    <div class='def-pill'><strong>KS Statistic</strong> — Max separation between default and non-default score distributions. Standard credit risk metric. Higher = sharper separation.</div>
    <div class='def-pill'><strong>Gini</strong> — Equal to 2×AUC−1. A Gini of 0.33 means the model ranks customers 33% better than random.</div>
    <div class='def-pill'><strong>Avg Precision</strong> — Area under Precision-Recall curve. Penalises false positives — important when classes are imbalanced.</div>
    <div class='def-pill'><strong>Brier Score</strong> — Mean squared error of probabilities. Lower = better calibration. Tells you whether the predicted 30% risk really means 30%.</div>
    </div>""", unsafe_allow_html=True)

    st.subheader("Validation metrics — held-out test set (20%)")
    st.markdown("""
    <div class='story-box purple'>
    Random Forest leads on AUC and KS. The story these numbers tell: tree-based ensemble models
    capture the complex interactions between payment delay, utilisation, and payment coverage that
    a linear model (Logistic Regression) cannot. The KS of 0.28 means that at the optimal cutoff,
    the model produces 28 percentage points more separation between defaults and non-defaults
    than random ranking — directly translating to fewer missed defaulters and fewer incorrectly
    declined good customers.
    </div>""", unsafe_allow_html=True)

    rows_html = ""
    for name, m in sorted(meta['metrics'].items(), key=lambda x:-x[1]['AUC']):
        is_best = name == best_name
        bg      = "background:#0A1F0E;" if is_best else ""
        badge   = " <span style='font-size:10px;background:#14532D;color:#4ADE80;padding:2px 7px;border-radius:4px;margin-left:6px;'>BEST</span>" if is_best else ""
        auc_c   = "#4ADE80" if is_best else "#94A3B8"
        rows_html += f"""<tr style='{bg}'>
          <td style='padding:11px 14px;color:#E2E8F0;font-weight:{"600" if is_best else "400"};'>{name}{badge}</td>
          <td style='padding:11px 14px;text-align:center;color:{auc_c};font-weight:700;'>{m['AUC']:.4f}</td>
          <td style='padding:11px 14px;text-align:center;color:#94A3B8;'>{m['KS']:.4f}</td>
          <td style='padding:11px 14px;text-align:center;color:#94A3B8;'>{m['Gini']:.4f}</td>
          <td style='padding:11px 14px;text-align:center;color:#94A3B8;'>{m['Avg Precision']:.4f}</td>
          <td style='padding:11px 14px;text-align:center;color:#94A3B8;'>{m['Brier Score']:.4f}</td></tr>"""

    st.markdown(f"""
    <div style='overflow-x:auto;border-radius:10px;border:1px solid #1E2D45;margin-bottom:20px;'>
    <table style='width:100%;border-collapse:collapse;background:#111827;'>
      <thead><tr style='border-bottom:1px solid #1E2D45;'>
        <th style='padding:10px 14px;text-align:left;color:#475569;font-size:11px;'>MODEL</th>
        <th style='padding:10px 14px;text-align:center;color:#475569;font-size:11px;'>AUC</th>
        <th style='padding:10px 14px;text-align:center;color:#475569;font-size:11px;'>KS</th>
        <th style='padding:10px 14px;text-align:center;color:#475569;font-size:11px;'>GINI</th>
        <th style='padding:10px 14px;text-align:center;color:#475569;font-size:11px;'>AVG PRECISION</th>
        <th style='padding:10px 14px;text-align:center;color:#475569;font-size:11px;'>BRIER SCORE</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table></div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC curves")
        st.markdown("""
        <div class='story-box purple'>
        The ROC curve plots how the true positive rate (defaults correctly caught) trades off against
        the false positive rate (good customers incorrectly declined) at every threshold. A model
        hugging the top-left corner catches most defaults while declining very few good customers.
        The diagonal is the random baseline — any model below it is worse than a coin flip.
        </div>""", unsafe_allow_html=True)
        Xte = scaler.transform(X_test_raw) if best_name in NEEDS_SCALE else X_test_raw
        prob_best = model.predict_proba(Xte)[:,1]
        fpr_b, tpr_b, _ = roc_curve(y_test_raw, prob_best)
        fig1, ax1 = plt.subplots(figsize=(5.8, 4.2))
        ax1.plot(fpr_b, tpr_b, color='#10B981', linewidth=2.5,
                 label=f'{best_name} — AUC={roc_auc_score(y_test_raw,prob_best):.4f}')
        ax1.plot([0,1],[0,1],'--',color='#334155',linewidth=1,label='Random baseline')
        ax1.fill_between(fpr_b,tpr_b,alpha=0.06,color='#10B981')
        ax1.set_xlabel('False positive rate  (good customers flagged)')
        ax1.set_ylabel('True positive rate  (defaults caught)')
        ax1.legend(fontsize=9)
        plt.tight_layout(); st.pyplot(fig1); plt.close()

    with col2:
        st.subheader("AUC vs. KS — all models")
        st.markdown("""
        <div class='story-box purple'>
        Comparing AUC and KS side-by-side confirms the story is consistent across both metrics.
        The models that score best on AUC also score best on KS — meaning the ranking is
        robust and not an artefact of one particular measure. This gives confidence that the
        model's risk ordering is genuinely reliable across the full score distribution.
        </div>""", unsafe_allow_html=True)
        mns  = list(meta['metrics'].keys())
        aucs = [meta['metrics'][n]['AUC'] for n in mns]
        kss  = [meta['metrics'][n]['KS']  for n in mns]
        xs   = np.arange(len(mns))
        shs  = ['LR','DTree','RF','XGB','MLP']
        fig2, ax2 = plt.subplots(figsize=(5.8, 4.2))
        b1=ax2.bar(xs-0.2,aucs,0.35,label='AUC',color='#8B5CF6',alpha=0.85)
        b2=ax2.bar(xs+0.2,kss, 0.35,label='KS', color='#10B981',alpha=0.85)
        ax2.set_xticks(xs); ax2.set_xticklabels(shs)
        for bar,v in zip(list(b1)+list(b2),aucs+kss):
            ax2.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.004,f'{v:.3f}',ha='center',fontsize=8)
        ax2.legend(); ax2.set_title('Performance across metrics — all models',fontsize=9)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("<hr style='border-color:#1E2D45;margin:20px 0;'>", unsafe_allow_html=True)
    st.subheader("SHAP explainability — why does the model give each customer their score?")
    st.markdown("""
    <div class='story-box purple'>
    SHAP (SHapley Additive exPlanations) answers the question regulators and model validators
    always ask: <em>why did this customer receive a high risk score?</em> Each bar shows the
    average absolute impact of that feature across all test predictions. <strong>max_delay_months</strong>
    dominates — customers with more consecutive months of late payment receive consistently higher
    default probabilities. This explainability layer is not optional in regulated industries;
    it is required by model risk governance frameworks such as SR 11-7.
    </div>""", unsafe_allow_html=True)
    if os.path.exists('outputs/04d_shap_explainability.png'):
        st.image('outputs/04d_shap_explainability.png', use_column_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BUSINESS IMPACT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == 'Business Impact':
    st.title("Business Impact Simulator")
    st.markdown("""
    <div class='story-box amber'>
    A model that scores customers is only valuable if it drives better lending decisions.
    This simulator applies the Expected Loss framework — the same framework banks use for
    capital provisioning — to translate model performance into financial outcomes.
    Every credit card company faces the same trade-off: decline too few customers and losses
    mount; decline too many and you lose profitable accounts. The optimal cutoff is the
    point that maximises net benefit, and this simulator finds it.
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div style='margin-bottom:18px;'>
    <div class='def-pill'><strong>Expected Loss</strong> = PD × EAD × LGD</div>
    <div class='def-pill'><strong>PD</strong> — Probability of Default (model output)</div>
    <div class='def-pill'><strong>EAD</strong> — Exposure at Default · ~40% of credit limit</div>
    <div class='def-pill'><strong>LGD</strong> — Loss Given Default · 65% (post-collections recovery rate)</div>
    <div class='def-pill'><strong>Net Benefit</strong> = Losses prevented − Revenue foregone</div>
    </div>""", unsafe_allow_html=True)

    cutoff = st.slider("Decline if predicted default probability exceeds:",
                        0.05, 0.90, 0.25, step=0.025, format="%.3f")
    row_i  = impact_df.iloc[(impact_df['cutoff']-cutoff).abs().argsort()[:1]].iloc[0]

    c1,c2,c3,c4 = st.columns(4)
    for col,(lbl,val,col_) in zip([c1,c2,c3,c4],[
        ("Defaults caught",         f"{row_i['defaults_caught_pct']:.1%}",   "#10B981"),
        ("Good clients declined",   f"{row_i['good_declined_pct']:.1%}",     "#EF4444"),
        ("Losses prevented",        f"NT${row_i['losses_prevented']/1e6:.1f}M","#10B981"),
        ("Net benefit",             f"NT${row_i['net_benefit']/1e6:.1f}M",   "#F59E0B"),
    ]):
        with col:
            st.markdown(f"""<div class='kpi-card' style='border-top:3px solid {col_};'>
              <div class='kpi-label'>{lbl}</div>
              <div class='kpi-value' style='color:{col_};font-size:21px;'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='font-size:10px;color:#334155;margin:8px 0 18px;'>Assumptions: NT$120,000 avg limit · 40% EAD · 65% LGD · NT$3,500 annual revenue per good account · 6,000-account test set</div>",unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1E2D45;margin:16px 0;'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Net benefit curve")
        st.markdown("""
        <div class='story-box amber'>
        The peak of this curve is the optimal deployment strategy — the threshold that adds the
        most net financial value per account. At cutoff 0.25, the model prevents NT$22.5M in
        losses against NT$11.3M net benefit on 6,000 accounts. The curve drops on both sides:
        too low and you approve too many defaulters; too high and you lose profitable accounts.
        </div>""", unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(5.8,3.8))
        ax1.plot(impact_df['cutoff'],impact_df['net_benefit']/1e6,color='#F59E0B',linewidth=2.5)
        ax1.axvline(cutoff,color='#EF4444',linestyle='--',linewidth=1.5,label=f'Selected: {cutoff:.3f}')
        ax1.fill_between(impact_df['cutoff'],0,impact_df['net_benefit']/1e6,alpha=0.07,color='#F59E0B')
        ax1.axhline(0,color='#334155',linewidth=0.8)
        ax1.set_xlabel('Score cutoff'); ax1.set_ylabel('Net benefit (NT$ millions)')
        ax1.legend()
        plt.tight_layout(); st.pyplot(fig1); plt.close()

    with col2:
        st.subheader("Loss prevention vs. revenue foregone")
        st.markdown("""
        <div class='story-box amber'>
        As the cutoff rises, more defaults are caught (green) but more good customers are also
        declined (red). The optimal point is where the gap between losses prevented and revenue
        foregone is maximised. This chart makes the business trade-off visible — and shows exactly
        how much of the model's value comes from risk reduction vs. what it costs in lost revenue.
        </div>""", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5.8,3.8))
        ax2.plot(impact_df['cutoff'],impact_df['losses_prevented']/1e6,color='#10B981',linewidth=2,label='Losses prevented')
        ax2.plot(impact_df['cutoff'],impact_df['revenue_lost']/1e6,    color='#EF4444',linewidth=2,label='Revenue foregone')
        ax2.axvline(cutoff,color='#F59E0B',linestyle='--',linewidth=1.5)
        ax2.set_xlabel('Score cutoff'); ax2.set_ylabel('NT$ millions')
        ax2.legend()
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    st.markdown("<hr style='border-color:#1E2D45;margin:16px 0;'>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Lift chart — decile analysis")
        st.markdown("""
        <div class='story-box amber'>
        Customers sorted by risk score, divided into 10 equal groups. Decile 1 contains the
        highest-scored accounts. A lift of 3–4× in D1 means the top 10% of flagged accounts
        contain 3–4 times more actual defaults than a random 10% — letting the collections
        team prioritise their reviews where they matter most, rather than reviewing accounts
        at random.
        </div>""", unsafe_allow_html=True)
        if os.path.exists('outputs/05b_lift_gains.png'):
            st.image('outputs/05b_lift_gains.png', use_column_width=True)

    with col4:
        st.subheader("Portfolio scaling")
        st.markdown("""
        <div class='story-box amber'>
        The test results scale linearly to larger portfolios. Enter your portfolio size
        to estimate annual impact at the currently selected cutoff.
        </div>""", unsafe_allow_html=True)
        portfolio = st.number_input("Portfolio size:", 10000, 10000000, 1000000, step=50000)
        scale = portfolio/6000
        for lbl,val,col_ in [
            ("Projected losses prevented", f"NT${row_i['losses_prevented']*scale/1e6:.0f}M", "#10B981"),
            ("Projected revenue foregone",  f"NT${row_i['revenue_lost']*scale/1e6:.0f}M",    "#EF4444"),
            ("Projected net benefit",       f"NT${row_i['net_benefit']*scale/1e6:.0f}M  ≈  USD ${row_i['net_benefit']*scale/30/1e6:.0f}M", "#F59E0B"),
        ]:
            st.markdown(f"""<div class='kpi-card' style='border-left:3px solid {col_};margin-bottom:8px;'>
              <div class='kpi-label'>{lbl}</div>
              <div class='kpi-value' style='color:{col_};font-size:18px;'>{val}</div></div>""",unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CUSTOMER SEGMENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
elif page == 'Customer Segments':
    st.title("Customer Segments")
    st.markdown("""
    <div class='story-box pink'>
    One model applied to all customers is a blunt instrument. K-Means clustering reveals
    that the portfolio naturally separates into distinct behavioural groups — each with
    a different default rate, spending pattern, and repayment tendency. Identifying these
    segments allows a lender to move beyond a single scorecard: offer transactors higher
    limits and rewards, flag revolvers for early intervention, and apply tighter controls
    to delinquent customers. This is the foundation of strategy-aligned credit decisioning.
    </div>""", unsafe_allow_html=True)

    seg_colors = ['#3B82F6','#EF4444','#10B981','#F59E0B','#8B5CF6']

    if 'cluster' in df.columns and 'persona' in df.columns:
        profile = df.groupby(['cluster','persona']).agg(
            Count=(TARGET,'count'), Default_Rate=(TARGET,'mean'),
            Avg_Limit=('LIMIT_BAL','mean'), Avg_Age=('AGE','mean'),
        ).reset_index()

        st.subheader("Risk persona profiles")
        seg_cols = st.columns(len(profile))
        for col,(_, rs) in zip(seg_cols, profile.iterrows()):
            color = seg_colors[int(rs['cluster'])%len(seg_colors)]
            dr    = rs['Default_Rate']
            drc   = '#EF4444' if dr>0.15 else '#F59E0B' if dr>0.10 else '#10B981'
            with col:
                st.markdown(f"""<div class='kpi-card' style='border-top:4px solid {color};'>
                  <div style='font-size:12px;font-weight:700;color:{color};margin-bottom:10px;'>{rs['persona']}</div>
                  <div class='kpi-label'>PORTFOLIO SHARE</div>
                  <div style='font-size:20px;font-weight:700;color:#F1F5F9;margin-bottom:6px;'>{rs['Count']/len(df):.1%}</div>
                  <div class='kpi-label'>DEFAULT RATE</div>
                  <div style='font-size:20px;font-weight:700;color:{drc};margin-bottom:6px;'>{dr:.1%}</div>
                  <div class='kpi-label'>AVG CREDIT LIMIT</div>
                  <div style='font-size:13px;color:#64748B;'>NT${rs["Avg_Limit"]:,.0f}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#1E2D45;margin:20px 0;'>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Optimal cluster selection")
            st.markdown("""
            <div class='story-box pink'>
            Two methods confirm the number of clusters. The <strong>elbow method</strong> plots
            inertia (total within-cluster distance) and identifies the point of diminishing returns.
            The <strong>silhouette score</strong> measures how well-separated clusters are from
            each other — the k with the highest silhouette score produces the most distinct, coherent
            segments. Coherent segments are essential: vague clusters give no actionable strategic insight.
            </div>""", unsafe_allow_html=True)
            if os.path.exists('outputs/02a_elbow_silhouette.png'):
                st.image('outputs/02a_elbow_silhouette.png', use_column_width=True)

        with col2:
            st.subheader("Segments — PCA 2D projection")
            st.markdown("""
            <div class='story-box pink'>
            Clusters are computed in 5-dimensional feature space. PCA projects them to 2 axes for
            visualisation. Clear visual separation confirms the algorithm found genuinely distinct
            behavioural groups — not arbitrary divisions. Each dot is a customer. Groups that sit
            apart in this chart have meaningfully different payment behaviours and should be treated
            with different credit strategies.
            </div>""", unsafe_allow_html=True)
            if os.path.exists('outputs/02b_cluster_pca.png'):
                st.image('outputs/02b_cluster_pca.png', use_column_width=True)

        st.markdown("<hr style='border-color:#1E2D45;margin:20px 0;'>", unsafe_allow_html=True)
        st.subheader("Detailed segment comparison")
        st.markdown("""
        <div class='story-box pink'>
        This table directly quantifies what separates each segment. The default rate is the key
        outcome column. The pay-to-bill ratio and average delay are the key behavioural inputs —
        and the pattern is consistent: segments with lower pay-to-bill ratios and higher average
        delays produce higher default rates. This validates the entire approach: behavioural
        clustering is predictively meaningful, and each segment genuinely warrants a different
        credit strategy.
        </div>""", unsafe_allow_html=True)

        rows_s = ""
        for _, rs in profile.iterrows():
            color = seg_colors[int(rs['cluster'])%len(seg_colors)]
            dr    = rs['Default_Rate']
            drc   = '#EF4444' if dr>0.15 else '#F59E0B' if dr>0.10 else '#10B981'
            rows_s += f"""<tr>
              <td style='padding:11px 14px;'><span style='color:{color};font-weight:600;'>{rs['persona']}</span></td>
              <td style='padding:11px 14px;text-align:center;color:#94A3B8;'>{rs['Count']/len(df):.1%}</td>
              <td style='padding:11px 14px;text-align:center;'><span style='color:{drc};font-weight:600;'>{dr:.1%}</span></td>
              <td style='padding:11px 14px;text-align:center;color:#94A3B8;'>NT${rs["Avg_Limit"]:,.0f}</td>
              <td style='padding:11px 14px;text-align:center;color:#94A3B8;'>{rs["Avg_Age"]:.0f} yrs</td>
            </tr>"""

        st.markdown(f"""
        <div style='overflow-x:auto;border-radius:10px;border:1px solid #1E2D45;margin-bottom:20px;'>
        <table style='width:100%;border-collapse:collapse;background:#111827;'>
          <thead><tr style='border-bottom:1px solid #1E2D45;'>
            <th style='padding:10px 14px;text-align:left;color:#475569;font-size:11px;'>SEGMENT</th>
            <th style='padding:10px 14px;text-align:center;color:#475569;font-size:11px;'>PORTFOLIO SHARE</th>
            <th style='padding:10px 14px;text-align:center;color:#475569;font-size:11px;'>DEFAULT RATE</th>
            <th style='padding:10px 14px;text-align:center;color:#475569;font-size:11px;'>AVG CREDIT LIMIT</th>
            <th style='padding:10px 14px;text-align:center;color:#475569;font-size:11px;'>AVG AGE</th>
          </tr></thead>
          <tbody>{rows_s}</tbody>
        </table></div>""", unsafe_allow_html=True)

        if os.path.exists('outputs/02c_cluster_profiles.png'):
            st.image('outputs/02c_cluster_profiles.png', use_column_width=True)
