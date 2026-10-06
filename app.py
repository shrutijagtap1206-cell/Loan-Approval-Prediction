"""
streamlit_app.py
================
Streamlit Web Application for the Loan Application Approval Prediction System.

Run locally:
    streamlit run streamlit_app.py

Deploy on Streamlit Community Cloud:
    Connect your GitHub repository, specify 'streamlit_app.py' as the entrypoint.
"""

import streamlit as st
import numpy as np
from engine import predict, get_model

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=" SMART Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich styling & typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3a8a, #0284c7);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 14px rgba(30, 58, 138, 0.25);
    }
    
    .main-header h1 {
        margin: 0;
        font-weight: 800;
        font-size: 2.2rem;
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        margin: 0.5rem 0 0;
        opacity: 0.9;
        font-size: 1.05rem;
    }

    .section-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
    }
    
    .decision-approved {
        background: #ecfdf5;
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 1.5rem;
        color: #065f46;
        margin-bottom: 1.5rem;
    }
    
    .decision-rejected {
        background: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        color: #991b1b;
        margin-bottom: 1.5rem;
    }
    
    .metric-container {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def format_inr(val):
    if val is None or np.isnan(val):
        return "N/A"
    return f"₹{round(val):,}"


# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/bank-building.png", width=70)
    st.title("About the System")
    st.info(
        "This enterprise prediction system uses a unified **Gradient Boosting Pipeline** "
        "trained on multi-source borrower data. It assesses financial solvency, "
        "debt-to-income balance, and creditworthiness in real-time."
    )
    
    try:
        bundle = get_model()
        metrics = bundle.get("metrics", {})
        st.subheader("Model Performance")
        st.write(f"**Algorithm:** {bundle.get('model_name', 'Gradient Boosting')}")
        st.write(f"**Accuracy:** {metrics.get('Accuracy', 0.975):.2%}")
        st.write(f"**ROC-AUC:** {metrics.get('ROC-AUC', 0.996):.4f}")
        st.write(f"**F1-Score:** {metrics.get('F1-Score', 0.980):.4f}")
    except Exception:
        pass
    
    st.divider()
    st.caption("© Loan Application Approval Prediction System")


# -----------------------------------------------------------------------------
# MAIN HEADER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🏦 SMART Loan  Approval Prediction</h1>
    <p>Real-time borrower credit assessment, debt-to-income analysis, and approval probability scoring.</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# FORM INPUTS
# -----------------------------------------------------------------------------
with st.form("loan_application_form"):
    
    # --- 1. APPLICANT INFORMATION ---
    st.subheader("1. Applicant Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    with c2:
        married = st.selectbox("Marital Status", ["Yes", "No"], format_func=lambda x: "Married" if x == "Yes" else "Single / Unmarried")
        self_employed = st.selectbox("Employment Type", ["No", "Yes"], format_func=lambda x: "Salaried Employee" if x == "No" else "Self-Employed / Business")
    with c3:
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1, step=1)
        property_area = st.selectbox("Property / Residential Area", ["Urban", "Semiurban", "Rural"])

    st.divider()

    # --- 2. FINANCIAL INFORMATION ---
    st.subheader("2. Financial Information")
    f1, f2 = st.columns(2)
    with f1:
        applicant_income = st.number_input(
            "Applicant Annual Income (₹)",
            min_value=10000.0,
            value=600000.0,
            step=25000.0,
            help="Gross annual earnings of the primary applicant before deductions."
        )
        coapplicant_income = st.number_input(
            "Co-applicant Annual Income (₹)",
            min_value=0.0,
            value=240000.0,
            step=25000.0,
            help="Annual income of spouse, parent, or co-borrower (enter 0 if none)."
        )
    with f2:
        st.caption("Declared Assets (Estimated market value in ₹):")
        a1, a2 = st.columns(2)
        with a1:
            res_assets = st.number_input("Residential Property", min_value=0.0, value=2500000.0, step=50000.0)
            comm_assets = st.number_input("Commercial Property", min_value=0.0, value=0.0, step=50000.0)
        with a2:
            lux_assets = st.number_input("Luxury / Vehicles", min_value=0.0, value=500000.0, step=25000.0)
            bank_assets = st.number_input("Bank Deposits / Liquid", min_value=0.0, value=400000.0, step=25000.0)

    st.divider()

    # --- 3. LOAN INFORMATION ---
    st.subheader("3. Loan Information")
    l1, l2 = st.columns(2)
    with l1:
        loan_amount = st.number_input(
            "Requested Loan Amount (₹)",
            min_value=10000.0,
            value=1800000.0,
            step=50000.0,
            help="Total principal loan amount requested from the bank."
        )
    with l2:
        loan_term_months = st.number_input(
            "Repayment Term (Months)",
            min_value=12,
            max_value=480,
            value=240,
            step=12,
            help="Loan duration in months (e.g. 120 = 10 yrs, 240 = 20 yrs, 360 = 30 yrs)."
        )
        st.caption(f"Equivalent Tenure: **{loan_term_months / 12:.1f} Years**")

    st.divider()

    # --- 4. CREDIT INFORMATION ---
    st.subheader("4. Credit Information")
    
    is_new_to_credit = st.checkbox(
        "I am new to credit / I don't have a CIBIL score yet",
        value=False,
        help="Check this if the applicant has no prior loan or credit card record."
    )
    
    cr1, cr2 = st.columns(2)
    with cr1:
        if is_new_to_credit:
            st.info("ℹ️ CIBIL score disabled for first-time / new-to-credit borrower.")
            cibil_score = None
        else:
            cibil_score = st.slider(
                "CIBIL Bureau Score",
                min_value=300,
                max_value=900,
                value=750,
                help="Continuous credit bureau rating between 300 (Poor) and 900 (Excellent)."
            )
            
            # Badge tier
            if cibil_score >= 750:
                st.success(f"Score: {cibil_score} (Excellent)")
            elif cibil_score >= 680:
                st.info(f"Score: {cibil_score} (Good)")
            elif cibil_score >= 580:
                st.warning(f"Score: {cibil_score} (Fair)")
            else:
                st.error(f"Score: {cibil_score} (Poor)")
                
    with cr2:
        if is_new_to_credit:
            credit_history = None
            st.caption("Past Credit History: **Not on file**")
        else:
            ch_choice = st.selectbox(
                "Past Repayment Track Record",
                ["Clean Record (No defaults)", "Adverse Record (Past delays / defaults)", "No history on file"]
            )
            if ch_choice == "Clean Record (No defaults)":
                credit_history = 1.0
            elif ch_choice == "Adverse Record (Past delays / defaults)":
                credit_history = 0.0
            else:
                credit_history = None

    submit_button = st.form_submit_button("Predict Loan Approval", use_container_width=True, type="primary")


# -----------------------------------------------------------------------------
# PREDICTION LOGIC & PRESENTATION
# -----------------------------------------------------------------------------
if submit_button:
    # Build payload
    payload = {
        "gender": gender,
        "married": married,
        "dependents": float(dependents),
        "education": education,
        "self_employed": self_employed,
        "applicant_income": float(applicant_income),
        "coapplicant_income": float(coapplicant_income),
        "loan_amount": float(loan_amount),
        "loan_term_months": float(loan_term_months),
        "credit_history": credit_history,
        "cibil_score": float(cibil_score) if cibil_score is not None else None,
        "property_area": property_area,
        "residential_assets": float(res_assets),
        "commercial_assets": float(comm_assets),
        "luxury_assets": float(lux_assets),
        "bank_asset_value": float(bank_assets)
    }

    with st.spinner("Processing applicant profile through decision engine..."):
        result = predict(payload)

    if not result.get("success", False):
        st.error("Validation Error:")
        for err in result.get("errors", []):
            st.write(f"- {err}")
    else:
        is_approved = result["approved"]
        prob = result["probability"]
        fin = result.get("financial_summary", {})

        st.subheader("Decision & Risk Analysis")

        # Top Decision Banner
        if is_approved:
            st.markdown(f"""
            <div class="decision-approved">
                <h2 style="margin:0 0 6px; font-weight:800; font-size:1.8rem;">✓ LOAN LIKELY TO BE APPROVED</h2>
                <p style="margin:0; font-size:1.05rem;">Approval Probability: <strong>{prob}%</strong></p>
                <p style="margin:4px 0 0; opacity:0.85; font-size:0.95rem;">Applicant demonstrates healthy financial solvency and manageable leverage.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="decision-rejected">
                <h2 style="margin:0 0 6px; font-weight:800; font-size:1.8rem;">✕ LOAN LIKELY TO BE REJECTED</h2>
                <p style="margin:0; font-size:1.05rem;">Approval Probability: <strong>{prob}%</strong></p>
                <p style="margin:4px 0 0; opacity:0.85; font-size:0.95rem;">Application presents elevated credit risk or high debt-to-income leverage.</p>
            </div>
            """, unsafe_allow_html=True)

        # 4 Key Financial Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Annual Income", format_inr(fin.get("total_income")))
        m2.metric("Loan-to-Income (LTI)", f"{fin.get('loan_to_income_ratio', 0):.2f}x")
        m3.metric("Est. Monthly Principal", f"{format_inr(fin.get('approx_monthly_emi'))}/mo")
        m4.metric("Declared Total Assets", format_inr(fin.get("total_assets")))

        # Factors breakdown
        col_pos, col_risk = st.columns(2)
        with col_pos:
            st.markdown("#### ✓ Profile Strengths")
            pos_factors = result.get("positive_factors", [])
            if pos_factors:
                for pf in pos_factors:
                    st.write(f"- {pf}")
            else:
                st.write("- Standard profile credentials.")

        with col_risk:
            st.markdown("#### ⚠️ Risk Observations")
            risk_factors = result.get("risk_factors", [])
            if risk_factors:
                for rf in risk_factors:
                    st.write(f"- {rf}")
            else:
                st.write("- No critical risk flags detected.")
