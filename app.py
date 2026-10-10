"""
app.py
======
Streamlit Web Application for the Loan Application Approval Prediction System.

Run locally:
    streamlit run app.py

Deployment:
    Deploy this file through Streamlit Community Cloud using GitHub.
"""

import streamlit as st
import numpy as np

from engine import predict, get_model


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="SMART Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-header {
        padding: 2rem 2.2rem;
        border-radius: 18px;
        margin-bottom: 2rem;
        background: linear-gradient(
            135deg,
            #0f4c81 0%,
            #1769aa 50%,
            #2196f3 100%
        );
        color: white;
        box-shadow: 0 8px 25px rgba(15, 76, 129, 0.18);
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    .main-header p {
        margin: 0.6rem 0 0;
        font-size: 1rem;
        opacity: 0.92;
    }

    .section-card {
        padding: 1.2rem;
        border-radius: 14px;
        background: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.04);
    }

    .decision-approved {
        padding: 1.4rem;
        border-radius: 14px;
        margin: 1rem 0 1.5rem;
        background: #e8f5e9;
        border: 1px solid #81c784;
        color: #1b5e20;
    }

    .decision-rejected {
        padding: 1.4rem;
        border-radius: 14px;
        margin: 1rem 0 1.5rem;
        background: #ffebee;
        border: 1px solid #ef9a9a;
        color: #b71c1c;
    }

    .metric-container {
        padding: 1rem;
        border-radius: 12px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }

    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 12px;
    }

    div[data-testid="stForm"] {
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.2rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def format_inr(val):
    """
    Format numeric values as Indian Rupee currency.
    """
    if val is None:
        return "N/A"

    try:
        if np.isnan(val):
            return "N/A"
    except (TypeError, ValueError):
        pass

    return f"₹{round(float(val)):,}"


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    st.image(
        "https://img.icons8.com/isometric/100/bank-building.png",
        width=70
    )

    st.title("About the System")

    st.info(
        "This prediction system uses a machine-learning pipeline trained "
        "on borrower application data. It evaluates applicant information, "
        "income, assets, loan characteristics, and credit information "
        "to estimate the probability of loan approval."
    )

    try:
        bundle = get_model()
        metrics = bundle.get("metrics", {})

        st.subheader("Model Performance")

        st.write(
            f"**Algorithm:** "
            f"{bundle.get('model_name', 'Gradient Boosting')}"
        )

        if metrics.get("Accuracy") is not None:
            st.write(
                f"**Accuracy:** "
                f"{metrics.get('Accuracy'):.2%}"
            )

        if metrics.get("ROC-AUC") is not None:
            st.write(
                f"**ROC-AUC:** "
                f"{metrics.get('ROC-AUC'):.4f}"
            )

        if metrics.get("F1-Score") is not None:
            st.write(
                f"**F1-Score:** "
                f"{metrics.get('F1-Score'):.4f}"
            )

    except Exception:
        pass

    st.divider()

    st.caption(
        "© Loan Application Approval Prediction System"
    )


# =============================================================================
# MAIN HEADER
# =============================================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🏦 UrApproval-SMART Loan Approval </h1>
        <p>
            AI-powered loan application assessment using applicant,
            financial, asset, and credit information.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# LOAN APPLICATION FORM
# =============================================================================

with st.form("loan_application_form"):

    # -------------------------------------------------------------------------
    # 1. APPLICANT INFORMATION
    # -------------------------------------------------------------------------

    st.subheader("1. Applicant Information")

    c1, c2, c3 = st.columns(3)

    with c1:

        gender = st.selectbox(
            "Gender",
            [
                "Select gender",
                "Male",
                "Female"
            ]
        )

        education = st.selectbox(
            "Education Level",
            [
                "Select education",
                "Graduate",
                "Not Graduate"
            ]
        )

    with c2:

        married = st.selectbox(
            "Marital Status",
            [
                "Select marital status",
                "Yes",
                "No"
            ],
            format_func=lambda x: (
                "Select marital status"
                if x == "Select marital status"
                else "Married"
                if x == "Yes"
                else "Single / Unmarried"
            )
        )

        self_employed = st.selectbox(
            "Employment Type",
            [
                "Select employment type",
                "No",
                "Yes"
            ],
            format_func=lambda x: (
                "Select employment type"
                if x == "Select employment type"
                else "Salaried Employee"
                if x == "No"
                else "Self-Employed / Business"
            )
        )

    with c3:

        dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=10,
            value=None,
            step=1,
            placeholder="Enter number of dependents"
        )

        property_area = st.selectbox(
            "Property / Residential Area",
            [
                "Select property area",
                "Urban",
                "Semiurban",
                "Rural"
            ]
        )


    st.divider()


    # -------------------------------------------------------------------------
    # 2. FINANCIAL INFORMATION
    # -------------------------------------------------------------------------

    st.subheader("2. Financial Information")

    f1, f2 = st.columns(2)

    with f1:

        applicant_income = st.number_input(
            "Applicant Annual Income (₹)",
            min_value=10000.0,
            value=None,
            step=25000.0,
            placeholder="Enter annual income",
            help=(
                "Gross annual earnings of the primary applicant "
                "before deductions."
            )
        )

        coapplicant_income = st.number_input(
            "Co-applicant Annual Income (₹)",
            min_value=0.0,
            value=None,
            step=25000.0,
            placeholder="Enter 0 if there is no co-applicant",
            help=(
                "Annual income of spouse, parent, or co-borrower. "
                "Enter 0 if there is no co-applicant."
            )
        )

    with f2:

        st.caption(
            "Declared Assets (Estimated market value in ₹):"
        )

        a1, a2 = st.columns(2)

        with a1:

            res_assets = st.number_input(
                "Residential Property",
                min_value=0.0,
                value=None,
                step=50000.0,
                placeholder="Enter value"
            )

            comm_assets = st.number_input(
                "Commercial Property",
                min_value=0.0,
                value=None,
                step=50000.0,
                placeholder="Enter value"
            )

        with a2:

            lux_assets = st.number_input(
                "Luxury / Vehicles",
                min_value=0.0,
                value=None,
                step=25000.0,
                placeholder="Enter value"
            )

            bank_assets = st.number_input(
                "Bank Deposits / Liquid",
                min_value=0.0,
                value=None,
                step=25000.0,
                placeholder="Enter value"
            )


    st.divider()


    # -------------------------------------------------------------------------
    # 3. LOAN INFORMATION
    # -------------------------------------------------------------------------

    st.subheader("3. Loan Information")

    l1, l2 = st.columns(2)

    with l1:

        loan_amount = st.number_input(
            "Requested Loan Amount (₹)",
            min_value=10000.0,
            value=None,
            step=50000.0,
            placeholder="Enter requested loan amount",
            help=(
                "Total principal loan amount requested "
                "from the lender."
            )
        )

    with l2:

        loan_term_months = st.number_input(
            "Repayment Term (Months)",
            min_value=12,
            max_value=480,
            value=None,
            step=12,
            placeholder="Enter tenure in months",
            help=(
                "Loan duration in months. "
                "Example: 240 = 20 years."
            )
        )

        if loan_term_months is not None:

            st.caption(
                f"Equivalent Tenure: "
                f"**{loan_term_months / 12:.1f} Years**"
            )


    st.divider()


    # -------------------------------------------------------------------------
    # 4. CREDIT INFORMATION
    # -------------------------------------------------------------------------

    st.subheader("4. Credit Information")

    is_new_to_credit = st.checkbox(
        "I am new to credit / I don't have a CIBIL score yet",
        value=False,
        help=(
            "Check this if the applicant has no prior loan "
            "or credit card record."
        )
    )

    cr1, cr2 = st.columns(2)

    with cr1:

        if is_new_to_credit:

            st.info(
                "ℹ️ CIBIL score disabled for first-time / "
                "new-to-credit borrower."
            )

            cibil_score = None

        else:

            cibil_score = st.number_input(
                "CIBIL Bureau Score",
                min_value=300,
                max_value=900,
                value=None,
                step=1,
                placeholder="Enter CIBIL score",
                help=(
                    "Credit bureau rating between "
                    "300 (Poor) and 900 (Excellent)."
                )
            )

            if cibil_score is not None:

                if cibil_score >= 750:

                    st.success(
                        f"Score: {cibil_score} (Excellent)"
                    )

                elif cibil_score >= 680:

                    st.info(
                        f"Score: {cibil_score} (Good)"
                    )

                elif cibil_score >= 580:

                    st.warning(
                        f"Score: {cibil_score} (Fair)"
                    )

                else:

                    st.error(
                        f"Score: {cibil_score} (Poor)"
                    )

    with cr2:

        if is_new_to_credit:

            credit_history = None

            st.caption(
                "Past Credit History: **Not on file**"
            )

        else:

            ch_choice = st.selectbox(
                "Past Repayment Track Record",
                [
                    "Select repayment history",
                    "Clean Record (No defaults)",
                    "Adverse Record (Past delays / defaults)",
                    "No history on file"
                ]
            )

            if ch_choice == "Clean Record (No defaults)":

                credit_history = 1.0

            elif ch_choice == "Adverse Record (Past delays / defaults)":

                credit_history = 0.0

            else:

                credit_history = None


    # -------------------------------------------------------------------------
    # SUBMIT BUTTON
    # -------------------------------------------------------------------------

    submit_button = st.form_submit_button(
        "Predict Loan Approval",
        use_container_width=True,
        type="primary"
    )


# =============================================================================
# VALIDATION + PREDICTION
# =============================================================================

if submit_button:

    validation_errors = []


    # -------------------------------------------------------------------------
    # APPLICANT VALIDATION
    # -------------------------------------------------------------------------

    if gender == "Select gender":

        validation_errors.append(
            "Please select your gender."
        )

    if education == "Select education":

        validation_errors.append(
            "Please select your education level."
        )

    if married == "Select marital status":

        validation_errors.append(
            "Please select your marital status."
        )

    if self_employed == "Select employment type":

        validation_errors.append(
            "Please select your employment type."
        )

    if dependents is None:

        validation_errors.append(
            "Please enter the number of dependents."
        )

    if property_area == "Select property area":

        validation_errors.append(
            "Please select the property / residential area."
        )


    # -------------------------------------------------------------------------
    # FINANCIAL VALIDATION
    # -------------------------------------------------------------------------

    if applicant_income is None:

        validation_errors.append(
            "Please enter the applicant's annual income."
        )

    if coapplicant_income is None:

        validation_errors.append(
            "Please enter the co-applicant's annual income, "
            "or enter 0 if there is no co-applicant."
        )

    if res_assets is None:

        validation_errors.append(
            "Please enter the residential property value."
        )

    if comm_assets is None:

        validation_errors.append(
            "Please enter the commercial property value."
        )

    if lux_assets is None:

        validation_errors.append(
            "Please enter the luxury / vehicle asset value."
        )

    if bank_assets is None:

        validation_errors.append(
            "Please enter the bank deposits / liquid asset value."
        )


    # -------------------------------------------------------------------------
    # LOAN VALIDATION
    # -------------------------------------------------------------------------

    if loan_amount is None:

        validation_errors.append(
            "Please enter the requested loan amount."
        )

    if loan_term_months is None:

        validation_errors.append(
            "Please enter the repayment term."
        )


    # -------------------------------------------------------------------------
    # CREDIT VALIDATION
    # -------------------------------------------------------------------------

    if not is_new_to_credit:

        if cibil_score is None:

            validation_errors.append(
                "Please enter the CIBIL score, "
                "or select 'I am new to credit'."
            )

        if credit_history is None:

            validation_errors.append(
                "Please select the past repayment track record."
            )


    # -------------------------------------------------------------------------
    # SHOW VALIDATION ERRORS
    # -------------------------------------------------------------------------

    if validation_errors:

        st.error(
            "Please complete the required fields "
            "before requesting a prediction."
        )

        for error in validation_errors:

            st.write(f"- {error}")


    # -------------------------------------------------------------------------
    # PREDICTION
    # -------------------------------------------------------------------------

    else:

        payload = {

            "gender": gender,

            "married": married,

            "dependents": float(
                dependents
            ),

            "education": education,

            "self_employed": self_employed,

            "applicant_income": float(
                applicant_income
            ),

            "coapplicant_income": float(
                coapplicant_income
            ),

            "loan_amount": float(
                loan_amount
            ),

            "loan_term_months": float(
                loan_term_months
            ),

            "credit_history": credit_history,

            "cibil_score": (
                float(cibil_score)
                if cibil_score is not None
                else None
            ),

            "property_area": property_area,

            "residential_assets": float(
                res_assets
            ),

            "commercial_assets": float(
                comm_assets
            ),

            "luxury_assets": float(
                lux_assets
            ),

            "bank_asset_value": float(
                bank_assets
            )
        }


        # ---------------------------------------------------------------------
        # CALL MODEL
        # ---------------------------------------------------------------------

        with st.spinner(
            "Processing applicant profile through decision engine..."
        ):

            result = predict(payload)


        # ---------------------------------------------------------------------
        # ENGINE VALIDATION ERROR
        # ---------------------------------------------------------------------

        if not result.get("success", False):

            st.error("Validation Error:")

            for err in result.get("errors", []):

                st.write(f"- {err}")


        # ---------------------------------------------------------------------
        # SUCCESSFUL PREDICTION
        # ---------------------------------------------------------------------

        else:

            is_approved = result["approved"]

            prob = result["probability"]

            fin = result.get(
                "financial_summary",
                {}
            )


            # -----------------------------------------------------------------
            # DECISION SECTION
            # -----------------------------------------------------------------

            st.subheader(
                "Decision & Risk Analysis"
            )


            # -----------------------------------------------------------------
            # APPROVED
            # -----------------------------------------------------------------

            if is_approved:

                st.markdown(
                    f"""
                    <div class="decision-approved">

                        <h2 style="
                            margin:0 0 6px;
                            font-weight:800;
                            font-size:1.8rem;
                        ">
                            ✓ LOAN LIKELY TO BE APPROVED
                        </h2>

                        <p style="
                            margin:0;
                            font-size:1.05rem;
                        ">
                            Approval Probability:
                            <strong>{prob}%</strong>
                        </p>

                        <p style="
                            margin:4px 0 0;
                            opacity:0.85;
                            font-size:0.95rem;
                        ">
                            Applicant demonstrates healthy financial
                            solvency and manageable leverage.
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # -----------------------------------------------------------------
            # REJECTED
            # -----------------------------------------------------------------

            else:

                st.markdown(
                    f"""
                    <div class="decision-rejected">

                        <h2 style="
                            margin:0 0 6px;
                            font-weight:800;
                            font-size:1.8rem;
                        ">
                            ✕ LOAN LIKELY TO BE REJECTED
                        </h2>

                        <p style="
                            margin:0;
                            font-size:1.05rem;
                        ">
                            Approval Probability:
                            <strong>{prob}%</strong>
                        </p>

                        <p style="
                            margin:4px 0 0;
                            opacity:0.85;
                            font-size:0.95rem;
                        ">
                            Application presents elevated credit
                            risk or high debt-to-income leverage.
                        </p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # -----------------------------------------------------------------
            # FINANCIAL METRICS
            # -----------------------------------------------------------------

            m1, m2, m3, m4 = st.columns(4)


            with m1:

                st.metric(
                    "Total Annual Income",
                    format_inr(
                        fin.get("total_income")
                    )
                )


            with m2:

                st.metric(
                    "Loan-to-Income (LTI)",
                    f"{fin.get('loan_to_income_ratio', 0):.2f}x"
                )


            with m3:

                monthly_principal = fin.get(
                    "approx_monthly_emi"
                )

                monthly_value = (
                    f"{format_inr(monthly_principal)}/mo"
                    if monthly_principal is not None
                    else "N/A"
                )

                st.metric(
                    "Est. Monthly Principal",
                    monthly_value
                )


            with m4:

                st.metric(
                    "Declared Total Assets",
                    format_inr(
                        fin.get("total_assets")
                    )
                )


            # -----------------------------------------------------------------
            # PROFILE FACTORS
            # -----------------------------------------------------------------

            st.divider()

            col_pos, col_risk = st.columns(2)


            # -----------------------------------------------------------------
            # POSITIVE FACTORS
            # -----------------------------------------------------------------

            with col_pos:

                st.markdown(
                    "#### ✓ Profile Strengths"
                )

                pos_factors = result.get(
                    "positive_factors",
                    []
                )

                if pos_factors:

                    for pf in pos_factors:

                        st.write(
                            f"- {pf}"
                        )

                else:

                    st.write(
                        "- Standard profile credentials."
                    )


            # -----------------------------------------------------------------
            # RISK FACTORS
            # -----------------------------------------------------------------

            with col_risk:

                st.markdown(
                    "#### ⚠️ Risk Observations"
                )

                risk_factors = result.get(
                    "risk_factors",
                    []
                )

                if risk_factors:

                    for rf in risk_factors:

                        st.write(
                            f"- {rf}"
                        )

                else:

                    st.write(
                        "- No critical risk flags detected."
                    )


# =============================================================================
# DISCLAIMER
# =============================================================================

st.divider()

st.caption(
    "⚠️ This system is a machine-learning prediction tool for "
    "educational/project purposes. The prediction should not be treated "
    "as an actual lending or financial decision."
)