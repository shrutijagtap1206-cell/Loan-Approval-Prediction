
"""
engine.py
=========
Production Prediction Engine for the Loan Application Approval Prediction System.

Responsibilities:
  1. Load pre-trained model bundle (Pipeline + metadata) from loan_model.joblib
  2. Validate incoming applicant parameters (ranges, types, business rules)
  3. Feature engineering (income sum, asset aggregation, financial ratios)
  4. Perform inference with full preprocessing pipeline
  5. Return decision, probability, and risk/financial factor explanations.
"""

import os
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "loan_model.joblib")
_MODEL_BUNDLE = None


def get_model():
    """Lazy loader / singleton for the trained model bundle."""
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file '{MODEL_PATH}' not found. Please run train_model.py first."
            )
        _MODEL_BUNDLE = joblib.load(MODEL_PATH)
    return _MODEL_BUNDLE


def validate_applicant(data):
    """
    Validates applicant input data.
    Returns: (cleaned_dict, error_list)
    """
    errors = []
    cleaned = {}

    # 1. Categorical fields
    gender = data.get("gender")
    if gender not in (None, "", "Male", "Female", "Other"):
        errors.append("Gender must be 'Male', 'Female', 'Other', or left empty.")
    cleaned["gender"] = gender if gender in ("Male", "Female") else None

    married = data.get("married")
    if married not in (None, "", "Yes", "No"):
        errors.append("Marital status must be 'Yes', 'No', or left empty.")
    cleaned["married"] = married if married in ("Yes", "No") else None

    education = data.get("education", "Graduate")
    if education not in ("Graduate", "Not Graduate"):
        errors.append("Education must be 'Graduate' or 'Not Graduate'.")
    cleaned["education"] = education

    self_employed = data.get("self_employed", "No")
    if self_employed not in ("Yes", "No"):
        errors.append("Self employed must be 'Yes' or 'No'.")
    cleaned["self_employed"] = self_employed

    property_area = data.get("property_area")
    if property_area not in (None, "", "Urban", "Semiurban", "Rural"):
        errors.append("Property Area must be 'Urban', 'Semiurban', 'Rural', or left empty.")
    cleaned["property_area"] = property_area if property_area in ("Urban", "Semiurban", "Rural") else None

    # 2. Numerical fields
    def parse_float(val, name, min_val=None, max_val=None, default=None, required=False):
        if val is None or val == "":
            if required:
                errors.append(f"{name} is required.")
                return default
            return default
        try:
            num = float(val)
            if min_val is not None and num < min_val:
                errors.append(f"{name} cannot be less than {min_val}.")
            if max_val is not None and num > max_val:
                errors.append(f"{name} cannot be greater than {max_val}.")
            return num
        except (ValueError, TypeError):
            errors.append(f"{name} must be a valid number.")
            return default

    cleaned["dependents"] = parse_float(data.get("dependents"), "Dependents", min_val=0, max_val=10, default=0.0)
    cleaned["applicant_income"] = parse_float(data.get("applicant_income"), "Applicant Income", min_val=0.0, required=True)
    cleaned["coapplicant_income"] = parse_float(data.get("coapplicant_income"), "Co-applicant Income", min_val=0.0, default=0.0)
    cleaned["loan_amount"] = parse_float(data.get("loan_amount"), "Loan Amount", min_val=1.0, required=True)
    cleaned["loan_term_months"] = parse_float(data.get("loan_term_months"), "Loan Term (Months)", min_val=1.0, max_val=600.0, required=True)

    # Total income validation
    if cleaned["applicant_income"] is not None and cleaned["coapplicant_income"] is not None:
        total_income = cleaned["applicant_income"] + cleaned["coapplicant_income"]
        if total_income <= 0:
            errors.append("Total income (Applicant + Co-applicant) must be greater than 0.")
        cleaned["total_income"] = total_income
    else:
        cleaned["total_income"] = None

    # Credit info
    cleaned["credit_history"] = parse_float(data.get("credit_history"), "Credit History", min_val=0.0, max_val=1.0, default=None)
    cleaned["cibil_score"] = parse_float(data.get("cibil_score"), "CIBIL Score", min_val=300.0, max_val=900.0, default=None)

    # Assets
    cleaned["residential_assets"] = parse_float(data.get("residential_assets"), "Residential Assets", min_val=0.0, default=None)
    cleaned["commercial_assets"] = parse_float(data.get("commercial_assets"), "Commercial Assets", min_val=0.0, default=None)
    cleaned["luxury_assets"] = parse_float(data.get("luxury_assets"), "Luxury Assets", min_val=0.0, default=None)
    cleaned["bank_asset_value"] = parse_float(data.get("bank_asset_value"), "Bank Asset Value", min_val=0.0, default=None)

    return cleaned, errors


def predict(applicant_data):
    """
    Main entry point for loan assessment:
      1. Validates input
      2. Computes ratios
      3. Passes through preprocessor + model
      4. Returns structured results
    """
    cleaned, errors = validate_applicant(applicant_data)
    if errors:
        return {
            "success": False,
            "errors": errors
        }

    # Derive Asset Totals if any asset was provided
    asset_fields = ["residential_assets", "commercial_assets", "luxury_assets", "bank_asset_value"]
    has_any_asset = any(cleaned[f] is not None for f in asset_fields)

    if has_any_asset:
        cleaned["total_assets"] = sum((cleaned[f] or 0.0) for f in asset_fields)
    else:
        cleaned["total_assets"] = np.nan

    # Derive Ratios
    total_income = cleaned["total_income"]
    loan_amount = cleaned["loan_amount"]
    total_assets = cleaned["total_assets"]

    cleaned["loan_to_income_ratio"] = loan_amount / total_income if total_income > 0 else np.nan

    if not np.isnan(total_assets) and total_income > 0:
        cleaned["asset_to_income_ratio"] = total_assets / total_income
    else:
        cleaned["asset_to_income_ratio"] = np.nan

    if not np.isnan(total_assets) and total_assets > 0:
        cleaned["loan_to_asset_ratio"] = loan_amount / total_assets
    else:
        cleaned["loan_to_asset_ratio"] = np.nan

    # Load model and prepare feature row in exact expected order
    bundle = get_model()
    pipeline = bundle["pipeline"]
    feature_order = bundle["feature_order"]

    feature_df = pd.DataFrame([cleaned])[feature_order]

    # Predict
    prob = float(pipeline.predict_proba(feature_df)[0, 1])
    is_approved = prob >= 0.50

    # Risk & financial indicators
    notes = []
    positive_factors = []

    cibil = cleaned.get("cibil_score")
    ch = cleaned.get("credit_history")

    if cibil is not None:
        if cibil >= 750:
            positive_factors.append(f"Excellent CIBIL score ({cibil:.0f}).")
        elif cibil < 600:
            notes.append(f"Low CIBIL score ({cibil:.0f}) indicates high historical credit risk.")
    elif ch is not None:
        if ch == 1.0:
            positive_factors.append("Clean historical credit repayment record.")
        else:
            notes.append("Adverse credit history on file.")
    else:
        positive_factors.append("New to credit: assessed based on income capacity and debt-to-income balance.")

    lti = cleaned["loan_to_income_ratio"]
    if lti > 4.5:
        notes.append(f"High loan-to-income ratio ({lti:.2f}x annual income).")
    elif lti < 2.0:
        positive_factors.append(f"Healthy loan-to-income ratio ({lti:.2f}x annual income).")

    if not np.isnan(total_assets) and total_assets > 0:
        lar = cleaned["loan_to_asset_ratio"]
        if lar < 0.5:
            positive_factors.append(f"Strong asset backing (loan is {lar*100:.0f}% of assets).")
        elif lar > 1.0:
            notes.append("Loan requested exceeds declared total assets.")

    # Approximate Monthly EMI (Simple amortization estimate for context)
    term_months = cleaned["loan_term_months"]
    approx_emi = loan_amount / term_months if term_months > 0 else 0
    monthly_income = total_income / 12.0
    emi_share = (approx_emi / monthly_income) * 100.0 if monthly_income > 0 else 0

    return {
        "success": True,
        "approved": is_approved,
        "decision": "APPROVED" if is_approved else "REJECTED",
        "probability": round(prob * 100.0, 1),
        "icon": "✓" if is_approved else "✕",
        "summary": (
            "LOAN LIKELY TO BE APPROVED"
            if is_approved
            else "LOAN LIKELY TO BE REJECTED"
        ),
        "model_used": bundle.get("model_name", "Gradient Boosting Pipeline"),
        "financial_summary": {
            "total_income": round(total_income),
            "loan_amount": round(loan_amount),
            "loan_term_months": int(term_months),
            "loan_to_income_ratio": round(lti, 2),
            "total_assets": round(total_assets) if not np.isnan(total_assets) else None,
            "approx_monthly_emi": round(approx_emi),
            "emi_share_of_income_pct": round(emi_share, 1)
        },
        "positive_factors": positive_factors,
        "risk_factors": notes
    }
