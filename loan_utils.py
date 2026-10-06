"""Shared helpers for the loan model (used by the notebook and by app.py)."""
import numpy as np
import pandas as pd

THRESHOLD = 0.60            # approve only when chance of approval >= 60%
BORDERLINE = (0.50, 0.70)   # chances in this band are flagged as borderline
MAX_PAYMENT_SHARE = 0.50    # monthly payment above 50% of income -> manual review
NUMERIC_CHECKS = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
ALL_FIELDS = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome',
              'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area']
REQUIRED_FIELDS = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']


def add_ratios(df):
    """Turn raw income / loan columns into ratios (these stay in a normal range)."""
    df = df.copy()
    df['CoapplicantIncome'] = df['CoapplicantIncome'].fillna(0)      # no co-applicant = 0
    total = (df['ApplicantIncome'] + df['CoapplicantIncome']).replace(0, np.nan)
    term = df['Loan_Amount_Term'].replace(0, np.nan)
    df['EMI_to_Income'] = df['LoanAmount'] * 1000 / term / total     # monthly payment / income
    df['Loan_to_Income'] = df['LoanAmount'] * 1000 / total           # loan size / income
    return df.drop(columns=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount'])


def compute_ranges(train):
    """Lowest / highest value of each numeric column seen in training."""
    return {c: (float(train[c].min()), float(train[c].max())) for c in NUMERIC_CHECKS}


def predict_applicant(model, ranges, applicant):
    """applicant: dict with the 11 input fields. Returns a dict with the decision."""
    df = pd.DataFrame([applicant]).reindex(columns=ALL_FIELDS)    # absent fields become blank
    for c in NUMERIC_CHECKS + ['Credit_History']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    missing = [f for f in REQUIRED_FIELDS if pd.isna(df.loc[0, f])]
    if missing:
        return {"probability": None, "decision": "INCOMPLETE", "model_lean": None,
                "payment_share": None, "severe": True,
                "notes": ["Cannot predict. Missing or invalid: " + ", ".join(missing)]}

    feats = add_ratios(df)
    prob = float(model.predict_proba(feats)[0, 1])
    emi = feats['EMI_to_Income'].iloc[0]
    emi = None if pd.isna(emi) else float(emi)

    mild, severe = [], []
    for col in NUMERIC_CHECKS:
        v = df.loc[0, col]
        lo, hi = ranges[col]
        if pd.isna(v) or lo <= v <= hi:
            continue
        if v > hi:
            times = v / hi
            (mild if times <= 2 else severe).append(
                f"{col} = {v:,.0f} is {times:.1f}x the highest value seen ({hi:,.0f})")
        else:
            times = lo / max(v, 1e-9)
            (mild if times <= 2 else severe).append(
                f"{col} = {v:,.0f} is below the lowest value seen ({lo:,.0f})")
    if emi is not None and emi > MAX_PAYMENT_SHARE:
        severe.append(f"Monthly payment is {emi*100:.0f}% of income (limit {MAX_PAYMENT_SHARE*100:.0f}%)")
    if emi is None:
        severe.append("Total income or loan term is zero/missing, so the payment cannot be checked")

    lean = "APPROVED" if prob >= THRESHOLD else "REJECTED"
    if severe:
        decision = "MANUAL REVIEW"
    else:
        decision = lean
    notes = severe + mild
    if not severe and BORDERLINE[0] <= prob < BORDERLINE[1]:
        notes.append("Borderline case: the model is not confident, a second look is recommended")

    return {"probability": round(prob * 100, 1), "decision": decision, "model_lean": lean,
            "payment_share": None if emi is None else round(emi * 100, 1),
            "notes": notes, "severe": bool(severe)}
