"""
combine_datasets.py
===================
Combines and standardizes:
  - Dataset 1: loan_train.csv (Home Loan demographic & credit history dataset)
  - Dataset 2: loan_approval_dataset.csv (Financial & asset-backed CIBIL loan dataset)

Produces:
  clean_loan_approval.csv

Follows the unified 22-column specification:
  - Personal: gender, married, dependents, education, self_employed
  - Income: applicant_income, coapplicant_income, total_income
  - Loan: loan_amount, loan_term_months
  - Credit: credit_history, cibil_score (kept distinct!)
  - Property: property_area
  - Assets: residential_assets, commercial_assets, luxury_assets, bank_asset_value, total_assets
  - Ratios: loan_to_income_ratio, asset_to_income_ratio, loan_to_asset_ratio
  - Target: target (1 = Approved, 0 = Rejected)
"""

import pandas as pd
import numpy as np


def load_and_standardize():
    print("=" * 60)
    print("STEP 1: LOADING RAW DATASETS")
    print("=" * 60)

    df1_raw = pd.read_csv("loan_train.csv")
    df2_raw = pd.read_csv("loan_approval_dataset.csv")

    # Clean whitespace in column names
    df1_raw.columns = df1_raw.columns.str.strip()
    df2_raw.columns = df2_raw.columns.str.strip()

    print(f"Dataset 1 (loan_train.csv) raw shape: {df1_raw.shape}")
    print(f"Dataset 2 (loan_approval_dataset.csv) raw shape: {df2_raw.shape}")

    # ==========================================
    # DATASET 1 STANDARDIZATION
    # ==========================================
    print("\n" + "=" * 60)
    print("STEP 2: STANDARDIZING DATASET 1 (loan_train.csv)")
    print("=" * 60)

    df1 = pd.DataFrame()

    # Demographics
    df1["gender"] = df1_raw["Gender"].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan})
    df1["married"] = df1_raw["Married"].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan})
    
    # Dependents: '3+' -> 3
    df1["dependents"] = (
        df1_raw["Dependents"]
        .astype(str)
        .str.strip()
        .replace({"3+": "3", "nan": np.nan, "None": np.nan})
    )
    df1["dependents"] = pd.to_numeric(df1["dependents"], errors="coerce")

    # Education: strip whitespace
    df1["education"] = df1_raw["Education"].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan})

    # Self Employed: strip whitespace
    df1["self_employed"] = df1_raw["Self_Employed"].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan})

    # Income: Dataset 1 records monthly income. Annualize (* 12) so units match Dataset 2 (income_annum)
    # Unit alignment verified: LoanAmount is in thousands (* 1000).
    df1["applicant_income"] = pd.to_numeric(df1_raw["ApplicantIncome"], errors="coerce") * 12
    df1["coapplicant_income"] = pd.to_numeric(df1_raw["CoapplicantIncome"], errors="coerce").fillna(0) * 12
    df1["total_income"] = df1["applicant_income"] + df1["coapplicant_income"]

    # Loan info: LoanAmount in df1 is in thousands of currency -> multiply by 1000
    df1["loan_amount"] = pd.to_numeric(df1_raw["LoanAmount"], errors="coerce") * 1000
    # Loan_Amount_Term is already in months
    df1["loan_term_months"] = pd.to_numeric(df1_raw["Loan_Amount_Term"], errors="coerce")

    # Credit History: binary flag (0 or 1)
    df1["credit_history"] = pd.to_numeric(df1_raw["Credit_History"], errors="coerce")
    # CIBIL score is not present in Dataset 1 -> NaN
    df1["cibil_score"] = np.nan

    # Property Area: Urban / Semiurban / Rural
    df1["property_area"] = df1_raw["Property_Area"].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan})

    # Assets: Not present in Dataset 1 -> keep NaN
    df1["residential_assets"] = np.nan
    df1["commercial_assets"] = np.nan
    df1["luxury_assets"] = np.nan
    df1["bank_asset_value"] = np.nan
    df1["total_assets"] = np.nan

    # Target: Y -> 1, N -> 0
    df1["target"] = df1_raw["Loan_Status"].astype(str).str.strip().map({"Y": 1, "N": 0})

    print(f"Dataset 1 standardized rows: {len(df1)}")

    # ==========================================
    # DATASET 2 STANDARDIZATION
    # ==========================================
    print("\n" + "=" * 60)
    print("STEP 3: STANDARDIZING DATASET 2 (loan_approval_dataset.csv)")
    print("=" * 60)

    df2 = pd.DataFrame()

    # Demographics not present in Dataset 2
    df2["gender"] = np.nan
    df2["married"] = np.nan

    # Dependents: numeric (0 to 5)
    df2["dependents"] = pd.to_numeric(df2_raw["no_of_dependents"], errors="coerce")

    # Education & Self Employed: strip leading/trailing spaces
    df2["education"] = df2_raw["education"].astype(str).str.strip()
    df2["self_employed"] = df2_raw["self_employed"].astype(str).str.strip()

    # Income: income_annum is annual income; coapplicant_income is 0
    df2["applicant_income"] = pd.to_numeric(df2_raw["income_annum"], errors="coerce")
    df2["coapplicant_income"] = 0.0
    df2["total_income"] = df2["applicant_income"]

    # Loan info: loan_amount is absolute currency
    df2["loan_amount"] = pd.to_numeric(df2_raw["loan_amount"], errors="coerce")
    # loan_term in df2 is in years (2 to 20) -> multiply by 12 to convert to months
    df2["loan_term_months"] = pd.to_numeric(df2_raw["loan_term"], errors="coerce") * 12

    # Credit: CIBIL score is 300-900; credit_history flag is not present -> NaN
    df2["credit_history"] = np.nan
    df2["cibil_score"] = pd.to_numeric(df2_raw["cibil_score"], errors="coerce")

    # Property Area not present in Dataset 2
    df2["property_area"] = np.nan

    # Assets: present in Dataset 2
    # (clip small negative data entry artifacts at 0)
    df2["residential_assets"] = pd.to_numeric(df2_raw["residential_assets_value"], errors="coerce").clip(lower=0)
    df2["commercial_assets"] = pd.to_numeric(df2_raw["commercial_assets_value"], errors="coerce").clip(lower=0)
    df2["luxury_assets"] = pd.to_numeric(df2_raw["luxury_assets_value"], errors="coerce").clip(lower=0)
    df2["bank_asset_value"] = pd.to_numeric(df2_raw["bank_asset_value"], errors="coerce").clip(lower=0)
    df2["total_assets"] = (
        df2["residential_assets"]
        + df2["commercial_assets"]
        + df2["luxury_assets"]
        + df2["bank_asset_value"]
    )

    # Target: Approved -> 1, Rejected -> 0
    df2["target"] = (
        df2_raw["loan_status"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"approved": 1, "rejected": 0})
    )

    print(f"Dataset 2 standardized rows: {len(df2)}")

    # ==========================================
    # COMBINE DATASETS
    # ==========================================
    print("\n" + "=" * 60)
    print("STEP 4: COMBINING DATASETS")
    print("=" * 60)

    combined = pd.concat([df1, df2], ignore_index=True)

    # ==========================================
    # FEATURE ENGINEERING (FINANCIAL RATIOS)
    # ==========================================
    print("\n" + "=" * 60)
    print("STEP 5: COMPUTING FINANCIAL RATIOS")
    print("=" * 60)

    # 1. Loan-to-Income: loan_amount / total_income
    combined["loan_to_income_ratio"] = np.where(
        combined["total_income"] > 0,
        combined["loan_amount"] / combined["total_income"],
        np.nan
    )

    # 2. Asset-to-Income: total_assets / total_income
    combined["asset_to_income_ratio"] = np.where(
        (combined["total_income"] > 0) & combined["total_assets"].notnull(),
        combined["total_assets"] / combined["total_income"],
        np.nan
    )

    # 3. Loan-to-Asset: loan_amount / total_assets
    combined["loan_to_asset_ratio"] = np.where(
        (combined["total_assets"] > 0) & combined["total_assets"].notnull(),
        combined["loan_amount"] / combined["total_assets"],
        np.nan
    )

    # ==========================================
    # COLUMN ORDER & VALIDATION
    # ==========================================
    ordered_columns = [
        "gender",
        "married",
        "dependents",
        "education",
        "self_employed",
        "applicant_income",
        "coapplicant_income",
        "total_income",
        "loan_amount",
        "loan_term_months",
        "credit_history",
        "cibil_score",
        "property_area",
        "residential_assets",
        "commercial_assets",
        "luxury_assets",
        "bank_asset_value",
        "total_assets",
        "loan_to_income_ratio",
        "asset_to_income_ratio",
        "loan_to_asset_ratio",
        "target"
    ]

    combined = combined[ordered_columns]

    # Drop any row where target is missing
    combined = combined.dropna(subset=["target"])
    combined["target"] = combined["target"].astype(int)

    # ==========================================
    # SAVE CLEAN DATASET
    # ==========================================
    output_path = "clean_loan_approval.csv"
    combined.to_csv(output_path, index=False)
    print(f"\n[OK] Clean combined dataset successfully saved to: {output_path}")

    # ==========================================
    # INSPECTION REPORT
    # ==========================================
    print("\n" + "=" * 60)
    print("DATASET INSPECTION REPORT")
    print("=" * 60)

    print(f"\n1. Final Shape: {combined.shape}")

    print("\n2. Columns (total 22):")
    for i, col in enumerate(combined.columns, 1):
        print(f"   {i:2d}. {col} ({combined[col].dtype})")

    print("\n3. Target Distribution:")
    val_counts = combined["target"].value_counts()
    for val, count in val_counts.items():
        label = "Approved (1)" if val == 1 else "Rejected (0)"
        pct = (count / len(combined)) * 100
        print(f"   {label}: {count} ({pct:.2f}%)")

    print("\n4. Missing Values Summary:")
    missing = combined.isnull().sum()
    for col, count in missing[missing > 0].items():
        pct = (count / len(combined)) * 100
        print(f"   {col:25s}: {count:4d} missing ({pct:.1f}%)")

    print("\n5. Loan Amount Summary:")
    print(combined["loan_amount"].describe())

    print("\n6. Total Income Summary:")
    print(combined["total_income"].describe())

    print("\n7. Loan Term (Months) Value Counts:")
    print(combined["loan_term_months"].value_counts().sort_index())

    print("\n8. Credit Features:")
    print("   credit_history distribution (non-null):")
    print(combined["credit_history"].value_counts(dropna=False))
    print("   cibil_score summary (non-null):")
    print(combined["cibil_score"].describe())

    print("\n9. Assets Summary:")
    print("   total_assets summary (non-null):")
    print(combined["total_assets"].describe())

    print("\n10. Financial Ratios Summary:")
    for r in ["loan_to_income_ratio", "asset_to_income_ratio", "loan_to_asset_ratio"]:
        print(f"\n   {r}:")
        print(combined[r].describe())

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return combined


if __name__ == "__main__":
    load_and_standardize()