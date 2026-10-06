# Loan Application Approval Prediction System

An end-to-end Machine Learning prediction system that takes an applicant's financial, demographic, and credit profile to predict:
- **Decision**: `APPROVED` / `REJECTED`
- **Approval Probability**: `0%` – `100%`
- **Key Financial Ratios**: Loan-to-Income, Asset-to-Income, and estimated monthly repayment obligations.

---

## 1. Project Overview & Architecture

This system generalizes loan approval prediction beyond home-loan-specific scenarios by combining multi-source applicant data:
1. **Source Datasets**:
   - `loan_train.csv` (Demographics, employment, loan terms, binary credit history)
   - `loan_approval_dataset.csv` (Financials, continuous CIBIL score, declared asset values)
   - `application_train.csv` (*Preserved for future default-risk modeling; excluded from approval model*)
2. **Data Pipeline**:
   - `combine_datasets.py` harmonizes units, computes debt & asset ratios, and exports `clean_loan_approval.csv`.
3. **Machine Learning Pipeline**:
   - `train_model.py` / `loanprediction.ipynb` compares **Logistic Regression**, **Random Forest**, and **Gradient Boosting**.
   - The best model (**Gradient Boosting**, ROC-AUC `0.9966`, F1-Score `0.9807`) is bundled with the complete preprocessing pipeline into `loan_model.joblib`.
4. **Inference & Service**:
   - `engine.py` provides validation, ratio computation, and model inference without retraining.
   - `app.py` exposes `POST /predict` and serves an interactive web interface.

---

## 2. Standardized Feature Set (22 Columns)

| Category | Columns |
| :--- | :--- |
| **Personal** | `gender`, `married`, `dependents`, `education`, `self_employed` |
| **Income** | `applicant_income`, `coapplicant_income`, `total_income` |
| **Loan** | `loan_amount`, `loan_term_months` |
| **Credit** | `credit_history` (0 or 1), `cibil_score` (300 to 900) *(Kept distinct)* |
| **Property** | `property_area` (`Urban`, `Semiurban`, `Rural`) |
| **Assets** | `residential_assets`, `commercial_assets`, `luxury_assets`, `bank_asset_value`, `total_assets` |
| **Financial Ratios** | `loan_to_income_ratio`, `asset_to_income_ratio`, `loan_to_asset_ratio` |
| **Target** | `target` (`1` = Approved, `0` = Rejected) |

---

## 3. Model Benchmark & Evaluation

Tested on an 80/20 hold-out stratified test set (977 test samples):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 89.87% | 0.9033 | 0.9399 | 0.9212 | 0.9675 | 0.9555 ± 0.0076 |
| **Random Forest** | 97.75% | 0.9714 | 0.9935 | 0.9823 | 0.9948 | 0.9937 ± 0.0015 |
| **Gradient Boosting (Selected)** | **97.54%** | **0.9728** | **0.9886** | **0.9807** | **0.9966** | **0.9956 ± 0.0020** |

---

## 4. Quick Start

### 1. Run Data Preparation & Inspection
```bash
python combine_datasets.py
```

### 2. Train and Export Model Pipeline
```bash
python train_model.py
```
*(Or run the interactive cells in `loanprediction.ipynb`)*

### 3. Run Automated Test Suite
```bash
python test_system.py
```

### 4. Launch the Web Application
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

---

## 5. API Reference

### `POST /predict`
**Request Payload Example:**
```json
{
  "gender": "Male",
  "married": "Yes",
  "dependents": 2,
  "education": "Graduate",
  "self_employed": "No",
  "applicant_income": 600000,
  "coapplicant_income": 240000,
  "loan_amount": 2000000,
  "loan_term_months": 240,
  "credit_history": 1,
  "cibil_score": 780,
  "property_area": "Urban",
  "residential_assets": 3000000,
  "commercial_assets": 1000000,
  "luxury_assets": 500000,
  "bank_asset_value": 800000
}
```

**Response Example:**
```json
{
  "approved": true,
  "decision": "APPROVED",
  "probability": 73.9,
  "summary": "LOAN LIKELY TO BE APPROVED",
  "financial_summary": {
    "total_income": 840000,
    "loan_amount": 2000000,
    "loan_term_months": 240,
    "loan_to_income_ratio": 2.38,
    "total_assets": 5300000,
    "approx_monthly_emi": 8333,
    "emi_share_of_income_pct": 11.9
  },
  "positive_factors": [
    "Excellent CIBIL score (780).",
    "Strong asset backing (loan is 38% of assets)."
  ],
  "risk_factors": []
}
```
