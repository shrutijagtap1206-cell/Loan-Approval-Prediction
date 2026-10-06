"""
train_model.py
==============
Trains, compares, evaluates, and exports the Loan Approval Prediction model.
Compares:
  - Logistic Regression (Baseline, highly interpretable)
  - Random Forest (Non-linear, robust to outliers)
  - Gradient Boosting (High predictive power)

Evaluates on:
  - Accuracy, Precision, Recall, F1-Score, ROC-AUC, 5-fold CV ROC-AUC, Confusion Matrix

Saves the best Pipeline (Preprocessing + Model) to loan_model.joblib.
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def train_and_evaluate():
    print("=" * 60)
    print("LOAN APPROVAL MODEL TRAINING & COMPARISON")
    print("=" * 60)

    # 1. Load data
    df = pd.read_csv("clean_loan_approval.csv")
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    X = df.drop(columns=["target"])
    y = df["target"]

    cat_cols = ["gender", "married", "education", "self_employed", "property_area"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    print(f"\nCategorical features ({len(cat_cols)}): {cat_cols}")
    print(f"Numerical features ({len(num_cols)}): {num_cols}")

    # 2. Train / Test Split (80 / 20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

    # 3. Preprocessing Definition
    # Median imputation with missing indicator + Standard Scaling for numerics
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler())
    ])

    # Constant 'Missing' imputation + One-Hot Encoding for categoricals
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    # 4. Candidate Models
    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
    }

    results = {}
    fitted_pipelines = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    for name, model in candidate_models.items():
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        # Cross-validation
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

        # Fit on training data
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        # Test evaluation
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": roc,
            "CV ROC-AUC (mean)": cv_scores.mean(),
            "CV ROC-AUC (std)": cv_scores.std(),
            "Confusion Matrix": cm
        }

        print(f"\n--- {name} ---")
        print(f"  Accuracy:           {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Precision:          {prec:.4f}")
        print(f"  Recall:             {rec:.4f}")
        print(f"  F1-Score:           {f1:.4f}")
        print(f"  ROC-AUC:            {roc:.4f}")
        print(f"  5-Fold CV ROC-AUC:  {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print(f"  Confusion Matrix (TN, FP / FN, TP):")
        print(f"    [[{cm[0,0]:3d}, {cm[0,1]:3d}],")
        print(f"     [{cm[1,0]:3d}, {cm[1,1]:3d}]]")

    # 5. Select Best Model by ROC-AUC and F1-Score
    best_model_name = max(results.keys(), key=lambda k: results[k]["ROC-AUC"])
    best_pipe = fitted_pipelines[best_model_name]
    best_metrics = results[best_model_name]

    print("\n" + "=" * 60)
    print(f"BEST MODEL SELECTED: {best_model_name}")
    print(f"ROC-AUC: {best_metrics['ROC-AUC']:.4f} | F1-Score: {best_metrics['F1-Score']:.4f}")
    print("=" * 60)

    # 6. Save Model Bundle (Pipeline + Metadata)
    # Bundle contains the entire self-contained pipeline (preprocessor + model)
    metadata = {
        "pipeline": best_pipe,
        "model_name": best_model_name,
        "cat_features": cat_cols,
        "num_features": num_cols,
        "feature_order": list(X.columns),
        "metrics": {
            k: (v.tolist() if isinstance(v, np.ndarray) else float(v))
            for k, v in best_metrics.items()
        }
    }

    joblib.dump(metadata, "loan_model.joblib")
    print(f"\n[OK] Successfully saved best model bundle to: loan_model.joblib")

    return results, best_model_name


if __name__ == "__main__":
    train_and_evaluate()
