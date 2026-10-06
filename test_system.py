"""
test_system.py
==============
Comprehensive end-to-end test suite for the Loan Application Approval Prediction System.
"""

import json
from app import app


def run_tests():
    client = app.test_client()

    print("=" * 60)
    print("RUNNING END-TO-END TEST SUITE")
    print("=" * 60)

    # 1. Health check
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    health_data = res_health.get_json()
    print(f"1. Health Check: PASS (Model: {health_data['model_name']}, Features: {health_data['features_count']})")

    # 2. Frontend Home Page
    res_home = client.get("/")
    assert res_home.status_code == 200
    assert b"Loan Approval Predictor" in res_home.data
    assert b"Applicant Information" in res_home.data
    assert b"Financial Information" in res_home.data
    assert b"Loan Information" in res_home.data
    assert b"Credit Information" in res_home.data
    print("2. Frontend UI Route: PASS (All sections present)")

    # 3. Test Cases
    test_cases = [
        {
            "desc": "Prime Applicant (High Income, 810 CIBIL, Strong Assets)",
            "expected_decision": "APPROVED",
            "payload": {
                "gender": "Male",
                "married": "Yes",
                "dependents": 2,
                "education": "Graduate",
                "self_employed": "No",
                "applicant_income": 900000,
                "coapplicant_income": 300000,
                "loan_amount": 2500000,
                "loan_term_months": 240,
                "credit_history": 1,
                "cibil_score": 810,
                "property_area": "Urban",
                "residential_assets": 4000000,
                "commercial_assets": 1000000,
                "luxury_assets": 600000,
                "bank_asset_value": 800000
            }
        },
        {
            "desc": "High Risk Applicant (Low Income, 380 CIBIL, Adverse History)",
            "expected_decision": "REJECTED",
            "payload": {
                "gender": "Female",
                "married": "No",
                "dependents": 3,
                "education": "Not Graduate",
                "self_employed": "Yes",
                "applicant_income": 300000,
                "coapplicant_income": 0,
                "loan_amount": 3000000,
                "loan_term_months": 60,
                "credit_history": 0,
                "cibil_score": 380,
                "property_area": "Rural",
                "residential_assets": 100000,
                "commercial_assets": 0,
                "luxury_assets": 50000,
                "bank_asset_value": 20000
            }
        },
        {
            "desc": "Traditional Home Loan Applicant (Only credit_history = 1, no assets/CIBIL)",
            "expected_decision": "APPROVED",
            "payload": {
                "gender": "Male",
                "married": "Yes",
                "dependents": 1,
                "education": "Graduate",
                "self_employed": "No",
                "applicant_income": 600000,
                "coapplicant_income": 200000,
                "loan_amount": 1600000,
                "loan_term_months": 240,
                "credit_history": 1,
                "cibil_score": 750,
                "property_area": "Semiurban",
                "residential_assets": None,
                "commercial_assets": None,
                "luxury_assets": None,
                "bank_asset_value": None
            }
        }
    ]

    print("\n3. Testing API Prediction Scenarios:")
    for i, tc in enumerate(test_cases, 1):
        res = client.post("/predict", data=json.dumps(tc["payload"]), content_type="application/json")
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        print(f"   Case {i}: {tc['desc']}")
        print(f"           Result: {data['decision']} ({data['probability']}%) | Expected: {tc['expected_decision']}")
        assert data["decision"] == tc["expected_decision"]

    # 4. Input Validation Edge Cases (Section 25)
    print("\n4. Testing Validation Rules (Section 25):")
    invalid_cases = [
        ({"applicant_income": -5000, "loan_amount": 100000, "loan_term_months": 120}, "Negative income rejection"),
        ({"applicant_income": 50000, "loan_amount": -1000, "loan_term_months": 120}, "Negative loan amount rejection"),
        ({"applicant_income": 50000, "loan_amount": 100000, "loan_term_months": -12}, "Negative loan term rejection"),
        ({"applicant_income": 50000, "loan_amount": 100000, "loan_term_months": 120, "cibil_score": 950}, "CIBIL > 900 rejection"),
        ({"applicant_income": 50000, "loan_amount": 100000, "loan_term_months": 120, "dependents": -2}, "Negative dependents rejection")
    ]

    for payload, desc in invalid_cases:
        res = client.post("/predict", data=json.dumps(payload), content_type="application/json")
        assert res.status_code == 400
        data = res.get_json()
        assert data["success"] is False
        assert len(data["errors"]) > 0
        print(f"   [PASS] {desc}: Caught {len(data['errors'])} validation errors")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
