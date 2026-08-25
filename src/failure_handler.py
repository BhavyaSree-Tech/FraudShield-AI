import os
import joblib
import pandas as pd

# ============================================================
# FRAUDSHIELD AI - FAILURE HANDLING
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "fraud_detection_model.joblib"
)

AUDIT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "failure_audit_log.csv"
)

print()
print("=" * 70)
print("FRAUDSHIELD AI - FAILURE HANDLING TEST")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load model
# ------------------------------------------------------------

print()
print("Loading trained model...")

pipeline = joblib.load(MODEL_PATH)

print("Model loaded successfully. ✅")

# ------------------------------------------------------------
# 2. Create sample transactions
# ------------------------------------------------------------

normal_transaction = {
    "amount": 750.00,
    "payment_method": "UPI",
    "device": "Mobile",
    "location": "Mumbai",
    "account_age_days": 500,
    "transaction_frequency": 3,
    "previous_fraud_count": 0,
    "new_device": 0,
    "location_distance_km": 10.0,
    "hour": 14,
    "day_of_week": 1,
    "is_weekend": 0
}

# ------------------------------------------------------------
# 3. Failure case
# ------------------------------------------------------------

failure_transaction = {
    "amount": None,
    "payment_method": "UPI",
    "device": "Mobile",
    "location": "Mumbai",
    "account_age_days": 500,
    "transaction_frequency": 3,
    "previous_fraud_count": 0,
    "new_device": 0,
    "location_distance_km": 10.0,
    "hour": 14,
    "day_of_week": 1,
    "is_weekend": 0
}

# ------------------------------------------------------------
# 4. Safe prediction function
# ------------------------------------------------------------

def safe_predict(transaction):

    try:

        # Convert transaction into DataFrame
        input_data = pd.DataFrame(
            [transaction]
        )

        # Validate required fields
        required_fields = [
            "amount",
            "payment_method",
            "device",
            "location",
            "account_age_days",
            "transaction_frequency",
            "previous_fraud_count",
            "new_device",
            "location_distance_km",
            "hour",
            "day_of_week",
            "is_weekend"
        ]

        missing_fields = [
            field
            for field in required_fields
            if field not in input_data.columns
            or pd.isna(input_data.iloc[0][field])
        ]

        # If required data is missing,
        # DO NOT make an automatic fraud decision.
        if missing_fields:

            return {
                "status": "MANUAL_REVIEW",
                "risk_score": None,
                "action": "Hold transaction",
                "reason": (
                    "Required transaction data is missing: "
                    + ", ".join(missing_fields)
                )
            }

        # Generate probability
        probability = pipeline.predict_proba(
            input_data
        )[0][1]

        risk_score = round(
            probability * 100,
            2
        )

        # Risk decision
        if risk_score >= 70:

            action = "Block / Manual Review"

        elif risk_score >= 30:

            action = "Additional Verification"

        else:

            action = "Approve"

        return {
            "status": "SUCCESS",
            "risk_score": risk_score,
            "action": action,
            "reason": "Prediction completed successfully"
        }

    except Exception as error:

        # Graceful fallback
        return {
            "status": "MANUAL_REVIEW",
            "risk_score": None,
            "action": "Hold transaction",
            "reason": (
                "Prediction failed safely: "
                + str(error)
            )
        }


# ------------------------------------------------------------
# 5. Test normal transaction
# ------------------------------------------------------------

print()
print("=" * 70)
print("TEST 1 - NORMAL TRANSACTION")
print("=" * 70)

result_1 = safe_predict(
    normal_transaction
)

print()
print(
    "Status      :",
    result_1["status"]
)

print(
    "Risk Score  :",
    result_1["risk_score"]
)

print(
    "Action      :",
    result_1["action"]
)

print(
    "Reason      :",
    result_1["reason"]
)

# ------------------------------------------------------------
# 6. Test failure transaction
# ------------------------------------------------------------

print()
print("=" * 70)
print("TEST 2 - FAILURE CASE")
print("=" * 70)

result_2 = safe_predict(
    failure_transaction
)

print()
print(
    "Status      :",
    result_2["status"]
)

print(
    "Risk Score  :",
    result_2["risk_score"]
)

print(
    "Action      :",
    result_2["action"]
)

print(
    "Reason      :",
    result_2["reason"]
)

# ------------------------------------------------------------
# 7. Audit trail
# ------------------------------------------------------------

audit_record = pd.DataFrame([
    {
        "test_case": "Normal transaction",
        "status": result_1["status"],
        "risk_score": result_1["risk_score"],
        "action": result_1["action"],
        "reason": result_1["reason"]
    },
    {
        "test_case": "Missing amount failure",
        "status": result_2["status"],
        "risk_score": result_2["risk_score"],
        "action": result_2["action"],
        "reason": result_2["reason"]
    }
])

audit_record.to_csv(
    AUDIT_PATH,
    index=False
)

# ------------------------------------------------------------
# 8. Final result
# ------------------------------------------------------------

print()
print("=" * 70)
print("FAILURE HANDLING COMPLETED SUCCESSFULLY. ✅")
print("=" * 70)

print()
print("Audit log saved at:")

print(AUDIT_PATH)

print()