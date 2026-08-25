import os
import joblib
import pandas as pd
import numpy as np

# ============================================================
# FRAUDSHIELD AI - RISK SCORING
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "transactions.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "fraud_detection_model.joblib"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "risk_scored_transactions.csv"
)

print()
print("=" * 70)
print("FRAUDSHIELD AI - RISK SCORING")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

print()
print("Loading transactions...")

df = pd.read_csv(DATA_PATH)

print(
    f"Transactions loaded: {len(df):,}"
)

# ------------------------------------------------------------
# 2. Prepare timestamp features
# ------------------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df["hour"] = df["timestamp"].dt.hour

df["day_of_week"] = df["timestamp"].dt.dayofweek

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# ------------------------------------------------------------
# 3. Features used by the model
# ------------------------------------------------------------

features = [
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

X = df[features]

# ------------------------------------------------------------
# 4. Load trained model
# ------------------------------------------------------------

print()
print("Loading trained ML model...")

model = joblib.load(
    MODEL_PATH
)

print(
    "Model loaded successfully. ✅"
)

# ------------------------------------------------------------
# 5. Generate fraud probabilities
# ------------------------------------------------------------

print()
print("Calculating fraud risk...")

fraud_probability = model.predict_proba(
    X
)[:, 1]

# ------------------------------------------------------------
# 6. Convert probability to risk score
# ------------------------------------------------------------

df["fraud_probability"] = fraud_probability

df["risk_score"] = (
    fraud_probability * 100
).round(2)

# ------------------------------------------------------------
# 7. Risk categories
# ------------------------------------------------------------

def classify_risk(score):

    if score < 30:
        return "Low Risk"

    elif score < 70:
        return "Medium Risk"

    else:
        return "High Risk"


df["risk_level"] = df["risk_score"].apply(
    classify_risk
)

# ------------------------------------------------------------
# 8. Generate simple explanations
# ------------------------------------------------------------

def generate_reasons(row):

    reasons = []

    if row["amount"] > 10000:
        reasons.append(
            "High transaction amount"
        )

    if row["new_device"] == 1:
        reasons.append(
            "New device detected"
        )

    if row["transaction_frequency"] >= 8:
        reasons.append(
            "Unusually high transaction frequency"
        )

    if row["previous_fraud_count"] >= 1:
        reasons.append(
            "Previous fraud history"
        )

    if row["account_age_days"] < 30:
        reasons.append(
            "Very new account"
        )

    if row["location_distance_km"] > 200:
        reasons.append(
            "Unusual location distance"
        )

    if (
        row["amount"] > 10000
        and row["new_device"] == 1
    ):
        reasons.append(
            "High amount combined with new device"
        )

    if (
        row["transaction_frequency"] >= 10
        and row["location_distance_km"] > 200
    ):
        reasons.append(
            "High frequency combined with unusual location"
        )

    if len(reasons) == 0:
        reasons.append(
            "No major suspicious indicators detected"
        )

    return "; ".join(reasons)


df["risk_reasons"] = df.apply(
    generate_reasons,
    axis=1
)

# ------------------------------------------------------------
# 9. Recommended action
# ------------------------------------------------------------

def recommend_action(row):

    if row["risk_score"] >= 70:

        return "Block / Manual Review"

    elif row["risk_score"] >= 30:

        return "Additional Verification"

    else:

        return "Approve"


df["recommended_action"] = df.apply(
    recommend_action,
    axis=1
)

# ------------------------------------------------------------
# 10. Save risk-scored dataset
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ------------------------------------------------------------
# 11. Display summary
# ------------------------------------------------------------

print()
print("=" * 70)
print("RISK DISTRIBUTION")
print("=" * 70)

print(
    df["risk_level"].value_counts()
)

print()
print("=" * 70)
print("RECOMMENDED ACTIONS")
print("=" * 70)

print(
    df["recommended_action"].value_counts()
)

# ------------------------------------------------------------
# 12. Show high-risk examples
# ------------------------------------------------------------

high_risk = df[
    df["risk_level"] == "High Risk"
].sort_values(
    "risk_score",
    ascending=False
).head(10)

print()
print("=" * 70)
print("TOP 10 HIGH-RISK TRANSACTIONS")
print("=" * 70)

for _, row in high_risk.iterrows():

    print()
    print(
        f"Transaction : {row['transaction_id']}"
    )

    print(
        f"Amount      : ₹{row['amount']:,.2f}"
    )

    print(
        f"Risk Score  : {row['risk_score']}/100"
    )

    print(
        f"Risk Level  : {row['risk_level']}"
    )

    print(
        f"Action      : {row['recommended_action']}"
    )

    print(
        f"Reasons     : {row['risk_reasons']}"
    )

# ------------------------------------------------------------
# 13. Final message
# ------------------------------------------------------------

print()
print("=" * 70)
print("RISK SCORING COMPLETED SUCCESSFULLY. ✅")
print("=" * 70)

print()
print(
    "Risk-scored dataset saved at:"
)

print(
    OUTPUT_PATH
)

print()