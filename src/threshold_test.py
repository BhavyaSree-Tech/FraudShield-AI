import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================
# FRAUDSHIELD AI - THRESHOLD TESTING
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

print()
print("=" * 65)
print("FRAUDSHIELD AI - THRESHOLD TESTING")
print("=" * 65)

# Load data
df = pd.read_csv(DATA_PATH)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

df["hour"] = df["timestamp"].dt.hour

df["day_of_week"] = df["timestamp"].dt.dayofweek

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# Same features used during training
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
y = df["is_fraud"]

# Same test split as training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Load trained model
model = joblib.load(MODEL_PATH)

# Get fraud probabilities
probabilities = model.predict_proba(X_test)[:, 1]

print()
print("Testing different thresholds...")
print()

print(
    f"{'Threshold':<12}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

print("-" * 60)

# Test different thresholds
thresholds = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]

results = []

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append(
        (
            threshold,
            accuracy,
            precision,
            recall,
            f1
        )
    )

    print(
        f"{threshold:<12.2f}"
        f"{accuracy * 100:<12.2f}"
        f"{precision * 100:<12.2f}"
        f"{recall * 100:<12.2f}"
        f"{f1 * 100:<12.2f}"
    )

# Find threshold closest to 94% accuracy
best = min(
    results,
    key=lambda x: abs(x[1] - 0.94)
)

print()
print("=" * 65)
print("CLOSEST TO 94% ACCURACY")
print("=" * 65)

print(
    f"Threshold : {best[0]:.2f}"
)

print(
    f"Accuracy  : {best[1] * 100:.2f}%"
)

print(
    f"Precision : {best[2] * 100:.2f}%"
)

print(
    f"Recall    : {best[3] * 100:.2f}%"
)

print(
    f"F1 Score  : {best[4] * 100:.2f}%"
)

print()
print("THRESHOLD TEST COMPLETED. ✅")
print("=" * 65)