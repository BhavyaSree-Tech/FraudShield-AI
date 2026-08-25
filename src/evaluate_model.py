import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)

# ============================================================
# FRAUDSHIELD AI - FINAL MODEL EVALUATION
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

REPORT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "model_evaluation_report.csv"
)

print()
print("=" * 70)
print("FRAUDSHIELD AI - FINAL MODEL EVALUATION")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

print()
print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(
    f"Dataset loaded: {len(df):,} transactions"
)

# ------------------------------------------------------------
# 2. Create time features
# ------------------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df["hour"] = df["timestamp"].dt.hour

df["day_of_week"] = (
    df["timestamp"].dt.dayofweek
)

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# ------------------------------------------------------------
# 3. Features
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

y = df["is_fraud"]

# ------------------------------------------------------------
# 4. Same train/test split used during training
# ------------------------------------------------------------

print()
print("Creating held-out test set...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    f"Training transactions : {len(X_train):,}"
)

print(
    f"Held-out test set     : {len(X_test):,}"
)

print(
    f"Fraud cases in test   : {y_test.sum():,}"
)

# ------------------------------------------------------------
# 5. Load trained model
# ------------------------------------------------------------

print()
print("Loading trained model...")

model = joblib.load(
    MODEL_PATH
)

print(
    "Model loaded successfully. ✅"
)

# ------------------------------------------------------------
# 6. Generate probabilities
# ------------------------------------------------------------

print()
print("Generating fraud probabilities...")

probabilities = model.predict_proba(
    X_test
)[:, 1]

# ------------------------------------------------------------
# 7. Evaluate at 0.50 threshold
# ------------------------------------------------------------

threshold = 0.50

predictions = (
    probabilities >= threshold
).astype(int)

# ------------------------------------------------------------
# 8. Confusion matrix
# ------------------------------------------------------------

tn, fp, fn, tp = confusion_matrix(
    y_test,
    predictions
).ravel()

# ------------------------------------------------------------
# 9. Metrics
# ------------------------------------------------------------

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

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

# ------------------------------------------------------------
# 10. False-positive cost
# ------------------------------------------------------------

# Assumption for demonstration:
# Each false positive requires a manual review
# costing ₹100.

COST_PER_FALSE_POSITIVE = 100

false_positive_cost = (
    fp * COST_PER_FALSE_POSITIVE
)

# ------------------------------------------------------------
# 11. Display results
# ------------------------------------------------------------

print()
print("=" * 70)
print("HELD-OUT TEST SET PERFORMANCE")
print("=" * 70)

print()

print(
    f"Threshold             : {threshold:.2f}"
)

print(
    f"Accuracy              : {accuracy * 100:.2f}%"
)

print(
    f"Precision             : {precision * 100:.2f}%"
)

print(
    f"Recall                : {recall * 100:.2f}%"
)

print(
    f"F1 Score              : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC               : {roc_auc:.4f}"
)

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()

print(
    f"True Negatives        : {tn:,}"
)

print(
    f"False Positives       : {fp:,}"
)

print(
    f"False Negatives       : {fn:,}"
)

print(
    f"True Positives        : {tp:,}"
)

print()
print("=" * 70)
print("FALSE-POSITIVE COST")
print("=" * 70)

print()

print(
    f"False positives       : {fp:,}"
)

print(
    f"Cost per false positive: ₹{COST_PER_FALSE_POSITIVE}"
)

print(
    f"Estimated FP cost     : ₹{false_positive_cost:,}"
)

# ------------------------------------------------------------
# 12. Save report
# ------------------------------------------------------------

report = pd.DataFrame({
    "metric": [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
        "cost_per_false_positive",
        "estimated_false_positive_cost"
    ],

    "value": [
        threshold,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        tn,
        fp,
        fn,
        tp,
        COST_PER_FALSE_POSITIVE,
        false_positive_cost
    ]
})

report.to_csv(
    REPORT_PATH,
    index=False
)

# ------------------------------------------------------------
# 13. Final message
# ------------------------------------------------------------

print()
print("=" * 70)
print("FINAL EVALUATION COMPLETED SUCCESSFULLY. ✅")
print("=" * 70)

print()
print("Evaluation report saved at:")

print(
    REPORT_PATH
)

print()