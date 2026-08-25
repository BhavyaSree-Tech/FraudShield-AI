import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ============================================================
# FRAUDSHIELD AI
# Fraud Detection Model Training
# ============================================================

# ------------------------------------------------------------
# 1. Project paths
# ------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "transactions.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

# ------------------------------------------------------------
# 2. Load dataset
# ------------------------------------------------------------

print()
print("=" * 70)
print("FRAUDSHIELD AI - MODEL TRAINING")
print("=" * 70)

print()
print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(
    f"Dataset loaded: {len(df):,} transactions"
)

# ------------------------------------------------------------
# 3. Convert timestamp
# ------------------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

# Extract useful time-based features

df["hour"] = df["timestamp"].dt.hour

df["day_of_week"] = (
    df["timestamp"].dt.dayofweek
)

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# ------------------------------------------------------------
# 4. Select features
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

target = "is_fraud"

X = df[features]

y = df[target]

# ------------------------------------------------------------
# 5. Display class distribution
# ------------------------------------------------------------

print()
print("Class distribution:")
print(
    y.value_counts().to_string()
)

# ------------------------------------------------------------
# 6. Identify feature types
# ------------------------------------------------------------

categorical_features = [
    "payment_method",
    "device",
    "location"
]

numeric_features = [
    "amount",
    "account_age_days",
    "transaction_frequency",
    "previous_fraud_count",
    "new_device",
    "location_distance_km",
    "hour",
    "day_of_week",
    "is_weekend"
]

# ------------------------------------------------------------
# 7. Preprocessing
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# ------------------------------------------------------------
# 8. Create Random Forest model
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# ------------------------------------------------------------
# 9. Create complete ML pipeline
# ------------------------------------------------------------

pipeline = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)

# ------------------------------------------------------------
# 10. Train / Test split
# ------------------------------------------------------------

print()
print("Splitting dataset...")

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
    f"Testing transactions  : {len(X_test):,}"
)

print(
    f"Training fraud cases  : {y_train.sum():,}"
)

print(
    f"Testing fraud cases   : {y_test.sum():,}"
)

# ------------------------------------------------------------
# 11. Train model
# ------------------------------------------------------------

print()
print("Training Random Forest model...")
print("Please wait...")

pipeline.fit(
    X_train,
    y_train
)

print(
    "Model training completed. ✅"
)

# ------------------------------------------------------------
# 12. Predictions
# ------------------------------------------------------------

print()
print("Generating predictions...")

y_pred = pipeline.predict(
    X_test
)

y_probability = pipeline.predict_proba(
    X_test
)[:, 1]

# ------------------------------------------------------------
# 13. Evaluation metrics
# ------------------------------------------------------------

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

# ------------------------------------------------------------
# 14. Display results
# ------------------------------------------------------------

print()
print("=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

# ------------------------------------------------------------
# 15. Classification report
# ------------------------------------------------------------

print()
print("CLASSIFICATION REPORT")
print("-" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Fraud"
        ],
        zero_division=0
    )
)

# ------------------------------------------------------------
# 16. Confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print()
print("CONFUSION MATRIX")
print("-" * 70)

print(cm)

tn, fp, fn, tp = cm.ravel()

print()
print(f"True Negatives  : {tn:,}")
print(f"False Positives : {fp:,}")
print(f"False Negatives : {fn:,}")
print(f"True Positives  : {tp:,}")

# ------------------------------------------------------------
# 17. Save model
# ------------------------------------------------------------

model_path = os.path.join(
    MODEL_DIR,
    "fraud_detection_model.joblib"
)

joblib.dump(
    pipeline,
    model_path
)

print()
print("=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    f"Model path: {model_path}"
)

print()
print("STEP 4 COMPLETED SUCCESSFULLY. ✅")
print("=" * 70)