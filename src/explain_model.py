import os
import joblib
import pandas as pd
import shap

# ============================================================
# FRAUDSHIELD AI - MODEL EXPLAINABILITY
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
    "feature_importance.csv"
)

print()
print("=" * 70)
print("FRAUDSHIELD AI - MODEL EXPLAINABILITY")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

print()
print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

# Create time features
df["timestamp"] = pd.to_datetime(df["timestamp"])

df["hour"] = df["timestamp"].dt.hour

df["day_of_week"] = df["timestamp"].dt.dayofweek

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

# ------------------------------------------------------------
# 2. Select features
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
# 3. Load trained pipeline
# ------------------------------------------------------------

print()
print("Loading trained model...")

pipeline = joblib.load(MODEL_PATH)

print("Model loaded successfully. ✅")

# ------------------------------------------------------------
# 4. Extract correct pipeline steps
# ------------------------------------------------------------

print()
print("Extracting model components...")

preprocessing = pipeline.named_steps["preprocessing"]

model = pipeline.named_steps["model"]

print("Preprocessing component found. ✅")
print("Random Forest model found. ✅")

# ------------------------------------------------------------
# 5. Transform data
# ------------------------------------------------------------

print()
print("Transforming data...")

X_transformed = preprocessing.transform(X)

# Get feature names after One-Hot Encoding
feature_names = (
    preprocessing.get_feature_names_out()
)

print(
    f"Total model features: {len(feature_names)}"
)

# ------------------------------------------------------------
# 6. Convert transformed data to DataFrame
# ------------------------------------------------------------

X_transformed = pd.DataFrame(
    X_transformed,
    columns=feature_names
)

# ------------------------------------------------------------
# 7. Take sample for SHAP
# ------------------------------------------------------------

sample_size = min(
    1000,
    len(X_transformed)
)

X_sample = X_transformed.sample(
    sample_size,
    random_state=42
)

print()
print(
    f"Calculating SHAP values for "
    f"{sample_size:,} transactions..."
)

# ------------------------------------------------------------
# 8. Create SHAP explainer
# ------------------------------------------------------------

explainer = shap.TreeExplainer(
    model
)

shap_values = explainer.shap_values(
    X_sample
)

# ------------------------------------------------------------
# 9. Handle SHAP output
# ------------------------------------------------------------

if isinstance(shap_values, list):

    values = shap_values[-1]

else:

    values = shap_values

    if len(values.shape) == 3:

        values = values[:, :, -1]

# ------------------------------------------------------------
# 10. Calculate feature importance
# ------------------------------------------------------------

importance = pd.DataFrame({
    "feature": X_sample.columns,
    "mean_abs_shap": abs(values).mean(axis=0)
})

importance = importance.sort_values(
    "mean_abs_shap",
    ascending=False
)

# ------------------------------------------------------------
# 11. Save feature importance
# ------------------------------------------------------------

importance.to_csv(
    OUTPUT_PATH,
    index=False
)

# ------------------------------------------------------------
# 12. Display top features
# ------------------------------------------------------------

print()
print("=" * 70)
print("TOP FRAUD RISK FEATURES")
print("=" * 70)

print()

for i, row in enumerate(
    importance.head(15).itertuples(),
    start=1
):

    print(
        f"{i:2}. "
        f"{row.feature:<50} "
        f"{row.mean_abs_shap:.6f}"
    )

# ------------------------------------------------------------
# 13. Completion
# ------------------------------------------------------------

print()
print("=" * 70)
print("EXPLAINABILITY COMPLETED SUCCESSFULLY. ✅")
print("=" * 70)

print()
print("Feature importance saved at:")

print(
    OUTPUT_PATH
)

print()