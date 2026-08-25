import os
import pandas as pd

# ============================================================
# FRAUDSHIELD AI
# Dataset Inspection & Validation
# ============================================================

# ------------------------------------------------------------
# 1. Locate the dataset
# ------------------------------------------------------------
project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

data_path = os.path.join(
    project_root,
    "data",
    "transactions.csv"
)

# ------------------------------------------------------------
# 2. Check whether dataset exists
# ------------------------------------------------------------
if not os.path.exists(data_path):
    print("ERROR: transactions.csv was not found.")
    print(f"Expected location: {data_path}")
    raise SystemExit

# ------------------------------------------------------------
# 3. Load dataset
# ------------------------------------------------------------
df = pd.read_csv(data_path)

print()
print("=" * 65)
print("FRAUDSHIELD AI - DATASET INSPECTION")
print("=" * 65)

# ------------------------------------------------------------
# 4. Dataset shape
# ------------------------------------------------------------
rows, columns = df.shape

print()
print("1. DATASET SIZE")
print("-" * 65)
print(f"Rows    : {rows:,}")
print(f"Columns : {columns}")

# ------------------------------------------------------------
# 5. Column names
# ------------------------------------------------------------
print()
print("2. COLUMNS")
print("-" * 65)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")

# ------------------------------------------------------------
# 6. First five records
# ------------------------------------------------------------
print()
print("3. FIRST 5 TRANSACTIONS")
print("-" * 65)

print(
    df.head(5).to_string(index=False)
)

# ------------------------------------------------------------
# 7. Data types
# ------------------------------------------------------------
print()
print("4. DATA TYPES")
print("-" * 65)

print(df.dtypes)

# ------------------------------------------------------------
# 8. Missing values
# ------------------------------------------------------------
print()
print("5. MISSING VALUES")
print("-" * 65)

missing_values = df.isnull().sum()

if missing_values.sum() == 0:
    print("No missing values found. ✅")
else:
    print(missing_values[missing_values > 0])

# ------------------------------------------------------------
# 9. Duplicate transactions
# ------------------------------------------------------------
print()
print("6. DUPLICATE TRANSACTIONS")
print("-" * 65)

duplicate_count = df["transaction_id"].duplicated().sum()

print(
    f"Duplicate transaction IDs: {duplicate_count}"
)

if duplicate_count == 0:
    print("No duplicate transaction IDs found. ✅")

# ------------------------------------------------------------
# 10. Fraud distribution
# ------------------------------------------------------------
print()
print("7. FRAUD DISTRIBUTION")
print("-" * 65)

fraud_counts = df["is_fraud"].value_counts()

normal_count = fraud_counts.get(0, 0)
fraud_count = fraud_counts.get(1, 0)

print(f"Normal transactions : {normal_count:,}")
print(f"Fraud transactions  : {fraud_count:,}")

fraud_rate = (
    fraud_count / len(df)
) * 100

print(f"Fraud rate          : {fraud_rate:.2f}%")

# ------------------------------------------------------------
# 11. Class imbalance warning
# ------------------------------------------------------------
print()
print("8. CLASS BALANCE")
print("-" * 65)

if fraud_rate < 10:
    print(
        "Fraud is much rarer than normal transactions."
    )
    print(
        "This is an imbalanced classification problem. ✅"
    )
else:
    print(
        "Fraud rate is relatively high."
    )

# ------------------------------------------------------------
# 12. Numerical statistics
# ------------------------------------------------------------
print()
print("9. NUMERICAL FEATURE SUMMARY")
print("-" * 65)

numeric_columns = [
    "amount",
    "account_age_days",
    "transaction_frequency",
    "previous_fraud_count",
    "new_device",
    "location_distance_km"
]

print(
    df[numeric_columns].describe().round(2).to_string()
)

# ------------------------------------------------------------
# 13. Payment method distribution
# ------------------------------------------------------------
print()
print("10. PAYMENT METHOD DISTRIBUTION")
print("-" * 65)

payment_distribution = (
    df["payment_method"]
    .value_counts()
)

print(payment_distribution)

# ------------------------------------------------------------
# 14. Fraud by payment method
# ------------------------------------------------------------
print()
print("11. FRAUD RATE BY PAYMENT METHOD")
print("-" * 65)

fraud_by_payment = (
    df.groupby("payment_method")["is_fraud"]
    .agg(
        transactions="count",
        fraud_count="sum",
        fraud_rate="mean"
    )
    .sort_values(
        "fraud_rate",
        ascending=False
    )
)

fraud_by_payment["fraud_rate"] = (
    fraud_by_payment["fraud_rate"] * 100
).round(2)

print(
    fraud_by_payment.to_string()
)

# ------------------------------------------------------------
# 15. Fraud by device
# ------------------------------------------------------------
print()
print("12. FRAUD RATE BY DEVICE")
print("-" * 65)

fraud_by_device = (
    df.groupby("device")["is_fraud"]
    .agg(
        transactions="count",
        fraud_count="sum",
        fraud_rate="mean"
    )
    .sort_values(
        "fraud_rate",
        ascending=False
    )
)

fraud_by_device["fraud_rate"] = (
    fraud_by_device["fraud_rate"] * 100
).round(2)

print(
    fraud_by_device.to_string()
)

# ------------------------------------------------------------
# 16. Fraud by new-device status
# ------------------------------------------------