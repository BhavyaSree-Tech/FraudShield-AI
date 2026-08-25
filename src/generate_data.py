import os
import numpy as np
import pandas as pd

# ============================================================
# FRAUDSHIELD AI
# Improved Synthetic Transaction Dataset Generator
# ============================================================

np.random.seed(42)

# ------------------------------------------------------------
# 1. Dataset size
# ------------------------------------------------------------
N = 50000

# ------------------------------------------------------------
# 2. Basic transaction information
# ------------------------------------------------------------
transaction_ids = [
    f"TXN{i:06d}" for i in range(1, N + 1)
]

merchant_ids = np.random.choice(
    [f"M{i:04d}" for i in range(1, 501)],
    size=N
)

customer_ids = np.random.choice(
    [f"C{i:06d}" for i in range(1, 10001)],
    size=N
)

timestamps = pd.date_range(
    start="2026-01-01",
    periods=N,
    freq="10min"
)

# ------------------------------------------------------------
# 3. Transaction amount
# ------------------------------------------------------------
amounts = np.random.lognormal(
    mean=6.0,
    sigma=1.0,
    size=N
)

amounts = np.clip(
    np.round(amounts, 2),
    10,
    200000
)

# ------------------------------------------------------------
# 4. Payment method
# ------------------------------------------------------------
payment_methods = np.random.choice(
    [
        "UPI",
        "Credit Card",
        "Debit Card",
        "Net Banking",
        "Wallet"
    ],
    size=N,
    p=[0.45, 0.25, 0.15, 0.10, 0.05]
)

# ------------------------------------------------------------
# 5. Device
# ------------------------------------------------------------
devices = np.random.choice(
    [
        "Mobile",
        "Desktop",
        "Tablet"
    ],
    size=N,
    p=[0.70, 0.20, 0.10]
)

# ------------------------------------------------------------
# 6. Location
# ------------------------------------------------------------
locations = np.random.choice(
    [
        "Bangalore",
        "Hyderabad",
        "Mumbai",
        "Delhi",
        "Chennai",
        "Pune",
        "Kolkata",
        "Other"
    ],
    size=N
)

# ------------------------------------------------------------
# 7. Account age
# ------------------------------------------------------------
account_age_days = np.random.randint(
    1,
    2500,
    size=N
)

# ------------------------------------------------------------
# 8. Transaction frequency
# ------------------------------------------------------------
transaction_frequency = (
    np.random.poisson(
        lam=3,
        size=N
    ) + 1
)

# Create a small group of unusually high-frequency transactions

high_frequency_mask = (
    np.random.rand(N) < 0.04
)

transaction_frequency[high_frequency_mask] = (
    np.random.randint(
        10,
        30,
        size=high_frequency_mask.sum()
    )
)

# ------------------------------------------------------------
# 9. Previous fraud count
# ------------------------------------------------------------
previous_fraud_count = np.random.poisson(
    lam=0.12,
    size=N
)

# Small group with previous suspicious history

history_mask = (
    np.random.rand(N) < 0.03
)

previous_fraud_count[history_mask] = (
    np.random.randint(
        1,
        4,
        size=history_mask.sum()
    )
)

# ------------------------------------------------------------
# 10. New device
# ------------------------------------------------------------
new_device = np.random.choice(
    [0, 1],
    size=N,
    p=[0.82, 0.18]
)

# ------------------------------------------------------------
# 11. Location distance
# ------------------------------------------------------------
location_distance_km = np.random.exponential(
    scale=50,
    size=N
)

location_distance_km = np.clip(
    np.round(
        location_distance_km,
        2
    ),
    0,
    1000
)

# ------------------------------------------------------------
# 12. Create stronger fraud-risk patterns
# ------------------------------------------------------------

risk_score = np.zeros(N)

# High transaction amount
risk_score += (
    amounts > 10000
) * 1.0

risk_score += (
    amounts > 50000
) * 1.5

# New device
risk_score += (
    new_device == 1
) * 1.2

# High transaction frequency
risk_score += (
    transaction_frequency >= 8
) * 1.4

risk_score += (
    transaction_frequency >= 15
) * 1.2

# Previous fraud history
risk_score += (
    previous_fraud_count >= 1
) * 1.2

risk_score += (
    previous_fraud_count >= 2
) * 1.0

# Very young account
risk_score += (
    account_age_days < 30
) * 1.3

risk_score += (
    account_age_days < 7
) * 0.8

# Unusual location
risk_score += (
    location_distance_km > 200
) * 1.2

risk_score += (
    location_distance_km > 500
) * 1.0

# Risky combinations
risk_score += (
    (new_device == 1)
    & (amounts > 10000)
) * 1.5

risk_score += (
    (transaction_frequency >= 10)
    & (location_distance_km > 200)
) * 1.5

risk_score += (
    (account_age_days < 30)
    & (new_device == 1)
) * 1.4

# ------------------------------------------------------------
# 13. Add random uncertainty
# ------------------------------------------------------------

risk_score += np.random.normal(
    loc=0,
    scale=0.8,
    size=N
)

# ------------------------------------------------------------
# 14. Convert risk score into fraud probability
# ------------------------------------------------------------

fraud_probability = (
    1 /
    (
        1 +
        np.exp(
            -(risk_score - 4.8)
        )
    )
)

# ------------------------------------------------------------
# 15. Generate fraud labels
# ------------------------------------------------------------

is_fraud = np.random.binomial(
    1,
    fraud_probability
)

# ------------------------------------------------------------
# 16. Create DataFrame
# ------------------------------------------------------------

df = pd.DataFrame({

    "transaction_id": transaction_ids,

    "merchant_id": merchant_ids,

    "customer_id": customer_ids,

    "timestamp": timestamps,

    "amount": amounts,

    "payment_method": payment_methods,

    "device": devices,

    "location": locations,

    "account_age_days": account_age_days,

    "transaction_frequency": transaction_frequency,

    "previous_fraud_count": previous_fraud_count,

    "new_device": new_device,

    "location_distance_km": location_distance_km,

    "is_fraud": is_fraud
})

# ------------------------------------------------------------
# 17. Add fraud spike periods
# ------------------------------------------------------------

spike_periods = [

    (
        "2026-01-15 10:00:00",
        "2026-01-15 20:00:00"
    ),

    (
        "2026-02-10 08:00:00",
        "2026-02-10 18:00:00"
    ),

    (
        "2026-03-05 12:00:00",
        "2026-03-05 22:00:00"
    )
]

for start, end in spike_periods:

    mask = (
        (df["timestamp"] >= start)
        &
        (df["timestamp"] <= end)
    )

    spike_indices = df.index[mask]

    if len(spike_indices) > 0:

        # Increase fraud probability during spike periods
        random_values = np.random.rand(
            len(spike_indices)
        )

        fraud_indices = spike_indices[
            random_values < 0.35
        ]

        df.loc[
            fraud_indices,
            "is_fraud"
        ] = 1

# ------------------------------------------------------------
# 18. Sort transactions chronologically
# ------------------------------------------------------------

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

# ------------------------------------------------------------
# 19. Create correct data directory
# ------------------------------------------------------------

project_root = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

data_directory = os.path.join(
    project_root,
    "data"
)

os.makedirs(
    data_directory,
    exist_ok=True
)

# ------------------------------------------------------------
# 20. Save dataset
# ------------------------------------------------------------

output_path = os.path.join(
    data_directory,
    "transactions.csv"
)

df.to_csv(
    output_path,
    index=False
)

# ------------------------------------------------------------
# 21. Display results
# ------------------------------------------------------------

print()

print("=" * 65)

print(
    "FRAUDSHIELD AI - IMPROVED DATASET GENERATED"
)

print("=" * 65)

print()

print(
    f"Total transactions : {len(df):,}"
)

print(
    f"Fraud transactions  : "
    f"{df['is_fraud'].sum():,}"
)

print(
    f"Normal transactions : "
    f"{(df['is_fraud'] == 0).sum():,}"
)

print(
    f"Fraud rate          : "
    f"{df['is_fraud'].mean() * 100:.2f}%"
)

print()

print(
    "Dataset saved at:"
)

print(
    output_path
)

print()

print("=" * 65)