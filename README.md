🛡️ FraudShield AI

Explainable AI-Powered Fraud Detection & Risk Management System

FraudShield AI is an end-to-end machine learning system designed to detect suspicious financial transactions, calculate transaction risk, explain the reasons behind fraud predictions, and safely handle uncertain or incomplete transaction data.

The project combines Machine Learning, Risk Scoring, SHAP Explainability, Failure Handling, and an Interactive Streamlit Dashboard into one practical fraud detection solution.

---

🎯 Problem Statement

Digital payment systems process a large number of transactions every day. Detecting fraudulent transactions is challenging because:

- Fraudulent transactions are much rarer than normal transactions.
- Fraud patterns can change over time.
- Incorrectly blocking genuine transactions creates a poor customer experience.
- Fraud decisions should be explainable instead of being a black-box prediction.
- Missing or invalid transaction information should not result in unsafe automatic decisions.

FraudShield AI addresses these challenges using a combination of machine learning and explainable risk management.

---

💡 Solution

FraudShield AI follows this workflow:

Transaction Data
       ↓
Data Validation & Inspection
       ↓
Feature Engineering
       ↓
Machine Learning Model
       ↓
Fraud Probability
       ↓
Risk Scoring
       ↓
Risk Level
       ↓
Recommended Action
       ↓
SHAP Explanation
       ↓
Dashboard / Investigation

The system produces three major risk categories:

Risk Level| Recommended Action
Low Risk| Approve
Medium Risk| Additional Verification
High Risk| Block / Manual Review

---

🤖 Machine Learning

The project uses a Random Forest Classifier with class balancing to handle the imbalanced fraud classification problem.

Dataset

- Total transactions: 50,000
- Normal transactions: 48,373
- Fraud transactions: 1,627
- Fraud rate: 3.25%

Important Features

The model uses transaction and behavioral information such as:

- Transaction amount
- Transaction frequency
- Previous fraud count
- Account age
- New device indicator
- Location distance
- Payment method
- Device
- Location
- Transaction hour
- Day of week

---

📊 Model Performance

The model was evaluated on a held-out test set of 10,000 transactions.

Metric| Result
Accuracy| 88.45%
Precision| 11.79%
Recall| 39.38%
F1 Score| 18.14%
ROC-AUC| 0.7317

Confusion Matrix

| Predicted Normal| Predicted Fraud
Actual Normal| 8,717| 958
Actual Fraud| 197| 128

The evaluation highlights the trade-off between detecting fraudulent transactions and avoiding unnecessary false positives.

---

🧠 Explainable AI with SHAP

FraudShield AI uses SHAP (SHapley Additive exPlanations) to understand which features contribute most to the model's fraud predictions.

Top risk-related features identified by SHAP include:

1. New Device
2. Transaction Frequency
3. Previous Fraud Count
4. Account Age
5. Location Distance
6. Transaction Amount
7. Transaction Hour
8. Day of Week

This makes the system more transparent and helps users understand why a transaction was considered risky.

---

🚦 Risk Scoring

Each transaction receives a risk score from 0 to 100.

The system converts the model prediction into practical risk categories:

0 ───────────── 40 ───────────── 70 ───────────── 100
     Low Risk          Medium Risk        High Risk

High-risk transactions can be sent for blocking or manual review, while medium-risk transactions can undergo additional verification.

---

🛡️ Safe Failure Handling

FraudShield AI does not make an automatic decision when required transaction information is missing.

Example:

Required field missing
        ↓
Prediction stopped
        ↓
Transaction placed on hold
        ↓
Manual review

This prevents the system from making unsafe decisions based on incomplete data.

---

🖥️ Interactive Dashboard

The project includes a Streamlit dashboard providing:

- Transaction overview
- Risk distribution
- Recommended actions
- Model performance
- Confusion matrix results
- False-positive cost analysis
- SHAP feature importance
- High-risk transaction table
- Transaction investigation
- Risk score visualization
- Safe failure audit information

---

🛠️ Technology Stack

Programming

- Python

Machine Learning

- Scikit-learn
- Random Forest
- SHAP

Data Processing

- Pandas
- NumPy

Visualization

- Streamlit

Model Storage

- Joblib

Development

- VS Code
- Git
- GitHub

---

📁 Project Structure

FraudShield-AI/
│
├── data/
│   ├── transactions.csv
│   ├── risk_scored_transactions.csv
│   ├── feature_importance.csv
│   ├── model_evaluation_report.csv
│   └── failure_audit_log.csv
│
├── models/
│   └── fraud_detection_model.joblib
│
├── src/
│   ├── generate_data.py
│   ├── inspect_data.py
│   ├── train_model.py
│   ├── threshold_test.py
│   ├── risk_scoring.py
│   ├── explain_model.py
│   ├── evaluate_model.py
│   └── failure_handler.py
│
├── dashboard/
│   └── app.py
│
├── notebooks/
│
├── .gitignore
└── README.md

---

▶️ How to Run

1. Clone the repository

git clone https://github.com/BhavyaSree-Tech/FraudShield-AI.git
cd FraudShield-AI

2. Create a virtual environment

python -m venv venv

3. Activate the environment

Windows:

venv\Scripts\activate

4. Install dependencies

pip install pandas numpy scikit-learn joblib shap streamlit

5. Run the Streamlit dashboard

cd dashboard
streamlit run app.py

The dashboard will open in your browser.

---

🔍 Example Investigation

A transaction such as:

Transaction ID: TXN048089
Risk Score: 96.87 / 100
Risk Level: High Risk
Action: Block / Manual Review

can be investigated through the dashboard.

The system can explain the risk using signals such as:

- New device detected
- High transaction frequency
- Previous fraud history
- Other behavioral or transaction-level signals

---

💰 False-Positive Cost Analysis

The project includes an estimated false-positive cost analysis to demonstrate the business impact of incorrectly flagging legitimate transactions.

Current project assumption:

False Positives: 958
Cost per False Positive: ₹100

Estimated Impact: ₹95,800

This cost is a project assumption used for demonstration and can be replaced with a real business-specific cost model.

---

🚀 Future Improvements

Future versions of FraudShield AI can include:

- Real-time transaction streaming
- Advanced anomaly detection
- XGBoost / LightGBM comparison
- Model calibration
- Real-time API deployment
- Cloud deployment
- Continuous model monitoring
- Concept-drift detection
- Customer-level behavioral profiling
- Automated feedback from confirmed fraud cases
- Production-grade authentication and audit logging

---

🏆 Project Goal

FraudShield AI demonstrates how machine learning + explainable AI + risk management + safe failure handling can be combined to create a practical fraud detection system.

The goal is not only to predict fraud, but also to provide:

«A risk-aware, explainable, and safer decision-support system for digital transactions.»

---

👩‍💻 Author

BhavyaSree-Tech

GitHub:
https://github.com/BhavyaSree-Tech

---

📌 Project Status

Completed Prototype — Ready for Demonstration

Core components implemented:

- ✅ Dataset generation
- ✅ Data inspection
- ✅ Fraud detection model
- ✅ Threshold analysis
- ✅ Risk scoring
- ✅ SHAP explainability
- ✅ Model evaluation
- ✅ Failure handling
- ✅ Streamlit dashboard
- ✅ GitHub repository
