# Customer Churn Prediction Engine

An end-to-end machine learning pipeline predicting subscriber attrition on telecom customer records (7,043 rows) using XGBoost, handling class imbalance with SMOTE, and explaining predictions with SHAP values.

## Problem Statement

Customer churn costs subscription businesses millions of dollars annually. Identifying customers likely to cancel *before* they leave allows retention teams to deploy targeted offers. However, churn datasets suffer from significant class imbalance (~73% non-churn vs ~27% churn), where standard accuracy metrics can be deceiving.

## Tech Stack

- **Python 3.8+**
- **Scikit-learn** for preprocessing, pipelines, and baseline models
- **Imbalanced-learn** for SMOTE (Synthetic Minority Over-sampling Technique)
- **XGBoost** for gradient-boosted decision trees
- **SHAP** for tree-based feature explainability
- **Pandas & NumPy** for data manipulation
- **Matplotlib & Seaborn** for evaluation charts

## Pipeline Architecture

```
Raw Telco Data (7,043 rows)
        │
        ▼
Data Preprocessing (TotalCharges cleaning, One-Hot Encoding)
        │
        ▼
Train-Test Split (80/20 Stratified)
        │
        ▼
SMOTE Resampling (Balances training minority class without data leakage)
        │
        ▼
Model Training & Comparison (Logistic Regression vs Random Forest vs XGBoost)
        │
        ▼
Model Explainability (SHAP summary plot & feature attribution)
        │
        ▼
Export Model Artifact (model/churn_model.pkl for dashboard serving)
```

## Key Results

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 74.8% | 0.51 | **0.80** | 0.62 | 0.835 |
| Random Forest | 78.6% | 0.60 | 0.52 | 0.56 | 0.824 |
| **XGBoost (Optimized)** | **80.2%** | **0.64** | **0.62** | **0.63** | **0.844** |

- **Best Model:** Tuned XGBoost achieved **0.844 ROC-AUC**.
- **Top 3 Churn Drivers (via SHAP):**
  1. `Contract Type` (Month-to-month contracts have 3.8x higher churn risk)
  2. `Tenure` (Higher tenure strongly protects against churn)
  3. `TotalCharges` & `MonthlyCharges` (Higher pricing without bundled perks drives disengagement)

## Project Structure

```
customer-churn-prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── model/
│   ├── churn_model.pkl          # Trained XGBoost model artifact
│   └── feature_names.pkl        # Encoded feature column list
├── notebooks/
│   └── churn_modeling.ipynb     # Interactive exploration & modeling
├── src/
│   ├── train.py                 # Preprocessing, SMOTE, and model training
│   └── evaluate.py              # Metrics, ROC curve, and SHAP visualizations
├── outputs/
│   └── charts/                  # Confusion matrix, ROC curve, SHAP summary
├── requirements.txt
└── README.md
```

## Setup & Running

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model pipeline
python src/train.py

# 3. Generate evaluation metrics and SHAP charts
python src/evaluate.py
```

## License

This project is for educational and portfolio demonstration. The Telco Customer Churn dataset is available under open Kaggle license.
