"""
src/train.py
End-to-end data preprocessing, SMOTE resampling, and model training.
Trains Logistic Regression, Random Forest, and XGBoost.
Exports best model artifact and test splits for evaluation.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score

def get_paths():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    model_dir = os.path.join(base_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    return base_dir, data_path, model_dir

def preprocess_data(data_path):
    print("1. Loading raw customer dataset...")
    df = pd.read_csv(data_path)
    print(f"   Dataset shape: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Drop unique ID
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Clean TotalCharges (spaces to NaN, impute with median)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    median_val = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_val)

    # Encode binary target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Separate target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # One-hot encode categorical features
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    print(f"   Encoded feature matrix: {X_encoded.shape[1]} features")
    return X_encoded, y

def train_pipeline():
    base_dir, data_path, model_dir = get_paths()
    X, y = preprocess_data(data_path)

    # Stratified 80/20 split
    print("2. Performing stratified 80/20 train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"   Training rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    print(f"   Training churn distribution: {np.bincount(y_train)}")

    # Apply SMOTE to training set only (prevent leakage)
    print("3. Applying SMOTE resampling to training set...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"   Balanced training set: {np.bincount(y_train_res)}")

    # 1. Baseline: Logistic Regression
    print("\n4. Training Logistic Regression baseline...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_res, y_train_res)
    lr_preds = lr.predict(X_test)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    print(f"   LR ROC-AUC: {roc_auc_score(y_test, lr_probs):.4f}")

    # 2. Random Forest
    print("\n5. Training Random Forest classifier...")
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_res, y_train_res)
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    print(f"   RF ROC-AUC: {roc_auc_score(y_test, rf_probs):.4f}")

    # 3. XGBoost
    print("\n6. Training Tuned XGBoost Classifier...")
    xgb = XGBClassifier(
        n_estimators=180,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    xgb.fit(X_train_res, y_train_res)
    xgb_preds = xgb.predict(X_test)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    print(f"   XGBoost ROC-AUC: {xgb_auc:.4f}")

    print("\n--- Final XGBoost Test Performance ---")
    print(classification_report(y_test, xgb_preds, target_names=["No Churn", "Churn"]))

    # Save artifacts
    print("7. Exporting trained artifacts...")
    model_path = os.path.join(model_dir, "churn_model.pkl")
    features_path = os.path.join(model_dir, "feature_names.pkl")
    test_data_path = os.path.join(base_dir, "data", "test_data.pkl")

    joblib.dump(xgb, model_path)
    joblib.dump(list(X.columns), features_path)
    joblib.dump((X_test, y_test), test_data_path)

    print(f"   Artifact saved: {model_path}")
    print(f"   Features saved: {features_path}")
    print(f"   Test data saved: {test_data_path}")
    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    train_pipeline()
