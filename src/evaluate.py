"""
src/evaluate.py
Loads trained XGBoost model and test data.
Computes ROC curve, confusion matrix, and SHAP explainability visualizations.
Saves all diagnostic plots to outputs/charts/.
"""

import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, auc

sns.set_theme(style="whitegrid")

def ensure_charts_dir():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "outputs", "charts")
    os.makedirs(out_dir, exist_ok=True)
    return base_dir, out_dir

def evaluate():
    base_dir, out_dir = ensure_charts_dir()
    model_path = os.path.join(base_dir, "model", "churn_model.pkl")
    test_data_path = os.path.join(base_dir, "data", "test_data.pkl")

    if not os.path.exists(model_path) or not os.path.exists(test_data_path):
        raise FileNotFoundError("Model or test data not found. Run train.py first.")

    print("1. Loading model and test splits...")
    model = joblib.load(model_path)
    X_test, y_test = joblib.load(test_data_path)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 1. Confusion Matrix
    print("2. Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Retained", "Churned"],
                yticklabels=["Retained", "Churned"])
    plt.title("XGBoost Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"   Saved: {cm_path}")

    # 2. ROC Curve
    print("3. Generating ROC Curve...")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(7, 5.5))
    plt.plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"XGBoost (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.7, label="Random Guess (AUC = 0.500)")
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate (Recall)", fontsize=11)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    roc_path = os.path.join(out_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print(f"   Saved: {roc_path}")

    # 3. SHAP Explainability
    print("4. Computing SHAP Explainability metrics (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # SHAP Summary Beeswarm Plot
    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X_test, show=False, max_display=12)
    plt.title("SHAP Feature Importance (Impact on Churn Prediction)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    shap_path = os.path.join(out_dir, "shap_summary.png")
    plt.savefig(shap_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {shap_path}")

    # SHAP Global Bar Importance
    plt.figure(figsize=(9, 5.5))
    shap.plots.bar(shap_values, max_display=10, show=False)
    plt.title("Mean |SHAP Value| (Global Feature Importance)", fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    shap_bar_path = os.path.join(out_dir, "shap_bar_importance.png")
    plt.savefig(shap_bar_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {shap_bar_path}")

    print("All diagnostic evaluation charts successfully created!")

if __name__ == "__main__":
    evaluate()
