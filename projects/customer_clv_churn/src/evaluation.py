import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import pandas as pd
import numpy as np
import os

def evaluate_model(model, X_test, y_test, model_name="model"):
    # Predict classes and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Print classification report and ROC-AUC score
    print(f"Classification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc:.4f}")

    # Prepare output directories if not exist
    reports_dir = "outputs/reports"
    figures_dir = "outputs/figures"
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Save classification report as CSV
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    report_path = os.path.join(reports_dir, f"{model_name}_classification_report.csv")
    df_report.to_csv(report_path)

    # Plot and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    cm_path = os.path.join(figures_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # Plot and save ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"{model_name} - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    roc_path = os.path.join(figures_dir, f"{model_name}_roc_curve.png")
    plt.savefig(roc_path)
    plt.close()


def plot_feature_importance(model, feature_names, model_name="model"):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-20:]  # Top 20 features
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.title(f"Top Feature Importances - {model_name}")
        plt.tight_layout()

        figures_dir = "outputs/figures"
        os.makedirs(figures_dir, exist_ok=True)
        fi_path = os.path.join(figures_dir, f"{model_name}_feature_importance.png")
        plt.savefig(fi_path)
        plt.close()
