
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def preprocess_for_rf(X_raw, train_columns):
    """
    Preprocess test data for Random Forest:
    - One-hot encode categorical columns
    - Align columns to training columns (add missing cols with zeros)
    """
    X = pd.get_dummies(X_raw)
    # Reindex to train columns, fill missing columns with 0
    X = X.reindex(columns=train_columns, fill_value=0)
    return X




def preprocess_for_logreg(X_raw, train_columns, scaler):
    """
    Preprocess test data for Logistic Regression:
    - One-hot encode categorical columns
    - Align columns to training columns
    - Scale using pre-fitted scaler
    """
    X = pd.get_dummies(X_raw)
    X = X.reindex(columns=train_columns, fill_value=0)
    X_scaled = scaler.transform(X)
    return X_scaled




def evaluate_model(model, X_test, y_test, model_name="model"):
    """
    Evaluates the model, prints metrics, plots confusion matrix and ROC curve,
    saves classification report CSV and plots to outputs folder.
    """
    # Ensure output dirs
    report_dir = "/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/outputs/reports"
    figures_dir = "/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/outputs/figures"
    ensure_dir(report_dir)
    ensure_dir(figures_dir)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred = pd.Series(y_pred).map({'No': 0, 'Yes': 1}).values

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    # Print classification report
    print(f"\n=== Evaluation Report: {model_name} ===")
    print(classification_report(y_test, y_pred))
    if y_prob is not None:
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    else:
        roc_auc = None
        print("ROC-AUC Score: Not available (no predict_proba method)")

    # Save classification report as CSV
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()
    report_path = os.path.join(report_dir, f"{model_name.lower().replace(' ', '_')}_classification_report.csv")
    df_report.to_csv(report_path)
    print(f"Saved classification report to {report_path}")

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = os.path.join(figures_dir, f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_path}")

    # Plot ROC curve 
    if y_prob is not None:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        roc_path = os.path.join(figures_dir, f"{model_name.lower().replace(' ', '_')}_roc_curve.png")
        plt.savefig(roc_path)
        plt.close()
        print(f"Saved ROC curve plot to {roc_path}")

#clv_model


def evaluate_clv_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")

    # Save figure
    figures_dir = "/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/outputs/figures"
    reports_dir = "/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/outputs/reports"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.xlabel("Actual CLV")
    plt.ylabel("Predicted CLV")
    plt.title("Actual vs Predicted CLV")
    plt.savefig(f"{figures_dir}/clv_actual_vs_predicted.png")
    plt.close()


    # Save evaluation report as CSV
    metrics_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'R2'],
        'Value': [mae, rmse, r2]
    })
    csv_path = f"{reports_dir}/clv_regression_report.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved CLV regression report to {csv_path}")

