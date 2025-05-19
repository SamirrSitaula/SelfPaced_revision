import pandas as pd
import joblib
from evaluation import evaluate_model, preprocess_for_rf, preprocess_for_logreg

if __name__ == "__main__":
    # Paths (update if needed)
    X_test_path = "/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/data/processed/X_test.csv"
    y_test_path = "/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/data/processed/y_test.csv"
    rf_model_path = "/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/outputs/models/random_forest_model.pkl"
    logreg_model_path = "/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/outputs/models/logistic_model.pkl"
    scaler_path = "/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/outputs/models/scaler.pkl"
    train_columns_path = "/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/data/processed/X_train_columns.csv"

    # Load test data
    X_test_raw = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()
    y_test = y_test.map({'No': 0, 'Yes': 1})

# Convert y_test from 'No'/'Yes' to 0/1
   

    # Load models and scaler
    rf_model = joblib.load(rf_model_path)
    logreg_model = joblib.load(logreg_model_path)
    scaler = joblib.load(scaler_path)

    # Load train columns list for dummy alignment
    train_columns = pd.read_csv(train_columns_path, header=None).squeeze().tolist()

    # Preprocess test data for Random Forest and evaluate
    X_test_rf = preprocess_for_rf(X_test_raw, train_columns)
    evaluate_model(rf_model, X_test_rf, y_test, model_name="Random Forest")

    # Preprocess test data for Logistic Regression and evaluate
    X_test_logreg = preprocess_for_logreg(X_test_raw, train_columns, scaler)
    evaluate_model(logreg_model, X_test_logreg, y_test, model_name="Logistic Regression")
