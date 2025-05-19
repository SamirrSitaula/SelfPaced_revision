import pandas as pd
from sklearn.model_selection import train_test_split
from cleaning import clean_data
from feature_engineering import create_features, create_clv_target
from model_training import train_clv_model
from evaluation import evaluate_clv_model
import matplotlib as plt
import seaborn as sns
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# Load processed data
df = pd.read_csv("../data/processed/final_dataset_with_clv.csv")

# Optional: clean again (if needed)
df = clean_data(df)
df = create_features(df)
df = create_clv_target(df)

# Select features and target
features = ['MonthlyCharges', 'tenure', 'TotalCharges', 'Contract_encoded', 'PaymentMethod_encoded']  # Add more as needed
X = df[features]
y = df['CLV']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = train_clv_model(X_train, y_train, model_type='random_forest')

# Evaluate
evaluate_clv_model(model, X_test, y_test)

# Predict values for residual plot
y_pred = model.predict(X_test)




 #saving regression model as csv   


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

