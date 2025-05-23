from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os
import pickle


app = Flask(__name__)

# Load trained model and scaler
model = joblib.load('outputs/models/logistic_model.pkl')
scaler = joblib.load('outputs/models/scaler.pkl')
with open("outputs/models/features.pkl", "rb") as f:
    features = pickle.load(f)

    import pickle

# Load trained columns
# trained_columns = pd.read_csv("data/processed/X_train_columns.csv", header=None)[0].tolist()
#trained_columns = pd.read_csv("/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/data/processed/X_train_columns.csv", header=None)[0].tolist()
# Get root of the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COLUMNS_PATH = os.path.join(BASE_DIR, "data", "processed", "X_train_columns.csv")

trained_columns = pd.read_csv(COLUMNS_PATH, header=None)[0].tolist()



# One-hot encode user input
# (Moved to inside the predict() function after input_df is created)

# Align columns
# (Moved to inside the predict() function after input_df is created)

# Your full list of 45 feature names (same order as training)
feature_columns = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Female', 'gender_Male',
    'Partner_No', 'Partner_Yes',
    'Dependents_No', 'Dependents_Yes',
    'PhoneService_No', 'PhoneService_Yes',
    'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_No', 'PaperlessBilling_Yes',
    'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form

        # 1. Numeric features
        numeric_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
        numeric_data = {f: float(form_data[f]) if f != 'SeniorCitizen' else int(form_data[f]) for f in numeric_features}

        # 2. Categorical features
        categorical_fields = [
            'gender', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]
        categorical_data = {field: form_data.get(field) for field in categorical_fields}

        # 3. Create DataFrame
        input_df = pd.DataFrame([{**numeric_data, **categorical_data}])

        # 4. One-hot encode categorical
        input_df = pd.get_dummies(input_df)

        # 5. Align columns with features.pkl
        input_df = input_df.reindex(columns=features, fill_value=0)

        # 6. Scale entire feature set
        input_scaled = scaler.transform(input_df)
        input_df_scaled = pd.DataFrame(input_scaled, columns=features)

        # 7. Predict
        prediction = model.predict(input_df_scaled)[0]
        prob = model.predict_proba(input_df_scaled)[0][1]

        result = "Yes" if prediction == 1 else "No"
        return render_template('index.html', prediction=result, probability=round(prob * 100, 2))
    
    except Exception as e:
        return f"Something went wrong: {e}", 500

# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

