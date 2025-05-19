from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load('../outputs/models/logistic_model.pkl')
scaler = joblib.load('../outputs/models/scaler.pkl')

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
    # 1. Get form data
    form_data = request.form

    # 2. Initialize input vector with 0s
    input_data = dict.fromkeys(feature_columns, 0)

    # 3. Fill numeric fields
    input_data['SeniorCitizen'] = int(form_data['SeniorCitizen'])
    input_data['tenure'] = float(form_data['tenure'])
    input_data['MonthlyCharges'] = float(form_data['MonthlyCharges'])
    input_data['TotalCharges'] = float(form_data['TotalCharges'])

    # 4. Handle categorical one-hot fields
    categorical_fields = [
        'gender', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]

    for field in categorical_fields:
        selected_option = form_data.get(field)
        if selected_option in input_data:
            input_data[selected_option] = 1

    # 5. Convert to DataFrame and order columns
    input_df = pd.DataFrame([input_data])[feature_columns]

    # 6. Scale numerical features only
    numeric_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])

    # 7. Predict
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]  # Probability of churn

    result = "Yes" if prediction == 1 else "No"
    return render_template('index.html', prediction=result, probability=round(prob * 100, 2))


if __name__ == '__main__':
    app.run(debug=True)
