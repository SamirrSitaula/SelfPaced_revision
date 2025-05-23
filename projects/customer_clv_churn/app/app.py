from flask import Flask, render_template, request
import pandas as pd
import joblib
import pickle

app = Flask(__name__)

# Load churn prediction model, scaler, and features
model = joblib.load('outputs/models/logistic_model.pkl')
scaler = joblib.load('outputs/models/scaler.pkl')
with open('outputs/models/features.pkl', 'rb') as f:
    churn_features = pickle.load(f)

# Load CLV prediction model and features
clv_model = joblib.load('outputs/models/clv_model.pkl')
with open('outputs/models/features_clv.pkl', 'rb') as f:
    clv_features = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html', prediction=None, probability=None, clv=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form

        # Numeric features expected as float or int
        numeric_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
        numeric_data = {f: float(form_data[f]) if f != 'SeniorCitizen' else int(form_data[f]) for f in numeric_features}

        # Categorical features expected as strings
        categorical_fields = [
            'gender', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod'
        ]
        categorical_data = {field: form_data.get(field) for field in categorical_fields}

        # Combine all inputs into one DataFrame
        input_df = pd.DataFrame([{**numeric_data, **categorical_data}])

        # One-hot encode categorical features
        input_df_encoded = pd.get_dummies(input_df)

        # --- Prepare input for churn model ---
        # Align columns with churn model features, fill missing with 0
        churn_input = input_df_encoded.reindex(columns=churn_features, fill_value=0)

        # Scale churn model input
        churn_scaled = scaler.transform(churn_input)

        # Predict churn probability and class
        churn_prob = model.predict_proba(churn_scaled)[0][1]
        churn_pred = "Yes" if churn_prob > 0.5 else "No"

        # --- Prepare input for CLV model ---
        # Align columns with CLV model features, fill missing with 0
        clv_input = input_df_encoded.reindex(columns=clv_features, fill_value=0)

        # Predict CLV (usually no scaling; if your clv_model needs scaling, apply here)
        clv_value = clv_model.predict(clv_input)[0]

        return render_template(
            'index.html',
            prediction=churn_pred,
            probability=round(churn_prob * 100, 2),
            clv=round(clv_value, 2)
        )

    except Exception as e:
        return f"Something went wrong: {e}", 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
