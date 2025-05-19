# 📊 Customer CLV & Churn Prediction

This project predicts **Customer Lifetime Value (CLV)** and **Churn** using machine learning models on a telecom dataset. It includes a full ML pipeline — from raw data to a deployable Flask web app — and provides business insights to help reduce churn and boost revenue.

---

## 📁 Project Structure

customer_clv_churn/
│
├── data/
│ ├── raw/ # Original CSV dataset
│ └── processed/ # Cleaned data
│
├── notebooks/ # EDA & modeling
│ ├── eda.ipynb
│ └── modeling.ipynb
│
├── src/ # Scripts for processing
│ ├── acquisition.py
│ ├── cleaning.py
│ ├── feature_engineering.py
│ └── model_training.py
| └── evaluate_clv.py
| └── evaluate_script.py
| └── evaluation.py
│
├── outputs/
│ ├── models/ # Trained models (.pkl)
│ ├── figures/ # Charts & visuals
│ └── reports/ # Summary reports
│
├── app/ # Flask app
│ ├── app.py
│ ├── templates/
│ │ └── index.html
│ └── pipeline.py # Optional for API structure
│
├── requirements.txt
└── README.md




---

## 📂 Dataset

- **Source**: [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- **Size**: ~7,000 customers
- **Features**:
  - Demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - Services: `InternetService`, `OnlineSecurity`, `TechSupport`, etc.
  - Billing: `MonthlyCharges`, `TotalCharges`, `PaymentMethod`
  - Contract details: `tenure`, `Contract`, `PaperlessBilling`
  - Target: `Churn`

---

## 🛠️ Libraries Used

- `pandas`, `numpy` — data processing  
- `matplotlib`, `seaborn` — visualization  
- `scikit-learn` — ML models and preprocessing  
- `joblib` — model serialization  
- `Flask` — web application  

---

## 🔄 Workflow

### 1. **Data Acquisition & Cleaning**
- Loaded the dataset using `pandas`.
- Converted `TotalCharges` to numeric, handled missing values and duplicates.
- Transformed categorical variables with **one-hot encoding**.
- Scaled numerical features (`tenure`, `MonthlyCharges`, etc.) with **StandardScaler**.

### 2. **Exploratory Data Analysis (EDA)**
- Identified key churn indicators: short tenure, monthly contract, high monthly charges.
- Visualized churn distribution across demographics and service categories.

### 3. **Feature Engineering**
- One-hot encoded all categorical columns to match model input requirements.
- Created a clean, transformed DataFrame with **45 input features**.

### 4. **Modeling**
Tried several models:
- ✅ **Logistic Regression** (best performance overall)
- Decision Tree
- Random Forest
- XGBoost (slight overfitting on small dataset)

**Best model**: **Logistic Regression**
- Chosen for interpretability and consistent performance.
- Achieved high **precision** and good **ROC-AUC score**.

### 5. **Deployment**
- Built a Flask web app for real-time prediction.
- Integrated the trained model (`logistic_model.pkl`) and scaler.
- HTML form collects all 45 required features.
- Displays predicted churn (`Yes`/`No`) and probability.

---

## 🌐 Web App

Users can input customer data via the form and get:
- **Prediction**: Will the customer churn?
- **Probability**: Confidence of the model.

🔗 _Coming soon: Live deployment on Render or Streamlit._

---

## 📈 Business Impact

This project helps telecom companies:
- Identify at-risk customers
- Tailor retention strategies
- Maximize customer lifetime value
- Reduce marketing costs via proactive interventions

---

## 🚀 How to Run

1. Clone the repo  
   ```bash
   git clone https://github.com/SamirrSitaula/customer_clv_churn.git
   cd customer_clv_churn

## install requirements

pip install -r requirements_model.txt

## requirements for docker app
pip install -r requirements.txt

## Launch the app
cd app
python app.py

open in browser: http://127.0.0.1:5000

## 👤 Author

**Samir Sitaula**  
_Data Analyst \| ML Enthusiast \| Crisis Management Grad_  
📍 California, USA  
🔗 [LinkedIn](https://www.linkedin.com/in/Whoissamir) • 🐙 [GitHub](https://github.com/SamirrSitaula)
