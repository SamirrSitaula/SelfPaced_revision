# src/preprocessing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess(path="data/raw/hr_data.csv"):
    df = pd.read_csv(path)
    
    # Fill missing values
    df["MonthlyIncome"].fillna(df["MonthlyIncome"].median(), inplace=True)
    df["JobSatisfaction"].fillna(df["JobSatisfaction"].mode()[0], inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for col in ["Department", "JobRole", "Attrition"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Scale numerical variables
    scaler = StandardScaler()
    num_cols = ["Age", "MonthlyIncome", "DistanceFromHome", "YearsAtCompany"]
    df[num_cols] = scaler.fit_transform(df[num_cols])

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler, label_encoders
