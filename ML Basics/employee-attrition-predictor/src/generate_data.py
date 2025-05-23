# src/data_creation.py
import pandas as pd
import numpy as np
import random

np.random.seed(42)

def generate_synthetic_hr_data(n=1000):
    data = pd.DataFrame({
        "Age": np.random.randint(20, 60, size=n),
        "Department": np.random.choice(["Sales", "HR", "R&D"], size=n),
        "MonthlyIncome": np.random.randint(2000, 20000, size=n),
        "DistanceFromHome": np.random.randint(1, 30, size=n),
        "JobRole": np.random.choice(["Manager", "Executive", "Engineer", "Clerk"], size=n),
        "Education": np.random.randint(1, 5, size=n),
        "JobSatisfaction": np.random.randint(1, 5, size=n),
        "YearsAtCompany": np.random.randint(0, 40, size=n),
    })

    # Introduce some missing values
    for col in ["MonthlyIncome", "JobSatisfaction"]:
        data.loc[data.sample(frac=0.05).index, col] = np.nan

    # Generate Attrition (target) with some logic
    def generate_attrition(row):
        if row["JobSatisfaction"] == 1 and row["YearsAtCompany"] < 2:
            return "Yes"
        elif row["MonthlyIncome"] < 4000 and row["DistanceFromHome"] > 20:
            return "Yes"
        else:
            return np.random.choice(["Yes", "No"], p=[0.1, 0.9])

    data["Attrition"] = data.apply(generate_attrition, axis=1)
    return data

if __name__ == "__main__":
    df = generate_synthetic_hr_data()
    df.to_csv("data/raw/hr_data.csv", index=False)
