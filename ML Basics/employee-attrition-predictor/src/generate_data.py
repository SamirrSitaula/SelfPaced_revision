# src/generate_data.py

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

    # Add target with some patterns
    data["Attrition"] = data.apply(
        lambda row: "Yes" if (row["JobSatisfaction"] <= 2 and row["YearsAtCompany"] < 3) else "No", axis=1
    )

    # Introduce some missing values
    for col in ["MonthlyIncome", "JobSatisfaction"]:
        data.loc[data.sample(frac=0.05).index, col] = np.nan

    return data

if __name__ == "__main__":
    df = generate_synthetic_hr_data()
    df.to_csv("data/raw/hr_data.csv", index=False)
