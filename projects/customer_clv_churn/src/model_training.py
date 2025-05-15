import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# Load your processed dataset (if you're reading from file)
# df = pd.read_csv('data/processed/your_file.csv')


df = pd.read_csv('/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/data/processed/cleaned_data.csv')

# 1. Feature-target split
X = df.drop('Churn', axis=1)
y = df['Churn']

# 2. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#check for object(non- numeric column)
print(X_train.dtypes[X_train.dtypes == 'object'])

#convert to numeric columns
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Align test set columns with train
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


# 3. Scaling (Optional but good for models like Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Model Training

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)  # RF doesn't need scaling

# 5. Evaluation
print("=== Logistic Regression ===")
y_pred_lr = log_reg.predict(X_test_scaled)
print(classification_report(y_test, y_pred_lr))
print("ROC-AUC:", roc_auc_score(y_test, log_reg.predict_proba(X_test_scaled)[:, 1]))

print("\n=== Random Forest ===")
y_pred_rf = rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]))

# 6. Save best model
joblib.dump(rf, '/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/outputs/models/random_forest_model.pkl')
joblib.dump(log_reg, '/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/outputs/models/logistic_model.pkl')
joblib.dump(scaler, '/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/outputs/models/scaler.pkl')

