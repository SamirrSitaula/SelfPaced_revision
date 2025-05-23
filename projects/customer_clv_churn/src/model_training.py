import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

# Load your processed dataset (if you're reading from file)
# df = pd.read_csv('data/processed/your_file.csv')


df = pd.read_csv('/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/data/processed/cleaned_data.csv')

# 1. Feature-target split
X = df.drop('Churn', axis=1)
y = df['Churn']





X = df.drop("Churn", axis=1)
X = pd.get_dummies(X)  # ✅ Encode all string columns

y = df["Churn"].map({"Yes": 1, "No": 0})  # ✅ Convert target to int




# 2. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Save test sets
X_test.to_csv("/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/data/processed/X_test.csv", index=False)
y_test.to_csv("/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/data/processed/y_test.csv", index=False)

#check for object(non- numeric column)
print(X_train.dtypes[X_train.dtypes == 'object'])

#convert to numeric columns
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)


#save features
features = X_train.columns.tolist()  # after all preprocessing steps
with open("../outputs/models/features.pkl", "wb") as f:

    pickle.dump(features, f)


# Align test set columns with train
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Save the training columns (feature names) so you can align test data later
pd.Series(X_train.columns).to_csv(
    '/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/data/processed/X_train_columns.csv',
    index=False,
    header=False
)




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






def train_clv_model(X_train, y_train, model_type='random_forest'):
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'linear':
        model = LinearRegression()
    else:
        raise ValueError("Unsupported model type")
    
    model.fit(X_train, y_train)
    return model


#overfitting and underfitting
from sklearn.metrics import accuracy_score, roc_auc_score

# Split again if needed (if you're not using a hold-out test set)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Example: Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

# Predictions
train_preds = lr_model.predict(X_train)
test_preds = lr_model.predict(X_test)

# Accuracy and AUC
train_acc = accuracy_score(y_train, train_preds)
test_acc = accuracy_score(y_test, test_preds)

train_auc = roc_auc_score(y_train, lr_model.predict_proba(X_train)[:, 1])
test_auc = roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1])

# Output
print("Logistic Regression Performance:")
print(f"Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
print(f"Train ROC-AUC: {train_auc:.4f}, Test ROC-AUC: {test_auc:.4f}")

# Optional: Repeat for Random Forest or any other model
