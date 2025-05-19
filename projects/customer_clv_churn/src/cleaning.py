import pandas as pd
df = pd.read_csv('/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/data/processed/acquired_data.csv')

# Perform cleaning tasks
df = df.dropna() 



# Drop the customerID column as it's not needed for prediction
df.drop('customerID', axis=1, inplace=True)

# Convert 'TotalCharges' to numeric, forcing errors to NaN (i.e., invalid data)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Drop rows where 'TotalCharges' is NaN (invalid values after conversion)
df.dropna(subset=['TotalCharges'], inplace=True)

# Replace any remaining spaces with NaN
df.replace(" ", pd.NA, inplace=True)

# Drop any duplicate rows
df.drop_duplicates(inplace=True)

# Reset the index after cleaning
df.reset_index(drop=True, inplace=True)

# Show the cleaned data
df.head()

#save the cleaned file
df.to_csv('/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/data/processed/cleaned_data.csv', index=False)
df.to_csv("/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/data/processed/final_dataset_with_clv.csv", index=False)

