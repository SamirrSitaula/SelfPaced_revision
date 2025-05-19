#importing data from local repo
#Data Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

import pandas as pd
#load the dataset
#read the dataset

df = pd.read_csv('/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')
#generate first five rows
df.head()

#basic information
df.info()

#data description
df.describe()

#shape of datasets(no. of rows and columns)
df.shape

#checking for missing 
df.isnull().sum()

#checking for duplicates
df.duplicated().sum()


#data preview of the targeted columns
df['Churn'].value_counts()

df['MonthlyCharges'].describe()

df['TotalCharges'].describe()

# Perform acquisition tasks
df.to_csv('/Users/samirsitaula/Documents/Selfpaced_Practice/projects/customer_clv_churn/data/processed/acquired_data.csv', index=False)