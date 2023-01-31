import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv", delimiter=",")

#stratify by y balances the split of data by the column specified
offline_df, online_df = train_test_split(data,test_size=0.2, random_state=42,stratify=data['Diabetes_binary'])

offline_df.to_csv('offline.csv')
online_df.to_csv('online.csv')