import pandas as pd

data_path="../data"

df=pd.read_csv(f"{data_path}/processed/application_train_clean.csv")

files=[
    "historical_features.csv",
    "installment_features.csv",
    "credit_card_features.csv",
    "pos_cash_features.csv",
    "bureau_balance_features.csv"
]

for file in files:
    features=pd.read_csv(f"{data_path}/processed/{file}")
    df=df.merge(features,on="SK_ID_CURR",how="left")

feature_cols=df.columns.drop(["SK_ID_CURR","TARGET"])
df[feature_cols]=df[feature_cols].fillna(0)

df.to_csv(
    f"{data_path}/processed/final_dataset.csv",
    index=False
)

print("Final shape:",df.shape)
print("Missing values:",df.isnull().sum().sum())
print("Target distribution:")
print(df["TARGET"].value_counts())