import pandas as pd

data_path="../data"

bureau_balance=pd.read_csv(f"{data_path}/bureau_balance.csv")

bureau_balance_features=bureau_balance.groupby("SK_ID_BUREAU").agg(
    BUREAU_MONTH_COUNT=("MONTHS_BALANCE","count"),
    BUREAU_STATUS_COUNT=("STATUS","count")
).reset_index()

bureau_balance_features=bureau_balance_features.merge(
    pd.read_csv(f"{data_path}/bureau.csv",usecols=["SK_ID_BUREAU","SK_ID_CURR"]),
    on="SK_ID_BUREAU",
    how="left"
)

final_features=bureau_balance_features.groupby("SK_ID_CURR").agg(
    BUREAU_BALANCE_MONTHS=("BUREAU_MONTH_COUNT","sum"),
    BUREAU_BALANCE_RECORDS=("BUREAU_STATUS_COUNT","sum")
).reset_index()

feature_cols=final_features.columns.drop("SK_ID_CURR")
final_features[feature_cols]=final_features[feature_cols].fillna(0)

final_features.to_csv(
    f"{data_path}/processed/bureau_balance_features.csv",
    index=False
)

print("Bureau balance features:",final_features.shape)
print("Missing values:",final_features.isnull().sum().sum())