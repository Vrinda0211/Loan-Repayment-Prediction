import pandas as pd
from pathlib import Path

DATA_PATH=Path("../data")
OUTPUT_PATH=Path("../data/processed/historical_features.csv")

bureau=pd.read_csv(DATA_PATH/"bureau.csv")

bureau_features=bureau.groupby("SK_ID_CURR").agg(
    BUREAU_CREDIT_COUNT=("SK_ID_BUREAU","count"),
    BUREAU_CREDIT_TOTAL=("AMT_CREDIT_SUM","sum"),
    BUREAU_DEBT_TOTAL=("AMT_CREDIT_SUM_DEBT","sum"),
    BUREAU_OVERDUE_TOTAL=("AMT_CREDIT_SUM_OVERDUE","sum"),
    BUREAU_CREDIT_MEAN=("AMT_CREDIT_SUM","mean"),
    BUREAU_ACTIVE_COUNT=("CREDIT_ACTIVE",lambda x:(x=="Active").sum())
).reset_index()

previous=pd.read_csv(DATA_PATH/"previous_application.csv")

previous_features=previous.groupby("SK_ID_CURR").agg(
    PREV_APPLICATION_COUNT=("SK_ID_PREV","count"),
    PREV_CREDIT_TOTAL=("AMT_CREDIT","sum"),
    PREV_CREDIT_MEAN=("AMT_CREDIT","mean"),
    PREV_APPLICATION_TOTAL=("AMT_APPLICATION","sum"),
    PREV_APPLICATION_MEAN=("AMT_APPLICATION","mean"),
    PREV_APPROVED_COUNT=("NAME_CONTRACT_STATUS",lambda x:(x=="Approved").sum()),
    PREV_REFUSED_COUNT=("NAME_CONTRACT_STATUS",lambda x:(x=="Refused").sum())
).reset_index()

features=bureau_features.merge(
    previous_features,
    on="SK_ID_CURR",
    how="outer"
)

features.to_csv(OUTPUT_PATH,index=False)

print("Bureau features:",bureau_features.shape)
print("Previous application features:",previous_features.shape)
print("Combined features:",features.shape)
print("Missing values:",features.isnull().sum().sum())
print("Saved to:",OUTPUT_PATH)