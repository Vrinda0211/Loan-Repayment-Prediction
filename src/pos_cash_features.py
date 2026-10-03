import pandas as pd

data_path="../data"

pos=pd.read_csv(f"{data_path}/POS_CASH_balance.csv")

pos_features=pos.groupby("SK_ID_CURR").agg(
    POS_RECORD_COUNT=("SK_ID_PREV","count"),
    POS_DPD_MEAN=("SK_DPD","mean"),
    POS_DPD_MAX=("SK_DPD","max"),
    POS_DPD_COUNT=("SK_DPD",lambda x:(x>0).sum()),
    POS_DPD_DEF_MEAN=("SK_DPD_DEF","mean"),
    POS_DPD_DEF_COUNT=("SK_DPD_DEF",lambda x:(x>0).sum()),
    POS_INSTALLMENT_FUTURE_MEAN=("CNT_INSTALMENT_FUTURE","mean"),
    POS_INSTALLMENT_COUNT_MEAN=("CNT_INSTALMENT","mean")
).reset_index()

feature_cols=pos_features.columns.drop("SK_ID_CURR")
pos_features[feature_cols]=pos_features[feature_cols].fillna(0)

pos_features.to_csv(
    f"{data_path}/processed/pos_cash_features.csv",
    index=False
)

print("POS Cash features:",pos_features.shape)
print("Missing values:",pos_features.isnull().sum().sum())