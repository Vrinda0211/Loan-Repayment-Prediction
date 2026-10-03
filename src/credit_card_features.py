import pandas as pd

data_path="../data"

credit_card=pd.read_csv(f"{data_path}/credit_card_balance.csv")

credit_card_features=credit_card.groupby("SK_ID_CURR").agg(
    CREDIT_CARD_RECORD_COUNT=("SK_ID_PREV","count"),
    CREDIT_CARD_BALANCE_MEAN=("AMT_BALANCE","mean"),
    CREDIT_CARD_BALANCE_MAX=("AMT_BALANCE","max"),
    CREDIT_LIMIT_MEAN=("AMT_CREDIT_LIMIT_ACTUAL","mean"),
    CREDIT_LIMIT_MAX=("AMT_CREDIT_LIMIT_ACTUAL","max"),
    PAYMENT_TOTAL_MEAN=("AMT_PAYMENT_TOTAL_CURRENT","mean"),
    DRAWINGS_TOTAL_MEAN=("AMT_DRAWINGS_CURRENT","mean"),
    DPD_MEAN=("SK_DPD","mean"),
    DPD_MAX=("SK_DPD","max"),
    DPD_COUNT=("SK_DPD",lambda x:(x>0).sum())
).reset_index()

feature_cols=credit_card_features.columns.drop("SK_ID_CURR")
credit_card_features[feature_cols]=credit_card_features[feature_cols].fillna(0)

credit_card_features.to_csv(
    f"{data_path}/processed/credit_card_features.csv",
    index=False
)

print("Credit card features:",credit_card_features.shape)
print("Missing values:",credit_card_features.isnull().sum().sum())