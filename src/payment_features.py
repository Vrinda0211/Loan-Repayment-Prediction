import pandas as pd

data_path="../data"

installments=pd.read_csv(f"{data_path}/installments_payments.csv")

installments["PAYMENT_DELAY"]=installments["DAYS_ENTRY_PAYMENT"]-installments["DAYS_INSTALMENT"]

installment_features=installments.groupby("SK_ID_CURR").agg(
    INSTALLMENT_COUNT=("SK_ID_PREV","count"),
    INSTALLMENT_PAYMENT_TOTAL=("AMT_PAYMENT","sum"),
    INSTALLMENT_PAYMENT_MEAN=("AMT_PAYMENT","mean"),
    INSTALLMENT_AMOUNT_TOTAL=("AMT_INSTALMENT","sum"),
    INSTALLMENT_AMOUNT_MEAN=("AMT_INSTALMENT","mean"),
    PAYMENT_DELAY_MEAN=("PAYMENT_DELAY","mean"),
    PAYMENT_DELAY_MAX=("PAYMENT_DELAY","max"),
    LATE_PAYMENT_COUNT=("PAYMENT_DELAY",lambda x:(x>0).sum())
).reset_index()


feature_cols=installment_features.columns.drop("SK_ID_CURR")
installment_features[feature_cols]=installment_features[feature_cols].fillna(0)
installment_features.to_csv(
    f"{data_path}/processed/installment_features.csv",
    index=False
)

print("Installment features:",installment_features.shape)
print("Missing values:",installment_features.isnull().sum().sum())
print(installment_features.describe().T)