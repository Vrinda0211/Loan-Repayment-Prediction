import pandas as pd

application=pd.read_csv("../data/processed/application_train_clean.csv")
historical=pd.read_csv("../data/processed/historical_features.csv")

df=application.merge(
    historical,
    on="SK_ID_CURR",
    how="left"
)

historical_cols=historical.columns.drop("SK_ID_CURR")

df[historical_cols]=df[historical_cols].fillna(0)

df.to_csv(
    "../data/processed/application_train_features.csv",
    index=False
)

print("Application shape:",application.shape)
print("Final shape:",df.shape)
print("Missing values:",df.isnull().sum().sum())
print("Target distribution:")
print(df["TARGET"].value_counts())