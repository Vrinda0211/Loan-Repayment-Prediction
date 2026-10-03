import pandas as pd
from pathlib import Path

DATA_PATH=Path("../data/application_train.csv")
OUTPUT_PATH=Path("../data/processed/application_train_clean.csv")

df=pd.read_csv(DATA_PATH)

id_col=df["SK_ID_CURR"]
target=df["TARGET"]

df=df.drop(columns=["SK_ID_CURR","TARGET"])
df["AGE_YEARS"]=-df["DAYS_BIRTH"]/365.25
df["EMPLOYED_YEARS"]=-df["DAYS_EMPLOYED"]/365.25

df["CREDIT_INCOME_RATIO"]=df["AMT_CREDIT"]/df["AMT_INCOME_TOTAL"]
df["ANNUITY_INCOME_RATIO"]=df["AMT_ANNUITY"]/df["AMT_INCOME_TOTAL"]
df["CREDIT_GOODS_RATIO"]=df["AMT_CREDIT"]/df["AMT_GOODS_PRICE"]

df["INCOME_PER_FAMILY_MEMBER"]=df["AMT_INCOME_TOTAL"]/df["CNT_FAM_MEMBERS"]
df["CHILDREN_RATIO"]=df["CNT_CHILDREN"]/df["CNT_FAM_MEMBERS"]

df=df.replace([float("inf"),float("-inf")],pd.NA)

missing_pct=df.isnull().mean()*100
drop_cols=missing_pct[missing_pct>60].index
df=df.drop(columns=drop_cols)

categorical_cols=df.select_dtypes(include=["object","str"]).columns
numerical_cols=df.select_dtypes(include=["number"]).columns

df[categorical_cols]=df[categorical_cols].fillna("Unknown")
df[numerical_cols]=df[numerical_cols].fillna(df[numerical_cols].median())

df=pd.get_dummies(df,columns=categorical_cols,dtype=int)

df.insert(0,"SK_ID_CURR",id_col)
df["TARGET"]=target

OUTPUT_PATH.parent.mkdir(parents=True,exist_ok=True)
df.to_csv(OUTPUT_PATH,index=False)

print("Original shape:",len(id_col),len(id_col)+122-1)
print("Dropped columns:",len(drop_cols))
print("Final shape:",df.shape)
print("Saved to:",OUTPUT_PATH)