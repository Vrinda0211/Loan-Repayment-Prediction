import pandas as pd

train=pd.read_csv("../data/application_train.csv")

categorical=train.select_dtypes(include=["object","str"]).columns
numerical=train.select_dtypes(include=["number"]).columns

print("Categorical columns:",len(categorical))
print(list(categorical))

print("\nNumerical columns:",len(numerical))