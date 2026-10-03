import pandas as pd
from imblearn.over_sampling import SMOTE

train=pd.read_csv("../data/processed/train.csv")

X=train.drop(columns=["TARGET"])
y=train["TARGET"]

print("Before SMOTE:")
print(y.value_counts())

smote=SMOTE(random_state=42)
X_resampled,y_resampled=smote.fit_resample(X,y)

train_balanced=pd.concat(
    [pd.DataFrame(X_resampled,columns=X.columns),
     pd.Series(y_resampled,name="TARGET")],
    axis=1
)

train_balanced.to_csv(
    "../data/processed/train_balanced.csv",
    index=False
)

print("\nAfter SMOTE:")
print(train_balanced["TARGET"].value_counts())
print("\nBalanced shape:",train_balanced.shape)