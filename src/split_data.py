import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv("../data/processed/final_dataset.csv")

X=df.drop(columns=["TARGET"])
y=df["TARGET"]

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

train=pd.concat([X_train,y_train],axis=1)
test=pd.concat([X_test,y_test],axis=1)

train.to_csv("../data/processed/train.csv",index=False)
test.to_csv("../data/processed/test.csv",index=False)

print("Train shape:",train.shape)
print("Test shape:",test.shape)

print("\nTrain TARGET distribution:")
print(train["TARGET"].value_counts())
print(train["TARGET"].value_counts(normalize=True)*100)

print("\nTest TARGET distribution:")
print(test["TARGET"].value_counts())
print(test["TARGET"].value_counts(normalize=True)*100)