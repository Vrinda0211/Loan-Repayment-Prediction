import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score

train=pd.read_csv("../data/processed/train_balanced.csv")
test=pd.read_csv("../data/processed/test.csv")

X_train=train.drop(columns=["TARGET","SK_ID_CURR"])
y_train=train["TARGET"]

X_test=test.drop(columns=["TARGET","SK_ID_CURR"])
y_test=test["TARGET"]

categorical_cols=X_train.select_dtypes(include=["object"]).columns

X_train=pd.get_dummies(X_train,columns=categorical_cols,dtype=int)
X_test=pd.get_dummies(X_test,columns=categorical_cols,dtype=int)

X_test=X_test.reindex(columns=X_train.columns,fill_value=0)

model=GaussianNB()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)
y_prob=model.predict_proba(X_test)[:,1]

print("Naive Bayes Results")
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Precision:",precision_score(y_test,y_pred))
print("Recall:",recall_score(y_test,y_pred))
print("F1 Score:",f1_score(y_test,y_pred))
print("ROC-AUC:",roc_auc_score(y_test,y_prob))