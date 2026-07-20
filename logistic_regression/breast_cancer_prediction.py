from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import(precision_score,accuracy_score,recall_score,f1_score,confusion_matrix)
import pandas as pd
data=load_breast_cancer()

df=pd.DataFrame(data.data,columns=data.feature_names)
df["target"]=data.target

#train_test_split
X=df.drop(columns="target")
Y=df["target"]
scaler=StandardScaler()
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=18)
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#model
model=LogisticRegression()
model.fit(X_train,y_train)
preds=model.predict(X_test)

#eval
a=accuracy_score(y_test,preds)
p=precision_score(y_test,preds)
r=recall_score(y_test,preds)
f1=f1_score(y_test,preds)
conf_matrix=confusion_matrix(y_test,preds)

print(a, p, r, f1, conf_matrix)