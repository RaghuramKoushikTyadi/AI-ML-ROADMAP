from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge,LinearRegression,Lasso
from sklearn.preprocessing import StandardScaler
import pandas as pd

path=r"B:\AI-ML-Roadmap\lin_vs_ridge_vs_lasso_reg\student-mat-cleaned.csv"

df=pd.read_csv(path)

orig=df.copy()


X=df.drop(columns="G3")
Y=df["G3"]

#k-fold
kfold=KFold(n_splits=5,shuffle=True,random_state=18)
linear=LinearRegression()
ridge=Ridge(0.1)
lasso=Lasso(0.1)
lin_score=[]
lasso_score=[]
ridge_score=[]
for train_i,test_i in kfold.split(X):
    scaler=StandardScaler()

    X_train=X.iloc[train_i]
    X_test=X.iloc[test_i]
    y_train = Y.iloc[train_i]
    y_test = Y.iloc[test_i]

    X_train_scaled=scaler.fit_transform(X_train)
    X_test_scaled=scaler.transform(X_test)

    linear.fit(X_train,y_train)
    ridge.fit(X_train_scaled,y_train)
    lasso.fit(X_train_scaled,y_train)

    lin_score.append(linear.score(X_test,y_test))
    ridge_score.append(ridge.score(X_test_scaled,y_test))
    lasso_score.append(lasso.score(X_test_scaled,y_test))

print("Linear:", lin_score)
print("Ridge :", ridge_score)
print("Lasso :", lasso_score)

print("Linear_avg:", sum(lin_score)/len(lin_score))
print("Ridge_avg :", sum(ridge_score)/len(ridge_score))
print("Lasso_avg :", sum(lasso_score)/len(ridge_score))



    





