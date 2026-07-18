from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

path=r"B:\AI-ML-Roadmap\lin_vs_ridge_vs_lasso_reg\student-mat-cleaned.csv"

df=pd.read_csv(path)
orig=df.copy()

X=df.drop(columns="G3")
Y=df["G3"]

#k-fold
kfold=KFold(n_splits=5, shuffle=True, random_state=18)

for i in [0.15,0.175,0.2,0.3,0.5,0.75,1]:
    lasso_score=[]
    ridge_score=[]
    for train_i, test_i in kfold.split(X):
        scaler=StandardScaler()

        X_train=scaler.fit_transform(X.iloc[train_i])
        X_test=scaler.transform(X.iloc[test_i])

        y_train=Y.iloc[train_i]
        y_test=Y.iloc[test_i]

        ridge=Ridge(alpha=i)
        lasso=Lasso(alpha=i)

        ridge.fit(X_train, y_train)
        lasso.fit(X_train, y_train)

        ridge_score.append(ridge.score(X_test, y_test))
        lasso_score.append(lasso.score(X_test, y_test))

    print(f"\nAlpha={i}")
    print("Ridge Scores :", ridge_score)
    print("Lasso Scores :", lasso_score)
    print("Ridge Avg :", np.mean(ridge_score))
    print("Lasso Avg :", np.mean(lasso_score))