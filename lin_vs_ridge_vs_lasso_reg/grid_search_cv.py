from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge,Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

path=r"B:\AI-ML-Roadmap\lin_vs_ridge_vs_lasso_reg\student-mat-cleaned.csv"
df=pd.read_csv(path)

X=df.drop(columns="G3")
Y=df["G3"]

ridge_pipeline=Pipeline([("scaler",StandardScaler()),("ridge",Ridge())])
lasso_pipeline=Pipeline([("scaler",StandardScaler()),("lasso",Lasso())])


ridge_para={"ridge__alpha":[0.0001,0.001,0.01,0.1,0.15]}
lasso_para={"lasso__alpha":[0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.3]}



lasso_grid=GridSearchCV(lasso_pipeline,lasso_para,cv=5)
ridge_grid=GridSearchCV(ridge_pipeline,ridge_para,cv=5)


ridge_grid.fit(X,Y)
lasso_grid.fit(X,Y)

print(ridge_grid.best_params_)
print(ridge_grid.best_score_)

print(lasso_grid.best_params_)
print(lasso_grid.best_score_)
