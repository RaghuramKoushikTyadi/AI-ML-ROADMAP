from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd 

path=r"B:\AI-ML-Roadmap\lin_vs_ridge_vs_lasso_reg\student-mat-cleaned.csv"

df=pd.read_csv(path)

X=df.drop(columns="G3")
Y=df["G3"]

ridge_pipeline=Pipeline([("scaler",StandardScaler()),("ridge",Ridge(0.15))])
lasso_pipeline=Pipeline([("scaler",StandardScaler()),("lasso",Lasso(0.15))])
linear_pipeline=Pipeline([("scaler",StandardScaler()),("linear",LinearRegression())])
ridge_scores=cross_val_score(ridge_pipeline,X,Y,cv=5)
lasso_scores=cross_val_score(lasso_pipeline,X,Y,cv=5)
linear_scores=cross_val_score(linear_pipeline,X,Y,cv=5)

print("Ridge Score: ",ridge_scores)
print("Lasso Score: ",lasso_scores)
print("Linear Score: ",linear_scores)

print("Linear Mean: ",linear_scores.mean())
print("Linear Std: ",linear_scores.std())

print("Ridge Mean: ",ridge_scores.mean())
print("Ridge Std: ",ridge_scores.std())

print("Lasso Mean: ",lasso_scores.mean())
print("Lasso Std: ",lasso_scores.std())
