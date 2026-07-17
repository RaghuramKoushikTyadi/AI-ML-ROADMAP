from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import pandas as pd
import numpy as np

path=r"B:\AI-ML-Roadmap\lin_vs_ridge_vs_lasso_reg\student-mat-cleaned.csv"

df=pd.read_csv(path)
orig=df.copy()

#train-test-split
scaler=StandardScaler()
X=df.drop(columns="G3")
Y=df["G3"]

X_train,X_test,y_train,y_test=train_test_split(
    X,Y,test_size=0.2,random_state=18
)

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#fun time :)
model=Ridge(alpha=0.1)
model.fit(X_train,y_train)
preds=model.predict(X_test)

#weights
print(model.coef_)
print(model.intercept_)#pred when all feature values are zero

#eval
print(preds[:10])
print(y_test.iloc[:10])

mse=mean_squared_error(y_test,preds)
mae=mean_absolute_error(y_test,preds)
r2=r2_score(y_test,preds)

print(f"MAE:{mae:.2f}")
print(f"MSE:{mse:.2f}")
print(f"R²:{r2:.2f}")





