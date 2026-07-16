from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

from sklearn.metrics import(
    mean_absolute_error, mean_squared_error,r2_score 
)

# mae=sum(|actual-predicted|)/n
# mse=sum((actual-predicted)^2)/n
# rmse=sqrt(mse)
# r2=1-(sum((actual-predicted)^2)/sum((actual-mean)^2))

path=r"B:\AI-ML-Roadmap\linear_regression\student-mat-cleaned.csv"

df=pd.read_csv(path)
orig=df.copy()


print(df.head(5))
print(df.info())

#train_test_split
X=df.drop(columns=["G1","G2","G3"])  #G1 and G2 higly correlate to final grade G3 so we dropped them alongn with G3
Y=df["G3"]

X_train,X_test,Y_train,Y_test=train_test_split(
    X,Y,test_size=0.2,random_state=18  #I like 18 :)
)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#now the fun begins :)
model=LinearRegression()
print(type(model))
model.fit(X_train,Y_train)

#weights
print(model.coef_)
print(model.intercept_)#pred when all feature values are zero

#moment of truth
preds=model.predict(X_test)
print(preds[:10])
print(Y_test.iloc[:10])

mae=mean_absolute_error(Y_test,preds)
mse=mean_squared_error(Y_test,preds)
r2=r2_score(Y_test,preds)

print(f"MAE:{mae:.2f}")
print(f"MSE:{mse:.2f}")
print(f"R²:{r2:.2f}")






