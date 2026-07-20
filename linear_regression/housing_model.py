from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import(mean_absolute_error,mean_squared_error,r2_score)
import pandas as pd

path=r"B:\AI-ML-Roadmap\linear_regression\cleaned_housing.csv"
df=pd.read_csv(path)

X=df.drop(columns="median_house_value")
Y=df["median_house_value"]

#train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=18)

#model
model=LinearRegression()
model.fit(X_train,y_train)
preds=model.predict(X_test)

mae=mean_absolute_error(y_test,preds)
mse=mean_squared_error(y_test,preds)
r2=r2_score(y_test,preds)

print(model.coef_)
print(model.intercept_)
print(f"MAE:{mae:.2f}")
print(f"MSE:{mse:.2f}")
print(f"R²:{r2:.2f}")
