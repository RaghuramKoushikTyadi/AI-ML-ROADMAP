from sklearn.datasets import load_breast_cancer
import pandas as pd
data=load_breast_cancer()

df=pd.DataFrame(data.data,columns=data.feature_names)
df["target"]=data.target

print(df.head(5))
print(df.info())
print(df.isnull().sum)