import pandas as pd
path="B:\AI-ML-Roadmap\linear_regression\housing.csv"

df=pd.read_csv(path)
print(df.info(),df.head(5),df.isnull().sum())

df=pd.get_dummies(df,columns=["ocean_proximity"],drop_first=True,dtype=int)

print(df.head(5))

df=df.dropna()
df.to_csv("cleaned_housing.csv", index=False)