
import pandas as pd
import numpy as np


df=pd.read_csv('amazon.csv')

df=df.drop('product_link',axis=1)

df=df.drop('img_link',axis=1)
print(df.head())

print(df.describe())

print(df.duplicated().value_counts())

df=df.drop_duplicates()

print(df.duplicated().value_counts())

print(df.isna().sum())

df['rating_count']=df['rating_count'].str.replace(',','',regex=False)
df['rating_count']=pd.to_numeric(df['rating_count'])

df['rating_count']=df['rating_count'].fillna(df['rating_count'].mean())

df['rating']=df['rating'].str.replace('|','',regex=False)

df['rating']=pd.to_numeric(df['rating'])

print(df.dtypes)

df["discounted_price"]=df["discounted_price"].str.replace(",",'',regex=False)
df["discounted_price"]=df["discounted_price"].str.replace("₹",'',regex=False)
df["actual_price"]=df["actual_price"].str.replace(",",'',regex=False)
df["actual_price"]=df["actual_price"].str.replace("₹",'',regex=False)

df=df.copy()
df['discount_percentage']=df['discount_percentage'].str.replace('%','',regex=False)

print(df['discount_percentage'].head(5).tolist())

df

df['rating_count']=df['rating_count'].astype(int)

df


print(df.dtypes)

print(df.columns)
print(df["discounted_price"].head())