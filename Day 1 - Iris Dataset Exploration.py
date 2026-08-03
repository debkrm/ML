#import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris=load_iris()
df=pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"]=iris.target
print("First 5 records")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nColumn names")
print(df.columns)

print("\nDataset info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\n Stats Summary")
print(df.describe())

print("\nTarget class Distribution")
print(df["target"].value_counts())

plt.figure(figsize=(8,5))
plt.hist(df["sepal length (cm)"])
plt.xlabel("Sepal Length(cm)")
plt.ylabel("Frequency")
plt.title("Distribution of Sepal Length")
plt.show()
