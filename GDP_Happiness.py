
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


df_happiness = pd.read_csv("data.csv", usecols=["Country", "Indicator", "Unit", "Value"])
df_happiness = df_happiness[(df_happiness["Indicator"]=="Life satisfaction") & (df_happiness["Unit"]=="Average score") ]

df_happiness.sort_values("Country")
df_happiness = df_happiness.groupby(["Country"]).max().reset_index()

df_gdp = pd.read_excel("WEO_Data_p.xls", usecols=["Country", "2015"] )
df_gdp.sort_values("Country")
df_gdp = df_gdp.groupby(["Country"]).max().reset_index()


df_joint = pd.merge(df_happiness, df_gdp, on="Country", how="inner")
df_joint = df_joint[["Country", "Value", "2015"]]

# print(df_joint)

(sns.scatterplot(x="2015", y="Value", data=df_joint)).set(xlim=(0,120000) , ylim=(1, 10))
plt.show()

model = LinearRegression()

x = [df_joint["2015"]]  
X = np.c_[df_joint["2015"]]

y = np.c_[df_joint["Value"]]

model.fit(X, y)

X_new = [[120000]]
y_predict = model.predict(X_new)
print(y_predict)
