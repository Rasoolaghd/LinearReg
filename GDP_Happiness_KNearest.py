
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor   
import numpy as np

# Read file into Dataframe
df_happiness = pd.read_csv("data.csv", usecols=["Country", "Indicator", "Unit", "Value"])
df_happiness = df_happiness[(df_happiness["Indicator"]=="Life satisfaction") & (df_happiness["Unit"]=="Average score") ]

df_happiness.sort_values("Country")
# in case there are multiple rows, group by and pick the max
df_happiness = df_happiness.groupby(["Country"]).max().reset_index()

# Read file into Dataframe
df_gdp = pd.read_excel("WEO_Data_p.xls", usecols=["Country", "2015"] )
df_gdp.sort_values("Country")
# in case there are multiple rows, group by and pick the max
df_gdp = df_gdp.groupby(["Country"]).max().reset_index()

# inner join the two dataframe on Country as the key
df_joint = pd.merge(df_happiness, df_gdp, on="Country", how="inner")
df_joint = df_joint[["Country", "Value", "2015"]]

# plot the scatter plot to visualize the relation
(sns.scatterplot(x="2015", y="Value", data=df_joint)).set(xlim=(0,120000) , ylim=(1, 10))
# plt.show()

# model for learning the relation 
model = KNeighborsRegressor(n_neighbors=3)

X = np.c_[df_joint["2015"]]
y = np.c_[df_joint["Value"]]

model.fit(X, y)

#  prediction based on the model for Cyprus
X_new = [[22587]]
y_predict = model.predict(X_new)
print(y_predict) # 6.26666667
