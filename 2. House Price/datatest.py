import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./train.csv')

# variable = 'BsmtCond'

# print("min: ", end="")
# print(df[variable].min())
# print("max: ", end="")
# print(df[variable].max())
# print("mean: ", end="")
# print(df[variable].mean())
# print("median: ", end="")
# print(df[variable].median())

# plt.scatter(df[variable], df['SalePrice'])
# plt.show()

# sns.histplot(df['SalePrice'], kde=True)
# plt.show()

# ----------------------------------------------------------------

# numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
# corr = df[numeric_columns].corr()

# # show tuples of the categories that have a correlation bigger than 0.5 (or smaller than -0.5), no dupes
# for i in range(len(corr.columns)):
#     for j in range(i):
#         if abs(corr.iloc[i, j]) > 0.7:
#             print(corr.columns[i], "\t", corr.columns[j], "\t", corr.iloc[i, j])
#             # also print each of their correlation with SalePrice
#             print(">", corr.columns[i], "\t", "SalePrice", "\t", corr.iloc[i, -1])
#             print(">", corr.columns[j], "\t", "SalePrice", "\t", corr.iloc[j, -1])
#             print("--------------------")


# plt.figure(figsize=(10, 10))
# sns.heatmap(corr, square=True, cmap='coolwarm')
# plt.show()

# ----------------------------------------------------------------

pd.set_option('display.max_rows', None)
# show percentage of missing values in each column, don't show columns with 0 missing values
print(df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False))