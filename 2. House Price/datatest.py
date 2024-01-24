import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./train.csv')

# plt:
# x axis: LotConfig
# y axis: SalePrice

variable = 'SaleCondition'

print("min: ", end="")
print(df[variable].min())
print("max: ", end="")
print(df[variable].max())
print("mean: ", end="")
print(df[variable].mean())
print("median: ", end="")
print(df[variable].median())

plt.scatter(df[variable], df['SalePrice'])
plt.show()