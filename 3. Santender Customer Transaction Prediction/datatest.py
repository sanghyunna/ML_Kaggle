import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
관찰 :
- target 평균이 약 0.1
- 표준편차가 매우 큼 (3 이상인 특성이 122개)
- 변수 간 상관관계가 매우 작음
"""

df = pd.read_csv('train.csv')
df.drop(['ID_code'], axis=1, inplace=True)

df_target = df['target']
df_input = df.drop(['target'], axis=1)

target_corr = df.corr()['target'].drop('target')

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

print(df.head())
# print(df.describe())
# print(df.isna().sum()[df.isna().sum() > 0].sort_values(ascending=False))

# print((df.std() > 3).sum())

# plt.figure(figsize=(20, 20))
# sns.heatmap(df.corr(), cmap='coolwarm')
# plt.show() 

# show each column's correlation with target in barplot
# plt.figure(figsize=(20, 20))
# sns.barplot(x=target_corr.index, y=target_corr.values)
# plt.xticks(rotation=90)
# plt.show()
  