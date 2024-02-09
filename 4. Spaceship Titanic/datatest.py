import seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
특징:
* passenger_id를 쪼개서 동승자 존재여부와 생존여부를 feature로 추가할 수 있을 듯
* 동승자가 있다면, 동승자의 수와 가족인지 여부 추가
* RoomService, FoodCourt, ShoppingMall, Spa, VRDeck 의 합계(소비 총액) 추가
* CyroSleep가 True인 경우, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck의 결측치는 0으로 채움
* CyroSleep가 True인 경우 중 vip 신청자는 단 한명, 결측치는 false로 채우기
* 이름 리스트를 남성과 여성을 구분할 수 없을 것 같음
* 이름은 모든 경우에서 반드시 두 단어임. 한 단어나 세 단어 이상은 없음.
* cabin 을 deck과 num과 side로 분리
* deck끼리 묶었을 때의 평균 생존율, deck과 num끼리 묶었을 때의 평균 생존율을 feature로 추가
* 동승자 중 한 명만 돈을 내는 경향 발견, 이를 feature로 추가하고 spending은 평균 내서 반영한다

-> 동승자의 생존 여부가 결정적일 것 같은데, test 데이터에는 동승자의 정보가 없음
    => 동승자 생존 여부가 없는 학습을 한번 돌리고, 그 결과를 기반으로 재예측

의문점:
* 동승자가 비용을 지불하는 경우 소비 총액이 작은가? 아니면 평균 내야?
"""

# Load the data
df = pd.read_csv('train.csv')

# print all rows and columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# 히트맵
# numerical_df = df.select_dtypes(include='number')
# #show correl
# correlation = numerical_df.corr()
# seaborn.heatmap(correlation, annot=True, cmap='coolwarm')
# plt.show()

# 냉동수면 특징 확인
# print(df[df['CryoSleep'] == True][['VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])


# 결측치 비중 확인 (생존율을 대체로 반반)
# print(df.isnull().sum())


# 이름으로 성별 유추?
# first_names = []
# for name in df['Name'].unique():
#     if not isinstance(name, str):
#         continue
#     if len(name.split()) == 2:
#         print(name)
    # if len(name.split()) > 2:
    #     last_name = name.split()[2]
    # first_names.append(first_name)

# print(first_names)

# HomePlanet과 Destination 쌍을 feature로 추가하고, 생존비율 확인
# df["HomePlanet-Destination"] = df["HomePlanet"] + " -> " + df["Destination"]
# df["HomePlanet-Destination"] = df.groupby("HomePlanet-Destination")["Transported"].mean()

# print(df["HomePlanet"].value_counts(), df.groupby("HomePlanet")["Transported"].mean())
# print()
# print(df["Destination"].value_counts(), df.groupby("Destination")["Transported"].mean())
# print()

# # 생존율
# print(df.groupby("HomePlanet-Destination")["Transported"].mean())


df['Group'] = df['PassengerId'].astype(str).str[:4].astype(int)
df['MemberCount'] = df['Group'].map(df['Group'].value_counts())
df['TotalSpending'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)

df['LastName'] = df['Name'].str.split().str[0]
df['Group'] = df['PassengerId'].astype(str).str[:4].astype(int)
# membercount가 2 이상인 모든 사람들의 Name과 TotalSpending 출력
# print(df[df['MemberCount'] > 1][['Name', 'TotalSpending']])
# 같은 그룹 사람들끼리 묶어서 TotalSpending의 총액을 구하고 그 총액을 membercount로 나눈 값을 AvgSpending으로 추가
df['AvgSpending'] = df.groupby(['Group'])['TotalSpending'].transform('sum') / df['MemberCount']
print(df[df['MemberCount'] > 1][['Name', 'TotalSpending', 'AvgSpending']])


