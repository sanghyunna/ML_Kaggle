from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np
import warnings
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# 핵심:
# 'transported' 와 '기타 feature'를 활용해 트레이닝셋에 '동승자의 생존 여부'라는 feature를 만듦
# 트레이닝셋의 '기타 feature'와 '동승자의 생존 여부'를 활용해 'transported'를 예측하는 모델을 훈련함
# 테스트셋의 '기타 feature'를 활용해 'transported'를 예측
# 예측한 'transported'와 '기타 feature'를 활용해 테스트셋의 '동승자의 생존 여부'를 예측
# 이렇게 예측한 '동승자의 생존 여부'를 활용해 'transported'를 재예측

# 발전 가능성 : cabin 활용

train_val_dict = {}

df = pd.read_csv('train.csv')


def preprocess(df):
    # ========== Feature 추가 ==========
    # LastName 추가
    df['LastName'] = df['Name'].str.split().str[0]
    # PassengerId를 통해 Group 이라는 feature 만듦
    df['Group'] = df['PassengerId'].astype(str).str[:4].astype(int)
    # 동승자의 수
    df['MemberCount'] = df['Group'].map(df['Group'].value_counts())
    # 동승자가 없으면 0, 있으면 1, 가족이면 2. 가족인지 판단은 LastName이 중복되는지 여부로 판단
    df['Family'] = 0
    df.loc[df['MemberCount'] > 1, 'Family'] = 1
    df.loc[(df['MemberCount'] > 1) & (df['LastName'].duplicated(keep=False)), 'Family'] = 2
    # RoomService, FoodCourt, ShoppingMall, Spa, VRDeck 의 합계(소비 총액) 추가
    df['TotalSpending'] = df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].sum(axis=1)
    # 같은 그룹 사람들끼리 묶어서 TotalSpending의 총액을 구하고 그 총액을 membercount로 나눈 값을 AvgSpending으로 추가
    df['AvgSpending'] = df.groupby(['Group'])['TotalSpending'].transform('sum') / df['MemberCount']
    # 돈을 내는 사람은 가장일 가능성이 높음, 생존율과 연관이 있을 수 있기 때문에 TotalSpending이 AvgSpending보다 큰 사람은 FamilyLeader라는 feature 가 true, 아닌 사람은 false
    df['FamilyLeader'] = df['TotalSpending'] > df['AvgSpending'] # 1인 가구는 가장 역할은 아니므로 false (>= 가 아니라 >)

    # ========== Feature 저장했다가 활용 ==========
    df["HomePlanet-Destination-String"] = df["HomePlanet"] + " -> " + df["Destination"]
    if 'Transported' in df.columns:
        # 출발지와 도착지를 반영한 생존율 활용 (출발지와 도착지가 같은 사람끼리 묶고 생존율을 구함)
        df["HomePlanet-Destination"] = df["HomePlanet-Destination-String"].map(df.groupby("HomePlanet-Destination-String")["Transported"].mean())
        # train_val_dict에 각 "HomePlanet-Destination-String"에 해당하는 생존율을 저장
        train_val_dict["HomePlanet-Destination-String"] = df.groupby("HomePlanet-Destination-String")["Transported"].mean()
    else:
        # train_val_dict에 저장된 생존율을 활용
        df["HomePlanet-Destination"] = df["HomePlanet-Destination-String"].map(train_val_dict["HomePlanet-Destination-String"])

    # ========== 결측치 처리 ==========
    # CyroSleep가 True인 경우, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck(소비 내용)의 결측치는 0으로 채움
    df.loc[df['CryoSleep'] == True, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = df.loc[df['CryoSleep'] == True, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)
    # CyroSleep가 True인 경우 중 vip 결측치는 false로 채우기
    df.loc[df['CryoSleep'] == True, 'VIP'] = df.loc[df['CryoSleep'] == True, 'VIP'].fillna(False)
    # 소비 내용이 모두 0인 경우 CyroSleep를 False로 채움
    df.loc[(df['RoomService'] == 0) & (df['FoodCourt'] == 0) & (df['ShoppingMall'] == 0) & (df['Spa'] == 0) & (df['VRDeck'] == 0), 'CryoSleep'] = False
    # membercount>1인 경우, VIP 결측치는 같은 그룹의 VIP의 평균으로 채움
    df.loc[df['MemberCount'] > 1, 'VIP'] = df.groupby(['Group'])['VIP'].transform('mean')
    # 소비 내용 중 결측값은 vip 가 true/false인 탑승자의 중간값으로 채움
    df.loc[df['VIP'] == True, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = \
        df.loc[df['VIP'] == True, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(
            df[df['VIP'] == True][['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].median()
        )
    df.loc[df['VIP'] == False, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = \
        df.loc[df['VIP'] == False, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(
            df[df['VIP'] == False][['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].median()
        )
    # 남은 VIP 결측치는 평균으로 채움
    df['VIP'] = df['VIP'].fillna(df['VIP'].mean())
    # 남은 소비 결측치는 중간값으로 채움
    df.loc[df['RoomService'].isnull(), 'RoomService'] = df['RoomService'].median()
    df.loc[df['FoodCourt'].isnull(), 'FoodCourt'] = df['FoodCourt'].median()
    df.loc[df['ShoppingMall'].isnull(), 'ShoppingMall'] = df['ShoppingMall'].median()
    df.loc[df['Spa'].isnull(), 'Spa'] = df['Spa'].median()
    df.loc[df['VRDeck'].isnull(), 'VRDeck'] = df['VRDeck'].median()
    # CryoSleep, HomePlanet-Destination 결측치는 전체 평균으로 채움
    df['CryoSleep'] = df['CryoSleep'].fillna(df['CryoSleep'].mean())
    df['HomePlanet-Destination'] = df['HomePlanet-Destination'].fillna(df['HomePlanet-Destination'].mean())
    # Age 결측치는 전체 중간값으로 채움
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    df['CryoSleep'] = df['CryoSleep'].astype(float)
    df['VIP'] = df['VIP'].astype(float)

    # ========== 불필요 Feature 드랍 ==========
    df = df.drop(['PassengerId', 'Name', 'LastName', 'Cabin', 'HomePlanet', 'Destination', 'HomePlanet-Destination-String'], axis=1)
    return df

def preprocess_for_second(df, predictions):
    # predictions를 df에 새로운 컬럼으로 추가
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df['TransportedPredictions'] = predictions.astype(int)

    df['GroupTransportedPredictionSum'] = df.groupby('Group')['TransportedPredictions'].transform('sum')
    df['GroupTransportedPredictionSum'] = df['GroupTransportedPredictionSum'] - df['TransportedPredictions'] # 자신을 빼줌 

    # membercount가 1인 사람들은 df['GroupSurvivalRate']을 TransportedPredictions의 평균으로 설정,
    # 1 이상인 사람들은 df['GroupTransportedPredictionSum'] / (df['MemberCount'] - 1) 으로 설정
    df['GroupTransportedPredictionAvg'] = np.where(
        df['MemberCount'] == 1, # 조건
        df['TransportedPredictions'].mean(), # 참
        df['GroupTransportedPredictionSum'] / (df['MemberCount'] - 1) # 거짓
        )

    df.drop("TransportedPredictions", axis=1, inplace=True)
    df.drop("GroupTransportedPredictionSum", axis=1, inplace=True)

    si = SimpleImputer(strategy='mean')
    df = pd.DataFrame(si.fit_transform(df), columns=df.columns)
    return df

def predict(df, lgb, lgb_two=False):
    df = preprocess(df.copy())
    df = df.astype(int)
    
    predictions = lgb.predict(df)

    if lgb_two:
        df = preprocess_for_second(df, predictions)
        predictions = lgb_two.predict(df)

    return predictions

def predict_and_save(lgb, lgb_two=False):
    test_df = pd.read_csv('test.csv')
    test_target = predict(test_df, lgb, lgb_two)  

    test_target = test_target.astype(bool)

    results = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Transported': test_target})
    results.to_csv('lgbm_second.csv', index=False)
    print("Saved as lgbm_second.csv")
    
space_titanic = preprocess(df.copy())

titanic_target = space_titanic['Transported']
titanic_input = space_titanic.drop(['Transported'], axis=1)

params = {
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 32,
    'sub_feature': 0.5,
    'n_jobs': -1,
    'verbose': -1,
}

lgb = LGBMClassifier(**params)


lgb.fit(titanic_input, titanic_target)

predictions = lgb.predict(titanic_input)

space_titanic = preprocess_for_second(space_titanic, predictions)

titanic_target = space_titanic['Transported']
titanic_input = space_titanic.drop(['Transported'], axis=1)

lgb_two = LGBMClassifier(**params)

lgb_two.fit(titanic_input, titanic_target)

print("Train accuracy:", lgb_two.score(titanic_input, titanic_target))


# 테스트셋에 대해 예측하고 csv파일로 저장
predict_and_save(lgb, lgb_two)

# print the importance of features
print("\n[ Feature importance ]")
for i in range(len(titanic_input.columns)):
    print(f"{titanic_input.columns[i]}: {lgb_two.feature_importances_[i]}")

print("\n[ True or False ratio ]")
print(f"True: {np.sum(predictions) / len(predictions)}")
print(f"False: {1 - np.sum(predictions) / len(predictions)}")