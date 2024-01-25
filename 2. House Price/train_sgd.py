from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, SGDRegressor

import warnings
warnings.filterwarnings("ignore", message="'squared' is deprecated in version 1.4 and will be removed in 1.6.")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

no_of_is = 50

# Preprocessing
df = pd.read_csv('./train.csv')
pd.set_option('display.max_rows', None)

def preprocess(df):
    mean_subclass_saleprice = df.groupby('MSSubClass')['SalePrice'].median()
    df['SubClassSalePriceMean'] = df['MSSubClass'].map(mean_subclass_saleprice)
    df.drop('MSSubClass', axis=1, inplace=True)

    mean_zoning_saleprice = df.groupby('MSZoning')['SalePrice'].median()
    df['ZoningSalePriceMean'] = df['MSZoning'].map(mean_zoning_saleprice)
    df.drop('MSZoning', axis=1, inplace=True)

    df['Street'] = df['Street'].map({'Grvl': 0, 'Pave': 1})
    df['Alley'] = df['Alley'].map({'Grvl': 0, 'Pave': 1})
    df['LotShape'] = df['LotShape'].map({'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3})
    df['LandContour'] = df['LandContour'].map({'Lvl': 0, 'Bnk': 1, 'HLS': 2, 'Low': 3})
    df['Utilities'] = df['Utilities'].map({'AllPub': 0, 'NoSewr': 1, 'NoSeWa': 2, 'ELO': 3})

    median_lot_config_saleprice = df.groupby('LotConfig')['SalePrice'].median()
    df['LotConfigSalePriceMedian'] = df['LotConfig'].map(median_lot_config_saleprice)
    df.drop('LotConfig', axis=1, inplace=True)

    df['LandSlope'] = df['LandSlope'].map({'Gtl': 0, 'Mod': 1, 'Sev': 2})

    mean_neighborhood_saleprice = df.groupby('Neighborhood')['SalePrice'].mean()
    df['NeighborhoodSalePriceMean'] = df['Neighborhood'].map(mean_neighborhood_saleprice)
    df.drop('Neighborhood', axis=1, inplace=True)

    mean_condition1_saleprice = df.groupby('Condition1')['SalePrice'].mean()
    mean_condition2_saleprice = df.groupby('Condition2')['SalePrice'].mean()
    df['ConditionTotal'] = df.apply(lambda row: mean_condition1_saleprice[row['Condition1']] + mean_condition2_saleprice[row['Condition2']] /2, axis=1)
    df.drop('Condition1', axis=1, inplace=True)
    df.drop('Condition2', axis=1, inplace=True)

    median_bldgtype_saleprice = df.groupby('BldgType')['SalePrice'].median()
    df['BldgTypeSalePriceMedian'] = df['BldgType'].map(median_bldgtype_saleprice)
    df.drop('BldgType', axis=1, inplace=True)

    median_housestyle_saleprice = df.groupby('HouseStyle')['SalePrice'].median()
    df['HouseStyleSalePriceMedian'] = df['HouseStyle'].map(median_housestyle_saleprice)
    df.drop('HouseStyle', axis=1, inplace=True)

    df['YearBuilt'] = df['YearBuilt'].map(lambda x: x - 1872)
    df['YearRemodAdd'] = df['YearRemodAdd'].map(lambda x: x - 1950)

    median_roofstyle_saleprice = df.groupby('RoofStyle')['SalePrice'].median()
    df['RoofStyleSalePriceMedian'] = df['RoofStyle'].map(median_roofstyle_saleprice)
    df.drop('RoofStyle', axis=1, inplace=True)

    median_roofmatl_saleprice = df.groupby('RoofMatl')['SalePrice'].median()
    df['RoofMatlSalePriceMedian'] = df['RoofMatl'].map(median_roofmatl_saleprice)
    df.drop('RoofMatl', axis=1, inplace=True)

    median_exterior1st_saleprice = df.groupby('Exterior1st')['SalePrice'].median()
    df['Exterior1stSalePriceMedian'] = df['Exterior1st'].map(median_exterior1st_saleprice)
    df.drop('Exterior1st', axis=1, inplace=True)

    median_exterior2nd_saleprice = df.groupby('Exterior2nd')['SalePrice'].median()
    df['Exterior2ndSalePriceMedian'] = df['Exterior2nd'].map(median_exterior2nd_saleprice)
    df.drop('Exterior2nd', axis=1, inplace=True)

    median_masvnrtype_saleprice = df.groupby('MasVnrType')['SalePrice'].median()
    df['MasVnrTypeSalePriceMedian'] = df['MasVnrType'].map(median_masvnrtype_saleprice)
    df.drop('MasVnrType', axis=1, inplace=True)

    mean_masvnrarea_saleprice = df.groupby('MasVnrArea')['SalePrice'].mean()
    df['MasVnrAreaSalePriceMean'] = df['MasVnrArea'].map(mean_masvnrarea_saleprice)
    df.drop('MasVnrArea', axis=1, inplace=True)

    df['ExterQual'] = df['ExterQual'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4})
    df['ExterCond'] = df['ExterCond'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4})

    median_foundation_saleprice = df.groupby('Foundation')['SalePrice'].median()
    df['FoundationSalePriceMedian'] = df['Foundation'].map(median_foundation_saleprice)
    df.drop('Foundation', axis=1, inplace=True)

    df['BsmtQual'] = df['BsmtQual'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4, 'NA': 5})
    df['BsmtCond'] = df['BsmtCond'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4, 'NA': 5})
    df['BsmtExposure'] = df['BsmtExposure'].map({'Gd': 0, 'Av': 1, 'Mn': 2, 'No': 3, 'NA': 4})
    df['BsmtFinType1'] = df['BsmtFinType1'].map({'GLQ': 0, 'ALQ': 1, 'BLQ': 2, 'Rec': 3, 'LwQ': 4, 'Unf': 5, 'NA': 6})
    df['BsmtFinType2'] = df['BsmtFinType2'].map({'GLQ': 0, 'ALQ': 1, 'BLQ': 2, 'Rec': 3, 'LwQ': 4, 'Unf': 5, 'NA': 6})
    df['Heating'] = df['Heating'].map({'Floor': 0, 'GasA': 1, 'GasW': 2, 'Grav': 3, 'OthW': 4, 'Wall': 5})
    df['HeatingQC'] = df['HeatingQC'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4})
    df['CentralAir'] = df['CentralAir'].map({'N': 0, 'Y': 1})

    median_electrical_saleprice = df.groupby('Electrical')['SalePrice'].median()
    df['ElectricalSalePriceMedian'] = df['Electrical'].map(median_electrical_saleprice)
    df.drop('Electrical', axis=1, inplace=True)

    df['KitchenQual'] = df['KitchenQual'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4})
    df['Functional'] = df['Functional'].map({'Typ': 0, 'Min1': 1, 'Min2': 2, 'Mod': 3, 'Maj1': 4, 'Maj2': 5, 'Sev': 6, 'Sal': 7})
    df['FireplaceQu'] = df['FireplaceQu'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4, 'NA': 5})

    median_garagetype_saleprice = df.groupby('GarageType')['SalePrice'].median()
    df['GarageTypeSalePriceMedian'] = df['GarageType'].map(median_garagetype_saleprice)
    df.drop('GarageType', axis=1, inplace=True)

    df['GarageYrBlt'] = df['GarageYrBlt'].map(lambda x: x - 1900)
    df['GarageFinish'] = df['GarageFinish'].map({'Fin': 0, 'RFn': 1, 'Unf': 2, 'NA': 3})
    df['GarageQual'] = df['GarageQual'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4, 'NA': 5})
    df['GarageCond'] = df['GarageCond'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4, 'NA': 5})
    df['PavedDrive'] = df['PavedDrive'].map({'Y': 0, 'P': 1, 'N': 2})
    df['PoolQC'] = df['PoolQC'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'NA': 4})
    df['Fence'] = df['Fence'].map({'GdPrv': 0, 'MnPrv': 1, 'GdWo': 2, 'MnWw': 3, 'NA': 4})

    median_miscfeature_saleprice = df.groupby('MiscFeature')['SalePrice'].median()
    df['MiscFeatureSalePriceMedian'] = df['MiscFeature'].map(median_miscfeature_saleprice)
    df.drop('MiscFeature', axis=1, inplace=True)

    median_saletype_saleprice = df.groupby('SaleType')['SalePrice'].median()
    df['SaleTypeSalePriceMedian'] = df['SaleType'].map(median_saletype_saleprice)
    df.drop('SaleType', axis=1, inplace=True)

    median_salecondition_saleprice = df.groupby('SaleCondition')['SalePrice'].median()
    df['SaleConditionSalePriceMedian'] = df['SaleCondition'].map(median_salecondition_saleprice)
    df.drop('SaleCondition', axis=1, inplace=True)

    df.drop('Id', axis=1, inplace=True)

    return df

df = preprocess(df)

# 아웃라이어 제거
z_scores = stats.zscore(df)
abs_z_scores = np.abs(z_scores)
outliers = (abs_z_scores >= 3)
df[outliers] = np.nan

# NaN 제거
df = df.fillna(df.mean())

# input / target 분리
saleprice_target = df['SalePrice']
saleprice_input = df.drop('SalePrice', axis=1)

random = np.random.randint(0, 1000, no_of_is)
# 하이퍼파라미터 확인을 위한 루프
# iteration = [10, 100, 1000, 10000, 100000, 1000000]
# iteration = np.arange(1000, 100000, 1000)

train_result = []
test_result = []
for i in range(100):
    train_input, test_input, train_target, test_target = train_test_split(saleprice_input, saleprice_target, random_state=i)

    ss = StandardScaler()
    ss.fit(train_input)
    train_scaled = ss.transform(train_input)
    test_scaled = ss.transform(test_input)

    sgd = SGDRegressor(max_iter=100000, alpha=1, tol=0.1, random_state=i, learning_rate='constant', eta0=0.001)
    # sgd = SGDRegressor(max_iter=100000, alpha=0.01, tol=0.1, random_state=i, learning_rate='constant', eta0=0.01)
    sgd.fit(train_scaled, train_target)

    train_predictions = sgd.predict(train_scaled)
    test_predictions = sgd.predict(test_scaled)

    train_predictions[train_predictions <= 0] = 1e-10
    test_predictions[test_predictions <= 0] = 1e-10

    train_log_predictions = np.log(train_predictions)
    test_log_predictions = np.log(test_predictions)
    train_log_target = np.log(train_target)
    test_log_target = np.log(test_target)

    train_rmse = mean_squared_error(train_log_target, train_log_predictions, squared=False)
    test_rmse = mean_squared_error(test_log_target, test_log_predictions, squared=False)

    train_result.append(train_rmse)
    test_result.append(test_rmse)

    # print("train: ", end="")
    # print(train_rmse, end=" / ")
    # print("test: ", end="")
    # print(test_rmse)


print("\n\nRESULT\n==============================")
print("TRAIN")
print("mean: ", end="")
print(np.mean(train_result))
print("TEST")
print("mean: ", end="")
print(np.mean(test_result)) 
