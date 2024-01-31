import pandas as pd
import numpy as np
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor 

import warnings
warnings.filterwarnings("ignore", message="'squared' is deprecated in version 1.4 and will be removed in 1.6.")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Preprocessing
df = pd.read_csv('./train.csv')

pd.set_option('display.max_rows', None)

median_saleprice_dictionary = {}

def preprocess(df, isTest=False):
    if not isTest:
        mean_subclass_saleprice = df.groupby('MSSubClass')['SalePrice'].median()
        median_saleprice_dictionary['SubClassSalePriceMean'] = mean_subclass_saleprice
    else:
        mean_subclass_saleprice = median_saleprice_dictionary['SubClassSalePriceMean']
    df['SubClassSalePriceMean'] = df['MSSubClass'].map(mean_subclass_saleprice)
    df.drop('MSSubClass', axis=1, inplace=True)

    if not isTest:
        mean_zoning_saleprice = df.groupby('MSZoning')['SalePrice'].median()
        median_saleprice_dictionary['ZoningSalePriceMean'] = mean_zoning_saleprice
    else:
        mean_zoning_saleprice = median_saleprice_dictionary['ZoningSalePriceMean']
    df['ZoningSalePriceMean'] = df['MSZoning'].map(mean_zoning_saleprice)
    df.drop('MSZoning', axis=1, inplace=True)

    df['Street'] = df['Street'].map({'Grvl': 0, 'Pave': 1})
    df['LotShape'] = df['LotShape'].map({'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3})
    df['LandContour'] = df['LandContour'].map({'Lvl': 0, 'Bnk': 1, 'HLS': 2, 'Low': 3})
    df['Utilities'] = df['Utilities'].map({'AllPub': 0, 'NoSewr': 1, 'NoSeWa': 2, 'ELO': 3})

    if not isTest:            
        median_lot_config_saleprice = df.groupby('LotConfig')['SalePrice'].median()
        median_saleprice_dictionary['LotConfigSalePriceMedian'] = median_lot_config_saleprice
    else:
        median_lot_config_saleprice = median_saleprice_dictionary['LotConfigSalePriceMedian']
    df['LotConfigSalePriceMedian'] = df['LotConfig'].map(median_lot_config_saleprice)
    df.drop('LotConfig', axis=1, inplace=True)

    df['LandSlope'] = df['LandSlope'].map({'Gtl': 0, 'Mod': 1, 'Sev': 2})

    if not isTest:
        mean_neighborhood_saleprice = df.groupby('Neighborhood')['SalePrice'].mean()
        median_saleprice_dictionary['NeighborhoodSalePriceMean'] = mean_neighborhood_saleprice
    else:
        mean_neighborhood_saleprice = median_saleprice_dictionary['NeighborhoodSalePriceMean']
    df['NeighborhoodSalePriceMean'] = df['Neighborhood'].map(mean_neighborhood_saleprice)
    df.drop('Neighborhood', axis=1, inplace=True)

    if not isTest:
        mean_condition1_saleprice = df.groupby('Condition1')['SalePrice'].mean()
        mean_condition2_saleprice = df.groupby('Condition2')['SalePrice'].mean()
        median_saleprice_dictionary['Condition1Mean'] = mean_condition1_saleprice
        median_saleprice_dictionary['Condition2Mean'] = mean_condition2_saleprice
    else:
        mean_condition1_saleprice = median_saleprice_dictionary['Condition1Mean']
        mean_condition2_saleprice = median_saleprice_dictionary['Condition2Mean']
    df['ConditionTotal'] = df['Condition1'].map(mean_condition1_saleprice) + df['Condition2'].map(mean_condition2_saleprice)
    df.drop('Condition1', axis=1, inplace=True)
    df.drop('Condition2', axis=1, inplace=True)

    if not isTest:
        median_bldgtype_saleprice = df.groupby('BldgType')['SalePrice'].median()
        median_saleprice_dictionary['BldgTypeSalePriceMedian'] = median_bldgtype_saleprice
    else:
        median_bldgtype_saleprice = median_saleprice_dictionary['BldgTypeSalePriceMedian']
    df['BldgTypeSalePriceMedian'] = df['BldgType'].map(median_bldgtype_saleprice)
    df.drop('BldgType', axis=1, inplace=True)

    if not isTest:
        median_housestyle_saleprice = df.groupby('HouseStyle')['SalePrice'].median()
        median_saleprice_dictionary['HouseStyleSalePriceMedian'] = median_housestyle_saleprice
    else:
        median_housestyle_saleprice = median_saleprice_dictionary['HouseStyleSalePriceMedian']
    df['HouseStyleSalePriceMedian'] = df['HouseStyle'].map(median_housestyle_saleprice)
    df.drop('HouseStyle', axis=1, inplace=True)

    df['YearBuilt'] = df['YearBuilt'].map(lambda x: 0 if x - 1872 < 0 else x - 1872)
    df['YearRemodAdd'] = df['YearRemodAdd'].map(lambda x: 0 if x - 1950 < 0 else x - 1950)

    if not isTest:
        median_roofstyle_saleprice = df.groupby('RoofStyle')['SalePrice'].median()
        median_saleprice_dictionary['RoofStyleSalePriceMedian'] = median_roofstyle_saleprice
    else:
        median_roofstyle_saleprice = median_saleprice_dictionary['RoofStyleSalePriceMedian']
    df['RoofStyleSalePriceMedian'] = df['RoofStyle'].map(median_roofstyle_saleprice)
    df.drop('RoofStyle', axis=1, inplace=True)

    if not isTest:
        median_roofmatl_saleprice = df.groupby('RoofMatl')['SalePrice'].median()
        median_saleprice_dictionary['RoofMatlSalePriceMedian'] = median_roofmatl_saleprice
    else:
        median_roofmatl_saleprice = median_saleprice_dictionary['RoofMatlSalePriceMedian']
    df['RoofMatlSalePriceMedian'] = df['RoofMatl'].map(median_roofmatl_saleprice)
    df.drop('RoofMatl', axis=1, inplace=True)

    if not isTest:
        median_exterior1st_saleprice = df.groupby('Exterior1st')['SalePrice'].median()
        median_saleprice_dictionary['Exterior1stSalePriceMedian'] = median_exterior1st_saleprice
    else:
        median_exterior1st_saleprice = median_saleprice_dictionary['Exterior1stSalePriceMedian']
    df['Exterior1stSalePriceMedian'] = df['Exterior1st'].map(median_exterior1st_saleprice)
    df.drop('Exterior1st', axis=1, inplace=True)

    if not isTest:
        median_exterior2nd_saleprice = df.groupby('Exterior2nd')['SalePrice'].median()
        median_saleprice_dictionary['Exterior2ndSalePriceMedian'] = median_exterior2nd_saleprice
    else:
        median_exterior2nd_saleprice = median_saleprice_dictionary['Exterior2ndSalePriceMedian']
    df['Exterior2ndSalePriceMedian'] = df['Exterior2nd'].map(median_exterior2nd_saleprice)
    df.drop('Exterior2nd', axis=1, inplace=True)

    if not isTest:
        median_masvnrtype_saleprice = df.groupby('MasVnrType')['SalePrice'].median()
        median_saleprice_dictionary['MasVnrTypeSalePriceMedian'] = median_masvnrtype_saleprice
    else:
        median_masvnrtype_saleprice = median_saleprice_dictionary['MasVnrTypeSalePriceMedian']
    df['MasVnrTypeSalePriceMedian'] = df['MasVnrType'].map(median_masvnrtype_saleprice)
    df.drop('MasVnrType', axis=1, inplace=True)

    if not isTest:
        mean_masvnrarea_saleprice = df.groupby('MasVnrArea')['SalePrice'].mean()
        median_saleprice_dictionary['MasVnrAreaSalePriceMean'] = mean_masvnrarea_saleprice
    else:
        mean_masvnrarea_saleprice = median_saleprice_dictionary['MasVnrAreaSalePriceMean']
    df['MasVnrAreaSalePriceMean'] = df['MasVnrArea'].map(mean_masvnrarea_saleprice)
    df.drop('MasVnrArea', axis=1, inplace=True)

    df['ExterQual'] = df['ExterQual'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4})
    df['ExterCond'] = df['ExterCond'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4})

    if not isTest:
        median_foundation_saleprice = df.groupby('Foundation')['SalePrice'].median()
        median_saleprice_dictionary['FoundationSalePriceMedian'] = median_foundation_saleprice
    else:
        median_foundation_saleprice = median_saleprice_dictionary['FoundationSalePriceMedian']
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

    if not isTest:
        median_electrical_saleprice = df.groupby('Electrical')['SalePrice'].median()
        median_saleprice_dictionary['ElectricalSalePriceMedian'] = median_electrical_saleprice
    else:
        median_electrical_saleprice = median_saleprice_dictionary['ElectricalSalePriceMedian']
    df['ElectricalSalePriceMedian'] = df['Electrical'].map(median_electrical_saleprice)
    df.drop('Electrical', axis=1, inplace=True)

    df['KitchenQual'] = df['KitchenQual'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4})
    df['Functional'] = df['Functional'].map({'Typ': 0, 'Min1': 1, 'Min2': 2, 'Mod': 3, 'Maj1': 4, 'Maj2': 5, 'Sev': 6, 'Sal': 7})
    df['FireplaceQu'] = df['FireplaceQu'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4, 'NA': 5})

    if not isTest:
        median_garagetype_saleprice = df.groupby('GarageType')['SalePrice'].median()
        median_saleprice_dictionary['GarageTypeSalePriceMedian'] = median_garagetype_saleprice
    else:
        median_garagetype_saleprice = median_saleprice_dictionary['GarageTypeSalePriceMedian']
    df['GarageTypeSalePriceMedian'] = df['GarageType'].map(median_garagetype_saleprice)
    df.drop('GarageType', axis=1, inplace=True)

    df['GarageYrBlt'] = df['GarageYrBlt'].map(lambda x: x - 1900)
    df['GarageFinish'] = df['GarageFinish'].map({'Fin': 0, 'RFn': 1, 'Unf': 2, 'NA': 3})
    df['GarageQual'] = df['GarageQual'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4, 'NA': 5})
    df['GarageCond'] = df['GarageCond'].map({'Ex': 0, 'Gd': 1, 'TA': 2, 'Fa': 3, 'Po': 4, 'NA': 5})
    df['PavedDrive'] = df['PavedDrive'].map({'Y': 0, 'P': 1, 'N': 2})

    if not isTest:
        median_saletype_saleprice = df.groupby('SaleType')['SalePrice'].median()
        median_saleprice_dictionary['SaleTypeSalePriceMedian'] = median_saletype_saleprice
    else:
        median_saletype_saleprice = median_saleprice_dictionary['SaleTypeSalePriceMedian']
    df['SaleTypeSalePriceMedian'] = df['SaleType'].map(median_saletype_saleprice)
    df.drop('SaleType', axis=1, inplace=True)

    if not isTest:
        median_salecondition_saleprice = df.groupby('SaleCondition')['SalePrice'].median()
        median_saleprice_dictionary['SaleConditionSalePriceMedian'] = median_salecondition_saleprice
    else:
        median_salecondition_saleprice = median_saleprice_dictionary['SaleConditionSalePriceMedian']
    df['SaleConditionSalePriceMedian'] = df['SaleCondition'].map(median_salecondition_saleprice)
    df.drop('SaleCondition', axis=1, inplace=True)

    df.drop('Id', axis=1, inplace=True)

    # 서로 상관 계수가 높은 값의 경우 하나만 남김
    df.drop('1stFlrSF', axis=1, inplace=True)
    df.drop('TotRmsAbvGrd', axis=1, inplace=True)
    df.drop('GarageYrBlt', axis=1, inplace=True)
    df.drop('GarageArea', axis=1, inplace=True)

    # null 값이 너무 많은 컬럼 드랍
    df.drop('PoolQC', axis=1, inplace=True)
    df.drop('MiscFeature', axis=1, inplace=True)
    df.drop('Alley', axis=1, inplace=True)
    df.drop('Fence', axis=1, inplace=True)
    
    # 아웃라이어 제거
    z_scores = stats.zscore(df)
    abs_z_scores = np.abs(z_scores)
    outliers = (abs_z_scores >= 3)
    df[outliers] = np.nan

    # NaN 제거
    df = df.fillna(df.median())

    # input / target 분리
    if not isTest:
        saleprice_target = df['SalePrice']
        saleprice_input = df.drop('SalePrice', axis=1)

    if isTest:
        return df, None
    else:
        return saleprice_input, saleprice_target

saleprice_input, saleprice_target = preprocess(df, False)

ss = StandardScaler()
ss.fit(saleprice_input)
train_scaled = ss.transform(saleprice_input)

sgd = SGDRegressor(max_iter=100000, alpha=1, tol=0.1, random_state=1, learning_rate='constant', eta0=0.001)
sgd.fit(train_scaled, saleprice_target)

train_predictions = sgd.predict(train_scaled)


# 음수나 0이 나오면 log를 취할 수 없으므로 1e-10으로 바꿈
train_predictions[train_predictions <= 0] = 1e-10

train_log_predictions = np.log(train_predictions)

train_log_target = np.log(saleprice_target)

train_rmse = mean_squared_error(train_log_target, train_log_predictions, squared=False)

print(train_rmse)


# SUBMISSION
# test_df = pd.read_csv('./test.csv')
test_df = pd.read_csv('./train.csv')

testdf_input, testdf_target = preprocess(test_df.copy(), True) # testdf_target은 없음

testdf_input = testdf_input.drop('SalePrice', axis=1)
testdf_target = test_df['SalePrice']
 
testdf_scaled = ss.transform(testdf_input)
testdf_predictions = sgd.predict(testdf_scaled)

# results = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': testdf_predictions})
# results.to_csv('./submission.csv', index=False)

testdf_log_predictions = np.log(testdf_predictions)
testdf_log_target = np.log(testdf_target)
testdf_rmse = mean_squared_error(testdf_log_target, testdf_log_predictions, squared=False)
print(testdf_rmse)