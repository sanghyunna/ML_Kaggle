import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

print('Load Train Data.')
train_df = pd.read_csv('train.csv')

train_target = np.array(train_df['target'])
train_ids = np.array(train_df.index)
train_df.drop(['ID_code', 'target'], axis=1, inplace=True)

params={
    'max_depth': -1,
    'n_estimators': 1000,
    'early_stopping_rounds': 100,
    'learning_rate': 0.02,
    'colsample_bytree': 0.3,
    'num_leaves': 2,
    'metric': 'auc',
    'objective': 'binary',
    'verbose': -1,
    'device': 'gpu', 
    'gpu_platform_id': 1,
    'gpu_device_id': 0,
    'n_jobs': -1,
}

lgb = LGBMClassifier(**params)

param_grid = {
    'max_depth': [-1, 3, 5],
    'n_estimators': [i for i in range(100, 1000, 100)],
    'early_stopping_rounds': [100, 500, 1000],
    'learning_rate': [0.01, 0.02, 0.05, 0.1],
    'colsample_bytree': [0.3, 0.5, 0.7],
    'num_leaves': [i for i in range(5, 20, 5)],
    'min_data_in_leaf': [i for i in range(100, 1000, 200)],
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

grid_search = GridSearchCV(lgb, param_grid, cv=skf, scoring='roc_auc', verbose=2)

print('\nGrid Search Fitting...')
grid_search.fit(train_df, train_target)

print('\nBest parameters:', grid_search.best_params_)
print('\nBest ROC AUC score:', round(grid_search.best_score_, 4))

best_lgb_model = grid_search.best_estimator_

print('Loading Test Data..')
test_df = pd.read_csv('test.csv')

test_df.drop(['ID_code'], axis=1, inplace=True)

print('\nMake predictions..\n')
lgb_result = best_lgb_model.predict_proba(test_df)[:, 1]

submission = pd.DataFrame()
submission['ID_code'] = test_df.index
submission['target'] = (lgb_result > 0.5).astype(int)

submission.to_csv('lgb_retry.csv', index=False)