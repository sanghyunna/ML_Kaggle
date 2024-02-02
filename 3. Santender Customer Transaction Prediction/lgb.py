from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from lightgbm import LGBMClassifier, plot_importance
import matplotlib.pyplot as plt
from submit import create_submission_csv
import pandas as pd
import numpy as np
import joblib
import time
import os

def shuffle_feature(input_data, feature_index):
    shuffled_data = input_data.copy()
    np.random.shuffle(shuffled_data[:, feature_index])
    return shuffled_data

def print_feature_importance(model, input_data, target):
    original_accuracy = accuracy_score(target, model.predict(input_data))
    print("Original Accuracy:", original_accuracy)

    feature_contributions = {}
    for feature_index in range(input_data.shape[1]):
        print(f"[{feature_index}] ",end="")
        shuffled_input = shuffle_feature(input_data, feature_index)
        shuffled_accuracy = accuracy_score(target, model.predict(shuffled_input))
        contribution = original_accuracy - shuffled_accuracy
        feature_contributions[feature_index] = contribution

    sorted_contributions = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)

    print("Feature Importance based on Contribution to Accuracy:")
    for feature_index, contribution in sorted_contributions:
        print(f"Feature {feature_index}: {contribution}")

df = pd.read_csv('train.csv')
df.drop(['ID_code'], axis=1, inplace=True)
df.drop(['var_17'], axis=1, inplace=True)
df.drop(['var_14'], axis=1, inplace=True)
df.drop(['var_61'], axis=1, inplace=True)
df.drop(['var_38'], axis=1, inplace=True)
df.drop(['var_161'], axis=1, inplace=True)
df.drop(['var_185'], axis=1, inplace=True)
df.drop(['var_117'], axis=1, inplace=True)

df_target = df['target']
df_input = df.drop(['target'], axis=1)

ms = MinMaxScaler()
ms.fit(df_input)
input_scaled = ms.transform(df_input)

# start = time.time()
print("Starting..")

skf = StratifiedKFold(n_splits=5, shuffle=True)

params = {
    'n_estimators': 500,
    'max_depth': -1,
    'learning_rate': 0.05,
    'num_leaves': 50,
    'n_jobs': -1,
    'device': 'gpu',
    'gpu_platform_id': 1,
    'gpu_device_id': 0,
    'verbose': -1,
}

lgb_model_path = 'model.pkl'

if os.path.exists(lgb_model_path):
    lgb = joblib.load(lgb_model_path)
else:
    lgb = LGBMClassifier(**params)
    lgb.fit(input_scaled, df_target)
    joblib.dump(lgb, lgb_model_path)

# print_feature_importance(lgb, input_scaled, df_target)

scores = cross_validate(lgb, input_scaled, df_target, cv=skf, return_train_score=True, n_jobs=-1)

# end = time.time()
# print(f"Time taken: {end-start:.2f}sec")

print(f"Train accuracy: {np.mean(scores['train_score'])}")
print(f"Test accuracy: {np.mean(scores['test_score'])}")

# print(np.sort(lr.coef_))

# exit()

test_csv = pd.read_csv('test.csv')
create_submission_csv(lgb, test_csv, 'lgb_featshuf.csv', ms)
