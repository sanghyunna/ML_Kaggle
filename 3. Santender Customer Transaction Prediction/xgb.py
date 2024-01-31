from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
from submit import create_submission_csv

df = pd.read_csv('train.csv')
df.drop(['ID_code'], axis=1, inplace=True)

df_target = df['target']
df_input = df.drop(['target'], axis=1)

ms = MinMaxScaler()
ms.fit(df_input)
input_scaled = ms.transform(df_input)

skf = StratifiedKFold(n_splits=5, shuffle=True)

xgb = XGBClassifier(tree_method='hist', n_estimators=1000, max_depth=5, learning_rate=0.1, n_jobs=-1)

scores = cross_validate(xgb, input_scaled, df_target, cv=skf, return_train_score=True, n_jobs=-1)

xgb.fit(input_scaled, df_target)

print(f"Train accuracy: {np.mean(scores['train_score'])}")
print(f"Test accuracy: {np.mean(scores['test_score'])}")

# print(np.sort(lr.coef_))

# import matplotlib.pyplot as plt
# from xgboost import plot_tree
# plot_tree(xgb)
# plt.show()



# exit()

test_csv = pd.read_csv('test.csv')
create_submission_csv(xgb, test_csv, 'xgb.csv', ms) 