from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from submit import create_submission_csv

df = pd.read_csv('train.csv')
df.drop(['ID_code'], axis=1, inplace=True)

df_target = df['target']
df_input = df.drop(['target'], axis=1)

ss = StandardScaler()
ss.fit(df_input)
input_scaled = ss.transform(df_input)

skf = StratifiedKFold(n_splits=5, shuffle=True)

lr = LogisticRegression()
lr.fit(input_scaled, df_target)

cv_results = cross_validate(lr, input_scaled, df_target, cv=skf, scoring=['accuracy'], return_train_score=True)

print(f"Train accuracy: {cv_results['train_accuracy'].mean()}")
print(f"Test accuracy: {cv_results['test_accuracy'].mean()}")

# print(np.sort(lr.coef_))

test_csv = pd.read_csv('test.csv')
create_submission_csv(lr, test_csv, 'logis.csv', ss)