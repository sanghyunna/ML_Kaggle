import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split, cross_validate
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform

# Train Model

data = pd.read_csv('./train.csv')
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# mean_embarked = data.groupby('Embarked')['Survived'].mean()
# data['MeanEmbarked'] = data['Embarked'].map(mean_embarked)

# mean_pclass = data.groupby('Pclass')['Survived'].mean()
# data['MeanPclass'] = data['Pclass'].map(mean_pclass)

titanic_input = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].to_numpy()
titanic_target = data['Survived'].to_numpy()

# ss = StandardScaler()
# ss.fit(titanic_input)
# titanic_scaled = ss.transform(titanic_input)

si = SimpleImputer(strategy='mean')
si.fit(titanic_input)
titanic_imputed_input = si.transform(titanic_input)


lgb = LGBMClassifier()
params={
    'num_leaves': randint(10, 100),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'n_estimators': randint(100, 500),
    'colsample_bytree': uniform(0.6, 1.0),
    'subsample': uniform(0.6, 1.0),
    'reg_alpha': uniform(0, 0.5),
    'reg_lambda': uniform(0, 0.5),
}

rs = RandomizedSearchCV(lgb, params, n_iter=50, scoring='accuracy', cv=5, n_jobs=-1)
rs.fit(titanic_imputed_input, titanic_target)

best_lgb = rs.best_estimator_
best_lgb.fit(titanic_imputed_input, titanic_target)

print(best_lgb.score(titanic_imputed_input, titanic_target))

# print('MeanPclass', 'Sex', '  Age', '  SibSp', 'Parch', 'Fare','MeanEmbarked')
# np.set_printoptions(threshold=np.inf, precision=3, suppress=True)
# print(titanic_imputed_input)
# exit()

# Save Trained Results to CSV

test_data = pd.read_csv('./test.csv')

test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})

# test_data['MeanEmbarked'] = test_data['Embarked'].map(mean_embarked)
# test_data['MeanPclass'] = data['Pclass'].map(mean_pclass)

titanic_test_input = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].to_numpy()

# test_scaled = ss.transform(titanic_test_input)

test_imputed_input = si.transform(titanic_test_input)

predictions = best_lgb.predict(test_imputed_input)

results_df = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})

results_df.to_csv('titanic_results_lgbm_rand.csv', index=False)