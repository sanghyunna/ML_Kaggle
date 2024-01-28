import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint

# Train Model

data = pd.read_csv('./train.csv')
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

titanic_input = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].to_numpy()
titanic_target = data['Survived'].to_numpy()

ss = StandardScaler()
ss.fit(titanic_input)
titanic_scaled = ss.transform(titanic_input)

si = SimpleImputer(strategy='mean')
si.fit(titanic_scaled)
titanic_imputed_input = si.transform(titanic_scaled)

dt = DecisionTreeClassifier(random_state=1)

param_dist = {
    'max_depth': randint(1, 10),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(dt, param_dist, cv=5, n_iter=100, random_state=1)

random_search.fit(titanic_imputed_input, titanic_target)

print(random_search.best_score_)
print(random_search.best_params_)

test_data = pd.read_csv('./test.csv')

test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
titanic_test_input = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].to_numpy()

test_scaled = ss.transform(titanic_test_input)
test_imputed_input = si.transform(test_scaled)

predictions = random_search.best_estimator_.predict(test_imputed_input)
results_df = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})

results_df.to_csv('titanic_results_rand_dt.csv', index=False)