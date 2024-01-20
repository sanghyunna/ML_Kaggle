import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

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
titanic_imputed = si.transform(titanic_scaled)

lr = LogisticRegression()
lr.fit(titanic_imputed, titanic_target)

print(lr.score(titanic_imputed, titanic_target))


# Save Trained Results to CSV

test_data = pd.read_csv('./test.csv')

test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
titanic_test_input = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].to_numpy()

test_scaled = ss.transform(titanic_test_input)
test_imputed_input = si.transform(test_scaled)

predictions = lr.predict(test_imputed_input)

results_df = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})

results_df.to_csv('titanic_results.csv', index=False)