import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt

# Train Model

data = pd.read_csv('./train.csv')
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

titanic_input = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].to_numpy()
titanic_target = data['Survived'].to_numpy()

train_input, test_input, train_target, test_target = train_test_split(titanic_input, titanic_target, stratify=titanic_target, random_state=1)

ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

si = SimpleImputer(strategy='mean')
si.fit(train_input)
train_imputed_input = si.transform(train_scaled)
test_imputed_input = si.transform(test_scaled)

dt = DecisionTreeClassifier(max_depth=3)
dt.fit(train_imputed_input, train_target)

print(dt.score(train_imputed_input, train_target))
print(dt.score(test_imputed_input, test_target))

plt.figure(figsize=(16,12))
plot_tree(dt, filled=True, feature_names=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'])
plt.show()