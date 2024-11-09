import pandas as pd

# load data
data = pd.read_csv('src/main/resources/train.csv')

# preprocess data
data['Age'].fillna(data['Age'].mean(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Fare'].fillna(data['Fare'].mean(), inplace=True)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# drop data that isn't relevant
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# target features
X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].copy()
X['Embarked'] = X['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
y = data['Survived'].values