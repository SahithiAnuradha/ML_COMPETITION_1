import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#Acquire data
train_df = pd.read_csv(r'C:\Users\psahi\OneDrive\Desktop\ASSIGNMENT\ML\ML_HW1\train.csv')
test_df = pd.read_csv(r'C:\Users\psahi\OneDrive\Desktop\ASSIGNMENT\ML\ML_HW1\test.csv')
test_ids = test_df["PassengerId"]
combine = [train_df, test_df]

#Features available in the dataset
print(train_df.columns.values)

# preview the data
print(train_df.head())
print(train_df.tail())
print('end')

#Datatypes of features and finding the features with null values
print(train_df.info())
print('_'*40)
print(test_df.info())

#Numerical
print(train_df.describe())

#Categorical
print(train_df.describe(include=['O']))

print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print(train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print(train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))

print(train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


# grid = sns.FacetGrid(train_df, col='Embarked', hue='Survived', palette={0: 'k', 1: 'w'})
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()

print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

#Changing Cabin value to numerical in training data
train_df['Cabin'] = train_df['Cabin'].str[0]
train_df['Cabin'] = train_df['Cabin'].fillna(0)
train_df['Cabin'] = train_df['Cabin'].replace('A', 1)
train_df['Cabin'] = train_df['Cabin'].replace('B', 2)
train_df['Cabin'] = train_df['Cabin'].replace('C', 3)
train_df['Cabin'] = train_df['Cabin'].replace('D', 4)
train_df['Cabin'] = train_df['Cabin'].replace('E', 5)
train_df['Cabin'] = train_df['Cabin'].replace('F', 6)
train_df['Cabin'] = train_df['Cabin'].replace('G', 7)
train_df['Cabin'] = train_df['Cabin'].replace('T', 8)
print(train_df['Cabin'])

#Changing Cabin value to numerical in testing data
test_df['Cabin'] = test_df['Cabin'].str[0]
test_df['Cabin'] = test_df['Cabin'].fillna(0)
test_df['Cabin'] = test_df['Cabin'].replace('A', 1)
test_df['Cabin'] = test_df['Cabin'].replace('B', 2)
test_df['Cabin'] = test_df['Cabin'].replace('C', 3)
test_df['Cabin'] = test_df['Cabin'].replace('D', 4)
test_df['Cabin'] = test_df['Cabin'].replace('E', 5)
test_df['Cabin'] = test_df['Cabin'].replace('F', 6)
test_df['Cabin'] = test_df['Cabin'].replace('G', 7)
test_df['Cabin'] = test_df['Cabin'].replace('T', 8)
print(test_df['Cabin'])

#Removing unnecessary features

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

#Changing Cabin value to numerical
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

print(pd.crosstab(train_df['Title'], train_df['Sex']))

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

print(train_df.head())

#Removing unnecessary features

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
print(train_df.shape, test_df.shape)

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

print(train_df.head())

grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

guess_ages = np.zeros((2,3))
print(guess_ages)

#Removing and filling null values
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

print(train_df.head())

#Categorizing Age data into numeric values

train_df['AgeBand'] = pd.cut(train_df['Age'], 9)
print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True))

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 5, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 5) & (dataset['Age'] <= 15), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 15) & (dataset['Age'] <= 20), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 25), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 35), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 45), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 45) & (dataset['Age'] <= 55), 'Age'] = 6
    dataset.loc[(dataset['Age'] > 55) & (dataset['Age'] <= 65), 'Age'] = 7
    dataset.loc[ dataset['Age'] > 65, 'Age']
print(train_df.head())

train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
print(train_df.head())

#Categorizing family members into a single feature

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))


for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

train_df = train_df.drop(['Parch'], axis=1)
test_df = test_df.drop(['Parch'], axis=1)
combine = [train_df, test_df]

print(train_df.head())

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))

#Filling null values in Embarked 

for dataset in combine:
   dataset['Embarked'] = dataset['Embarked'].fillna('S')
    
print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))


for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#Changing the Fare details into numeric values

print(train_df.head())

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
print(test_df.head())

train_df['FareBand'] = pd.qcut(train_df['Fare'], 2)
print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True))

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 8.662, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 8.662) & (dataset['Fare'] <= 26), 'Fare'] = 1
    dataset.loc[ dataset['Fare'] > 26, 'Fare'] = 2
    
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

print(train_df.head(10))

print(test_df.head(10))


#Model

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
print(X_train.shape, Y_train.shape, X_test.shape)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print("LogisticRegression")
print(acc_log)

#Submission CSV file
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
test_data=test_df.drop("PassengerId", axis=1).copy()
prediction=logreg.predict(test_data)
submission=pd.DataFrame({
    "PassengerID":test_df["PassengerId"],
    "Survived":prediction
    })

submission.to_csv('submission_old1.csv', index=False)
submission=pd.read_csv('submission_old1.csv')

# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print("Support Vector Machine")
print(acc_svc)

#KNN

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print("KNN")
print(acc_knn)

#Gaussian

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print("Gaussian NB")
print(acc_gaussian)

#Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print("Perception")
print(acc_perceptron)

#Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print("Linear SVC")
print(acc_linear_svc)

#SGD

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print("SGD")
print(acc_sgd)

#Decission Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print("Decision Tree")
print(acc_decision_tree)

#Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print("Random Forest")
print(acc_random_forest)

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
print(models.sort_values(by='Score', ascending=False))








