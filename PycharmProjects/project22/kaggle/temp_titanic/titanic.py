import pandas as pd
import numpy as np
import statistics
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

train = pd.read_csv("C:/data_minsung/kaggle/titanic/train.csv")
test = pd.read_csv("C:/data_minsung/kaggle/titanic/test.csv")
test_label = pd.read_csv("C:/data_minsung/kaggle/titanic/gender_submission.csv")

print('aa')
"""
train
"""
# data check
train.info()
test.info()

# category to int : male = 1, female = 0
train["Sex"][train["Sex"]=='male'] = 1
train["Sex"][train["Sex"]=='female'] = 0
train["Sex"].fillna(statistics.mode(train["Sex"]), inplace=True) # data to int64 : why?


# fillna with mode value : Embarked (C,Q,S)
temp_li = list(set(train["Embarked"]))
temp_li = [x for x in temp_li if str(x) != 'nan']

for i in range(len(temp_li)):
    train["Embarked"][train["Embarked"]==temp_li[i]] = i
train["Embarked"].fillna(statistics.mode(train["Embarked"]), inplace=True)

# fillna with average value : Age
train["Age"].fillna(train["Age"].mean(), inplace=True)

# drop data(smaller than the half of train data)
train.drop(["Name", "Cabin", "Ticket", "PassengerId"], axis=1, inplace=True)

X_train = train.drop(["Survived"], axis=1)
y_train = train["Survived"]

# data visualization
sns.countplot(x='Survived', data=train, palette='pastel')
sns.distplot(train["Survived"])
plt.show()

sns.distplot(train["Age"])
plt.show()

# More woman were prone to survive
sns.countplot(data=train, x = 'Survived', hue='Sex').set(xticklabels = ['Died', 'Survied'])
plt.show()

# correlation matrix
corr = train.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True)


# Oversampling(SMOTE) for imbalanced data -> It causes Over-fitting
# smote = SMOTE(random_state=42)
# X_train, y_train = smote.fit_resample(X_train, y_train)

# standardize
std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)



"""
test
"""
test.drop(["Name", "Cabin", "Ticket"], axis=1, inplace=True)
# male = 1, female = 0
test["Sex"][test["Sex"]=='male'] = int(1)
test["Sex"][test["Sex"]=='female'] = int(0)
test["Sex"].fillna(statistics.mode(test["Sex"]), inplace=True)

# Embarked 해결
for i in range(len(temp_li)):
    test["Embarked"][test["Embarked"]==temp_li[i]] = i
test["Embarked"] = test["Embarked"].fillna(statistics.mode(test["Embarked"]))
test["Age"].fillna(test["Age"].mean(), inplace=True)

# fillna with mean value : "Fare"
test["Fare"].fillna(test["Fare"].mean(), inplace=True)

std_scaler = StandardScaler()
X_test = std_scaler.fit_transform(test.drop(["PassengerId"], axis=1))

# PCA on data
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
X_train.shape
pca = PCA(random_state=0)
pca.fit(X_train)
var_cumu = np.cumsum(pca.explained_variance_ratio_)

incre_pca = IncrementalPCA(n_components=6)

train_pca = incre_pca.fit_transform(X_train)
test_pca = incre_pca.fit_transform(X_test)

# Gaussian Naive Bayes - pca 사용시 성능악화
from sklearn.naive_bayes import GaussianNB
gaus_clf = GaussianNB( )
gaus_clf.fit(X_train, y_train)
gaus_preds = gaus_clf.predict(X_test)

#  Decision Tree Classifier
DTC = DecisionTreeClassifier(max_features=6, max_depth=5)
DTC.fit(X_train, y_train)

DTC_preds = DTC.predict(X_test)
preds_df = pd.DataFrame(DTC_preds, columns=["Survived"])
preds_df["PassengerId"] = test["PassengerId"]
preds_df = preds_df[["PassengerId", "Survived"]]


# logistic regression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
clf_preds = clf.predict(X_test)

preds_df = pd.DataFrame(clf_preds, columns=["Survived"])
preds_df["PassengerId"] = test["PassengerId"]
preds_df = preds_df[["PassengerId", "Survived"]]
preds_df.to_csv("C://data_minsung/kaggle/titanic/result_logis.csv", index=False)

# evaluate
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,plot_roc_curve,accuracy_score,recall_score
result1= pd.read_csv("C:/data_minsung/kaggle/titanic/result.csv").iloc[:,0]
accuracy_score( test_label.values[:,1], gaus_preds)



