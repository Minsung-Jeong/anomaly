import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

cancer = load_breast_cancer()

data_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
data_df.head()
data_df.info()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=156)



lr_clf = LogisticRegression()

knn_clf = KNeighborsClassifier()


vo_clf = VotingClassifier(estimators=[('LR', lr_clf),('KNN', knn_clf)], voting='soft')
vo_clf.fit(X_train, y_train)
pred = vo_clf.predict(X_test)
acc = accuracy_score(y_test, pred)
print("ensemble accuracy :{0:4f}".format(acc))

classifiers = [lr_clf, knn_clf]
for clf in classifiers:
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    class_name = clf.__class__.__name__
    print("{0}'s accuracy = {1:4f}".format(class_name, acc))

