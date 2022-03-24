from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz

import graphviz
import numpy as np
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# DecisionTree 생성
dt_clf = DecisionTreeClassifier(random_state=156)

# 데이터 로딩 및 데이터 세트 분리
iris_data = load_iris()

iris_data.keys()
sum(np.isnan(iris_data.data))

X_train, X_test, Y_train, Y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2, random_state=11)

# DecisionTreeClassifier 학습
dt_clf.fit(X_train, Y_train)

# graphviz 생성하고 저장 후 불러오기
export_graphviz(dt_clf, out_file="tree.dot", class_names=iris_data.target_names, feature_names=iris_data.feature_names, impurity=True, filled=True)

with open("tree.dot") as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)


# feature importance 추출
print("변수중요도\n{0}".format(np.round(dt_clf.feature_importances_, 3)))
for name, value in zip(iris_data.feature_names, dt_clf.feature_importances_):
    print("{0} : {1:4f}".format(name, value))

sns.barplot(x=dt_clf.feature_importances_, y=iris_data.feature_names)