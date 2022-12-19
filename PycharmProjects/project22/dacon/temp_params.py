import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from catboost import CatBoostClassifier

import catboost as cb
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.naive_bayes import GaussianNB
import os
from sklearn.preprocessing import LabelEncoder

def get_dict(data):
    dic_x = {}
    for i, x in enumerate((data), start=0):
        dic_x[x] = i
    return dic_x

os.getcwd()
os.chdir("C://data_minsung/dacon")
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')


# 데이터 전처리
snp_col = train_df.columns[5:-1]
for col in snp_col:
    dic = get_dict(list(set(train_df[col])))
    for i in range(len(train_df)):
        train_df[col][i] = dic[train_df[col][i]]
    for i in range(len(test_df)):
        test_df[col][i] = dic[test_df[col][i]]


train_df = train_df.drop(['id','father','mother', 'gender'], axis=1)
test_df = test_df.drop(['id','father','mother', 'gender'], axis=1)

le = LabelEncoder()  # from sklearn.preprocessing
temp = train_df['class'].copy()
label_dic =  {'A': 0, 'B': 1, 'C': 2}
# train_y = train_df['class'].copy()
# for i in range(len(train_y)):
#     train_y[i] = label_dic[train_y[i]]
train_y = le.fit_transform(train_df['class'].copy())
train_x = train_df.drop('class', axis=1)

# train_df.groupby(by='class').count()


# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)


def check_i(data, len):
    for i in range(len):
        if np.median(data) == data[i]:
            midd = i
        if max(data) == data[i]:
            maxx = i
        if min(data) == data[i]:
            minn = i
    return maxx, midd, minn



################## 3. Random Forest
# max 34, mid47(96.56), min48(96.18)
## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}

RFC = RandomForestClassifier(random_state=i)
gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsRFC.fit(train_x, train_y)
RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_


######################### 4. Gradient boosting tunning(성능 안 좋음) - acc 77나옴

gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1]
              }
#randomstate :98 => 0.8396
for i in range(100):
GBC = GradientBoostingClassifier(random_state=i)
gsGBC = GridSearchCV(GBC, param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsGBC.fit(train_x,train_y)
GBC_best = gsGBC.best_estimator_
# gb_rs.append(i)
# gb_score.append(gsGBC.best_score_)
# Best score
gsGBC.best_score_


#################### 5. SVC classifier
svc_param_grid = {'kernel': ['rbf'],
                  'gamma': [0.0001, 0.001, 0.01, 0.1],
                  'C': [50, 100,200,300, 1000]}

# random state 상관없이 다 같음
SVMC = SVC(probability=True, random_state=0)
gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsSVMC.fit(train_x,train_y)
SVMC_best = gsSVMC.best_estimator_
gsSVMC.best_score_

##################### 6. MLP Classifier
#random_state : 33, 56 => 93.148
mlp_param_grid = {'max_iter' : [300,400,500],
                  'hidden_layer_sizes' : [32, 64, 128, 256],
                    'alpha': 10.0 ** -np.arange(3, 8)
                  }
MLP = MLPClassifier(random_state=33)
gsMLP = GridSearchCV(MLP, param_grid=mlp_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
gsMLP.fit(train_x, train_y)
MLP_best = gsMLP.best_estimator_
gsMLP.best_score_
##################### 7. Logistic Regression(92.39)
lr_param_grid = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}

LR = LogisticRegression()
gsLR = GridSearchCV(LR, param_grid=lr_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
gsLR.fit(train_x,train_y)
LR_best = gsLR.best_estimator_
gsLR.best_score_


####################### 9. KNeighborsClassifier
KC = KNeighborsClassifier()
kc_param_grid = {'n_neighbors':list(range(2,30))}
gsKC = GridSearchCV(KC, param_grid=kc_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
gsKC.fit(train_x, train_y)
KC_best = gsKC.best_estimator_
gsKC.best_score_

######################## 10. Gaussian Naive Bayes
GB = GaussianNB()
gb_param_grid = {'var_smoothing': np.logspace(0,-9, num=50)}
gsGB = GridSearchCV(GB, param_grid=gb_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1)
gsGB.fit(train_x, train_y)
GB_best = gsGB.best_estimator_
gsGB.best_score_


