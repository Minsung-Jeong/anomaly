import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
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

def check_bool(temp1, temp2):
    diff = sum(temp1['class'] != temp2['class'])
    li = []
    for i in range(len(temp1)):
        if temp1['class'][i] != temp2['class'][i]:
            li.append(i)
    print('개수', diff, 'list', li)
    return diff, li

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

# y = le.fit_transform(train_df['class'].copy())
# x = train_df.drop('class', axis=1)
#
# train_size = int(len(x)*0.8)
# train_x = x.iloc[:train_size]
# train_y = y[:train_size]
# test_x = x.iloc[train_size:]
# test_y = y[train_size:]


# train_df.groupby(by='class').count()


# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)

# adaboost
DTC = DecisionTreeClassifier(random_state=7)
ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :[1,2, 3, 4],
              "learning_rate":  [0.000001, 0.00001, 0.001, 0.01]}

adaDTC = AdaBoostClassifier(DTC, random_state=99)
gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsadaDTC.fit(train_x,train_y)
ada_best = gsadaDTC.best_estimator_
print(gsadaDTC.best_score_)


#참고 et_cl = ExtraTreesClassifier(n_estimators=1000, min_samples_leaf=9, min_samples_split=6, max_features=40)
# 결과 : ExtraTreesClassifier(max_features=1, min_samples_split=10, n_estimators=300,
#                      random_state=38)
# 더높은점수결과 : ExtraTreesClassifier(max_features=3, min_samples_split=6, random_state=38)
# rs(이전 버전) : max38(95), mid99, min91
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10, 30, 40],
              "min_samples_split": [2, 6, 10],
              "min_samples_leaf": [1, 3, 9],
              "bootstrap": [False],
              "n_estimators" :[100, 300, 500, 1000],
              "criterion": ["gini"]}

ExtC = ExtraTreesClassifier(random_state=38)
gsExtC = GridSearchCV(ExtC,param_grid=ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsExtC.fit(train_x,train_y)
ExtC_best = gsExtC.best_estimator_
gsExtC.best_score_


# rf_reg = RandomForestRegressor(n_estimators=1000, min_samples_leaf=9, min_samples_split=6, max_features=20)
################## 3. Random Forest
# max 34, mid47(96.56), min48(96.18)
## Search grid for optimal parameters

# # 결과
# RandomForestClassifier(bootstrap=False, max_features=1, min_samples_leaf=2,
#                        min_samples_split=7, n_estimators=500, random_state=47)
# rf_param_grid = {"max_depth": [None],
#               "max_features": range(10),
#               "min_samples_split": range(10),
#               "min_samples_leaf": range(10),
#               "bootstrap": [False],
#               "n_estimators" :[100,300, 500],
#               "criterion": ["gini"]}
#
# RFC = RandomForestClassifier(random_state=47)
# gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
# gsRFC.fit(train_x, train_y)
# RFC_best = gsRFC.best_estimator_
#
# # Best score
# gsRFC.best_score_

##################################svc(95)
svc_param_grid = {'kernel': ['rbf'],
                  'gamma': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
                  'C': range(20,60)}

# random state 상관없이 다 같음
SVMC = SVC(probability=True, random_state=0)
gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsSVMC.fit(train_x,train_y)
SVMC_best = gsSVMC.best_estimator_
print(gsSVMC.best_score_)

#############################################\

ada = AdaBoostClassifier(algorithm='SAMME',
                   base_estimator=DecisionTreeClassifier(random_state=7,
                                                         splitter='random'),
                   learning_rate=1e-05, n_estimators=1, random_state=99)

# extc = ExtraTreesClassifier(max_features=1, min_samples_split=10, n_estimators=300,
#                      random_state=38)
extc = ExtraTreesClassifier(max_features=3, min_samples_split=6, random_state=38)

rfc = RandomForestClassifier(bootstrap=False, max_features=1, min_samples_leaf=2,
                       min_samples_split=7, n_estimators=500, random_state=47)

gnb = GaussianNB(var_smoothing=0.12067926406393285)
svc = SVC(C=23, gamma=0.01, probability=True, random_state=0)
xgb = xgb.XGBClassifier(max_depth = 6,gamma=0,colsample_bytree=1,learning_rate=0.300000012,n_estimators = 100)

# votingC = VotingClassifier(estimators=[('rfc', rfc), ('extc', extc), ('gnb', gnb),('svc', svc)], voting='soft', n_jobs=4) #temp
# votingC = VotingClassifier(estimators=[('rfc', rfc), ('gnb', gnb),('svc', svc)], voting='soft', n_jobs=4) #temp2
# votingC = VotingClassifier(estimators=[('ada', ada),('extc', extc), ('rfc', rfc), ('gnb', gnb),('svc', svc)], voting='soft', n_jobs=4) #temp3
votingC = VotingClassifier(estimators=[('xgb', xgb),('extc', extc), ('rfc', rfc), ('gnb', gnb),('svc', svc)], voting='soft', n_jobs=4) #temp4
votingC = votingC.fit(train_x.astype(int), train_y)

pred = votingC.predict(test_df.astype(int))
# label_dic = get_dict(list(set(train_df['class'])))
label_rev_dic = {}
for i, x in enumerate(label_dic):
    label_rev_dic[i] = x

result = []
for x in pred:
    result.append(label_rev_dic[x])

temp = pd.read_csv('./test.csv')
temp['id']

result_df = pd.DataFrame(result, index=temp['id'], columns=['class'])
result_df.to_csv("./model_result_temp4.csv")


temp1 = pd.read_csv("./model_result_temp4.csv")

temp2 = pd.read_csv("./model_result_temp_max.csv")
temp3 = pd.read_csv("./model_result_14.csv")
#


check_bool(temp1, temp2)
check_bool(temp2, temp3)
check_bool(temp1, temp3)


