import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
# from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
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

import optuna
from optuna import Trial
from optuna.samplers import TPESampler

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
# kfold = StratifiedKFold(n_splits=10)
#
# # adaboost
# DTC = DecisionTreeClassifier(random_state=7)
# ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
#               "base_estimator__splitter" :   ["best", "random"],
#               "algorithm" : ["SAMME","SAMME.R"],
#               "n_estimators" :[1,2, 3, 4],
#               "learning_rate":  [0.000001, 0.00001, 0.001, 0.01]}
#
# adaDTC = AdaBoostClassifier(DTC, random_state=99)
# gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
# gsadaDTC.fit(train_x,train_y)
# ada_best = gsadaDTC.best_estimator_
# print(gsadaDTC.best_score_)


#참고 et_cl = ExtraTreesClassifier(n_estimators=1000, min_samples_leaf=9, min_samples_split=6, max_features=40)
# 결과 : ExtraTreesClassifier(max_features=1, min_samples_split=10, n_estimators=300,
#                      random_state=38)
# 더높은점수결과 : ExtraTreesClassifier(max_features=3, min_samples_split=6, random_state=38)
# rs(이전 버전) : max38(95), mid99, min91
# ex_param_grid = {"max_depth": [None],
#               "max_features": [1, 3, 10, 30, 40],
#               "min_samples_split": [2, 6, 10],
#               "min_samples_leaf": [1, 3, 9],
#               "bootstrap": [False],
#               "n_estimators" :[100, 300, 500, 1000],
#               "criterion": ["gini"]}
#
# ExtC = ExtraTreesClassifier(random_state=38)
# gsExtC = GridSearchCV(ExtC,param_grid=ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
# gsExtC.fit(train_x,train_y)
# ExtC_best = gsExtC.best_estimator_
# gsExtC.best_score_


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
# svc_param_grid = {'kernel': ['rbf'],
#                   'gamma': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
#                   'C': range(20,60)}
#
# # random state 상관없이 다 같음
# SVMC = SVC(probability=True, random_state=0)
# gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
# gsSVMC.fit(train_x,train_y)
# SVMC_best = gsSVMC.best_estimator_
# print(gsSVMC.best_score_)

#############################################\

# ada = AdaBoostClassifier(algorithm='SAMME',
#                    base_estimator=DecisionTreeClassifier(random_state=7,
#                                                          splitter='random'),
#                    learning_rate=1e-05, n_estimators=1, random_state=99)

# extc = ExtraTreesClassifier(max_features=1, min_samples_split=10, n_estimators=300,
#                      random_state=38)

extc = ExtraTreesClassifier(max_features=3, min_samples_split=6, random_state=42)
rfc = RandomForestClassifier(bootstrap=False, max_features=1, min_samples_leaf=2,
                       min_samples_split=7, n_estimators=500, random_state=47)
gnb = GaussianNB(var_smoothing=0.12067926406393285)
svc = SVC(C=23, gamma=0.01, probability=True, random_state=0)
xgbc = xgb.XGBClassifier(objective='multi:softprob', random_state=10, var_smoothing=1.0)

# XGboost 하이퍼파라미터 튜닝
def objective(trial):
    X_train, X_valid, Y_train, Y_valid = train_test_split(train_x.astype(int), train_y.astype(int), test_size=0.2)

    dtrain = xgb.DMatrix(X_train, label=Y_train)
    dtest = xgb.DMatrix(X_valid, label=Y_valid)
    param = {
        "random_state": 42,
        "verbosity": 0,
        "objective": "multi:softmax",
        "num_class": 5,
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "early_stopping_rounds":200,
        "evals":([X_valid,Y_valid])
    }

    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    fitted_model = xgb.train(param, dtrain)
    preds = fitted_model.predict(dtest)
    xgb_pred = np.rint(preds)
    accuracy = accuracy_score(Y_valid, xgb_pred)
    # logloss = log_loss(Y_valid, xgb_pred)
    return accuracy
    # return logloss

sampler = TPESampler(seed=42)
study = optuna.create_study(
    study_name="xgb_parameter_opt",
    direction="maximize",
    # direction="minimize",
    sampler=sampler,
)
study.optimize(objective, n_trials=100)
print("Best Score:", study.best_value)
print("Best trial:", study.best_trial.params)

xgbc = xgb.XGBClassifier(**study.best_trial.params)

# Light gbm 하이퍼파라미터 튜닝
def objective(trial: Trial) -> float:
    params_lgb = {
        "random_state": 42,
        "verbosity": -1,
        "learning_rate": 0.05,
        "n_estimators": 10000,
        "objective": "multiclass",
        "metric": "multi_logloss",
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 3e-5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 9e-2),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "max_bin": trial.suggest_int("max_bin", 200, 500),
    }

    X_train, X_valid, Y_train, Y_valid = train_test_split(train_x.astype(int), train_y.astype(int), test_size=0.2)

    model = LGBMClassifier(**params_lgb)
    model.fit(
        X_train,
        Y_train,
        eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
        early_stopping_rounds=100,
        verbose=False,
    )

    # lgb_pred = model.predict_proba(X_valid)
    log_score = log_loss(Y_valid, lgb_pred)
    accuracy = accuracy_score(Y_valid, lgb_pred)
    return accuracy

sampler = TPESampler(seed=42)
study = optuna.create_study(
    study_name="lgbm_parameter_opt",
    direction="minimize",
    # direction="maximize",
    sampler=sampler,
)
study.optimize(objective, n_trials=100)
print("Best Score:", study.best_value)
print("Best trial:", study.best_trial.params)

lgbmc = LGBMClassifier(**study.best_trial.params)



# kc
kc = KNeighborsClassifier(n_neighbors=7)

# votingC = VotingClassifier(estimators=[('rfc', rfc), ('extc', extc), ('gnb', gnb),('svc', svc)], voting='soft', n_jobs=4) #temp
# votingC = VotingClassifier(estimators=[('rfc', rfc), ('gnb', gnb),('svc', svc)], voting='soft', n_jobs=4) #temp2
# votingC = VotingClassifier(estimators=[('ada', ada),('extc', extc), ('rfc', rfc), ('gnb', gnb),('svc', svc)], voting='soft', n_jobs=4) #temp3
# votingC = VotingClassifier(estimators=[('xgb', xgbc),('extc', extc), ('rfc', rfc), ('gnb', gnb),('svc', svc)], voting='soft', n_jobs=4) #temp4
# votingC = VotingClassifier(estimators=[('xgb', xgbc),('extc', extc), ('rfc', rfc), ('gnb', gnb)], voting='soft', n_jobs=4) #temp5(best) 1은 이전 xgb, 2은  나중 xgb
# votingC = VotingClassifier(estimators=[('extc', extc), ('rfc', rfc),('gnb', gnb)], voting='soft', n_jobs=4) #temp6(=temp5)
# votingC = VotingClassifier(estimators=[('extc', extc), ('rfc', rfc),('gnb', gnb), ('lgbm', lgbmc), ('xgbc', xgbc)], voting='soft', n_jobs=4) #temp7

# optuna 통한 lgb, xgb 기반, [rfc, extc, svc]는 하던대로 > 다음 버전은 rfc, extx도 ouptuna로 튜닝해서 진행
# 8은 result14랑 1개 차이, 13은 result14랑 3개 차이이지만 점수는 같음
votingC = VotingClassifier(estimators=[('lgbm', lgbmc), ('xgbc', xgbc)], voting='soft', n_jobs=4) #temp8 = 96
votingC = VotingClassifier(estimators=[('lgbm', lgbmc), ('xgbc', xgbc), ('extc', extc)], voting='soft', n_jobs=4) #temp9
votingC = VotingClassifier(estimators=[('lgbm', lgbmc), ('xgbc', xgbc), ('rfc', rfc)], voting='soft', n_jobs=4) #temp10
votingC = VotingClassifier(estimators=[('lgbm', lgbmc), ('xgbc', xgbc), ('rfc', rfc), ('extc', extc)], voting='soft', n_jobs=4) #temp11 - 여기까지
votingC = VotingClassifier(estimators=[('lgbm', lgbmc), ('xgbc', xgbc), ('rfc', rfc), ('extc', extc), ('gnb', gnb)], voting='soft', n_jobs=4) #temp12 = 3개가 다르지만 점수는 같음
votingC = VotingClassifier(estimators=[('lgbm', lgbmc), ('xgbc', xgbc), ('gnb', gnb)], voting='soft', n_jobs=4) #temp13
votingC = VotingClassifier(estimators=[('lgbm', lgbmc), ('xgbc', xgbc), ('svc', svc)], voting='soft', n_jobs=4) #temp14


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
result_df.to_csv("./model_result_temp14.csv")


temp1 = pd.read_csv("./model_result_temp13.csv")
temp2 = pd.read_csv("./model_result_temp_max.csv")
temp3 = pd.read_csv("./model_result_14.csv")
#
check_bool(temp1, temp2)
check_bool(temp2, temp3)
check_bool(temp1, temp3)


