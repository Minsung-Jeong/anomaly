import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
# from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss, accuracy_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split
import catboost as cb
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import os
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


train_y = le.fit_transform(train_df['class'].copy())
train_x = train_df.drop('class', axis=1)

SEEDS = [42, 1028, 1234, 0, 24]

def RF_objective(trial):
    max_depth = trial.suggest_int('max_depth', 1, 10)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 1000)
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    model = RandomForestClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, n_estimators=n_estimators,
                                   n_jobs=2, random_state=seed)
    model.fit(train_x, train_y)

    kfold = StratifiedKFold(n_splits=10)

    score = cross_val_score(model, train_x, train_y, cv=kfold, scoring=make_scorer(f1_score,average='micro'))
    f1_mean = score.mean()
    return f1_mean
# Execute optuna and set hyperparameters
parameter  = []
for seed in SEEDS:
    sampler = TPESampler(seed=seed)
    RF_study = optuna.create_study(direction='maximize', sampler=sampler)
    RF_study.optimize(RF_objective, n_trials=100)
    print("Best Score:", RF_study.best_value)
    print("Best trial:", RF_study.best_trial.params)
    parameter.append(RF_study.best_trial.params)


rfc1 = RandomForestClassifier(**parameter[0], random_state=SEEDS[0])
rfc2 = RandomForestClassifier(**parameter[1], random_state=SEEDS[1])
rfc3 = RandomForestClassifier(**parameter[2], random_state=SEEDS[2])
rfc4 = RandomForestClassifier(**parameter[3], random_state=SEEDS[3])
rfc5 = RandomForestClassifier(**parameter[4], random_state=SEEDS[4])


# optuna
# rfc = RandomForestClassifier(max_depth=6, max_leaf_nodes=268, n_estimators=286,
#                        random_state=42) #rfc1=97.1917
# rfc = RandomForestClassifier(max_depth=5, max_leaf_nodes=364, n_estimators=335, random_state=42) # 100번 튜닝, rfc2 = rfc1

# grid
# rfc = RandomForestClassifier(bootstrap=False, max_features=1, min_samples_leaf=2,
#                        min_samples_split=7, n_estimators=500, random_state=47)



# rfc ensemble

votingC = VotingClassifier(estimators=[('1', rfc1), ('2', rfc2),('3', rfc3), ('4', rfc4),('5', rfc5)], voting='soft', n_jobs=4) #temp8 = 96


# # 단일모델
# rfc.fit(train_x, train_y)
# pred = rfc.predict(test_df)


# 앙상블 모델
votingC = votingC.fit(train_x.astype(int), train_y)
pred = votingC.predict(test_df.astype(int))

label_rev_dic = {}
for i, x in enumerate(label_dic):
    label_rev_dic[i] = x

result = []
for x in pred:
    result.append(label_rev_dic[x])

temp = pd.read_csv('./test.csv')

result_df = pd.DataFrame(result, index=temp['id'], columns=['class'])
result_df.to_csv("./result/model_result_temp_rfc_ensemble.csv")

# rfc(1개) > xgb(temp5와 비교 3개) > lgbm(4개)
# rfc 4개, xgb 26개, lgbm, lgbm 4개
temp1 = pd.read_csv("./result/model_result_temp_rfc_ensemble.csv")
temp2 = pd.read_csv("./result/model_result_temp_rfc.csv")
temp3 = pd.read_csv("./result/model_result_14.csv")
#
check_bool(temp1, temp2)
check_bool(temp2, temp3)
check_bool(temp1, temp3)



