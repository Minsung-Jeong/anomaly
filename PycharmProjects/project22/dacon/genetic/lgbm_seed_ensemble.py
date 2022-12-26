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
def objective(trial: Trial) -> float:
    params_lgb = {
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

    X_train, X_valid, Y_train, Y_valid = train_test_split(train_x.astype(int), train_y.astype(int), test_size=0.2, random_state=seed)

    model = LGBMClassifier(**params_lgb, random_state=42)

    model.fit(
        X_train,
        Y_train,
        eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
        early_stopping_rounds=200
    )

    # log-score
    # lgb_pred = model.predict_proba(X_valid)
    # log_score = log_loss(Y_valid, lgb_pred)

    # f1-score
    lgb_pred = model.predict(X_valid)
    f1 = f1_score(Y_valid, lgb_pred, average='micro')
    # return log_score
    return f1

parameters = []
for seed in SEEDS:
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        study_name="lgbm_parameter_opt",
        # direction="minimize",
        direction="maximize",
        sampler=sampler,
    )
    study.optimize(objective, n_trials=100)
    print("Best Score:", study.best_value)
    print("Best trial:", study.best_trial.params)
    parameters.append(study.best_trial.params)


lgmbc1 = LGBMClassifier(**parameters[0], random_state=SEEDS[0])
lgmbc2 = LGBMClassifier(**parameters[1], random_state=SEEDS[1])
lgmbc3 = LGBMClassifier(**parameters[2], random_state=SEEDS[2])
lgmbc4 = LGBMClassifier(**parameters[3], random_state=SEEDS[3])
lgmbc5= LGBMClassifier(**parameters[4], random_state=SEEDS[4])

votingC = VotingClassifier(estimators=[('1', lgmbc1), ('2', lgmbc2),('3', lgmbc3), ('4', lgmbc4),('5', lgmbc5)], voting='soft', n_jobs=4) #temp8 = 96





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
result_df.to_csv("./result/model_result_temp_lgbm_ensemble.csv")

# rfc(1개) > xgb(temp5와 비교 3개) > lgbm(4개)
# rfc 4개, xgb 26개, lgbm, lgbm 4개
temp1 = pd.read_csv("./result/model_result_temp_lgbm_ensemble.csv")
temp2 = pd.read_csv("./result/model_result_temp_rfc.csv")
temp3 = pd.read_csv("./result/model_result_14.csv")
#
check_bool(temp1, temp2)
check_bool(temp2, temp3)
check_bool(temp1, temp3)



