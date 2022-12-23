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
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
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
# train_y = train_df['class'].copy()
# for i in range(len(train_y)):
#     train_y[i] = label_dic[train_y[i]]


train_y = le.fit_transform(train_df['class'].copy())
train_x = train_df.drop('class', axis=1)

extc = ExtraTreesClassifier(max_features=3, min_samples_split=6, random_state=42)
rfc = RandomForestClassifier(bootstrap=False, max_features=1, min_samples_leaf=2,
                       min_samples_split=7, n_estimators=500, random_state=47)
gnb = GaussianNB(var_smoothing=0.12067926406393285)
svc = SVC(C=23, gamma=0.01, probability=True, random_state=0)
xgbc = xgb.XGBClassifier(objective='multi:softprob', random_state=10, var_smoothing=1.0)

# randomforest 하이퍼파라미터 튜닝
def RF_objective(trial):
    max_depth = trial.suggest_int('max_depth', 1, 10)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 1000)
    n_estimators = trial.suggest_int('n_estimators', 100, 500)
    model = RandomForestClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, n_estimators=n_estimators,
                                   n_jobs=2, random_state=42)
    model.fit(train_x, train_y)
    kfold = StratifiedKFold(n_splits=5)
    score = cross_val_score(model, train_x, train_y, cv=kfold, scoring=make_scorer(f1_score,average='micro'))
    f1_mean = score.mean()
    return f1_mean
# Execute optuna and set hyperparameters
sampler = TPESampler(seed=42)
RF_study = optuna.create_study(direction='maximize', sampler=sampler)
RF_study.optimize(RF_objective, n_trials=1000)
print("Best Score:", RF_study.best_value)   
print("Best trial:", RF_study.best_trial.params)

# RandomForestClassifier(max_depth=5, max_leaf_nodes=786, n_estimators=180, random_state=42) #10번
# RandomForestClassifier(max_depth=5, max_leaf_nodes=364, n_estimators=335, random_state=42) # 100번
# rfc = RandomForestClassifier(max_depth=6, max_leaf_nodes=268, n_estimators=286, random_state=42) # 95.82
# {'max_depth': 6, 'max_leaf_nodes': 157, 'n_estimators': 162} # 95.43
rfc = RandomForestClassifier(**RF_study.best_trial.params)

# XGboost 하이퍼파라미터 튜닝
def xgb_objective(trial):
    X_train, X_valid, Y_train, Y_valid = train_test_split(train_x.astype(int), train_y.astype(int), test_size=0.2, random_state=42)

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

    # acc 기준 학습
    # fitted_model = xgb.train(param, dtrain)
    # preds = fitted_model.predict(dtest)
    # xgb_pred = np.rint(preds)
    # accuracy = accuracy_score(Y_valid, xgb_pred)

    # f1-score 기준 학습
    fitted_model = xgb.train(param, dtrain)
    preds = fitted_model.predict(dtest)
    # xgb_pred = np.rint(preds)
    f1 = f1_score(Y_valid, preds, average='micro')
    # return accuracy
    return f1


sampler = TPESampler(seed=42)
study = optuna.create_study(
    study_name="xgb_parameter_opt",
    direction="maximize",
    # direction="minimize",
    sampler=sampler,
)
study.optimize(xgb_objective, n_trials=1000)
print("Best Score:", study.best_value)
print("Best trial:", study.best_trial.params)

xgbc = xgb.XGBClassifier(**study.best_trial.params)
# 100
# XGBClassifier(base_score=None, booster=None, callbacks=None,
#               colsample_bylevel=None, colsample_bynode=None,
#               colsample_bytree=None, early_stopping_rounds=None,
#               enable_categorical=False, eval_metric=None, feature_types=None,
#               gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
#               interaction_constraints=None, learning_rate=None, max_bin=None,
#               max_cat_threshold=None, max_cat_to_onehot=None,
#               max_delta_step=None, max_depth=None, max_leaves=None,
#               min_child_weight=None, missing=nan, monotone_constraints=None,
#               n_estimators=100, n_jobs=None, num_parallel_tree=None,
#               objective='multi:softprob', predictor=None, ...)


# Light gbm 하이퍼파라미터 튜닝
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

    X_train, X_valid, Y_train, Y_valid = train_test_split(train_x.astype(int), train_y.astype(int), test_size=0.2, random_state=42)

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

sampler = TPESampler(seed=42)
study = optuna.create_study(
    study_name="lgbm_parameter_opt",
    # direction="minimize",
    direction="maximize",
    sampler=sampler,
)
study.optimize(objective, n_trials=100)
print("Best Score:", study.best_value)
print("Best trial:", study.best_trial.params)

# 둘 다 1
# LGBMClassifier(colsample_bytree=0.9427779694979184, max_bin=280, max_depth=8,
#                min_child_samples=11, num_leaves=208,
#                reg_alpha=2.6609721865173934e-05, reg_lambda=0.02820657881767179,
#                subsample=0.4489768874984188, subsample_freq=3)
# LGBMClassifier(colsample_bytree=0.851261132978881, max_bin=446, max_depth=13,
#                min_child_samples=19, num_leaves=169,
#                reg_alpha=2.2714647832604017e-06, reg_lambda=0.0671564944943035,
#                subsample=0.6846758987443338, subsample_freq=9)

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
votingC = VotingClassifier(estimators=[('lgbm', lgbmc), ('xgbc', xgbc), ('rfc', rfc), ('gnb', gnb)], voting='soft', n_jobs=4) #temp12-1
votingC = VotingClassifier(estimators=[('lgbm', lgbmc), ('rfc', rfc), ('gnb', gnb)], voting='soft', n_jobs=4) #temp12-2

votingC = VotingClassifier(estimators=[('lgbm', lgbmc), ('xgbc', xgbc), ('gnb', gnb)], voting='soft', n_jobs=4) #temp13
votingC = VotingClassifier(estimators=[('lgbm', lgbmc), ('xgbc', xgbc), ('svc', svc)], voting='soft', n_jobs=4) #temp14
votingC = VotingClassifier(estimators=[('temp', xgbc), ('rfc', rfc)], voting='soft', n_jobs=4) #temp15 0.9622
votingC = VotingClassifier(estimators=[('temp', lgbmc), ('rfc', rfc)], voting='soft', n_jobs=4) #temp16_1(lgbm이전버전, rfc새버전) 0.9622 = 2(새버전, 새버전)


# # 단일모델

# rfc.fit(train_x, train_y)
# pred = rfc.predict(test_df)

# xgbc.fit(train_x.astype(int), train_y.astype(int))
# pred = xgbc.predict(test_df.astype(int))
#
lgbmc.fit(train_x.astype(int), train_y.astype(int))
pred = lgbmc.predict(test_df.astype(int))

# 앙상블 모델
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

result_df = pd.DataFrame(result, index=temp['id'], columns=['class'])
result_df.to_csv("./model_result_temp12_2.csv")

# rfc(1개) > xgb(temp5와 비교 3개) > lgbm(4개)
temp1 = pd.read_csv("./model_result_temp12_2.csv")
temp2 = pd.read_csv("./model_result_temp5.csv")
temp3 = pd.read_csv("./model_result_14.csv")
#
check_bool(temp1, temp2)
check_bool(temp2, temp3)
check_bool(temp1, temp3)

"""
기록
12월 22일 밤(집)
model_resulttemp12 : 0.971919 == temp12_1
votingC = VotingClassifier(estimators=[('lgbm', lgbmc), ('xgbc', xgbc), ('rfc', rfc), ('extc', extc), ('gnb', gnb)], voting='soft', n_jobs=4)
1.LGBMClassifier(colsample_bytree=0.851261132978881, max_bin=446, max_depth=13,
               min_child_samples=19, num_leaves=169,
               reg_alpha=2.2714647832604017e-06, reg_lambda=0.0671564944943035,
               subsample=0.6846758987443338, subsample_freq=9)
2.XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
              colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
              early_stopping_rounds=None, enable_categorical=False,
              eval_metric=None, feature_types=None, gamma=0, gpu_id=-1,
              grow_policy='depthwise', importance_type=None,
              interaction_constraints='', learning_rate=0.300000012,
              max_bin=256, max_cat_threshold=64, max_cat_to_onehot=4,
              max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=1,
              missing=nan, monotone_constraints='()', n_estimators=100,
              n_jobs=0, num_parallel_tree=1, objective='multi:softprob',
              predictor='auto', ...)
3. RandomForestClassifier(max_depth=6, max_leaf_nodes=268, n_estimators=286,
                       random_state=42)
4. ExtraTreesClassifier(max_features=3, min_samples_split=6, random_state=42)
5. GaussianNB(var_smoothing=0.12067926406393285)

"""
