# xgboost, lightGBM 사용하는 실습

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from xgboost import plot_importance
from lightgbm import LGBMClassifier

os.chdir("C://data_minsung")

cust_df = pd.read_csv("./santander/train.csv")
cust_df.info()

# 데이터 라벨값 확인
cust_df['TARGET'].value_counts()

# 데이터 대충 보기
cust_df.describe()

# 몇 개인지 세기
cust_df[cust_df['var3'] == -99999].var3.count()

cust_df.replace(-99999, 2, inplace=True)
cust_df.drop('ID', axis=1, inplace=True)

X_features = cust_df.iloc[:, :-1]
y_labels = cust_df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2, random_state=156)

sum(y_train==1)/y_train.count()
sum(y_test==1)/y_test.count()

xgb_clf = XGBClassifier(n_estimators=500, random_state=156)
xgb_clf.fit(X_train, y_train, early_stopping_rounds=100, eval_metric='auc',
            eval_set=[(X_train, y_train), (X_test, y_test)])

xgb_auc_roc = roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:,1], average='macro')
print(xgb_auc_roc)

# Grid Search 진행
xgb_clf = XGBClassifier(n_estimators=100)

params = {'max_depth':[5,7], 'min_child_weight':[1,3], 'colsample_bytree':[0.5, 0.75]}

gridcv = GridSearchCV(xgb_clf, param_grid=params, cv=3)
gridcv.fit(X_train, y_train, early_stopping_rounds=30, eval_metric="auc",
           eval_set=[(X_train, y_train),(X_test, y_test)])
print('GridSearch 최적파라미터\n', gridcv.best_params_, '최적스코어:{0:3f}'.format(gridcv.best_score_))

# Grid Search 의 하이퍼파라미터 기반으로 다시 xgb 돌리기

xgb_clf = XGBClassifier(n_estimators=1000, colsample_bytree=0.75,
                        max_depth=5, min_child_weight=1, random_state=156, learning_rate=0.02, reg_alpha=0.03)
xgb_clf.fit(X_train, y_train, early_stopping_rounds=200, eval_metric='auc', eval_set=[(X_train, y_train),(X_test, y_test)])
xgb_roc_score = roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:,1], average='macro')
print('roc score:{0:4f}'.format(xgb_roc_score))
acc = sum(y_test == xgb_clf.predict(X_test))/len(X_test)

fig, ax = plt.subplots(1, 1, figsize=(10,8))
plot_importance(xgb_clf, ax=ax, max_num_features=20, height=0.4)


# light GBM 실행
lgbm_clf = LGBMClassifier(n_estimators=500)
evals = [(X_test, y_test)]
lgbm_clf.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="auc", eval_set=evals, verbose=True)
lgbm_roc_score = roc_auc_score(y_test, lgbm_clf.predict_proba(X_test)[:,1], average='macro')

# Grid Search 실행
lgbm_clf = LGBMClassifier(n_estimators=200)
params = {'num_leaves':[32, 64],
          'max_depth':[128, 160],
          'min_child_samples':[60, 100],
          'subsample':[0.8, 1]}
gridcv = GridSearchCV(lgbm_clf, param_grid=params, cv=3)
gridcv.fit(X_train, y_train, early_stopping_rounds=100,
           eval_metric='auc', eval_set=evals, verbose=True)

gridcv.best_params_
lgbm_roc_score = roc_auc_score(y_test, gridcv.predict_proba(X_test)[:,1], average='macro')
