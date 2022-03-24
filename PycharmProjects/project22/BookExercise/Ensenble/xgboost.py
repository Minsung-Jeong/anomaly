import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

def get_clf_eval(y_test, pred=None, pred_proba = None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    print("acc :{0:4f}, f1-score:{1:4f}".format(accuracy, f1))

# 코드 소실분 채우기 1/18


#
# dataset = load_breast_cancer()
# X_features = dataset.data
# y_label = dataset.target
#
# cancer_df = pd.DataFrame(data=X_features, columns=dataset.feature_names)
# cancer_df['target'] = y_label
#
# cancer_df.info()
# cancer_df.describe()
# print(cancer_df['target'].value_counts())
#
# X_train, X_test, y_train, y_test = train_test_split(X_features, y_label, test_size=0.2, random_state=156)
#
#
# # 1. 파이썬 xgBoost 사용
# dtrain = xgb.DMatrix(data=X_train, label=y_train)
# dtest = xgb.DMatrix(data=X_test, label=y_test)
#
# params = {'max_depth':3,
#           'eta':0.1,
#           'objective':'binary:logistic',
#           'eval_metric':'logloss',
#           'early_stoppings':100}
# num_rounds = 400
# wlist = [(dtrain, 'train'), (dtest, 'eval')]
#
# xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_rounds, early_stopping_rounds=100, evals=wlist)
# pred_probs = xgb_model.predict(dtest)
# preds = [1 if x>0.5 else 0 for x in pred_probs]
# get_clf_eval(y_test, preds, pred_probs)
#
# # 2. 싸이킷런 xgBoost 사용
# xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
# xgb_wrapper.fit(X_train, y_train)
# w_preds = xgb_wrapper.predict(X_test)
# w_pred_proba = xgb_wrapper.predict_proba(X_test)[:,1]
#
# get_clf_eval(y_test, w_preds, w_pred_proba)
#
# #2.1 싸이킷런 xgboost 통한 조기중단
# xgb_wrapper = XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=3)
# evals = [(X_test, y_test)]
# xgb_wrapper.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="logloss", eval_set=evals, verbose=True)
# ws100_preds = xgb_wrapper.predict(X_test)
# ws100_pred_proba = xgb_wrapper.predict_proba(X_test)[:,1]
#
# get_clf_eval(y_test, ws100_preds, ws100_pred_proba)
#
