import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from lightgbm import LGBMClassifier
import seaborn as sns
from sklearn.preprocessing import StandardScaler


import warnings
warnings.filterwarnings('ignore')



def get_clf_eval(y_test, pred=None, pred_proba = None):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba, average='macro')
    print("confusion:{0}\n acc :{1:4f}, f1-score:{2:4f}, roc-auc:{3:4f}".format(confusion,accuracy, f1, roc_auc))

def get_model_train_eval(model, ftr_train=None, ftr_test=None, tgt_train=None, tgt_test=None):
    model.fit(ftr_train, tgt_train)
    pred = model.predict(ftr_test)
    pred_proba = model.predict_proba(ftr_test)[:,1]
    get_clf_eval(tgt_test, pred, pred_proba)

def get_preprocessed_df(df=None):
    df_copy =df.copy()
    amount_n = np.log1p(df_copy['Amount'])
    df_copy.insert(0, 'Amount-Scaled', amount_n)
    df_copy.drop(['Time','Amount'], axis=1, inplace=True)

    outlier_index = get_outlier(df=df_copy, column='V14', weight=1.5)
    df_copy.drop(outlier_index, axis=0, inplace=True)
    return df_copy



def get_outlier(df=None, column=None, weight=1.5):
    # 특정 column에서 fraud 에 해당 하는 값 추출
    fraud = df[df['Class']==1][column]
    quantile_25 = np.percentile(fraud.values, 25)
    quantile_75 = np.percentile(fraud.values, 75)

    iqr = quantile_75 - quantile_25
    iqr_weight = iqr * weight
    lowest_val = quantile_25 - iqr_weight
    highest_val = quantile_75 + iqr_weight

    outlier_index = fraud[(fraud < lowest_val)|(fraud > highest_val)].index
    return outlier_index


os.getcwd()
card_df = pd.read_csv('./creditcard_forGBM.csv')
pro_df = card_df.drop('Time', axis=1)

X_features = pro_df.iloc[:, :-1]
y_target = pro_df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3,
                                                    random_state=0, stratify=y_target)

# logistic regression 써보기
lr_clf = LogisticRegression()
lr_clf.fit(X_train, y_train)
lr_pred = lr_clf.predict(X_test)
lr_pred_prob = lr_clf.predict_proba(X_test)[:,1]

get_clf_eval(y_test, lr_pred, lr_pred_prob)

# light BGM 써보기
lgbm_clf = LGBMClassifier(n_estimators=1000, num_leaves=64, n_jobs=1, boost_from_average=False)
get_model_train_eval(lgbm_clf, X_train, X_test, y_train, y_test)


# --------------------------------- 데이터 분포 변환 후 다시 학습 및 예측--------------------
plt.figure(figsize=(8,4))
plt.xticks(range(0,30000,1000), rotation=60)
sns.distplot(card_df["Amount"])

scaler = StandardScaler()
amount_n = scaler.fit_transform(pro_df['Amount'].values.reshape(-1,1))
pro_df.insert(0, 'Amount-scaled', amount_n)
pro_df.drop('Amount', axis=1, inplace=True)

X_features = pro_df.iloc[:, :-1]
y_target = pro_df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3,
                                                    random_state=0, stratify=y_target)
get_model_train_eval(lgbm_clf, X_train, X_test, y_train, y_test)


# IQR 통해 이상치 제거( IQR*1.5 값을 플러스 마이너스 해서 그 범위 넘는 값을 제거)
plt.figure(figsize=(9,9))
corr = card_df.corr()
sns.heatmap(corr, cmap='RdBu')

pro_df = get_preprocessed_df(card_df)

X_features = pro_df.iloc[:, :-1]
y_target = pro_df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.3,
                                                    random_state=0, stratify=y_target)

get_model_train_eval(lgbm_clf, X_train, X_test, y_train, y_test)    