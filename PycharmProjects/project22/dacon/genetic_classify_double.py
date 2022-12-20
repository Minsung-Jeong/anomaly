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


train_x
train_y