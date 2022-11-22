"""
input : macros, etfs(미래참조 금지)
output : portfolio weight, max return
seq2seq, transformer 다 해보기
"""
import pandas as pd
import numpy as np
import os
from scipy import stats, interpolate
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
import statsmodels.api as sm
# import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import mean_absolute_error
# from statsmodels.stats.outliers_influence import variance_inflation_factor
os.chdir('C://data_minsung/finance/Qraft')

"""
1. Data EDA
-시계열 특성에 대한 체크
-보간법, 이상치 제거, 수치변환, 다중 공선성 제거, feature engineering 등 수행
-보간법에 따른 성능차이? -이전값 constant, 
-input : etfs, macros / output : 횡적리스크 모델 결과물
"""
lable = pd.read_csv('./result/sharpeMax_port.csv').set_index('Date')
lable.index = pd.to_datetime(lable.index)
etfs = pd.read_csv('./indexPortfolio/etfs.csv').set_index('Date')
etfs.index = pd.to_datetime(etfs.index)
macros = pd.read_csv('./indexPortfolio/macros.csv').set_index('Unnamed: 0')
macros.index = pd.to_datetime(macros.index)
# macros 길이에 맞춰서 데이터 합치기
etfs = etfs.iloc[len(etfs)-len(macros):]

# lable : 94/2/28~21/4/30
# input : 90/1/1~21/5/13 -> 94/2/25~21/4/29,(2/27 X -> 2/25)
input = macros.copy()
input[etfs.columns.values] = etfs.copy()

start = 0
end = 0
for i in range(len(input)):
    if input.index[i] == lable.index[0]:
        start = i
    if input.index[i] == lable.index[-1]:
        end = i

input = input.iloc[start-1:end].copy()

# input 결측치 'MXUSMMT Index'에서 67개
input.isna().sum()
lable
input.iloc[0]
# 보간법 1. IterativeImputer / 2.KNNImputer
imp = IterativeImputer(max_iter=10, random_state=0)
imp.fit(input)
X = pd.DataFrame(imp.transform(input), index=input.index, columns=input.columns)

Kimp = KNNImputer(n_neighbors=10)
X2 = pd.DataFrame(Kimp.fit_transform(input), index=input.index, columns=input.columns)
