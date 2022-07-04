import pandas_datareader as pdr
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import math
import quantstats as qs
import numpy as np
from sklearn import preprocessing
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
# import tensorflow as tf
# from tensorflow.keras import datasets, layers, models

import statistics
from scipy import stats

os.chdir('C:/data_minsung')

# MAA_data에서 받아온 데이터
# y_cut

start_date = '2000-01-01'
end_date = '2022-04-15'



t10y2y = pd.read_csv('./finance/new_data/T10Y2Y.csv').set_index('DATE')
t10y3m = pd.read_csv('./finance/new_data/T10Y3M.csv').set_index('DATE')
acogno = pd.read_csv('./finance/new_data/ACOGNO.csv').set_index('DATE')



def csvProcessing(x):
    input = x.values
    idx = x.index
    Y = []
    for ob in input:
        if ob == '.':
            Y.append(Y[-1])
        else:
            Y.append(float(ob))
    output = pd.DataFrame(Y, index=idx)
    return output

t10y2y = csvProcessing(t10y2y)
t10y2y = t10y2y.iloc[len(t10y2y) - len(t10y3m):]
t10y3m = csvProcessing(t10y3m)

idx = t10y2y.index[0]
idx_l = t10y2y.index[-1]
t10y2y.loc[idx:idx_l]

#
# train_sz = int(len(t10y2y)*0.8)
# x_train = t10y2y[:train_sz]
# y_train = t10y3m[:train_sz]
#
# x_test = t10y2y[train_sz:]
# y_test = t10y3m[train_sz:]
#
# reg = LinearRegression().fit(x_train, y_train)
# reg_pred = reg.predict(x_test)
# mean_absolute_error(y_test, reg_pred)
#
# x_sample = t10y2y[:100].values.squeeze()
# y_sample = t10y3m[:100].values.squeeze()
#
# stats.pearsonr(x_sample, y_sample)
# stats.kendalltau(x_sample, y_sample)
#
# a = x_sample.values.squeeze()
#
# plt.scatter(x_train, y_train)