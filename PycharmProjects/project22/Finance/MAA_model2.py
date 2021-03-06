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
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from statsmodels.tsa.arima_model import ARIMA

# model = RandomForestRegressor()
# params = {'n_estimators' : [10, 100],
#           'max_depth' : [6, 8, 10, 12],
#           'min_samples_leaf' : [8, 12, 18],
#           'min_samples_split' : [8, 16, 20]
#           }
#
# rf_clf = RandomForestRegressor(random_state = 221, n_jobs = -1)
# grid_cv = GridSearchCV(rf_clf,
#                        param_grid = params,
#                        cv = 3,
#                        n_jobs = -1)
# grid_cv.fit(x_train, y_train)
#
# print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
# print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))
# 값 바이너리화 양수면 1, 음수면 0
def get_binary(y_analy):
    y_binary = []
    # 양수 1, 음수 0
    for val in y_analy:
        if val > 0:
            y_binary.append(1)
        else:
            y_binary.append(0)
    return y_binary

########## MAA_data 파일에서 만든 데이터
x_cut
idx_x
y_cut

y_index = y_cut.index
# 뒤에 def로 감싸지 않은 데이터 위해 그냥 둔다
train_size = round(len(x_cut)*0.8)
x_train = x_cut.iloc[:train_size]
y_train = y_cut.iloc[:train_size]

x_test = x_cut.iloc[train_size:]
y_test = y_cut.iloc[train_size:]
test_idx = y_index[train_size:]

#########################################  RNN 하는 부분
# (batch, seq_len, features) : seq_len 을 일종의 window_size로 생각하면 됨
trn_prop = 0.8
seq_len = 20 #20으로 해야 한 달(데이터에 주말 없음)
n_feature = np.shape(x_cut)[-1]

def data_for_rnn(x, y, idx_x, seq_len, trn_prop):
    train_size = round(len(x)*trn_prop)
    idx_train = pd.to_datetime(idx_x).values[:train_size]
    idx_test = pd.to_datetime(idx_x).values[train_size:]


    # train은 일별로 test는 월별로 진행
    x_train = x[:train_size].values
    x_test = x[train_size:].values
    x_test = pd.DataFrame(x_test, index=idx_test)
    x_test = x_test.resample(rule='M').last()

    idx_test = x_test.index
    x_test = x_test.values

    y = y.iloc[seq_len:]
    y_train = y[:train_size]
    y_test = y[train_size:]
    y_test = y_test.resample(rule='M').last()

    def new_dim(input):
        input_for_rnn = []
        for i in range(len(input)-seq_len):
            input_for_rnn.append(input[i:i+seq_len])
        return input_for_rnn

    x_train = new_dim(x_train)
    x_test = new_dim(x_test)

    idx_train = idx_train[:-seq_len]
    return x_train, x_test, y_train, y_test, idx_train, idx_test

def RNNmodel(x_train, x_test, y_train, y_test):

    x_train = tf.convert_to_tensor(x_train)
    y_train = tf.convert_to_tensor(y_train)
    x_test = tf.convert_to_tensor(x_test)
    y_test = tf.convert_to_tensor(y_test)

    model = models.Sequential()
    model.add(layers.Input(shape=(seq_len, n_feature)))
    model.add(layers.LSTM(128, activation='relu', return_sequences=True))
    model.add(layers.LSTM(64, activation='relu'))
    # classify면 2개, regress면 1개로 설정하도록
    model.add(layers.Dense(1))
    model.summary()
    model.compile(optimizer='adam',
                  loss = 'mse',
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=10,
                        # validation_data=(x_val, y_val)
                        )
    pred = model.predict(tf.convert_to_tensor(x_test))

    return pred, mean_absolute_error(y_test, pred)

# ################################# RDF 하는 부분
def RDFregressor(x, y, trn_prop):
    train_size = round(len(x)*trn_prop)
    x_train = x.iloc[:train_size]
    y_train = y.iloc[:train_size]

    x_test = x.iloc[train_size:]
    y_test = y.iloc[train_size:]
    rf_clf1 = RandomForestRegressor(n_estimators = 100,
                                    max_depth = 150,
                                    min_samples_leaf = 6,
                                    min_samples_split = 2,
                                    random_state = 0,
                                    n_jobs = -1)
    rf_clf1.fit(x_train, y_train)
    pred = rf_clf1.predict(x_test)

    return pred, mean_absolute_error(y_test, pred), y_test


# rnn 위한 x,y값 생성
x_train, x_test, y_train, y_test, idx_train, idx_test = data_for_rnn(x_cut, y_cut, idx_x, 20, 0.8)
y_rnn_idx = y_index[:-seq_len]

RNN_pred, RNN_mae = RNNmodel(x_train, x_test, y_train, y_test)
RDF_pred, RDF_mae, RDF_test = RDFregressor(x_cut, y_cut, 0.8)

plt.bar(np.arange(len(RDF_pred)),RDF_pred)
plt.bar(RDF_test.index ,RDF_test.values)
plt.plot(np.zeros(len(RDF_pred)))
plt.bar(RDF_test[-50:].index, RDF_test[-50:].values)


RNN_binary = get_binary(RNN_pred)
RNN_pd = pd.DataFrame(RNN_binary, index=tst_index)
RDF_binary = get_binary(RDF_pred)

# # ########################## 선형회귀 하는 부분
# reg = LinearRegression().fit(x_train, y_train)
# reg_pred = reg.predict(x_test)
#
# idx_test = y_test.index
# plt.bar(idx_test, reg_pred, color='r')
# plt.bar(idx_test, y_test, color='b')
# plt.plot()


