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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
# from statsmodels.tsa.arima_model import ARIMA

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
from project22.tutorial.DCGAN.DCGAN import checkpoint_dir


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

def RDF_CLF(x, y, trn_prop):
    train_size = round(len(x)*trn_prop)
    x_train = x.iloc[:train_size]
    y_train = y.iloc[:train_size]

    x_test = x.iloc[train_size:]
    y_test = y.iloc[train_size:]
    rf_clf1 = RandomForestClassifier(n_estimators = 100,
                                    max_depth = 150,
                                    min_samples_leaf = 6,
                                    min_samples_split = 2,
                                    random_state = 0,
                                    n_jobs = -1)
    rf_clf1.fit(x_train, y_train)
    pred = rf_clf1.predict(x_test)

    return pred, accuracy_score(y_test, pred), y_test

#########################################  RNN 하는 부분
# (batch, seq_len, features) : seq_len 을 일종의 window_size로 생각하면 됨
trn_prop = 0.8
seq_len = 20
n_feature = np.shape(x_cut)[-1]

def data_for_rnn(input, idx_x, seq_len):
    input_for_rnn = []
    for i in range(len(input)-seq_len):
        input_for_rnn.append(input[i:i+seq_len])
    idx_x = idx_x[:-seq_len]
    return input_for_rnn, idx_x

def RNNmodel(x_rnn, y_rnn, trn_prop, index):
    checkpoint_path = 'training_1/cp-{epoch:04d}.ckpt'
    os.chdir('C://data_minsung/finance/MAA_RNN')
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=5)

    train_size = round(len(x_rnn) * trn_prop)
    x_train = tf.convert_to_tensor(x_rnn[:train_size])
    y_train = tf.convert_to_tensor(y_rnn[:train_size])
    x_test = tf.convert_to_tensor(x_rnn[train_size:])
    y_test = tf.convert_to_tensor(y_rnn[train_size:])
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
    history = model.fit(x_train, y_train, epochs=50,
                        # validation_data=(x_val, y_val)
                        callbacks=[cp_callback]
                        )
    pred = model.predict(tf.convert_to_tensor(x_test))
    return pred, mean_absolute_error(y_test, pred),  y_test, index[train_size:]

def create_model():
    model = models.Sequential()
    model.add(layers.Input(shape=(seq_len, n_feature)))
    model.add(layers.LSTM(128, activation='relu', return_sequences=True))
    model.add(layers.LSTM(64, activation='relu'))
    # classify면 2개, regress면 1개로 설정하도록
    model.add(layers.Dense(1))
    model.summary()
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])
    return model



# rnn 위한 x,y값 생성
x_rnn, idx_rnn = data_for_rnn(x_cut.values, idx_x, seq_len)
y_rnn = y_cut[seq_len:] #seq_len 단위로 예측하다 보니 앞을 삭제해야함
y_rnn_idx = y_rnn.index

train_prop = 0.5
RDF_pred, RDF_mae, RDF_test = RDFregressor(x_cut, y_cut, train_prop)
# RNN_pred, RNN_mae, RNN_test, tst_index = RNNmodel(x_rnn, y_rnn, train_prop, y_rnn_idx)


# 모델 load - 지저분해지므로 전체모델 저장과 load하는 방식으로 진행하기
os.chdir('C://data_minsung/finance/MAA_RNN')
checkpoint_path = 'training_1/cp-{epoch:04d}.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
train_size = round(len(x_rnn)*train_prop)

latest = tf.train.latest_checkpoint(checkpoint_dir)
model = create_model()
model.load_weights(latest)
x_test = tf.convert_to_tensor(x_rnn[train_size:])

RNN_pred = model.predict(x_test)
tst_index = y_rnn[train_size:].index



# y_rnn_bin = pd.DataFrame(get_binary(y_rnn.values), index=y_rnn.index)
# y_cut_bin = pd.DataFrame(get_binary(y_cut.values), index=y_cut.index)
# RNN_CLF_pred, RNN_CLF_acc, RNN_CLF_test, tst_CLF_index = RNNmodel(x_rnn, y_rnn_bin, 0.8, y_rnn_idx)
# RDF_CLF_pred, RDF_CLF_acc, RDF_CLF_test = RDF_CLF(x_cut, y_cut_bin,  0.8)


# plt.bar(np.arange(len(RDF_pred)),RDF_pred)
# plt.bar(RDF_test.index ,RDF_test.values)
# plt.plot(np.zeros(len(RDF_pred)))
# plt.bar(RDF_test[-50:].index, RDF_test[-50:].values)

RNN_binary = get_binary(RNN_pred)
RNN_pd = pd.DataFrame(RNN_pred, index=tst_index).resample('M').mean()
# RNN_pd = pd.DataFrame(RNN_binary, index=tst_index).resample('M').mean()
RDF_binary = get_binary(RDF_pred)
RDF_pd = pd.DataFrame(RDF_pred, index=RDF_test.index).resample('M').mean()
# RDF_pd = pd.DataFrame(RDF_binary, index=RDF_test.index).resample('M').mean()

sum(RNN_binary)/len(RNN_binary)
# 단순 평균통한 앙상블(soft 앙상블)
pred_average = (RNN_pd +RDF_pd)/2

# # ########################## 선형회귀 하는 부분
# reg = LinearRegression().fit(x_train, y_train)
# reg_pred = reg.predict(x_test)
#
# idx_test = y_test.index
# plt.bar(idx_test, reg_pred, color='r')
# plt.bar(idx_test, y_test, color='b')
# plt.plot()