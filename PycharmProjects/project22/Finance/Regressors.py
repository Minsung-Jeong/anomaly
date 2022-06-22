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


x_cut = x_analy
y_cut = y_analy

train_size = round(len(x_cut)*0.8)
x_train = x_cut.iloc[:train_size]
y_train = y_cut.iloc[:train_size]

x_test = x_cut.iloc[train_size:]
y_test = y_cut.iloc[train_size:]

# idx_test = idx_cut[train_size:]


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


# # rnn regressor 로 fit 꽉 차게 해보기기
# def RNNregressor(x, y, trn_prop):
# (batch, seq_len, features) : seq_len 을 일종의 window_size로 생각하면 됨
x = x_cut
y = y_cut
trn_prop = 0.8

train_size = round(len(x) * trn_prop)
val_size = round(len(x)* 0.1)

seq_len = 30
n_feature = x.shape[1]



x_train = x.iloc[:train_size]
y_train = y.iloc[:train_size]
x_val = x.iloc[train_size : train_size+val_size]
y_val = y.iloc[train_size : train_size+val_size]
x_test = x.iloc[train_size+val_size:]
y_test = y.iloc[train_size+val_size:]

abc = np.zeros((1,15))
x_train[7, [-1]]

model = models.Sequential()
# Add a LSTM layer with 128 internal units.
model.add(layers.LSTM(128, activation='relu', input_shape=()))
# Add a Dense layer with 10 units.
model.add(layers.Dense(1))

model.compile(optimizer='adam',
              loss = tf.keras.losses.mean_absolute_error(from_logits=True),
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10,
                    validation_data=(x_val, y_val))
pred = model.predict(x_test)

def RDFregressor(x, y, trn_prop):
    train_size = round(len(x)*trn_prop)
    x_train = x.iloc[:train_size]
    y_train = y.iloc[:train_size]

    x_test = x.iloc[train_size:]
    y_test = y.iloc[train_size:]
    rf_clf1 = RandomForestRegressor(n_estimators = 150,
                                    max_depth = 50,
                                    min_samples_leaf = 6,
                                    min_samples_split = 2,
                                    random_state = 0,
                                    n_jobs = -1)
    rf_clf1.fit(x_train, y_train)
    pred = rf_clf1.predict(x_test)

    return pred, mean_absolute_error(y_test, pred), y_test


RDF_pred, RDFmae, y_test = RDFregressor(x_cut, y_cut, 0.8)

plt.bar(np.arange(len(RDF_pred)),RDF_pred)
plt.bar(y_test.index ,y_test.values)
plt.plot(np.zeros(len(RDF_pred)))
plt.bar(y_test[-50:].index, y_test[-50:].values)

reg = LinearRegression().fit(x_train, y_train)
reg_pred = reg.predict(x_test)

idx_test = y_test.index
plt.bar(idx_test, reg_pred, color='r')
plt.bar(idx_test, y_test, color='b')
plt.plot()


count = 0
li = []
for i, x in enumerate(RDF_pred):
    if x > 0:
        count +=1
        li.append(i)


tst_count = 0
li = []
for x in y_test:
    if x > 0:
        tst_count += 1
        li.append(x)

tst_count
print(tst_count)

li
len(y_test)