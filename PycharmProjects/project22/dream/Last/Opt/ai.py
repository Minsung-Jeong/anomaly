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
import matplotlib.pyplot as plt
import seaborn as sns
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
etfs = pd.read_csv('./indexPortfolio/etfs.csv').set_index('Date')
macros = pd.read_csv('./indexPortfolio/macros.csv').set_index('Unnamed: 0')
lable.index = pd.to_datetime(lable.index)
etfs.index = pd.to_datetime(etfs.index)
macros.index = pd.to_datetime(macros.index)

# visualize data
etfs.plot()
macros.plot(figsize=(15,10))

# 아웃라이어 제거
def remove_out(data, remove_col):
    inp = data
    for k in remove_col:
        col_data  = inp[k]
        Q1 = inp[k].quantile(0.25)
        Q3 = inp[k].quantile(0.75)
        IQR = Q3 - Q1
        IQR = IQR*2
        low = Q1 - IQR
        high = Q3 + IQR
        outlier_idx = col_data[((col_data < low) | (col_data > high))].index
        inp.drop(outlier_idx, axis=0, inplace=True)
    return inp

def normalize(data):
    mean = data.mean()
    std = data.std()
    return (data-mean)/std

def data_process(etfs, macros):
    # macros 길이에 맞춰서 데이터 합치기
    etfs = etfs.iloc[len(etfs) - len(macros):]
    input = macros.copy()
    input[etfs.columns.values] = etfs.copy()

    # 보간1 : null값 보간(KNN Imputation)
    Kimp = KNNImputer(n_neighbors=10)
    input = pd.DataFrame(Kimp.fit_transform(input), index=input.index, columns=input.columns)

    # 데이터 정규화
    input = normalize(input)

    # 영향력 클 것으로 예상되는 'VIX Index', 'MOVE Index'
    sns.heatmap(input.corr())

    # 위 2개 변수에 대해서 outlier 제거
    remove_col = ['VIX Index', 'MOVE Index']
    # Outlier 제거
    result = remove_out(input, remove_col)

    # 보간2 : 월말 데이터 결측치 이전 날 기준으로 보간
    # ex) 94/4/30의 부재는 94/4/29를 이용하여 보간
    # input_month = input.resample('M').last()
    # temp = input.append(input_month)
    # temp = temp.sort_values(by='Unnamed: 0')
    # temp = temp[~temp.index.duplicated(keep='first')]

    result_month = result.resample('M').last()
    result = result.append(result_month)
    result = result.sort_values(by='Unnamed: 0')
    result = result[~result.index.duplicated(keep='first')]
    return result

def data_for_rnn(input, lable, window_size):
    index_sync = []
    for i in range(len(lable)):
        for j in range(len(input)):
            if input.index[j] == lable.index[i]:
                index_sync.append(j)

    # lable 데이터 기준으로 이전 30일의 데이터를 input 데이터로 지정
    input_li = [input.iloc[i-window_size : i].values for i in index_sync]
    index_li = [input.index[i] for i in index_sync]

    return input_li, index_li

def RNNmodel(x_trn ,x_tst, y_trn, y_tst, seq_len, n_feature):
    checkpoint_path = 'training_1/cp-{epoch:04d}.ckpt'
    os.chdir('C://data_minsung/result/AIport')
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=5)

    x_train = tf.convert_to_tensor(x_trn)
    x_test = tf.convert_to_tensor(x_tst)
    y_train = tf.convert_to_tensor(y_trn)
    y_test = tf.convert_to_tensor(y_tst)
    out_shape = tf.shape(y_test)[1]

    model = models.Sequential()
    model.add(layers.Input(shape=(seq_len, n_feature), name='input'))
    model.add(layers.LSTM(256, activation='relu', return_sequences=True, name='first'))
    model.add(layers.LSTM(128, activation='relu', name='second'))
    # model.add(layers.LSTM(64, activation='relu', name='third'))
    # classify면 2개, regress면 1개로 설정하도록
    model.add(layers.Dense(out_shape))
    model.summary()
    model.compile(optimizer='adam',
                  loss = 'mse',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=100,
                        # validation_data=(x_val, y_val)
                        callbacks=[cp_callback]
                        )
    pred = model.predict(tf.convert_to_tensor(x_test))
    return pred, mean_absolute_error(y_test, pred)


input = data_process(etfs, macros)

trn_size = int(len(lable)*0.7)
trn_lable = lable.iloc[:trn_size]
tst_lable = lable.iloc[trn_size:]

data_rnn, idx_rnn = data_for_rnn(input, lable, 20)

# input_trn, idx_trn = data_for_rnn(trn_input, trn_lable, 20)
# input_tst, idx_tst = data_for_rnn(tst_input, tst_lable, 20)

# pred, mae = RNNmodel(input_trn, input_tst, trn_lable, tst_lable, seq_len=20, n_feature=30 )

