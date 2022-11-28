"""
input : macros, etfs(미래참조 금지)
output : portfolio weight, max return
seq2seq, transformer 다 해보기
"""
import pandas as pd
import numpy as np
import os
# import matplotlib.pyplot as plt
# import seaborn as sns
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import mean_absolute_error
# from statsmodels.stats.outliers_influence import variance_inflation_factor
os.chdir('C://data_minsung/finance/Qraft')

from tensorflow.python.client import device_lib
device_lib.list_local_devices()
tf.version.VERSION


def normalize(data):
    mean = data.mean()
    std = data.std()
    return (data-mean)/std

# (n, feature) => (n, window, feature)
def generate_window(data, seq_len):
    # 인덱싱은 0부터 시작이므로 고려(12번째 = index 11)
    input_idx = data.index[seq_len-1:-1]
    output_idx = data.index[seq_len:]
    data = data.values
    input = []
    for i in range(len(data)-seq_len):
        input.append(data[i:i+seq_len])
    output = data[seq_len:]

    return input, output, input_idx, output_idx

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


etfs = pd.read_csv('./indexPortfolio/etfs.csv').set_index('Date')
macros = pd.read_csv('./indexPortfolio/macros.csv').set_index('Unnamed: 0')

total = pd.concat((etfs[macros.index[0]:], macros), axis=1)
total.index = pd.to_datetime(total.index)

"""
데이터 경우의 수
1. 학습 일별, 테스트 월별 : 학습데이터 양이 많음
2. 학습 월별, 테스트 월별 : 월, 일 간의 데이터 형태 비유사시 문제

"""
# 학습 일별, 테스트 월별

# 보간법 : 0으로 채울까 선형으로 채울까(사잇값이 아닌 시작값이 결측치)
    # 선형보간법 사용시 : Nan에서 최초값까지 상승추세로 값이 채워짐 bad
    # Next observation carried backward(NOCB) : 직후 관측값으로 결측치 대체가 better
# 차분(differencing) 통해 시계열의 평균 변화 일정하게 해줌
trn_size = int(len(total)*0.7)
trn_data = total.iloc[:trn_size].pct_change().fillna(0.0)
tst_data = total.iloc[trn_size:].resample('M').last().pct_change().fillna(0.0)
# normalize
trn_data, tst_data = normalize(trn_data), normalize(tst_data)

# in/out : 12개월 치 데이터를 통해 다음 달 etfs_diff를 예측 - single step model
seq_len = 12
n_feature = np.shape(trn_data)[1]
x_trn, y_trn, x_trn_idx, y_trn_idx = generate_window(trn_data, seq_len)
x_tst, y_tst, x_tst_idx, y_tst_idx = generate_window(tst_data, seq_len)

# execute the model
# 집 컴퓨터 numpy tensor 싱크 문제 있는 듯
pred, mae = RNNmodel(x_trn, x_tst, y_trn, y_tst, seq_len, n_feature )



