import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
os.chdir('C://data_minsung/finance/Qraft')
etfs = pd.read_csv('./indexPortfolio/etfs.csv').set_index('Date')
macros = pd.read_csv('./indexPortfolio/macros.csv').set_index('Unnamed: 0')

"""
1.etfs가 시계열 데이터 특성을 가졌는지 확인
2. 보간법, 이상치제거, 수치변환, 다중공선성 제거, feature engineering 등
"""
# etfs:육안으로 보아도 추세나, 계절성 내포, white noise와 거리가 먼 모양
# 차분(difference)화 하기
etfs.plot()
etfs_ret = etfs.pct_change()
etfs_ret.index = pd.to_datetime(etfs_ret.index)

macros_ret = macros.pct_change()
macros_ret.index = pd.to_datetime(macros_ret.index)

st_date = macros_ret.index[0]
etfs_ret = etfs_ret.apply(lambda x: x[st_date:])

etfs_ret.plot()

#보간법 : nan값은 이전과 변동 없는 것으로 간주하여 0.0으로 결측치 채움
macros_ret = macros_ret.fillna(0.0)
etfs_ret = etfs_ret.fillna(0.0)

# 상관계수 heatmap 통해 대략적인 파악
cmap = sns.light_palette("darkgray", as_cmap=True)
sns.heatmap(etfs_ret.corr(), annot=True, cmap=cmap)
plt.show()

sns.heatmap(macros_ret.corr(), annot=True, cmap=cmap)
plt.show()

# 다중공선성 제거 : VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(etfs_ret.values, i) for i in range(etfs_ret.shape[1])]
vif.index = etfs_ret.columns

# 데이터 합치고, train, test split
total_data = pd.concat((etfs_ret, macros_ret), axis=1)
trn_size = int(len(total_data)*0.7)

trn_data = total_data[:trn_size]
tst_data = total_data[trn_size:]

# normalize data
trn_mean = trn_data.mean()
trn_std = trn_data.std()

trn_data = (trn_data - trn_mean) / trn_std
tst_data = (tst_data - trn_mean) / trn_std


def data_for_rnn(input, idx_x, seq_len):
    input_for_rnn = []
    for i in range(len(input)-seq_len):
        input_for_rnn.append(input[i:i+seq_len])
    idx_x = idx_x[:-seq_len]
    return input_for_rnn, idx_x

# 월별 인덱스
add_ = etfs_ret.index[0]
last_idx = etfs_ret.resample('M').last().index
first_idx = last_idx + timedelta(days=1)
first_idx = list(first_idx)[:-1]
first_idx.insert(0, add_)

mth_len = []
for i in range(len(first_idx)):
    a= len(trn_data.TLT[first_idx[0]:last_idx[0]])
    mth_len.append(a)

# (obs, window, n_features)형태
window_size = int(np.mean(mth_len))
x_trn, xidx_trn = data_for_rnn(trn_data.values,trn_data.index ,window_size)
x_tst, xidx_tst = data_for_rnn(tst_data.values,trn_data.index ,window_size)

y_trn = trn_data.iloc[window_size:, :np.shape(etfs)[1]]
y_tst = tst_data.iloc[window_size:, :np.shape(etfs)[1]]


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
    model.add(layers.Input(shape=(seq_len, n_feature)))
    model.add(layers.LSTM(256, activation='relu', return_sequences=True))
    model.add(layers.LSTM(128, activation='relu'))
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
    return pred, mean_absolute_error(y_test, pred),  y_test

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