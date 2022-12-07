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
from sklearn.impute import KNNImputer
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.metrics import mean_absolute_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score


# 아웃라이어 제거
def remove_out(data, remove_col):
    inp = data.copy()
    idx = []
    for k in remove_col:
        col_data = inp[k]
        Q1 = inp[k].quantile(0.25)
        Q3 = inp[k].quantile(0.75)
        IQR = Q3 - Q1
        IQR = IQR*2 #2배수에 대한 논리 필요
        low = Q1 - IQR
        high = Q3 + IQR
        outlier_idx = col_data[((col_data < low) | (col_data > high))].index
        inp.drop(outlier_idx, axis=0, inplace=True)
        idx.extend(outlier_idx)
    return inp, idx

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
    result, out_idx = remove_out(input, remove_col)

    # 보간2 : 월말 데이터 결측치 이전 날 기준으로 보간
    # ex) 94/4/30의 부재는 94/4/29를 이용하여 보간
    result_month = result.resample('M').last()
    result = result.append(result_month)
    result = result.sort_values(by='Unnamed: 0')
    result = result[~result.index.duplicated(keep='first')]

    # outlier 제거로 인해 유실된 데이터에 맞춤
    null_idx = result_month[result_month.iloc[:, 0].isna()].index.values
    result.drop(null_idx, axis=0, inplace=True)

    # etfs를 lable data로 이용
    cols = etfs.columns.values
    lable = result[cols]
    return result, lable

def data_for_rnn(input, seq_len):
    x_li = []
    x_idx = input.index[seq_len-1:-1]
    for i in range(len(input)-seq_len):
        x_li.append(input.iloc[i:i+seq_len].values)
    return x_li, x_idx

def RNNmodel(out_shape, seq_len, n_feature):
    checkpoint_path = 'training_1/cp-{epoch:04d}.ckpt'
    os.chdir('C://data_minsung/result/AIport')
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=5)
    out_shape = out_shape

    model = models.Sequential()
    model.add(layers.Input(shape=(seq_len, n_feature), name='input'))
    model.add(layers.LSTM(128, activation='relu', return_sequences=True, name='first'))
    model.add(layers.LSTM(64, activation='relu', name='second'))
    # model.add(layers.LSTM(64, activation='relu', name='third'))
    model.add(layers.Dense(out_shape))
    model.summary()
    model.compile(optimizer='adam',
                  loss = 'mse',
                  metrics=['accuracy'])
    return model
    # history = model.fit(x_train, y_train, epochs=100,
    #                     # validation_data=(x_val, y_val)
    #                     callbacks=[cp_callback]
    #                     )
    # pred = model.predict(tf.convert_to_tensor(x_test))
    # return pred, mean_absolute_error(y_test, pred)

def del_multicolin(input):
    # 다중공선성 제거
    vif = pd.DataFrame()
    vif['VIF_Factor'] = [variance_inflation_factor(input.values, i) for i in range(input.shape[1])]
    vif['feature'] = input.columns
    vif.sort_values(by='VIF_Factor', ascending=True)
    # SPX Index에서 압도적으로 높은 다중공선성을 보이므로 해당값 삭제
    input.drop('SPX Index', axis=1, inplace=True)
    return input

"""
1. Data EDA
-시계열 특성에 대한 체크
-보간법, 이상치 제거, 수치변환, 다중 공선성 제거, feature engineering 등 수행
-보간법에 따른 성능차이? -이전값 constant, 
-input : etfs, macros / output : 횡적리스크 모델 결과물
"""
os.chdir('C://data_minsung/finance/Qraft')
etfs = pd.read_csv('./indexPortfolio/etfs.csv').set_index('Date')
macros = pd.read_csv('./indexPortfolio/macros.csv').set_index('Unnamed: 0')

etfs.index = pd.to_datetime(etfs.index)
macros.index = pd.to_datetime(macros.index)


# etfs 시계열 데이터 특성 확인
check_etfs = etfs.dropna()
f, axes = plt.subplots(3,3)
f.set_size_inches((10,10))
cols = etfs.columns
# 모든 값에 trend는 있지만 seasonality는 없음

# adf - h0 : 비정상시계열, kpss-h0 : 정상시계열
# adf의 p-val 가장 작은 값이 0.43 모든 변수가 비정상시계열(trend가 너무 강함)
# 딥러닝을 적용할 것이기 때문에 차분 등의 과정 불필요하다고 판단
for col in cols:
    # sm.tsa.seasonal_decompose(check_etfs[col]).plot()
    print("Augmented Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(check_etfs[col])[1])

# 데이터 전처리
input, lable = data_process(etfs, macros)
input = del_multicolin(input)

seq_len = 20
# 원래 data len = 8106
y_rnn = lable.iloc[seq_len:]
x_rnn, input_idx = data_for_rnn(input, seq_len)

trn_size = int(len(input)*0.7)
y_trn = y_rnn.iloc[:trn_size]
y_tst = y_rnn.iloc[trn_size:]

x_trn = x_rnn[:trn_size]
x_tst = x_rnn[trn_size:]

n_feature = np.shape(x_trn)[-1]

model = RNNmodel(np.shape(y_tst)[1], seq_len=20, n_feature=np.shape(x_trn)[-1])
# -------------------------model 실행부
x_test = tf.convert_to_tensor(x_tst)
y_test = tf.convert_to_tensor(y_tst)

for i in range(len(x_trn)-1):
    x_trn_t = x_trn[:i+1]
    y_trn_t = y_trn.iloc[:i+1]

    x_val = np.expand_dims(x_trn[i+1], axis=0)
    y_val = np.expand_dims(y_trn.iloc[i+1], axis=0)

    x_train = tf.convert_to_tensor(x_trn_t)
    y_train = tf.convert_to_tensor(y_trn_t)

    x_val = tf.convert_to_tensor(x_val)
    y_val = tf.convert_to_tensor(y_val)

    model.fit(x_train, y_train, epochs=5,
                        validation_data=(x_val, y_val)
                        # callbacks=[cp_callback]
                        )
pred = model.predict(tf.convert_to_tensor(x_test))

mean_absolute_error(y_tst, pred)
