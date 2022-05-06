import numpy as np
import tensorflow as tf
import pandas as pd
# import tqdm
from tqdm.notebook import trange, tqdm
from tensorflow.keras import Model, models, layers, optimizers, utils
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from celluloid import Camera
import os

os.getcwd()
os.chdir('C://data_minsung/finance')

SNP = pd.read_csv('./SNP500_anomal.csv')
X = pd.read_csv('./total_data.csv').iloc[:,2:]
X_date = X.iloc[:,1]

X_scaled = tf.keras.utils.normalize(X, axis=1)

# normal, abnormal 나누기
# [samples, seq_len, x_dim]
abnormal_idx = SNP.loc[SNP['STATUS']==1].values[:,0]
normal_input = X_scaled.drop(list(abnormal_idx)).values.astype('float')
normal_input = normal_input.reshape((len(normal_input), 1, normal_input.shape[1]))
normal_date = X_date.drop(list(abnormal_idx)).values

abnormal_input = X_scaled.values[list(abnormal_idx)].astype('float')
abnormal_input = abnormal_input.reshape((len(abnormal_input), 1, abnormal_input.shape[1]))
abnormal_date = X_date.values[list(abnormal_idx)]
len(abnormal_input)

# 학습, 검증, 실험용으로 나누기
interval_n = int(len(normal_input)/10)
interval_ab = int(len(abnormal_input)/2)

# 학습 및 mean, std 생성용
normal_df1 = normal_input[0:interval_n*9]
normal_df2 = normal_input[interval_n*9:interval_n*10]

abnormal_df1 = abnormal_input[0:interval_ab]
abnormal_df2 = abnormal_input[interval_ab:interval_ab*2]

mean_df = normal_df1.mean()
std_df = normal_df1.std()

observation = normal_df1.shape[0]
seq_len = 1
x_dim = normal_df1.shape[-1]
h_dim = 100

# # prepare output sequence
# seq_out = x[:, 1:, :]
# n_out = seq_len - 1

class Lstm_AutoEncoder(keras.Model):
    def __init__(self, seq_len, x_dim, h_dim, name='autoencoder', **kwargs):
        super(Lstm_AutoEncoder, self).__init__(name=name, **kwargs)

    # define encoder
        self.inputlayer = layers.Input(shape=(seq_len, x_dim))
        self.encoder = layers.LSTM(h_dim, activation='relu', name='enc_lstm')(self.inputlayer)

        decoder = layers.RepeatVector(seq_len)(self.encoder)
        decoder = layers.LSTM(h_dim, activation='relu', return_sequences=True, name='dec_lstm')(decoder)
        self.decoder = layers.TimeDistributed(layers.Dense(x_dim))(decoder)

        self.model = Model(inputs=self.inputlayer, outputs=[self.decoder])
    #     self.model.compile(optimizer='adam', loss='mse')
    #
    # def call(self, x, epochs):
    #     x = tf.convert_to_tensor(x)
    #     self.model.fit(x, x, epochs=epochs, verbose=0)
    # def predict(self, x):
    #     return self.model.predict(x, verbose=0)

def anomaly_score(err, mean, std):
    x = (err-mean)
    return np.matmul(np.matmul(x, std), x.T)

AE = Lstm_AutoEncoder(seq_len=seq_len, x_dim=x_dim, h_dim=h_dim)
AE.model.compile(optimizer='adam', loss='mse')
AE.model.fit(normal_input, normal_input, epochs=300, verbose=0)

n_loss_list = []
for i in range(len(normal_df2)):
    ex_normal_df = np.expand_dims(normal_df2[i], axis=0)
    n_loss_list.append(AE.model.evaluate(ex_normal_df, ex_normal_df))

ab_loss_list =[]
for i in range(len(abnormal_df1)):
    ex_abnormal_df = np.expand_dims(abnormal_df1[i], axis=0)
    ab_loss_list.append(AE.model.evaluate(ex_abnormal_df, ex_abnormal_df))

ab_total_loss_list =[]
for i in range(len(abnormal_input)):
    ex_abnormal_df = np.expand_dims(abnormal_input[i], axis=0)
    ab_total_loss_list.append(AE.model.evaluate(ex_abnormal_df, ex_abnormal_df))

X_total_tst = X_scaled.values
X_total_tst = X_total_tst.reshape(len(X_scaled), 1, 14)

total_loss_list = []
for i in range(len(X_scaled)):
    total_data = X_total_tst
    df = np.expand_dims(total_data[i], axis=0)
    total_loss_list.append(AE.model.evaluate(df, df))




np.mean(n_loss_list)
np.mean(ab_total_loss_list)
np.mean(total_loss_list)

#
plt.plot(ab_total_loss_list)
plt.plot(n_loss_list, color='red')

# 전체와 abnormal 비교
# plt.plot(total_loss_list)
# plt.scatter(x=abnormal_idx, y=ab_total_loss_list, color='red')



# -----------data
# df = pd.read_csv("C://data_minsung/anomalyDetection/sensor.csv", index_col=0)
# df.head()
#
# ## 데이터 Type 변경
# df['date'] = pd.to_datetime(df['timestamp'])
# for var_index in [item for item in df.columns if 'sensor_' in item]:
#     df[var_index] = pd.to_numeric(df[var_index], errors='coerce')
# del df['timestamp']
#
# ## date를 index로 변환
# df = df.set_index('date')
# # 데이터 결측치 확인
# (df.isnull().sum()/len(df)).plot.bar(figsize=(18, 8), colormap='Paired')
# ## 중복된 데이터를 삭제
# df = df.drop_duplicates()
# ## 센서 15번, 센서 50 은 삭제
# del df['sensor_15']
# del df['sensor_50']
# ## 이전 시점의 데이터로 보간
# df = df.fillna(method='ffill')
#
# # 데이터 정규화
# df_temp = df.drop(['machine_status'], axis=1)
# df_scaled = keras.utils.normalize(df_temp, axis=1)
#
# normal_df = df_scaled[df['machine_status']=='NORMAL']
# abnormal_df = df_scaled[df['machine_status']!='NORMAL']
#
# ## 시계열 데이터이고, 입력의 형태가 특정 길이(window size)의 sequence 데이터 이므로 shuffle은 사용하지 않습니다.
# ## Normal 데이터는 학습데이터, 파라미터 설정데이터, 검증용데이터, 실험용데이터의 비율을 7:1:1:1 로 나누어서 사용합니다.
# interval_n = int(len(normal_df)/10)
# normal_df1 = normal_df.iloc[0:interval_n*7]
# normal_df2 = normal_df.iloc[interval_n*7:interval_n*8]
# normal_df3 = normal_df.iloc[interval_n*8:interval_n*9]
# normal_df4 = normal_df.iloc[interval_n*9:interval_n*10]
#
# interval_ab = int(len(abnormal_df)/2)
# abnormal_df1 = abnormal_df.iloc[0:interval_ab]
# abnormal_df2 = abnormal_df.iloc[interval_ab:interval_ab*2]
#
# mean_df = normal_df1.mean()
# std_df = normal_df1.std()

# # 데이터 구조만들기- index으로 불러오기
# def make_data_idx(dates, window_size):
#     input_idx = []
#     for idx in range(window_size-1, len(dates)):
#         cur_date = dates[idx].to_pydatetime()
#         in_date = dates[idx - (window_size-1)].to_pydatetime()
#
#         _in_period = (cur_date -in_date).days * 24 * 60 + (cur_date-in_date).seconds/60
#
#         if _in_period == (window_size-1):
#             input_idx.append(list(range(idx-window_size+1, idx+1)))
#     return input_idx
#
# dates = list(df.index)
# input_ids = make_data_idx(dates,1)

# def plot_sensor(temp_df, save_path='sample.gif'):
#     fig = plt.figure(figsize=(16, 6))
#     ## 에니메이션 만들기
#     camera = Camera(fig)
#     ax = fig.add_subplot(111)
#
#     ## 불량 구간 탐색 데이터
#     labels = temp_df['machine_status'].values.tolist()
#     dates = temp_df.index
#
#     for var_name in tqdm([item for item in df.columns if 'sensor_' in item]):
#         ## 센서별로 사진 찍기
#         temp_df[var_name].plot(ax=ax)
#         ax.legend([var_name], loc='upper right')
#
#         ## 고장구간 표시
#         temp_start = dates[0]
#         temp_date = dates[0]
#         temp_label = labels[0]
#
#         for xc, value in zip(dates, labels):
#             if temp_label != value:
#                 if temp_label == "BROKEN":
#                     ax.axvspan(temp_start, temp_date, alpha=0.2, color='blue')
#                 if temp_label == "RECOVERING":
#                     ax.axvspan(temp_start, temp_date, alpha=0.2, color='orange')
#                 temp_start = xc
#                 temp_label = value
#             temp_date = xc
#         if temp_label == "BROKEN":
#             ax.axvspan(temp_start, xc, alpha=0.2, color='blue')
#         if temp_label == "RECOVERING":
#             ax.axvspan(temp_start, xc, alpha=0.2, color='orange')
#         ## 카메라 찍기
#         camera.snap()
#
#     animation = camera.animate(500, blit=True)
#     # .gif 파일로 저장하면 끝!
#     animation.save(
#         save_path,
#         dpi=100,
#         savefig_kwargs={
#             'frameon': False,
#             'pad_inches': 'tight'
#         }
#     )
#
#
# plot_sensor(df)