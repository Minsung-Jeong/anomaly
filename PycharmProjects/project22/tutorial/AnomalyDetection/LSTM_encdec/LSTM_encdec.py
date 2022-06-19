import numpy as np
import tensorflow as tf
import pandas as pd
# import tqdm
from tqdm.notebook import trange, tqdm
from tensorflow.keras import Model, models, layers, optimizers, utils
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from celluloid import Camera

# (samples, seq_length, x_dim)
x = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])
np.shape(x)
seq_len = len(x)
x_dim = 9
x = x.reshape((1, seq_len, x_dim))
# prepare output sequence
seq_out = x[:, 1:, :]
n_out = seq_len - 1
h_dim = 100

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
        self.model.compile(optimizer='adam', loss='mse')

    def call(self, x, epochs):
        self.model.fit(x, x, epochs=epochs, verbose=0)
    def predict(self, x):
        return self.model.predict(x, verbose=0)

def anomaly_score(err, mean, std):
    x = (err-mean)
    return np.matmul(np.matmul(x, std), x.T)
model = Lstm_AutoEncoder(seq_len=seq_len, x_dim=x_dim, h_dim=h_dim)
model(x=x, epochs=300)
predict4 = model.predict(x)

# -----------data

df = pd.read_csv("C://data_minsung/anomalyDetection/sensor.csv", index_col=0)
df.head()

## 데이터 Type 변경
df['date'] = pd.to_datetime(df['timestamp'])
for var_index in [item for item in df.columns if 'sensor_' in item]:
    df[var_index] = pd.to_numeric(df[var_index], errors='coerce')
del df['timestamp']

## date를 index로 변환
df = df.set_index('date')

# 데이터 결측치 확인
(df.isnull().sum()/len(df)).plot.bar(figsize=(18, 8), colormap='Paired')

## 중복된 데이터를 삭제
df = df.drop_duplicates()

## 센서 15번, 센서 50 은 삭제
del df['sensor_15']
del df['sensor_50']

## 이전 시점의 데이터로 보간
df = df.fillna(method='ffill')

normal_df = df[df['machine_status']=='NORMAL']
abnormal_df = df[df['machine_status']!='NORMAL']


def plot_sensor(temp_df, save_path='sample.gif'):
    fig = plt.figure(figsize=(16, 6))
    ## 에니메이션 만들기
    camera = Camera(fig)
    ax = fig.add_subplot(111)

    ## 불량 구간 탐색 데이터
    labels = temp_df['machine_status'].values.tolist()
    dates = temp_df.index

    for var_name in tqdm([item for item in df.columns if 'sensor_' in item]):
        ## 센서별로 사진 찍기
        temp_df[var_name].plot(ax=ax)
        ax.legend([var_name], loc='upper right')

        ## 고장구간 표시
        temp_start = dates[0]
        temp_date = dates[0]
        temp_label = labels[0]

        for xc, value in zip(dates, labels):
            if temp_label != value:
                if temp_label == "BROKEN":
                    ax.axvspan(temp_start, temp_date, alpha=0.2, color='blue')
                if temp_label == "RECOVERING":
                    ax.axvspan(temp_start, temp_date, alpha=0.2, color='orange')
                temp_start = xc
                temp_label = value
            temp_date = xc
        if temp_label == "BROKEN":
            ax.axvspan(temp_start, xc, alpha=0.2, color='blue')
        if temp_label == "RECOVERING":
            ax.axvspan(temp_start, xc, alpha=0.2, color='orange')
        ## 카메라 찍기
        camera.snap()

    animation = camera.animate(500, blit=True)
    # .gif 파일로 저장하면 끝!
    animation.save(
        save_path,
        dpi=100,
        savefig_kwargs={
            'frameon': False,
            'pad_inches': 'tight'
        }
    )


plot_sensor(df)