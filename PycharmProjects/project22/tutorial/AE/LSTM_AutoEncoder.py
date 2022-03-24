import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, models, layers, optimizers, utils
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.utils import plot_model

# -------------first lstm auto-encoder
def temporalize(X, y, lookback):
    output_X = []
    output_y = []
    for i in range(len(X)-lookback-1):
        t = []
        for j in range(1,lookback+1):
            # Gather past records upto the lookback period
            t.append(X[[(i+j+1)], :])
        output_X.append(t)
        output_y.append(y[i+lookback+1])
    return output_X, output_y


timeseries = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                       [0.1**3, 0.2**3, 0.3**3, 0.4**3, 0.5**3, 0.6**3, 0.7**3, 0.8**3, 0.9**3]]).transpose()
np.shape(timeseries)
timesteps = timeseries.shape[0]
n_features = timeseries.shape[1]

timesteps = 3
X, y = temporalize(X = timeseries, y = np.zeros(len(timeseries)), lookback = timesteps)

n_features = 2
X = np.array(X)
np.shape(X)
# 결과 데이터 형태 = [batch, seq(timesteps), n_features]
X = X.reshape(X.shape[0], timesteps, n_features)
np.shape(X)

model = models.Sequential()
model.add(layers.LSTM(128, activation='relu', input_shape=(timesteps, n_features), return_sequences=True))
# return sequence가 없으면 (none, 64) = 마지막 cell에서 나오는 output만 가지고 간다, 있으면 (none, 3, 64),
# return_sequences=False로 설정하면, 그 다음에는 RepeatVector가 이용될 수 밖에
model.add(layers.LSTM(64, activation='relu', return_sequences=False)) #false면 마지막 output. true면 전체 sequence
model.add(layers.RepeatVector(timesteps))
model.add(layers.LSTM(64, activation='relu', return_sequences=True))
model.add(layers.LSTM(128, activation='relu', return_sequences=True))
model.add(layers.TimeDistributed(layers.Dense(n_features)))
model.compile(optimizer='adam', loss='mse')

model.summary()

model.fit(X, X, epochs=300, batch_size=5, verbose=0)
yhat = model.predict(X, verbose=0)
print('---------predict-----')
print(np.round(yhat,1))
print('---------actual---------')
print(np.round(X, 1))
print('비교 : \t', np.round(yhat,1)==np.round(X,1))

#-------------------------------second lstm auto-encoder
# define input sequence
sequence = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# reshape input into [samples(batch), timesteps(seq), features]
n_in = len(sequence)
sequence = np.reshape(sequence, (1,n_in, 1))

seq_out = sequence[:,1:,:]
n_out = n_in - 1

# temp data
sequence = tf.zeros([90])
timesteps = 9
x_dim = 1
sequence = tf.reshape(sequence, (10, timesteps, x_dim))
trn_data = tf.data.Dataset.from_tensor_slices(sequence)
trn_data = trn_data.shuffle(buffer_size=100).batch(batch_size=2)

trn_data = np.zeros((10,9,1))
trn_data = tf.data.Dataset.from_tensor_slices(trn_data)
# define model
model = models.Sequential()
# model.add(layers.Input(shape=(n_in,1)))
model.add(layers.Input(shape=(timesteps,x_dim)))
model.add(layers.LSTM(100, activation='relu'))
# RepeatVector : Repeats the input n times
# model.add(layers.RepeatVector(n_in))
model.add(layers.RepeatVector(timesteps))
# return_sequences = True 면, last ouput 도 출력
model.add(layers.LSTM(100, activation='relu', return_sequences=True))
# 시계열을 고려한 연산 가능(?) - 차원 둘째자리가 timestamp으로 인지하는 방식인 듯,
# ex) (32, 10, 128, 128, 3)이 주어지면 10개의 temporal dimension 에 대해 같은 layer를(같은 layer가 포인트인 듯)
model.add(layers.TimeDistributed(layers.Dense(1)))
model.compile(optimizer='adam', loss='mse')
model.summary()
# fit model
for subset in trn_data:
    model.fit(subset, subset, epochs=100, verbose=1)
yhat = model.predict(sequence)
np.shape(yhat)
# ----------------------------third composition lstm auto-encoder(reconstruct + predict)
# define input sequence
seq_in = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# reshape input into [samples, timesteps, features]
n_in = len(seq_in)
seq_in = seq_in.reshape((1, n_in, 1))
# prepare output sequence
seq_out = seq_in[:, 1:, :]
n_out = n_in - 1
# define encoder
visible = Input(shape=(n_in, 1))
encoder = LSTM(100, activation='relu')(visible)

decoder1 = RepeatVector(n_in)(encoder)
decoder1 = LSTM(100, activation='relu', return_sequences=True)(decoder1)
decoder1 = TimeDistributed(Dense(1))(decoder1)

decoder2 = RepeatVector(n_out)(encoder)
decoder2 = LSTM(100, activation='relu', return_sequences=True)(decoder2)
decoder2 = TimeDistributed(Dense(1))(decoder2)

model = Model(inputs=visible, outputs=[decoder1, decoder2])
model.compile(optimizer='adam', loss='mse')
plot_model(model, show_shapes=True, to_file='composite_lstm_autoencoder.png')
# fit model
model.fit(seq_in, [seq_in,seq_out], epochs=300, verbose=0)
# demonstrate prediction
yhat = model.predict(seq_in, verbose=0)
print(yhat)