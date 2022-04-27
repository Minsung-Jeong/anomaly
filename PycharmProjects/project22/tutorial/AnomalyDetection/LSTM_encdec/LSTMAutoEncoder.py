import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, models, layers, optimizers, utils
import tensorflow.keras as keras

class Encoder(keras.Model):
    def __init__(self, time_step, x_dim, lstm_h_dim, batch_size=None, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        # self.encoder_inputs = keras.Input(shape=(time_step, x_dim))
        """
        LSTM(return_state = True) : output(hidden_state), hidden_state, cell_state 출력
        LSTM(return_sequences = True) : output_sequence 출력(batch, seq, h_dim)
        data_inputs = [samples, timesteps, features]
        """
        # self.inputs = keras.Input(shape=(time_step, x_dim))
        self.encoder = layers.LSTM(lstm_h_dim, name='encoder_lstm', return_state=True)

    def call(self, x):
        # x_inputs = self.inputs(x)
        output, hidden, cell = self.encoder(x)
        return hidden, cell

# class Decoder(keras.Model):
class Decoder():
    def __init__(self, time_step, x_dim, h_dim, batch_size, hidden, cell, name='decoder', **kwargs):
        # super(Decoder, self).__init__(name=name, **kwargs)

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.batch_size = batch_size

        self.h_0 = tf.convert_to_tensor(hidden)
        self.c_0 = tf.convert_to_tensor(cell)

        self.inputs = keras.Input(shape=(time_step, x_dim), batch_size=batch_size)
        self.decoder = layers.LSTM(h_dim, name='decoder_lstm', return_sequences=True)
        self.lstm_out = self.decoder(self.inputs, initial_state=[self.h_0, self.c_0])


        # 논문은 linear지만 nn이 나을 것 같아서 Dense 사용
        self.output = layers.Dense(x_dim)(self.lstm_out)
        self.model = Model(inputs=self.inputs, outputs = self.output)

    # def get_zero_initial_state(self, inputs):
    #     return [tf.zeros((self.batch_size, self.h_dim)),tf.zeros((self.batch_size, self.h_dim))]

    def call(self, x):
        # self.model = Model(inputs=self.inputs, outputs=)
        # x_input = self.inputs(x)
        # lstm_out = self.decoder(x_input)
        # prediction = self.fc_layer(lstm_out)
        prediction = self.model(x)
        return prediction


# (timestep, x_dim)
inputs = tf.keras.Input(shape=(5,16))
lstm = layers.LSTM(8)(inputs)
model = tf.keras.Model(inputs=inputs, outputs=lstm)
model.summary()
seq_in = np.zeros((10,5,16))
lstm  = layers.LSTM(8)
a= lstm(seq_in)



np.shape(a)

# ---------------
# keras.Input 의 사용이유가 없는데?
x_dim = 1
seq_in = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                   ])
seq_in = seq_in.reshape((5,9, x_dim))

inputs = Input(shape=(9, 1))
# LSTM(return_state = True)이면 output(hidden_state), hidden_state, cell_state 출력
encoder_lstm = LSTM(30, activation='relu', return_state=True)(inputs)
model = Model(inputs=inputs, outputs=encoder_lstm)
a,b,c = model(seq_in)
np.shape(b)

batch_size = 1
inputs = Input(shape=(9, 1), batch_size=batch_size)
cell = layers.LSTMCell(30)
lstm = layers.RNN(cell, return_sequences=True, return_state=True)

c_0 = tf.convert_to_tensor(np.random.random([batch_size, 30]).astype(np.float32))
h_0 = tf.convert_to_tensor(np.random.random([batch_size, 30]).astype(np.float32))

lstm_out, hidden, cell =lstm(inputs, initial_state=[h_0, c_0])
output = layers.Dense(1)(lstm_out)
model3 = Model(inputs=inputs, outputs=[output, hidden, cell])



model_out, hidden, _= model3(seq_in)
np.shape(model_out)
np.shape(seq_in)
np.shape(hidden)
