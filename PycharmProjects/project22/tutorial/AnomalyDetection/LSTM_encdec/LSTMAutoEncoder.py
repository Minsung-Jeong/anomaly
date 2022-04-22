import numpy as np
import tensorflow as tf

from tensorflow.keras import Model, models, layers, optimizers, utils
import tensorflow.keras as keras

class Encoder(layers.layer):
    def __init__(self, time_step, x_dim, lstm_h_idm, batch_size ,name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        # state 이용하여면 batch_shape=(batch, sez, dim)으로 넣어줘야 함
        self.encoder_inputs = keras.Input(shape=(time_step, x_dim))

        """
        lstm = tf.keras.layers.LSTM(4, return_sequences=True, return_state=True)
        whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
        """

        self.encoder_lstm = layers.LSTM(lstm_h_idm, return_state=True ,activation='tanh', name='encoder_lstm')

    def call(self, inputs):
        self.encoder_inputs = inputs
        output = self.encoder_lstm(self.encoder_inputs)


