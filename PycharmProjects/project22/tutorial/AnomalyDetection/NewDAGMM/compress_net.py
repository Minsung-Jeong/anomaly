import tensorflow as tf
import numpy as np


class CompressNet:
    """
    This network(Auto-encoder) converts input to
    1. (compressed) low dimensional Representation
    2. reconstruction error
    """

    def __init__(self, hidden_layer_sizes, activation=tf.nn.tanh):
        """
        param
        ------------------------
        1. hidden_layer_size : list of int(hidden layer sizes)
        ex) hidden_layer_size = [n1, n2]
        input => n1 => n2 => n1 => input
        2. activation : activation function of hidden layer
        ------------------------
        """

        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation

    def encoder(self, x):
        self.input_size = x.shape[1]

        n_layer = 0
        encoder = tf.keras.Sequential(name='encoder')
        # input layer 추가해주고 가는 것도 좋을 것 같은데??

        for size in self.hidden_layer_sizes[:-1]:
            n_layer += 1
            encoder.add(tf.keras.layers.Dense(size,
                                              activation=self.activation,
                                              name='layer_{}'.format(n_layer)))
        n_layer += 1
        encoder.add(tf.keras.layers.Dense(self.hidden_layer_sizes[-1],
                                          name='layer_{}'.format(n_layer)
                                          ))
        return encoder(x)

    def decoder(self, z):
        n_layer = 0
        decoder = tf.keras.Sequential(name='decoder')
        for size in self.hidden_layer_sizes[:-1][::-1]:
            n_layer += 1
            decoder.add(tf.keras.layers.Dense(size,
                                              activation=self.activation,
                                              name="layer_{}".format(n_layer)))
        n_layer += 1
        decoder.add(tf.keras.layers.Dense(self.input_size,
                                          name="layer_{}".format(n_layer)))
        return decoder(z)





    def loss_as_feature(self, x, x_re):
        """ Loss for z_r, there are two losses
        1. loss_E : relative Euclidean distance
        2. loss_C : Cosine similarity
        """

        # input type must be tf.float
        norm_x = tf.norm(x, ord='euclidean', axis=1)
        norm_x_re = tf.norm(x_re, ord='euclidean', axis=1)
        dist_x = tf.norm(x - x_re, axis=1)
        dot_x = tf.reduce_sum(x*x_re, axis=1)

        min_val = 1e-3
        loss_E = dist_x / (norm_x + min_val)
        loss_C = 0.5 * (1.0 - dot_x / (norm_x * norm_x_re + min_val))
        return tf.concat([loss_E[:, None], loss_C[:, None]], axis=1)

    def extract_feature(self, x, x_re, z_c):
        z_r = self.loss_as_feature(x, x_re)
        return tf.concat([z_c, z_r], axis=1)

    def compress(self, x):
        """convert input x to output z=[z_c,z_r]
        z is the input of the GMM algorithm
        It is composed of low dimensional representation, reconstruction error
        -------
        x : tf.Tensor shape : (n_samples, n_features)
        x_re : reconstructed x value
        -------
        z : tf.Tensor shape : (n_samples, n2 + 2)
            Result data
            Second dimension of this data is equal to
            sum of compressed representation size and
            number of loss function (=2)
        """
        z_c = self.encoder(x)
        x_re = self.decoder(z_c)

        z = self.extract_feature(x, x_re, z_c)

        return z, x_re

    def reconstruction_error(self, x, x_re):
        return tf.reduce_mean(tf.reduce_sum(
            tf.square(x - x_re), axis=1), axis=0)


# 데이터 형태에 맞는 방식으로 불가, 2차원에 입력에 맞춰진 형태
x = tf.ones([32,180], dtype=tf.float32)

model = CompressNet([60,30, 10, 1])
z_c = model.encoder(x)
z_c.shape
x_re = model.decoder(z)
x_re.shape
model.compress(x)

