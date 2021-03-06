# def __init__ 은 생성자
# def __call__ 은 인스턴스 호출 후,

import numpy as np

np.random.seed(0)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

import tensorflow as tf

tf.random.set_seed(0)
from tensorflow import keras, data
import tensorflow_probability as tfp
from tensorflow.keras import layers, regularizers, activations
from tensorflow.keras import backend as K
import seaborn as sns
import matplotlib.pyplot as plt


# Dataset = pd.DataFrame(np.random.rand(10000, 8))
Dataset = np.random.rand(10000,8)

dataset_name = "bearing_dataset"
# train_ratio = 0.75

trn_size = int(len(Dataset)*0.5)
batch_size = 10
time_step = 1
x_dim = 4
lstm_h_dim = 8
z_dim = 4
epoch_num = 10
threshold = 0.03

mode = 'train'
model_dir = "./lstm_vae_model/"
image_dir = "./lstm_vae_images/"

def split_normalize_data(all_df):
    # row_mark = int(all_df.shape[0] * train_ratio)
    train_df = all_df[:trn_size]
    test_df = all_df[trn_size:]

    scaler = MinMaxScaler()

    # scaler.fit(np.array(all_df)[:, 1:])
    # train_scaled = scaler.transform(np.array(train_df)[:, 1:])
    # test_scaled = scaler.transform(np.array(test_df)[:, 1:])

    scaler.fit(np.array(all_df)[:, :])
    train_scaled = scaler.transform(np.array(train_df)[:, :])
    test_scaled = scaler.transform(np.array(test_df)[:, :])

    return train_scaled, test_scaled

def reshape(input):
    input = input.__array__()
    return np.reshape(input, (input.shape[0], time_step, input.shape[1])).astype("float32")
    # return da.reshape(da.shape[0], time_step, da.shape[1]).astype("float32")


class Sampling(layers.Layer):
    def __init__(self, name='sampling_z'):
        # 부모 클래스에 넘길 값을 마지막 괄호에 기입
        super(Sampling, self).__init__(name=name)

    def call(self, inputs):
        mu, logvar = inputs
        # print('mu: ', mu)
        sigma = K.exp(logvar * 0.5)
        epsilon = K.random_normal(shape=(mu.shape[0], z_dim), mean=0.0, stddev=1.0)
        return mu + epsilon * sigma



class Encoder(layers.Layer):
    def __init__(self, time_step, x_dim, lstm_h_dim, z_dim, name='encoder', **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        # stateful 하려면 batch_size 나 batch_input_shape 설정이 있어야 하는데???
        self.encoder_inputs = keras.Input(shape=(time_step, x_dim))
        # stateful = True 이면 배치 내의 i번째 샘플의 마지막 state가, 다음 배치의 i번째 initial state로 사용
        # This assumes a one-to-one mapping between samples in different successive batches. = 굳이 사용해야 하나???
        # stateful 하려면 batch_input_shape 를 첫 레이어에 넣어야 하는데 그런 부분이 없어서 에러 나올 확률이 있다
        self.encoder_lstm = layers.LSTM(lstm_h_dim, activation='softplus', name='encoder_lstm')
        # self.encoder_lstm = layers.LSTM(lstm_h_dim, activation='softplus', name='encoder_lstm', stateful=True)
        self.z_mean = layers.Dense(z_dim, name='z_mean')
        self.z_logvar = layers.Dense(z_dim, name='z_log_var')
        self.z_sample = Sampling()

    def call(self, inputs):
        self.encoder_inputs = inputs
        hidden = self.encoder_lstm(self.encoder_inputs)
        mu_z = self.z_mean(hidden)
        logvar_z = self.z_logvar(hidden)
        z = self.z_sample((mu_z, logvar_z))
        return mu_z, logvar_z, z

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'name': self.name,
            'z_sample': self.z_sample.get_config()
        })
        return config

class Decoder(layers.Layer):
    def __init__(self, time_step, x_dim, lstm_h_dim, z_dim, name='decoder', **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.z_inputs = layers.RepeatVector(time_step, name='repeat_vector')
        self.decoder_lstm_hidden = layers.LSTM(lstm_h_dim, activation='softplus', return_sequences=True,
                                               name='decoder_lstm')
        self.x_mean = layers.Dense(x_dim, name='x_mean')
        self.x_sigma = layers.Dense(x_dim, name='x_sigma', activation='tanh')

    def call(self, inputs):
        z = self.z_inputs(inputs)
        hidden = self.decoder_lstm_hidden(z)
        mu_x = self.x_mean(hidden)
        sigma_x = self.x_sigma(hidden)
        return mu_x, sigma_x

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            'name': self.name
        })
        return config

# 학습용이 아님(찍어서 보는 용도)
loss_metric = keras.metrics.Mean(name='loss')
likelihood_metric = keras.metrics.Mean(name='log likelihood')


class LSTM_VAE(keras.Model):
    def __init__(self, time_step, x_dim, lstm_h_dim, z_dim, name='lstm_vae', **kwargs):
        super(LSTM_VAE, self).__init__(name=name, **kwargs)

        self.encoder = Encoder(time_step, x_dim, lstm_h_dim, z_dim, **kwargs)
        self.decoder = Decoder(time_step, x_dim, lstm_h_dim, z_dim, **kwargs)

    def call(self, inputs):
        mu_z, logvar_z, z = self.encoder(inputs)
        mu_x, sigma_x = self.decoder(z)

        var_z = tf.math.exp(logvar_z)
        # 원래 vae loss와 크게 다르지 않음
        # print("shape of 1.var_z, 2.logvar_z, 3.mu_z", np.shape(var_z), np.shape(logvar_z), np.shape(mu_z))


        # kl_loss = K.mean(-0.5 * K.sum(var_z - logvar_z + tf.square(1 - mu_z), axis=1), axis=0)
        kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(var_z - logvar_z + tf.square(1 - mu_z), axis=1), axis=0, name='kl_loss')
        # kl_loss = tf.reduce_mean(-0.5 * tf.zeros_like(mu_z))
        # print("kl_loss=========", kl_loss)
        # 상속관계에 있는 모듈의 add_loss


        # 이 식(람다)이 문제
        # self.add_loss(lambda: kl_loss)
        self.add_loss(kl_loss)
        dist = tfp.distributions.Normal(loc=mu_x, scale=tf.abs(sigma_x))
        log_px = -dist.log_prob(inputs)

        return mu_x, sigma_x, log_px

    def get_config(self):
        config = {
            'encoder': self.encoder.get_config(),
            'decoder': self.decoder.get_config(),
            'name': self.name
        }
        return config

    # gaussian 가정했을 때의 reconstruct loss
    def reconstruct_loss(self, x, mu_x, sigma_x):
        var_x = K.square(sigma_x)
        reconst_loss = -0.5 * K.sum(K.log(var_x), axis=2) + K.sum(K.square(x - mu_x) / var_x, axis=2)
        # print("recons loss shape:", np.shape(reconst_loss))
        reconst_loss = K.reshape(reconst_loss, shape=(x.shape[0], 1))
        # return K.mean(reconst_loss, axis=0)
        return tf.reduce_mean(reconst_loss, axis=0, name='reconstruct')

    def mean_log_likelihood(self, log_px):
        # print("log_px shape", np.shape(log_px))
        log_px = K.reshape(log_px, shape=(log_px.shape[0], log_px.shape[2]))
        # mean_log_px = K.mean(log_px, axis=1)
        mean_log_px = tf.reduce_mean(log_px, axis=1, name='mean_log_likeli_px')

        # return K.mean(mean_log_px, axis=0)
        return tf.reduce_mean(mean_log_px, axis=0, name='meanOfmean')

    def train_step(self, data):
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        with tf.GradientTape() as tape:
            mu_x, sigma_x, log_px = self(x)
            loss = self.reconstruct_loss(x, mu_x, sigma_x)
            # add_loss 통해 추가되었던 KLD loss를 추가
            loss += sum(self.losses)
            mean_log_px = self.mean_log_likelihood(log_px)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        loss_metric.update_state(loss)
        likelihood_metric.update_state(mean_log_px)
        return {'loss': loss_metric.result(), 'log_likelihood': likelihood_metric.result()}


def plot_loss_moment(history):
    _, ax = plt.subplots(figsize=(14, 6), dpi=80)
    ax.plot(history['loss'], 'blue', label='Loss', linewidth=1)
    ax.plot(history['log_likelihood'], 'red', label='Log likelihood', linewidth=1)
    ax.set_title('Loss and log likelihood over epochs')
    ax.set_ylabel('Loss and log likelihood')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    plt.savefig(image_dir + 'loss_lstm_vae_' + mode + '.png')


def plot_log_likelihood(df_log_px):
    plt.figure(figsize=(14, 6), dpi=80)
    plt.title("Log likelihood")
    sns.set_color_codes()
    sns.distplot(df_log_px, bins=40, kde=True, rug=True, color='blue')
    plt.savefig(image_dir + 'log_likelihood_' + mode + '.png')


def save_model(model):
    with open(model_dir + 'lstm_vae.json', 'w') as f:
        f.write(model.to_json())
    model.save_weights(model_dir + 'lstm_vae_ckpt')


def load_model():
    lstm_vae_obj = {'Encoder': Encoder, 'Decoder': Decoder, 'Sampling': Sampling}
    with keras.utils.custom_object_scope(lstm_vae_obj):
        with open(model_dir + 'lstm_vae.json', 'r'):
            model = keras.models.model_from_json(model_dir + 'lstm_vae.json')
        model.load_weights(model_dir + 'lstem_vae_ckpt')
    return model

# 0-00----------

    # dataset = Dataset.get_by_name(ws, dataset_name)
dataset = Dataset
print("Dataset found: ", dataset_name)
all_df = dataset
train_scaled, test_scaled = split_normalize_data(all_df)
x_dim = train_scaled.shape[1]
print("train and test data shape after scaling: ", train_scaled.shape, test_scaled.shape)


train_X = reshape(train_scaled)
test_X = reshape(test_scaled)

opt = keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-6, amsgrad=True)

if mode == "train":
    model = LSTM_VAE(time_step, x_dim, lstm_h_dim, z_dim, dtype='float32')
    model.compile(optimizer=opt)
    train_dataset = data.Dataset.from_tensor_slices(train_X)
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)

    history = model.fit(train_dataset, epochs=epoch_num, shuffle=False).history

    model.summary()
    plot_loss_moment(history)
    save_model(model)
elif mode == "infer":
    model = load_model()
    model.compile(optimizer=opt)
else:
    print("Unknown mode: ", mode)
    exit(1)

_, _, train_log_px = model.predict(train_X, batch_size=1)
train_log_px = train_log_px.reshape(train_log_px.shape[0], train_log_px.shape[2])
df_train_log_px = pd.DataFrame()
df_train_log_px['log_px'] = np.mean(train_log_px, axis=1)
plot_log_likelihood(df_train_log_px)

_, _, test_log_px = model.predict(test_X, batch_size=1)
test_log_px = test_log_px.reshape(test_log_px.shape[0], test_log_px.shape[2])
df_log_px = pd.DataFrame()
df_log_px['log_px'] = np.mean(test_log_px, axis=1)
df_log_px = pd.concat([df_train_log_px, df_log_px])
df_log_px['threshold'] = 0.65
df_log_px['anomaly'] = df_log_px['log_px'] > df_log_px['threshold']
df_log_px.index = np.array(all_df)[:, 0]

df_log_px.plot(logy=True, figsize=(16, 9), color=['blue', 'red'])
plt.savefig(image_dir + 'anomaly_lstm_vae_' + mode + '.png')




# # ------------
# def main():
#     try:
#         # dataset = Dataset.get_by_name(ws, dataset_name)
#         dataset = Dataset
#         print("Dataset found: ", dataset_name)
#     except Exception:
#         print("Dataset not found: ", dataset_name)
#
#     all_df = dataset
#     train_scaled, test_scaled = split_normalize_data(all_df)
#     x_dim = train_scaled.shape[1]
#     print("train and test data shape after scaling: ", train_scaled.shape, test_scaled.shape)
#
#     train_X = reshape(train_scaled)
#     test_X = reshape(test_scaled)
#
#     opt = keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-6, amsgrad=True)
#
#     if mode == "train":
#         model = LSTM_VAE(time_step, x_dim, lstm_h_dim, z_dim, dtype='float32')
#         model.compile(optimizer=opt)
#         train_dataset = data.Dataset.from_tensor_slices(train_X)
#         train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)
#
#         history = model.fit(train_dataset, epochs=epoch_num, shuffle=False).history
#
#         model.summary()
#         plot_loss_moment(history)
#         save_model(model)
#     elif mode == "infer":
#         model = load_model()
#         model.compile(optimizer=opt)
#     else:
#         print("Unknown mode: ", mode)
#         exit(1)
#
#     _, _, train_log_px = model.predict(train_X, batch_size=1)
#     train_log_px = train_log_px.reshape(train_log_px.shape[0], train_log_px.shape[2])
#     df_train_log_px = pd.DataFrame()
#     df_train_log_px['log_px'] = np.mean(train_log_px, axis=1)
#     plot_log_likelihood(df_train_log_px)
#
#     _, _, test_log_px = model.predict(test_X, batch_size=1)
#     test_log_px = test_log_px.reshape(test_log_px.shape[0], test_log_px.shape[2])
#     df_log_px = pd.DataFrame()
#     df_log_px['log_px'] = np.mean(test_log_px, axis=1)
#     df_log_px = pd.concat([df_train_log_px, df_log_px])
#     df_log_px['threshold'] = 0.65
#     df_log_px['anomaly'] = df_log_px['log_px'] > df_log_px['threshold']
#     df_log_px.index = np.array(all_df)[:, 0]
#
#     df_log_px.plot(logy=True, figsize=(16, 9), color=['blue', 'red'])
#     plt.savefig(image_dir + 'anomaly_lstm_vae_' + mode + '.png')
#
#
# if __name__ == "__main__":
#     main()