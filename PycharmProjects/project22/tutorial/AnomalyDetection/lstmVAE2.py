# def __init__ 은 생성자
# def __call__ 은 인스턴스 호출 후,
import numpy as np
import os
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

mode = 'train'
model_dir = "./lstm_vae_model/"
image_dir = "./lstm_vae_images/"


Dataset = np.random.rand(10000,8)
trn_size = int(len(Dataset)*0.5)
batch_size = 10
time_step = 1
x_dim = 8
lstm_h_dim = 8
z_dim = 4
epoch_num = 10
threshold = 0.03

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

loss_metric = keras.metrics.Mean(name='loss')
likelihood_metric = keras.metrics.Mean(name='log likelihood')

# model 형식으로 만들어보자
class LSTM_VAE(tf.keras.Model):
    def __init__(self, time_step, x_dim, lstm_h_dim, z_dim, name='lstm_vae', **kwargs):
        super(LSTM_VAE, self).__init__(name=name, **kwargs)
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(time_step, x_dim), name='encoder_input'),
                tf.keras.layers.LSTM(lstm_h_dim, activation='softplus', name='encoder_lstm'),
                tf.keras.layers.Dense(z_dim + z_dim, name='enc_representation')
            ], name='encoder'
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.RepeatVector(time_step, name='repeat_vector'),
                tf.keras.layers.LSTM(lstm_h_dim, activation='softplus', return_sequences=True,
                                               name='decoder_lstm'),
                tf.keras.layers.Dense(x_dim + x_dim)
            ], name='decoder'
        )

    def call(self, x):
        # mean, logvar 값을 구한 뒤 reparameterize
        print('encoding 시작')
        mu_z, logvar_z = tf.split(self.encoder(x), num_or_size_splits=2, axis=-1)
        eps = tf.random.normal(shape=mu_z.shape)
        z = eps * tf.exp(logvar_z * 0.5) + mu_z

        mu_x, sigma_x = tf.split(self.decoder(z), num_or_size_splits=2, axis=-1)
        var_z = tf.math.exp(logvar_z)

        kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(var_z - logvar_z + tf.square(1 - mu_z), axis=1), axis=0, name='kl_loss')
        # 논문에서  전개한 식
        # kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(var_z - logvar_z - 1 + tf.square(mu_z), axis=1), axis=0, name='kl_loss')


        self.add_loss(kl_loss)

        dist = tfp.distributions.Normal(loc=mu_x, scale=tf.abs(sigma_x))
        log_px = -dist.log_prob(x)
        print('call 완료')
        return mu_x, sigma_x, log_px

    # Model 상속해서 만드는 것인데 왜 명시해줘야 작동하지???
    def get_config(self):
        config = {
            'encoder': self.encoder.get_config(),
            'decoder': self.decoder.get_config(),
            'name': self.name
        }
        return config

    def reconstruct_loss(self, x, mu_x, sigma_x):
        var_x = tf.square(sigma_x)
        reconst_loss = -0.5 * tf.reduce_sum(tf.math.log(var_x), axis=2) + tf.reduce_sum(tf.square(x - mu_x)/ var_x, axis=2)
        reconst_loss = tf.reshape(reconst_loss, shape=(x.shape[0], 1))
        return tf.reduce_mean(reconst_loss, axis=0, name='reconstruct')

    def mean_log_likelihood(self, log_px):
        log_px = tf.reshape(log_px, shape=(log_px.shape[0], log_px.shape[2]))
        mean_log_px = tf.reduce_mean(log_px, axis=1, name='mean_log_likeli_px')
        return tf.reduce_mean(mean_log_px, axis=0, name='meanofmean')

    def train_step(self, data):
        if isinstance(data, tuple):
            x = data[0]
        else:
            x = data

        with tf.GradientTape() as tape:
            mu_x, sigma_x, log_px = self(x)
            loss = self.reconstruct_loss(x, mu_x, sigma_x)
            loss += sum(self.losses)
            print('loss까지 왔다')
            mean_log_px = self.mean_log_likelihood(log_px)

        # compute gradients
        grads = tape.gradient(loss, self.trainable_variables)
        # update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        loss_metric.update_state(loss)
        likelihood_metric.update_state(mean_log_px)
        # print('-----------done with train step, loss:', loss)
        # return loss
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



# reparameterize에 사용되는 파라미터는 굳이 저장할 필요 없을 것 같으므로 삭제
def load_model():
    # lstm_vae_obj = {'Encoder': LSTM_VAE.encoder, 'Decoder': LSTM_VAE.decoder, 'Sampling': Sampling}
    lstm_vae_obj = {'Encoder': LSTM_VAE.encoder, 'Decoder': LSTM_VAE.decoder}
    with keras.utils.custom_object_scope(lstm_vae_obj):
        with open(model_dir + 'lstm_vae.json', 'r'):
            model = keras.models.model_from_json(model_dir + 'lstm_vae.json')
        model.load_weights(model_dir + 'lstem_vae_ckpt')
    return model





# tf.function 이용할 때는 해당 명령통해 eager execution 하는 편인 듯
# tf.config.run_functions_eagerly(True)
# Dataset = pd.DataFrame(np.random.rand(10000, 8))

all_df = Dataset
train_scaled, test_scaled = split_normalize_data(all_df)
x_dim = train_scaled.shape[1]
print("train and test data shape after scaling: ", train_scaled.shape, test_scaled.shape)

train_X = reshape(train_scaled)
test_X = reshape(test_scaled)

# train_ratio = 0.75

mode = 'train'
model_dir = "./lstm_vae_model/"
image_dir = "./lstm_vae_images/"

opt = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-6, amsgrad=True)
if mode == "train":
    model = LSTM_VAE(time_step, x_dim, lstm_h_dim, z_dim, dtype='float32')
    ## https: // www.tensorflow.org / guide / keras / customizing_what_happens_in_fit
    model.compile(optimizer=opt)
    train_dataset = tf.data.Dataset.from_tensor_slices(train_X)
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

