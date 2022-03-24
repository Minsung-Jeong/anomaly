import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
# import tensorflow_probability as tfp
import time
import os

os.getcwd()


(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()



def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  # np.where 은 조건에 맞으면 1, 아니면 0을 return
  return np.where(images > .5, 1.0, 0.0).astype('float32')

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)



train_size = 6000
batch_size = 32
test_size = 1000

train_dataset = (tf.data.Dataset.from_tensor_slices(train_images).
                 shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))

class CVAE(tf.keras.Model):

    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28,28,1)),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2,2), activation='relu'
                ),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=(2,2), activation='relu'
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim, )),
                tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7,7,32)),
                tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')
            ]
        )

    @tf.function
    # 시각화에 사용되는 sample, 모델 흐름에는 상관 없음
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean
        # return tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits



# 이 부분 바꿔도 잘 작동하는지 실험해보기,
# 비교대상 pdf 공식과 코드 내용 비교
def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)

    # 1차-good
    # return tf.reduce_sum(
    #     -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
    #     axis=raxis
    # )

    # 2차-good
    # return tf.reduce_sum((1/np.sqrt(2*np.pi*np.exp(logvar)))*np.exp(-((sample-mean)**2)/(2*np.exp(logvar))),
    #                      axis=raxis)

    # 3차 아무값-bad
    # return tf.reduce_sum((sample-mean)/logvar)

    # 4차 좀 연관 있어보이지만 다르게 설정-bad
    return tf.reduce_sum((3/np.sqrt(10*np.exp(logvar)))*np.exp(-((sample-mean)**3)/(7*np.exp(logvar))),
                         axis=raxis)


def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    # reconstruction error 부분이 cross entropy = decoder에 대해 베르누이를 가정했다.
    # 가우시안으로 하면 MSE 가 error가 될 것
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels = x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1,2,3])

    ## log 1 = 0 이므로 세번째 매개변수 맞음
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

@tf.function
def train_step(model, x, optimizer):
    # 모델 불러오는 부분
    ckpt.restore(manager.latest_checkpoint)

    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
        ckpt.step.assign_add(1)
        if int(ckpt.step) % 10 == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        print("loss {:1.2f}".format(loss))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# tf.function 이용할 때는 해당 명령통해 eager execution 하는 편인 듯
tf.config.run_functions_eagerly(True)
epochs = 4
latent_dim = 2
num_examples_to_generate = 16
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)
model.save_weights('vae_checkpoint')

optimizer = tf.keras.optimizers.Adam(1e-4)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(ckpt, './experimental_result/cvae_checkpoint/training_1/vae_ckpts', max_to_keep=None)

def generate_and_save_images(model, epoch, test_sample):
  mean, logvar = model.encode(test_sample)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z)
  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('cvae_image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

# Pick a sample of the test set for generating output images
assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
  test_sample = test_batch[0:num_examples_to_generate, :, :, :]

generate_and_save_images(model, 0, test_sample)

for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train_x in train_dataset:
    train_step(model, train_x, optimizer)
  end_time = time.time()
  loss = tf.keras.metrics.Mean()
  for test_x in test_dataset:
    loss(compute_loss(model, test_x))
  elbo = -loss.result()
  print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'.format(epoch, elbo, end_time - start_time))
  generate_and_save_images(model, epoch, test_sample)