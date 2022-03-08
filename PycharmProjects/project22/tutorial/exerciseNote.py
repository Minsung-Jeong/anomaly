import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

a = np.array([[1,2,3,4], [1,2,3,4]])
b = np.array([[10,20,30,40], [10,20,30,40]])

ab = tf.keras.backend.sum(a - b, axis=1)
np.shape(a)


model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28,28,1)),
    tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, strides=(2,2), activation='relu'
    ),
    tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, strides=(2,2), activation='relu'
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2+2)
])

model.summary()

decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim, )),
                tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7,7,32)),
                tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
                tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')
            ]
        )

decoder.summary()



from tensorflow.keras import layers, regularizers, activations
from tensorflow.keras import backend as K
import tensorflow.keras as keras

row_mark = 740
batch_size = 10
time_step = 1
x_dim = 4
lstm_h_dim = 8
z_dim = 4
epoch_num = 100
threshold = 0.03

inputs = keras.Input(shape=(time_step, x_dim), batch_size=10)
lstm = layers.LSTM(lstm_h_dim, activation='softplus', name='enc_lstm', stateful=True)
z_mean = layers.Dense(z_dim, name='z_mean')
z_logvar = layers.Dense(z_dim, name='z_logvar')

hidden = lstm(inputs)
mu = z_mean(hidden)
logvar = z_logvar(hidden)


R_model = keras.Sequential([
    keras.Input(shape=(time_step, x_dim), name="ddo"),
    layers.LSTM(lstm_h_dim, activation='softplus', name='DDODO'),
    layers.Dense(z_dim, name='dense')
])

R_model.summary()
R_model.get_config()


class Person:
    # 클래스 변수는 모든 객체가 공유함
    total_count = 0
    def __init__(self, name='wi', age=11):
        self.name = name
        self.age = age
        Person.total_count+=1
    def introduce(self):
        print('{0}은 "만{1}세" 이다'.format(self.name, self.age))
    def get_aged(self):
        self.age +=1

class Student(Person):
    def __init__(self):
        super().__init__()

    def introduce(self):
        super().introduce()
        print('그 학생은 건드리지 마쇼')

s2 = Student()
s2.introduce()
s2.get_aged()
s2.age

Student.mro()



# 0---------------------------------------------------------------------------------
batch_size = 32
time_step = 1
x_dim = 4
lstm_h_dim = 8
z_dim = 4

tf.keras.layers.Dense(z_dim, name='z_mean')

inputs = tf.keras.Input(shape=(time_step, x_dim), batch_size=5)
encoder_lstm = tf.keras.layers.LSTM(z_dim, activation='softplus', name='encoder_lstm', stateful=True)(inputs)



# ---------------------------------
Dataset = pd.DataFrame(np.random.rand(10000, 8))
train_scaled, test_scaled = split_normalize_data(Dataset)
x_dim = train_scaled.shape[1]

train_X = reshape(train_scaled)
test_X = reshape(test_scaled)

train_dataset = data.Dataset.from_tensor_slices(train_X)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)




enc = Encoder(time_step, x_dim, lstm_h_dim, z_dim)
dec = Decoder(time_step, x_dim, lstm_h_dim, z_dim)
inputs = np.stack(list(train_dataset))
np.shape(inputs)
mu_z, logvar_z, z = enc(inputs)