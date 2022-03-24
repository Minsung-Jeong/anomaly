import tensorflow as tf
from tensorflow.keras import layers

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255, x_test/255

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])


modelDiffer = tf.keras.models.Sequential()
modelDiffer.add(layers.Flatten(input_shape=(28,28)))
modelDiffer.add(layers.Dense(128, activation='relu'))
modelDiffer.add(layers.Dropout(0.2))
modelDiffer.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

modelDiffer.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

modelDiffer.fit(x_train, y_train, epochs=5)