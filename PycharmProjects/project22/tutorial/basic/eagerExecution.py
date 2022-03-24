import tensorflow as tf

import cProfile
import matplotlib.pyplot as plt


tf.executing_eagerly()
x = [[2.]]
m = tf.matmul(x,x)
print('hello {}'.format(m))

a = tf.constant([[1,2],
                 [3,4]])
b = tf.add(a,1)
print(a*b)

c=  np.multiply(a,b)

tf.matmul(a,b)

def fizzbuzz(max_num):
  counter = tf.constant(0)
  max_num = tf.convert_to_tensor(max_num)
  for num in range(1, max_num.numpy()+1):
    num = tf.constant(num)
    if int(num % 3) == 0 and int(num % 5) == 0:
      print('FizzBuzz')
    elif int(num % 3) == 0:
      print('Fizz')
    elif int(num % 5) == 0:
      print('Buzz')
    else:
      print(num.numpy())
    counter += 1


# tf.GradientTape : 그래디언트 연산 추적을 위해 사용
w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
    loss = w*w
grad = tape.gradient(loss, w)
print(grad)


# ------------mnist 데이터 가져오기
(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()

dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float32),
   tf.cast(mnist_labels,tf.int64)))
dataset = dataset.shuffle(1000).batch(32)

# ==-------------------- 모델 생성
mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,[3,3], activation='relu',
                           input_shape=(None, None, 1)),
    tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10)
])

for images, labels in dataset.take(1):

    # print('logits: ', mnist_model(images[0:1]).numpy())
    print('logits: ', mnist_model(images[0]).numpy())

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_history = []

def train_step(images, labels):
    with tf.GradientTape() as tape:
        logits = mnist_model(images, training=True)
        # 결과의 형태 확인 위해 단언문 추가
        tf.debugging.assert_equal(logits.shape, (32,10))
        loss_value = loss_object(labels, logits)

    loss_history.append(loss_value.numpy().mean())
    grads = tape.gradient(loss_value, mnist_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, mnist_model.trainable_variables))

def train():
    for epoch in range(3):
        for (batch, (images, labels)) in enumerate(dataset):
            train_step(images, labels)
        print('에포크{} 종료'.format(epoch))

train()

plt.plot(loss_history)
plt.xlabel('Batch #')
plt.ylabel('Loss [entropy]')
# ------------------------------------------------------
a, y = tf.constant(3.0), tf.constant(8.0)
x = tf.Variable(10.0)

loss = tf.abs(y-a*x)
losss = tf.math.abs(y-a*x)
losss

def train_func():
    with tf.GradientTape() as tape:
        loss = tf.math.abs(a*x-y)
    dx = tape.gradient(loss, x)
    print('x={}, dx={:.2f}, loss={}'.format(x.numpy(), dx, loss))
    x.assign(x-dx)

x = tf.Variable(4.0)

with tf.GradientTape() as tape:
    y = x**2

# dy = (2x) * dx
dy_dx = tape.gradient(y,x)
dy_dx.numpy()

w = tf.Variable(tf.random.normal((4,2)), name='w')
b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
x = [[1.0, 2.0, 3.0, 4.0]]

with tf.GradientTape(persistent=True) as tape:
    y = x @ w + b
    loss = tf.reduce_mean(y**2)

[dl_dw, dl_db] = tape.gradient(loss, [w,b])

# ---------------------------------------------------즉시 실행에서 상태를 위한 객체 사용

if tf.config.experimental.list_physical_devices("GPU"):
    with tf.device("gpu:0"):
        print("GPU 사용가능")
        v = tf.Varialbe(tf.random.normal([1000,1000]))
        v = None



