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
row_mark = int(len(Dataset)*0.5)
batch_size = 10
time_step = 1
x_dim = 4
lstm_h_dim = 8
z_dim = 4
epoch_num = 100
threshold = 0.03

Dataset = pd.DataFrame(np.random.rand(10000, 8))
train_scaled, test_scaled = split_normalize_data(Dataset)
x_dim = train_scaled.shape[1]
np.shape(train_scaled)

train_X = reshape(train_scaled)
test_X = reshape(test_scaled)

np.shape(train_X)


m = tf.keras.metrics.Mean()
m.update_state([1,2,3,4])
m.update_state([0,0,0,0])
m.update_state([0,0,0,0])
m.result()



# 상속 체크
inputs = tf.keras.layers.Input((32,))
outputs = tf.keras.layers.Dense(1)(inputs)
abc = tf.keras.Model(inputs, outputs)
abc.get_config()

abc.to_json()

abc.get_config()
abc.to_json()



class ComputeSum(keras.layers.Layer):
    def __init__(self, input_dim):
        super().__init__()
        # super(ComputeSum, self).__init__()
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)

    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.total

input_dim = 2

x = tf.ones((2, 2))
inputs = x

model = ComputeSum(2)
y = model(x)
print(y.numpy())
y = model(x)
print(y.numpy())

model.get_config()


# gmm python
from scipy.stats import norm

x = np.linspace(-5, 5, 20)
x1 = x * np.random.rand(20)
x2 = x * np.random.rand(20) + 10
x3 = x * np.random.rand(20) - 10

xt = np.hstack((x1,x2,x3))

max_iterations = 10
pi = np.array([1/3, 1/3, 1/3])
mu = np.array([5,6,-3])
var = np.array([1,3,9])
r = np.zeros((len(xt), 3))


gauss1 = norm(loc=mu[0], scale=var[0])
gauss2 = norm(loc=mu[1], scale=var[1])
gauss3 = norm(loc=mu[2], scale=var[2])


# E-step, 최종적으로 r = gamma
for c, g, p in zip(range(3), [gauss1, gauss2, gauss3], pi):
    # E-step에서 감마값 구하는 과정의 분자
    r[:, c] = p*g.pdf(xt[:])

for i in range(len(r)):
    r[i, :] /= np.sum(r[i,:])

fig = plt.figure(figsize=(10,10))
ax0 = fig.add_subplot(111)

for i in range(len(r)):
    ax0.scatter(xt[i], 0, c=r[i,:], s=100)

# pdf에 np.linspace()넣는 건 Gaussian 표현 위해서 해주는 것
for g,c in zip([gauss1.pdf(np.linspace(-15,15)),
                gauss2.pdf(np.linspace(-15,15)),
                gauss3.pdf(np.linspace(-15,15))], ['r','g','b']):
    ax0.plot(np.linspace(-15, 15), g, c=c, zorder=0)
ax0.set_xlabel('X-axis')
ax0.set_ylabel('Gaussian pdf value')
ax0.legend(['Gaussian 1', 'Gaussian 2', 'Gaussian 3'])

plt.show()

# M-step
mc = np.sum(r, axis=0)
pi = mc / len(xt)
mu =  np.sum(r*np.vstack((xt, xt, xt)).T, axis=0)/mc
var = []

for c in range(len(pi)):
    var.append(np.sum(np.dot(r[:,c]*(xt[i] - mu[c]).T, r[:,c]*(xt[i] - mu[c])))/mc[c])



class GMM1D:

    def __init__(self, X, max_iterations):
        self.X = X
        self.max_iterations = max_iterations

    def run(self):
        self.pi = np.array([1 / 3, 1 / 3, 1 / 3])
        self.mu = np.array([5, 8, 1])
        self.var = np.array([5, 3, 1])

        gam = np.zeros((len(self.X), 3))

        for itr in range(self.max_iterations):
            gauss1 = norm(loc=self.mu[0], scale=self.var[0])
            gauss2 = norm(loc=self.mu[1], scale=self.var[1])
            gauss3 = norm(loc=self.mu[2], scale=self.var[2])

            # E-step
            for count, gauss, pi in zip(range(3), [gauss1, gauss2, gauss3], self.pi):
                gam[:, count] = pi * gauss.pdf(xt[:])

            for i in range(len(gam)):
                gam[i,:] /= np.sum(gam[i,:])

            fig = plt.figure(figsize=(10,10))
            ax0 = fig.add_subplot(111)

            for i in range(len(gam)):
                ax0.scatter(xt[i], 0, c=gam[i,:], s=100)

            for gauss, count in zip([gauss1.pdf(np.linspace(-15, 15)), gauss2.pdf(np.linspace(-15, 15)),
                             gauss3.pdf(np.linspace(-15, 15))], ['r', 'g', 'b']):
                ax0.plot(np.linspace(-15, 15), gauss, c=count, zorder=0)
            plt.show()

            # M-step
            mc = np.sum(gam, axis=0)
            self.pi = mc / len(self.X)
            self.mu = np.sum(gam*np.vstack((self.X, self.X, self.X)).T, axis=0)/mc
            self.var = []

            for c in range(len(self.pi)):
                self.var.append(np.sum(np.dot(gam[:,c]*(self.X[i] - self.mu[c]).T, gam[:,c]*(self.X[i] - self.mu[c])))/mc[c])

gmm = GMM1D(xt, 10)
gmm.run()






import numpy as np

class GMM2D:
    """Apply GMM to 2D data"""

    def __init__(self, num_clusters, max_iterations):

        """Initialize num_clusters(K) and max_iterations for the model"""

        self.num_clusters = num_clusters
        self.max_iterations = max_iterations

    def run(self, X):

        """Initialize parameters and run E and M step storing log-likelihood value after every iteration"""

        self.pi = np.ones(self.num_clusters) / self.num_clusters
        self.mu = np.random.randint(min(X[:, 0]), max(X[:, 0]), size=(self.num_clusters, len(X[0])))
        self.cov = np.zeros((self.num_clusters, len(X[0]), len(X[0])))


        for n in range(len(self.cov)):
            np.fill_diagonal(self.cov[n], 5)

        # reg_cov is used for numerical stability i.e. to check singularity issues in covariance matrix
        self.reg_cov = 1e-6 * np.identity(len(X[0]))

        #
        a = np.array([1,2,3])
        b = np.array([4,5,6])
        np.meshgrid(a, b)
        #

        x, y = np.meshgrid(np.sort(X[:, 0]), np.sort(X[:, 1]))
        self.XY = np.array([x.flatten(), y.flatten()]).T
        # Plot the data and the initial model

        fig0 = plt.figure(figsize=(10, 10))
        ax0 = fig0.add_subplot(111)
        ax0.scatter(X[:, 0], X[:, 1])
        ax0.set_title("Initial State")

        for m, c in zip(self.mu, self.cov):
            c += self.reg_cov
            multi_normal = multivariate_normal(mean=m, cov=c)
            ax0.contour(np.sort(X[:, 0]), np.sort(X[:, 1]), multi_normal.pdf(self.XY).reshape(len(X), len(X)),
                        colors='black', alpha=0.3)
            ax0.scatter(m[0], m[1], c='grey', zorder=10, s=100)

        fig0.savefig('GMM2D Initial State.png')
        plt.show()
        self.log_likelihoods = []

        for iters in range(self.max_iterations):
            # E-Step

            self.ric = np.zeros((len(X), len(self.mu)))

            for pic, muc, covc, r in zip(self.pi, self.mu, self.cov, range(len(self.ric[0]))):
                covc += self.reg_cov
                mn = multivariate_normal(mean=muc, cov=covc)
                self.ric[:, r] = pic * mn.pdf(X)

            for r in range(len(self.ric)):
                self.ric[r, :] = self.ric[r, :] / np.sum(self.ric[r, :])

            # M-step

            self.mc = np.sum(self.ric, axis=0)
            self.pi = self.mc / np.sum(self.mc)
            self.mu = np.dot(self.ric.T, X) / self.mc.reshape(self.num_clusters, 1)

            self.cov = []

            for r in range(len(self.pi)):
                covc = 1 / self.mc[r] * (np.dot((self.ric[:, r].reshape(len(X), 1) * (X - self.mu[r])).T,
                                                X - self.mu[r]) + self.reg_cov)
                self.cov.append(covc)

            self.cov = np.asarray(self.cov)

            likelihood_sum = np.sum(
                [self.pi[r] * multivariate_normal(self.mu[r], self.cov[r] + self.reg_cov).pdf(X) for r in
                 range(len(self.pi))])
            self.log_likelihoods.append(np.sum(np.log(likelihood_sum)))

            fig1 = plt.figure(figsize=(10, 10))
            ax1 = fig1.add_subplot(111)
            ax1.scatter(X[:, 0], X[:, 1])
            ax1.set_title("Iteration " + str(iters))

            for m, c in zip(self.mu, self.cov):
                c += self.reg_cov
                multi_normal = multivariate_normal(mean=m, cov=c)
                ax1.contour(np.sort(X[:, 0]), np.sort(X[:, 1]), multi_normal.pdf(self.XY).reshape(len(X), len(X)),
                            colors='black', alpha=0.3)
                ax1.scatter(m[0], m[1], c='grey', zorder=10, s=100)

            fig1.savefig("GMM2D Iter " + str(iters) + ".png")
            plt.show()

        fig2 = plt.figure(figsize=(10, 10))
        ax2 = fig2.add_subplot(111)
        ax2.plot(range(0, iters + 1, 1), self.log_likelihoods)
        ax2.set_title('Log Likelihood Values')
        fig2.savefig('GMM2D Log Likelihood.png')
        plt.show()

    def predict(self, Y):

        """Predicting cluster for new samples in array Y"""

        predictions = []

        for pic, m, c in zip(self.pi, self.mu, self.cov):
            prob = pic * multivariate_normal(mean=m, cov=c).pdf(Y)
            predictions.append([prob])

        predictions = np.asarray(predictions).reshape(len(Y), self.num_clusters)
        predictions = np.argmax(predictions, axis=1)

        fig2 = plt.figure(figsize=(10, 10))
        ax2 = fig2.add_subplot(111)
        ax2.scatter(X[:, 0], X[:, 1], c='c')
        ax2.scatter(Y[:, 0], Y[:, 1], marker='*', c='k', s=150, label='New Data')
        ax2.set_title("Predictions on New Data")

        colors = ['r', 'b', 'g']

        for m, c, col, i in zip(self.mu, self.cov, colors, range(len(colors))):
            #         c += reg_cov
            multi_normal = multivariate_normal(mean=m, cov=c)
            ax2.contour(np.sort(X[:, 0]), np.sort(X[:, 1]), multi_normal.pdf(self.XY).reshape(len(X), len(X)),
                        colors='black', alpha=0.3)
            ax2.scatter(m[0], m[1], marker='o', c=col, zorder=10, s=150, label='Centroid ' + str(i + 1))

        for i in range(len(Y)):
            ax2.scatter(Y[i, 0], Y[i, 1], marker='*', c=colors[predictions[i]], s=150)

        ax2.set_xlabel('X-axis')
        ax2.set_ylabel('Y-axis')
        ax2.legend()
        fig2.savefig('GMM2D Predictions.png')
        plt.show()

        return predictions



# lstm enc-dec test
# input = (samples, seq, x_dim)
x_dim = 1
seq_in = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                   ])
seq_in = seq_in.reshape((5,9, x_dim))

time_step = 9
lstm_h_dim = 30

np.shape(seq_in)

enc = Encoder(time_step=time_step, x_dim=x_dim, lstm_h_dim=lstm_h_dim )
hidden, cell = enc(seq_in)

dec = Decoder(time_step=time_step, x_dim=x_dim, h_dim=lstm_h_dim, hidden=hidden, cell = cell , batch_size=5)