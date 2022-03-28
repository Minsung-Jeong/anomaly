import numpy as np
import tensorflow as tf

class GMM:

    def __init__(self, n_comp):
        self.n_comp = n_comp
        self.phi = self.mu = self.sigma = None
        self.training = False

    def create_variables(self, n_features):
        phi = tf.Variable(tf.zeros(shape=[self.n_comp]),
                          dtype=tf.float32, name="phi")
        mu = tf.Variable(tf.zeros(shape=[self.n_comp, n_features]),
                         dtype=tf.float32, name="mu")
        sigma = tf.Variable(tf.zeros(
            shape=[self.n_comp, n_features, n_features]),
            dtype=tf.float32, name="sigma")
        L = tf.Variable(tf.zeros(
            shape=[self.n_comp, n_features, n_features]),
            dtype=tf.float32, name="L")

        return phi, mu, sigma, L

    def fit(self, z, gamma):
        """ fit data to GMM model
        Parameters
        ----------
        z : tf.Tensor, shape (n_samples, n_features)
            data fitted to GMM.
        gamma : tf.Tensor, shape (n_samples, n_comp)
            probability. each row is correspond to row of z.
        """
        gamma_sum = tf.reduce_sum(gamma, axis=0)
        self.phi = phi = tf.reduce_mean(gamma, axis=0)
        self.mu = mu = tf.einsum('ik,il->kl', gamma, z) / gamma_sum[:,None]

n_features = 2
n_samples = 10
x = np.random.randint(10, size=(n_samples, n_features))

a = np.ones([n_samples])
ab = np.ones([n_samples])*2
b = np.ones([n_samples])*3
gamma = np.array([a, ab, b]).T
gamma.shape
gamma_sum = np.sum(gamma, axis=0)
sibling = np.einsum('ik,il->kl', gamma, x)
sig = np.einsum('ik,il->kl', gamma, x) / gamma_sum[:,None]
sibling[1] / 20