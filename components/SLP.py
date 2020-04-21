import tensorflow as tf
from Environment import d

initializer = tf.keras.initializers.GlorotNormal()

class SLP():
    def __init__(self, T):
        self.weights_dict = {}
        self.biases_dict = {}

        for t in range(1, T+1):
            self.weights_dict[t] = tf.Variable(initializer([d, d]), trainable=True)
            self.biases_dict[t] = tf.Variable(initializer([d]), trainable=True)
    
    def compute(self, q, t):
        w_t = self.weights_dict[t]
        b_t = self.biases_dict[t]

        logits = tf.matmul(q, w_t)
        logits = logits + b_t
        q_t = tf.math.tanh(logits)

        return q_t


