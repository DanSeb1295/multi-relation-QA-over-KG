import tensorflow as tf
from Environment import d

initializer = tf.contrib.layers.xavier_initializer()

class SLP():
  def __init__(self, T):
    self.weights_dict = {}
    self.biases_dict = {}

    for t in range(T):
      self.weights_dict[t] = tf.Variable(initializer([d, d]))
      self.biases_dict[t] = tf.Variable(initializer([d]))
  
  def compute(self, q, t):
    w_t = self.weights_dict[t]
    b_t = self.biases_dict[t]
    
    logits = tf.matmul(w_t, q) + b_t
    q_t = tf.math.tanh(logits)

    return q_t


