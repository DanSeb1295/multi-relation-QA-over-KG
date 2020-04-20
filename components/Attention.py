import tensorflow as tf
from Environment import d

initializer = tf.contrib.layers.xavier_initializer()

class Attention():
  def __init__(self):
    self.W = tf.Variable(initializer([d]))
    self.B = tf.Variable(initializer(0))

  def compute(self, r_star, q_t):
    if r_star.shape == (d, 1):
      r_star = tf.transpose(r_star)

    r_star = tf.reshape(r_star, [d])

    beta_stars = []
    for w_t_m in q_t:
      output = tf.multiply(r_star, w_t_m)
      b_star = tf.matmul(self.W, output) + self.b
      beta_stars.append(b_star)
    alpha_stars tf.nn.softmax(beta_stars)
    
    q_t_stars = []
    for i, a_star in enumerate(alpha_stars):
      q = a_star * q_t[i]
      q_t_stars.append(q)
    q_t_star = tf.reduce_sum(q_t_stars, 0)

    return q_t_star

