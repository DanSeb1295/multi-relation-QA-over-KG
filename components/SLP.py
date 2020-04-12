import tensorflow as tf
from Environment import d

class SLP():
	def __init__(self, T):
		self.weights_dict = {}
		self.biases_dict = {}

		for t in T:
			self.weights_dict[t] = tf.Variable(tf.truncated_normal([d, d], stddev=1.0/math.sqrt(float(d))), name='weights_{}'.format(t))
			self.biases_dict[t] = tf.Variable(tf.zeros([d]), name='biases_{}'.format(d))
	
	def compute(self, q, t):
		w_t = self.weights_dict[t]
		b_t = self.biases_dict[t]
		
		logits = tf.matmul(w_t, q) + b_t
		q_t = tf.math.tanh(logits)

		return q_t


