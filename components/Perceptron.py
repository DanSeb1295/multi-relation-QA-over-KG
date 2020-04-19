import tensorflow as tf
from Environment import d

class Perceptron():
	def __init__(self):
		self.W_L1 = tf.Variable(tf.truncated_normal([d, 2*d], stddev=1.0/math.sqrt(float(d))), name='weights_L1')
		self.W_L2 = tf.Variable(tf.truncated_normal([d], stddev=1.0/math.sqrt(float(d))), name='weights_L2')

	def compute(self, r_star, H_t, q_t_star):
		if r_star.shape == (d, 1):
			r_star = tf.transpose(r_star)

		r_star = tf.reshape(r_star, [d])

		input = tf.concat(H_t, q_t_star, axis = 0)
		
		logits_1 = tf.matmul(self.W_L1, input)
		out_1 = tf.nn.relu(logits_1)
		out_2 = tf.matmul(self.W_L2, out_1)
		semantic_score = tf.matmul(r_star, logits_2)

		return semantic_score

