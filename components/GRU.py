import tensorflow as tf
from Environment import d

dropout = 0.3
initializer = tf.contrib.layers.xavier_initializer()

class GRU():
	def __init__(self):
		# TODO: DROPOUT
		cell_1 = tf.nn.rnn_cell.GRUCell(d, dropout=dropout, recurrent_dropout=dropout, kernel_initializer=initializer, recurrent_initializer=initializer, bias_initializer=initializer)
		cell_2 = tf.nn.rnn_cell.GRUCell(d, dropout=dropout, recurrent_dropout=dropout, kernel_initializer=initializer, recurrent_initializer=initializer, bias_initializer=initializer)
		cell_3 = tf.nn.rnn_cell.GRUCell(d, dropout=dropout, recurrent_dropout=dropout, kernel_initializer=initializer, recurrent_initializer=initializer, bias_initializer=initializer)
		self.GRU = tf.nn.rnn_cell.MultiRNNCell([cell_1, cell_2, cell_3])

	def compute(self, r_t):
		# TODO: DOES THIS RETAIN PREVIOUS TIMESTEPS?
		H_t_1, _ = tf.nn.static_rnn(self.GRU, [r_t])
		return H_t_1
