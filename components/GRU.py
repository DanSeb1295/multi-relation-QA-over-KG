import tensorflow as tf
from Environment import d

dropout = 0.3
KEEP_PROB = 1 - dropout

class GRU():
    def __init__(self):
        # cell_1 = tf.nn.rnn_cell.GRUCell(d, kernel_initializer='glorot_uniform', bias_initializer='zeros')
        # cell_2 = tf.nn.rnn_cell.GRUCell(d, kernel_initializer='glorot_uniform', bias_initializer='zeros')
        # cell_3 = tf.nn.rnn_cell.GRUCell(d, kernel_initializer='glorot_uniform', bias_initializer='zeros')
        
        # self.cell_1 = tf.nn.rnn_cell.DropoutWrapper(cell_1, output_keep_prob=KEEP_PROB, state_keep_prob=KEEP_PROB, input_keep_prob=KEEP_PROB)
        # self.cell_2 = tf.nn.rnn_cell.DropoutWrapper(cell_2, output_keep_prob=KEEP_PROB, state_keep_prob=KEEP_PROB, input_keep_prob=KEEP_PROB)
        # self.cell_3 = tf.nn.rnn_cell.DropoutWrapper(cell_3, output_keep_prob=KEEP_PROB, state_keep_prob=KEEP_PROB, input_keep_prob=KEEP_PROB)
        # self.states = (np.zeros(d), np.zeros(d), np.zeros(d))

        cell_1 = tf.keras.layers.GRU(d, dropout=dropout, recurrent_dropout=dropout) 
        cell_2 = tf.keras.layers.GRU(d, dropout=dropout, recurrent_dropout=dropout) 
        cell_3 = tf.keras.layers.GRU(d, dropout=dropout, recurrent_dropout=dropout) 
        self.GRU = tf.keras.layers.RNN([cell_1, cell_2, cell_3])

    def compute(self, r_t):
        # state_1, state_2, state_3 = self.states
        # output_1, state_1 = self.cell_1(r_t, state_1)
        # output_2, state_2 = self.cell_2(output_1, state_2)
        # H_t_1, state_3 = self.cell_3(output_2, state_3)
        # self.states = (state_1, state_2, state_3)

        r_t = tf.expand_dims(r_t, 0)
        H_t_1 = self.GRU(r_t)
        H_t_1 = tf.reshape([d])

        return H_t_1
