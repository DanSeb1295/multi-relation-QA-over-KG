import tensorflow as tf
from tensorflow import keras
from Environment import d

HIDDEN_DIM = int(d / 2)
dropout = 0.3

class BiGRU():
    def __init__(self):
        self.fw_1 = keras.layers.GRU(HIDDEN_DIM, dropout=dropout, recurrent_dropout=dropout, return_sequences=True)
        self.fw_2 = keras.layers.GRU(HIDDEN_DIM, dropout=dropout, recurrent_dropout=dropout, return_sequences=True)
        self.bw_1 = keras.layers.GRU(HIDDEN_DIM, go_backwards=True, dropout=dropout, recurrent_dropout=dropout, return_sequences=True)
        self.bw_2 = keras.layers.GRU(HIDDEN_DIM, go_backwards=True, dropout=dropout, recurrent_dropout=dropout, return_sequences=True)

    def compute(self, q):
        n = q.shape[1]

        fw_out_1 = self.fw_1(q)
        bw_out_1 = self.bw_1(q)
        hidden_out = tf.concat([fw_out_1, bw_out_1], 1)
        hidden_out = tf.reshape(hidden_out, (1, n, d))
        
        fw_out_2 = self.fw_2(hidden_out)
        bw_out_2 = self.bw_2(hidden_out)
        output = tf.concat([fw_out_2, bw_out_2], 1)
        output = tf.reshape(output, (n, d))

        return output

