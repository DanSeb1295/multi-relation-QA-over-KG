import tensorflow as tf
from tf import keras
from Environment import d

HIDDEN_DIM = d / 2
dropout = 0.3
initializer = tf.contrib.layers.xavier_initializer()

class BiGRU():
  def __init__(self):
    self.fw_1 = keras.layers.GRU(HIDDEN_DIM, dropout=dropout, recurrent_dropout=dropout, kernel_initializer=initializer, recurrent_initializer=initializer, bias_initializer=initializer)
    self.fw_2 = keras.layers.GRU(HIDDEN_DIM, dropout=dropout, recurrent_dropout=dropout, kernel_initializer=initializer, recurrent_initializer=initializer, bias_initializer=initializer)
    self.bw_1 = keras.layers.GRU(HIDDEN_DIM, go_backwards=True, dropout=dropout, recurrent_dropout=dropout, kernel_initializer=initializer, recurrent_initializer=initializer, bias_initializer=initializer)
    self.bw_2 = keras.layers.GRU(HIDDEN_DIM, go_backwards=True, dropout=dropout, recurrent_dropout=dropout, kernel_initializer=initializer, recurrent_initializer=initializer, bias_initializer=initializer)

  def compute(self, q):
    fw_out_1 = self.fw_1(q)
    bw_out_1 = self.bw_1(q)
    hidden_out = tf.concat([fw_out_1, bw_out_1], 0)
    
    fw_out_2 = self.fw_2(hidden_out)
    bw_out_2 = self.bw_2(hidden_out)
    output = tf.concat([fw_out_2, bw_out_2], 0)

    return output

