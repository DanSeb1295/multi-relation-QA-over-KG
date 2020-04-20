from PolicyNetwork import PolicyNetwork
from Environment import Environment
from util import prep_dataset
from matplotlib import pyplot as plt
import pandas as pd

# set seeds for np and tf
import numpy as np
import tensorflow as tf
seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)

epochs = 10

# paths for the KG and QA files
path_KB = r"datasets\3H-kb.txt"
path_QA = r"datasets\PQ-3H.txt"

# Experiment Settings
T = 2                 # To change according to QA type
attention = True    # Use Attention Model or not
perceptron = True     # Use Perceptron for semantic similary scores

# Prep Data
KG, dataset = prep_dataset(path_KB, path_QA)
inputs = (KG, dataset, T)

# Initialise Policy Network
saved_model_path = 'model.meta'
policy_network = PolicyNetwork(T, saved_model_path)

# Run Experiments
train_acc_att_per, val_acc_att_per = policy_network.train(inputs, epochs=epochs)                # Model uses both attention & perceptro layers
train_acc_per, val_acc_per = policy_network.train(inputs, epochs=epochs, attention=False)         # Model does not use attention layer
train_acc_att, val_acc_att = policy_network.train(inputs, epochs=epochs, perceptron=False)        # Model does not use perceptron layer

# TODO: Plot Results
print(pd.DataFrame({'SRN': [np.mean(val_acc_att_per)], 'w/o Attention': [np.mean(val_acc_per)],
                    'w/o Perceptron': [np.mean(val_acc_att)]}, index=["PQ-2H"]).T)

# # plot epochs vs acc
# xaxis = range(len(epochs))
# ymin = 0
# ymax = 1
# fig = plt.figure()

# plt.title("Ablation Study of PQ-2H")
# plt.xlabel('Epochs')
# plt.ylabel('Validation Set Hits@1')
# plt.plot(xaxis, val_acc_att_per, 'r', label='SRN')
# plt.plot(xaxis, val_acc_per, 'b', label='no Attention layer')
# plt.plot(xaxis, val_acc_att, 'g', label='no Perceptron layer')
# axes = plt.gca()
# axes.set_ylim([ymin, ymax])

# plt.legend()
# plt.show()
