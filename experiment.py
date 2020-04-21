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
path_KB = "./datasets/3H-kb.txt"
path_QA = "./datasets/PQ-3H.txt"

# Experiment Settings
T = 2                 # To change according to QA type
attention = True    # Use Attention Model or not
perceptron = True     # Use Perceptron for semantic similary scores

# Prep Data
KG, dataset = prep_dataset(path_KB, path_QA)
inputs = (KG, dataset, T)

# Initialise Policy Network
saved_model_name = 'model'
policy_network = PolicyNetwork(T, saved_model_name)

# Run Experiments
print('*********** Policy Network with Perceptron & Attention ***********')
train_att_per, val_att_per = policy_network.train(inputs, epochs=epochs)                # Model uses both attention & perceptro layers

print('*********** Policy Network with Perceptron Only ***********')
train_per, val_per = policy_network.train(inputs, epochs=epochs, attention=False)         # Model does not use attention layer

print('*********** Policy Network with Attention Only ***********')
train_att, val_att = policy_network.train(inputs, epochs=epochs, perceptron=False)        # Model does not use perceptron layer

# TODO: Plot Results
print(pd.DataFrame({'SRN': [np.mean(val_att_per[0])], 'w/o Attention': [np.mean(val_per[0])],
                    'w/o Perceptron': [np.mean(val_att[0])]}, index=["PQ-3H"]).T)

# plot epochs vs acc
xaxis = range(epochs)
ymin = 0
ymax = 1
fig = plt.figure()

plt.title("Ablation Study of PQ-2H")
plt.xlabel('Epochs')
plt.ylabel('Validation Set Hits@1')
plt.plot(xaxis, val_att_per[0], 'r', label='SRN')
plt.plot(xaxis, val_per[0], 'b', label='no Attention layer')
plt.plot(xaxis, val_att[0], 'g', label='no Perceptron layer')
axes = plt.gca()
axes.set_ylim([ymin, ymax])

plt.legend()
plt.show()
