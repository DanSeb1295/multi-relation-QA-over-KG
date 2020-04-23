from PolicyNetwork import PolicyNetwork
from Environment import Environment
from util import prep_dataset, fetch_model_name
from matplotlib import pyplot as plt
import pandas as pd

# set seeds for np and tf
import numpy as np
import tensorflow as tf
seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)

epochs = 8

# paths for the KG and QA files
path_KB = "./datasets/3H-kb.txt"
path_QA = "./datasets/PQ-3H.txt"

# Experiment Settings
T = 3                 # To change according to QA type
attention = True    # Use Attention Model or not
perceptron = True     # Use Perceptron for semantic similary scores

# Prep Data
KG, dataset = prep_dataset(path_KB, path_QA)
# dataset = dataset[:100]
inputs = (KG, dataset, T)

# Initialise Policy Network

# Run Experiments
print('\n\n*********** Policy Network with Perceptron & Attention ***********')
model_name = fetch_model_name('combined')
policy_network = PolicyNetwork(T, model_name)
# Model uses both attention & perceptro layers
train_att_per, val_att_per = policy_network.train(inputs, epochs=epochs)

# print('\n\n*********** Policy Network with Perceptron Only ***********')
# model_name = fetch_model_name('perceptron')
# policy_network = PolicyNetwork(T, model_name)
# train_per, val_per = policy_network.train(inputs, epochs=epochs, attention=False)         # Model does not use attention layer

# print('\n\n*********** Policy Network with Attention Only ***********')
# model_name = fetch_model_name('attention')
# policy_network = PolicyNetwork(T, model_name)
# train_att, val_att = policy_network.train(inputs, epochs=epochs, perceptron=False)        # Model does not use perceptron layer

# print(pd.DataFrame({'SRN': [np.mean(val_att_per[0])], 'w/o Attention': [np.mean(val_per[0])],
#                     'w/o Perceptron': [np.mean(val_att[0])]}, index=["PQ-3H"]).T)

# Plot Results
xaxis = range(1, epochs + 1)
ymin = 0
ymax = 1
fig = plt.figure()

plt.title("Ablation Study of PQ-3H")
plt.xlabel('Epochs')
plt.ylabel('Validation Set Hits@1')
plt.plot(xaxis, val_att_per[0], 'r', label='SRN')
plt.plot(xaxis, val_per[0], 'b', label='no Attention layer')
plt.plot(xaxis, val_att[0], 'g', label='no Perceptron layer')
axes = plt.gca()
axes.set_ylim([ymin, ymax])

plt.legend()
plt.show()
