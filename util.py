import csv
import configparser
import networkx as nx
import pandas as pd
import numpy as np
from math import ceil
import tensorflow as tf
from components import EntityLinker
from matplotlib import pyplot as plt


model_names_path = './saved_models/model_names.ini'
results_path = './saved_models/'

seed = 2020
train_split = 0.8


def prep_dataset(path_KB, path_QA):
    '''
    Input:
            path_KB.txt, path_QA.txt
    Return:
            KG as network x graph object
            list of ([q_word1, q_word2,...,], e_s, ans)             # e_s should be replaced inside the questions also
    '''
    # get KG
    df_graph = pd.read_csv(path_KB, sep=r'\s', header=None, names=['e_subject', 'relation', 'e_object'])
    KB = nx.from_pandas_edgelist(
        df_graph, "e_subject", "e_object", edge_attr="relation", create_using=nx.DiGraph())

    # get questions
    df_qn = pd.read_csv(path_QA, sep=r'\t', header=None, names=['question_sentence', 'answer_set', 'answer_path'])
    df_qn['answer'] = df_qn['answer_set'].apply(lambda x: x.split('(')[0])

    # Initialize Entity Linker
    entity_linker = EntityLinker(path_KB, path_QA)

    # get parsed qn and the topic entity
    df_qn['q'], df_qn['e_s'] = zip(*df_qn['question_sentence'].apply(lambda x: entity_linker.find_entity(x)))
    qn_list = df_qn[['q', 'e_s', 'answer']].values.tolist()

    # convert to list of tuples
    final_qn_list = [tuple(x) for x in qn_list]
    return KB, final_qn_list


def train_test_split(dataset, seed=seed, train_split=train_split):
    '''
    Input:
            List of (q, e_s, ans), train_split, seed
    Return:
            train_set: list of (q, e_s, ans)
            test_set: list (q, e_s, ans)
    '''
    n_samples = len(dataset)
    n_train = ceil(n_samples * train_split)
    n_test = n_samples - n_train

    if n_train + n_test > n_samples:
        raise ValueError('The sum of train_size and test_size = %d, '
                         'should be smaller than the number of '
                         'samples %d.' % (n_train + n_test, n_samples))

    np.random.seed(seed)
    shuffled = np.random.permutation(dataset)
    train_set = shuffled[:n_train]
    test_set = shuffled[n_train:]

    if len(train_set) + len(test_set) > n_samples:
        raise ValueError('The sum of train_set and test_set = %d, '
                         'should be smaller than the number of '
                         'samples %d.' % (len(train_set) + len(test_set), n_samples))

    return train_set, test_set


def save_checkpoint(policy_network, save_path, step, write_meta_graph=False):
    '''
    Input: PolicyNetwork
    Return: None
    Output: Appropriate save file of learned parameters weights and values for all labelled #Trainable in PolicyNetwork
                    (Label file_extension according to date_time e.g. T_<T>_model_HHMM_DDMM, savedir = ./models)

    # TODO: Implement
    '''

    saver = tf.train.Saver()
    saver.save(policy_network.sess, save_path, max_to_keep=5, keep_checkpoint_every_n_hours=1,
                 global_step=step, write_meta_graph=write_meta_graph)

def write_model_name(model_name, model_type='combined'):
    config = configparser.ConfigParser()
    config.read(model_names_path)
    config['Models'][model_type] = model_name
    with open(model_names_path, 'w') as configfile:
        config.write(configfile)

def fetch_model_name(model_type='combined'):
    config = configparser.ConfigParser()
    config.read(model_names_path)
    name = config['Models'][model_type]
    if not name: return 'model'
    return name

def plot_results(file_path=results_path):
    epochs, results_dic = getAllResults(file_path, model_types=['combined','attention'])
    xaxis = range(1, epochs + 2)
    ymin = 0
    ymax = 1

    fig1 = plt.figure(1)

    plt.subplot(2, 1, 1)
    plt.gca().set_title('Training')
    plt.ylabel('Loss')
    plt.plot(xaxis, results_dic['combined']['train_loss'], 'ro-', label='SRN')
    # plt.plot(xaxis, results_dic['attention']['train_loss'], 'go-', label='no Perceptron layer')

    ax = plt.subplot(2, 1, 2)
    plt.gca().set_ylim([ymin, ymax])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(xaxis, results_dic['combined']['train_acc'], 'ro-', label='SRN')
    # plt.plot(xaxis, results_dic['attention']['train_acc'], 'go-', label='no Perceptron layer')

    handles, labels = ax.get_legend_handles_labels()
    fig1.legend(handles, labels, loc='upper right')
    fig1.suptitle("Ablation Study of PQ-3H")
    fig1.tight_layout()
    fig1.subplots_adjust(top=0.85)

    fig2 = plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.gca().set_title('Validation')
    plt.ylabel('Loss')
    plt.plot(xaxis, results_dic['combined']['val_loss'], 'ro-', label='SRN')
    # plt.plot(xaxis, results_dic['attention']['val_loss'], 'go-', label='no Perceptron layer')

    ax = plt.subplot(2, 1, 2)
    plt.gca().set_ylim([ymin, ymax])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(xaxis, results_dic['combined']['val_acc'], 'ro-', label='SRN')
    # plt.plot(xaxis, results_dic['attention']['val_acc'], 'go-', label='no Perceptron layer')


    handles, labels = ax.get_legend_handles_labels()
    fig2.legend(handles, labels, loc='upper right')
    fig2.suptitle("Ablation Study of PQ-3H")
    fig2.tight_layout()
    fig2.subplots_adjust(top=0.85)
    plt.show()


def getAllResults(file_path, model_types):
    all_results_dic = {}
    epochs = 0
    for model_type in model_types:
        file_name = model_type + '_results.csv'
        try:
            with open(file_path + file_name) as data_file:
                reader = csv.reader(data_file, delimiter=',')
                final_row = ''
                for row in reader:
                    final_row = row
        except FileNotFoundError:
            print('Can\'t find {} results file!'.format(model_type))
            continue
        all_results_dic[model_type] = parseResults(final_row[1:])
        epochs = max(epochs, int(final_row[0].strip()[-1]))
    return epochs, all_results_dic



def parseResults(results):
    results_dic = {}
    result_names = iter(['train_acc','train_loss','val_acc','val_loss'])
    name = next(result_names)
    values = []
    for result in results:
        try:
            result = result.strip()
            if result[-1] == ']':
                values.append(float(result[:-1]))
                results_dic[name] = values
                name = next(result_names)
                values = []
            elif result[0] == '[':
                values.append(float(result[1:]))
            else:
                values.append(float(result))
        except StopIteration:
            break
    return results_dic
