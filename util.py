import networkx as nx
import pandas as pd
from math import ceil
import numpy as np
import tensorflow as tf
from components import Entity_linker

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
    entity_linker = Entity_linker(path_KB, path_QA)

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
