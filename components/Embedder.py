import numpy as np
import pandas as pd
import csv
import networkx as nx

class Embedder:
    def __init__(self):
        freebase_path = "./datasets/Freebase/"
        glove_data_file = "{}glove.6B/glove.6B.50d.txt".format(freebase_path)
        mid_to_name_file = '{}mid2name.tsv'.format(freebase_path)
        entity2id_file = '{}knowledge_graphs/entity2id.txt'.format(freebase_path)
        relation2id_file = '{}knowledge_graphs/relation2id.txt'.format(freebase_path)
        embedding_entity_file = '{}embeddings/dimension_50/transe/entity2vec.bin'.format(freebase_path)
        embedding_relation_file = '{}embeddings/dimension_50/transe/relation2vec.bin'.format(freebase_path)

        print('Importing Glove Word Embeddings')
        words = pd.read_csv(glove_data_file, sep=" ", index_col=0, header=None, na_values=None, quoting=csv.QUOTE_NONE) 
        self.word_embeddings = words
        
        print('Getting relation2id')
        self.relation2id = self.read_tsv(relation2id_file)
        
        print('Getting relation Embeddings')
        self.relation_embedding = np.memmap(embedding_relation_file , dtype='float32', mode='r')

    #one function to handle all the file reading
    def read_tsv(self, path_name):
        tsv_file = open(path_name)
        read_tsv = csv.reader(tsv_file, delimiter='\t')
        output_dict = {}
        if 'mid2name' in path_name:
            for row in read_tsv:
                temp_list = list(row[0])
                temp_list[0] = ''
                temp_list[2] = '.'
                modified_row0 = ''.join(temp_list)
                modified_row1 = ''.join(row[1].lower().split())
                output_dict[modified_row1] = modified_row0
        elif 'relation2id' in path_name:
            for row in read_tsv:
                item_list = row[0].split('.')
                if 'people' in item_list:
                    output_dict[item_list[-1]] = row[1]
        else:
            for row in read_tsv:
                if len(row) > 1:
                    output_dict[row[0]] = row[1]
        return output_dict
        
    #finding the word embeddings from table
    def vec(self, w):
        return self.word_embeddings.loc[w].values

    #Try catch for finding the word embeddings from GLOVE
    def embed_word(self, word):
        # embedder.embed_word(word), where word is a string, would return a 50d vector (np.array)
        try:
            original_embedding = self.vec(word)
            return original_embedding
        except Exception as e:
            print('Get a larger pre-trained GLOVE model', e)

    #Try catch for finding the entity embeddings from Freebase
    def embed_entity(self, name):
        # embedder.embed_entity(entity_name), where entity_name is a string, would return a 50d vector (np.memmap, like np.array)
        modified_name = ''
        if '_' in name:
            modified_name = ''.join(name.lower().split('_'))
        else:
            modified_name = name
        print('modified_name', modified_name)
        try: 
            mid = self.mid_to_name[modified_name]
            print(mid)
            try:
                index = int(self.entity2id[mid])
                print(index)
                vector_index = index * 50
                return self.entity_embedding[vector_index:vector_index+50]
            except Exception as e:
                print('Not found in entity2id', e)
        except Exception as e:
            print('Not found in mid_to_name', e)
        

    # Try catch for finding the relation embeddings from Freebase
    def embed_relation(self, relation): 
        # embedder.embed_relation(relation_name), where relation_name is a string, would return a 50d vector (np.memmap, like np.array)
        try:
            index = int(self.relation2id[relation])
            vector_index = index * 50
            return self.relation_embedding[vector_index:vector_index+50]
        except Exception as e:
            return

