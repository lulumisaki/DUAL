from bert_serving.client import BertClient
import numpy as np

def get_bert_embedding(file_name_01,file_name_02):
    attr_01 = []
    attr_02 = []
    # file_name_01 = "data/en_de_15k_V1/ent_attr_1"
    # file_name_02 = "data/en_de_15k_V1/ent_attr_2"
    with open(file_name_01, 'r', encoding='utf-8') as file:
        for line in file:
            attr = line.split()[1]
            attr_01.append(attr)
    with open(file_name_02, 'r', encoding='utf-8') as file:
        for line in file:
            attr = line.split()[1]
            attr_02.append(attr)
    bc = BertClient()
    attr_embedding_01 = bc.encode(attr_01)
    attr_embedding_02 = bc.encode(attr_02)
    # attr_embedding = np.concatenate((attr_embedding_01, attr_embedding_02), axis=0)
    attr_embedding_01 = attr_embedding_01[:10500, :]
    attr_embedding_02 = attr_embedding_02[:10500, :]
    sim_matrix = np.dot(attr_embedding_01, attr_embedding_02.T) / (np.linalg.norm(attr_embedding_01, axis=1, keepdims=True) * np.linalg.norm(attr_embedding_02, axis=1, keepdims=True).T)
    return sim_matrix

