import pickle
from numpy.linalg import norm
import numpy as np
import csv





def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict



def save_pickle(path, dict):
    with open(path, 'bw') as file:
        pickle.dump(dict, file)  



def compute_sim(emb1, emb2):
    sim = np.dot(emb1[0], emb2[0]) / (norm(emb1[0]) * norm(emb2[0]))
    return sim



def read_csv(PATH_CSV):
    with open(PATH_CSV, mode='r', encoding='UTF-8-SIG') as csv_file:
        csv_reader = csv.reader(csv_file)
        return list(csv_reader)