from pykg2vec.utils.visualization import *
import os
import csv
import pandas as pd
import numpy as np
import sklearn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


os.getcwd()

os.chdir("C:\\data_minsung\customdataset\embeddings\TransE")


def read_tsv(path, entity=None, relation=None):

    file = pd.read_csv(path, delimiter='\t', header=None)
    file = file.values.tolist()
    return file

entity_embed = read_tsv(".\ent_embedding.tsv")
entity_label = read_tsv(".\ent_labels.tsv")

E_labels = []
for i in range(len(entity_label)):
    E_labels.append(entity_label[i][0])

entity_tsne = np.load(".\TransE_TSNE.npy")
relation_label = read_tsv(".\_rel_labels.tsv")

import numpy as np
np.shape(entity_tsne)
type(entity_embed)
draw_embedding(np.matrix(entity_embed), E_labels, resultpath=".\embeddings", algos='TransE', show_label=True)
