import os
import csv
import pandas as pd
import numpy as np
import sklearn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

os.chdir("C:\\data_minsung\customdataset_merged\embeddings\TransH")


def read_tsv(path, entity=None, relation=None):
    file = pd.read_csv(path, delimiter='\t', header=None)
    file = file.values.tolist()
    return file


############ entities embeddings & labels
entity_embed = read_tsv(".\ent_embedding.tsv")
entity_label = read_tsv(".\ent_labels.tsv")

# relation_embed = read_tsv(".\_rel_embedding.tsv")
# relation_label = read_tsv(".\_rel_labels.tsv")

entity_tsne = TSNE(n_components=2).fit_transform(entity_embed)
np.save(".\TransE_TSNE.npy",entity_tsne)
# relation_tsne = TSNE(n_components=2).fit_transform(relation_embed)
# np.save(".\TransE_label_TSNH.npy", relation_tsne)

entity_tsne = np.load(".\TransE_TSNE.npy")

############ k-means clustering
from sklearn.cluster import KMeans

# 입력 넣으면 plot까지
def Dt_ready(entity_tsne, entity_label, NumC,NumEnt):

    E_labels = [x[0] for x in entity_label]
    E_x = entity_tsne[:,0]
    E_y = entity_tsne[:,1]

    concat_dt = np.column_stack([np.array(E_x), np.array(E_y), np.array(E_labels)])
    concat_dt = np.random.permutation(concat_dt)

    inputAll = concat_dt[:NumEnt]

    x = inputAll[:,0].astype(float)
    y = inputAll[:,1].astype(float)
    xyaxis = np.column_stack([x,y])

    kmeans = KMeans(n_clusters=NumC, random_state=0).fit(xyaxis)
    kmeans_labels = kmeans.labels_


    # plotting embedding
    fig, ax = plt.subplots(figsize=(50,50))
    ax.scatter(x, y, s=30, c=kmeans_labels.astype(np.float))
    for i, txt in enumerate(inputAll[:,2]):
        ax.annotate(txt, (x[i],y[i]))

Dt_ready(entity_tsne, entity_label, NumC = 3,NumEnt=150)