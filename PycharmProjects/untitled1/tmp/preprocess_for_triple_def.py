import re
import pandas as pd
import bs4
import requests
import spacy
from spacy import displacy
import os
import numpy as np

nlp = spacy.load("en_core_web_sm")
from spacy.matcher import Matcher
from spacy.tokens import Span
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_entities(sent):
    ## chunk 1
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""  # dependency tag of previous token in the sentence
    prv_tok_text = ""  # previous token in the sentence

    prefix = ""
    modifier = ""

    #############################################################

    for tok in nlp(sent):
        ## chunk 2
        # if token is a punctuation mark then move on to the next token
        if tok.dep_ != "punct":
            # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text

            # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            ## chunk 3
            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

                ## chunk 4
            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text

            ## chunk 5
            # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
    #############################################################

    return [ent1.strip(), ent2.strip()]


def get_relation(sent):
    doc = nlp(sent)

    # Matcher class object
    matcher = Matcher(nlp.vocab)

    # define the pattern
    pattern = [{'DEP': 'ROOT'},
               {'DEP': 'prep', 'OP': "?"},
               {'DEP': 'agent', 'OP': "?"},
               {'POS': 'ADJ', 'OP': "?"}]

    matcher.add("matching_1", None, pattern)

    matches = matcher(doc)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]]

    return (span.text)


# HRT to pykg2vec
def Dt2Pykg(head, relations, tail, stop=False):
    # Get rid of blank
    headNoB = [i.replace(" ", "") for i in head]
    tailNoB = [j.replace(" ", "") for j in tail]
    relationsNoB = [k.replace(" ", "") for k in relations]


    # check No Entities or Relations indices
    del_idx = []
    for i in range(len(head)):
        if len(head[i]) == 0:
            del_idx.append(i)
        elif len(tail[i]) == 0:
            del_idx.append(i)
        elif len(relations) == 0:
            del_idx.append(i)

    if stop == True:
        # Delete stopwords
        stopwords = ['I', 'you', 'You', 'we', 'We', 'He', 'It', 'it', 'That', 'that', 'she', 'She', 'They', 'they', 'them',
                     'Them', 'a', 'A', 'who', 'Who', 'how', 'How', 'This', 'this', 'When', 'when', 'us', 'Us']
        stopidx = []
        for i in range(len(head)):
            if head[i] in stopwords:
                stopidx.append(i)
        # merge to idx list
        del_idx.extend(stopidx)

    del_clear = list(set(del_idx))

    # Make the triplet
    triplet_1 = [""] * len(headNoB)
    triplet_2 = [""] * len(headNoB)
    for i in range(len(headNoB)):
        triplet_1[i] = headNoB[i] + str("\t") + relationsNoB[i]
        triplet_2[i] = triplet_1[i] + str("\t") + tailNoB[i]

    # stopwords and omissions
    triplet_np = np.array(triplet_2)
    triplet_done = np.delete(triplet_np, del_clear, axis=0).tolist()

    return triplet_done




pd.set_option('display.max_colwidth', 200)
os.chdir("C:\\data_minsung")
# sent = pd.read_csv("./trav_papers.csv").values[:, 1]

# 동필's data
merge = []
f = open("./mergefile.txt", "r")
for i in f:
    merge.append(i)
f.close()

sent = merge

# extract entities(Use modules)
entity_pairs = []
for i in tqdm(sent):
    entity_pairs.append(get_entities(i))

relations = [get_relation(i) for i in sent]
# No empty space, No omission
head = np.array(entity_pairs)[:, 0].tolist()
tail = np.array(entity_pairs)[:, 1].tolist()

# delete stopwords & blank
triplet_done = Dt2Pykg(head, relations, tail, stop=True)

# delete blank only
# triplet_done = Dt2Pykg(head, relations, tail, stop=False)
# f = open("./trav_Npre.txt", "w")
# for i in range(len(triplet_done)):
#     data = triplet_done[i] + "\n"
#     f.write(data)
# f.close()


############################
train_size = int(len(triplet_done) * 0.8)
val_size = int((len(triplet_done) - train_size) * 0.3)

f = open("./merged-train.txt", "w", encoding='utf-8')
for i in range(train_size):
    data = triplet_done[i] + "\n"
    f.write(data)
f.close()

f = open("./merged-valid.txt", "w",encoding='utf-8')
for i in triplet_done[train_size:train_size + val_size]:
    data = i + "\n"
    f.write(data)
f.close()

f = open("./merged-test.txt", "w",encoding='utf-8')
for i in triplet_done[train_size + val_size:]:
    data = i + "\n"
    f.write(data)
f.close()

###################################################.
### mixed : travel, business

os.getcwd()

travel_paper = []
f = open("./customdataset_Npre/trav_Npre.txt", "r")
lines = f.readlines()
for line in lines:
    travel_paper.append(line)

bus_paper = []
f = open("./customdataset_Npre/bus_Npre.txt", "r")
lines = f.readlines()
for line in lines:
    bus_paper.append(line)

travel_half = travel_paper[:len(bus_paper)]
travel_half.extend(bus_paper)
total_corp = travel_half

train_size = int(len(total_corp) * 0.7)
val_size = int((len(total_corp) - train_size) * 0.3)

f = open("./total_Npre-train.txt", "w")
for i in range(train_size):
    data = total_corp[i]
    f.write(data)
f.close()

f = open("./total_Npre-valid.txt", "w")
for i in total_corp[train_size:train_size + val_size]:
    data = i
    f.write(data)
f.close()

f = open("./total_Npre-test.txt", "w")
for i in total_corp[train_size + val_size:]:
    data = i
    f.write(data)
f.close()
