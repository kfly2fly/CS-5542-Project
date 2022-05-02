import glob
import os
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import math
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TN, FP, FN, TP)

# convert a onehot method name with mulitple labels to name
def onehot_to_name(onehot, dict):
    r, l = (onehot == 1).nonzero(as_tuple=True)
    counts = torch.bincount(r)
    cindex = torch.cumsum(counts, dim=-1)
    s = ""
    for i, item in enumerate(l):
        if i in cindex:
            s += ","
        s += dict[str(item.item() + 1)] + " "
    return s, r, l


def onehot_to_name_list(onehot, dict):
    ps = PorterStemmer()
    name_list = []
    for t in onehot:
        index = t.nonzero()
        s = ''
        for i in index:
            s += ps.stem(dict[str(i.item() + 1)]) + " "

        name_list.append(s[:-1])

    return name_list


def onehot_to_name_list_1(onehot, dict):
    ps = PorterStemmer()
    r, l = (onehot == 1).nonzero(as_tuple=True)
    counts = torch.bincount(r)
    cindex = torch.cumsum(counts, dim=-1)
    s = ""
    name_list = []
    for i, item in enumerate(l):
        s += ps.stem(dict[str(item.item() + 1)]) + " "
        if i in cindex:
            if len(s) > 0:
                name_list.append(s[:-1])
            else:
                name_list.append(s)
            s = ""
    if len(s) > 0:
        name_list.append(s[:-1])
    else:
        name_list.append(s)
    return name_list


def stem_onehot(onehot, dict):
    ps = PorterStemmer()
    new_dict = {y: x for x, y in dict.items()}
    name_list = onehot_to_name_list(onehot, dict)
    for i in range(0, len(name_list)):
        for name in name_list[i].split(" "):
            stem_name = ps.stem(name)
            if stem_name in new_dict:
                if name != stem_name and int(onehot[i, int(new_dict[stem_name]) - 1]) == 0:
                    onehot[i, int(new_dict[name]) - 1] = 0
                    onehot[i, int(new_dict[stem_name]) - 1] = 1
    return onehot


def x_to_name(x, dict):
    node_indx = torch.argmax(x, dim=1)
    s = ""
    for key in node_indx:
        key = str(key.item() + 1)
        s += dict[key] + " "
    return s;


def y_to_name(y, dict):
    node_indx = torch.nonzero(y, as_tuple=True)[0]
    s = ""
    for key in node_indx:
        key = str(key.item() + 1)
        s += dict[key] + " "
    return s;


def onehot_Prediction(out, truth):
    # init results as 0
    r = torch.zeros(truth.shape)
    topPicks = []
    # number of labels to pick based on truth
    n2p = torch.count_nonzero(truth, dim=1)
    for i in np.arange(out.shape[0]):
        _, indx = torch.topk(out[i], n2p[i])
        topPicks.append(indx)
    for i in np.arange(len(topPicks)):
        # print(r[i])
        r[i][topPicks[i]] = 1
    return r


def onehot_Prediction_compressed(out, truth):
    # init results as 0
    r = torch.zeros(truth.shape)
    topPicks = []
    # number of labels to pick based on truth
    n2p = torch.count_nonzero(truth, dim=1)
    for i in np.arange(out.shape[0]):
        _, indx = torch.topk(out[i], n2p[i])
        topPicks.append(indx)
    for i in np.arange(len(topPicks)):
        # print(r[i])
        r[i][topPicks[i]] = 1
    return r


def showPreTruth(pre, label, dict):
    n, _, _ = onehot_to_name(pre, dict)
    print("     ===========pre and target ================")
    print("     =========", n)

    n, _, _ = onehot_to_name(label, dict)
    print("     =========", n)


def void_nan(x):
    return 0 if math.isnan(float(x)) else float(x)


def get_nf_dict(path):
    dict = {}
    with open(path, 'r') as f:
        srcs = f.read().split('\n')[:-1]
    for src in srcs:
        dict[src.split(' ')[0]] = src.split(' ')[1]
    return dict

def node_to_vec(file_path, dict_path):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    dict = get_nf_dict(dict_path)
    files = [f for f in glob.glob(os.path.join(
        file_path, '*/*.nvcb'), recursive=True)]
    for file in files:
        with open (file, 'r') as f:
            srcs = f.readlines()
        vec_list = []
        for src in srcs:
            if src.strip() in dict:
                vec_list.append(dict[str(src.strip())])
            else:
                vec = ''
                for s in model.encode(src.strip()):
                    vec = vec + str(s) + ','
                vec_list.append(vec[:-1])
                dict[src.strip()] = vec[:-1]
                with open (dict_path, 'a') as wf:
                    wf.writelines(src.strip() + " " + vec[:-1] + "\n")
        with open(file.replace('nvocab','nvocabf',1).replace('.nvcb', '.ndf', 1), "w") as wf:
            for line in vec_list:
                wf.writelines(line + '\n')