import collections
import os
import os.path as osp
import random

from sklearn import metrics
from mynn import myGlobalAttentionGATNet3
from GCN_model import GCN
import gc
import networkx as nx
from torch_geometric.data import DataLoader, DataListLoader
#from torchviz import make_dot
from visualdl import LogWriter
from sklearn.model_selection import StratifiedShuffleSplit
import evaluation as evl
import utility as ut
from AstDataset import AstDataset
import torch
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.nn import DataParallel
from collections import Counter

mm = 'GATConv'
hidden_channels = 256
weight_decay = 0 # 1e-8
lr = 0.0005
threshold = 0.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

exp = 'multi_final'
kfold_num = 7
log_path = f'./log/scalar/{exp}_{kfold_num}_{mm}_{threshold}_{hidden_channels}_{weight_decay}_{lr}'

test_dataset = AstDataset('E:\\project\\' + exp + "\\kfold_" + str(kfold_num) + '\\test')
torch.manual_seed(1006)

test_dataset = test_dataset.shuffle()
print(test_dataset)

test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
checkpoint = torch.load('C:\\Users\\ruoya\\PycharmProjects\\GCN_cus\\log\\scalar\\multi_final_7_GCN_0.5_256_0_0.00025\\model_49.pth')
#checkpoint = torch.load('C:\\Users\\ruoya\\PycharmProjects\\GCN_cus\\model\\multi_GAT_14\\model20.pt')
model = GCN(test_dataset.num_node_features, hidden_channels=hidden_channels, num_label_features=44)
model.load_state_dict(checkpoint['model'])

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = torch.nn.MultiLabelSoftMarginLoss()


def test(loader):
    model.eval()

    y_pred = []
    y_true = []

    with torch.no_grad():
        for data_list in loader:  # Iterate in batches over the training/test dataset.
            # input = data.to(device)
            out = model(data_list)

            # out, h, h2, h768 = model(data_list)
            # out = torch.sigmoid(out)
            # out = (out >= threshold).float()
            out = torch.argmax(out, dim=1)

            y = data_list.y.view(data_list.num_graphs, -1).float()
            y = torch.argmax(y, dim=1)

            y_pred.extend(out.detach().cpu().numpy())
            y_true.extend(y.detach().cpu().numpy())

        # only show the last batch accuracy
        return y_pred, y_true  # Derive ratio of correct predictions.


# === create a log directory
# to view logs, use the command
# visualdl --logdir ./log/scalar
writer = LogWriter(logdir=log_path)
for data in test_loader:
    print('====================Testing  results accuracy comparison ')
    test_y_pred, test_y_true = test(test_loader)
    print(metrics.confusion_matrix(test_y_true, test_y_pred))
    test_tp, test_precision_den, test_recall_den = evl.extract_info_from_matrix(
        metrics.confusion_matrix(test_y_true, test_y_pred))
    total_test_tp = 0
    total_test_recall_den = 0
    pos_test_tp = 0
    pos_test_precision_den = 0
    pos_test_recall_den = 0
    for i, tp in enumerate(test_tp):
        total_test_tp += test_tp[i]
        total_test_recall_den += test_recall_den[i]
        if i != len(test_tp) - 1:
            pos_test_tp += test_tp[i]
            pos_test_precision_den += test_precision_den[i]
            pos_test_recall_den += test_recall_den[i]

    print(total_test_tp)
    print(total_test_recall_den)
    print(pos_test_tp)
    print(pos_test_precision_den)
    print(pos_test_recall_den)
    test_precision_acc = pos_test_tp / pos_test_precision_den
    test_recall_acc = pos_test_tp / pos_test_recall_den
    test_f1_score = evl.f1_score(test_precision_acc, test_recall_acc)
    print(f'Test Accuracy: {(total_test_tp / total_test_recall_den):.4f}, '
          f'Test Precision: {test_precision_acc :.4f}, '
          f'Test Recall Acc: {test_recall_acc:.4f}, '
          f'Test F1 score: {test_f1_score:.4f}'
          )
    print(metrics.classification_report(test_y_true, test_y_pred, digits=3))


