import collections
import os
import os.path as osp
import random

from sklearn import metrics
from mynn import myGlobalAttentionGATNet3
import gc
import networkx as nx
from torch_geometric.data import DataLoader, DataListLoader
#from torchviz import make_dot
from visualdl import LogWriter
from sklearn.model_selection import StratifiedShuffleSplit
import evaluation as evl
import utility as ut
#from AstDataset import AstDataset
from CFGDataset import CFGDataset
import torch
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.nn import DataParallel
from collections import Counter

mm = 'GATConv'
hidden_channels = 1024
weight_decay = 0 # 1e-8
lr = 0.0005
threshold = 0.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

exp = 'multi_final'
kfold_num = 4
log_path = f'./log/scalar/{exp}_{kfold_num}_{mm}_{threshold}_{hidden_channels}_{weight_decay}_{lr}'

train_dataset = CFGDataset('E:\\project\\' + exp + "\\kfold_" + str(kfold_num) + '\\train')
#validation_dataset = AstDataset('E:\\project\\' + exp + "\\kfold_" + str(kfold_num) + '\\validation')
test_dataset = CFGDataset('E:\\project\\' + exp + "\\kfold_" + str(kfold_num) + '\\test')
#dataset = AstDataset(root='/home/ylzqn/data/' + exp)
print(f'Dataset: {train_dataset}:')
print('====================')
print(f'Number of train graphs: {len(train_dataset)}')


print(f'Dataset: {test_dataset}:')
print('====================')
print(f'Number of test graphs: {len(test_dataset)}')

print('===============Preparing data ==============================================')
# shuffle will increase accuracy, remember to turn it on


torch.manual_seed(1006)

train_dataset = train_dataset.shuffle()
test_dataset = test_dataset.shuffle()


train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


model = myGlobalAttentionGATNet3(test_dataset.num_node_features, hidden_channels=hidden_channels, num_label_features=44)
print(model)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = DataParallel(model)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = torch.nn.MultiLabelSoftMarginLoss()


def train():
    model.train()

    for data_list in train_loader:  # Iterate in batches over the training dataset.
        out = model(data_list)  # Perform a single forward pass.
        #y = torch.cat([torch.unsqueeze(data.y, 0).float() for data in data_list], 0).to(out.device)
        y = data_list.y.view(data_list.num_graphs, -1).float()
        loss = criterion(out, y)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    print("  Loss: ", loss.data.mean())
    print("Train Done")


def test(loader):
    model.eval()

    y_pred = []
    y_true = []

    with torch.no_grad():
        for data_list in loader:  # Iterate in batches over the training/test dataset.
            # input = data.to(device)
            out = model(data_list)

            # out, h, h2, h768 = model(data_list)
            #out = torch.sigmoid(out)
            #out = (out >= threshold).float()
            out=torch.argmax(out, dim=1)

            y = data_list.y.view(data_list.num_graphs, -1).float()
            y = torch.argmax(y, dim=1)

            y_pred.extend(out.detach().cpu().numpy())
            y_true.extend(y.detach().cpu().numpy())

        # only show the last batch accuracy
        return y_pred, y_true  # Derive ratio of correct predictions.

if not os.path.exists("C:\\Users\\ruoya\\PycharmProjects\\GCN_cus\\model\\multi_GAT_1" + str(kfold_num)):
    os.mkdir("C:\\Users\\ruoya\\PycharmProjects\\GCN_cus\\model\\multi_GAT_1" + str(kfold_num))


# === create a log directory
# to view logs, use the command
# visualdl --logdir ./log/scalar
writer = LogWriter(logdir=log_path)
for epoch in range(1, 50):
    print('Train Start')
    train()
    print(f'  ==Epoch: {epoch:03d}==')
    print('====================Training results accuracy comparison ')
    train_y_pred, train_y_true = test(train_loader)
    print(metrics.confusion_matrix(train_y_true, train_y_pred))
    train_tp, train_precision_den, train_recall_den = evl.extract_info_from_matrix(metrics.confusion_matrix(train_y_true, train_y_pred))
    total_train_tp = 0
    total_train_recall_den = 0
    pos_train_tp = 0
    pos_train_precision_den = 0
    pos_train_recall_den = 0
    for i, tp in enumerate(train_tp):
        total_train_tp +=  train_tp[i]
        total_train_recall_den += train_recall_den[i]
        if i != len(train_tp) - 1:
            pos_train_tp += train_tp[i]
            pos_train_precision_den += train_precision_den[i]
            pos_train_recall_den += train_recall_den[i]

    print(total_train_tp)
    print(total_train_recall_den)
    print(pos_train_tp)
    print(pos_train_precision_den)
    print(pos_train_recall_den)
    train_precision_acc = pos_train_tp/pos_train_precision_den
    train_recall_acc = pos_train_tp/pos_train_recall_den
    train_f1_score = evl.f1_score(train_precision_acc, train_recall_acc)
    print(f'Train Accuracy: {(total_train_tp/total_train_recall_den):.4f}, '
          f'Train Precision: {(train_precision_acc):.4f}, '
          f'Train Recall Acc: {train_recall_acc:.4f}, '
          f'Train F1 score: {train_f1_score :.4f}'
          )
    print(metrics.classification_report(train_y_true, train_y_pred, digits=3))

    print('====================Testing  results accuracy comparison ')
    test_y_pred, test_y_true = test(test_loader)
    print(metrics.confusion_matrix(test_y_true, test_y_pred))
    test_tp, test_precision_den, test_recall_den = evl.extract_info_from_matrix(metrics.confusion_matrix(test_y_true, test_y_pred))
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
    test_precision_acc = pos_test_tp/pos_test_precision_den
    test_recall_acc = pos_test_tp/pos_test_recall_den
    test_f1_score = evl.f1_score(test_precision_acc, test_recall_acc)
    print(f'Test Accuracy: {(total_test_tp/total_test_recall_den):.4f}, '
          f'Test Precision: {test_precision_acc :.4f}, '
          f'Test Recall Acc: {test_recall_acc:.4f}, '
          f'Test F1 score: { test_f1_score:.4f}'
          )
    print(metrics.classification_report(test_y_true, test_y_pred, digits=3))
    FILE = "C:\\Users\\ruoya\\PycharmProjects\\GCN_cus\\model\\multi_GAT_1" + str(kfold_num) + "/model" + str(
        epoch) + ".pt"
    torch.save(model.state_dict(), FILE)
    # save accuracy for
    writer.add_scalar(tag="Train Accuracy", step=epoch,
                      value=ut.void_nan(total_train_tp/total_train_recall_den))
    writer.add_scalar(tag="Train Precision", step=epoch,
                      value=ut.void_nan(train_precision_acc))
    writer.add_scalar(tag="Train Recall", step=epoch,
                      value=ut.void_nan(train_recall_acc))
    writer.add_scalar(tag="Train F1  Score", step=epoch,
                      value=ut.void_nan(train_f1_score))

    writer.add_scalar(tag="Test Accuracy", step=epoch,
                      value=ut.void_nan(total_test_tp/total_test_recall_den))
    writer.add_scalar(tag="Test Precision", step=epoch,
                      value=ut.void_nan(test_precision_acc))
    writer.add_scalar(tag="Test Recall", step=epoch,
                      value=ut.void_nan(test_recall_acc))
    writer.add_scalar(tag="Test F1  Score", step=epoch,
                      value=ut.void_nan(test_f1_score))
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    file_name = 'model_' + str(epoch) + '.pth'
    torch.save(state, osp.join(log_path, file_name))
writer.close()