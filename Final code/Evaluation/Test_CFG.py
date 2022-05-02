import torch
#from tu_dataset_cus import TUDataset
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix
from torch_geometric.data import DataLoader
from visualdl import LogWriter
from GCN_model import GCN
import utility as ut
from AstDataset import AstDataset
import os

exp='CFG_exp1_test_bb'
hidden_channels=256
weight_decay= 1e-8
lr=0.00025
log_path=f'./log/scalar/{exp}_{hidden_channels}_{weight_decay}_{lr}'

dataset = AstDataset(root='data/' + exp)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = dataset[0]
print(data.y)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
checkpoint = torch.load('/Users/ericxiao/PycharmProjects/Experiment/model/CFG_exp1_bb_GCN/8/model19.pt')
model = GCN(dataset.num_node_features, hidden_channels=hidden_channels).to(device)
model.load_state_dict(checkpoint)


def test(loader):
    model.eval()
    pred = []
    target = []
    with torch.no_grad():
        for data in loader:  # Iterate in batches over the training/test dataset.
            data.to(device)
            out = model(data)
            out = torch.sigmoid(out)
            out = (out >= 0.5).float().detach().cpu().numpy()
            data.y = data.y.view(data.num_graphs, -1).detach().cpu().numpy()
            for i, y in enumerate(out):
                pred.append(y)
                target.append(data.y[i])

        accuracy_score = (target, pred)
        precision_acc = precision_score(target, pred)
        recall_acc = recall_score(target, pred)
        f1_acc = f1_score(target, pred)
        TN, FP, FN, TP = ut.perf_measure(target, pred)
        tn1, fp1, fn1, tp1 = confusion_matrix(target, pred).ravel()

        return accuracy_score, precision_acc, recall_acc, \
               f1_acc, TN, FP, FN, TP, tn1, fp1, fn1, tp1  # Derive ratio of correct predictions.


test(test_loader)
print("===========Test result accuracy===========")
test_accuracy_score, test_precision_acc, test_recall_acc, test_f1_score, \
test_TN, test_FP, test_FN, test_TP, tn2, fp2, fn2, tp2= test(test_loader)
print('Test Precision Macro: {:.4f}, '.format(test_precision_acc),
'Test Recall Macro: {:.4f},'.format(test_recall_acc),
'Test F1 Macro: {:.4f}'.format(test_f1_score)
)

for data in test_loader:
    out = model(data)
    out = torch.sigmoid(out)
    pred = (out >= 0.5).int().detach().cpu().numpy()
    data.y = data.y.view(data.num_graphs, -1).detach().cpu().numpy()
    if pred != data.y:
        print("==================VAL Result================")
        print(data)
        print("=====Graph Name=====")
        print(data.z)
        print("====Predict Result(Pred)=====")
        print(pred)
        print("=====Ground Truth(data.y)=====")
        print(data.y)