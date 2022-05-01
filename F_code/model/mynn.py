import torch
from torch.nn import Linear
from torch_geometric.nn import GlobalAttention,GATConv
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


import torch.nn as nn

class myGlobalAttentionGATNet3(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_label_features):
        super(myGlobalAttentionGATNet3, self).__init__()
        self.conv1 = GATConv(num_node_features, int(num_node_features/2))
        self.conv2 = GATConv(int(num_node_features/2), hidden_channels)
        self.pooling_gate_nn = Linear(hidden_channels, 1)
        self.pooling = GlobalAttention(self.pooling_gate_nn)
        self.lin = Linear(hidden_channels, num_label_features)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.pooling.reset_parameters()
        self.lin.reset_parameters()


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        # it returns the mean of graph. Because each patch has a different graph,
        # the returns [batch_size, hidden_channels]
        # the hidden_channels is the number of features
        # x = global_mean_pool(x, batch)

        # x = global_add_pool(x, batch)
        x = self.pooling(x, batch)
        x = F.relu(x)

        # x= global_sort_pool(x, batch, 1)

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        #x = torch.softmax(x) #output is the probability of the label, sum=1
        # https://discuss.pytorch.org/t/what-is-the-difference-between-bcewithlogitsloss-and-multilabelsoftmarginloss/14944/3
        return x