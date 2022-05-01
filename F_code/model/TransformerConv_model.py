import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import global_mean_pool

class Trans(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_label_features):
        super(Trans, self).__init__()
        #torch.manual_seed(12345)
        self.conv1 = TransformerConv(num_node_features, hidden_channels)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels)
        self.conv3 = TransformerConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_label_features)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
