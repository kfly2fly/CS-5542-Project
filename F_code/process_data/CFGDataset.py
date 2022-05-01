import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
from torch_geometric.io import read_txt_array
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
import glob

print(f"Torch version: {torch.__version__}")
#print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")


class CFGDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(AstDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')

    @property
    def raw_file_names(self):
        return ''

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        # return "not_implemented"
        return ['data_0.pt', 'data_1.pt']

    def download(self):
        pass

    def process(self):
        #      adj_path=os.path.join(self.raw_dir,'adj')
        #     adj_files=[f for f in glob.glob(os.path.join(adj_path, '*/*.adj'), recursive=True)]
        #    graph_label_files= [f.replace('adj','graphlabel',1).replace('.adj', '.gf', 1)
        #                        for f in adj_files]


    #  node_feature_files =[ f.replace('adj','nvocabf',1).replace('.adj','.ndf', 1) for f in adj_files]

        gf_path = os.path.join(self.raw_dir, 'graphlabel')
        graph_label_files = [f for f in glob.glob(os.path.join(gf_path, '*/*.gf'), recursive=True)]
        adj_files = [f.replace('graphlabel', 'adj', 1).replace('.gf', '.adj', 1)
                     for f in graph_label_files]
        node_feature_files = [f.replace('graphlabel', 'nvocabf', 1).replace('.gf', '.ndf', 1) for f in
                              graph_label_files]

        for i, f in enumerate(adj_files):
            # print("Processing {}".format(f))
            edge_index = self.get_edge_index(f)
            x = self.get_node_feats_new(node_feature_files[i])
            y = self.get_graph_label(graph_label_files[i])
            z = f.split('/')[-1][:-4]
            print(z)

            data = Data(x=x, edge_index=edge_index, y=y, z=z)
            torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))

            if i % 100 == 0:
                print(f'processed {i} graphs')

        self.num_of_examples = len(adj_files)

    def len(self):
        num_files = (glob.glob(os.path.join(self.processed_dir, 'data_*.pt')))
        return len(num_files)

    '''def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)), map_location="cuda:0")
        return data'''

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)), map_location=torch.device('cpu'))
        return data

    #multi-label method name prediction
    '''def get_dict(self):
        path = os.path.join(self.raw_dir, 'MUTAG_node_dict_limited.txt')
        dict = {}
        with open(path, 'r') as f:
            src = f.read().split('\n')[:-1]
            dict = {}
            for line in src:
                t1, t2, t3 = line.split(',')
                dict[t2] = t1
        return dict'''

    def get_edge_index(self, adj_file):
        # -1 because node ID starts from 0
        edge_index = read_txt_array(adj_file, sep=',', dtype=torch.long).t() - 1
        #print("edge_index dim:")
        if len(list(edge_index.size()))==1:
            edge_index = edge_index.unsqueeze(dim=1)
        #print(edge_index.shape)
        return edge_index

    def get_node_feats(self, node_file):
        node_labels = read_txt_array(node_file, sep=',', dtype=torch.long)
        node_labels = node_labels.unsqueeze(-1)
        # node index from txt starts from 1, torch graph starts from 0
        # so we -1
        node_labels = node_labels - 1
        node_labels = node_labels.unbind(dim=-1)
        node_labels = [F.one_hot(x, num_classes=len(self.get_dict())) for x in node_labels]
        node_labels = torch.cat(node_labels, dim=-1).to(torch.float)
        return node_labels

    def get_graph_label(self, method_name_file):
        return read_txt_array(method_name_file, sep=',', dtype=torch.long)

    def get_node_feats_new(self, node_file):
        node_labels = read_txt_array(node_file, sep=',', dtype=torch.float)
        return node_labels



