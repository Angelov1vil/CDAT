from dgl.data import FraudYelpDataset, FraudAmazonDataset, CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data.utils import load_graphs, save_graphs
import dgl
import numpy as np
import torch


class Dataset:
    def __init__(self, name='tfinance', homo=True):
        self.name = name
        graph = None
        if name == 'tfinance':
            graph, label_dict = load_graphs('dataset/tfinance')
            graph = graph[0]
            graph.ndata['label'] = graph.ndata['label'].argmax(1)

        elif name == 'tsocial':
            graph, label_dict = load_graphs('dataset/tsocial')
            graph = graph[0]

        elif name == 'yelp':
            dataset = FraudYelpDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
        elif name == 'cora':
            dataset = CoraGraphDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['train_mask', 'val_mask', 'test_mask', 'feat', 'label'])
                graph = dgl.add_self_loop(graph)
        elif name == 'citeseer':
            dataset = CiteseerGraphDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['train_mask', 'val_mask', 'test_mask', 'feat', 'label'])
                graph = dgl.add_self_loop(graph)
        elif name == 'pubmed':
            dataset = PubmedGraphDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['train_mask', 'val_mask', 'test_mask', 'feat', 'label'])
                graph = dgl.add_self_loop(graph)
        elif name == 'amazon':
            dataset = FraudAmazonDataset()
            graph = dataset[0]
            if homo:
                graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
                graph = dgl.add_self_loop(graph)
           
        else:
            print('no such dataset')
            exit(1)

        graph.ndata['label'] = graph.ndata['label'].long().squeeze(-1)

        if name in ['cora', 'citeseer', 'pubmed']:
            graph.ndata['feature'] = graph.ndata['feat'].float()
        else:
            graph.ndata['feature'] = graph.ndata['feature'].float()
        print(graph)

        self.graph = graph
