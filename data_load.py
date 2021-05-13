# -*- coding: utf-8 -*-
"""
Created on Tue May 11 17:23:20 2021

@author: user
"""

import torch
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from torch_geometric.data import InMemoryDataset, Data
import warnings 
warnings.filterwarnings(action='ignore')


class Co_contribution(InMemoryDataset) :
    
    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None) :
        super(Co_contribution, self).__init__()
        
        # load dataset 
        adjacency = pd.read_csv(root, index_col=0)
        attr = pd.read_csv('network_data/attribute/project_network_attr.csv')
        table = pd.read_csv('data/data.csv')
        
        
        
        # data preprocess
        diag = pd.DataFrame(np.eye(adjacency.shape[0], dtype=int), columns=adjacency.columns, index=adjacency.index)
        adjacency = adjacency + diag
        merged = pd.concat([table.set_index('full_name'), attr.set_index('Id')], axis=1)
        
        G = nx.from_numpy_matrix(adjacency.values)
        G = nx.relabel_nodes(G, dict(enumerate(adjacency.columns)))
        
        
        
        # make node feature matrix
        
            # only using topological data 
        # x = torch.eye(G.number_of_nodes(), dtype=torch.float)
        
            # event information
        node_feature = merged[['contributor_counts', 'stargazer_counts', 'forker_counts']]

        for col in node_feature :
            scaler = RobustScaler().fit(node_feature[col].to_numpy().reshape(-1, 1))
            node_feature[col] = scaler.transform(node_feature[col].to_numpy().reshape(-1, 1))
        node_feature = node_feature.values
           
            # community detection results    
        onehot = OneHotEncoder().fit(merged.modularity_class.to_numpy().reshape(-1,1))
        communities = onehot.transform(merged.modularity_class.to_numpy().reshape(-1,1)).toarray()
        
        x = np.concatenate((node_feature, communities), axis=1)
        x = torch.tensor(x, dtype=torch.float)
        
        
        
        # make edge list 
        adj = nx.to_scipy_sparse_matrix(G).tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)
        
        data = Data(x=x, edge_index=edge_index)
        
        self.data, self.slices = self.collate([data]) 
        self.labels = adjacency.columns
        
