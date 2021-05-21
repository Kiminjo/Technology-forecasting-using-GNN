# -*- coding: utf-8 -*-
"""
Created on Tue May 11 17:23:20 2021

@author: InjoKim

Data sceintist of Seoultech 
"""

import torch
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from torch_geometric.data import InMemoryDataset, Data
from node2vec import node2vec
from sklearn.metrics import jaccard_score
import warnings 
warnings.filterwarnings(action='ignore')


class Co_contribution(InMemoryDataset) :
    
    def __init__(self, root, exe_node2vec=False, train=True, transform=None, pre_transform=None, pre_filter=None) :
        super(Co_contribution, self).__init__()
        
        # load dataset 
        self.original = pd.read_csv(root, index_col=0)
        self.attr = pd.read_csv('network_data/attribute/project_network_attr.csv')
        self.table = pd.read_csv('data/data.csv')
        
        
        
        # data preprocess
        diag = pd.DataFrame(np.eye(self.original.shape[0], dtype=int), columns=self.original.columns, index=self.original.index)
        self.adjacency = self.original + diag
        merged = pd.concat([self.table.set_index('full_name'), self.attr.set_index('Id')], axis=1)
        
        G = nx.from_numpy_matrix(self.adjacency.values)
        G = nx.relabel_nodes(G, dict(enumerate(self.adjacency.columns)))
        
        
        
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
        self.labels = self.original.columns
        self.label_dict = {idx : label for idx, label in enumerate(self.labels)}
        
        
        # node embedding vector
        if exe_node2vec == True :
            n2v = self.node2vec(self.data)
            self.node2vec_vector = n2v(torch.arange(self.data.num_nodes, device='cpu'))
            self.data.x = torch.cat((self.data.x, self.node2vec_vector), dim=1)
        
    
    def node2vec(self, dataset) :
        model = node2vec(dataset)
        return model
    
    
if __name__=='__main__' :
    def sigmoid(x) :
        return 1/(1+np.exp(-x))
    
    EDGE_TYPE = 'normal'
    ROOT = 'network_data/'
    
    if EDGE_TYPE == 'normal' :
        ROOT = ROOT + 'gnn_contributor_coupling.csv'
        
    else :
        ROOT = ROOT + 'contributor_coupling.csv'
    
    DATASET = Co_contribution(ROOT)
    DATA = DATASET.data
    
    embedding_vector = DATA.x[:, -32:]
    
    network = pd.read_csv('network_data/contributor_coupling.csv').values
    col = network[: ,0]
    network = network[:, 1:].astype(np.float32) + np.eye(network.shape[0])
    
    G = nx.from_numpy_array(network)
    
    similarity_matrix = np.zeros(network.shape)
    for node1 in range(network.shape[0]) :
        for node2 in range(network.shape[1]) :
            similarity_matrix[node1, node2] = list(nx.adamic_adar_index(G, [(node1,node2)]))[0][2]
    similarity_matrix_sig = sigmoid(similarity_matrix)
            
    vector_sim_matrix = np.zeros(network.shape)        
    for node1 in range(network.shape[0]) :
        for node2 in range(network.shape[1]) :
            vector_sim_matrix[node1, node2] = embedding_vector[node1].T @ embedding_vector[node2] 
    vector_sim_matrix_sig = sigmoid(vector_sim_matrix)        
    
            
    objective = jaccard_score(similarity_matrix, vector_sim_matrix)
    
    
