# -*- coding: utf-8 -*-
"""
Created on Tue May 11 17:23:20 2021

@author: InjoKim

Data sceintist of Seoultech 
"""

from networkx.generators.small import truncated_cube_graph
import torch
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import RobustScaler
from torch_geometric.data import InMemoryDataset, Data
from model.node_embedding import Node2vec
from model import document_embedding
from utils import non_isolated_node_name
import warnings 
warnings.filterwarnings(action='ignore')


class Load_data(InMemoryDataset) :
    
    def __init__(self, network_path, feature_path, exe_node2vec=False, exe_doc2vec=False, basic_feature=False, train=True, feature_col_names=None) :
        super(Load_data, self).__init__()
        
        ### load dataset 
        self.original = pd.read_csv(network_path, index_col=0)
        # self.attr = pd.read_csv('network_data/attribute/project_network_attr.csv')
        self.feature = pd.read_csv(feature_path)
        
        ### data preprocess
        # drop isolated nodes
        self.adjacency = self.original.values
        non_isolated_node = non_isolated_node_name(self.original)
        self.original = self.original.loc[non_isolated_node, non_isolated_node]
        self.feature = pd.DataFrame([row for idx, row in self.feature.iterrows() if row['full_name'] in non_isolated_node]).reset_index(drop=True)

        # add dioagonal term 
        diag = pd.DataFrame(np.eye(self.original.shape[0], dtype=int), columns=self.original.columns, index=self.original.index)
        self.adjacency = self.original + diag
        #merged = pd.concat([self.feature.set_index('full_name'), self.attr.set_index('Id')], axis=1)
        
        G = nx.from_numpy_matrix(self.adjacency.values)
        G = nx.relabel_nodes(G, dict(enumerate(self.adjacency.columns)))
        
        # make node feature matrix 
            # event information
        if basic_feature == True:
            node_feature = self.feature[feature_col_names]

            for col in node_feature :
                scaler = RobustScaler().fit(node_feature[col].to_numpy().reshape(-1, 1))
                node_feature[col] = scaler.transform(node_feature[col].to_numpy().reshape(-1, 1))
            node_feature = node_feature.values

        """           
            # community detection results    
        onehot = OneHotEncoder().fit(merged.modularity_class.to_numpy().reshape(-1,1))
        communities = onehot.transform(merged.modularity_class.to_numpy().reshape(-1,1)).toarray()
        """

        if basic_feature==False :
            x = torch.eye(G.number_of_nodes(), dtype=torch.float)

        elif basic_feature==True :
            x = torch.tensor(node_feature, dtype=torch.float)
        
        
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
            if basic_feature == True :
                self.data.x = torch.cat((self.data.x, self.node2vec_vector), dim=1)
            else :
                self.data.x = torch.tensor(self.node2vec_vector, dtype=torch.float)

        if exe_doc2vec == True :
            text_data = document_embedding.read_documents(self.feature.repo_id)
            doc2vec_vector = torch.tensor(document_embedding.embedding_readme(text_data), dtype=torch.float)
            if basic_feature == True or exe_node2vec == True :
                self.data.x = torch.cat((self.data.x, doc2vec_vector), dim=1)
            else : 
                self.data.x = doc2vec_vector


    def node2vec(self, dataset) :
        model = Node2vec(dataset)
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
    
    DATASET = Load_data(ROOT)
    DATA = DATASET.data
    