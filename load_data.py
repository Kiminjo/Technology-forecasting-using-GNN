"""
Created on Wed July 14 10:44:00 2021

@ author : Injo Kim
Data sceintist of Seoultech

Data load using torch data loader
"""
import numpy as np
import pandas as pd
import networkx as nx

import torch
from torch_geometric.data import InMemoryDataset, Data

from sklearn.preprocessing import RobustScaler

import document_embedding
from node_embedding import Node2vec
import warnings

warnings.filterwarnings(action='ignore')

class Load_data(InMemoryDataset) :
    
    def __init__(self, network_file_path, feature_file_path, basic_feature, doc2vec, node2vec, feature_col_names=None) :
        super(Load_data, self).__init__()

        ### load dataset
        self.adjacency = pd.read_csv(network_file_path, index_col=0) 
        self.feature = pd.read_csv(feature_file_path)


        ### data preprocess
        ## Remove isolate nodes
        self.adjacency, self.feature = self.remove_isolated_nodes()


        ## Change network data type from dataframe to networkx
        G = nx.from_numpy_matrix(self.adjacency.values)
        G = nx.relabel_nodes(G, dict(enumerate(self.adjacency.columns)))

        adj = nx.to_scipy_sparse_matrix(G).tocoo()
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)


        ## normalize feature data 
        if basic_feature == True : 
            social_feature = self.feature[feature_col_names]
            
            for col in social_feature : 
                scaler = RobustScaler().fit(self.feature[col].to_numpy().reshape(-1, 1))
                social_feature[col] = scaler.transform(self.feature[col].to_numpy().reshape(-1, 1))
            social_feature = social_feature.values

        ## Select node feature matrix 
        '''
        There are three main features in this task
        1. social metric
        2. doc2vec embedding vector
        3. node2vec embedding vector

        and also, there are three parameters in this class
        1. basic feature (social metric)
        2. doc2vec
        3. node2vec

        default is False and if you want to use some feature, change it True
        if all parameters are False, use only topological data(identify matrix) 
        '''

        if (basic_feature == False) and (doc2vec == False) :
            print('basic False, doc2vec False')
            x = torch.eye(G.number_of_nodes(), dtype=torch.float)
            print(x)

        elif (basic_feature == True) and (doc2vec == False) :
            print('basic True, doc2vec False')
            x = torch.tensor(social_feature, dtype=torch.float64)

        elif (basic_feature == False) and (doc2vec == True) :
            print('basic False, doc2vec True')
            text_data = document_embedding.read_documents(self.feature.repo_id)
            x = document_embedding.embedding_readme(text_data)
            x = torch.tensor(x, dtype=torch.float)

        else:
            print('basic True, doc2vec True')
            x_basic = torch.tensor(social_feature, dtype=torch.float64)
            
            text_data = document_embedding.read_documents(self.feature.repo_id)
            x_doc2vec = document_embedding.embedding_readme(text_data)
            x_doc2vec = torch.tensor(x_doc2vec, dtype=torch.float)
            
            x = torch.cat((x_basic, x_doc2vec), dim=1)

        x = x.double()
        ### Make data object 
        data = Data(x=x, edge_index=edge_index)

        self.data, self.slices = self.collate([data]) 
        self.labels = self.adjacency.columns
        self.label_dict = {idx : label for idx, label in enumerate(self.labels)}

        if node2vec == True :
            model = Node2vec(self.data)
            node2vec_embedding = model(torch.arange(self.data.num_nodes, device='cpu'))

            if basic_feature == False and doc2vec == False :
                self.data.x = torch.tensor(node2vec_embedding)
            else :
                self.data.x = torch.cat((self.data.x, node2vec_embedding), dim=1)


    def remove_isolated_nodes(self) :
        '''
        Get adjacency matrix and node feature matrix as input 

        check all nodes which have no link and remove it 

        finally return non isolated adjacency matrix and node feature matrix

        * input : network(dataframe), node_feature(dataframe)
        * output : non_isoltaed_network(dataframe), non_isolated_node_feature(dataframe)
        '''
        non_isolated_node_names = self.adjacency.index[self.adjacency.sum(axis=1) > 0]

        non_isolated_network = self.adjacency.loc[non_isolated_node_names, non_isolated_node_names]
        non_isolated_node_feature = pd.DataFrame([row for idx, row in self.feature.iterrows() if row['full_name'] in non_isolated_node_names]).reset_index(drop=True)

        return non_isolated_network, non_isolated_node_feature


if __name__ == '__main__':

    network_file_path = 'network_data/contributor_coupling.csv'
    feature_file_path = 'data/data.csv'

    loader = Load_data(network_file_path=network_file_path, feature_file_path=feature_file_path, basic_feature=True, doc2vec=True, node2vec=True, feature_col_names=['contributor_counts', 'stargazer_counts', 'forker_counts'])
    print(loader.data)