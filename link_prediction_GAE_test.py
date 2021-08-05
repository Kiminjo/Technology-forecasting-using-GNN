# -*- coding: utf-8 -*-
"""
Created on Fri May 14 15:26:41 2021

@author: InjoKim

Data sceintist of Seoultech 
"""

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch_geometric.utils import train_test_split_edges, negative_sampling
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from data_load import Load_data
from model.gnn_model import VGAE
import utils

import argparse
import warnings
from networkx.generators.small import truncated_cube_graph

warnings.filterwarnings(action='ignore')

 
"""""""""""""""""
Conduct link prediction using graph autoencoder
"""""""""""""""""

def train(model, optimizer, pos_edge, neg_edge) :
    model.train()

    optimizer.zero_grad()
    latent = model.encoder()
    link_logits = model.decoder(latent, pos_edge, neg_edge)
    link_labels = utils.get_link_labels(pos_edge, neg_edge)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward(retain_graph=True)
    optimizer.step()
    
    return loss



@torch.no_grad()
def test(model, optimizer) :
    model.eval()
    output = []
    
    for prefix in ['val', 'test'] :
        pos_edge = data[f'{prefix}_pos_edge_index']
        neg_edge = data[f'{prefix}_neg_edge_index']
        
        latent = model.encoder()
        link_logits = model.decoder(latent, pos_edge, neg_edge)
        link_prob = link_logits.sigmoid()
        link_labels = utils.get_link_labels(pos_edge, neg_edge)
        output.append(roc_auc_score(link_labels, link_prob))
    return output
    


if __name__=='__main__' :

    parser = argparse.ArgumentParser()
    parser.add_argument('--network_path', required=True, help='path of adjacency matrix')
    parser.add_argument('--feature_path', required=True, help='path of node feature')
    parser.add_argument('--basic_feature', required=False, default=False, type=bool, help='determine whether to use node feature information')
    parser.add_argument('--doc2vec', required=False, default=False, type=bool, help='determine whether to use doc2vec embedding vecotr as a node feature')
    parser.add_argument('--node2vec', required=False, default=False, type=bool, help='determine whether to use node2vec embedding vecotr as a node feature')
    parser.add_argument('--feature_col_names', required=False, default=None, nargs='+', help='get feature columns name')
    
    args = parser.parse_args()
    
    total_epoch = 10
    total_threshold = 0.8
    original_edges = {}
    output = {}

    # set test data set
    # edge number : 1,038
    # test edge : 200 edges
    dataset = Load_data(network_path = args.network_path, feature_path=args.feature_path, exe_node2vec=False, basic_feature=True, exe_doc2vec=False, feature_col_names=args.feature_col_names)
    data = dataset.data
    test_index = torch.tensor(np.random.randint(low=0, high=data.edge_index.size(1), size=200, dtype=np.int64))
    train_index = torch.tensor([ele for ele in range(data.edge_index.size(1)) if ele not in test_index])
    test_pos_edges = torch.index_select(data.edge_index, dim=1, index=test_index)
    test_neg_edges = negative_sampling(data.edge_index, num_neg_samples=200)
    test_edges = torch.cat((test_pos_edges, test_neg_edges), 1)
    test_labels = torch.zeros(test_edges.size(1))
    test_labels[:test_pos_edges.size(1)] = 1

    original_adjacency_matrix = torch.tensor(dataset.adjacency.values, dtype=torch.float)
    
    for epoch in range(total_epoch) :
        print('\n')
        print('======================    {}    ================================='.format(epoch+1))
        DATASET = Load_data(network_path = args.network_path, feature_path=args.feature_path, exe_node2vec=args.node2vec, basic_feature=args.basic_feature, exe_doc2vec=args.doc2vec, feature_col_names=args.feature_col_names)
        DATA = DATASET.data
        DATA.edge_index = torch.index_select(DATA.edge_index, dim=1, index=train_index)
        data = train_test_split_edges(DATA)

        pos_edge_index = data.train_pos_edge_index
        neg_edge_index = negative_sampling(edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes,
                                           num_neg_samples=data.train_pos_edge_index.size(1))

        THRESHOLD = 0.8
    
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = VGAE(data, 16).to(DEVICE)
        data = data.to(DEVICE)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
        epochs = 200
        
        print('\n')
        print('VGAE study start')
        best_val_loss = test_loss = 0
        for mini_epoch in range(1, epochs+1):
            train_loss = train(model, optimizer, pos_edge_index, neg_edge_index)
            val_loss, temp_test_loss = test(model, optimizer)
            
            if val_loss > best_val_loss:
                best_val_loss = val_loss
                test_loss = temp_test_loss

            if mini_epoch % 10 == 0 :
                log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
                print(log.format(mini_epoch, train_loss, best_val_loss, test_loss))
            
        latent = model.encoder()
        new_adj = model.decode_all(latent, THRESHOLD)

        if epoch == 0 :
            new_adjacency_matrix = new_adj
        else :
            new_adjacency_matrix += new_adj
        
    new_adjacency_matrix /= total_epoch
    new_adjacency_matrix = torch.where(new_adjacency_matrix>total_threshold, 1, 0).numpy()

    adj_hat = pd.DataFrame(new_adjacency_matrix, columns=dataset.feature.full_name, index=dataset.feature.full_name)
    adj_hat.to_csv('result/new_adj.csv', index=True)

    

