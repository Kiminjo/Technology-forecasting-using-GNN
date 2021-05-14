# -*- coding: utf-8 -*-
"""
Created on Fri May 14 15:26:41 2021

@author: user
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges, negative_sampling, remove_self_loops
from sklearn.metrics import roc_auc_score
from data_load import Co_contribution



class VGAE(nn.Module) :
    def __init__(self, data, output_dim) :
        super(VGAE, self).__init__()
        
        self.data = data
        self.conv1 = GCNConv(data.num_features, 2* output_dim)
        self.conv2 = GCNConv(2* output_dim, output_dim)
        self.relu = nn.ReLU()
        
    def encoder(self) :
        x = self.conv1(x=data.x, edge_index=data.train_pos_edge_index)
        x = self.relu(x)
        latent = self.conv2(x=x, edge_index=data.train_pos_edge_index)
        return latent
    
    def decoder(self, latent, pos_edge, neg_edge) :
        edge_index = torch.cat([pos_edge, neg_edge], dim=-1)
        logits = (latent[edge_index[0]] * latent[edge_index[1]]).sum(dim=-1)
        return logits
    
    def decode_all(self, latent):
        prob_adj = latent @ latent.t()
        return prob_adj
    
    
    
def get_link_labels(pos_edge, neg_edge) :
    E = pos_edge.size(1) + neg_edge.size(1)
    link_labels = torch.zeros(E, dtype=torch.float)
    link_labels[:pos_edge.size(1)] = 1.
    return link_labels
    
    
    
def train(model, optimizer, pos_edge, neg_edge) :
    model.train()

    optimizer.zero_grad()
    latent = model.encoder()
    link_logits = model.decoder(latent, pos_edge, neg_edge)
    link_labels = get_link_labels(pos_edge, neg_edge)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
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
        link_labels = get_link_labels(pos_edge, neg_edge)
        output.append(roc_auc_score(link_labels, link_prob))
    return output
    

def convert_adj(model, edge_list) :
    adj = torch.zeros((model.data.x.shape[0], model.data.x.shape[0]), dtype=float)
    for loc in edge_list.T.split(1) :
        adj[loc[0][0], loc[0][1]] = 1.
    return adj
    
    
    

if __name__=='__main__' :
    edge_type = 'dichotomize'
    root = 'network_data/'
    
    if edge_type == 'normal' :
        root = root + 'gnn_contributor_coupling.csv'
        
    else :
        root = root + 'contributor_coupling.csv'
    
    dataset = Co_contribution(root)
    data = dataset.data
    data = train_test_split_edges(data)
    pos_edge_index = data.train_pos_edge_index
    neg_edge_index = negative_sampling(edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes,
                                       num_neg_samples=data.train_pos_edge_index.size(1))
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGAE(data, 16).to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    epochs = 200
    
    best_val_perf = test_perf = 0
    for epoch in range(1, epochs+1):
        train_loss = train(model, optimizer, pos_edge_index, neg_edge_index)
        val_perf, tmp_test_perf = test(model, optimizer)
        
        if val_perf > best_val_perf:
            best_val_perf = val_perf
            test_perf = tmp_test_perf
        log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_loss, best_val_perf, test_perf))
        
    latent = model.encoder()
    new_adj = remove_self_loops(model.decode_all(latent))[0]
    
    
    """
    new_link = adj - dataset.adjacency
    print('Number of origianl edges : {}'.format(torch.count_nonzero(torch.tensor(dataset.adjacency))))
    print('NUmber of new adjacency edges : {}'.format(torch.count_nonzero(adj)))
    print('New edges : {}'.format(torch.count_nonzero(new_link)))
    """