# -*- coding: utf-8 -*-
"""
Created on Thu May 20 15:58:41 2021

@author: InjoKim

Data sceintist of Seoultech 
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv 



class VGAE(nn.Module) :
    def __init__(self, data, output_dim) :
        super(VGAE, self).__init__()
        
        self.data = data
        self.conv1 = GCNConv(data.num_features, 2* output_dim, cached=True)
        self.conv2 = GCNConv(2* output_dim, output_dim, cached=True)
        self.relu = nn.ReLU()
        
    def encoder(self) :
        x = self.conv1(x=self.data.x, edge_index=self.data.train_pos_edge_index)
        x = self.relu(x)
        latent = self.conv2(x=x, edge_index=self.data.train_pos_edge_index)
        return latent
    
    def decoder(self, latent, pos_edge, neg_edge) :
        edge_index = torch.cat([pos_edge, neg_edge], dim=-1)
        logits = (latent[edge_index[0]] * latent[edge_index[1]]).sum(dim=-1)
        return logits
    
    def decode_all(self, latent, threshold):
        prob_adj = latent @ latent.t()
        prob_adj = prob_adj.sigmoid()
        prob_adj = torch.where(prob_adj > threshold, prob_adj, torch.tensor(0.0, dtype=torch.float))
        return prob_adj
    
    
    
    
class GCNEncoder(torch.nn.Module) :
    def __init__(self, in_channel, out_channel) :
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channel, 2 * out_channel, cached=True)
        self.conv2 = GCNConv(2 * out_channel, out_channel, cached=True)
        
    def forward(self, x, edge_index) :
        middle = self.conv1(x, edge_index).relu()
        output = self.conv2(middle, edge_index)
        
        return output