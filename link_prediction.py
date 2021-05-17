# -*- coding: utf-8 -*-
"""
Created on Fri May 14 15:26:41 2021

@author: user
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges, negative_sampling, remove_self_loops
from sklearn.metrics import roc_auc_score
from data_load import Co_contribution


#%%
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
    
    def decode_all(self, latent, threshold):
        prob_adj = latent @ latent.t()
        prob_adj = prob_adj.sigmoid()
        prob_adj = torch.where(prob_adj > threshold, prob_adj, torch.tensor(0.0, dtype=torch.float))
        return prob_adj
    


#%%   
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
    

#%%
def get_link_labels(pos_edge, neg_edge) :
    E = pos_edge.size(1) + neg_edge.size(1)
    link_labels = torch.zeros(E, dtype=torch.float)
    link_labels[:pos_edge.size(1)] = 1.
    return link_labels



def convert_adj(model, edge_list) :
    adj = torch.zeros((model.data.x.shape[0], model.data.x.shape[0]), dtype=float)
    for loc in edge_list.T.split(1) :
        adj[loc[0][0], loc[0][1]] = 1.
    return adj
    


def make_new_edge_list(adj, object_) :
    edge_list = pd.DataFrame((adj >0).nonzero().numpy(), columns=['node_1_idx', 'node_2_idx'])
    
    new_edge_list = []
    for idx, row in edge_list.iterrows() :
        if [row['node_2_idx'], row['node_1_idx']] not in new_edge_list :
            new_edge_list.append([row['node_1_idx'], row['node_2_idx']])
    new_edge_list = pd.DataFrame(new_edge_list)
    new_edge_list = pd.merge(new_edge_list, new_edge_list.replace(object_.label_dict), left_index=True, right_index=True)
    new_edge_list.columns = ['node_1_idx', 'node_2_idx', 'node_1', 'node_2'] 
            
    return new_edge_list


def get_result_table(object_, new_adj, new_edge, threshold) :
    # result table contains below data
    # 1. Each node name
    # 2. Probability of edge generation
    # 3. Presence on pos val or pos test edge list
    # 4. Presence on neg val or neg test edge list
    mode = 'new'
    org_adj = object_.original
    result = pd.DataFrame(new_edge, columns=['node_1_idx', 'node_2_idx', 'node_1', 'node_2'])
    
    if mode == 'all' :
        new_adj = pd.DataFrame(new_adj.detach().numpy())
    elif mode == 'new' :
        new_adj = (torch.tensor(org_adj.values) - new_adj).detach().numpy()
        new_adj = np.where(new_adj<0, -new_adj, 0)
        new_adj = pd.DataFrame(new_adj)
        
    
    # add presence probability 
    presence_prob = []
    for idx, row in result.iterrows() :
        presence_prob.append(new_adj.loc[row.node_1_idx, row.node_2_idx])
    result['prob_edge'] = presence_prob
    
    # add val or test pos existence
    existance = []
    for idx, row in result.iterrows() :
        link = [row['node_1_idx'], row['node_2_idx']] 
        for pre_edge in object_.data :
            if link in pre_edge[1].T.tolist() :
                existance.append([str(link[0]) + ',' + str(link[1]), link[0], link[1], pre_edge[0][:-11]])           
            else :
                existance.append([str(link[0]) + ',' + str(link[1]), link[0], link[1], ''])
    existance = pd.DataFrame(existance, columns=['link', 'node1', 'node2', 'type']).groupby('link').agg({'node1' : 'first',
                                                                                                        'node2' : 'first',
                                                                                                        'type' : ''.join})
    existance = existance.sort_values(by=['node1', 'node2']).reset_index(drop=True).drop(['node1', 'node2'], axis=1)
    result = pd.merge(result, existance, left_index=True, right_index=True)
    result = result.sort_values(by=['prob_edge', 'node_1_idx', 'node_2_idx'], ascending=False)
    result = result[result['prob_edge'] > threshold].reset_index(drop=True)
    
    return result
    



#%%
if __name__=='__main__' :
    EDGE_TYPE = 'dichotomize'
    ROOT = 'network_data/'
    
    if EDGE_TYPE == 'normal' :
        ROOT = ROOT + 'gnn_contributor_coupling.csv'
        
    else :
        ROOT = ROOT + 'contributor_coupling.csv'
    
    DATASET = Co_contribution(ROOT)
    DATA = DATASET.data
    data = train_test_split_edges(DATA)
    pos_edge_index = data.train_pos_edge_index
    neg_edge_index = negative_sampling(edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes,
                                       num_neg_samples=data.train_pos_edge_index.size(1))
    THRESHOLD = 0.8
    
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGAE(data, 16).to(DEVICE)
    data = data.to(DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    EPOCHS = 200
    
    BEST_VAL_PERF = TEST_PERF = 0
    for EPOCH in range(1, EPOCHS+1):
        train_loss = train(model, optimizer, pos_edge_index, neg_edge_index)
        VAL_PERF, TMP_TEST_PERF = test(model, optimizer)
        
        if VAL_PERF > BEST_VAL_PERF:
            BEST_VAL_PERF = VAL_PERF
            TEST_PERF = TMP_TEST_PERF
        LOG = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(LOG.format(EPOCH, train_loss, BEST_VAL_PERF, TEST_PERF))
        
    latent = model.encoder()
    new_adj = model.decode_all(latent, THRESHOLD)
    new_edge_list = make_new_edge_list(new_adj, DATASET)

    result = get_result_table(DATASET, new_adj, new_edge_list, THRESHOLD)
    
    print('\n')
    print('Number of origianl edges : {}'.format(torch.count_nonzero(torch.tensor(DATASET.original.values))))
    print('NUmber of new adjacency edges : {}'.format(len(result)))
    print('Number of restored edges : {}'.format(len(new_edge_list) - len(result)))
    print('Number of new edges  {}'.format(2*len(result)-len(new_edge_list)))
    
    result[['node_1_idx', 'node_2_idx']].replace(DATASET.label_dict).to_csv('result/new_edge_1.csv', index=False)
    