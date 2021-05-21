# -*- coding: utf-8 -*-
"""
Created on Fri May 14 15:26:41 2021

@author: InjoKim

Data sceintist of Seoultech 
"""

import torch
import torch.nn.functional as F
from torch_geometric.utils import train_test_split_edges, negative_sampling
from sklearn.metrics import roc_auc_score
from data_load import Co_contribution
from gnn_model import VGAE
import utils



#%%   
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
    


#%%
if __name__=='__main__' :
    EDGE_TYPE = 'normal'
    ROOT = 'network_data/'
    
    if EDGE_TYPE == 'normal' :
        ROOT = ROOT + 'gnn_contributor_coupling.csv'
        
    else :
        ROOT = ROOT + 'contributor_coupling.csv'
    
    total_epoch = 10
    original_edges = {}
    output = {}
    
    for epoch in range(total_epoch) :
        print('======================    {}    ================================='.format(epoch))
        DATASET = Co_contribution(ROOT, exe_node2vec=True)
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
        new_edge_list = utils.make_new_edge_list(new_adj, DATASET)
    
        result = utils.get_result_table(DATASET, new_adj, new_edge_list, THRESHOLD)
        result = utils.remove_original_edge_from_new_edges(ROOT, result)
        
        original_edge, new_edge = result[result['original']==1], result[result['original']==0] 
        original_edges[epoch+1] = original_edge
        output[epoch+1] = new_edge
    
    
    print('\n')
    print('Number of origianl edges : {}'.format(torch.count_nonzero(torch.tensor(DATASET.adjacency.values))))
    print('Number of new adjacency edges : {}'.format(len(result)))
    
    constructed_edge = list(utils.find_edge_occur_more_than_threshold(original_edges, threshold=8).keys())
    meaning_edge = list(utils.find_edge_occur_more_than_threshold(output, threshold=8).keys())
    

    
    #result[['node_1_idx', 'node_2_idx']].replace(DATASET.label_dict).to_csv('result/new_edge_1.csv', index=False)
    