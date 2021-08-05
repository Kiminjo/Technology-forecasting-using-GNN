# -*- coding: utf-8 -*-
"""
Created on Fri May 21 09:31:12 2021

@author: InjoKim

Data sceintist of Seoultech 
"""
import torch
import numpy as np
import pandas as pd
from collections import Counter


def non_isolated_node_name(network) :
    # network type : dataframe
    non_isolated_node = network.index[network.sum(axis=1) > 0]
    return non_isolated_node

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
    mode = 'all'
    org_adj = object_.adjacency
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
    

def remove_original_edge_from_new_edges(edge_index, edge_list) :
    org_edge = edge_index.T.tolist()
    edge_list['original'] = 0
    for idx, row in edge_list.iterrows() :
        if [row['node_1_idx'], row['node_2_idx']] in org_edge :
            edge_list['original'][idx] = 1
            
    return edge_list
    

def find_edge_occur_more_than_threshold(output_dict, threshold) :
    edges = []
    for df in output_dict.values() :
        edges += list(zip(df['node_1_idx'], df['node_2_idx']))
    
    edges = Counter(edges)
    
    return {key : value for key, value in edges.items() if value>=threshold}