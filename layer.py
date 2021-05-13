# -*- coding: utf-8 -*-
"""
Created on Tue May 11 20:38:10 2021

@author: user
"""

import numpy as np
import pandas as pd
import torch 
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.utils import train_test_split_edges
from data_load import Co_contribution
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class GCNEncoder(torch.nn.Module) :
    def __init__(self, in_channel, out_channel) :
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channel, 2 * out_channel, cached=True)
        self.conv2 = GCNConv(2 * out_channel, out_channel, cached=True)
        
    def forward(self, x, edge_index) :
        middle = self.conv1(x, edge_index).relu()
        output = self.conv2(middle, edge_index)
        
        return output


def train(model, x, edge_index) :
    model.train()
    optimizer.zero_grad()
    embedding_vector = model.encode(x, edge_index)
    loss = model.recon_loss(embedding_vector, edge_index)
    
    loss.backward()
    optimizer.step()
    return embedding_vector, float(loss) 

def test(model, edge_index, test_pos, test_neg) :
    model.eval()
    with torch.no_grad() :
        encoding_vector = model.encode(x, edge_index)
    return model.test(encoding_vector, test_pos, test_neg)


def add_cluster_feature(cluster, dataset) :
    dict_ = dict(zip(dataset.labels, cluster))
    att = pd.read_csv('network_data/attribute/project_network_attr.csv')
    att['cluster'] = 0
    for idx in range(att.shape[0]) :
        att['cluster'][idx] = dict_[att['Id'][idx]]
        
    return att

        
    
if __name__ == '__main__' :
    # edge type has two classes
    # 1. normal
    # 2. dichotomize
    edge_type = 'normal'
    root = 'network_data/'
    
    if edge_type == 'normal' :
        root = root + 'gnn_contributor_coupling.csv'
        
    else :
        root = root + 'contributor_coupling.csv'
    
    cc_dataset = Co_contribution(root)
    dataset = cc_dataset.data
    dataset = train_test_split_edges(dataset)

    in_channel = dataset.num_features
    out_channel = 128
    epochs = 20
     
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x, edge_index = dataset.x.to(device), dataset.train_pos_edge_index.to(device)
    
    
    
    # graph embedding using graph autoencoder
    model = GAE(GCNEncoder(in_channel, out_channel))
    model = model.to(device)
    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)
    
    for epoch in range(epochs) :
        embedding, loss = train(model, x, edge_index)
        auc, ap = test(model, edge_index, dataset.test_pos_edge_index, dataset.test_neg_edge_index)
        print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
    embedding = embedding.detach()  
    
    
    
    # vector clustering with k-means clustering
    clt = KMeans(n_clusters=6)
    cluster = clt.fit(embedding).labels_
    
    
    
    # draw scatter plot 
    tsne = TSNE(n_components=2).fit_transform(embedding)
    x, y = np.array(tsne[:, 0]), np.array(tsne[:, 1])
    colormap = np.array(['r', 'g', 'b', 'y', 'o', 'p'])
    fig, ax = plt.subplots(figsize=(16,10))
    
    ax.scatter(x, y)
    plt.show()
    
    
    
    # add cluster to network attribute
    att = add_cluster_feature(cluster, cc_dataset)
    att.to_csv('network_data/attribute/project_add_cluster.csv', index=False)
    


