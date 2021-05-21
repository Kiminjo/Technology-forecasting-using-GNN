# -*- coding: utf-8 -*-
"""
Created on Mon May 17 20:24:02 2021

@author: InjoKim

Data sceintist of Seoultech 
"""

import torch
from torch_geometric.nn import Node2Vec


def train(model, loader, optimizer, device) :
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader :
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)



@torch.no_grad()
def test(model, optimizer, dataset) :
    model.eval()
    z = model()
    acc = model.loss()
    
    return z, acc
    
    
    

def node2vec(dataset) :
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(edge_index=dataset.edge_index, embedding_dim=16, walk_length=8, p=1, q=4,
                   context_size=8, walks_per_node=50, num_negative_samples=1, sparse=True).to(device)
    
    loader = model.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    epochs = 100
    
    for epoch in range(1, epochs +1) :
        loss = train(model, loader, optimizer, device)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
        
    return model
    

    

   