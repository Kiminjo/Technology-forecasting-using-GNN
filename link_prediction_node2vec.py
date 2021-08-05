# data analysis
import numpy as np
import pandas as pd

# personal library
from data_load import Co_contribution
from node2vec import node2vec
import utils

# GNN model 
import torch
from torch.nn import functional as F
from torch_geometric.utils import train_test_split_edges, negative_sampling

# evaluation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# ETC
from tqdm import tqdm


def embedding_using_node2vec(data, mode='train') : 
    n2v = node2vec(data, mode=mode)
    n2v_latent = n2v(torch.arange(data.num_nodes, device='cpu'))

    return n2v_latent



def make_eval_dataset(data, latent) :
    val_pos_df = pd.DataFrame(harmard(latent, data.val_pos_edge_index, 1))
    test_pos_df = pd.DataFrame(harmard(latent, data.test_pos_edge_index, 1))
    val_neg_df = pd.DataFrame(harmard(latent, data.val_neg_edge_index, 0))
    test_neg_df = pd.DataFrame(harmard(latent, data.test_neg_edge_index, 0))

    eval_dataset = pd.concat([val_pos_df, test_pos_df, val_neg_df, test_neg_df], ignore_index=True)

    return eval_dataset


def harmard(vectors, edge_index, label) :
    edge_index = edge_index.T
    output = []
    for edge in edge_index :
        first_node = int(edge[0]); second_node = int(edge[1])
        first_vector = vectors[first_node]; second_vector = vectors[second_node]
        edge_vector = first_vector * second_vector

        row = edge_vector.detach().tolist()
        row.append(label)
        output.append(row)

    return output


if __name__ == '__main__':
    ROOT = 'network_data/contributor_coupling.csv'
    dataset = Co_contribution(ROOT, feature_type='topological')
    data = dataset.data
    test_index = torch.tensor(np.random.randint(low=0, high=data.edge_index.size(1), size=200, dtype=np.int64))
    train_index = torch.tensor([ele for ele in range(data.edge_index.size(1)) if ele not in test_index])
    test_pos_edges = torch.index_select(data.edge_index, dim=1, index=test_index)
    test_neg_edges = negative_sampling(data.edge_index, num_neg_samples=200)
    test_edges = torch.cat((test_pos_edges, test_neg_edges), 1)
    test_labels = torch.zeros(test_edges.size(1))
    test_labels[:test_pos_edges.size(1)] = 1
    
    split_data = train_test_split_edges(data, val_ratio=0.3, test_ratio=0.2)


    # node embedding using node2vec
    print('\n \n')
    print('Node2vec embedding start')
    n2v_latent = embedding_using_node2vec(split_data)
    n2v_eval_dataset = make_eval_dataset(split_data, n2v_latent)


    # binary classifier learning
    X = n2v_eval_dataset.iloc[:, :-1] ; y = n2v_eval_dataset.iloc[: , -1]

    rf_models = {}; rf_auc, rf_f1, rf_precision, rf_recall = {}, {}, {}, {}
    eval_epochs = 100
    for epoch in range(eval_epochs) : 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        rf_models[epoch] = rf
        rf_auc[epoch] = roc_auc_score(y_test, rf.predict(X_test))
        rf_f1[epoch] = f1_score(y_test, rf.predict(X_test))
        rf_precision[epoch] = precision_score(y_test, rf.predict(X_test))
        rf_recall[epoch] = recall_score(y_test, rf.predict(X_test))

    print('\n')
    print('node2vec AUC score : {}'.format(sum([auc for auc in rf_auc.values()])/eval_epochs))
    print('node2vec F1 score score : {}'.format(sum([auc for auc in rf_f1.values()])/eval_epochs))
    print('node2vec precision score : {}'.format(sum([auc for auc in rf_precision.values()])/eval_epochs))
    print('node2vec recall score : {}'.format(sum([auc for auc in rf_recall.values()])/eval_epochs))


    """
    # Link prediction using original data 
    new_dataset = Co_contribution(ROOT, feature_type='topological')
    new_data = new_dataset.data
    threshold = 8

    latent_vector = embedding_using_node2vec(new_data, mode='all')
    new_adjacency = latent_vector @ latent_vector.T
    new_adjacency = new_adjacency.sigmoid()
    print(new_adjacency)
    print(new_adjacency.shape)
    
    new_adjacency = np.zeros((latent_vector.shape[0], latent_vector.shape[0]))
    latent_vector = latent_vector.detach().numpy()
    for row in tqdm(range(new_adjacency.shape[0])) :
        for col in range(new_adjacency.shape[1]) :
            edge_vector = latent_vector[row] * latent_vector[col]
            edge_connection = [model.predict([edge_vector]) for model in rf_models.values()]
            if edge_connection.count(1) > threshold :
                new_adjacency[row, col] = 1
    
    # Compare original network and new adjacency network
    adjacency = new_dataset.adjacency
    new_minus_origin = new_adjacency - adjacency

    
    print('generated new edge count : {}'.format((new_minus_origin==1).sum().sum()))
    print('edges exists in original network but not in : {}'.format((new_minus_origin==-1).sum().sum()))
    """
