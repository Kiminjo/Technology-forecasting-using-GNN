# data analysis
import numpy as np
import pandas as pd

# personal library
from data_load import Co_contribution
from node2vec import node2vec
from gnn_model import VGAE
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


def embedding_using_gcn(data) :
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VGAE(data, 16).to(DEVICE)


    pos_edge_index = data.train_pos_edge_index
    neg_edge_index = negative_sampling(edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes,
                                        num_neg_samples=data.train_pos_edge_index.size(1))


    data = data.to(DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    epochs = 200
        
    BEST_VAL_PERF = TEST_PERF = 0
    for epoch in range(1, epochs+1):
        train_loss = train(model, optimizer, pos_edge_index, neg_edge_index)
        VAL_PERF, TMP_TEST_PERF = test(model, optimizer)
            
        if VAL_PERF > BEST_VAL_PERF:
            BEST_VAL_PERF = VAL_PERF
            TEST_PERF = TMP_TEST_PERF
        
        if epoch % 10 == 0 :
            LOG = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(LOG.format(epoch, train_loss, BEST_VAL_PERF, TEST_PERF))
            
    latent = model.encoder()

    return latent

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


def link_pred(eval_data) :
    X = eval_data.iloc[:, :-1].values; y = eval_data.iloc[:, -1]

    rf_result = {}
    for i in range(100) : 
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3)

        rf = RandomForestClassifier()
        rf.fit(train_X, train_y)
        rf_result[i] = roc_auc_score(test_y, rf.predict(test_X))
 
    rf_auc_result = sum([out for out in rf_result.values()])/100
    return rf_auc_result


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


def compare_models() :
    dataset = Co_contribution('network_data/gnn_contributor_coupling.csv', feature_type='topological')
    data = dataset.data
    split_data = train_test_split_edges(data, val_ratio=0.3, test_ratio=0.2)

    print('\n \n')
    print('GCN with topological data embedding start')
    gcn_latent = embedding_using_gcn(split_data)
    gcn_eval_dataset = make_eval_dataset(split_data, gcn_latent)

    print('GCN with topological AUC score : {}'.format(link_pred(gcn_eval_dataset)))

    dataset = Co_contribution('network_data/gnn_contributor_coupling.csv', feature_type='node_feature')
    data = dataset.data
    split_data = train_test_split_edges(data, val_ratio=0.3, test_ratio=0.2)

    print('\n \n')
    print('GCN with node feature embedding start')
    gcn_latent = embedding_using_gcn(split_data)
    gcn_eval_dataset = make_eval_dataset(split_data, gcn_latent)

    print('GCN with node feature AUC score : {}'.format(link_pred(gcn_eval_dataset)))

    dataset = Co_contribution('network_data/gnn_contributor_coupling.csv', exe_node2vec=True, feature_type='node_feature')
    data = dataset.data
    split_data = train_test_split_edges(data, val_ratio=0.3, test_ratio=0.2)

    print('\n \n')
    print('GCN with node feature and n2v embedding start')
    gcn_latent = embedding_using_gcn(split_data)
    gcn_eval_dataset = make_eval_dataset(split_data, gcn_latent)

    print('GCN with node feature and n2v AUC score : {}'.format(link_pred(gcn_eval_dataset)))


if __name__ == '__main__':

    """
    Run below code if you want to test, which model make best performance.
    # compare_models()

    The results of comparing the performance of each model are as follows.
    The evaluation index used AUC.
    Node feature : Three event frequency(contributors, stargazers, forkers) + One-hot vector of community -> 11 lengths of vector

    # Node2vec : 0.78
    # GCN with topological data : 0.84
    # GCN with node features : 0.87
    # GCN with node features and node2vec : 0.89
    """

    # main 
    # using GCN with node feature and node2vec 

    ROOT = 'network_data/gnn_contributor_coupling.csv'
    dataset = Co_contribution(ROOT, exe_node2vec=True, feature_type='topological')
    data = dataset.data
    split_data = train_test_split_edges(data, val_ratio=0.3, test_ratio=0.2)

    print('\n \n')
    print('GCN with node feature and n2v embedding start')
    gcn_latent = embedding_using_gcn(split_data)
    gcn_eval_dataset = make_eval_dataset(split_data, gcn_latent)

    # binary classifier learning
    X = gcn_eval_dataset.iloc[:, :-1] ; y = gcn_eval_dataset.iloc[: , -1]

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
    print('GCN with node feature and node2vec AUC score : {}'.format(sum([auc for auc in rf_auc.values()])/eval_epochs))
    print('GCN with node feature and node2vec f1 score score : {}'.format(sum([auc for auc in rf_f1.values()])/eval_epochs))
    print('GCN with node feature and node2vec precision score : {}'.format(sum([auc for auc in rf_precision.values()])/eval_epochs))
    print('GCN with node feature and node2vec recall score : {}'.format(sum([auc for auc in rf_recall.values()])/eval_epochs))

    """
    # Link prediction using original data 
    new_dataset = Co_contribution(ROOT, exe_node2vec=True, feature_type='node_feature')
    new_data = new_dataset.data
    new_data = train_test_split_edges(new_data, val_ratio=0, test_ratio=0)
    threshold = 8

    latent_vector = embedding_using_gcn(new_data)

    new_adjacency = latent_vector @ latent_vector.T
    new_adjacency = new_adjacency.sigmoid()
    print(new_adjacency)
    
    
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