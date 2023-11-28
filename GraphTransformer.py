import math
import time
import torch
import numpy as np
import scipy as sp
import pandas as pd
import torch.nn as nn
import networkx as nx
import seaborn as sns
from torch import Tensor
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

if torch.cuda.is_available():
  torch.device('cuda')

"""
Utils:
Data Loader
Feature Matrix Constructor
Random Node Remover
"""

def Graph_load_batch(min_num_nodes=20, max_num_nodes=1000, name='ENZYMES', node_attributes=True, graph_labels=True):
    '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
        print('Loading graph dataset: ' + str(name))
        G = nx.Graph()
        # load data
        # path = '../dataset/' + name + '/'
        path = '/content/gdrive/My Drive/' + name + '/'
        data_adj = np.loadtxt(path + name + '_A.txt', delimiter=',').astype(int)
        if node_attributes:
            data_node_att = np.loadtxt(path + name + '_node_attributes.txt', delimiter=',')
        data_node_label = np.loadtxt(path + name + '_node_labels.txt', delimiter=',').astype(int)
        data_graph_indicator = np.loadtxt(path + name + '_graph_indicator.txt', delimiter=',').astype(int)
        if graph_labels:
            data_graph_labels = np.loadtxt(path + name + '_graph_labels.txt', delimiter=',').astype(int)
        data_tuple = list(map(tuple, data_adj))
        G.add_edges_from(data_tuple)
        for i in range(data_node_label.shape[0]):
            if node_attributes:
                G.add_node(i + 1, feature=data_node_att[i])
            G.add_node(i + 1, label=data_node_label[i])
        G.remove_nodes_from(list(nx.isolates(G)))
        graph_num = data_graph_indicator.max()
        node_list = np.arange(data_graph_indicator.shape[0]) + 1
        graphs = []
        max_nodes = 0
        for i in range(graph_num):
            nodes = node_list[data_graph_indicator == i + 1]
            G_sub = G.subgraph(nodes)
            if graph_labels:
                G_sub.graph['label'] = data_graph_labels[i]

            if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes() <= max_num_nodes:
                graphs.append(G_sub)
                if G_sub.number_of_nodes() > max_nodes:
                    max_nodes = G_sub.number_of_nodes()
        print('Loaded')

        return graphs


def feature_matrix(g):
    '''
    constructs the feautre matrix (N x 3) for the enzymes datasets
    '''
        esm = nx.get_node_attributes(g, 'label')
        piazche = np.zeros((len(esm), 3))
        for i, (k, v) in enumerate(esm.items()):
            piazche[i][v-1] = 1

        return piazche


# def remove_random_node(graph, max_size=40, min_size=10):
#     '''
#     removes a random node from the gragh
#     returns the remaining graph matrix and the removed node links
#     '''
#     if len(graph.nodes()) >= max_size or len(graph.nodes()) < min_size:
#         return None
#     relabeled_graph = nx.relabel.convert_node_labels_to_integers(graph)
#     choice = np.random.choice(list(relabeled_graph.nodes()))
#     remaining_graph = nx.to_numpy_matrix(relabeled_graph.subgraph(filter(lambda x: x != choice, list(relabeled_graph.nodes()))))
#     removed_node = nx.to_numpy_matrix(relabeled_graph)[choice]
#     graph_length = len(remaining_graph)
#     # source_graph = np.pad(remaining_graph, [(0, max_size - graph_length), (0, max_size - graph_length)])
#     # target_graph = np.copy(source_graph)
#     removed_node_row = np.asarray(removed_node)[0]
#     # target_graph[graph_length] = np.pad(removed_node_row, [(0, max_size - len(removed_node_row))])
#     return remaining_graph, removed_node_row

def prepare_graph_data(graph, max_size=40, min_size=10):
  '''
  gets a graph as an input
  returns a graph with a randomly removed node adj matrix [0], its feature matrix [1], the removed node true links [2]
  '''
        if len(graph.nodes()) >= max_size or len(graph.nodes()) < min_size:
            return None
        relabeled_graph = nx.relabel.convert_node_labels_to_integers(graph)
        choice = np.random.choice(list(relabeled_graph.nodes()))
        remaining_graph = relabeled_graph.subgraph(filter(lambda x: x != choice, list(relabeled_graph.nodes())))
        remaining_graph_adj = nx.to_numpy_matrix(remaining_graph)
        graph_length = len(remaining_graph)
        remaining_graph_adj = np.pad(remaining_graph_adj, [(0, max_size - graph_length), (0, max_size - graph_length)])
        removed_node = nx.to_numpy_matrix(relabeled_graph)[choice]
        removed_node_row = np.asarray(removed_node)[0]
        removed_node_row = np.pad(removed_node_row, [(0, max_size - len(removed_node_row))])
        return remaining_graph_adj, feature_matrix(remaining_graph),  removed_node_row

""""
Layers:
Graph Convolution
Graph Multihead Attention
Feed-Forward (as a MLP)
"""

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        '''
        x is the feature matrix constructed in feature_matrix function
        adj ham ke is adjacency matrix of the graph
        '''
        y = torch.matmul(adj, x)
        # print(y.shape)
        # print(self.weight.shape)
        y = torch.matmul(y, self.weight.double())

        return y


class GraphAttn(nn.Module):
  def __init__(self, heads, model_dim, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.key_dim = model_dim // heads
        self.heads = heads

        self.q_linear = nn.Linear(model_dim, model_dim).cuda()
        self.v_linear = nn.Linear(model_dim, model_dim).cuda()
        self.k_linear = nn.Linear(model_dim, model_dim).cuda()

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(model_dim, model_dim).cuda()

  def forward(self, query, key, value):
        # print(q, k, v)
        bs = query.size(0)

        key = self.k_linear(key.float()).view(bs, -1, self.heads, self.key_dim)
        query = self.q_linear(query.float()).view(bs, -1, self.heads, self.key_dim)
        value = self.v_linear(value.float()).view(bs, -1, self.heads, self.key_dim)

        key = key.transpose(1,2)
        query = query.transpose(1,2)
        value = value.transpose(1,2)

        scores = attention(query, key, value, self.key_dim)
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.model_dim)
        output = self.out(concat)
        output = output.view(bs, self.model_dim)

        return output


class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fully_connected1 = nn.Linear(self.input_size, self.hidden_size).cuda()
        self.relu = nn.ReLU()
        self.fully_connected2 = nn.Linear(self.hidden_size, 1).cuda()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden = self.fully_connected1(x.float())
        relu = self.relu(hidden)
        output = self.fully_connected2(relu)
        output = self.sigmoid(output)

        return output


class Hydra(nn.Module):
    def __init__(self, gcn_input, model_dim, head):
        super().__init__()

        self.GCN = GraphConv(input_dim=gcn_input, output_dim=model_dim).cuda()
        self.GAT = GraphAttn(heads=head, model_dim=model_dim).cuda()
        self.MLP = FeedForward(input_size=model_dim, hidden_size=gcn_input).cuda()

        def forward(self, x, adj):
        gcn_outputs = self.GCN(x, adj)
        gat_output = self.GAT(gcn_outputs)
        mlp_output = self.MLP(gat_output).reshape(1,-1)

        return mlp_output

""""
Train the Model
Prepare data using DataLoader
(data can't be batched)
"""

def build_model(gcn_input, model_dim, head):
    model = Hydra(gcn_input, model_dim, head).cuda()
    return model


def fn(batch):
    return batch[0]


def train_model(model, trainloader, epoch, print_every=100):
        optim = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

        model.train()
        start = time.time()
        temp = start
        total_loss = 0

        for i in range(epoch):
            for batch, data in enumerate(trainloader, 0):
                adj, features, true_links = data
                adj, features, true_links  = torch.tensor(adj).cuda(), torch.tensor(features).cuda(), torch.tensor(true_links).cuda()
                # print(adj.shape)
                # print(features.shape)
                # print(true_links.shape)
                preds = model(features, adj)
                optim.zero_grad()
                loss = F.binary_cross_entropy(preds.double(), true_links.double())
                loss.backward()
                optim.step()
                total_loss += loss.item()
            if (i + 1) % print_every == 0:
            loss_avg = total_loss / print_every
            print("time = %dm, epoch %d, iter = %d, loss = %.3f,\
            %ds per %d iters" % ((time.time() - start) // 60,\
            epoch + 1, i + 1, loss_avg, time.time() - temp,\
            print_every))
            total_loss = 0
            temp = time.time()


# prepare data
# coop = sum([list(filter(lambda x: x is not None, [prepare_graph_data(g) for g in graphs])) for i in range(10)], [])
coop = list(filter(lambda x: x is not None, [prepare_graph_data(g) for g in graphs]))
trainloader = torch.utils.data.DataLoader(coop, collate_fn=fn, batch_size=1)
model  = build_model(3, 243, 9)
train_model(model, trainloader, 100, 10)
