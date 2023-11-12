import torch
from torch import nn
import dgl
from dgl.nn.pytorch import SAGEConv

class Graph_Classification(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, activation = "relu", dropout = 0.0, layer = 3):
        super().__init__()
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif callable(activation):
            self.act = activation
            
        self.inp = nn.Sequential(
            nn.Linear(in_feats, hidden_size),
            self.act,
        )
        
        self.gcl = nn.Sequential()
        for _ in range(layer):
            self.gcl.append(SAGEConv(hidden_size, hidden_size, "mean", feat_drop = dropout, activation = self.act, norm = nn.LayerNorm(hidden_size)))
        
        self.out = nn.Sequential(
            nn.Linear(hidden_size, out_feats),
        )
            
    def forward(self, nodes, graph):
        with graph.local_scope():
            graph.ndata['feat'] = self.inp(nodes)
            for module in self.gcl:
                graph.ndata['feat'] = module(graph, graph.ndata['feat'])
            nodes = dgl.mean_nodes(graph, 'feat')
            nodes = self.out(nodes)
        return nodes
    
class Node_Classification(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, activation = "relu", dropout = 0.0, layer = 3):
        super().__init__()
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif callable(activation):
            self.act = activation
            
        self.inp = nn.Sequential(
            nn.Linear(in_feats, hidden_size),
            self.act,
            nn.Linear(hidden_size, hidden_size),
        )
        
        self.gcl = nn.Sequential()
        for _ in range(layer):
            self.gcl.append(SAGEConv(hidden_size, hidden_size, "mean", feat_drop = dropout, activation = self.act, norm = nn.LayerNorm(hidden_size)))
        
        self.out = nn.Sequential(
            nn.Linear(hidden_size, out_feats),
        )
            
    def forward(self, nodes, graph):
        with graph.local_scope():
            graph.ndata['feat'] = self.inp(nodes)
            for module in self.gcl:
                graph.ndata['feat'] = module(graph, graph.ndata['feat'])
            nodes = self.out(graph.ndata['feat'])
        return nodes