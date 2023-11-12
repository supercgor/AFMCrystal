import torch
from torch import nn
import dgl
from dgl.nn.pytorch import SAGEConv, GATv2Conv

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
        )
        
        self.gcl = nn.Sequential()
        for _ in range(layer):
            self.gcl.append(SAGEConv(hidden_size, hidden_size, "gcn", feat_drop = dropout, activation = self.act, norm = nn.LayerNorm(hidden_size)))
        
        self.out = nn.Sequential(
            nn.Linear(hidden_size, out_feats),
        )
            
    def forward(self, nodes, graph):
        nodes = self.inp(nodes)
        for module in self.gcl:
            nodes = module(graph, nodes)
        nodes = self.out(nodes)
        return nodes
    
class Node_Classification2(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, activation = "gelu", dropout = 0.0, layer = 12):
        super().__init__()
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif callable(activation):
            self.act = activation
            
        self.inp = nn.Sequential(
            nn.Linear(in_feats, hidden_size),
        )
        
        self.gcl = nn.Sequential()
        self.norms = nn.Sequential()
        for _ in range(layer):
            self.gcl.append(GATv2Conv(hidden_size, hidden_size, 1, feat_drop = dropout, attn_drop = dropout))
            self.norms.append(nn.LayerNorm(hidden_size))
        self.out = nn.Sequential(
            nn.Linear(hidden_size, out_feats),
        )
            
    def forward(self, nodes, graph):
        nodes = self.inp(nodes)
        for gcl, norm_fn in zip(self.gcl, self.norms):
            trans = gcl(graph, nodes).squeeze(1)
            trans = norm_fn(trans)
            trans = self.act(trans)
            nodes = nodes + trans
        nodes = self.out(nodes)
        return nodes