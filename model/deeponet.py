import torch
from torch import nn
import dgl
from .gnn import Graph_Classification, Node_Classification

class GraphDeepOnet(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, activation = "relu", dropout = 0.0, layer = 3):
        super().__init__()
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif callable(activation):
            self.act = activation
        
        self._inp_transform = None
        self._out_transform = None
        
        self.brunch = Graph_Classification(in_feats, hidden_size, hidden_size, activation, dropout, layer= layer)
        self.trunk = nn.Sequential(
            nn.Linear(in_feats, hidden_size),
            self.act,
            nn.Linear(hidden_size, hidden_size),
            self.act,
            nn.Linear(hidden_size, hidden_size),
        )
        
        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            self.act,
            nn.Linear(hidden_size, out_feats),
        )
            
    def forward(self, known_nodes, nodes, gk, g):
        if self._inp_transform is not None:
            known_nodes = self._inp_transform(known_nodes)
            nodes = self._inp_transform(nodes)
        
        field_feats = self.brunch(known_nodes, gk)
        coord_feats = self.trunk(nodes)
        coord_feats = torch.split(coord_feats, g.batch_num_nodes().tolist())
        nodes = torch.cat([field_feats[(i,), :] * coord_feats[i] for i in range(len(coord_feats))], dim=0)
        nodes = self.out(nodes)
        
        if self._out_transform is not None:
            nodes = self._out_transform(nodes)
        return nodes
    
    def apply_transform(self, inp_transform, out_transform):
        self._inp_transform = inp_transform
        self._out_transform = out_transform
        


class NodeDeepOnet(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats, activation = "relu", dropout = 0.0, layer = 3):
        super().__init__()
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif callable(activation):
            self.act = activation
        
        self._inp_transform = None
        self._out_transform = None
        
        self.brunch = Node_Classification(in_feats, hidden_size, hidden_size, activation, dropout, layer= layer)
        self.trunk = nn.Sequential(
            nn.Linear(in_feats, hidden_size),
            self.act,
            nn.Linear(hidden_size, hidden_size),
            self.act,
            nn.Linear(hidden_size, hidden_size),
        )
        
        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            self.act,
            nn.Linear(hidden_size, out_feats),
        )
            
    def forward(self, known_nodes, nodes, gk, g):
        if self._inp_transform is not None:
            known_nodes = self._inp_transform(known_nodes)
            nodes = self._inp_transform(nodes)
        
        field_feats = self.brunch(known_nodes, gk) # N * 128
        field_feats = torch.split(field_feats, gk.batch_num_nodes().tolist())
        coord_feats = self.trunk(nodes) # M * 128
        coord_feats = torch.split(coord_feats, g.batch_num_nodes().tolist())

        nodes = []
        for f, c in zip(field_feats, coord_feats):
            feat = c[None, :, :] * f[:, None, :]
            feat = feat.mean(dim=1)
            nodes.append(feat)
        nodes = torch.cat(nodes)
        nodes = self.out(nodes)
        
        if self._out_transform is not None:
            nodes = self._out_transform(nodes)
        return nodes
    
    def apply_transform(self, inp_transform, out_transform):
        self._inp_transform = inp_transform
        self._out_transform = out_transform
        
        