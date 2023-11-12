import h5py
import os
import torch

from torch import nn
from torch.utils.data import Dataset
import dgl

class WaterDataset(Dataset):
    def __init__(self, path, pos_filter = None, testing = False):
        self._path = path
        with h5py.File(path, 'r') as f:
            self._keys = list(f.keys())
            self._len = len(self._keys)
            if testing:
                self._len = 100
        if pos_filter is None:
            self.pos_filter = self.__pos_filter
        else:
            self.pos_filter = pos_filter
    
    def __pos_filter(self, coords):
        # print(coords.shape)
        first_layer = coords[...,2] < 4
        random_select = torch.randint(0, 2, (coords.shape[0],)).bool()
        #print(random_select)
        # print(first_layer.shape, random_select.shape)
        return torch.logical_or(first_layer, random_select)
    
    def __len__(self):
        return self._len
    
    def __further_sample(self, coords, num_points, cutoff = 0.05, mul = 1):
        n = num_points
        samples = torch.rand(n * mul, 3) * 2 -1
        cdist = torch.cdist(coords, samples)
        cdist_match = (cdist < cutoff).sum(dim=0) == 0
        # cdist = cdist[:, cdist_match].sum(dim=0)
        samples = samples[cdist_match][:num_points]
        if samples.shape[0] < n:
            samples = torch.cat((samples, self.__further_sample(coords, n - samples.shape[0], cutoff, mul+1)))
        return samples
        
    def __getitem__(self, idx):
        file_name = self._keys[idx]
        hfile = h5py.File(self._path, 'r')
        data = hfile[file_name]
                
        pos = data['pos'][...,0, :3] # N * 9
        pos = torch.as_tensor(pos, dtype=torch.float)
        real_size = torch.as_tensor(data['pos'].attrs['size'], dtype=torch.float) / 2
        
        known_mask = self.pos_filter(pos)
        pos = pos / real_size - 1
        known_pos = pos[known_mask]
        pos_pos = pos
        neg_pos = self.__further_sample(pos, len(pos))
        
        g_known = dgl.knn_graph(known_pos, 10)
        g_known.ndata['pos'] = known_pos
        
        g_pos = dgl.graph(([], []), num_nodes=len(pos_pos))
        g_pos.ndata['pos'] = pos_pos
        
        g_neg = dgl.graph(([], []), num_nodes=len(neg_pos))
        g_neg.ndata['pos'] = neg_pos
        
        hfile.close()
        return file_name, g_known, g_pos, g_neg

def collate_fn(batch):
    file_names, g_known, g_pos, g_neg = zip(*batch)
    g_known = dgl.batch(g_known)
    g_pos = dgl.batch(g_pos)
    g_neg = dgl.batch(g_neg)
    return file_names, g_known, g_pos, g_neg