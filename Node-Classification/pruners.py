import numpy as np
import random
from abc import ABC, abstractmethod

class BasePruner(ABC):
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    @abstractmethod
    def prune(self, data):
        pass
    
    def _gaussian_kernel(self, v1, v2, sigma=1.0):
        squared_distance = np.sum((v1 - v2) ** 2)
        return np.exp(-squared_distance / (2 * sigma ** 2))

class NoPrune(BasePruner):
    def prune(self, data):
        return data.edge_index

class MSTGaussPrune(BasePruner):
    def prune(self, data):
        edge_index = data.edge_index.cpu().numpy()
        x = data.x.cpu().numpy()
        train_mask = data.train_mask.cpu().numpy() if hasattr(data, 'train_mask') else np.ones(x.shape[0], dtype=bool)
        
        mst_edges = self._construct_mst(edge_index, x, train_mask, dissimilarity=True)
        return self._prune_edges(edge_index, mst_edges, train_mask)
    
    def _construct_mst(self, edge_index, x, train_mask, dissimilarity):
        n = x.shape[0]
        par = list(range(n))
        edges = []
        mst = set()
        
        def find(v):
            if par[v] != v:
                par[v] = find(par[v])
            return par[v]
        
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i], edge_index[1, i]
            if train_mask[u] and train_mask[v]:
                if dissimilarity:
                    edges.append((u, v, self._gaussian_kernel(x[u], x[v])))
                else:
                    edges.append((u, v))
        
        edges = sorted(edges, key=lambda x: x[2]) if dissimilarity else random.sample(edges, len(edges))
        
        for edge in edges:
            u, v = edge[0], edge[1]
            ru, rv = find(u), find(v)
            if ru != rv:
                par[rv] = ru
                mst.add((u, v))
                mst.add((v, u))
        
        return mst
    
    def _prune_edges(self, edge_index, selected_edges, train_mask):
        pruned_edges = [[], []]
        
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i], edge_index[1, i]
            if (u, v) in selected_edges or not (train_mask[u] and train_mask[v]):
                pruned_edges[0].append(u)
                pruned_edges[1].append(v)
        
        return np.array(pruned_edges)

class RandomSTPrune(MSTGaussPrune):
    def prune(self, data):
        edge_index = data.edge_index.cpu().numpy()
        x = data.x.cpu().numpy()
        train_mask = data.train_mask.cpu().numpy() if hasattr(data, 'train_mask') else np.ones(x.shape[0], dtype=bool)
        
        mst_edges = self._construct_mst(edge_index, x, train_mask, dissimilarity=False)
        return self._prune_edges(edge_index, mst_edges, train_mask)

class PrunerFactory:
    @staticmethod
    def create_pruner(prune_type, random_seed=42):
        pruners = {
            "NoPrune": NoPrune,
            "MSTGaussPrune": MSTGaussPrune,
            "randomSTPrune": RandomSTPrune
        }
        if prune_type not in pruners:
            raise ValueError(f"Unknown prune type: {prune_type}")
        return pruners[prune_type](random_seed)