import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, degree
import networkx as nx
from collections import defaultdict
import random
import numpy as np

class BasePruner:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    def prune(self, data):
        raise NotImplementedError

class NoPrune(BasePruner):
    def prune(self, data):
        return data

class RandomMSTPruner(BasePruner):
    def prune(self, data):
        if not data.is_undirected():
            raise ValueError("This implementation assumes undirected graphs")
        
        edges = data.edge_index.t().tolist()
        edges = [tuple(sorted([u, v])) for u, v in edges]
        edges = list(set(edges))
        random.shuffle(edges)
        
        parent = list(range(data.num_nodes))
        
        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]
        
        def union(u, v):
            parent[find(u)] = find(v)
        
        spanning_tree_edges = []
        for u, v in edges:
            if find(u) != find(v):
                union(u, v)
                spanning_tree_edges.append((u, v))
        
        edge_dict = defaultdict(list)
        for i, (u, v) in enumerate(data.edge_index.t().tolist()):
            edge_dict[tuple(sorted([u, v]))].append(i)
        
        selected_indices = [
            i for u, v in spanning_tree_edges for i in edge_dict[tuple(sorted([u, v]))]]
        new_edge_index = data.edge_index[:, selected_indices]
        new_edge_attr = data.edge_attr[selected_indices] if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        
        return Data(x=data.x, edge_index=new_edge_index, y=data.y, edge_attr=new_edge_attr)

class WeightedMSTPruner(BasePruner):
    def _compute_weighted_mst(self, data, weight_type):
        if not data.is_undirected():
            raise ValueError("This implementation assumes undirected graphs")
        
        G = to_networkx(data, edge_attrs=['edge_attr'] if hasattr(data, 'edge_attr') and data.edge_attr is not None else None, to_undirected=True)
        
        weights = {}
        if weight_type == 'low_degree':
            deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
            for i, (u, v) in enumerate(data.edge_index.t().tolist()):
                deg_u = deg[u].item()
                deg_v = deg[v].item()
                weight = deg_u + deg_v + torch.sum((data.x[u] - data.x[v]) ** 2).item()
                weights[(u, v)] = weight
                weights[(v, u)] = weight
        elif weight_type == 'high_degree':
            deg = degree(data.edge_index[0], num_nodes=data.num_nodes)
            for i, (u, v) in enumerate(data.edge_index.t().tolist()):
                deg_u = deg[u].item()
                deg_v = deg[v].item()
                weight = -deg_u - deg_v + torch.sum((data.x[u] - data.x[v]) ** 2).item()
                weights[(u, v)] = weight
                weights[(v, u)] = weight
        elif weight_type == 'gaussian':
            for i, (u, v) in enumerate(data.edge_index.t().tolist()):
                node_u_features = data.x[u]
                node_v_features = data.x[v]
                squared_distance = torch.sum((node_u_features - node_v_features) ** 2).item()
                weights[(u, v)] = squared_distance
                weights[(v, u)] = squared_distance
        
        for u, v, d in G.edges(data=True):
            key = tuple(sorted([u, v]))
            d['weight'] = weights.get((u, v)) or weights.get((v, u)) or 1
        
        spanning_tree = nx.minimum_spanning_tree(G, weight='weight')
        spanning_tree_edge_sets = [tuple(sorted(e)) for e in spanning_tree.edges()]
        
        edge_dict = defaultdict(list)
        for i, (u, v) in enumerate(data.edge_index.t().tolist()):
            edge_dict[tuple(sorted([u, v]))].append(i)
        
        selected_indices = [i for key in spanning_tree_edge_sets for i in edge_dict[key]]
        new_edge_index = data.edge_index[:, selected_indices]
        new_edge_attr = data.edge_attr[selected_indices] if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
        
        return Data(x=data.x, edge_index=new_edge_index, y=data.y, edge_attr=new_edge_attr)

class LDMSTPruner(WeightedMSTPruner):
    def prune(self, data):
        return self._compute_weighted_mst(data, 'low_degree')
    
class HDMSTPruner(WeightedMSTPruner):
    def prune(self, data):
        return self._compute_weighted_mst(data, 'high_degree')
    
class GMSTPruner(WeightedMSTPruner):
    def prune(self, data):
        return self._compute_weighted_mst(data, 'gaussian')

class PrunerFactory:
    @staticmethod
    def create_pruner(prune_type, random_seed=42):
        pruners = {
            "NoPrune": NoPrune,
            "RandomMST": RandomMSTPruner,
            "LDMST": LDMSTPruner,
            "HDMST": HDMSTPruner,
            "GMST": GMSTPruner
        }
        if prune_type not in pruners:
            raise ValueError(f"Unknown prune type: {prune_type}")
        return pruners[prune_type](random_seed)