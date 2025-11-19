import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GINConv, ChebConv, global_mean_pool
from torch_geometric.utils import degree

class ARMAGNNConv(nn.Module):
    def __init__(self, in_channels, out_channels, K=2):
        super(ARMAGNNConv, self).__init__()
        self.K = K
        self.lin = nn.Linear(in_channels, out_channels)
        self.weights = nn.Parameter(torch.ones(K) / K)
    
    def forward(self, x, edge_index):
        row, col = edge_index
        deg = torch.bincount(row, minlength=x.size(0)).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        out = self.lin(x)
        for k in range(self.K):
            x_k = out.clone()
            for _ in range(k + 1):
                x_k = torch.sparse.mm(torch.sparse_coo_tensor(
                    edge_index, norm, (x.size(0), x.size(0))), x_k)
            out += self.weights[k] * x_k
        return out

class GraphClassificationModel(nn.Module):
    def __init__(self, model_name, num_node_features, num_classes, num_layers=3, hidden_dim=64):
        super(GraphClassificationModel, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.model_name = model_name
        
        in_channels = num_node_features
        for _ in range(num_layers):
            if model_name == 'GraphSAGE':
                self.conv_layers.append(SAGEConv(in_channels, hidden_dim))
            elif model_name == 'GAT':
                self.conv_layers.append(GATConv(in_channels, hidden_dim // 8, heads=8))
            elif model_name == 'GCN':
                self.conv_layers.append(GCNConv(in_channels, hidden_dim))
            elif model_name == 'GIN':
                mlp = nn.Sequential(
                    nn.Linear(in_channels, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.conv_layers.append(GINConv(mlp))
            elif model_name == 'ChebNet':
                self.conv_layers.append(ChebConv(in_channels, hidden_dim, K=3))
            elif model_name == 'ARMAGNN':
                self.conv_layers.append(ARMAGNNConv(in_channels, hidden_dim))
            else:
                raise ValueError(f"Model {model_name} not supported")
            in_channels = hidden_dim
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for conv in self.conv_layers:
            x = F.relu(conv(x, edge_index))
        
        x = global_mean_pool(x, batch)
        return self.classifier(x)

class ModelFactory:
    @staticmethod
    def create_model(model_name, num_node_features, num_classes, num_layers=3, hidden_dim=64):
        return GraphClassificationModel(
            model_name, num_node_features, num_classes, num_layers, hidden_dim
        )