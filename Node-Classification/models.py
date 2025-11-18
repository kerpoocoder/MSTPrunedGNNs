import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, GATConv, GCNConv, SAGEConv, ARMAConv, GINConv
from torch.nn import Sequential, Linear, ReLU, BatchNorm1d

class BaseGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.setup_layers(in_channels, hidden_channels, out_channels)
    
    def setup_layers(self, in_channels, hidden_channels, out_channels):
        raise NotImplementedError
    
    def forward(self, x, edge_index):
        raise NotImplementedError

class GCN(BaseGNN):
    def setup_layers(self, in_channels, hidden_channels, out_channels):
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphSAGE(BaseGNN):
    def setup_layers(self, in_channels, hidden_channels, out_channels):
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GAT(BaseGNN):
    def setup_layers(self, in_channels, hidden_channels, out_channels):
        self.conv1 = GATConv(in_channels, hidden_channels, heads=8)
        self.conv2 = GATConv(hidden_channels * 8, out_channels, heads=1)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GIN(BaseGNN):
    def setup_layers(self, in_channels, hidden_channels, out_channels):
        self.conv1 = GINConv(Sequential(
            Linear(in_channels, hidden_channels), ReLU(),
            Linear(hidden_channels, hidden_channels)
        ))
        self.conv2 = GINConv(Sequential(
            Linear(hidden_channels, hidden_channels), ReLU(), 
            Linear(hidden_channels, hidden_channels)
        ))
        self.fc = Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(self.fc(x), dim=1)

class ChebNet(BaseGNN):
    def setup_layers(self, in_channels, hidden_channels, out_channels):
        self.conv1 = ChebConv(in_channels, hidden_channels, K=2)
        self.conv2 = ChebConv(hidden_channels, out_channels, K=2)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class ARMAGNN(BaseGNN):
    def setup_layers(self, in_channels, hidden_channels, out_channels):
        self.conv1 = ARMAConv(in_channels, hidden_channels, num_stacks=1)
        self.conv2 = ARMAConv(hidden_channels, out_channels, num_stacks=1)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class ModelFactory:
    @staticmethod
    def create_model(model_name, in_channels, hidden_channels, out_channels, dropout_rate=0.5):
        models = {
            "GCN": GCN,
            "GraphSAGE": GraphSAGE,
            "GAT": GAT,
            "GIN": GIN,
            "ChebNet": ChebNet,
            "ARMAGNN": ARMAGNN
        }
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")
        return models[model_name](in_channels, hidden_channels, out_channels, dropout_rate)