import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

class BaseTrainer:
    def __init__(self, model, optimizer, device, patience=50):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.patience = patience
        self.best_val_acc = 0
        self.best_model_state = None
        self.wait = 0
    
    def train_epoch(self, data):
        raise NotImplementedError
    
    def validate(self, data):
        raise NotImplementedError
    
    def train(self, data, epochs):
        for epoch in range(epochs):
            self.model.train()
            train_loss = self.train_epoch(data)
            
            self.model.eval()
            val_acc = self.validate(data)
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    break
        
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

class NodeClassificationTrainer(BaseTrainer):
    def train_epoch(self, data):
        self.optimizer.zero_grad()
        out = self.model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def validate(self, data):
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            pred = out[data.val_mask].max(dim=1)[1]
            correct = pred.eq(data.y[data.val_mask]).sum().item()
            return correct / data.val_mask.sum().item()

class PPITrainer(BaseTrainer):
    def __init__(self, model, optimizer, device, patience=50):
        super().__init__(model, optimizer, device, patience)
    
    def train_epoch(self, data_loaders):
        train_loader, _ = data_loaders
        self.model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = F.binary_cross_entropy_with_logits(out, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * data.num_graphs
        return total_loss / len(train_loader.dataset)
    
    def validate(self, data_loaders):
        _, val_loader = data_loaders
        self.model.eval()
        correct = total = 0
        for data in val_loader:
            data = data.to(self.device)
            with torch.no_grad():
                out = self.model(data.x, data.edge_index)
            pred = (out > 0).float()
            correct += (pred == data.y).sum().item()
            total += data.y.numel()
        return correct / total if total > 0 else 0

class TrainerFactory:
    @staticmethod
    def create_trainer(model, optimizer, device, dataset_type, patience=50):
        if dataset_type == "PPI":
            return PPITrainer(model, optimizer, device, patience)
        else:
            return NodeClassificationTrainer(model, optimizer, device, patience)