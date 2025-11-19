import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import random

class GraphClassificationTrainer:
    def __init__(self, model, device, patience=40):
        self.model = model
        self.device = device
        self.patience = patience
        self.best_val_acc = 0
        self.best_model_state = None
        self.wait = 0
    
    def train(self, train_loader, val_loader, test_loader, num_epochs=200, learning_rate=0.01, drop_rate=0.5):
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss().to(self.device)
        
        for epoch in range(num_epochs):
            self.model.train()
            
            # Apply DropGraph
            current_train_loader = self._apply_dropgraph(train_loader, drop_rate, epoch)
            
            total_loss = 0
            for data in current_train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                out = self.model(data)
                loss = criterion(out, data.y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            val_acc = self.evaluate(val_loader)
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    break
        
        # Load best model and evaluate on test set
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        test_acc = self.evaluate(test_loader)
        
        return test_acc
    
    def _apply_dropgraph(self, train_loader, drop_rate, epoch):
        if drop_rate == 0:
            return train_loader
        
        # Convert DataLoader to list of graphs
        train_data = []
        for batch in train_loader:
            # Split batch into individual graphs
            for i in range(batch.num_graphs):
                graph_data = batch[i]
                train_data.append(graph_data)
        
        if drop_rate == -1:
            random.seed(epoch)
            drop_rate = random.random() / 2 + 0.3
        
        num_keep = max(1, int(len(train_data) * (1 - drop_rate)))
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        keep_indices = indices[:num_keep]
        
        new_train_data = [train_data[i] for i in keep_indices]
        new_train_loader = DataLoader(new_train_data, batch_size=train_loader.batch_size, shuffle=True)
        
        return new_train_loader
    
    def evaluate(self, loader):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                out = self.model(data)
                pred = out.argmax(dim=1)
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
        return correct / total if total > 0 else 0