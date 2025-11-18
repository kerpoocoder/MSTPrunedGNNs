import torch
from torchmetrics.classification import MulticlassF1Score
from torch_geometric.loader import DataLoader

class Evaluator:
    def __init__(self, device):
        self.device = device
    
    def evaluate_node_classification(self, model, data):
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.max(dim=1)[1]
            
            correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
            accuracy = correct / data.test_mask.sum().item()
            
            num_classes = data.num_classes
            f1_macro = MulticlassF1Score(num_classes=num_classes, average='macro').to(self.device)
            f1_micro = MulticlassF1Score(num_classes=num_classes, average='micro').to(self.device)
            
            macro_f1 = f1_macro(pred[data.test_mask], data.y[data.test_mask])
            micro_f1 = f1_micro(pred[data.test_mask], data.y[data.test_mask])
            
            return {
                'accuracy': accuracy,
                'macro_f1': macro_f1.item(),
                'micro_f1': micro_f1.item()
            }
    
    def evaluate_ppi(self, model, test_loader):
        model.eval()
        correct = total = 0
        for data in test_loader:
            data = data.to(self.device)
            with torch.no_grad():
                out = model(data.x, data.edge_index)
            pred = (out > 0).float()
            correct += (pred == data.y).sum().item()
            total += data.y.numel()
        
        accuracy = correct / total if total > 0 else 0
        return {'accuracy': accuracy}