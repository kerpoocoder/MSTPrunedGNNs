import torch

class GraphClassificationEvaluator:
    def __init__(self, device):
        self.device = device
    
    def evaluate(self, model, test_loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                out = model(data)
                pred = out.argmax(dim=1)
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
        accuracy = correct / total if total > 0 else 0
        return {'accuracy': accuracy}