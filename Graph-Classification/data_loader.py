import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import OneHotDegree
from collections import defaultdict
import random

class GraphClassificationDataLoader:
    def __init__(self, data_dir, dataset_name, random_seed=42):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.random_seed = random_seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(random_seed)
        random.seed(random_seed)
    
    def load_dataset(self):
        # Apply OneHotDegree transform for certain datasets
        transform = None
        if self.dataset_name in ['IMDB-BINARY']:
            transform = OneHotDegree(max_degree=1000)
        
        dataset = TUDataset(
            root=f'{self.data_dir}/{self.dataset_name}', 
            name=self.dataset_name, 
            transform=transform
        )
        
        return dataset
    
    def split_dataset(self, dataset, train_ratio=0.8, val_ratio=0.1, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        
        labels = [data.y.item() if data.y.dim() == 0 else data.y[0].item()
                  for data in dataset]
        label_to_indices = defaultdict(list)
        for i, label in enumerate(labels):
            label_to_indices[label].append(i)
        
        train_indices, val_indices, test_indices = [], [], []
        for indices in label_to_indices.values():
            n = len(indices)
            perm = torch.randperm(n)
            train_end = int(train_ratio * n)
            val_end = train_end + int(val_ratio * n)
            train_indices.extend([indices[i] for i in perm[:train_end]])
            val_indices.extend([indices[i] for i in perm[train_end:val_end]])
            test_indices.extend([indices[i] for i in perm[val_end:]])
        
        return train_indices, val_indices, test_indices
    
    def create_data_loaders(self, dataset, train_indices, val_indices, test_indices, batch_size=32):
        train_data = [dataset[i] for i in train_indices]
        val_data = [dataset[i] for i in val_indices]
        test_data = [dataset[i] for i in test_indices]
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader, train_data