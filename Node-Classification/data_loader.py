import os
import pandas as pd
import numpy as np
import torch
import json
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.transforms import NormalizeFeatures

class BaseDataLoader:
    def __init__(self, data_dir, random_seed=42):
        self.data_dir = data_dir
        self.random_seed = random_seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    
    def load_data(self):
        raise NotImplementedError

class CitationDataLoader(BaseDataLoader):
    def __init__(self, dataset_name, data_dir, random_seed=42, run=0):
        super().__init__(data_dir, random_seed)
        self.dataset_name = dataset_name
        self.run = run
        self.masks_file = os.path.join(data_dir, 'masks.json')
    
    def load_data(self):
        # Try to load from CSV files first (your original format)
        dataset_path = os.path.join(self.data_dir, self.dataset_name)
        if os.path.exists(os.path.join(dataset_path, 'x.csv')):
            return self._load_from_csv(dataset_path)
        else:
            # Fall back to built-in dataset
            return self._load_from_pyg(dataset_path)
    
    def _load_from_csv(self, dataset_path):
        x = pd.read_csv(os.path.join(dataset_path, 'x.csv'), 
                       header=None).values.astype(np.float32)
        y = pd.read_csv(os.path.join(dataset_path, 'y.csv'), 
                       header=None).values.flatten().astype(np.int64)
        edge_index = pd.read_csv(os.path.join(dataset_path, 'edge_index.csv'), 
                               header=None).values.astype(np.int64)
        
        n = x.shape[0]
        
        # Load masks from file (your original approach)
        if os.path.exists(self.masks_file):
            with open(self.masks_file, 'r') as f:
                masks = json.load(f)
            
            if self.dataset_name in masks and len(masks[self.dataset_name]) > self.run:
                indices = masks[self.dataset_name][self.run]
            else:
                indices = np.random.permutation(n)
        else:
            indices = np.random.permutation(n)
        
        train_size = int(n * 0.6)
        val_size = int(n * 0.2)
        
        train_mask = np.isin(np.arange(n), indices[:train_size])
        val_mask = np.isin(np.arange(n), indices[train_size:train_size + val_size])
        test_mask = np.isin(np.arange(n), indices[train_size + val_size:])
        
        data = Data(
            x=torch.tensor(x),
            edge_index=torch.tensor(edge_index),
            y=torch.tensor(y),
            train_mask=torch.tensor(train_mask),
            val_mask=torch.tensor(val_mask),
            test_mask=torch.tensor(test_mask)
        )
        data.num_classes = int(torch.max(data.y)) + 1
        
        return data.to(self.device)
    
    def _load_from_pyg(self, dataset_path):
        # Load built-in dataset
        dataset = Planetoid(root=self.data_dir, name=self.dataset_name, transform=NormalizeFeatures())
        data = dataset[0]
        
        n = data.num_nodes
        
        # Load masks from file (your original approach)
        if os.path.exists(self.masks_file):
            with open(self.masks_file, 'r') as f:
                masks = json.load(f)
            
            if self.dataset_name in masks and len(masks[self.dataset_name]) > self.run:
                indices = masks[self.dataset_name][self.run]
            else:
                indices = np.random.permutation(n)
        else:
            indices = np.random.permutation(n)
        
        train_size = int(n * 0.6)
        val_size = int(n * 0.2)
        
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
        
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        data.num_classes = dataset.num_classes
        
        return data.to(self.device)

class PPIDataLoader(BaseDataLoader):
    def __init__(self, data_dir, random_seed=42):
        super().__init__(data_dir, random_seed)
    
    def load_data(self):
        path = os.path.join(self.data_dir, 'PPI')
        train_dataset = PPI(root=path, split='train')
        val_dataset = PPI(root=path, split='val') 
        test_dataset = PPI(root=path, split='test')
        
        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }

class DataLoaderFactory:
    @staticmethod
    def create_loader(dataset_name, data_dir, random_seed=42, run=0):
        if dataset_name in ["Cora", "Citeseer", "Pubmed"]:
            return CitationDataLoader(dataset_name, data_dir, random_seed, run)
        elif dataset_name == "PPI":
            return PPIDataLoader(data_dir, random_seed)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")