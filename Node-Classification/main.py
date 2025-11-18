import torch
import json
import os
import numpy as np
from argument_parser import create_parser
from data_loader import DataLoaderFactory
from models import ModelFactory
from pruners import PrunerFactory
from trainer import TrainerFactory
from evaluator import Evaluator
from reporter import Reporter

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize reporter
    reporter = Reporter(args.output_dir)
    
    results = []
    
    for run in range(args.num_runs):
        print(f"Run {run + 1}/{args.num_runs}")
        
        # Load data for this specific run
        data_loader = DataLoaderFactory.create_loader(
            args.dataset, args.data_dir, args.random_seed, run
        )
        data = data_loader.load_data()
        
        # Prune edges if needed
        pruner = PrunerFactory.create_pruner(args.prune_type, args.random_seed + run)
        
        if args.dataset == "PPI":
            # For PPI, we need to handle each split separately
            train_data = data['train']
            val_data = data['val']
            test_data = data['test']
            
            # Apply pruning to training data
            train_data.edge_index = torch.tensor(pruner.prune(train_data), dtype=torch.long)
            
        else:
            # For citation networks, apply pruning to the single graph
            pruned_edge_index = pruner.prune(data)
            data.edge_index = torch.tensor(pruned_edge_index, dtype=torch.long)
        
        # Create model
        if args.dataset == "PPI":
            in_channels = data['train'].num_features
            out_channels = data['train'].num_classes
        else:
            in_channels = data.x.shape[1]
            out_channels = data.num_classes
        
        model = ModelFactory.create_model(
            args.model, in_channels, args.hidden_channels, 
            out_channels, args.dropout_rate
        ).to(data_loader.device)
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
        
        # Create trainer and train
        trainer = TrainerFactory.create_trainer(
            model, optimizer, data_loader.device, args.dataset, args.patience
        )
        
        if args.dataset == "PPI":
            from torch_geometric.loader import DataLoader
            train_loader = DataLoader(data['train'], batch_size=2, shuffle=True)
            val_loader = DataLoader(data['val'], batch_size=2, shuffle=False)
            test_loader = DataLoader(data['test'], batch_size=2, shuffle=False)
            
            # Pass both loaders as a tuple
            trainer.train((train_loader, val_loader), args.epochs)
        else:
            trainer.train(data, args.epochs)
        
        # Evaluate
        evaluator = Evaluator(data_loader.device)
        
        if args.dataset == "PPI":
            metrics = evaluator.evaluate_ppi(model, test_loader)
        else:
            metrics = evaluator.evaluate_node_classification(model, data)
        
        # Report results
        run_result = {
            'dataset': args.dataset,
            'model': args.model,
            'prune_type': args.prune_type,
            'hidden_channels': args.hidden_channels,
            'run': run,
            **metrics
        }
        
        results.append(run_result)
        reporter.report_run(run_result)
    
    # Generate final report
    reporter.report_summary(results)

if __name__ == "__main__":
    main()