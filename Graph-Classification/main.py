import torch
import json
import os
import time
import numpy as np
from argument_parser import create_parser
from data_loader import GraphClassificationDataLoader
from models import ModelFactory
from pruners import PrunerFactory
from trainer import GraphClassificationTrainer
from evaluator import GraphClassificationEvaluator
from reporter import GraphClassificationReporter

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize components
    data_loader = GraphClassificationDataLoader(args.data_dir, args.dataset, args.random_seed)
    reporter = GraphClassificationReporter(args.output_dir)
    
    # Load dataset
    dataset = data_loader.load_dataset()
    print(f"Loaded dataset: {args.dataset} with {len(dataset)} graphs")
    print(f"Number of node features: {dataset.num_node_features}")
    print(f"Number of classes: {dataset.num_classes}")
    
    results = []
    execution_times = []
    
    for run in range(args.num_runs):
        print(f"Run {run + 1}/{args.num_runs}")
        run_start_time = time.time()
        
        # Split dataset
        train_indices, val_indices, test_indices = data_loader.split_dataset(
            dataset, seed=args.random_seed + run
        )
        
        # Apply pruning if needed
        if args.prune_type != 'NoPrune':
            pruner = PrunerFactory.create_pruner(args.prune_type, args.random_seed + run)
            pruned_dataset = [pruner.prune(data) for data in dataset]
        else:
            pruned_dataset = dataset
        
        # Create data loaders
        train_loader, val_loader, test_loader, train_data = data_loader.create_data_loaders(
            pruned_dataset, train_indices, val_indices, test_indices, args.batch_size
        )
        
        # Create model
        model = ModelFactory.create_model(
            args.model, dataset.num_node_features, dataset.num_classes,
            args.num_layers, args.hidden_dim
        )
        
        # Create trainer and train
        trainer = GraphClassificationTrainer(model, data_loader.device, args.patience)
        test_acc = trainer.train(
            train_loader, val_loader, test_loader,
            args.num_epochs, args.learning_rate, args.drop_rate
        )
        
        run_time = time.time() - run_start_time
        execution_times.append(run_time)
        
        # Report results
        run_result = {
            'dataset': args.dataset,
            'model': args.model,
            'prune_type': args.prune_type,
            'drop_rate': args.drop_rate,
            'run': run,
            'accuracy': test_acc,
            'execution_time': run_time
        }
        
        results.append(run_result)
        reporter.report_run(run_result)
        
        print(f"Run {run + 1} completed in {run_time:.2f}s, Accuracy: {test_acc * 100:.2f}%")
    
    # Generate final report
    reporter.report_summary(results, execution_times)

if __name__ == "__main__":
    main()