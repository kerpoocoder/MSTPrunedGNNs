import argparse

def create_parser():
    parser = argparse.ArgumentParser(description="MSTGNN Framework")
    
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["Cora", "Citeseer", "Pubmed", "PPI"],
                       help="Dataset name")
    parser.add_argument("--model", type=str, required=True,
                       choices=["GCN", "GraphSAGE", "GAT", "GIN", "ChebNet", "ARMAGNN"],
                       help="GNN model architecture")
    parser.add_argument("--prune_type", type=str, required=True,
                       choices=["NoPrune", "MSTGaussPrune", "randomSTPrune"],
                       help="Pruning strategy")
    
    # Optional arguments with defaults
    parser.add_argument("--hidden_channels", type=int, default=64,
                       help="Hidden channel dimension")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--dropout_rate", type=float, default=0.5,
                       help="Dropout rate")
    parser.add_argument("--epochs", type=int, default=200,
                       help="Maximum number of epochs")
    parser.add_argument("--patience", type=int, default=50,
                       help="Early stopping patience")
    
    parser.add_argument("--num_runs", type=int, default=30,
                       help="Number of experiment runs")
    parser.add_argument("--random_seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Output directory")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Data directory")
    
    return parser