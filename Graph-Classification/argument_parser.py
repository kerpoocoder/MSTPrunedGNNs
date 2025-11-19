import argparse

def create_parser():
    parser = argparse.ArgumentParser(
        description='Graph Classification with Spanning Tree Pruning and DropGraph')
    
    parser.add_argument('--prune_type', type=str, default='RandomMST',
                       choices=['RandomMST', 'LDMST', 'HDMST', 'GMST', 'NoPrune'],
                       help='Type of pruning to apply (default: RandomMST)')
    parser.add_argument('--dataset', type=str, default='MUTAG',
                       choices=['MUTAG', 'AIDS', 'NCI1', 'PROTEINS', 'IMDB-BINARY'],
                       help='Dataset to use (default: MUTAG)')
    parser.add_argument('--model', type=str, default='GraphSAGE',
                       choices=['GraphSAGE', 'GAT', 'GCN', 'GIN', 'ChebNet', 'ARMAGNN'],
                       help='GNN model to use (default: GraphSAGE)')
    
    # Model parameters
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of convolutional layers (default: 3)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension size for GNN layers (default: 64)')
    
    # Training parameters
    parser.add_argument('--num_runs', type=int, default=30,
                       help='Number of runs to perform (default: 30)')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='Number of training epochs (default: 200)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate for optimizer (default: 0.01)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--patience', type=int, default=40,
                       help='Early stopping patience (default: 40)')
    
    # DropGraph parameters
    parser.add_argument('--drop_rate', type=float, default=0.5,
                       help='Drop rate for DropGraph (default: 0.5)')
    
    # System parameters
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output_dir', type=str, default='results_gc',
                       help='Output directory (default: results_gc)')
    parser.add_argument('--data_dir', type=str, default='data_gc',
                       help='Data directory (default: data_gc)')
    
    return parser