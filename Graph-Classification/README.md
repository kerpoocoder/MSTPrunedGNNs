# Graph Classification Framework

A framework for Graph Classification with Spanning Tree Pruning and DropGraph.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kerpoocoder/MSTPrunedGNNs.git
```

2. cd to the current location:
```bash
cd MSTPrunedGNNs/Graph-Classification
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Jupyter Notebook Installation:
for Jupyter Notebooks like Kaggle or Google Colab just enough to install torch_geometric
```bash
!pip install torch_geometric
```
## Usage

### Basic Usage
```bash
python main.py --dataset MUTAG --model GraphSAGE --prune_type RandomMST
```

### All Parameters
```bash
python main.py \
    --dataset MUTAG \
    --model GraphSAGE \
    --prune_type RandomMST \
    --num_layers 3 \
    --hidden_dim 64 \
    --num_runs 30 \
    --num_epochs 200 \
    --learning_rate 0.01 \
    --batch_size 32 \
    --drop_rate 0.5 \
    --patience 40 \
    --random_seed 42 \
    --output_dir results \
    --data_dir data
```

### Supported Datasets
- MUTAG
- AIDS
- NCI1
- PROTEINS
- IMDB-BINARY

### Supported Models
- GraphSAGE
- GAT
- GCN
- GIN
- ChebNet
- ARMAGNN

### Pruning Strategies
- NoPrune: No pruning (original graph)
- RandomMST: Random Minimum Spanning Tree
- LDMST: Low Degree MST
- HDMST: High Degree MST
- GMST: Gaussian MST

## Features

- **Spanning Tree Pruning**: Convert graphs to spanning trees using various strategies
- **DropGraph**: Randomly drop graphs during training for regularization
- **Stratified Splitting**: Maintain class distribution in train/val/test splits
- **Early Stopping**: Prevent overfitting with patience-based early stopping
- **Comprehensive Reporting**: Detailed results with accuracy and robustness metrics

## Output

Results are saved in `results/graph_classification_results.json` with metrics for each run and summary statistics.

## File Structure
```
Graph-Classification/
├── argument_parser.py
├── data_loader.py
├── models.py
├── pruners.py
├── trainer.py
├── evaluator.py
├── main.py
├── reporter.py
├── requirements.txt
└── README.md
```

## Example Commands

### Run on MUTAG with GraphSAGE and Random MST pruning
```bash
python main.py --dataset MUTAG --model GraphSAGE --prune_type RandomMST --num_runs 10
```

### Run on PROTEINS with GAT and Gaussian MST pruning
```bash
python main.py --dataset PROTEINS --model GAT --prune_type GMST --hidden_dim 128 --drop_rate 0.3
```

### Run with custom parameters
```bash
python main.py --dataset IMDB-BINARY --model GIN --prune_type LDMST --learning_rate 0.005 --num_epochs 300 --batch_size 64
```

## Citation

If you use this framework in your research, please cite the original paper.