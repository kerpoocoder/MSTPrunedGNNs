# MSTGNN Framework

A framework for Graph Neural Networks with Minimum Spanning Tree-based pruning strategies.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kerpoocoder/MSTPrunedGNNs.git
```

2. cd to the current loacation
```bash
cd MSTPrunedGNNs/Node-Classification
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
python main.py --dataset Cora --model GCN --prune_type MSTGaussPrune
```

### All Parameters
```bash
python main.py \
    --dataset Cora \
    --model GCN \
    --prune_type MSTGaussPrune \
    --hidden_channels 64 \
    --learning_rate 0.01 \
    --weight_decay 1e-4 \
    --dropout_rate 0.5 \
    --epochs 200 \
    --patience 50 \
    --num_runs 30 \
    --random_seed 42 \
    --output_dir results \
    --data_dir data
```

### Supported Datasets
- Cora
- Citeseer  
- Pubmed
- PPI

### Supported Models
- GCN
- GraphSAGE
- GAT
- GIN
- ChebNet
- ARMAGNN

### Pruning Strategies
- NoPrune: No pruning (original graph)
- MSTGaussPrune: MST pruning with Gaussian similarity
- randomSTPrune: Random spanning tree pruning

## Data Format

For citation networks (Cora, Citeseer, Pubmed), place data in `data/<dataset_name>/` with:
- `x.csv`: Node features
- `y.csv`: Node labels  
- `edge_index.csv`: Edge indices

PPI dataset will be automatically downloaded.

## Output

Results are saved in `results/results.json` with metrics for each run and summary statistics.

## File Structure
```
mstgnn/
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

### Run on Cora with GCN and MST pruning
```bash
python main.py --dataset Cora --model GCN --prune_type MSTGaussPrune --num_runs 10
```

### Run on PPI with GAT and no pruning
```bash
python main.py --dataset PPI --model GAT --prune_type NoPrune --hidden_channels 256 --epochs 400
```

### Run with custom parameters
```bash
python main.py --dataset Pubmed --model GraphSAGE --prune_type randomSTPrune --learning_rate 0.005 --dropout_rate 0.3 --epochs 500
```

## Citation

If you use this framework in your research, please cite the original paper.