# SPINN: Sparse Physics-Informed Neural Network

A structured pruning approach for Physics-Informed Neural Networks (PINNs) applied to manufacturing digital twins.

## Overview

SPINN achieves **68.5% parameter reduction** while maintaining prediction accuracy for tool wear and thermal displacement prediction in CNC milling operations.

## Project Structure

```
├── models/
│   ├── dense_pinn.py          # Dense PINN baseline model
│   ├── sparse_pinn.py         # Sparse PINN with structured pruning
│   ├── structured_pruning.py  # Neuron-level pruning implementation
│   ├── physics_losses.py      # Physics-informed loss functions
│   └── pruning.py             # Magnitude-based pruning utilities
├── data/
│   ├── download_data.py       # NASA milling dataset downloader
│   └── preprocess.py          # Data preprocessing pipeline
├── results/
│   ├── checkpoints/           # Trained model weights
│   ├── figures/               # Generated plots
│   ├── metrics/               # Evaluation metrics
│   └── benchmarks/            # Performance benchmarks
├── train_baseline_simple.py   # Basic dense PINN training
├── train_baseline_improved.py # Improved dense PINN with weighted loss
├── train_spinn.py             # SPINN training with magnitude pruning
├── train_spinn_structured.py  # SPINN with true structured pruning
└── requirements.txt           # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Data
```bash
python data/preprocess.py
```

### 2. Train Baseline Model
```bash
python train_baseline_improved.py
```

### 3. Train SPINN (Structured Pruning)
```bash
python train_spinn_structured.py
```

## Results

| Model | Parameters | R² Score | Compression |
|-------|-----------|----------|-------------|
| Dense PINN | 666,882 | 0.85 | - |
| SPINN | 210,000 | 0.83 | 68.5% |

## Dataset

Uses the NASA Milling Dataset for tool wear prediction:
- Input features: cutting forces, spindle speed, feed rate, depth of cut
- Output targets: tool wear (VB), thermal displacement

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn

## License

MIT License
