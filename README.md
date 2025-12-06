# Signal Density and Circuit Formation in Character-Counting Transformers

This repository contains code and checkpoints for studying how training data distribution affects the emergence of counting circuits in small transformers.

## Overview

We train 2-layer transformer models on a character-counting task ("Count the letter X in: [string]") under varying data distributions and analyze the resulting circuits using mechanistic interpretability techniques.

### Key Findings

- **H1 (Signal Density)**: Models trained with high target multiplicity develop specialized character-detection circuits (L1H6), while models trained on sparse targets fail to learn robust counting mechanisms.
- **H2 (Compositional Structure)**: Successful models develop compositional circuits with L0H4 performing preprocessing, L1H6 detecting target characters, and L1H4 aggregating counts.
- **H3 (Transferability)**: Despite functional similarity, circuits do not transfer between models trained on different distributions (0% improvement via activation patching).

## Repository Structure
├── model.ipynb                       # Main training notebook
├── attention_analysis.py             # Head ablation & attention analysis
├── ov_qk_circuit_analysis.py         # OV/QK eigenvalue analysis
├── activation_patching.py            # Circuit transfer experiments
├── attension_visualization.py        # Attention pattern visualization
│
├── checkpoint-{model}-epoch-{N}.pt   # Model checkpoints
├── train-{model}-dataset.pkl         # Training datasets (100K examples)
├── test-{model}-dataset.pkl          # Test datasets (10K examples)
├── train-{model}-tokenizer.pkl       # Tokenizers
│
├── key_heads_{model}.json            # Critical head analysis results
├── attention_metrics.json            # Letter-match & entropy metrics
├── ov_qk_circuit_metrics.json        # OV/QK eigenvalue results
├── patching_results.json             # Activation patching results
│
├── l1h6_pattern_ex*.png              # L1H6 attention visualizations
├── l1h4_pattern_ex*.png              # L1H4 attention visualizations
└── requirements.txt                  # Dependencies
## Training Distributions

| Model | Sequence Length | Target Multiplicity | Description |
|-------|-----------------|---------------------|-------------|
| `easy` | 5-10 chars | 1-2 | Baseline (short, sparse) |
| `mult-hard` | 5-10 chars | 3-10 | High multiplicity only |
| `length-hard` | 20-50 chars | 1-2 | Long sequences only |
| `bpe-hard` | 5-10 chars | 1-2 | BPE tokenization |
| `all-hard` | 20-50 chars | 3-10 | Long + high multiplicity |
| `mixed` | Mixed | Mixed | Combination of all |

## Installation

```bash
git clone https://github.com/ijinyu1113/182_Project.git
cd 182_Project
pip install -r requirements.txt

## Dependencies
PyTorch
TransformerLens
NumPy
Matplotlib

## Usage
### Training Models
jupyter notebook model.ipynb
### Analyzing Attention Heads
from attention_analysis import build_model, load_pickle, evaluate_accuracy

tokenizer = load_pickle("train-all-hard-tokenizer.pkl")
dataset = load_pickle("test-all-hard-dataset.pkl")
model = build_model(vocab_size=tokenizer.vocab.size + 5, device="cuda")
### Running Activation Patching
python activation_patching.py

Model Architecture
Layers: 2
Attention Heads: 8 per layer
Embedding Dimension: 128
Context Length: 64
Tokenization: Character-level
Contributors
@ijinyu1113
@WKaiZ
@siva-tanikonda
@sanjay-adhikesaven
