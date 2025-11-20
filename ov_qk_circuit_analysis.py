#!/usr/bin/env python3
"""
OV/QK Circuit Computation script for Circuit Robustness project.

Computes QK and OV circuits for each attention head and analyzes eigenvalues
for copying behavior, following the Transformer Circuits framework.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

# ---------------------------------------------------------------------------
# Dataset/tokenizer definitions (needed for unpickling saved datasets)


class Vocabulary:
    class TrieNode:
        def __init__(self):
            self.id = None
            self.next = {}

    def __init__(self):
        self.root = self.TrieNode()
        self.token_map = {}
        self.size = 0

    def add_token(self, token):
        node = self.root
        for c in token:
            if c not in node.next:
                node.next[c] = self.TrieNode()
            node = node.next[c]
        if node.id is None:
            node.id = self.size
            self.token_map[self.size] = token
            self.size += 1

    def longest_prefix_token(self, text, start):
        longest_token = None
        longest_length = 0
        node = self.root
        for i in range(start, len(text)):
            if text[i] not in node.next:
                break
            node = node.next[text[i]]
            if node.id is not None:
                longest_token = node.id
                longest_length = i - start + 1
        if longest_token is None:
            raise ValueError(f"No token found for text starting at position {start}: {text}")
        return longest_token, longest_length

    def get_token(self, token_id):
        return self.token_map[token_id]


class CountingTokenizer:
    def __init__(self):
        self.vocab = Vocabulary()
        chars = list("abcdefghijklmnopqrstuvwxyz0123456789")
        special = ["<PAD>", "<BOS>", "<EOS>", ":", " ", "Count", "the", "letter", "in"]
        for token in special + chars:
            self.vocab.add_token(token)

    def encode(self, text, include_lengths=False):
        ids = []
        i = 0
        while i < len(text):
            token_id, token_len = self.vocab.longest_prefix_token(text, i)
            ids.append((token_id, token_len) if include_lengths else token_id)
            i += token_len
        return ids

    def decode(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        return "".join(self.vocab.get_token(token_id) for token_id in ids)

    def apply_bpe(self, words, max_token_length=3):
        text = "".join(f"<BOS>{word}<EOS>" for word in words)
        ignore_tokens = {"<PAD>", "<BOS>", "<EOS>", ":", " "}
        while True:
            encoded = self.encode(text, include_lengths=True)
            pairs = {}
            merge_pair = ()
            for i in range(len(encoded) - 1):
                first, second = encoded[i], encoded[i + 1]
                if first[1] + second[1] > max_token_length:
                    continue
                tokens = self.vocab.get_token(first[0]), self.vocab.get_token(second[0])
                if any(tok in ignore_tokens for tok in tokens):
                    continue
                pair = (first[0], second[0])
                pairs[pair] = pairs.get(pair, 0) + 1
                if not merge_pair or pairs[pair] > pairs[merge_pair]:
                    merge_pair = pair
            if not merge_pair or pairs[merge_pair] < 2:
                break
            new_token = "".join(self.vocab.get_token(idx) for idx in merge_pair)
            self.vocab.add_token(new_token)


# ---------------------------------------------------------------------------
# Circuit computation utilities


@dataclass
class CircuitMetrics:
    model: str
    layer: int
    head: int
    # QK circuit stats
    qk_frobenius_norm: float
    qk_max_entry: float
    qk_mean_entry: float
    qk_std_entry: float
    # OV circuit stats
    ov_frobenius_norm: float
    ov_max_entry: float
    ov_mean_entry: float
    ov_std_entry: float
    # OV eigenvalue analysis (for copying behavior)
    ov_num_positive_eigenvalues: int
    ov_num_negative_eigenvalues: int
    ov_largest_eigenvalue: float
    ov_smallest_eigenvalue: float
    ov_sum_eigenvalues: float  # trace
    ov_max_real_eigenvalue: float
    ov_positive_eigenvalue_ratio: float
    # Top eigenvalues (for detailed analysis)
    ov_top_eigenvalues: List[float]  # Top 5 eigenvalues
    # Diagonal analysis (another copying indicator)
    ov_diagonal_mean: float
    ov_diagonal_std: float
    ov_diagonal_max: float


def load_pickle(path: Path):
    import pickle

    with path.open("rb") as f:
        return pickle.load(f)


def build_model(vocab_size: int, device: torch.device) -> HookedTransformer:
    config = HookedTransformerConfig(
        n_layers=2,
        n_heads=8,
        d_model=128,
        d_head=16,
        d_mlp=None,
        attn_only=True,
        attention_dir="causal",
        normalization_type=None,
        d_vocab=vocab_size,
        n_ctx=100,  # Match the training configuration
        init_weights=True,
        device=str(device),
    )
    return HookedTransformer(config).to(device)


def compute_qk_circuit(
    model: HookedTransformer, layer: int, head: int, device: torch.device
) -> torch.Tensor:
    """
    Compute QK circuit: W_E @ W_Q[h] @ W_K[h].T @ W_E.T
    Returns [vocab_size, vocab_size] matrix where QK[i,j] is the attention score
    for query token i attending to key token j.
    
    The attention score is: (W_E[i] @ W_Q) @ (W_E[j] @ W_K).T
    = W_E[i] @ W_Q @ W_K.T @ W_E[j].T
    So QK = W_E @ W_Q @ W_K.T @ W_E.T
    """
    W_E = model.embed.W_E  # [vocab_size, d_model]
    W_Q = model.blocks[layer].attn.W_Q[head]  # [d_model, d_head]
    W_K = model.blocks[layer].attn.W_K[head]  # [d_model, d_head]

    # Handle different weight storage formats (robustness check)
    if W_Q.shape[0] < W_Q.shape[1]:  # [d_head, d_model] format
        W_Q = W_Q.T  # Convert to [d_model, d_head]
    if W_K.shape[0] < W_K.shape[1]:  # [d_head, d_model] format
        W_K = W_K.T  # Convert to [d_model, d_head]

    # QK[i,j] = attention score for query token i and key token j
    # = (W_E[i] @ W_Q) @ (W_E[j] @ W_K).T
    # = W_E[i] @ W_Q @ W_K.T @ W_E[j].T
    # So: QK = W_E @ W_Q @ W_K.T @ W_E.T
    W_QK = W_Q @ W_K.T  # [d_model, d_head] @ [d_head, d_model] = [d_model, d_model]
    QK = W_E @ W_QK @ W_E.T  # [vocab_size, d_model] @ [d_model, d_model] @ [d_model, vocab_size] = [vocab_size, vocab_size]
    
    return QK


def compute_ov_circuit(
    model: HookedTransformer, layer: int, head: int, device: torch.device
) -> torch.Tensor:
    """
    Compute OV circuit: W_U.T @ W_O[h] @ W_V[h] @ W_E
    Returns [vocab_size, vocab_size] matrix where OV[i,j] is the effect of
    token j on logit i when attended to.
    
    The effect is: W_U.T[i] @ W_O @ W_V @ W_E[j]
    So: OV = W_U.T @ W_O @ W_V @ W_E
    
    Note: In transformer_lens, W_U is [d_model, vocab_size] and W_O is [d_head, d_model]
    """
    W_E = model.embed.W_E  # [vocab_size, d_model]
    W_V = model.blocks[layer].attn.W_V[head]  # [d_model, d_head]
    W_O = model.blocks[layer].attn.W_O[head]  # [d_head, d_model]
    W_U = model.unembed.W_U  # [d_model, vocab_size]

    # OV[i,j] = effect of token j on logit i
    # For token j: value = W_V @ W_E[j], output = W_O @ value, logit_i = W_U.T[i] @ output
    # So: OV[i,j] = W_U.T[i] @ W_O @ W_V @ W_E[j]
    # 
    # For all tokens:
    # 1. Value vectors: V = W_E @ W_V = [vocab_size, d_head]
    # 2. Output vectors: O = V @ W_O = [vocab_size, d_head] @ [d_head, d_model] = [vocab_size, d_model]
    # 3. Logits: OV = O @ W_U = [vocab_size, d_model] @ [d_model, vocab_size] = [vocab_size, vocab_size]
    
    V = W_E @ W_V  # [vocab_size, d_head]
    O = V @ W_O  # [vocab_size, d_head] @ [d_head, d_model] = [vocab_size, d_model]
    OV = O @ W_U  # [vocab_size, d_model] @ [d_model, vocab_size] = [vocab_size, vocab_size]
    
    return OV


def analyze_circuits(
    model: HookedTransformer, device: torch.device, top_k_eigenvalues: int = 5
) -> List[CircuitMetrics]:
    """
    Compute QK and OV circuits for all heads and analyze them.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    metrics = []

    model.eval()
    with torch.no_grad():
        for layer in range(n_layers):
            for head in range(n_heads):
                # Compute QK circuit
                QK = compute_qk_circuit(model, layer, head, device)
                
                # Compute OV circuit
                OV = compute_ov_circuit(model, layer, head, device)

                # QK circuit statistics (compute on GPU, then convert)
                qk_frobenius = float(torch.norm(QK, p='fro').item())
                qk_max = float(QK.max().item())
                qk_mean = float(QK.mean().item())
                qk_std = float(QK.std().item())

                # OV circuit statistics (compute on GPU, then convert)
                ov_frobenius = float(torch.norm(OV, p='fro').item())
                ov_max = float(OV.max().item())
                ov_mean = float(OV.mean().item())
                ov_std = float(OV.std().item())

                # OV eigenvalue analysis (for copying behavior) - use GPU if available
                # torch.linalg.eig is much faster on GPU than numpy on CPU
                eigenvalues, _ = torch.linalg.eig(OV)
                eigenvalues_real = eigenvalues.real.cpu().numpy()
                
                num_positive = int(np.sum(eigenvalues_real > 0))
                num_negative = int(np.sum(eigenvalues_real < 0))
                largest_eig = float(np.max(eigenvalues_real))
                smallest_eig = float(np.min(eigenvalues_real))
                sum_eigenvalues = float(np.sum(eigenvalues_real))  # trace
                positive_ratio = float(num_positive / len(eigenvalues)) if len(eigenvalues) > 0 else 0.0

                # Top eigenvalues (sorted by absolute value)
                top_eigenvalues = sorted(eigenvalues_real, key=abs, reverse=True)[:top_k_eigenvalues]
                top_eigenvalues = [float(x) for x in top_eigenvalues]

                # Diagonal analysis (another copying indicator) - compute on GPU
                diagonal = torch.diag(OV)
                ov_diagonal_mean = float(diagonal.mean().item())
                ov_diagonal_std = float(diagonal.std().item())
                ov_diagonal_max = float(diagonal.max().item())

                metrics.append(
                    CircuitMetrics(
                        model="",  # Will be set by caller
                        layer=layer,
                        head=head,
                        qk_frobenius_norm=qk_frobenius,
                        qk_max_entry=qk_max,
                        qk_mean_entry=qk_mean,
                        qk_std_entry=qk_std,
                        ov_frobenius_norm=ov_frobenius,
                        ov_max_entry=ov_max,
                        ov_mean_entry=ov_mean,
                        ov_std_entry=ov_std,
                        ov_num_positive_eigenvalues=num_positive,
                        ov_num_negative_eigenvalues=num_negative,
                        ov_largest_eigenvalue=largest_eig,
                        ov_smallest_eigenvalue=smallest_eig,
                        ov_sum_eigenvalues=sum_eigenvalues,
                        ov_max_real_eigenvalue=largest_eig,  # Same as ov_largest_eigenvalue, kept for compatibility
                        ov_positive_eigenvalue_ratio=positive_ratio,
                        ov_top_eigenvalues=top_eigenvalues,
                        ov_diagonal_mean=ov_diagonal_mean,
                        ov_diagonal_std=ov_diagonal_std,
                        ov_diagonal_max=ov_diagonal_max,
                    )
                )

    return metrics


def analyze_model(
    model_name: str,
    checkpoint_path: Path,
    tokenizer_path: Path,
    device: torch.device,
    top_k_eigenvalues: int = 5,
) -> List[CircuitMetrics]:
    tokenizer = load_pickle(tokenizer_path)
    vocab_size = tokenizer.vocab.size + 5
    model = build_model(vocab_size, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict - use strict=False to allow mismatches in pos_embed and masks
    # (we don't need these for circuit analysis, only the weight matrices)
    state_dict = checkpoint["model_state_dict"]
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # Filter out expected mismatches (positional embeddings and masks)
    # These are expected because we use strict=False to handle n_ctx differences
    expected_missing = [k for k in missing_keys if 'pos_embed' in k or 'mask' in k]
    unexpected_missing = [k for k in missing_keys if k not in expected_missing]
    if unexpected_missing:
        print(f"  Warning: Some weights were not loaded: {unexpected_missing}")

    metrics = analyze_circuits(model, device, top_k_eigenvalues=top_k_eigenvalues)
    
    # Set model name for all metrics
    for m in metrics:
        m.model = model_name

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="OV/QK circuit computation for counting models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path.cwd(),
        help="Root directory containing checkpoints and tokenizers",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ov_qk_circuit_metrics.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--top_k_eigenvalues",
        type=int,
        default=5,
        help="Number of top eigenvalues to save per head",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=20,
        help="Epoch number to load from checkpoints (e.g., 20 for checkpoint-*-epoch-20.pt)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("  Warning: Running on CPU. This will be slow. Consider using GPU for faster computation.")
    print(f"Loading checkpoints from epoch {args.epoch}")
    model_names = ["easy", "bpe-hard", "mult-hard", "length-hard", "all-hard", "mixed"]
    all_metrics: List[CircuitMetrics] = []

    for name in model_names:
        checkpoint = args.project_root / f"checkpoint-{name}-epoch-{args.epoch}.pt"
        tokenizer = args.project_root / f"train-{name}-tokenizer.pkl"
        
        if not checkpoint.exists():
            print(f"Warning: Checkpoint not found: {checkpoint}, skipping {name}")
            continue
        if not tokenizer.exists():
            print(f"Warning: Tokenizer not found: {tokenizer}, skipping {name}")
            continue
            
        print(f"Analyzing {name}...")
        try:
            metrics = analyze_model(
                name,
                checkpoint_path=checkpoint,
                tokenizer_path=tokenizer,
                device=device,
                top_k_eigenvalues=args.top_k_eigenvalues,
            )
            all_metrics.extend(metrics)
            
            # Print summary for this model
            copying_heads = [m for m in metrics if m.ov_positive_eigenvalue_ratio > 0.5]
            print(f"  Heads with >50% positive eigenvalues (copying indicator): {len(copying_heads)}/{len(metrics)}")
            if copying_heads:
                print(f"    Examples: {[f'L{m.layer}H{m.head}' for m in copying_heads[:5]]}")
        except Exception as e:
            print(f"  Error analyzing {name}: {e}")
            # Only print full traceback if it's not a state_dict loading error
            if "state_dict" not in str(e).lower() and "size mismatch" not in str(e).lower():
                import traceback
                traceback.print_exc()

    with args.output.open("w") as f:
        json.dump([asdict(m) for m in all_metrics], f, indent=2)
    print(f"\nSaved circuit metrics to {args.output}")


if __name__ == "__main__":
    main()

