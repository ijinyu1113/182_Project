#!/usr/bin/env python3
"""
Attention analysis script for Circuit Robustness project.

Computes letter-matching score, attention entropy, and flags character-detection
heads for each of the six trained models (epoch 20 checkpoints).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
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


class CountingDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch, pad_id=0, max_len=60):
    tokens = [ex["tokens"] for ex in batch]
    max_batch_len = min(max(len(t) for t in tokens), max_len)

    padded_tokens = []
    for seq in tokens:
        truncated = seq[:max_batch_len]
        padded = truncated + [pad_id] * (max_batch_len - len(truncated))
        padded_tokens.append(padded)

    return {
        "input_ids": torch.tensor(padded_tokens, dtype=torch.long),
        "question_lengths": torch.tensor(
            [min(ex["question_length"], max_batch_len - 1) for ex in batch], dtype=torch.long
        ),
    }


# ---------------------------------------------------------------------------
# Analysis utilities


@dataclass
class HeadMetrics:
    model: str
    layer: int
    head: int
    letter_match: float
    entropy: float
    is_character_head: bool


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
        n_ctx=60,
        init_weights=True,
        device=str(device),
    )
    return HookedTransformer(config).to(device)


def compute_metrics(model, dataloader, device, pad_id=0, max_batches=50):
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    letter_score_sum = torch.zeros(n_layers, n_heads, device=device)
    entropy_sum = torch.zeros(n_layers, n_heads, device=device)
    count_valid = torch.zeros(n_layers, n_heads, device=device)
    entropy_count = torch.zeros(n_layers, n_heads, device=device)

    processed_batches = 0
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch["input_ids"].to(device)
            padding_mask = tokens != pad_id
            seq_mask = padding_mask.unsqueeze(1).unsqueeze(-1) & padding_mask.unsqueeze(1).unsqueeze(2)

            _, cache = model.run_with_cache(tokens)

            same_token = (tokens.unsqueeze(2) == tokens.unsqueeze(1)) & seq_mask.squeeze(1)
            diff_token = (~same_token) & seq_mask.squeeze(1)

            same_counts = same_token.sum(-1)
            diff_counts = diff_token.sum(-1)
            valid_query = padding_mask & (same_counts > 0) & (diff_counts > 0)

            for layer in range(n_layers):
                attn = cache[f"blocks.{layer}.attn.hook_pattern"]  # [batch, heads, seq, seq]

                entropy = -(attn * (attn.clamp_min(1e-9).log())).sum(-1)
                entropy_mask = padding_mask.unsqueeze(1)
                entropy_sum[layer] += (entropy_mask * entropy).sum(dim=(0, 2))
                entropy_count[layer] += entropy_mask.sum(dim=(0, 2))

                same_mean = (attn * same_token.unsqueeze(1)).sum(-1) / same_counts.unsqueeze(1).clamp_min(1)
                diff_mean = (attn * diff_token.unsqueeze(1)).sum(-1) / diff_counts.unsqueeze(1).clamp_min(1)
                delta = same_mean - diff_mean
                mask = valid_query.unsqueeze(1)
                letter_score_sum[layer] += (delta * mask).sum(dim=(0, 2))
                count_valid[layer] += mask.sum(dim=(0, 2))

            processed_batches += 1
            if processed_batches >= max_batches:
                break

    letter_match = (letter_score_sum / count_valid.clamp_min(1)).cpu()
    entropy = (entropy_sum / entropy_count.clamp_min(1)).cpu()
    return letter_match, entropy


def flag_character_heads(letter_match, entropy):
    flat_scores = letter_match.flatten()
    flat_entropy = entropy.flatten()
    score_thr = float(flat_scores.mean() + flat_scores.std())
    entropy_thr = float(flat_entropy.mean() - flat_entropy.std())
    mask = (letter_match >= score_thr) & (entropy <= entropy_thr)
    return mask


def analyze_model(
    model_name: str,
    checkpoint_path: Path,
    tokenizer_path: Path,
    dataset_path: Path,
    device: torch.device,
    batch_size: int,
    max_batches: int,
) -> List[HeadMetrics]:
    tokenizer = load_pickle(tokenizer_path)
    vocab_size = tokenizer.vocab.size + 5
    model = build_model(vocab_size, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = load_pickle(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    letter_match, entropy = compute_metrics(model, dataloader, device, pad_id=0, max_batches=max_batches)
    character_mask = flag_character_heads(letter_match, entropy)

    metrics = []
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            metrics.append(
                HeadMetrics(
                    model=model_name,
                    layer=layer,
                    head=head,
                    letter_match=float(letter_match[layer, head]),
                    entropy=float(entropy[layer, head]),
                    is_character_head=bool(character_mask[layer, head]),
                )
            )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Attention analysis for counting models.")
    parser.add_argument("--project_root", type=Path, default=Path.cwd())
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_batches", type=int, default=125)
    parser.add_argument("--output", type=Path, default=Path("attention_metrics.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_names = ["easy", "bpe-hard", "mult-hard", "length-hard", "all-hard", "mixed"]
    all_metrics: List[HeadMetrics] = []

    for name in model_names:
        checkpoint = args.project_root / f"checkpoint-{name}-epoch-20.pt"
        tokenizer = args.project_root / f"train-{name}-tokenizer.pkl"
        dataset = args.project_root / f"test-{name}-dataset.pkl"
        print(f"Analyzing {name}:")
        metrics = analyze_model(
            name,
            checkpoint_path=checkpoint,
            tokenizer_path=tokenizer,
            dataset_path=dataset,
            device=device,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
        )
        all_metrics.extend(metrics)
        flagged = [m for m in metrics if m.is_character_head]
        print(f"  Character heads: {[f'L{m.layer}H{m.head}' for m in flagged] or 'None'}")

    with args.output.open("w") as f:
        json.dump([asdict(m) for m in all_metrics], f, indent=2)
    print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()

