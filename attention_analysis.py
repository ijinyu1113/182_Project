#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer, HookedTransformerConfig


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


def collate_fn(batch, pad_id=0, max_len=100):
    tokens = [ex['tokens'] for ex in batch]
    max_batch_len = min(max(len(t) for t in tokens), max_len)
    
    padded_tokens = []
    masks = []
    
    for ex in batch:
        seq = ex['tokens'][:max_batch_len]
        q_len = min(ex['question_length'], max_batch_len - 1)
        
        padded = seq + [pad_id] * (max_batch_len - len(seq))
        padded_tokens.append(padded)
        
        mask = [0] * max_batch_len
        if q_len < len(seq):
            mask[q_len] = 1
        masks.append(mask)
    
    return {
        'input_ids': torch.tensor(padded_tokens, dtype=torch.long),
        'loss_mask': torch.tensor(masks, dtype=torch.float),
        'answers': torch.tensor([ex['answer'] for ex in batch], dtype=torch.long)
    }


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


def infer_config_from_checkpoint(checkpoint_path: Path, vocab_size: int) -> HookedTransformerConfig:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    
    embed_shape = state_dict["embed.W_E"].shape
    pos_shape = state_dict["pos_embed.W_pos"].shape
    attn_q_shape = state_dict["blocks.0.attn.W_Q"].shape
    
    d_model = embed_shape[1]
    n_ctx = pos_shape[0]
    n_heads = attn_q_shape[0]
    d_head = attn_q_shape[2]
    
    config = HookedTransformerConfig(
        n_layers=2,
        n_heads=n_heads,
        d_model=d_model,
        d_head=d_head,
        d_mlp=None,
        attn_only=True,
        attention_dir="causal",
        normalization_type=None,
        d_vocab=vocab_size,
        n_ctx=n_ctx,
        init_weights=True,
        device="cpu",
    )
    return config


def build_model(vocab_size: int, device: torch.device, checkpoint_path: Path | None = None) -> HookedTransformer:
    if checkpoint_path is not None:
        config = infer_config_from_checkpoint(checkpoint_path, vocab_size)
    else:
        config = HookedTransformerConfig(
            n_layers=2,
            n_heads=8,
            d_model=256,
            d_head=16,
            d_mlp=None,
            attn_only=True,
            attention_dir="causal",
            normalization_type=None,
            d_vocab=vocab_size,
            n_ctx=100,
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
                attn = cache[f"blocks.{layer}.attn.hook_pattern"]

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


def identify_key_heads(model, dataloader, device, max_batches=200):
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    
    baseline_acc = evaluate_accuracy(model, dataloader, device, max_batches=max_batches)
    
    accuracy_drops = torch.zeros(n_layers, n_heads)
    
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    print("Ablating heads to identify key heads...")
    
    for layer in range(n_layers):
        for head in range(n_heads):
            def make_ablation_hook(target_head):
                def hook_fn(activation, hook):
                    activation[:, :, target_head, :] = 0
                    return activation
                return hook_fn
            
            hook_name = f"blocks.{layer}.attn.hook_z"
            
            ablated_acc = evaluate_accuracy(
                model, dataloader, device, 
                hooks=[(hook_name, make_ablation_hook(head))],
                max_batches=max_batches
            )
            
            accuracy_drop = baseline_acc - ablated_acc
            accuracy_drops[layer, head] = accuracy_drop
            
            if accuracy_drop > 0.01:
                print(f"  L{layer}H{head}: drop={accuracy_drop:.4f} (baseline={baseline_acc:.4f} -> ablated={ablated_acc:.4f})")
    
    return accuracy_drops.cpu(), baseline_acc


def get_ablated_accuracy(baseline_acc, accuracy_drops):
    ablated_accuracies = {}
    for layer in range(accuracy_drops.shape[0]):
        for head in range(accuracy_drops.shape[1]):
            drop = float(accuracy_drops[layer, head])
            if drop > 0.01:
                ablated_acc = baseline_acc - drop
                ablated_accuracies[(layer, head)] = ablated_acc
    return ablated_accuracies


def evaluate_accuracy(model, dataloader, device, hooks=None, max_batches=200):
    model.eval()
    total = 0
    correct = 0
    
    with torch.no_grad():
        processed = 0
        hook_handles = []
        
        if hooks:
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                loss_mask = batch["loss_mask"].to(device)
                
                logits = model.run_with_hooks(
                    input_ids[:, :-1],
                    fwd_hooks=[(hook_name, hook_fn) for hook_name, hook_fn in hooks]
                )
                targets = input_ids[:, 1:]
                mask = loss_mask[:, 1:]
                
                preds = logits.argmax(dim=-1)
                
                per_row = mask.sum(dim=1)
                valid_rows = (per_row == 1)
                
                if not valid_rows.any():
                    processed += 1
                    if processed >= max_batches:
                        break
                    continue
                
                mask_valid = mask[valid_rows].bool()
                preds_valid = preds[valid_rows]
                targets_valid = targets[valid_rows]
                
                pred_answers = preds_valid[mask_valid]
                true_answers = targets_valid[mask_valid]
                
                correct += (pred_answers == true_answers).sum().item()
                total += pred_answers.numel()
                
                processed += 1
                if processed >= max_batches:
                    break
        else:
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                loss_mask = batch["loss_mask"].to(device)
                
                logits = model(input_ids[:, :-1])
                targets = input_ids[:, 1:]
                mask = loss_mask[:, 1:]
                
                preds = logits.argmax(dim=-1)
                
                per_row = mask.sum(dim=1)
                valid_rows = (per_row == 1)
                
                if not valid_rows.any():
                    processed += 1
                    if processed >= max_batches:
                        break
                    continue
                
                mask_valid = mask[valid_rows].bool()
                preds_valid = preds[valid_rows]
                targets_valid = targets[valid_rows]
                
                pred_answers = preds_valid[mask_valid]
                true_answers = targets_valid[mask_valid]
                
                correct += (pred_answers == true_answers).sum().item()
                total += pred_answers.numel()
                
                processed += 1
                if processed >= max_batches:
                    break
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


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
    model = build_model(vocab_size, device, checkpoint_path=checkpoint_path)
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

    def find_latest_epoch(model_name: str, project_root: Path) -> int:
        pattern = str(project_root / f"checkpoint-{model_name}-epoch-*.pt")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No checkpoints found for {model_name}")
        epochs = [int(f.split("epoch-")[1].split(".pt")[0]) for f in files]
        return max(epochs)

    for name in model_names:
        latest_epoch = find_latest_epoch(name, args.project_root)
        checkpoint = args.project_root / f"checkpoint-{name}-epoch-{latest_epoch}.pt"
        tokenizer = args.project_root / f"train-{name}-tokenizer.pkl"
        dataset = args.project_root / f"test-{name}-dataset.pkl"
        print(f"Analyzing {name} (epoch {latest_epoch}):")
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

    print("\n" + "=" * 70)
    print("IDENTIFYING KEY HEADS FOR ALL MODELS")
    print("=" * 70)
    
    models_to_analyze = ["easy", "bpe-hard", "mult-hard", "length-hard", "all-hard", "mixed"]
    
    for model_name in models_to_analyze:
        print(f"\n{'=' * 70}")
        print(f"Analyzing {model_name.upper()}")
        print("=" * 70)
        
        try:
            latest_epoch = find_latest_epoch(model_name, args.project_root)
            checkpoint = args.project_root / f"checkpoint-{model_name}-epoch-{latest_epoch}.pt"
            tokenizer_path = args.project_root / f"train-{model_name}-tokenizer.pkl"
            dataset_path = args.project_root / f"test-{model_name}-dataset.pkl"
            
            tokenizer = load_pickle(tokenizer_path)
            vocab_size = tokenizer.vocab.size + 5
            model = build_model(vocab_size, device, checkpoint_path=checkpoint)
            checkpoint_data = torch.load(checkpoint, map_location=device)
            model.load_state_dict(checkpoint_data["model_state_dict"])
            
            dataset = load_pickle(dataset_path)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
            
            accuracy_drops, baseline_acc = identify_key_heads(model, dataloader, device, max_batches=args.max_batches)
            
            print(f"\nKEY HEADS IDENTIFIED (by accuracy drop when ablated):")
            
            n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
            key_heads = []
            for layer in range(n_layers):
                for head in range(n_heads):
                    drop = float(accuracy_drops[layer, head])
                    if drop > 0.01:
                        ablated_acc = baseline_acc - drop
                        key_heads.append((layer, head, drop, ablated_acc))
            
            key_heads.sort(key=lambda x: x[2], reverse=True)
            
            if key_heads:
                print(f"\nTop {min(5, len(key_heads))} key heads:")
                for layer, head, drop, ablated in key_heads[:5]:
                    print(f"  L{layer}H{head}: {drop:.4f} accuracy drop (baseline: {baseline_acc:.4f} -> ablated: {ablated:.4f})")
            else:
                print("\nNo heads with significant accuracy drop found.")
                print("This suggests the model uses distributed computation across heads.")
            
            key_heads_output = args.project_root / f"key_heads_{model_name}.json"
            with key_heads_output.open("w") as f:
                json.dump({
                    "model": model_name,
                    "epoch": latest_epoch,
                    "baseline_accuracy": float(baseline_acc),
                    "key_heads": [
                        {
                            "layer": l,
                            "head": h,
                            "accuracy_drop": float(d),
                            "ablated_accuracy": float(a)
                        }
                        for l, h, d, a in key_heads
                    ]
                }, f, indent=2)
            print(f"\nSaved key heads analysis to {key_heads_output}")
            
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
