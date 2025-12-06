#!/usr/bin/env python3
"""
Visualize attention patterns for all-hard model key heads.
Confirms L1H6 detects characters, L1H4 outputs results.
"""
import torch
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from transformer_lens import HookedTransformer, HookedTransformerConfig

# ============================================================================
# Load model and tokenizer
# ============================================================================
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

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def build_model(vocab_size, checkpoint_path, device):
    """Build model with config inferred from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    
    # Infer config from checkpoint
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
        device=str(device),
    )
    
    model = HookedTransformer(config).to(device)
    model.load_state_dict(state_dict, strict=False)
    return model

# ============================================================================
# Visualization functions
# ============================================================================

def visualize_attention_pattern(attn_pattern, tokens, tokenizer, title, save_path=None):
    """
    Visualize attention pattern as heatmap.
    
    Args:
        attn_pattern: [seq_len, seq_len] attention weights
        tokens: [seq_len] token IDs
        tokenizer: tokenizer to decode tokens
        title: plot title
        save_path: optional path to save figure
    """
    # Decode tokens to strings
    token_strs = []
    for tok_id in tokens:
        try:
            tok_id_int = tok_id.item() if torch.is_tensor(tok_id) else tok_id
            token_str = tokenizer.decode([tok_id_int])

            # Clean up for display
            token_str = token_str.replace("<space>", "_")
            token_str = token_str.replace(" ", "_")
            if len(token_str) > 8:
                token_str = token_str[:8]
            token_strs.append(token_str)
        except:
            token_strs.append(f"[{tok_id}]")
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(
        attn_pattern.cpu().numpy(),
        xticklabels=token_strs,
        yticklabels=token_strs,
        cmap='viridis',
        cbar_kws={'label': 'Attention Weight'},
        square=True,
        vmin=0,
        vmax=attn_pattern.max().item()
    )
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel("Key Position", fontsize=12)
    plt.ylabel("Query Position", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def visualize_target_attention(attn_pattern, tokens, tokenizer, target_letter_id, title, save_path=None):
    """
    Visualize attention to target vs non-target positions.
    
    Shows bar plot of average attention to target positions for each query.
    """
    seq_len = len(tokens)
    
    # Identify target positions (after position 11)
    target_mask = (tokens[11:] == target_letter_id).cpu().numpy()
    target_positions = np.where(target_mask)[0] + 11
    
    # Decode tokens - FIX: Convert tensors to ints
    token_strs = []
    for i, tok_id in enumerate(tokens):
        tok_id_int = tok_id.item() if torch.is_tensor(tok_id) else tok_id  # Convert to int
        token_str = tokenizer.decode([tok_id_int])
        token_str = token_str.replace("<space>", "_").replace(" ", "_")
        if i in target_positions:
            token_str = f"*{token_str}*"  # Mark targets
        if len(token_str) > 10:
            token_str = token_str[:10]
        token_strs.append(token_str)
    
    # Rest stays the same...
    # Compute average attention to target vs non-target
    attn_np = attn_pattern.cpu().numpy()
    target_attn = np.zeros(seq_len)
    non_target_attn = np.zeros(seq_len)
    
    for query_pos in range(11, seq_len):  # Only string positions
        if len(target_positions) > 0:
            target_attn[query_pos] = attn_np[query_pos, target_positions].mean()
        
        non_target_pos = [i for i in range(11, seq_len) if i not in target_positions]
        if len(non_target_pos) > 0:
            non_target_attn[query_pos] = attn_np[query_pos, non_target_pos].mean()
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Bar plot
    x = np.arange(11, seq_len)
    width = 0.35
    ax1.bar(x - width/2, target_attn[11:], width, label='Attn to Targets', color='red', alpha=0.7)
    ax1.bar(x + width/2, non_target_attn[11:], width, label='Attn to Non-Targets', color='blue', alpha=0.7)
    ax1.set_xlabel("Query Position")
    ax1.set_ylabel("Average Attention")
    ax1.set_title(f"{title}\n(Target positions marked with *)")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Difference plot
    diff = target_attn[11:] - non_target_attn[11:]
    colors = ['green' if d > 0 else 'orange' for d in diff]
    ax2.bar(x, diff, color=colors, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel("Query Position")
    ax2.set_ylabel("Target Attn - Non-Target Attn")
    ax2.set_title("Differential Attention (positive = prefers targets)")
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    # Print statistics
    print(f"\nTarget positions: {target_positions}")
    print(f"Average attention to targets: {target_attn[11:].mean():.4f}")
    print(f"Average attention to non-targets: {non_target_attn[11:].mean():.4f}")
    print(f"Differential: {(target_attn[11:] - non_target_attn[11:]).mean():.4f}")




# ============================================================================
# Main analysis
# ============================================================================

def analyze_examples(model, tokenizer, dataset, device, n_examples=5):
    """Analyze and visualize n_examples from the dataset"""
    
    model.eval()
    
    for idx in range(min(n_examples, len(dataset))):
        example = dataset[idx]
        tokens = torch.tensor(example['tokens'], device=device).unsqueeze(0)  # [1, seq_len]
        answer = example['answer']
        
        print("\n" + "="*80)
        print(f"EXAMPLE {idx + 1}")
        print("="*80)
        
        # Decode and print
        decoded = tokenizer.decode(example['tokens'])
        print(f"Input: {decoded}")
        print(f"True answer: {answer}")
        
        # Get target letter (position 6)
        target_letter_id = tokens[0, 6].item()
        target_letter = tokenizer.decode([target_letter_id])
        print(f"Target letter: {target_letter}")
        
        # Run model (IMPORTANT: model sees tokens[:-1])
        model_input = tokens[:, :-1]  # Remove last token for prediction
        logits, cache = model.run_with_cache(model_input)
        
        pred = logits[0, -1].argmax().item()
        pred_token = tokenizer.decode([pred])
        answer_token = tokenizer.decode([answer])
        print(f"Predicted: {pred_token}")
        print(f"True answer token: {answer_token}")
        print(f"Correct: {pred_token == answer_token}")
        
        # FIXED: Find where 'f' appears in MODEL INPUT (not full sequence)
        seq_len = model_input.shape[1]  # Actual sequence length model sees
        f_positions = []
        for i in range(11, seq_len):  # String starts at position 11
            tok = model_input[0, i].item()
            if tok == target_letter_id:
                f_positions.append(i)
        
        print(f"Target '{target_letter}' appears at positions: {f_positions}")
        print(f"Attention pattern shape: {cache['blocks.1.attn.hook_pattern'].shape}")
        
        # FIXED: Use model_input length, not full tokens
        if len(f_positions) > 0:
            attn_l1h6 = cache["blocks.1.attn.hook_pattern"][0, 6]  # [seq_len, seq_len]
            
            # Build list of non-target positions
            non_target_positions = [i for i in range(11, seq_len) if i not in f_positions]
            
            if len(non_target_positions) > 0:
                avg_attn_to_targets = attn_l1h6[:, f_positions].mean().item()
                avg_attn_to_others = attn_l1h6[:, non_target_positions].mean().item()
                
                print(f"L1H6 average attention to '{target_letter}' positions: {avg_attn_to_targets:.4f}")
                print(f"L1H6 average attention to other positions: {avg_attn_to_others:.4f}")
                if avg_attn_to_others > 0:
                    print(f"Ratio (should be >1 for detector): {avg_attn_to_targets/avg_attn_to_others:.2f}")
        
        # ================================================================
        # Visualize L1H6 (Character Detector)
        # ================================================================
        
        attn_l1h6 = cache["blocks.1.attn.hook_pattern"][0, 6]  # [seq, seq]
        
        print("\n--- L1H6 (Character Detector) ---")
        visualize_attention_pattern(
            attn_l1h6,
            model_input[0],  # Use model input, not full tokens!
            tokenizer,
            f"Example {idx+1}: L1H6 Attention Pattern",
            save_path=f"l1h6_pattern_ex{idx+1}.png"
        )
        
        visualize_target_attention(
            attn_l1h6,
            model_input[0],  # Use model input!
            tokenizer,
            target_letter_id,
            f"Example {idx+1}: L1H6 Attention to Targets",
            save_path=f"l1h6_targets_ex{idx+1}.png"
        )
        
        # ================================================================
        # Visualize L1H4 (Output Head)
        # ================================================================
        
        # ================================================================
        # Visualize L1H4 (Output Head)
        # ================================================================

        attn_l1h4 = cache["blocks.1.attn.hook_pattern"][0, 4]  # [seq, seq]

        print("\n--- L1H4 (Output Head) ---")
        visualize_attention_pattern(
            attn_l1h4,
            model_input[0],
            tokenizer,
            f"Example {idx+1}: L1H4 Attention Pattern",
            save_path=f"l1h4_pattern_ex{idx+1}.png"
        )

        # ADD THIS: Target attention for L1H4
        visualize_target_attention(
            attn_l1h4,
            model_input[0],
            tokenizer,
            target_letter_id,
            f"Example {idx+1}: L1H4 Attention to Targets",
            save_path=f"l1h4_targets_ex{idx+1}.png"
        )

        # ALSO ADD: Compute ratio for comparison
        if len(f_positions) > 0 and len(non_target_positions) > 0:
            avg_attn_to_targets_h4 = attn_l1h4[:, f_positions].mean().item()
            avg_attn_to_others_h4 = attn_l1h4[:, non_target_positions].mean().item()
            
            print(f"L1H4 average attention to '{target_letter}' positions: {avg_attn_to_targets_h4:.4f}")
            print(f"L1H4 average attention to other positions: {avg_attn_to_others_h4:.4f}")
            if avg_attn_to_others_h4 > 0:
                print(f"L1H4 Ratio: {avg_attn_to_targets_h4/avg_attn_to_others_h4:.2f}")
            print(f"\nCOMPARISON:")
            print(f"  L1H6 ratio: {avg_attn_to_targets/avg_attn_to_others:.2f} (character detector)")
            print(f"  L1H4 ratio: {avg_attn_to_targets_h4/avg_attn_to_others_h4:.2f} (output head)")
                
        


# ============================================================================
# Run analysis
# ============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Paths
    project_root = Path(".")
    checkpoint_path = project_root / "checkpoint-all-hard-epoch-40.pt"
    tokenizer_path = project_root / "train-all-hard-tokenizer.pkl"
    dataset_path = project_root / "test-all-hard-dataset.pkl"
    
    # Load
    print("Loading model and data...")
    tokenizer = load_pickle(tokenizer_path)
    dataset = load_pickle(dataset_path)
    vocab_size = tokenizer.vocab.size + 5
    
    model = build_model(vocab_size, checkpoint_path, device)
    
    print(f"Loaded model with {model.cfg.n_layers} layers, {model.cfg.n_heads} heads")
    print(f"Dataset size: {len(dataset)} examples")
    
    # Analyze first 5 examples
    analyze_examples(model, tokenizer, dataset, device, n_examples=5)
    
    print("\n✓ Visualization complete!")