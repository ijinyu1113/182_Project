import torch
from torch.utils.data import DataLoader
from attention_analysis import (
    build_model,
    load_pickle,
    collate_fn,
    evaluate_accuracy,
    CountingTokenizer,
    Vocabulary,
    CountingDataset,
)
from pathlib import Path
import glob
import sys

# Make classes available to pickle if it's looking in __main__
sys.modules['__main__'].CountingTokenizer = CountingTokenizer
sys.modules['__main__'].Vocabulary = Vocabulary
sys.modules['__main__'].CountingDataset = CountingDataset

def get_latest_checkpoint(model_name, project_root):
    pattern = str(project_root / f"checkpoint-{model_name}-epoch-*.pt")
    files = glob.glob(pattern)
    epochs = [int(f.split("epoch-")[1].split(".pt")[0]) for f in files]
    latest = max(epochs)
    return project_root / f"checkpoint-{model_name}-epoch-{latest}.pt"

def run_patching(model_source, model_target, dataloader, device, layer_heads_list, max_batches=50):
    """Run patching for specified layer-head combinations"""
    total = 0
    correct = 0
    processed = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            
            _, cache_source = model_source.run_with_cache(input_ids[:, :-1])
            
            hooks = []
            for layer, heads in layer_heads_list:
                hook_name = f"blocks.{layer}.attn.hook_z"
                source_acts = cache_source[hook_name]
                
                def make_patch_hook(src_acts, head_list):
                    def patch_hook(activations, hook):
                        batch_size = min(activations.shape[0], src_acts.shape[0])
                        for h in head_list:
                            activations[:batch_size, :, h, :] = src_acts[:batch_size, :, h, :]
                        return activations
                    return patch_hook
                
                hooks.append((hook_name, make_patch_hook(source_acts, heads)))
            
            logits = model_target.run_with_hooks(
                input_ids[:, :-1],
                fwd_hooks=hooks
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
    
    return correct / total if total > 0 else 0.0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path.cwd()
    
    # Create output file
    output_file = project_root / "activation_patching_results.txt"
    output_lines = []
    
    def log(message):
        """Print to console and add to output lines"""
        print(message)
        output_lines.append(message)
    
    log("="*70)
    log("ACTIVATION PATCHING: All-hard → Easy on Mult-hard")
    log("="*70)
    
    # Load tokenizer
    tokenizer = load_pickle(project_root / "train-all-hard-tokenizer.pkl")
    vocab_size = tokenizer.vocab.size + 5
    
    # Load Mult-hard test data
    log("\nLoading Mult-hard test dataset...")
    dataset = load_pickle(project_root / "test-mult-hard-dataset.pkl")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    log(f"  {len(dataset)} examples")
    
    # Load models
    log("\nLoading models...")
    
    # DONOR: All-hard
    ckpt_hard = get_latest_checkpoint("all-hard", project_root)
    model_source = build_model(vocab_size, device, checkpoint_path=ckpt_hard)
    model_source.load_state_dict(torch.load(ckpt_hard, map_location=device)["model_state_dict"])
    model_source.eval()
    log(f"  ✓ Donor: All-hard")
    
    # RECEIVER: Easy
    ckpt_easy = get_latest_checkpoint("easy", project_root)
    model_target = build_model(vocab_size, device, checkpoint_path=ckpt_easy)
    model_target.load_state_dict(torch.load(ckpt_easy, map_location=device)["model_state_dict"])
    model_target.eval()
    log(f"  ✓ Receiver: Easy")
    
    # Baselines
    log("\n" + "="*70)
    log("BASELINES")
    log("="*70)
    
    acc_easy = evaluate_accuracy(model_target, dataloader, device, max_batches=50)
    log(f"Easy on Mult-hard:     {acc_easy:.1%}")
    
    acc_hard = evaluate_accuracy(model_source, dataloader, device, max_batches=50)
    log(f"All-hard on Mult-hard: {acc_hard:.1%}")
    
    # Patching experiments (5 conditions from paper)
    log("\n" + "="*70)
    log("PATCHING: All-hard → Easy")
    log("="*70)
    
    configs = [
        ("L1H6 only (detector)", [(1, [6])]),
        ("L1H4 only (output)", [(1, [4])]),
        ("L1H6 + L1H4", [(1, [4, 6])]),
        ("L0H4 (control)", [(0, [4])]),
        ("All three (L0H4+L1H6+L1H4)", [(0, [4]), (1, [4, 6])]),
    ]
    
    results = []
    for name, layer_heads in configs:
        acc = run_patching(model_source, model_target, dataloader, device, layer_heads)
        improvement = acc - acc_easy
        results.append((name, acc, improvement))
        log(f"{name}: {acc:.1%} (Δ = {improvement:+.1%})")
    
    # Summary table
    log("\n" + "="*70)
    log("SUMMARY (Table 4)")
    log("="*70)
    log(f"{'Condition':<35} {'Accuracy':>10} {'Improvement':>12}")
    log("-"*60)
    log(f"{'Easy on Mult-hard (baseline)':<35} {acc_easy:>10.1%} {'–':>12}")
    log(f"{'All-hard on Mult-hard':<35} {acc_hard:>10.1%} {'–':>12}")
    log("-"*60)
    for name, acc, imp in results:
        log(f"{name:<35} {acc:>10.1%} {imp:>+12.1%}")
    log("="*70)
    
    # Write results to file
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))
    
    print(f"\n✓ Results saved to {output_file}")

if __name__ == "__main__":
    main()
