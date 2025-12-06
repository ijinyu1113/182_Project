#!/usr/bin/env python3
"""
Complete diagnostic script for activation patching issues.
Checks tokenizers, models, datasets, and patching.
"""

import torch
from torch.utils.data import DataLoader
from attention_analysis import (
    build_model,
    load_pickle,
    collate_fn,
    evaluate_accuracy,
    CountingDataset,
    Vocabulary,
    CountingTokenizer
)
from pathlib import Path
import glob

#!/usr/bin/env python3
"""
Quick test: Patch All-Hard circuits into Easy, test on All-Hard data
"""

def get_latest_checkpoint(model_name, project_root):
    pattern = str(project_root / f"checkpoint-{model_name}-epoch-*.pt")
    files = glob.glob(pattern)
    epochs = [int(f.split("epoch-")[1].split(".pt")[0]) for f in files]
    latest = max(epochs)
    return project_root / f"checkpoint-{model_name}-epoch-{latest}.pt"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path.cwd()
    
    print("="*80)
    print("REVERSE PATCHING: Mult-Hard → All-Hard on Mult-Hard Test")
    print("  Can short-sequence circuits help long-sequence model?")
    print("="*80)
    
    # Load tokenizer
    tokenizer = load_pickle(project_root / "train-all-hard-tokenizer.pkl")
    vocab_size = tokenizer.vocab.size + 5
    
    # Load Mult-Hard test dataset
    print("\nLoading Mult-Hard test dataset...")
    dataset = load_pickle(project_root / "test-mult-hard-dataset.pkl")
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    print(f"  {len(dataset)} examples (5-10 chars, 3-10 mult)")
    
    # Load models
    print("\nLoading models...")
    
    # DONOR: Mult-Hard (knows how to count on short sequences)
    ckpt_mult = get_latest_checkpoint("mult-hard", project_root)
    model_source = build_model(vocab_size, device, checkpoint_path=ckpt_mult)
    model_source.load_state_dict(torch.load(ckpt_mult, map_location=device)["model_state_dict"])
    model_source.eval()
    print(f"  ✓ Loaded Mult-Hard (donor - short sequence expert)")
    
    # RECEIVER: All-Hard (fails on short sequences)
    ckpt_hard = get_latest_checkpoint("all-hard", project_root)
    model_target = build_model(vocab_size, device, checkpoint_path=ckpt_hard)
    model_target.load_state_dict(torch.load(ckpt_hard, map_location=device)["model_state_dict"])
    model_target.eval()
    print(f"  ✓ Loaded All-Hard (receiver - long sequence expert)")
    
    # Baselines
    print("\n" + "="*80)
    print("BASELINES")
    print("="*80)
    
    acc_mult = evaluate_accuracy(model_source, dataloader, device, max_batches=100)
    print(f"Mult-Hard on Mult-Hard (donor):     {acc_mult:.4f} ({acc_mult*100:.1f}%)")
    
    acc_hard = evaluate_accuracy(model_target, dataloader, device, max_batches=100)
    print(f"All-Hard on Mult-Hard (receiver):   {acc_hard:.4f} ({acc_hard*100:.1f}%)")
    print(f"  ← All-Hard FAILS on short sequences despite high multiplicity")
    
    if acc_hard > 0.50:
        print("\n⚠ All-Hard is doing better than expected - less room for improvement")
    else:
        print(f"\n✓ Large gap ({acc_mult - acc_hard:.1%}) - room for patching to help!")
    
    # Patching experiments
    print("\n" + "="*80)
    print("REVERSE PATCHING EXPERIMENTS")
    print("="*80)
    
    configs = [
        ("Layer 0 only (all heads)", 0, list(range(8))),
        ("L0H4 only (preprocessing)", 0, [4]),
        ("L1H6 only (detector)", 1, [6]),
        ("L1H4 only (output)", 1, [4]),
        ("L1H6 + L1H4", 1, [4, 6]),
        ("All critical heads (L0H4+L1H6+L1H4)", None, None),
        ("Layer 1 only (all heads)", 1, list(range(8))),
    ]
    
    results = {}
    
    for name, layer, heads in configs:
        print(f"\n--- {name} ---")
        
        if name == "All critical heads (L0H4+L1H6+L1H4)":
            # Patch multiple layers
            acc = run_multi_layer_patching(
                model_source, model_target, dataloader, device,
                [(0, [4]), (1, [4, 6])], max_batches=100
            )
        else:
            # Single layer
            acc = run_single_layer_patching(
                model_source, model_target, dataloader, device,
                layer, heads, max_batches=100
            )
        
        improvement = acc - acc_hard
        results[name] = (acc, improvement)
        
        print(f"Accuracy: {acc:.4f} ({acc*100:.1f}%)")
        print(f"Improvement: {improvement:+.4f} ({improvement*100:+.1f}%)")
        
        if improvement > 0.30:
            print("  ✅ HUGE improvement - circuits transfer!")
        elif improvement > 0.15:
            print("  ✅ STRONG improvement!")
        elif improvement > 0.05:
            print("  ~ Moderate improvement")
        elif improvement > 0.01:
            print("  ~ Slight improvement")
        else:
            print("  ❌ No improvement")
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    best_name = max(results.items(), key=lambda x: x[1][0])[0]
    best_acc, best_imp = results[best_name]
    
    print(f"Best patching: {best_name}")
    print(f"  Accuracy: {best_acc:.1%}")
    print(f"  Improvement: {best_imp:+.1%}")
    
    if best_imp > 0.50:
        print("\n✅ REVERSE PATCHING WORKS!")
        print("Interpretation:")
        print("  - Mult-Hard's circuits CAN help All-Hard on short sequences")
        print("  - All-Hard's failure is due to LATE-LAYER circuits")
        print("  - Early layers (position embeddings, L0) are compatible")
        print("  - The core counting circuits are TRANSFERABLE!")
    elif best_imp > 0.15:
        print("\n~ PARTIAL SUCCESS")
        print("Interpretation:")
        print("  - Some circuit components transfer")
        print("  - But distribution mismatch still causes issues")
    else:
        print("\n❌ REVERSE PATCHING FAILS TOO")
        print("Interpretation:")
        print("  - Even Mult-Hard's working circuits can't help All-Hard")
        print("  - Distribution specificity is PERVASIVE (all layers)")
        print("  - Position embeddings and early features are incompatible")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Mult-Hard (donor) baseline:  {acc_mult:.1%}")
    print(f"All-Hard (receiver) baseline: {acc_hard:.1%}")
    print(f"Best patched result:          {best_acc:.1%}")
    print(f"Maximum improvement:          {best_imp:+.1%}")
    print("="*80)

def run_single_layer_patching(model_source, model_target, dataloader, device, 
                               layer, heads, max_batches=100):
    """Run patching for a single layer"""
    hook_name = f"blocks.{layer}.attn.hook_z"
    
    total = 0
    correct = 0
    processed = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            
            # Cache source activations
            _, cache_source = model_source.run_with_cache(input_ids[:, :-1])
            source_acts = cache_source[hook_name]
            
            # Define patching hook
            def patch_hook(activations, hook):
                batch_size = min(activations.shape[0], source_acts.shape[0])
                for h in heads:
                    activations[:batch_size, :, h, :] = source_acts[:batch_size, :, h, :]
                return activations
            
            # Run with patching
            logits = model_target.run_with_hooks(
                input_ids[:, :-1],
                fwd_hooks=[(hook_name, patch_hook)]
            )
            
            # Evaluate
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

def run_multi_layer_patching(model_source, model_target, dataloader, device,
                              layer_heads_list, max_batches=100):
    """Run patching for multiple layers"""
    total = 0
    correct = 0
    processed = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            
            # Cache source activations for all layers
            _, cache_source = model_source.run_with_cache(input_ids[:, :-1])
            
            # Build hooks for all layers
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
            
            # Run with patching
            logits = model_target.run_with_hooks(
                input_ids[:, :-1],
                fwd_hooks=hooks
            )
            
            # Evaluate
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

if __name__ == "__main__":
    main()
