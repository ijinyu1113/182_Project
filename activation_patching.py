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
import argparse
from pathlib import Path
import glob

def get_latest_checkpoint(model_name, project_root):
    pattern = str(project_root / f"checkpoint-{model_name}-epoch-*.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoints found for {model_name}")
    epochs = [int(f.split("epoch-")[1].split(".pt")[0]) for f in files]
    latest = max(epochs)
    return project_root / f"checkpoint-{model_name}-epoch-{latest}.pt"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path.cwd()

    print(f"Using device: {device}")

    # FIX 1: Use Length-Hard test set, not All-Hard
    print("Loading Length-Hard test dataset...")
    dataset_path = project_root / "test-length-hard-dataset.pkl"
    dataset = load_pickle(dataset_path)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    tokenizer_path = project_root / "train-all-hard-tokenizer.pkl"
    tokenizer = load_pickle(tokenizer_path)
    vocab_size = tokenizer.vocab.size + 5

    print("Loading models...")
    
    ckpt_hard = get_latest_checkpoint("all-hard", project_root)
    print(f"Loading Source (All-Hard): {ckpt_hard.name}")
    model_source = build_model(vocab_size, device, checkpoint_path=ckpt_hard)
    state_dict_hard = torch.load(ckpt_hard, map_location=device)["model_state_dict"]
    model_source.load_state_dict(state_dict_hard)
    model_source.eval()

    ckpt_easy = get_latest_checkpoint("easy", project_root)
    print(f"Loading Target (Easy): {ckpt_easy.name}")
    model_target = build_model(vocab_size, device, checkpoint_path=ckpt_easy)
    state_dict_easy = torch.load(ckpt_easy, map_location=device)["model_state_dict"]
    model_target.load_state_dict(state_dict_easy)
    model_target.eval()

    print("\nComputing Baselines...")
    acc_easy = evaluate_accuracy(model_target, dataloader, device, max_batches=50)
    print(f"Baseline Easy Model Accuracy on Length-Hard: {acc_easy:.4f}")

    acc_hard = evaluate_accuracy(model_source, dataloader, device, max_batches=50)
    print(f"Baseline All-Hard Model Accuracy on Length-Hard: {acc_hard:.4f}")

    # FIX 2: Test different head combinations
    test_configs = [
        ("L1H6 only", 1, [6]),
        ("L1H4 only", 1, [4]),
        ("L1H6 + L1H4", 1, [4, 6]),
    ]
    
    print("\nRunning Activation Patching Experiments...")
    
    for config_name, layer, heads in test_configs:
        print(f"\n--- Testing: {config_name} ---")
        HOOK_NAME = f"blocks.{layer}.attn.hook_z"
        
        total = 0
        correct = 0
        processed = 0
        max_batches = 50

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                loss_mask = batch["loss_mask"].to(device)
                
                # Run source model and cache activations
                _, cache_source = model_source.run_with_cache(input_ids[:, :-1])
                source_acts = cache_source[HOOK_NAME]

                # FIX 3: Add dimension check
                def patch_hook(activations, hook):
                    batch_size = min(activations.shape[0], source_acts.shape[0])
                    for h in heads:
                        activations[:batch_size, :, h, :] = source_acts[:batch_size, :, h, :]
                    return activations

                # Run target model with patched activations
                logits = model_target.run_with_hooks(
                    input_ids[:, :-1],
                    fwd_hooks=[(HOOK_NAME, patch_hook)]
                )

                # Evaluate
                targets = input_ids[:, 1:]
                mask = loss_mask[:, 1:]
                preds = logits.argmax(dim=-1)
                
                per_row = mask.sum(dim=1)
                valid_rows = (per_row == 1)
                
                if not valid_rows.any():
                    processed += 1
                    if processed >= max_batches: break
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

        acc_patched = correct / total if total > 0 else 0
        print(f"{config_name} Accuracy: {acc_patched:.4f} (Δ = {acc_patched - acc_easy:+.4f})")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Easy Baseline:      {acc_easy:.4f}")
    print(f"All-Hard Baseline:  {acc_hard:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
