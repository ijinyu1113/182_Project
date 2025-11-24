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

    print("Loading dataset...")
    dataset_path = project_root / "test-all-hard-dataset.pkl"
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
    print(f"Baseline Easy Model Accuracy on All-Hard: {acc_easy:.4f}")

    acc_hard = evaluate_accuracy(model_source, dataloader, device, max_batches=50)
    print(f"Baseline All-Hard Model Accuracy on All-Hard: {acc_hard:.4f}")

    print("\nRunning Activation Patching...")
    
    LAYER = 1
    HEADS = [4, 6]
    HOOK_NAME = f"blocks.{LAYER}.attn.hook_z"
    
    print(f"Patching {HOOK_NAME} for heads {HEADS}")

    total = 0
    correct = 0
    processed = 0
    max_batches = 50

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            
            _, cache_source = model_source.run_with_cache(input_ids[:, :-1])
            source_acts = cache_source[HOOK_NAME]

            def patch_hook(activations, hook):
                for h in HEADS:
                    activations[:, :, h, :] = source_acts[:, :, h, :]
                return activations

            logits = model_target.run_with_hooks(
                input_ids[:, :-1],
                fwd_hooks=[(HOOK_NAME, patch_hook)]
            )

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
    print(f"Patched Model Accuracy: {acc_patched:.4f}")
    
    print("\nSummary:")
    print(f"Easy (Baseline): {acc_easy:.4f}")
    print(f"All-Hard (Source): {acc_hard:.4f}")
    print(f"Easy + Patched L1H{HEADS}: {acc_patched:.4f}")

if __name__ == "__main__":
    main()
