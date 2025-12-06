"""
Cross-distribution generalization test (Table 5)
Tests each model on multiple test distributions
"""

import torch
from torch.utils.data import DataLoader
from attention_analysis import (
    build_model,
    load_pickle,
    collate_fn,
    evaluate_accuracy,
)
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
    
    print("="*70)
    print("TABLE 5: Cross-Distribution Generalization")
    print("="*70)
    
    # Load tokenizer
    tokenizer = load_pickle(project_root / "train-all-hard-tokenizer.pkl")
    vocab_size = tokenizer.vocab.size + 5
    
    # Load test datasets
    print("\nLoading test datasets...")
    datasets = {
        "easy": load_pickle(project_root / "test-easy-dataset.pkl"),
        "length-hard": load_pickle(project_root / "test-length-hard-dataset.pkl"),
        "mult-hard": load_pickle(project_root / "test-mult-hard-dataset.pkl"),
    }
    
    dataloaders = {
        name: DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
        for name, ds in datasets.items()
    }
    
    for name, ds in datasets.items():
        print(f"  {name}: {len(ds)} examples")
    
    # Load models
    print("\nLoading models...")
    models = {}
    
    for model_name in ["easy", "length-hard", "all-hard"]:
        ckpt = get_latest_checkpoint(model_name, project_root)
        model = build_model(vocab_size, device, checkpoint_path=ckpt)
        model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
        model.eval()
        models[model_name] = model
        print(f"  ✓ Loaded {model_name}")
    
    # Test each model on each dataset
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    results = {}
    
    for model_name, model in models.items():
        results[model_name] = {}
        print(f"\n{model_name} model:")
        
        for test_name, dataloader in dataloaders.items():
            acc = evaluate_accuracy(model, dataloader, device, max_batches=100)
            results[model_name][test_name] = acc
            print(f"  on {test_name}: {acc*100:.1f}%")
    
    # Print Table 5 format
    print("\n" + "="*70)
    print("TABLE 5: Distribution Specificity")
    print("="*70)
    print(f"{'Model':<15} {'Training Dist.':<15} {'Own Test':>10} {'Length-Hard':>12} {'Mult-Hard':>10}")
    print("-"*70)
    
    # Easy
    print(f"{'Easy':<15} {'(5-10, 1-2)':<15} {results['easy']['easy']*100:>10.1f} {results['easy']['length-hard']*100:>12.1f} {results['easy']['mult-hard']*100:>10.1f}")
    
    # Length-hard
    print(f"{'Length-hard':<15} {'(20-50, 1-2)':<15} {results['length-hard']['length-hard']*100:>10.1f} {results['length-hard']['length-hard']*100:>12.1f} {'–':>10}")
    
    # All-hard
    print(f"{'All-hard':<15} {'(20-50, 3-10)':<15} {results['all-hard']['easy']*100:>10.1f} {results['all-hard']['length-hard']*100:>12.1f} {results['all-hard']['mult-hard']*100:>10.1f}")
    
    print("="*70)

if __name__ == "__main__":
    main()
