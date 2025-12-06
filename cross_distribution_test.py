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
    if not files:
        raise FileNotFoundError(f"No checkpoints found for {model_name}")
    epochs = [int(f.split("epoch-")[1].split(".pt")[0]) for f in files]
    latest = max(epochs)
    return project_root / f"checkpoint-{model_name}-epoch-{latest}.pt"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = Path.cwd()
    
    # Create output file
    output_file = project_root / "cross_distribution_test_results.txt"
    output_lines = []
    
    def log(message):
        """Print to console and add to output lines"""
        print(message)
        output_lines.append(message)
    
    log("="*70)
    log("TABLE 5: Cross-Distribution Generalization")
    log("="*70)
    
    # Load tokenizer
    tokenizer = load_pickle(project_root / "train-all-hard-tokenizer.pkl")
    vocab_size = tokenizer.vocab.size + 5
    
    # Load test datasets
    log("\nLoading test datasets...")
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
        log(f"  {name}: {len(ds)} examples")
    
    # Load models
    log("\nLoading models...")
    models = {}
    
    for model_name in ["easy", "length-hard", "all-hard"]:
        ckpt = get_latest_checkpoint(model_name, project_root)
        model = build_model(vocab_size, device, checkpoint_path=ckpt)
        model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
        model.eval()
        models[model_name] = model
        log(f"  ✓ Loaded {model_name}")
    
    # Test each model on each dataset
    log("\n" + "="*70)
    log("RESULTS")
    log("="*70)
    
    results = {}
    
    for model_name, model in models.items():
        results[model_name] = {}
        log(f"\n{model_name} model:")
        
        for test_name, dataloader in dataloaders.items():
            acc = evaluate_accuracy(model, dataloader, device, max_batches=100)
            results[model_name][test_name] = acc
            log(f"  on {test_name}: {acc*100:.1f}%")
    
    # Print Table 5 format
    log("\n" + "="*70)
    log("TABLE 5: Distribution Specificity")
    log("="*70)
    log(f"{'Model':<15} {'Training Dist.':<15} {'Own Test':>10} {'Length-Hard':>12} {'Mult-Hard':>10}")
    log("-"*70)
    
    # Easy
    log(f"{'Easy':<15} {'(5-10, 1-2)':<15} {results['easy']['easy']*100:>10.1f} {results['easy']['length-hard']*100:>12.1f} {results['easy']['mult-hard']*100:>10.1f}")
    
    # Length-hard
    log(f"{'Length-hard':<15} {'(20-50, 1-2)':<15} {results['length-hard']['length-hard']*100:>10.1f} {results['length-hard']['length-hard']*100:>12.1f} {results['length-hard']['mult-hard']*100:>10.1f}")
    
    # All-hard
    log(f"{'All-hard':<15} {'(20-50, 3-10)':<15} {results['all-hard']['easy']*100:>10.1f} {results['all-hard']['length-hard']*100:>12.1f} {results['all-hard']['mult-hard']*100:>10.1f}")
    
    log("="*70)
    
    # Write results to file
    with open(output_file, "w") as f:
        f.write("\n".join(output_lines))
    
    print(f"\n✓ Results saved to {output_file}")

if __name__ == "__main__":
    main()
