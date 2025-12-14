#!/usr/bin/env python3
"""
Signal Density (H1) visualization:

Line plot of Accuracy vs. target Multiplicity, with separate lines for
short vs. long sequence-length regimes.

This corresponds to the "Signal Density Enables Counting Circuits" hypothesis:
higher multiplicity -> denser supervision -> higher accuracy (especially in long sequences).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt


def get_latest_checkpoint(model_name: str, project_root: Path) -> Path:
    pattern = str(project_root / f"checkpoint-{model_name}-epoch-*.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoints found for {model_name} (pattern: {pattern})")
    epochs = [int(f.split("epoch-")[1].split(".pt")[0]) for f in files]
    latest = max(epochs)
    return project_root / f"checkpoint-{model_name}-epoch-{latest}.pt"


def read_baseline_accuracies_from_key_heads(project_root: Path) -> Dict[str, float]:
    """
    Fallback path that doesn't require TransformerLens.
    Uses baseline accuracies already stored in `key_heads_{model}.json`.
    """
    accs: Dict[str, float] = {}
    for model_name in ["easy", "mult-hard", "length-hard", "all-hard"]:
        path = project_root / f"key_heads_{model_name}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Either generate it via `python attention_analysis.py` "
                "or run this script with --source evaluate in an environment with transformer-lens installed."
            )
        data = json.loads(path.read_text())
        accs[model_name] = float(data["baseline_accuracy"])
    return accs


def evaluate_accuracies_from_checkpoints(project_root: Path) -> Dict[str, Dict[str, Any]]:
    """
    Full evaluation path (requires torch + transformer-lens via attention_analysis).
    Computes accuracy for each regime on its own test set.
    """
    import sys

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

    # Make classes available if any pickles reference __main__
    sys.modules["__main__"].CountingTokenizer = CountingTokenizer
    sys.modules["__main__"].Vocabulary = Vocabulary
    sys.modules["__main__"].CountingDataset = CountingDataset

    def full_accuracy(model, dataloader, device) -> float:
        # evaluate_accuracy stops after max_batches, so use the whole dataloader length
        return evaluate_accuracy(model, dataloader, device, max_batches=len(dataloader) + 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use a shared tokenizer for vocab size (all pickles were created with compatible vocab)
    tokenizer = load_pickle(project_root / "train-all-hard-tokenizer.pkl")
    vocab_size = tokenizer.vocab.size + 5

    regimes = {
        "easy": {"length": "short", "mult_range": (1, 2), "test_dataset": "test-easy-dataset.pkl"},
        "mult-hard": {"length": "short", "mult_range": (3, 10), "test_dataset": "test-mult-hard-dataset.pkl"},
        "length-hard": {"length": "long", "mult_range": (1, 2), "test_dataset": "test-length-hard-dataset.pkl"},
        "all-hard": {"length": "long", "mult_range": (3, 10), "test_dataset": "test-all-hard-dataset.pkl"},
    }

    results: Dict[str, Dict[str, Any]] = {}
    for model_name, spec in regimes.items():
        ckpt = get_latest_checkpoint(model_name, project_root)
        ds = load_pickle(project_root / spec["test_dataset"])
        dl = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

        model = build_model(vocab_size, device, checkpoint_path=ckpt)
        model.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
        model.eval()

        acc = full_accuracy(model, dl, device)
        results[model_name] = {
            "checkpoint": str(ckpt),
            "n_examples": len(ds),
            "accuracy": float(acc),
            "length_regime": spec["length"],
            "multiplicity_range": list(spec["mult_range"]),
            "multiplicity_midpoint": float(sum(spec["mult_range"]) / 2.0),
        }

        print(f"{model_name:>10} | mult={spec['mult_range']} | len={spec['length']:>5} | acc={acc*100:6.2f}%")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate signal-density line plot (Multiplicity vs Accuracy).")
    parser.add_argument("--project_root", type=Path, default=Path.cwd())
    parser.add_argument("--output_png", type=Path, default=Path("signal_density_accuracy.png"))
    parser.add_argument("--output_json", type=Path, default=Path("signal_density_accuracy.json"))
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--source",
        choices=["auto", "evaluate", "key_heads_json"],
        default="auto",
        help=(
            "Where to get accuracies. "
            "'evaluate' loads checkpoints + evaluates on test sets (requires transformer-lens). "
            "'key_heads_json' reads baseline accuracies from key_heads_*.json. "
            "'auto' tries evaluate then falls back to key_heads_json."
        ),
    )
    args = parser.parse_args()

    regimes = {
        "easy": {"length": "short", "mult_range": (1, 2)},
        "mult-hard": {"length": "short", "mult_range": (3, 10)},
        "length-hard": {"length": "long", "mult_range": (1, 2)},
        "all-hard": {"length": "long", "mult_range": (3, 10)},
    }

    results: Dict[str, Dict[str, Any]] = {}
    if args.source in ("auto", "evaluate"):
        try:
            results = evaluate_accuracies_from_checkpoints(args.project_root)
        except Exception as e:
            if args.source == "evaluate":
                raise
            print(f"[auto] evaluation path failed ({type(e).__name__}: {e}); falling back to key_heads_*.json")
            args.source = "key_heads_json"

    if args.source == "key_heads_json":
        accs = read_baseline_accuracies_from_key_heads(args.project_root)
        for model_name, spec in regimes.items():
            mult_range = spec["mult_range"]
            results[model_name] = {
                "checkpoint": None,
                "n_examples": None,
                "accuracy": float(accs[model_name]),
                "length_regime": spec["length"],
                "multiplicity_range": list(mult_range),
                "multiplicity_midpoint": float(sum(mult_range) / 2.0),
                "accuracy_source": "key_heads_json",
            }
        print("Loaded accuracies from key_heads_*.json:")
        for model_name in ["easy", "mult-hard", "length-hard", "all-hard"]:
            r = results[model_name]
            print(f"{model_name:>10} | mult={tuple(r['multiplicity_range'])} | len={r['length_regime']:>5} | acc={r['accuracy']*100:6.2f}%")

    # Prepare plot series: short vs long, accuracy on matching distributions
    def point(model_name: str):
        r = results[model_name]
        return r["multiplicity_midpoint"], r["accuracy"]

    x_short = [point("easy")[0], point("mult-hard")[0]]
    y_short = [point("easy")[1], point("mult-hard")[1]]

    x_long = [point("length-hard")[0], point("all-hard")[0]]
    y_long = [point("length-hard")[1], point("all-hard")[1]]

    # Plot
    plt.figure(figsize=(6.5, 4.2))
    plt.plot(x_short, [v * 100 for v in y_short], marker="o", linewidth=2, label="Short sequences (5–10)")
    plt.plot(x_long, [v * 100 for v in y_long], marker="o", linewidth=2, label="Long sequences (20–50)")

    # Use midpoint x-values but label with ranges (more faithful to your dataset setup)
    x_ticks = [1.5, 6.5]
    plt.xticks(x_ticks, ["1–2", "3–10"])

    plt.ylim(0, 102)
    plt.xlabel("Target multiplicity (range)")
    plt.ylabel("Accuracy (%)")
    plt.title("Signal Density Hypothesis (H1): Multiplicity vs Accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output_png, dpi=args.dpi)
    plt.close()

    payload = {
        "plot": {
            "x_axis": "multiplicity_midpoint (labeled by multiplicity_range)",
            "y_axis": "accuracy",
            "accuracy_source": args.source,
            "series": {
                "short": {"models": ["easy", "mult-hard"]},
                "long": {"models": ["length-hard", "all-hard"]},
            },
        },
        "results": results,
    }
    args.output_json.write_text(json.dumps(payload, indent=2))

    print(f"\nSaved plot: {args.output_png}")
    print(f"Saved data: {args.output_json}")


if __name__ == "__main__":
    main()


