import argparse
import json
import numpy as np
import jax
import jax.random as jr
import equinox as eqx

from models.pdssm import StateTrackingPDSSM
from data_dir.dataloaders import FLDataloader
from train import compute_accuracy


def evaluate(config_path, checkpoint_path, lengths=None, num_samples=2048, batch_size=64, seed=42):
    config = json.load(open(config_path))

    task         = config["task"]
    num_layers   = config["num_layers"]
    state_size   = config["state_size"]
    embed_size   = config["embed_size"]
    dictionary_size = config.get("dictionary_size", 6)

    # Import vocab_size from the task module
    import importlib
    module = importlib.import_module(f"data_dir.fl_tasks.{task}")
    vocab_size = module.vocab_size

    # Rebuild model architecture then load weights
    model = StateTrackingPDSSM(
        vocab_size=vocab_size,
        label_dim=vocab_size,
        N=state_size,
        H=embed_size,
        num_layers=num_layers,
        K=dictionary_size,
        key=jr.PRNGKey(0),  # structure only — weights overwritten below
    )
    model = eqx.tree_deserialise_leaves(checkpoint_path, model)
    print(f"Loaded model from {checkpoint_path}")

    if lengths is None:
        lengths = [10, 20, 40, 64, 128, 256, 512, 1024]

    print(f"\n{'Length':>8}  {'Accuracy':>10}  {'Correct':>10}")
    print("-" * 34)

    results = {}
    for length in lengths:
        loader = FLDataloader(
            task_name=task,
            min_length=length,
            max_length=length,
            num_val_samples=num_samples,
            val_seed=seed,
        )
        correct = 0
        total = 0
        for X, y, mask in loader.loop_epoch(batch_size):
            acc = compute_accuracy(model, X, y, mask)
            correct += float(acc) * X.shape[0]
            total += X.shape[0]
        acc = correct / total
        results[length] = acc
        print(f"{length:>8}  {acc:>10.4f}  {int(correct):>6}/{total}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True,
                        help="Config name in experiment_configs/ (without .json)")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .eqx checkpoint file")
    parser.add_argument("--lengths", nargs="+", type=int, default=None,
                        help="Sequence lengths to evaluate (default: 10 20 40 64 128 256 512 1024)")
    parser.add_argument("--num_samples", type=int, default=2048,
                        help="Number of test samples per length (default: 2048)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate(
        config_path=f"experiment_configs/{args.config}.json",
        checkpoint_path=args.checkpoint,
        lengths=args.lengths,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
