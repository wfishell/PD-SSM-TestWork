"""Train a PD-SSM or Transformer on Dyck-k membership classification.

Matches the hyperparameters of the existing state-tracking experiments
(parity_0.json): batch 512, 100k steps, AdamW lr=2e-3 wd=0.01, warmup
10% + cosine decay, early stop at 99.95% val accuracy. Uses on-the-fly
data generation via Dyck_K_experiments.data.DyckGenerator.

Input format (matches state_tracking_JAX harness):
  X:    (B, L) int32   — token 0 is padding, tokens 1..2k are brackets
  y:    (B,)  int32    — 1=invalid, 2=valid (label 0 reserved for pad)
  mask: (B, L) bool    — True at the last real token of each sequence;
                         loss and accuracy use only that position.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from Dyck_K_experiments.data.dyck_generator import DyckGenerator
from Dyck_K_experiments.models.pdssm import StateTrackingPDSSM
from Dyck_K_experiments.models.transformer import StateTrackingTransformer


# ----------------------------------------------------------------------
# Batch construction: Dyck generator -> (X, y, mask) in harness format.
# ----------------------------------------------------------------------

def make_batch(gen: DyckGenerator, batch_size: int, min_len: int, max_len: int,
               positive_ratio: float = 0.5,
               negative_strategy: str = "edit") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = gen.generate_dataset(
        n_samples=batch_size,
        length_range=(min_len, max_len),
        positive_ratio=positive_ratio,
        negative_strategy=negative_strategy,
        pad_value=-1,
    )
    tokens = out["tokens"]
    raw_labels = out["labels"]
    lengths = out["lengths"]

    X = np.where(tokens == -1, 0, tokens + 1).astype(np.int32)
    y = (raw_labels + 1).astype(np.int32)     # 0->1 invalid, 1->2 valid
    B, L = X.shape
    mask = np.zeros((B, L), dtype=bool)
    mask[np.arange(B), lengths - 1] = True
    return X, y, mask


# ----------------------------------------------------------------------
# Loss / accuracy — last-token classification.
# ----------------------------------------------------------------------

def _last_logits(model, X, mask):
    logits = jax.vmap(model)(X)                              # (B, L, C)
    return jnp.sum(logits * mask[:, :, None].astype(logits.dtype), axis=1)


def loss_fn(model, X, y, mask):
    last = _last_logits(model, X, mask)                      # (B, C)
    return optax.softmax_cross_entropy_with_integer_labels(last, y).mean()


def accuracy_fn(model, X, y, mask):
    last = _last_logits(model, X, mask)
    return jnp.mean(jnp.argmax(last, axis=-1) == y)


# ----------------------------------------------------------------------
# Train / eval steps (JIT closures capture the optimizer).
# ----------------------------------------------------------------------

def make_train_step(optimizer):
    @eqx.filter_jit
    def train_step(model, opt_state, X, y, mask):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, X, y, mask)
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss
    return train_step


@eqx.filter_jit
def eval_step(model, X, y, mask):
    return accuracy_fn(model, X, y, mask)


def eval_full(model, val_X, val_y, val_mask, val_batch_size):
    n = val_X.shape[0]
    total_correct = 0.0
    total = 0
    for start in range(0, n, val_batch_size):
        end = min(start + val_batch_size, n)
        acc = eval_step(
            model, val_X[start:end], val_y[start:end], val_mask[start:end]
        )
        count = end - start
        total_correct += float(acc) * count
        total += count
    return total_correct / total


# ----------------------------------------------------------------------
# Main.
# ----------------------------------------------------------------------

def build_model(args, key, vocab_size: int, label_dim: int):
    if args.model == "pdssm":
        return StateTrackingPDSSM(
            vocab_size=vocab_size, label_dim=label_dim,
            N=args.pdssm_N, H=args.pdssm_H,
            num_layers=args.num_layers, K=args.pdssm_K,
            key=key,
        )
    return StateTrackingTransformer(
        vocab_size=vocab_size, label_dim=label_dim,
        d_model=args.tf_d_model, n_heads=args.tf_n_heads, d_ff=args.tf_d_ff,
        num_layers=args.num_layers, key=key,
    )


def count_params(model) -> int:
    return int(sum(
        l.size for l in jax.tree_util.tree_leaves(model) if hasattr(l, "shape")
    ))


def main() -> None:
    p = argparse.ArgumentParser()
    # Task
    p.add_argument("--model", choices=("pdssm", "transformer"), default="pdssm")
    p.add_argument("--k", type=int, default=1, help="Dyck-k")
    p.add_argument("--max-depth", type=int, default=40)
    p.add_argument("--min-train-length", type=int, default=3)
    p.add_argument("--max-train-length", type=int, default=40)
    p.add_argument("--min-val-length", type=int, default=40)
    p.add_argument("--max-val-length", type=int, default=256)
    p.add_argument("--positive-ratio", type=float, default=0.5)
    p.add_argument("--negative-strategy", choices=("edit", "random"), default="edit")
    # Model
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--pdssm-N", type=int, default=192)
    p.add_argument("--pdssm-H", type=int, default=128)
    p.add_argument("--pdssm-K", type=int, default=10)
    p.add_argument("--tf-d-model", type=int, default=192)
    p.add_argument("--tf-n-heads", type=int, default=6)
    p.add_argument("--tf-d-ff", type=int, default=960)
    # Training
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--num-steps", type=int, default=10_001)
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="global-norm gradient clip; 0 to disable")
    p.add_argument("--log-every", type=int, default=1,
                   help="print loss every N steps (cheap, just a GPU->CPU sync)")
    p.add_argument("--print-steps", type=int, default=1000,
                   help="run full validation every N steps")
    p.add_argument("--num-val-samples", type=int, default=8192)
    p.add_argument("--val-batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-fraction", type=float, default=0.1)
    p.add_argument("--early-stop", type=float, default=0.9995,
                   help="stop when val_acc >= this threshold (on a val eval)")
    p.add_argument("--early-stop-loss", type=float, default=0.0001,
                   help="stop when training loss drops below this; 0 to disable")
    p.add_argument("--save-path", type=str, default=None,
                   help="where to write the final model (.eqx) and metrics (.json). "
                        "Default: Dyck_K_experiments/checkpoints/{model}_dyck{k}")
    p.add_argument("--no-save", action="store_true",
                   help="disable checkpointing")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    vocab_size = 2 * args.k + 1
    label_dim = 3

    print(f"# Dyck-{args.k}  max_depth={args.max_depth}  "
          f"train_len=[{args.min_train_length},{args.max_train_length}]  "
          f"val_len=[{args.min_val_length},{args.max_val_length}]")

    gen_train = DyckGenerator(k=args.k, max_depth=args.max_depth, seed=args.seed + 1000)
    gen_val = DyckGenerator(k=args.k, max_depth=args.max_depth, seed=args.seed + 2000)

    print(f"# Pre-generating {args.num_val_samples} val samples...")
    t0 = time.time()
    vX, vy, vM = make_batch(
        gen_val, args.num_val_samples,
        args.min_val_length, args.max_val_length,
        positive_ratio=args.positive_ratio,
        negative_strategy=args.negative_strategy,
    )
    val_X = jnp.array(vX)
    val_y = jnp.array(vy)
    val_mask = jnp.array(vM)
    print(f"# val generated in {time.time() - t0:.1f}s  shape={val_X.shape}")

    key = jr.PRNGKey(args.seed)
    model = build_model(args, key, vocab_size, label_dim)
    print(f"# model={args.model}  params={count_params(model):,}")

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr,
        warmup_steps=int(args.warmup_fraction * args.num_steps),
        decay_steps=args.num_steps, end_value=args.lr * 0.1,
    )
    adamw = optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay)
    if args.grad_clip > 0:
        optimizer = optax.chain(optax.clip_by_global_norm(args.grad_clip), adamw)
    else:
        optimizer = adamw
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    train_step = make_train_step(optimizer)

    # Quick sanity-eval before training starts
    acc0 = eval_full(model, val_X, val_y, val_mask, args.val_batch_size)
    print(f"# step 0 pre-train val_acc {acc0:.4f}  (random guess ≈ 0.5)")

    start = time.time()
    best = 0.0
    for step_idx in range(args.num_steps):
        X_np, y_np, M_np = make_batch(
            gen_train, args.batch_size,
            args.min_train_length, args.max_train_length,
            positive_ratio=args.positive_ratio,
            negative_strategy=args.negative_strategy,
        )
        X, y, M = jnp.array(X_np), jnp.array(y_np), jnp.array(M_np)
        model, opt_state, loss = train_step(model, opt_state, X, y, M)

        do_eval = step_idx % args.print_steps == 0 and step_idx > 0
        do_log = args.log_every > 0 and step_idx % args.log_every == 0

        if do_eval:
            acc = eval_full(model, val_X, val_y, val_mask, args.val_batch_size)
            best = max(best, acc)
            elapsed = time.time() - start
            sps = (step_idx + 1) / elapsed
            print(f"step {step_idx:>6}  loss {float(loss):.4f}  "
                  f"val {acc:.4f}  best {best:.4f}  "
                  f"{sps:.2f} steps/s  {elapsed:.0f}s", flush=True)
            if acc >= args.early_stop:
                print(f"[ok] early-stop at step {step_idx} (val_acc {acc:.4f})")
                break
        elif do_log:
            elapsed = time.time() - start
            sps = (step_idx + 1) / elapsed if elapsed > 0 else 0
            print(f"step {step_idx:>6}  loss {float(loss):.4f}  "
                  f"{sps:.1f} steps/s", flush=True)

        if args.early_stop_loss > 0 and float(loss) < args.early_stop_loss:
            acc = eval_full(model, val_X, val_y, val_mask, args.val_batch_size)
            best = max(best, acc)
            print(f"[ok] early-stop at step {step_idx} "
                  f"(loss {float(loss):.6f} < {args.early_stop_loss}; "
                  f"val_acc {acc:.4f})", flush=True)
            break

    final = eval_full(model, val_X, val_y, val_mask, args.val_batch_size)
    best = max(best, final)
    print(f"# final val_acc {final:.4f}  best {best:.4f}")

    if not args.no_save:
        save_path = args.save_path
        if save_path is None:
            save_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "checkpoints"
            )
            save_path = os.path.join(save_dir, f"{args.model}_dyck{args.k}")
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        model_path = save_path + ".eqx"
        meta_path = save_path + ".json"
        eqx.tree_serialise_leaves(model_path, model)
        meta = {
            "model": args.model,
            "k": args.k,
            "max_depth": args.max_depth,
            "min_train_length": args.min_train_length,
            "max_train_length": args.max_train_length,
            "min_val_length": args.min_val_length,
            "max_val_length": args.max_val_length,
            "num_layers": args.num_layers,
            "pdssm": {"N": args.pdssm_N, "H": args.pdssm_H, "K": args.pdssm_K},
            "transformer": {
                "d_model": args.tf_d_model,
                "n_heads": args.tf_n_heads,
                "d_ff": args.tf_d_ff,
            },
            "num_steps_target": args.num_steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_fraction": args.warmup_fraction,
            "grad_clip": args.grad_clip,
            "positive_ratio": args.positive_ratio,
            "negative_strategy": args.negative_strategy,
            "seed": args.seed,
            "vocab_size": vocab_size,
            "label_dim": label_dim,
            "final_val_acc": float(final),
            "best_val_acc": float(best),
            "param_count": count_params(model),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"# saved model -> {model_path}")
        print(f"# saved meta  -> {meta_path}")


if __name__ == "__main__":
    main()
