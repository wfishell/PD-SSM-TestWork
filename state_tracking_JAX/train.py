import datetime
import json
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from data_dir.dataloaders import create_fl_dataloaders
from models.pdssm import StateTrackingPDSSM


# ---------------------------------------------------------------------------
# Loss and training step
# ---------------------------------------------------------------------------

def loss_fn(model, X, y, mask):
    """
    X:    (B, L) int32  — padded token sequences
    y:    (B,)   int32  — class label per sequence
    mask: (B, L) bool   — True at the last real token position per sequence
    """
    logits = jax.vmap(model)(X)                           # (B, L, label_dim)
    last_pos = jnp.argmax(mask.astype(jnp.int32), axis=-1)  # (B,)
    B = X.shape[0]
    last_logits = logits[jnp.arange(B), last_pos]         # (B, label_dim)
    return jnp.mean(
        optax.softmax_cross_entropy_with_integer_labels(last_logits, y)
    )


@eqx.filter_jit
def compute_accuracy(model, X, y, mask):
    logits = jax.vmap(model)(X)
    last_pos = jnp.argmax(mask.astype(jnp.int32), axis=-1)
    B = X.shape[0]
    last_logits = logits[jnp.arange(B), last_pos]
    return jnp.mean(jnp.argmax(last_logits, axis=-1) == y)


def make_make_step(opt):
    """Returns a JIT-compiled training step function closed over the optimizer."""

    @eqx.filter_jit
    def make_step(model, opt_state, X, y, mask):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, X, y, mask)
        updates, new_opt_state = opt.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, new_opt_state, loss

    return make_step


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_experiment(config):
    seed = config.get("seed", 0)
    key = jr.PRNGKey(seed)

    task = config["task"]
    num_layers = config["num_layers"]
    state_size = config["state_size"]       # N: hidden state dimension
    embed_size = config["embed_size"]       # H: model dimension
    dictionary_size = config.get("dictionary_size", 6)  # K
    batch_size = config["batch_size"]
    num_steps = config["num_steps"]
    print_steps = config["print_steps"]
    learning_rate = config["learning_rate"]
    weight_decay = config.get("weight_decay", 1e-2)
    warmup_fraction = config.get("warmup_fraction", 0.1)
    early_stop_threshold = config.get("early_stop_threshold", 0.9995)
    num_val_samples = config.get("num_val_samples", 8192)
    val_batch_size = config.get("val_batch_size", 64)
    config_name = config.get("config_name", "unnamed")

    max_train_length = config.get("max_train_length", 40)
    min_train_length = config.get("min_train_length", 3)
    max_val_length = config.get("max_val_length", 256)
    min_val_length = config.get("min_val_length", max_train_length)

    # --- Dataloaders ---
    dataloaders, vocab_size = create_fl_dataloaders(
        task=task,
        min_train_length=min_train_length,
        max_train_length=max_train_length,
        min_val_length=min_val_length,
        max_val_length=max_val_length,
        num_val_samples=num_val_samples,
        val_seed=seed * 2,
    )
    label_dim = vocab_size  # labels are drawn from the same vocab

    # --- Model ---
    model_key, train_key = jr.split(key)
    model = StateTrackingPDSSM(
        vocab_size=vocab_size,
        label_dim=label_dim,
        N=state_size,
        H=embed_size,
        num_layers=num_layers,
        K=dictionary_size,
        key=model_key,
    )
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))
    print(f"Trainable parameters: {n_params:,}")

    # --- Optimizer ---
    warmup_steps = int(warmup_fraction * num_steps)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=num_steps,
        end_value=1e-5,
    )
    opt = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    make_step = make_make_step(opt)

    # --- Output dirs ---
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # --- Training loop ---
    steps, val_accs = [], []
    running_loss = jnp.zeros(())
    steps_since_print = 0
    start = time.time()
    early_stop = False

    for step, (X, y, mask) in zip(
        range(num_steps),
        dataloaders["train"].loop(batch_size, key=train_key),
    ):
        model, opt_state, loss = make_step(model, opt_state, X, y, mask)
        running_loss = running_loss + loss  # stays on device, no GPU→CPU sync
        steps_since_print += 1

        if step % print_steps == 0:
            # Validation accuracy over the pre-generated val set
            val_correct = 0
            val_total = 0
            for X_val, y_val, mask_val in dataloaders["val"].loop_epoch(val_batch_size):
                acc = compute_accuracy(model, X_val, y_val, mask_val)
                val_correct += float(acc) * X_val.shape[0]
                val_total += X_val.shape[0]

            val_acc = val_correct / val_total if val_total > 0 else 0.0
            avg_loss = float(running_loss) / max(steps_since_print, 1)
            elapsed = time.time() - start

            print(
                f"Step: {step}, Loss: {avg_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}, Time: {elapsed:.2f}s"
            )

            steps.append(step)
            val_accs.append(val_acc)
            running_loss = jnp.zeros(())
            steps_since_print = 0
            start = time.time()

            # Save results
            results = {"config": config, "steps": steps, "val_accs": val_accs,
                       "early_stop": early_stop}
            with open(f"results/results_{task}_pdssm_{config_name}.json", "w") as f:
                json.dump(results, f, indent=2)

            if val_acc >= early_stop_threshold:
                print("Early stop triggered.")
                early_stop = True
                break

    print("Training complete." if not early_stop else "Stopped early.")
    eqx.tree_serialise_leaves(f"checkpoints/model_{task}_pdssm_{config_name}.eqx", model)
    print(f"Model saved to checkpoints/model_{task}_pdssm_{config_name}.eqx")
    return model, steps, val_accs, early_stop
