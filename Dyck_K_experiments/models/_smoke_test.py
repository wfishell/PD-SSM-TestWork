"""Instantiate the Dyck PD-SSM and Transformer and verify shapes + param counts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from Dyck_K_experiments.models.pdssm import StateTrackingPDSSM
from Dyck_K_experiments.models.transformer import StateTrackingTransformer


def count_params(model) -> int:
    leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda x: x if hasattr(x, "shape") else None, model)
    )
    return int(sum(l.size for l in leaves if l is not None))


def predicted_pdssm_params(
    vocab_size: int, label_dim: int, N: int, H: int, num_layers: int, K: int
) -> int:
    layer = (
        2 * N * H          # B_re, B_im
        + 2 * H * N        # C_re, C_im
        + H                # D
        + K * N * N        # P_dict
        + K * H            # P_selector
        + N * H + N        # W1_mag, b1_mag
        + N * H + N        # W1_pha, b1_pha
    )
    block = layer + 2 * H  # LayerNorm (scale + bias)
    return num_layers * block + vocab_size * H + (H * label_dim + label_dim)


def predicted_transformer_params(
    vocab_size: int, label_dim: int, d: int, d_ff: int, num_layers: int
) -> int:
    # no biases on Q/K/V/O or FFN; LN has scale + bias (2d each)
    attn = 4 * d * d              # q, k, v, o
    ffn = 2 * d * d_ff            # up, down
    block = attn + ffn + 2 * (2 * d)  # 2 LayerNorms
    emb = vocab_size * d
    final_ln = 2 * d
    readout = d * label_dim + label_dim
    return num_layers * block + emb + final_ln + readout


def check_pdssm() -> None:
    vocab_size = 7
    label_dim = 3
    N, H, K, num_layers = 192, 128, 10, 2

    key = jr.PRNGKey(0)
    model = StateTrackingPDSSM(
        vocab_size=vocab_size, label_dim=label_dim,
        N=N, H=H, num_layers=num_layers, K=K, key=key,
    )

    x = jnp.array(
        [[0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        dtype=jnp.int32,
    )
    logits = jax.vmap(model)(x)
    assert logits.shape == (2, 16, label_dim), logits.shape

    measured = count_params(model)
    predicted = predicted_pdssm_params(vocab_size, label_dim, N, H, num_layers, K)
    print("---- PD-SSM ----")
    print(f"config:  vocab={vocab_size} label_dim={label_dim} "
          f"N={N} H={H} K={K} num_layers={num_layers}")
    print(f"predicted: {predicted:,}")
    print(f"measured:  {measured:,}")
    print(f"logits:    {logits.shape}")
    assert measured == predicted, (measured, predicted)
    print("[ok] PD-SSM measured == predicted")


def check_transformer() -> None:
    vocab_size = 7
    label_dim = 3
    d_model, n_heads, d_ff, num_layers = 192, 6, 960, 2

    key = jr.PRNGKey(1)
    model = StateTrackingTransformer(
        vocab_size=vocab_size, label_dim=label_dim,
        d_model=d_model, n_heads=n_heads, d_ff=d_ff,
        num_layers=num_layers, key=key,
    )

    x = jnp.array(
        [[0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        dtype=jnp.int32,
    )
    logits = jax.vmap(model)(x)
    assert logits.shape == (2, 16, label_dim), logits.shape

    measured = count_params(model)
    predicted = predicted_transformer_params(
        vocab_size, label_dim, d_model, d_ff, num_layers
    )
    print("---- Transformer ----")
    print(f"config:  vocab={vocab_size} label_dim={label_dim} "
          f"d={d_model} h={n_heads} d_ff={d_ff} num_layers={num_layers}")
    print(f"predicted: {predicted:,}")
    print(f"measured:  {measured:,}")
    print(f"logits:    {logits.shape}")
    assert measured == predicted, (measured, predicted)
    print("[ok] Transformer measured == predicted")


def check_causality() -> None:
    """Modifying x[t] must not change output at positions < t."""
    key = jr.PRNGKey(2)
    model = StateTrackingTransformer(
        vocab_size=7, label_dim=3,
        d_model=192, n_heads=6, d_ff=960, num_layers=2, key=key,
    )
    x1 = jnp.array([0, 1, 2, 3, 4, 5, 6, 0], dtype=jnp.int32)
    x2 = x1.at[5].set(3)  # perturb position 5
    y1 = model(x1)
    y2 = model(x2)
    # positions 0..4 should be identical
    for t in range(5):
        assert jnp.allclose(y1[t], y2[t], atol=1e-5), t
    # position 5 onwards should differ
    assert not jnp.allclose(y1[5], y2[5], atol=1e-5)
    print("[ok] Transformer is causal (prefix outputs invariant to future perturbations)")


def main() -> None:
    check_pdssm()
    check_transformer()
    check_causality()

    # Side-by-side summary.
    pd = predicted_pdssm_params(7, 3, 192, 128, 2, 10)
    tf = predicted_transformer_params(7, 3, 192, 960, 2)
    diff = pd - tf
    pct = 100.0 * diff / pd
    print(f"\nPD-SSM:      {pd:,}")
    print(f"Transformer: {tf:,}")
    print(f"delta:       {diff:,}  ({pct:+.2f}%)")


if __name__ == "__main__":
    main()
