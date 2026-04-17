"""Causal pre-norm Transformer with RoPE — param-matched to the Dyck PD-SSM.

Interface matches StateTrackingPDSSM:
    model(x_tokens: (L,) int32) -> logits: (L, label_dim) float

No bias on Q/K/V/O projections or FFN linears. LayerNorm keeps its
default scale+bias. RoPE is applied only to Q and K (standard). Causal
mask is built once per forward pass from the input length.
"""

from __future__ import annotations

from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


# ---------------------------------------------------------------------------
# RoPE: rotary positional embeddings (no trainable parameters).
# ---------------------------------------------------------------------------

def _rope_frequencies(head_dim: int, base: float = 10000.0) -> jnp.ndarray:
    """Inverse frequencies for RoPE: shape (head_dim // 2,)."""
    assert head_dim % 2 == 0, f"head_dim must be even for RoPE, got {head_dim}"
    half = head_dim // 2
    return 1.0 / (base ** (jnp.arange(half, dtype=jnp.float32) / half))


def apply_rope(x: jnp.ndarray, positions: jnp.ndarray, inv_freq: jnp.ndarray) -> jnp.ndarray:
    """Apply RoPE to x of shape (L, n_heads, head_dim).

    positions: (L,) int32 — absolute position for each token.
    inv_freq:  (head_dim // 2,) float32.
    """
    # (L, head_dim // 2)
    freqs = positions[:, None].astype(jnp.float32) * inv_freq[None, :]
    cos = jnp.cos(freqs)[:, None, :]  # (L, 1, head_dim // 2)
    sin = jnp.sin(freqs)[:, None, :]

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    # rotation in each consecutive pair of dims
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    # re-interleave
    out = jnp.stack([rot1, rot2], axis=-1)
    return out.reshape(x.shape)


# ---------------------------------------------------------------------------
# Causal self-attention with RoPE.
# ---------------------------------------------------------------------------

class CausalSelfAttention(eqx.Module):
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    o_proj: eqx.nn.Linear
    n_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)

    def __init__(self, d_model: int, n_heads: int, *, key):
        assert d_model % n_heads == 0, f"d_model={d_model} not divisible by n_heads={n_heads}"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        k1, k2, k3, k4 = jr.split(key, 4)
        self.q_proj = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k1)
        self.k_proj = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k2)
        self.v_proj = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k3)
        self.o_proj = eqx.nn.Linear(d_model, d_model, use_bias=False, key=k4)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (L, d_model) -> (L, d_model)."""
        L, _ = x.shape

        # Project and reshape into heads: (L, n_heads, head_dim)
        q = jax.vmap(self.q_proj)(x).reshape(L, self.n_heads, self.head_dim)
        k = jax.vmap(self.k_proj)(x).reshape(L, self.n_heads, self.head_dim)
        v = jax.vmap(self.v_proj)(x).reshape(L, self.n_heads, self.head_dim)

        # RoPE on Q and K only (standard); V stays untouched.
        # inv_freq is recomputed per call — it's tiny (head_dim/2 floats) and
        # avoiding a stored buffer keeps the trainable param count clean.
        inv_freq = _rope_frequencies(self.head_dim)
        positions = jnp.arange(L, dtype=jnp.int32)
        q = apply_rope(q, positions, inv_freq)
        k = apply_rope(k, positions, inv_freq)

        # Scaled dot-product attention, per head.
        # scores: (n_heads, L, L)
        scores = jnp.einsum('lhd,mhd->hlm', q, k) / jnp.sqrt(self.head_dim)

        # Causal mask: token at row l can attend to cols <= l.
        mask = jnp.tril(jnp.ones((L, L), dtype=bool))
        scores = jnp.where(mask[None, :, :], scores, -jnp.inf)

        attn = jax.nn.softmax(scores, axis=-1)  # (n_heads, L, L)
        out = jnp.einsum('hlm,mhd->lhd', attn, v)  # (L, n_heads, head_dim)
        out = out.reshape(L, self.n_heads * self.head_dim)
        return jax.vmap(self.o_proj)(out)  # (L, d_model)


# ---------------------------------------------------------------------------
# FFN: Linear -> GELU -> Linear, no biases.
# ---------------------------------------------------------------------------

class FFN(eqx.Module):
    up: eqx.nn.Linear
    down: eqx.nn.Linear

    def __init__(self, d_model: int, d_ff: int, *, key):
        k1, k2 = jr.split(key, 2)
        self.up = eqx.nn.Linear(d_model, d_ff, use_bias=False, key=k1)
        self.down = eqx.nn.Linear(d_ff, d_model, use_bias=False, key=k2)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda u: self.down(jax.nn.gelu(self.up(u))))(x)


# ---------------------------------------------------------------------------
# Pre-norm Transformer block: x + Attn(LN(x)); x + FFN(LN(x)).
# ---------------------------------------------------------------------------

class TransformerBlock(eqx.Module):
    norm1: eqx.nn.LayerNorm
    attn: CausalSelfAttention
    norm2: eqx.nn.LayerNorm
    ffn: FFN

    def __init__(self, d_model: int, n_heads: int, d_ff: int, *, key):
        k1, k2 = jr.split(key, 2)
        self.norm1 = eqx.nn.LayerNorm(shape=d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, key=k1)
        self.norm2 = eqx.nn.LayerNorm(shape=d_model)
        self.ffn = FFN(d_model, d_ff, key=k2)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x + self.attn(jax.vmap(self.norm1)(x))
        x = x + self.ffn(jax.vmap(self.norm2)(x))
        return x


# ---------------------------------------------------------------------------
# StateTrackingTransformer: matches StateTrackingPDSSM's interface.
# ---------------------------------------------------------------------------

class StateTrackingTransformer(eqx.Module):
    embedding: eqx.nn.Embedding
    blocks: List[TransformerBlock]
    final_norm: eqx.nn.LayerNorm
    readout: eqx.nn.Linear

    def __init__(
        self,
        vocab_size: int,
        label_dim: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_layers: int,
        *,
        key,
    ):
        """
        vocab_size: number of input token types (including padding token 0)
        label_dim:  number of output classes
        d_model:    embedding / residual-stream dimension
        n_heads:    number of attention heads (must divide d_model)
        d_ff:       hidden dim of the FFN
        num_layers: number of stacked TransformerBlocks
        """
        keys = jr.split(key, num_layers + 2)
        self.embedding = eqx.nn.Embedding(vocab_size, d_model, key=keys[0])
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, key=keys[i + 1])
            for i in range(num_layers)
        ]
        self.final_norm = eqx.nn.LayerNorm(shape=d_model)
        self.readout = eqx.nn.Linear(d_model, label_dim, key=keys[-1])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (L,) int  — token IDs for a single sequence
        returns: (L, label_dim) float logits
        """
        x = jax.vmap(self.embedding)(x)  # (L, d_model)
        for block in self.blocks:
            x = block(x)
        x = jax.vmap(self.final_norm)(x)
        return jax.vmap(self.readout)(x)  # (L, label_dim)
