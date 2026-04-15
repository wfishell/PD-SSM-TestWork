# Mamba (Selective SSM) for State Tracking Tasks

## Purpose

A selective state-space model baseline for the formal language state-tracking tasks. Mamba's key innovation is making the SSM parameters (B, C, and the discretisation step dt) **input-dependent** via a selection mechanism. This gives it more expressiveness than S4D (fixed dynamics) but with a different structure than PD-SSM (permutation + diagonal). The L* extraction will reveal whether Mamba's selection mechanism learns clean automata.

## Interface contract

Same signature as `StateTrackingPDSSM`:

```python
class StateTrackingMamba(eqx.Module):
    def __call__(self, x):
        """
        x: (L,) int32  — token IDs for a single sequence
        returns: (L, label_dim) float logits
        """
```

- Single sequence in, per-position logits out. Batching via `jax.vmap` externally.
- Reuses the existing `train.py`, `evaluate.py`, `learn_automaton.py`, and dataloaders.

## Architecture

```
Token IDs (L,) int32
    |
Embedding(vocab_size, H)
    |
[MambaBlock] x num_layers
    |
    |  Input projection: Linear(H, 2 * d_inner)  — split into x_branch and z_branch
    |
    |  x_branch (d_inner):
    |      Conv1D(d_inner, kernel=d_conv)  — short causal convolution
    |      SiLU activation
    |      |
    |      SSM core:
    |          — A: (d_inner, N) complex, FIXED (learned, structured init)
    |          — B(u_t): Linear(d_inner, N)        — input-dependent
    |          — C(u_t): Linear(d_inner, N)        — input-dependent
    |          — dt(u_t): Linear(d_inner, d_inner) — input-dependent discretisation
    |          — Discretise: A_bar_t = exp(A * dt_t), B_bar_t = dt_t * B_t
    |          — Selective scan: h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
    |          — Output: y_t = C_t * h_t
    |
    |  z_branch (d_inner):
    |      SiLU activation (gate)
    |
    |  y = x_branch_output * z_branch     — gated combination
    |  Output projection: Linear(d_inner, H)
    |  + residual connection
    |  LayerNorm(H)
    |
Linear(H, label_dim)
```

### How Mamba differs from PD-SSM and S4D

| | S4D | Mamba | PD-SSM |
|---|---|---|---|
| A matrix | Fixed | Fixed (but dt is input-dependent, so A_bar varies) | Input-dependent P(u)D(u) |
| B matrix | Fixed | **Input-dependent** B(u_t) | Fixed |
| C matrix | Fixed | **Input-dependent** C(u_t) | Fixed |
| dt (discretisation) | Learned scalar | **Input-dependent** dt(u_t) | N/A (discrete-time) |
| Core mechanism | LTI system | Selection (content-aware gating of what enters/exits state) | Structured permutation + diagonal |
| Sparsity | Dense diagonal | Dense diagonal | Sparse (permutation + diagonal) |

### The selection mechanism

Mamba's thesis: the SSM should **select** which information to let into and read from the state based on the current input. This is achieved by making B, C, and dt functions of the input:

- **B(u_t)** controls what gets written to the state — different inputs write different projections
- **C(u_t)** controls what gets read from the state — different queries for different contexts
- **dt(u_t)** controls the discretisation step — effectively how much to "forget" vs "remember"
  - Large dt → A_bar ≈ 0, state resets (forget everything)
  - Small dt → A_bar ≈ I, state persists (remember everything)

This is a softer form of input-dependence than PD-SSM's hard permutation switching. The question is whether this softer selection is sufficient to learn clean finite automata.

### Causal convolution

Before the SSM, Mamba applies a short 1-D causal convolution (kernel size d_conv, typically 4). This gives the model a small local receptive field, similar to how the SSM's B projection mixes adjacent tokens. For our short-alphabet tasks this may not matter much, but we include it for architectural faithfulness.

### Simplified implementation

The original Mamba uses a hardware-aware selective scan kernel (custom CUDA). For our tasks (short sequences, small state), we use a **pure JAX implementation** with `jax.lax.associative_scan`. Since A_bar and B_bar vary per timestep, the scan operator is:

```python
def binary_op(q_i, q_j):
    A_i, b_i = q_i    # A_i is (N,) diagonal, b_i is (N,)
    A_j, b_j = q_j
    return A_i * A_j, A_j * b_i + b_j
```

Same structure as S4D's scan but with **different A_bar_t at each position** (because dt varies).

### Gating

The z_branch acts as a multiplicative gate on the SSM output, similar to a GLU (gated linear unit). This is a standard Mamba design choice that helps gradient flow.

## Hyperparameters

```json
{
  "model_type": "mamba",
  "num_layers": 2,
  "embed_size": 128,
  "d_inner": 256,
  "state_size": 16,
  "d_conv": 4,
  "dt_rank": 16,
  "dt_min": 0.001,
  "dt_max": 0.1
}
```

Note: Mamba typically uses a **smaller state size** (N=16) than S4/PD-SSM (N=128) because the input-dependent B/C projections make each state dimension more expressive. `d_inner` is the expanded internal dimension (typically 2x embed_size).

### Parameter count estimate

Per MambaBlock:
- Input projection: H x (2 * d_inner) = 128 x 512 = 65,536
- Conv1D: d_inner x d_conv = 256 x 4 = 1,024
- SSM projections:
  - B projection: d_inner x N = 256 x 16 = 4,096
  - C projection: d_inner x N = 256 x 16 = 4,096
  - dt projection: d_inner x dt_rank = 256 x 16 = 4,096 (plus dt_rank x d_inner = 4,096)
  - A: d_inner x N = 4,096
  - D: d_inner = 256
- Output projection: d_inner x H = 256 x 128 = 32,768
- LayerNorm: 2 x 128 = 256
- Per block total: ~120K

Full model:
- Embedding: vocab_size x 128
- 2 blocks: ~240K
- Readout: 128 x vocab_size
- **Total: ~241K** (parity)

### Training

Same protocol as all models:
- AdamW, lr=0.002, weight_decay=0.01
- Warmup 10% + cosine decay
- Early stop at 99.95% validation accuracy
- Train lengths 3-40, validate 40-256
- 3 seeds per task

## Dependencies

- `jax`, `equinox`, `optax` — already installed, no new packages needed.
- No `mamba_ssm` CUDA package required — pure JAX implementation.

## Files to create

```
state_tracking_mamba_JAX/
  models/
    mamba.py            — StateTrackingMamba, MambaBlock, SelectiveSSM
  train.py              — adapted from state_tracking_JAX/train.py
  evaluate.py           — adapted from state_tracking_JAX/evaluate.py
  experiment_configs/
    parity_mamba_0.json
    parity_mamba_1.json
    parity_mamba_2.json
    even_pairs_mamba_0.json
    ...                 — 4 tasks x 3 seeds = 12 configs
```

## Expected behaviour under L*

| Task | Expected outcome |
|------|-----------------|
| parity | **Interesting test case.** Mamba can modulate dt per input, so it could in principle learn to toggle state on 'b' and preserve on 'a'. But the diagonal A with soft gating may not produce a clean 2-state separation. L* may extract a correct automaton with extra states, or a slightly noisy one. |
| even_pairs | **Likely succeeds.** Selection mechanism can easily learn to write the first token to state and compare at the end. Should extract a small automaton. |
| cycle_nav | **Uncertain.** Mamba can modulate B per input (+1, -1, STAY), which gives it a way to route information differently. But without PD-SSM's hard permutation structure, the state updates are soft — may work for short sequences but degrade on longer ones. |
| mod_arith | **Likely struggles.** Requires structured multi-step computation (precedence, modular ops). Mamba's selection is powerful but unstructured — it may learn approximate solutions that break on edge cases. L* will reveal this. |

### The key question for Mamba

Mamba's selection is **soft and continuous** — dt, B, C are smooth functions of the input. PD-SSM's transitions are **hard and discrete** — permutation matrices with argmax selection. The hypothesis is that hard discrete switching is better for learning clean finite automata, even though Mamba's soft selection is more general in principle. The L* extraction will test this directly.
