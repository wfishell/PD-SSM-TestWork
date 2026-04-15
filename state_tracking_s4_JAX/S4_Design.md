# S4 (S4D) for State Tracking Tasks

## Purpose

A diagonal state-space model baseline (S4D variant) for the formal language state-tracking tasks. S4D uses a **fixed, input-independent** transition matrix A — in contrast to PD-SSM where A(u_t) depends on the current input. This makes S4D a linear time-invariant (LTI) system, which is the core limitation the PD-SSM paper argues against. The L* extraction will reveal whether a fixed-dynamics SSM can learn clean automata for these tasks.

## Interface contract

Same signature as `StateTrackingPDSSM`:

```python
class StateTrackingS4D(eqx.Module):
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
[S4DBlock] x num_layers
    |  S4DLayer(N, H)
    |      — A: (N,) complex, FIXED (learned but input-independent)
    |          initialised via HiPPO or log-uniform on unit disk
    |      — B: (N, H) complex, FIXED
    |      — C: (H, N) complex, FIXED
    |      — D: (H,) real skip connection
    |      — dt: (H,) or scalar, learned log-discretisation step
    |      — Discretise: A_bar = exp(A * dt), B_bar = (A_bar - I) * A^{-1} * B
    |      — Parallel scan over sequence using (A_bar, B_bar * u_t)
    |  + residual connection (optional)
    |  LayerNorm(H)
    |
Linear(H, label_dim)
```

### The key difference from PD-SSM

| | S4D | PD-SSM |
|---|---|---|
| Transition matrix | A is fixed (same for every input token) | A(u_t) = P(u_t) D(u_t) changes per token |
| System type | Linear time-invariant (LTI) | Linear time-varying (LTV) |
| Expressiveness | Cannot represent input-dependent state transitions | Can represent any finite automaton (with enough states) |
| Parallel scan | Standard: scan over (A_bar, B_bar * u_t) pairs | Custom: scan over (P, D, Bu) triples with permutation composition |

Because A is fixed, S4D applies the **same dynamics** regardless of whether the input is 'a' or 'b'. The only input-dependence comes through B (the input projection). This is fundamentally limiting for automata that require different transitions per symbol.

### HiPPO initialisation

S4's key contribution is the HiPPO initialisation for A, which gives the SSM a good starting point for memorising history. For the diagonal (S4D) variant:

```
A_n = -1/2 + n*i    (HiPPO-LegS, diagonal approximation)
```

This places eigenvalues along the left half-plane with increasing imaginary parts, giving each state dimension a different oscillation frequency.

Alternative: **log-uniform initialisation** on the unit disk in discrete time:

```
|A_n| ~ exp(Uniform(log(dt_min), log(dt_max)))
angle(A_n) ~ Uniform(0, 2*pi)
```

We will implement both and default to HiPPO.

### Discretisation

Continuous-time parameters (A, B) are discretised via ZOH (zero-order hold):

```
A_bar = exp(A * dt)
B_bar = (exp(A * dt) - I) * A^{-1} * B
```

For diagonal A this is elementwise — no matrix exponentials needed.

The learned `log_dt` parameter controls the timescale. Initialised from `Uniform(log(dt_min), log(dt_max))`.

### Parallel scan

Since A_bar is the same at every timestep, the recurrence `h_t = A_bar * h_{t-1} + B_bar * u_t` can be computed with `jax.lax.associative_scan` using the standard binary operator:

```python
def binary_op(q_i, q_j):
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_i * A_j, A_j * b_i + b_j
```

This is simpler than PD-SSM's operator because there is no permutation matrix.

## Hyperparameters

```json
{
  "model_type": "s4d",
  "num_layers": 2,
  "embed_size": 128,
  "state_size": 128,
  "dt_min": 0.001,
  "dt_max": 0.1,
  "init": "hippo"
}
```

### Parameter count estimate

Per S4DLayer:
- A: N complex = 256 real params (but these are the critical ones)
- B: N x H complex = 2 x 128 x 128 = 32,768
- C: H x N complex = 2 x 128 x 128 = 32,768
- D: H = 128
- log_dt: H = 128
- Per layer total: ~66K

Full model:
- Embedding: vocab_size x 128
- 2 blocks (layer + layernorm): 2 x (66K + 256) ~ 133K
- Readout: 128 x vocab_size
- **Total: ~134K** (parity) — smaller than PD-SSM's ~397K

This is notably fewer parameters than PD-SSM because there are no permutation dictionary matrices (K x N x N). The comparison is fair in the sense that S4D has **fewer parameters to work with** — if it still fails, the failure is architectural, not capacity-limited. If desired, we can increase N to match parameter counts more closely.

### Training

Same protocol as all models:
- AdamW, lr=0.002, weight_decay=0.01
- Warmup 10% + cosine decay
- Early stop at 99.95% validation accuracy
- Train lengths 3-40, validate 40-256
- 3 seeds per task

## Dependencies

- `jax`, `equinox`, `optax` — already installed, no new packages needed.

## Files to create

```
state_tracking_s4_JAX/
  models/
    s4d.py              — StateTrackingS4D, S4DBlock, S4DLayer
  train.py              — adapted from state_tracking_JAX/train.py
  evaluate.py           — adapted from state_tracking_JAX/evaluate.py
  experiment_configs/
    parity_s4d_0.json
    parity_s4d_1.json
    parity_s4d_2.json
    even_pairs_s4d_0.json
    ...                 — 4 tasks x 3 seeds = 12 configs
```

## Expected behaviour under L*

| Task | Expected outcome |
|------|-----------------|
| parity | **Likely fails.** Parity requires different transitions for 'a' vs 'b'. S4D applies the same A regardless of input — it can only modulate through B, which is a weaker mechanism. May achieve moderate accuracy on short sequences but poor length generalisation. L* will likely extract a bloated or incorrect automaton. |
| even_pairs | **Possibly succeeds.** Only needs to compare first and last tokens. S4D might encode "remember first token" in its fixed dynamics and use B to flag the last. But distinguishing (a,a) from (a,b) at the end still requires some input-dependent logic. |
| cycle_nav | **Likely fails.** Counting +1/-1 modulo 5 requires input-dependent state transitions. Fixed A cannot route the state differently for + vs - inputs. |
| mod_arith | **Almost certainly fails.** The most complex task, requiring multiplication and addition with operator precedence. Fixed dynamics cannot handle this. |

This is the **intended negative result** — S4D's failure on these tasks demonstrates why input-dependent transitions (PD-SSM) are necessary for state tracking.
