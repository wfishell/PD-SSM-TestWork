# Transformer for State Tracking Tasks

## Purpose

A small causal Transformer baseline for the formal language state-tracking tasks (parity, even_pairs, cycle_nav, mod_arith_no_brack). This model serves as a comparison point against PD-SSM, S4, and Mamba — the central question is whether a purely attention-based model can learn a compact automaton that L* can extract.

## Interface contract

The model must match the same input/output signature as `StateTrackingPDSSM`:

```python
class StateTrackingTransformer(eqx.Module):
    def __call__(self, x):
        """
        x: (L,) int32  — token IDs for a single sequence
        returns: (L, label_dim) float logits
        """
```

- Input is a 1-D integer sequence (no batch dim — `jax.vmap` handles batching externally).
- Output is logits at every position. Training loss is computed only at the last real token (using the mask from the dataloader).
- The existing `train.py`, `evaluate.py`, `learn_automaton.py`, and dataloader infrastructure are reused without modification — only the model class changes.

## Architecture

```
Token IDs (L,) int32
    |
Embedding(vocab_size, H)          — learned lookup table
    |
+ Learned positional encoding(L, H)  — sinusoidal or learned, up to max_pos positions
    |
[TransformerBlock] x num_layers
    |  CausalSelfAttention(H, n_heads)
    |      — Q, K, V projections: Linear(H, H) each
    |      — causal mask: lower-triangular, prevents attending to future tokens
    |      — scaled dot-product: softmax(QK^T / sqrt(d_k)) V
    |      — output projection: Linear(H, H)
    |  + residual connection
    |  LayerNorm(H)
    |  FFN: Linear(H, d_ff) -> GELU -> Linear(d_ff, H)
    |  + residual connection
    |  LayerNorm(H)
    |
Linear(H, label_dim)              — readout to class logits
```

### Why causal masking

The state-tracking tasks read the label from the last token position. The model must not attend to future tokens — otherwise it could cheat by looking ahead. Causal masking makes the Transformer autoregressive, matching the left-to-right processing of the SSM models.

### Positional encoding

Unlike SSMs which process tokens sequentially (position is implicit in the recurrence), the Transformer has no notion of order without positional information. Two options:

- **Learned positional embeddings** (simpler, default choice): a `(max_pos, H)` parameter added to the token embeddings. Set `max_pos` to the longest validation length (256) plus margin.
- **Sinusoidal**: fixed, generalises to unseen lengths in theory. But in practice learned embeddings work fine when the validation range is known.

Use **learned positional embeddings** with `max_pos = 512` to cover the validation range with headroom.

### Why there is no recurrent state

This is the key theoretical difference. SSMs maintain a hidden state `h_t` that compresses the history into a fixed-size vector — this is structurally analogous to a DFA state. The Transformer has no such bottleneck: at position t it attends to all positions 1..t with O(t) memory. This means:

- It can in principle solve any finite-length task by memorising patterns.
- But it has no mechanism to *compress* history into a finite state, so L* extraction may yield bloated automata or fail to converge.
- Length generalisation is expected to be poor because the positional embeddings are trained on lengths 3-40 only.

## Hyperparameters

```json
{
  "model_type": "transformer",
  "num_layers": 2,
  "embed_size": 128,
  "n_heads": 4,
  "d_ff": 512,
  "max_pos": 512,
  "dropout": 0.0
}
```

### Parameter count estimate

Per TransformerBlock:
- Attention: 4 x Linear(128, 128) = 4 x 128 x 128 = 65,536
- FFN: Linear(128, 512) + Linear(512, 128) = 128 x 512 + 512 x 128 = 131,072
- LayerNorms: 2 x 2 x 128 = 512
- Per block total: ~197K

Full model:
- Embedding: vocab_size x 128 (384 for parity, 1152 for cycle_nav)
- Positional: 512 x 128 = 65,536
- 2 blocks: ~394K
- Readout: 128 x vocab_size
- **Total: ~460K** (parity) — comparable to PD-SSM's ~397K

### Training

Same protocol as PD-SSM:
- AdamW, lr=0.002, weight_decay=0.01
- Warmup 10% + cosine decay
- Early stop at 99.95% validation accuracy
- Train lengths 3-40, validate 40-256
- 3 seeds per task

## Dependencies

- `jax`, `equinox`, `optax` — already installed, no new packages needed.

## Files to create

```
state_tracking_transformer_JAX/
  models/
    transformer.py      — StateTrackingTransformer, TransformerBlock, CausalSelfAttention
  train.py              — copy of state_tracking_JAX/train.py, import swapped to transformer
  evaluate.py           — copy of state_tracking_JAX/evaluate.py, import swapped
  experiment_configs/
    parity_transformer_0.json
    parity_transformer_1.json
    parity_transformer_2.json
    even_pairs_transformer_0.json
    ...                 — 4 tasks x 3 seeds = 12 configs
```

## Expected behaviour under L*

| Task | Expected outcome |
|------|-----------------|
| parity | May learn correct behaviour on training lengths but fail to generalise. L* likely extracts a correct-ish automaton if the model generalises, otherwise a bloated one. |
| even_pairs | Only needs to remember first token — attention can do this easily. Should extract a small automaton. |
| cycle_nav | Requires counting modular position — attention has no built-in counter. Likely struggles on long sequences. |
| mod_arith | Most complex task. Transformer may handle short sequences but L* extraction will reveal whether it learned a clean computational structure or memorised. |
