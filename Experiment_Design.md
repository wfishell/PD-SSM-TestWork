# Experiment Design: Automata Extraction from Sequence Models

## Goal

Train four sequence model architectures on formal language (regular language) tasks, then use L* with PAC-closeness equivalence queries to extract Moore machines from each trained model. Compare:

1. Whether each architecture learns a correct automaton
2. How many states the extracted automaton has (minimality)
3. How many L* rounds / oracle queries are needed
4. Whether the extracted automaton generalises to longer sequences

## Models

| Model | Type | Key property |
|-------|------|-------------|
| **PD-SSM** | Parametrised diagonal SSM | Structured sparse A(u) = P(u)D(u), closed under composition |
| **S4** | Diagonal SSM (S4D variant) | Fixed A matrix, input-independent dynamics |
| **Mamba** | Selective SSM | Input-dependent A,B,C via selection mechanism |
| **Transformer** | Causal self-attention | No recurrent state, attends to full history |

### Implementation plan

- **PD-SSM**: Already implemented (`models/pdssm.py`). No changes needed.
- **S4 (S4D)**: Implement in JAX/Equinox. Diagonal state-space model with learnable (A, B, C, D) where A is a fixed complex diagonal. Use the same parallel scan infrastructure. This is effectively PD-SSM without input-dependent transitions.
- **Mamba**: Implement the selective SSM in JAX/Equinox. Input-dependent B, C, and discretisation step delta. Simplified version sufficient — no hardware-aware scan needed for these short sequences.
- **Transformer**: Implement a small causal Transformer in JAX/Equinox. Causal multi-head attention + FFN blocks. Same embedding/readout interface.

All models share the same `StateTracking*` wrapper pattern:
```
Embedding(vocab_size, H) → [Block] × num_layers → Linear(H, label_dim)
```

## Tasks

All four formal language tasks from the existing codebase:

| Task | Alphabet | States in minimal DFA | Output labels |
|------|----------|----------------------|---------------|
| **parity** | {a, b} | 2 | even (1), odd (2) |
| **even_pairs** | {a, b} | 4 | match (1), diff (2) |
| **cycle_nav** | {S, +, -} | 5 | positions 0-4 (labels 4-8) |
| **mod_arith** | {+,-,*,=,0,1,2,3,4} | ≥25 (structured) | residues 0-4 (labels 5-9) |

Training protocol (same for all models):
- Train on sequences of length 3-40
- Validate on sequences of length 40-256 (length generalisation)
- AdamW, warmup + cosine decay
- Early stop at 99.95% validation accuracy

## Hyperparameters

Match parameter count across models as closely as possible so comparisons are fair.

### Shared across all models
```json
{
  "num_layers": 2,
  "embed_size": 128,
  "state_size": 128,
  "batch_size": 512,
  "num_steps": 100001,
  "learning_rate": 0.002,
  "weight_decay": 0.01,
  "warmup_fraction": 0.1
}
```

### Model-specific
- **PD-SSM**: K=6 (dictionary size) — existing configs
- **S4D**: dt_min=0.001, dt_max=0.1 (discretisation range)
- **Mamba**: d_inner=256, dt_rank=16, d_conv=4
- **Transformer**: n_heads=4, d_ff=512, dropout=0.0

### Seeds
3 seeds per (model, task) pair: seeds 0, 1, 2.

## L* Extraction Protocol

After training, extract a Moore machine from each checkpoint:

```bash
python learn_automaton.py -c <config> --checkpoint <path> \
    --epsilon 0.01 --delta 0.01 --max_eq_length 64 --max_rounds 50 \
    --exact_check_max_length 10 --dot <output.dot>
```

**PAC parameters**: epsilon=0.01, delta=0.01 → ~530 samples per equivalence query. With geometric delta-splitting across rounds, this guarantees that with probability >= 99%, the learned automaton disagrees with the oracle on at most 1% of the uniform distribution.

## Metrics

### Training metrics (per model, per task, averaged over seeds)
- Final validation accuracy (length 40-256)
- Steps to convergence (99.95% threshold)
- Parameter count

### Automata extraction metrics (per model, per task, averaged over seeds)
- **Correctness**: Does the extracted automaton match the ground-truth minimal DFA? (exact check up to length 10)
- **Number of states**: How many states in the extracted Moore machine? (fewer = cleaner representation)
- **L* rounds**: How many equivalence query rounds before convergence?
- **Oracle queries**: Total membership queries to the neural model
- **Extraction time**: Wall-clock time for L*

### Expected outcomes
- **PD-SSM**: Should extract minimal or near-minimal automata (structured transitions → clean state space)
- **S4D**: May struggle on tasks requiring input-dependent dynamics; extracted automaton likely has extra states or fails
- **Mamba**: Input-dependent but less structured than PD-SSM; may extract correct but non-minimal automata
- **Transformer**: No recurrent state — unclear if L* can extract a compact automaton; may need many states or fail to converge

## Directory structure

```
state_tracking_JAX/
  models/
    pdssm.py          # existing
    s4d.py             # new
    mamba.py           # new
    transformer.py     # new
    generate_model.py  # new: factory function for all models
  experiment_configs/
    {task}_{model}_{seed}.json   # e.g. parity_s4d_0.json
  train.py             # generalise to accept model_type in config
  learn_automaton.py   # existing (already task-agnostic)
  run_all.py           # new: batch runner for all experiments
  results/             # training curves
  checkpoints/         # .eqx files
  automata/            # .dot files from L* extraction
```

## Execution order

1. Implement S4D, Mamba, Transformer models (same Equinox interface)
2. Add `generate_model.py` factory and update `train.py` to dispatch on `model_type`
3. Generate config JSONs for all (model, task, seed) combinations
4. Train all models (4 models x 4 tasks x 3 seeds = 48 runs)
5. Run L* extraction on all converged checkpoints
6. Collect and compare metrics
