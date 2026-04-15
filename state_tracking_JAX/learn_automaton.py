"""
Task-agnostic L* automata learner for PD-SSM checkpoints.

Treats a trained PD-SSM (.eqx checkpoint) as a black-box oracle and uses the
L* algorithm with PAC-closeness equivalence queries to extract a Moore machine
(a DFA where every state carries an output label).

Usage:
    python learn_automaton.py -c parity_0 \
        --checkpoint checkpoints/model_parity_pdssm_parity_0.eqx \
        --epsilon 0.01 --delta 0.01 --max_rounds 25

Works with any state-tracking task (parity, cycle_nav, even_pairs,
mod_arith_no_brack) — the alphabet and label space are read from the task
module automatically.
"""

import argparse
import importlib
import json
import math
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from models.pdssm import StateTrackingPDSSM


# ---------------------------------------------------------------------------
# Oracle: wraps a trained PD-SSM checkpoint
# ---------------------------------------------------------------------------

@eqx.filter_jit
def predict_last_logits(model, tokens):
    """Run model on a single token sequence, return logits at the last position."""
    logits = model(tokens)
    return logits[-1]


class NeuralOracle:
    """Black-box oracle backed by a PD-SSM checkpoint.

    For a non-empty word, the oracle returns argmax of the model's last-position
    logits.  For the empty word the oracle returns *empty_label*, which is the
    ground-truth output for the empty string under the task semantics (see
    ``default_empty_label``).
    """

    def __init__(self, model, empty_label):
        self.model = model
        self.empty_label = empty_label
        self.cache = {}
        self.membership_queries = 0

    def query(self, word):
        """Return the integer output label for *word*."""
        key = tuple(word)
        if key in self.cache:
            return self.cache[key]

        if not key:
            label = self.empty_label
        else:
            tokens = jnp.array(key, dtype=jnp.int32)
            logits = predict_last_logits(self.model, tokens)
            label = int(jnp.argmax(logits))

        self.cache[key] = label
        self.membership_queries += 1
        return label


# ---------------------------------------------------------------------------
# Task metadata helpers
# ---------------------------------------------------------------------------

# Ground-truth label for the empty string under each task's semantics.
_EMPTY_LABELS = {
    "parity": 1,            # zero b's → even → label 1
    "cycle_nav": 4,         # position 0 → label 4
    "even_pairs": 1,        # vacuously first==last → label 1
    "mod_arith_no_brack": 5,  # empty expression → 0 mod 5 → label 5
}

# Human-readable names for input tokens, per task.
_SYMBOL_NAMES = {
    "parity": {1: "a", 2: "b"},
    "even_pairs": {1: "a", 2: "b"},
    "cycle_nav": {1: "S", 2: "+", 3: "-"},
    "mod_arith_no_brack": {
        1: "+", 2: "-", 3: "*", 4: "=",
        5: "0", 6: "1", 7: "2", 8: "3", 9: "4",
    },
}

# Human-readable names for output labels, per task.
_LABEL_NAMES = {
    "parity": {1: "even", 2: "odd"},
    "even_pairs": {1: "match", 2: "diff"},
    "cycle_nav": {4: "p0", 5: "p1", 6: "p2", 7: "p3", 8: "p4"},
    "mod_arith_no_brack": {5: "0", 6: "1", 7: "2", 8: "3", 9: "4"},
}


def default_empty_label(task):
    return _EMPTY_LABELS.get(task, 1)


def symbol_name(task, tok):
    return _SYMBOL_NAMES.get(task, {}).get(tok, str(tok))


def label_name(task, lab):
    return _LABEL_NAMES.get(task, {}).get(lab, str(lab))


def format_word(task, word):
    if not word:
        return "ε"
    return "".join(symbol_name(task, s) for s in word)


# ---------------------------------------------------------------------------
# Observation table (multi-class generalisation of standard L*)
# ---------------------------------------------------------------------------

class ObservationTable:
    """L* observation table that stores output labels (integers) instead of
    accept/reject booleans.  Two prefixes are Nerode-equivalent iff they yield
    the same label for every experiment suffix."""

    def __init__(self, oracle, alphabet):
        self.oracle = oracle
        self.alphabet = tuple(sorted(alphabet))
        self.S = [()]           # prefix-closed set of access strings
        self.E = [()]           # suffix-closed set of experiments

    def output(self, word):
        return self.oracle.query(word)

    def row(self, prefix):
        return tuple(self.output(prefix + suffix) for suffix in self.E)

    # -- S·Σ extensions not already in S -----------------------------------

    def s_extensions(self):
        extensions = []
        seen = set(self.S)
        for prefix in self.S:
            for symbol in self.alphabet:
                candidate = prefix + (symbol,)
                if candidate not in seen:
                    seen.add(candidate)
                    extensions.append(candidate)
        return extensions

    # -- Closedness check --------------------------------------------------

    def find_unclosed(self):
        s_rows = {self.row(p) for p in self.S}
        for candidate in self.s_extensions():
            if self.row(candidate) not in s_rows:
                return candidate
        return None

    # -- Consistency check -------------------------------------------------

    def find_inconsistency(self):
        for i, left in enumerate(self.S):
            left_row = self.row(left)
            for right in self.S[i + 1:]:
                if left_row != self.row(right):
                    continue
                for symbol in self.alphabet:
                    left_ext = self.row(left + (symbol,))
                    right_ext = self.row(right + (symbol,))
                    if left_ext == right_ext:
                        continue
                    for suffix in self.E:
                        if self.output(left + (symbol,) + suffix) != \
                                self.output(right + (symbol,) + suffix):
                            return (symbol,) + suffix
        return None

    # -- Add counter-example (all prefixes) --------------------------------

    def add_counterexample(self, word):
        for end in range(len(word) + 1):
            prefix = tuple(word[:end])
            if prefix not in self.S:
                self.S.append(prefix)
        self.S.sort(key=lambda p: (len(p), p))

    # -- Main table-repair loop --------------------------------------------

    def close_and_make_consistent(self):
        while True:
            unclosed = self.find_unclosed()
            if unclosed is not None:
                self.S.append(unclosed)
                self.S.sort(key=lambda p: (len(p), p))
                continue

            witness = self.find_inconsistency()
            if witness is not None and witness not in self.E:
                self.E.append(witness)
                self.E.sort(key=lambda s: (len(s), s))
                continue
            return


# ---------------------------------------------------------------------------
# Moore machine (DFA with output labels on states)
# ---------------------------------------------------------------------------

@dataclass
class MooreMachine:
    states: tuple
    alphabet: tuple
    start_state: int
    transitions: dict            # (state, symbol) → state
    output_map: dict             # state → label
    representatives: dict        # state → access string (tuple of tokens)

    def trace(self, word):
        state = self.start_state
        for symbol in word:
            state = self.transitions[(state, symbol)]
        return state

    def output(self, word):
        return self.output_map[self.trace(word)]


def build_hypothesis(table):
    """Construct a Moore machine from a closed, consistent observation table."""
    row_to_state = {}
    representatives = {}

    for prefix in table.S:
        row = table.row(prefix)
        if row not in row_to_state:
            state = len(row_to_state)
            row_to_state[row] = state
            representatives[state] = prefix

    transitions = {}
    output_map = {}
    for prefix in table.S:
        state = row_to_state[table.row(prefix)]
        output_map[state] = table.output(prefix)
        for symbol in table.alphabet:
            next_row = table.row(prefix + (symbol,))
            transitions[(state, symbol)] = row_to_state[next_row]

    start_state = row_to_state[table.row(())]
    states = tuple(sorted(representatives))
    return MooreMachine(
        states=states,
        alphabet=table.alphabet,
        start_state=start_state,
        transitions=transitions,
        output_map=output_map,
        representatives=representatives,
    )


# ---------------------------------------------------------------------------
# PAC equivalence query (uniform random sampling)
# ---------------------------------------------------------------------------

def pac_equivalence_query(hypothesis, oracle, rng, epsilon, delta,
                          min_length, max_length, num_samples=None):
    """Draw uniform random words and check hypothesis vs oracle.

    Returns (counterexample, num_samples_drawn) where counterexample is None
    when no disagreement was found.  The PAC guarantee: if None is returned,
    then with probability ≥ 1-δ the hypothesis disagrees with the oracle on
    at most an ε fraction of the uniform distribution over words of length
    min_length..max_length.
    """
    if num_samples is None:
        num_samples = math.ceil(math.log(1.0 / delta) / epsilon)
        num_samples = max(num_samples, 1)

    alphabet = list(hypothesis.alphabet)
    for _ in range(num_samples):
        length = int(rng.integers(min_length, max_length + 1))
        word = tuple(rng.choice(alphabet, size=length).tolist())
        if hypothesis.output(word) != oracle.query(word):
            return word, num_samples

    return None, num_samples


# ---------------------------------------------------------------------------
# Exact check (brute-force up to a length bound)
# ---------------------------------------------------------------------------

def exact_check(hypothesis, oracle, alphabet, max_length):
    """Enumerate all words up to *max_length* and compare outputs."""
    checked = 0
    alphabet = list(alphabet)

    # BFS over words
    frontier = [()]
    while frontier:
        next_frontier = []
        for word in frontier:
            if hypothesis.output(word) != oracle.query(word):
                return False, word, checked + 1
            checked += 1
            if len(word) < max_length:
                for symbol in alphabet:
                    next_frontier.append(word + (symbol,))
        frontier = next_frontier

    return True, None, checked


# ---------------------------------------------------------------------------
# L* main loop
# ---------------------------------------------------------------------------

def learn_moore_machine(oracle, alphabet, epsilon, delta,
                        min_eq_length, max_eq_length, max_rounds,
                        eq_samples=None, verbose=True):
    table = ObservationTable(oracle=oracle, alphabet=alphabet)
    pac_samples_used = 0
    rng = np.random.default_rng(0)

    for round_idx in range(max_rounds):
        table.close_and_make_consistent()
        hypothesis = build_hypothesis(table)

        if verbose:
            print(f"  round {round_idx + 1}: hypothesis has "
                  f"{len(hypothesis.states)} states, "
                  f"|S|={len(table.S)}, |E|={len(table.E)}")

        round_delta = delta / (2 ** (round_idx + 1))
        counterexample, samples_this_round = pac_equivalence_query(
            hypothesis=hypothesis,
            oracle=oracle,
            rng=rng,
            epsilon=epsilon,
            delta=round_delta,
            min_length=min_eq_length,
            max_length=max_eq_length,
            num_samples=eq_samples,
        )
        pac_samples_used += samples_this_round

        if counterexample is None:
            return hypothesis, table, round_idx + 1, pac_samples_used

        if verbose:
            print(f"    counterexample (len={len(counterexample)}): "
                  f"hyp={hypothesis.output(counterexample)}, "
                  f"oracle={oracle.query(counterexample)}")

        table.add_counterexample(counterexample)

    raise RuntimeError(
        f"Failed to converge within {max_rounds} L* rounds."
    )


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_moore_machine(mm, task):
    print("\nLearned Moore Machine")
    print("---------------------")
    for state in mm.states:
        flags = []
        if state == mm.start_state:
            flags.append("start")
        lab = mm.output_map[state]
        flags.append(f"output={label_name(task, lab)} ({lab})")
        rep = format_word(task, mm.representatives[state])
        print(f"  q{state}: {', '.join(flags)}, representative={rep}")

    print("\nTransitions")
    for state in mm.states:
        targets = []
        for symbol in mm.alphabet:
            dest = mm.transitions[(state, symbol)]
            targets.append(f"{symbol_name(task, symbol)}→q{dest}")
        print(f"  q{state}: {', '.join(targets)}")


def export_dot(mm, task, path):
    """Write a Graphviz DOT file for the learned Moore machine."""
    with open(path, "w") as f:
        f.write("digraph MooreMachine {\n")
        f.write("  rankdir=LR;\n")
        f.write('  node [shape=circle];\n')
        f.write(f'  __start__ [shape=point];\n')
        f.write(f'  __start__ -> q{mm.start_state};\n')

        for state in mm.states:
            lab = label_name(task, mm.output_map[state])
            f.write(f'  q{state} [label="q{state}\\n[{lab}]"];\n')

        for state in mm.states:
            for symbol in mm.alphabet:
                dest = mm.transitions[(state, symbol)]
                sym = symbol_name(task, symbol)
                f.write(f'  q{state} -> q{dest} [label="{sym}"];\n')

        f.write("}\n")
    print(f"\nDOT file written to {path}")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_oracle_model(config_path, checkpoint_path):
    with open(config_path) as f:
        config = json.load(f)

    task = config["task"]
    module = importlib.import_module(f"data_dir.fl_tasks.{task}")
    vocab_size = module.vocab_size

    model = StateTrackingPDSSM(
        vocab_size=vocab_size,
        label_dim=vocab_size,
        N=config["state_size"],
        H=config["embed_size"],
        num_layers=config["num_layers"],
        K=config.get("dictionary_size", 6),
        key=jr.PRNGKey(0),
    )
    model = eqx.tree_deserialise_leaves(checkpoint_path, model)
    return model, config, task, vocab_size


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Learn a Moore machine from a trained PD-SSM checkpoint using "
            "L* with PAC-closeness equivalence queries."
        )
    )
    parser.add_argument(
        "-c", "--config", required=True,
        help="Config name in experiment_configs/ (without .json)",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the .eqx checkpoint",
    )
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--min_eq_length", type=int, default=1)
    parser.add_argument("--max_eq_length", type=int, default=64)
    parser.add_argument("--max_rounds", type=int, default=50)
    parser.add_argument(
        "--eq_samples", type=int, default=None,
        help="Override the PAC sample count per equivalence query",
    )
    parser.add_argument(
        "--exact_check_max_length", type=int, default=8,
        help="Exhaustive post-check length bound; set < 0 to skip",
    )
    parser.add_argument(
        "--dot", type=str, default=None,
        help="Path to write a Graphviz DOT file of the learned machine",
    )
    args = parser.parse_args()

    config_path = f"experiment_configs/{args.config}.json"
    model, config, task, vocab_size = load_oracle_model(
        config_path, args.checkpoint,
    )

    alphabet = tuple(range(1, vocab_size))  # exclude pad token 0
    empty_label = default_empty_label(task)

    print(f"Task: {task}")
    print(f"Alphabet: {{{', '.join(symbol_name(task, s) for s in alphabet)}}}")
    print(f"Empty-string label: {label_name(task, empty_label)} ({empty_label})")
    print(f"PAC parameters: ε={args.epsilon}, δ={args.delta}")
    print()

    oracle = NeuralOracle(model, empty_label=empty_label)

    hypothesis, table, eq_rounds, pac_samples_used = learn_moore_machine(
        oracle=oracle,
        alphabet=alphabet,
        epsilon=args.epsilon,
        delta=args.delta,
        min_eq_length=args.min_eq_length,
        max_eq_length=args.max_eq_length,
        max_rounds=args.max_rounds,
        eq_samples=args.eq_samples,
    )

    print_moore_machine(hypothesis, task)

    print("\nLearning summary")
    print("----------------")
    print(f"  Observation table prefixes (|S|): {len(table.S)}")
    print(f"  Observation table suffixes (|E|): {len(table.E)}")
    print(f"  States in learned machine:        {len(hypothesis.states)}")
    print(f"  PAC equivalence rounds:           {eq_rounds}")
    print(f"  PAC samples used:                 {pac_samples_used}")
    print(f"  Oracle membership queries:        {oracle.membership_queries}")

    if args.exact_check_max_length >= 0:
        matches, witness, checked = exact_check(
            hypothesis=hypothesis,
            oracle=oracle,
            alphabet=alphabet,
            max_length=args.exact_check_max_length,
        )
        status = "PASSED" if matches else "FAILED"
        print(f"\n  Exact check (len ≤ {args.exact_check_max_length}): "
              f"{status} over {checked} strings")
        if not matches:
            print(f"  First mismatch: {format_word(task, witness)}")

    if args.dot:
        export_dot(hypothesis, task, args.dot)


if __name__ == "__main__":
    main()
