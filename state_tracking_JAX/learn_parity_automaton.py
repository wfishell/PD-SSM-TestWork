import argparse
import importlib
import json
import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from models.pdssm import StateTrackingPDSSM


ALPHABET = (1, 2)
SYMBOL_NAMES = {1: "a", 2: "b"}
ACCEPT_LABEL = 1


def format_word(word):
    if not word:
        return "eps"
    return "".join(SYMBOL_NAMES[symbol] for symbol in word)


@eqx.filter_jit
def predict_last_logits(model, tokens):
    logits = model(tokens)
    return logits[-1]


def load_oracle_model(config_path, checkpoint_path):
    with open(config_path) as f:
        config = json.load(f)
    task = config["task"]

    if task != "parity":
        raise ValueError(
            f"This learner is implemented for the parity task only, got task={task!r}."
        )

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
    return model, config


class NeuralParityOracle:
    def __init__(self, model):
        self.model = model
        self.cache = {}
        self.membership_queries = 0

    def accepts(self, word):
        key = tuple(word)
        if key in self.cache:
            return self.cache[key]

        # The deployed checkpoint is trained on non-empty strings. We anchor eps
        # to the exact parity language and use the model for every sampled query.
        if not key:
            accepted = True
        else:
            tokens = jnp.array(key, dtype=jnp.int32)
            logits = predict_last_logits(self.model, tokens)
            accepted = int(jnp.argmax(logits)) == ACCEPT_LABEL

        self.cache[key] = bool(accepted)
        self.membership_queries += 1
        return self.cache[key]


class ObservationTable:
    def __init__(self, oracle, alphabet):
        self.oracle = oracle
        self.alphabet = tuple(alphabet)
        self.S = [tuple()]
        self.E = [tuple()]

    def membership(self, word):
        return self.oracle.accepts(word)

    def row(self, prefix):
        return tuple(self.membership(prefix + suffix) for suffix in self.E)

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

    def find_unclosed(self):
        s_rows = {self.row(prefix) for prefix in self.S}
        for candidate in self.s_extensions():
            if self.row(candidate) not in s_rows:
                return candidate
        return None

    def find_inconsistency(self):
        for i, left in enumerate(self.S):
            left_row = self.row(left)
            for right in self.S[i + 1 :]:
                if left_row != self.row(right):
                    continue
                for symbol in self.alphabet:
                    left_next = self.row(left + (symbol,))
                    right_next = self.row(right + (symbol,))
                    if left_next == right_next:
                        continue
                    for suffix in self.E:
                        if self.membership(left + (symbol,) + suffix) != self.membership(
                            right + (symbol,) + suffix
                        ):
                            return (symbol,) + suffix
        return None

    def add_counterexample(self, word):
        for end in range(len(word) + 1):
            prefix = tuple(word[:end])
            if prefix not in self.S:
                self.S.append(prefix)
        self.S.sort(key=lambda prefix: (len(prefix), prefix))

    def close_and_make_consistent(self):
        while True:
            unclosed = self.find_unclosed()
            if unclosed is not None:
                self.S.append(unclosed)
                self.S.sort(key=lambda prefix: (len(prefix), prefix))
                continue

            witness = self.find_inconsistency()
            if witness is not None and witness not in self.E:
                self.E.append(witness)
                self.E.sort(key=lambda suffix: (len(suffix), suffix))
                continue
            return


@dataclass
class DFA:
    states: tuple
    start_state: int
    accepting_states: frozenset
    transitions: dict
    representatives: dict

    def accepts(self, word):
        state = self.start_state
        for symbol in word:
            state = self.transitions[(state, symbol)]
        return state in self.accepting_states


def build_hypothesis(table):
    row_to_state = {}
    representatives = {}

    for prefix in table.S:
        row = table.row(prefix)
        if row not in row_to_state:
            state = len(row_to_state)
            row_to_state[row] = state
            representatives[state] = prefix

    transitions = {}
    accepting_states = set()
    for prefix in table.S:
        state = row_to_state[table.row(prefix)]
        if table.membership(prefix):
            accepting_states.add(state)
        for symbol in table.alphabet:
            next_row = table.row(prefix + (symbol,))
            transitions[(state, symbol)] = row_to_state[next_row]

    start_state = row_to_state[table.row(tuple())]
    states = tuple(sorted(representatives))
    return DFA(
        states=states,
        start_state=start_state,
        accepting_states=frozenset(accepting_states),
        transitions=transitions,
        representatives=representatives,
    )


def sample_balanced_parity_word(rng, min_length, max_length, accept):
    length = int(rng.integers(min_length, max_length + 1))
    if length <= 0:
        return tuple()

    if length == 1:
        return (1,) if accept else (2,)

    prefix = rng.integers(1, 3, size=length - 1).astype(np.int32)
    odd_prefix = int((prefix == 2).sum()) % 2 == 1

    if accept:
        last_symbol = 2 if odd_prefix else 1
    else:
        last_symbol = 1 if odd_prefix else 2

    return tuple(prefix.tolist() + [last_symbol])


def pac_equivalence_query(
    hypothesis,
    oracle,
    rng,
    epsilon,
    delta,
    min_length,
    max_length,
    num_samples=None,
):
    if num_samples is None:
        num_samples = math.ceil(math.log(1.0 / delta) / epsilon)
        num_samples = max(num_samples, 1)

    for sample_idx in range(num_samples):
        target_accept = sample_idx % 2 == 0
        word = sample_balanced_parity_word(
            rng=rng,
            min_length=min_length,
            max_length=max_length,
            accept=target_accept,
        )
        if hypothesis.accepts(word) != oracle.accepts(word):
            return word, num_samples

    return None, num_samples


def exact_check(hypothesis, oracle, max_length):
    checked = 0
    for length in range(max_length + 1):
        for mask in range(1 << length):
            word = tuple(2 if (mask >> i) & 1 else 1 for i in range(length))
            if hypothesis.accepts(word) != oracle.accepts(word):
                return False, word, checked + 1
            checked += 1
    return True, None, checked


def learn_dfa(
    oracle,
    epsilon,
    delta,
    min_eq_length,
    max_eq_length,
    max_rounds,
    eq_samples=None,
):
    table = ObservationTable(oracle=oracle, alphabet=ALPHABET)
    pac_samples_used = 0
    rounds = 0
    rng = np.random.default_rng(0)

    while rounds < max_rounds:
        table.close_and_make_consistent()
        hypothesis = build_hypothesis(table)

        round_delta = delta / (2 ** (rounds + 1))
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
            return hypothesis, table, rounds + 1, pac_samples_used

        table.add_counterexample(counterexample)
        rounds += 1

    raise RuntimeError(
        f"Failed to find a PAC-close DFA within {max_rounds} learning rounds."
    )


def print_dfa(dfa):
    print("\nLearned DFA")
    print("-----------")
    for state in dfa.states:
        flags = []
        if state == dfa.start_state:
            flags.append("start")
        if state in dfa.accepting_states:
            flags.append("accept")
        label = ", ".join(flags) if flags else "internal"
        rep = format_word(dfa.representatives[state])
        print(f"q{state}: {label}, representative={rep}")
    print("\nTransitions")
    for state in dfa.states:
        for symbol in ALPHABET:
            dest = dfa.transitions[(state, symbol)]
            print(f"q{state} --{SYMBOL_NAMES[symbol]}--> q{dest}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Learn a DFA for the parity task by treating a deployed JAX .eqx "
            "checkpoint as a membership oracle and using PAC-style equivalence queries."
        )
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Config name in experiment_configs/ (without .json)",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the deployed .eqx checkpoint",
    )
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--min_eq_length", type=int, default=1)
    parser.add_argument("--max_eq_length", type=int, default=64)
    parser.add_argument("--max_rounds", type=int, default=25)
    parser.add_argument(
        "--eq_samples",
        type=int,
        default=None,
        help="Override the PAC sample count per equivalence query",
    )
    parser.add_argument(
        "--exact_check_max_length",
        type=int,
        default=8,
        help="Exhaustive post-check length bound; set < 0 to skip",
    )
    args = parser.parse_args()

    config_path = f"experiment_configs/{args.config}.json"
    model, _ = load_oracle_model(config_path, args.checkpoint)
    oracle = NeuralParityOracle(model)

    hypothesis, table, eq_rounds, pac_samples_used = learn_dfa(
        oracle=oracle,
        epsilon=args.epsilon,
        delta=args.delta,
        min_eq_length=args.min_eq_length,
        max_eq_length=args.max_eq_length,
        max_rounds=args.max_rounds,
        eq_samples=args.eq_samples,
    )

    print_dfa(hypothesis)
    print("\nLearning summary")
    print("----------------")
    print(f"Observation table prefixes: {len(table.S)}")
    print(f"Observation table suffixes: {len(table.E)}")
    print(f"States in learned DFA: {len(hypothesis.states)}")
    print(f"PAC equivalence rounds: {eq_rounds}")
    print(f"PAC samples used: {pac_samples_used}")
    print(f"Oracle membership queries: {oracle.membership_queries}")

    if args.exact_check_max_length >= 0:
        matches, witness, checked = exact_check(
            hypothesis=hypothesis,
            oracle=oracle,
            max_length=args.exact_check_max_length,
        )
        print(
            f"Exact check up to length {args.exact_check_max_length}: "
            f"{'passed' if matches else 'failed'} over {checked} strings"
        )
        if not matches:
            print(f"First exact-check mismatch: {format_word(witness)}")


if __name__ == "__main__":
    main()
