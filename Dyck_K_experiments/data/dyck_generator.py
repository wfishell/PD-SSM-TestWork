"""Depth-bounded Dyck-k sampler.

Tokens are integers in [0, 2k):
    open_i  = i           for i in [0, k)
    close_i = k + i       for i in [0, k)
Bracket i closes bracket i.

Positive samples are drawn uniformly from the set of length-L Dyck-k
strings whose stack never exceeds max_depth. Negatives come from edit
perturbations (default) or uniform-random strings over the alphabet;
both are filtered through is_valid so returned negatives are guaranteed
to be outside the language.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

NegativeStrategy = Literal["edit", "random"]


class DyckGenerator:
    def __init__(self, k: int, max_depth: int, seed: int | None = 0):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        self.k = int(k)
        self.max_depth = int(max_depth)
        self.vocab_size = 2 * self.k
        self.rng = np.random.default_rng(seed)
        self._counts: list[list[int]] = []
        self._counts_built_up_to = -1

    # ------------------------------------------------------------------
    # Count table: counts[r][s] = number of valid completions with r
    # tokens remaining and current stack depth s, under the depth bound.
    # ------------------------------------------------------------------
    def _ensure_counts(self, max_len: int) -> None:
        if max_len <= self._counts_built_up_to:
            return
        D = self.max_depth
        counts: list[list[int]] = [[0] * (D + 1) for _ in range(max_len + 1)]
        counts[0][0] = 1
        for r in range(1, max_len + 1):
            row = counts[r]
            prev = counts[r - 1]
            for s in range(D + 1):
                total = 0
                if s < D:
                    total += self.k * prev[s + 1]
                if s > 0:
                    total += prev[s - 1]
                row[s] = total
        self._counts = counts
        self._counts_built_up_to = max_len

    def count_strings(self, length: int) -> int:
        """Number of depth-bounded Dyck-k strings of the given length."""
        if length < 0 or length % 2 != 0:
            return 0
        self._ensure_counts(length)
        return self._counts[length][0]

    # ------------------------------------------------------------------
    # Validity check.
    # ------------------------------------------------------------------
    def is_valid(self, tokens: np.ndarray | list[int]) -> bool:
        stack: list[int] = []
        for t in tokens:
            t = int(t)
            if t < 0 or t >= 2 * self.k:
                return False
            if t < self.k:
                stack.append(t)
                if len(stack) > self.max_depth:
                    return False
            else:
                if not stack or stack[-1] != t - self.k:
                    return False
                stack.pop()
        return not stack

    # ------------------------------------------------------------------
    # Positive sampling: exactly uniform over depth-bounded Dyck-k
    # strings of the given length.
    # ------------------------------------------------------------------
    def sample_positive(self, length: int) -> np.ndarray:
        if length % 2 != 0:
            raise ValueError(f"Dyck length must be even, got {length}")
        if self.count_strings(length) == 0:
            raise ValueError(
                f"No Dyck-{self.k} strings of length {length} "
                f"with depth <= {self.max_depth}"
            )

        tokens = np.empty(length, dtype=np.int32)
        stack: list[int] = []
        for i in range(length):
            r = length - i
            s = len(stack)
            n_open = self.k * self._counts[r - 1][s + 1] if s < self.max_depth else 0
            n_close = self._counts[r - 1][s - 1] if s > 0 else 0
            total = n_open + n_close
            # total is always > 0 here because counts[length][0] > 0
            # and we only ever move along supported paths.
            if self.rng.random() * total < n_open:
                bracket = int(self.rng.integers(self.k))
                tokens[i] = bracket
                stack.append(bracket)
            else:
                bracket = stack.pop()
                tokens[i] = self.k + bracket
        return tokens

    # ------------------------------------------------------------------
    # Negative sampling.
    # ------------------------------------------------------------------
    def sample_negative(
        self,
        length: int,
        strategy: NegativeStrategy = "edit",
        max_tries: int = 200,
    ) -> np.ndarray:
        if strategy == "edit":
            return self._negative_by_edit(length, max_tries)
        if strategy == "random":
            return self._negative_by_random(length, max_tries)
        raise ValueError(f"Unknown negative strategy: {strategy!r}")

    def _negative_by_edit(self, length: int, max_tries: int) -> np.ndarray:
        if length < 2:
            raise ValueError("Edit-based negatives require length >= 2")
        edits = ["flip_direction", "swap"]
        if self.k > 1:
            edits.append("flip_type")

        for _ in range(max_tries):
            cand = self.sample_positive(length).copy()
            edit = edits[int(self.rng.integers(len(edits)))]
            if edit == "flip_direction":
                i = int(self.rng.integers(length))
                cand[i] = (int(cand[i]) + self.k) % (2 * self.k)
            elif edit == "flip_type":
                i = int(self.rng.integers(length))
                t = int(cand[i])
                if t < self.k:
                    new = (t + 1 + int(self.rng.integers(self.k - 1))) % self.k
                    cand[i] = new
                else:
                    old = t - self.k
                    new = (old + 1 + int(self.rng.integers(self.k - 1))) % self.k
                    cand[i] = self.k + new
            else:  # swap
                i, j = self.rng.choice(length, size=2, replace=False)
                cand[i], cand[j] = cand[j], cand[i]
            if not self.is_valid(cand):
                return cand
        raise RuntimeError(
            f"Failed to produce an edit-based negative after {max_tries} tries "
            f"(length={length}, k={self.k}, max_depth={self.max_depth})"
        )

    def _negative_by_random(self, length: int, max_tries: int) -> np.ndarray:
        for _ in range(max_tries):
            cand = self.rng.integers(0, 2 * self.k, size=length, dtype=np.int32)
            if not self.is_valid(cand):
                return cand
        raise RuntimeError(
            f"Failed to produce a random negative after {max_tries} tries"
        )

    # ------------------------------------------------------------------
    # Vectorized batch primitives. All ops use numpy on the full batch
    # dimension; the only remaining Python loop is over positions in the
    # sequence (which is fine — at most max_val_length ~= 256 iterations).
    # ------------------------------------------------------------------

    def _batch_is_valid(self, tokens: np.ndarray) -> np.ndarray:
        """Vectorized Dyck-k validity check. tokens: (B, L) -> (B,) bool."""
        B, L = tokens.shape
        stack = np.zeros((B, self.max_depth + 1), dtype=np.int32)
        depths = np.zeros(B, dtype=np.int64)
        alive = np.ones(B, dtype=bool)
        row = np.arange(B)

        for pos in range(L):
            if not alive.any():
                break
            t = tokens[:, pos]

            # Out-of-range tokens kill the row.
            alive &= (t >= 0) & (t < 2 * self.k)

            is_open = alive & (t < self.k)
            is_close = alive & (t >= self.k) & (t < 2 * self.k)

            # Depth overflow on an open.
            alive &= ~(is_open & (depths >= self.max_depth))
            is_open &= alive

            # Close with empty stack.
            alive &= ~(is_close & (depths == 0))
            is_close &= alive

            # Close whose top-of-stack doesn't match.
            peek_idx = np.maximum(depths - 1, 0)
            top = stack[row, peek_idx]
            alive &= ~(is_close & (top != (t - self.k)))
            is_close &= alive

            # Push on opens.
            if is_open.any():
                stack[row[is_open], depths[is_open]] = t[is_open]

            depths = depths + is_open.astype(np.int64) - is_close.astype(np.int64)

        return alive & (depths == 0)

    def batch_sample_positive(self, length: int, n: int) -> np.ndarray:
        """Vectorized uniform sampling of n depth-bounded Dyck-k strings of
        exactly ``length``. Same distribution as `sample_positive`."""
        if length % 2 != 0:
            raise ValueError(f"Dyck length must be even, got {length}")
        if self.count_strings(length) == 0:
            raise ValueError(
                f"No Dyck-{self.k} strings of length {length} "
                f"with depth <= {self.max_depth}"
            )
        if n == 0:
            return np.empty((0, length), dtype=np.int32)

        # counts_np[r, s] = # completions with r remaining tokens, depth s.
        counts_np = np.asarray(self._counts, dtype=np.float64)

        tokens = np.empty((n, length), dtype=np.int32)
        depths = np.zeros(n, dtype=np.int64)
        stack = np.zeros((n, self.max_depth + 1), dtype=np.int32)
        row = np.arange(n)

        for i in range(length):
            r = length - i

            can_open = depths < self.max_depth
            open_target = np.where(can_open, depths + 1, 0)
            n_open = counts_np[r - 1, open_target] * self.k
            n_open = np.where(can_open, n_open, 0.0)

            can_close = depths > 0
            close_target = np.where(can_close, depths - 1, 0)
            n_close = counts_np[r - 1, close_target]
            n_close = np.where(can_close, n_close, 0.0)

            total = n_open + n_close  # > 0 at every reachable (r, s)
            is_open = self.rng.random(size=n) * total < n_open

            open_types = self.rng.integers(self.k, size=n).astype(np.int32)
            peek_idx = np.maximum(depths - 1, 0)
            close_types = stack[row, peek_idx]

            tokens[:, i] = np.where(is_open, open_types, self.k + close_types)

            push_mask = is_open
            if push_mask.any():
                stack[row[push_mask], depths[push_mask]] = open_types[push_mask]
            depths = depths + is_open.astype(np.int64) - (~is_open).astype(np.int64)

        return tokens

    def batch_sample_negative(
        self, length: int, n: int,
        strategy: NegativeStrategy = "edit", max_tries: int = 50,
    ) -> np.ndarray:
        """Vectorized negative sampling of n non-Dyck strings of exact length."""
        if n == 0:
            return np.empty((0, length), dtype=np.int32)
        if length < 2:
            raise ValueError("Negatives require length >= 2")

        out = np.empty((n, length), dtype=np.int32)
        need = np.ones(n, dtype=bool)

        for _ in range(max_tries):
            remaining = int(need.sum())
            if remaining == 0:
                break

            if strategy == "edit":
                cand = self._batch_edit_candidates(length, remaining)
            elif strategy == "random":
                cand = self.rng.integers(
                    0, 2 * self.k, size=(remaining, length), dtype=np.int32
                )
            else:
                raise ValueError(f"Unknown negative strategy: {strategy!r}")

            valid = self._batch_is_valid(cand)
            invalid_mask = ~valid  # shape (remaining,)

            # Write the invalid candidates into slots that still need filling.
            idx_need = np.where(need)[0]
            take = invalid_mask
            accept_idx = idx_need[take]
            out[accept_idx] = cand[take]
            need[accept_idx] = False

        if need.any():
            raise RuntimeError(
                f"Failed to produce {int(need.sum())} negatives after "
                f"{max_tries} tries (length={length}, k={self.k}, "
                f"strategy={strategy!r})"
            )
        return out

    def _batch_edit_candidates(self, length: int, n: int) -> np.ndarray:
        """Generate n candidate negatives by edit-perturbing positives.
        Some candidates may still be valid Dyck strings; caller filters."""
        cand = self.batch_sample_positive(length, n).copy()

        n_edit_types = 3 if self.k > 1 else 2
        edit = self.rng.integers(n_edit_types, size=n)
        pos1 = self.rng.integers(length, size=n)
        pos2_raw = self.rng.integers(max(length - 1, 1), size=n)
        pos2 = np.where(pos2_raw >= pos1, pos2_raw + 1, pos2_raw)

        rows = np.arange(n)

        # Edit 0: flip direction at pos1.
        m0 = edit == 0
        if m0.any():
            idx = rows[m0]
            p = pos1[m0]
            cand[idx, p] = (cand[idx, p] + self.k) % (2 * self.k)

        # Edit 1: swap pos1 and pos2.
        m1 = edit == 1
        if m1.any():
            idx = rows[m1]
            p, q = pos1[m1], pos2[m1]
            a = cand[idx, p].copy()
            cand[idx, p] = cand[idx, q]
            cand[idx, q] = a

        # Edit 2 (only if k > 1): change bracket type at pos1, keep direction.
        if self.k > 1:
            m2 = edit == 2
            if m2.any():
                idx = rows[m2]
                p = pos1[m2]
                cur = cand[idx, p]
                direction = cur // self.k           # 0 = open, 1 = close
                old_type = cur % self.k
                offset = self.rng.integers(self.k - 1, size=len(idx)) + 1
                new_type = (old_type + offset) % self.k
                cand[idx, p] = direction * self.k + new_type

        return cand

    # ------------------------------------------------------------------
    # Batch generation.
    # ------------------------------------------------------------------
    def generate_dataset(
        self,
        n_samples: int,
        length_range: tuple[int, int],
        positive_ratio: float = 0.5,
        negative_strategy: NegativeStrategy = "edit",
        pad_value: int = -1,
    ) -> dict[str, np.ndarray]:
        """Batch generator.

        Lengths are sampled uniformly from the even integers in
        [length_range[0], length_range[1]]. Each sample is positive with
        probability positive_ratio. Shorter sequences are right-padded
        with pad_value up to the max length in the batch window.

        Returns a dict with keys:
            tokens:  (n_samples, max_len)   int32
            labels:  (n_samples,)           int32, 1=positive, 0=negative
            lengths: (n_samples,)           int32
        """
        lo, hi = length_range
        if lo < 0 or hi < lo:
            raise ValueError(f"Invalid length_range={length_range}")
        even_lo = lo if lo % 2 == 0 else lo + 1
        even_hi = hi if hi % 2 == 0 else hi - 1
        if even_lo > even_hi:
            raise ValueError(
                f"No even lengths in range {length_range}"
            )
        self._ensure_counts(even_hi)

        half_lo = even_lo // 2
        half_hi = even_hi // 2
        lengths = (2 * self.rng.integers(half_lo, half_hi + 1, size=n_samples)).astype(np.int32)
        is_positive = self.rng.random(size=n_samples) < positive_ratio
        labels = is_positive.astype(np.int32)

        tokens = np.full((n_samples, even_hi), pad_value, dtype=np.int32)

        # Group samples by (length, positive/negative) so the vectorized
        # batch primitives can sample a whole block at once.
        for L in np.unique(lengths):
            L_int = int(L)
            at_len = lengths == L

            pos_mask = at_len & is_positive
            n_pos = int(pos_mask.sum())
            if n_pos > 0:
                batch_pos = self.batch_sample_positive(L_int, n_pos)
                tokens[pos_mask, :L_int] = batch_pos

            neg_mask = at_len & ~is_positive
            n_neg = int(neg_mask.sum())
            if n_neg > 0:
                batch_neg = self.batch_sample_negative(
                    L_int, n_neg, strategy=negative_strategy,
                )
                tokens[neg_mask, :L_int] = batch_neg

        return {"tokens": tokens, "labels": labels, "lengths": lengths}


# ----------------------------------------------------------------------
# Pretty-printing helper.
# ----------------------------------------------------------------------
def format_sequence(tokens: np.ndarray, k: int) -> str:
    """Render a token array as a bracket string for humans."""
    opens = "([{<abcdefghij"
    closes = ")]}>ABCDEFGHIJ"
    if k > len(opens):
        return " ".join(str(int(t)) for t in tokens)
    out = []
    for t in tokens:
        t = int(t)
        if t < k:
            out.append(opens[t])
        else:
            out.append(closes[t - k])
    return "".join(out)


# ----------------------------------------------------------------------
# Smoke test: validates the generator against known facts.
# ----------------------------------------------------------------------
def _catalan(n: int) -> int:
    c = 1
    for i in range(n):
        c = c * (2 * (2 * i + 1)) // (i + 2)
    return c


def _self_test() -> None:
    # 1. Counts match Catalan * k^(L/2) when max_depth is effectively unbounded.
    for k in (1, 2, 3):
        gen = DyckGenerator(k=k, max_depth=128, seed=0)
        for L in (0, 2, 4, 6, 8, 10, 12):
            expected = _catalan(L // 2) * (k ** (L // 2))
            got = gen.count_strings(L)
            assert got == expected, (k, L, expected, got)
    print("[ok] count_strings matches Catalan * k^(L/2) for k=1,2,3 up to L=12")

    # 2. Depth bound actually bounds depth.
    gen = DyckGenerator(k=2, max_depth=3, seed=1)
    for _ in range(200):
        s = gen.sample_positive(20)
        depth = 0
        peak = 0
        for t in s:
            if t < gen.k:
                depth += 1
                peak = max(peak, depth)
            else:
                depth -= 1
        assert peak <= 3
        assert gen.is_valid(s)
    print("[ok] positives respect depth bound and pass is_valid")

    # 3. Negatives really are outside the language.
    for strat in ("edit", "random"):
        gen = DyckGenerator(k=3, max_depth=6, seed=2)
        for L in (4, 10, 40, 100):
            for _ in range(50):
                n = gen.sample_negative(L, strategy=strat)
                assert not gen.is_valid(n), (strat, L, n)
    print("[ok] edit and random negatives are all invalid")

    # 4. Uniformity sanity check: for k=1, L=6, max_depth=large,
    # there are C_3 = 5 strings; each should appear with roughly 1/5 mass.
    gen = DyckGenerator(k=1, max_depth=10, seed=3)
    counts: dict[tuple[int, ...], int] = {}
    N = 20000
    for _ in range(N):
        s = tuple(int(x) for x in gen.sample_positive(6))
        counts[s] = counts.get(s, 0) + 1
    assert len(counts) == 5, counts
    freqs = np.array(sorted(counts.values())) / N
    # each should be near 0.2; allow generous tolerance for stochasticity
    assert np.all(np.abs(freqs - 0.2) < 0.02), freqs
    print("[ok] positive sampler is approximately uniform on Dyck-1 length 6")

    # 5. End-to-end batch generation.
    gen = DyckGenerator(k=2, max_depth=8, seed=4)
    batch = gen.generate_dataset(
        n_samples=1000,
        length_range=(3, 40),
        positive_ratio=0.5,
    )
    assert batch["tokens"].shape == (1000, 40)
    assert batch["labels"].shape == (1000,)
    pos_mask = batch["labels"] == 1
    neg_mask = batch["labels"] == 0
    # labels and validity agree
    for i in range(1000):
        L = int(batch["lengths"][i])
        seq = batch["tokens"][i, :L]
        valid = gen.is_valid(seq)
        assert valid == bool(batch["labels"][i]), (i, L, seq, valid)
    print(
        f"[ok] batch generation: {pos_mask.sum()} positives, "
        f"{neg_mask.sum()} negatives, all labels consistent with is_valid"
    )

    # 6. Vectorized batch primitives produce the same distribution as the
    # per-sample path. Check against Catalan count on k=1, length=6.
    gen = DyckGenerator(k=1, max_depth=10, seed=5)
    batch_pos = gen.batch_sample_positive(6, 20000)
    counts_v: dict[tuple[int, ...], int] = {}
    for row in batch_pos:
        key = tuple(int(x) for x in row)
        counts_v[key] = counts_v.get(key, 0) + 1
    assert len(counts_v) == 5, counts_v
    freqs = np.array(sorted(counts_v.values())) / 20000
    assert np.all(np.abs(freqs - 0.2) < 0.02), freqs
    print("[ok] batch_sample_positive matches uniform on Dyck-1 length 6")

    # 7. Vectorized negatives are all invalid.
    for strat in ("edit", "random"):
        gen = DyckGenerator(k=3, max_depth=8, seed=6)
        for L in (4, 10, 40, 100):
            batch_neg = gen.batch_sample_negative(L, 200, strategy=strat)
            assert batch_neg.shape == (200, L)
            for i in range(200):
                assert not gen.is_valid(batch_neg[i]), (strat, L, i)
    print("[ok] batch_sample_negative returns only invalid strings (edit & random)")

    # 8. Speed benchmark: batched vs per-sample.
    import time
    gen = DyckGenerator(k=1, max_depth=40, seed=7)
    N = 2048
    t0 = time.time()
    _ = gen.generate_dataset(
        n_samples=N, length_range=(3, 40), positive_ratio=0.5,
        negative_strategy="edit",
    )
    dt = time.time() - t0
    print(f"[bench] vectorized generate_dataset: {N} samples in {dt*1000:.0f} ms "
          f"({N/dt:.0f} samples/s)")


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Generate Dyck-k samples (positive and/or negative)."
    )
    p.add_argument("--k", type=int, default=2, help="number of bracket types (>=1)")
    p.add_argument("--max-depth", type=int, default=10, help="stack depth bound")
    p.add_argument(
        "--min-length",
        type=int,
        default=4,
        help="minimum sequence length (inclusive); even integers sampled uniformly",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=40,
        help="maximum sequence length (inclusive)",
    )
    p.add_argument("--n", type=int, default=5, help="number of samples to print")
    p.add_argument(
        "--positive-ratio",
        type=float,
        default=0.5,
        help="fraction of samples that are positive (1.0 = all positive)",
    )
    p.add_argument(
        "--negative-strategy",
        choices=("edit", "random"),
        default="edit",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--raw",
        action="store_true",
        help="print raw token ids instead of bracket characters",
    )
    p.add_argument(
        "--self-test",
        action="store_true",
        help="run internal correctness checks and exit",
    )
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    gen = DyckGenerator(k=args.k, max_depth=args.max_depth, seed=args.seed)
    batch = gen.generate_dataset(
        n_samples=args.n,
        length_range=(args.min_length, args.max_length),
        positive_ratio=args.positive_ratio,
        negative_strategy=args.negative_strategy,
    )

    print(
        f"# Dyck-{args.k} | max_depth={args.max_depth} | "
        f"length in [{args.min_length},{args.max_length}] | n={args.n} | "
        f"positive_ratio={args.positive_ratio} | "
        f"negative_strategy={args.negative_strategy} | seed={args.seed}"
    )
    for i in range(args.n):
        L = int(batch["lengths"][i])
        seq = batch["tokens"][i, :L]
        label = int(batch["labels"][i])
        tag = "POS" if label == 1 else "NEG"
        rendered = (
            " ".join(str(int(t)) for t in seq) if args.raw else format_sequence(seq, args.k)
        )
        print(f"[{tag}] len={L:>3}  {rendered}")


if __name__ == "__main__":
    _cli()
