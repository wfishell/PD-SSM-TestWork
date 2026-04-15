import numpy as np

vocab_size = 3
# Tokens: 0=padding, 1='a', 2='b'
# Labels: 1='a' (even number of b's), 2='b' (odd number of b's)


def generate_sample(min_length, max_length, rng):
    """
    rng: np.random.Generator
    returns: (tokens: np.ndarray int32, label: int)
    """
    length = int(rng.integers(min_length, max_length + 1))
    tokens = rng.integers(1, 3, size=length).astype(np.int32)  # 1='a', 2='b'
    num_b = int((tokens == 2).sum())
    label = 1 if num_b % 2 == 0 else 2
    return tokens, label


def generate_batch(batch_size, min_length, max_length, rng):
    """Vectorized batch generation — no Python loop over samples."""
    lengths = rng.integers(min_length, max_length + 1, size=batch_size)
    X = rng.integers(1, 3, size=(batch_size, max_length)).astype(np.int32)
    positions = np.arange(max_length)[None, :]          # (1, max_length)
    valid = positions < lengths[:, None]                # (B, max_length)
    X = np.where(valid, X, 0)
    mask = (positions == (lengths[:, None] - 1))
    b_counts = ((X == 2) & valid).sum(axis=1)
    y = (b_counts % 2 + 1).astype(np.int32)            # 1=even, 2=odd
    return X, y, mask
