import numpy as np

vocab_size = 3
# Tokens: 0=padding, 1='a', 2='b'
# Labels: 1='a' (first == last), 2='b' (first != last)


def generate_sample(min_length, max_length, rng):
    """
    rng: np.random.Generator
    returns: (tokens: np.ndarray int32, label: int)
    """
    length = int(rng.integers(min_length, max_length + 1))
    tokens = rng.integers(1, 3, size=length).astype(np.int32)  # 1='a', 2='b'
    label = 1 if tokens[0] == tokens[-1] else 2
    return tokens, label


def generate_batch(batch_size, min_length, max_length, rng):
    """Vectorized batch generation — no Python loop over samples."""
    lengths = rng.integers(min_length, max_length + 1, size=batch_size)
    X = rng.integers(1, 3, size=(batch_size, max_length)).astype(np.int32)
    positions = np.arange(max_length)[None, :]          # (1, max_length)
    valid = positions < lengths[:, None]                # (B, max_length)
    X = np.where(valid, X, 0)
    mask = (positions == (lengths[:, None] - 1))
    first = X[:, 0]
    last = X[np.arange(batch_size), lengths - 1]
    y = np.where(first == last, 1, 2).astype(np.int32)
    return X, y, mask
