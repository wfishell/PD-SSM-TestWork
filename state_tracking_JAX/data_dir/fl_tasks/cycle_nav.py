import numpy as np

max_position = 5
vocab_size = max_position + 4  # = 9
# Tokens: 0=padding, 1=STAY, 2=+1, 3=-1
# Labels: 4..8 (final position on cycle of length max_position)


def generate_sample(min_length, max_length, rng):
    """
    rng: np.random.Generator
    returns: (tokens: np.ndarray int32, label: int)
    """
    length = int(rng.integers(min_length, max_length + 1))
    movements = rng.integers(0, 3, size=length)  # 0=STAY, 1=+1, 2=-1
    tokens = (movements + 1).astype(np.int32)     # 1=STAY, 2=+1, 3=-1
    net = int((movements == 1).sum()) - int((movements == 2).sum())
    label = int(4 + net % max_position)
    return tokens, label


def generate_batch(batch_size, min_length, max_length, rng):
    """Vectorized batch generation — no Python loop over samples."""
    lengths = rng.integers(min_length, max_length + 1, size=batch_size)
    movements = rng.integers(0, 3, size=(batch_size, max_length))
    tokens = (movements + 1).astype(np.int32)
    positions = np.arange(max_length)[None, :]          # (1, max_length)
    valid = positions < lengths[:, None]                # (B, max_length)
    tokens = np.where(valid, tokens, 0)
    mask = (positions == (lengths[:, None] - 1))
    plus_counts  = ((movements == 1) & valid).sum(axis=1)
    minus_counts = ((movements == 2) & valid).sum(axis=1)
    net = plus_counts - minus_counts
    y = (net % max_position + 4).astype(np.int32)
    return tokens, y, mask
