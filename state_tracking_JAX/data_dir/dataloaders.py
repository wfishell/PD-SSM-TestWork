import importlib

import numpy as np
import jax.numpy as jnp
import jax.random as jr


class FLDataloader:
    """
    Dataloader for formal language tasks.

    Training batches are generated on-the-fly using numpy RNG seeded from JAX keys,
    so no large dataset needs to be held in memory.

    The validation set is pre-generated at construction time and served via loop_epoch.

    Each batch returns:
        X:    (batch_size, max_length)  int32  — token IDs, zero-padded
        y:    (batch_size,)             int32  — class label for each sequence
        mask: (batch_size, max_length)  bool   — True at the last real token position
    """

    def __init__(self, task_name, min_length, max_length, num_val_samples=8192, val_seed=42):
        module = importlib.import_module(f"data_dir.fl_tasks.{task_name}")
        self.generate_sample = module.generate_sample
        self.generate_batch_fn = getattr(module, "generate_batch", None)
        self.vocab_size = module.vocab_size
        self.min_length = min_length
        self.max_length = max_length

        rng = np.random.default_rng(val_seed)
        X, y, mask = self._generate_batch_np(num_val_samples, rng)
        self.val_X = jnp.array(X)
        self.val_y = jnp.array(y)
        self.val_mask = jnp.array(mask)

    def _generate_batch_np(self, batch_size, rng):
        if self.generate_batch_fn is not None:
            return self.generate_batch_fn(batch_size, self.min_length, self.max_length, rng)

        X = np.zeros((batch_size, self.max_length), dtype=np.int32)
        y = np.zeros(batch_size, dtype=np.int32)
        mask = np.zeros((batch_size, self.max_length), dtype=bool)

        for i in range(batch_size):
            tokens, label = self.generate_sample(self.min_length, self.max_length, rng)
            L = len(tokens)
            X[i, :L] = tokens
            y[i] = label
            mask[i, L - 1] = True

        return X, y, mask

    def loop(self, batch_size, *, key):
        """Infinite iterator of training batches. Each batch is freshly generated."""
        while True:
            subkey, key = jr.split(key)
            seed = int(jr.randint(subkey, shape=(), minval=0, maxval=2 ** 31 - 1))
            rng = np.random.default_rng(seed)
            X, y, mask = self._generate_batch_np(batch_size, rng)
            yield jnp.array(X), jnp.array(y), jnp.array(mask)

    def loop_epoch(self, batch_size):
        """Iterate once over the pre-generated validation set."""
        n = self.val_X.shape[0]
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield self.val_X[start:end], self.val_y[start:end], self.val_mask[start:end]


def create_fl_dataloaders(task, min_train_length, max_train_length,
                          min_val_length, max_val_length,
                          num_val_samples, val_seed):
    """
    Returns a dict with 'train' and 'val' FLDataloader instances.

    Train loader generates sequences in [min_train_length, max_train_length].
    Val loader pre-generates sequences in [min_val_length, max_val_length] to
    test generalisation to longer sequences.
    """
    module = importlib.import_module(f"data_dir.fl_tasks.{task}")
    vocab_size = module.vocab_size

    train_loader = FLDataloader(
        task_name=task,
        min_length=min_train_length,
        max_length=max_train_length,
        num_val_samples=num_val_samples,
        val_seed=val_seed,
    )
    val_loader = FLDataloader(
        task_name=task,
        min_length=min_val_length,
        max_length=max_val_length,
        num_val_samples=num_val_samples,
        val_seed=val_seed + 1,
    )
    return {"train": train_loader, "val": val_loader}, vocab_size
