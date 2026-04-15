from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import math

parallel_scan = jax.lax.associative_scan


# ---------------------------------------------------------------------------
# Parallel scan operators (verbatim from time_series_JAX)
# ---------------------------------------------------------------------------

@jax.vmap
def binary_operator_efficient(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence."""
    P_i, D_i, b_i = q_i
    P_j, D_j, b_j = q_j
    P_new = jnp.take(P_j, indices=P_i)
    D_new = jnp.take(D_j, indices=P_i) * D_i
    b_new = jnp.zeros_like(b_i).at[P_j].add(D_j * b_i) + b_j
    return P_new, D_new, b_new


@jax.vmap
def binary_operator_efficient_grad(q_i, q_j):
    """Binary operator for parallel scan of the backward pass."""
    P_i, D_i, b_i = q_i
    P_j, D_j, b_j = q_j
    P_new = jnp.take(P_i, indices=P_j)
    D_new = jnp.take(D_i, indices=P_j) * D_j
    b_new = b_i[P_j] * D_j + b_j
    return P_new, D_new, b_new


def indexing_parallel_scan(P, D, b):
    _, _, y = parallel_scan(binary_operator_efficient, (P, D, b), axis=0)
    return y


def indexing_grad_parallel_scan(P, D, b):
    _, _, y = parallel_scan(binary_operator_efficient_grad, (P, D, b), axis=0)
    return y


# ---------------------------------------------------------------------------
# Custom VJP to route gradients through the argmax (verbatim from time_series_JAX)
# ---------------------------------------------------------------------------

@jax.custom_vjp
def custom_function(P, D, b):
    return forward_pass(P, D, b)


def forward_pass(P, D, b):
    return indexing_parallel_scan(jnp.argmax(P, axis=-2), D, b)


def custom_fwd(P, D, b):
    y = forward_pass(P, D, b)
    return y, (y, jnp.argmax(P, axis=-2), D)


def batch_index(b_row, p_row):
    return b_row[p_row]


def custom_bwd(residuals, grad_xs):
    y, P, D = residuals
    T, N = D.shape
    y_padded = jnp.concatenate([jnp.zeros((1, N)), y[:-1, :]], axis=0)

    pscan_inputs = jnp.flip(grad_xs, axis=0)
    pscan_Ps = jnp.concatenate([jnp.arange(N)[None, :], jnp.flip(P[1:], axis=0)], axis=0)
    pscan_Ds = jnp.concatenate([jnp.ones((1, N)), jnp.flip(D[1:], axis=0)], axis=0)
    grad_ys = jnp.flip(indexing_grad_parallel_scan(pscan_Ps, pscan_Ds, pscan_inputs), axis=0)

    grad_Ps = jnp.einsum('tj,tl->tjl', grad_ys, D * y_padded)
    grad_Ds = jax.vmap(batch_index)(grad_ys, P) * y_padded
    grad_bs = grad_ys

    return grad_Ps.real, grad_Ds, grad_bs


custom_function.defvjp(custom_fwd, custom_bwd)


# ---------------------------------------------------------------------------
# PDSSMLayer: the core SSM with parallel scan (verbatim from time_series_JAX,
# minus unused W2/b2 parameters)
# ---------------------------------------------------------------------------

class PDSSMLayer(eqx.Module):
    B_re: jnp.ndarray
    B_im: jnp.ndarray
    C_re: jnp.ndarray
    C_im: jnp.ndarray
    D: jnp.ndarray
    P_dict: jnp.ndarray
    P_selector: jnp.ndarray
    W1_mag: jnp.ndarray
    b1_mag: jnp.ndarray
    W1_pha: jnp.ndarray
    b1_pha: jnp.ndarray

    def __init__(self, N, H, K=6, *, key):
        """
        N: hidden state dimension
        H: model (embedding) dimension
        K: dictionary size (number of learned permutation matrices)
        """
        keys = jr.split(key, 11)
        self.B_re = jr.normal(keys[0], shape=(N, H)) / jnp.sqrt(2 * H)
        self.B_im = jr.normal(keys[1], shape=(N, H)) / jnp.sqrt(2 * H)
        self.C_re = jr.normal(keys[2], shape=(H, N)) / jnp.sqrt(N)
        self.C_im = jr.normal(keys[3], shape=(H, N)) / jnp.sqrt(N)
        self.D = jr.normal(keys[4], shape=(H,))
        self.P_dict = jr.normal(keys[5], shape=(K, N, N)) / jnp.sqrt(N)
        self.P_selector = jr.normal(keys[6], shape=(K, H))
        self.W1_mag = jr.normal(keys[7], shape=(N, H)) / jnp.sqrt(H)
        self.b1_mag = jr.normal(keys[8], shape=(N,)) / jnp.sqrt(N)
        self.W1_pha = jr.normal(keys[9], shape=(N, H)) / jnp.sqrt(H)
        self.b1_pha = jr.normal(keys[10], shape=(N,)) / jnp.sqrt(N) - 2.5

    def __call__(self, x):
        """
        x: (L, H) float
        returns: (L, H) float
        """
        B = self.B_re + 1j * self.B_im
        C = self.C_re + 1j * self.C_im

        # Input-dependent diagonal matrices: magnitudes in (0,1), phases in (0, 2pi)
        magnitudes = jax.vmap(lambda u: jax.nn.sigmoid(self.W1_mag @ u + self.b1_mag))(x)
        phases = jax.vmap(lambda u: jax.nn.sigmoid(self.W1_pha @ u + self.b1_pha) * 2 * math.pi)(x)
        phases_complex = jax.vmap(lambda u: jnp.exp(1j * u))(phases)
        diagonal_matrices = magnitudes * phases_complex  # (L, N) complex

        # Input-dependent permutation matrices via dictionary lookup + column softmax
        selection_weights = jax.vmap(lambda u: jax.nn.softmax(self.P_selector @ u, axis=-1))(x)
        permutation_matrices = jax.vmap(lambda u: jnp.einsum('kmn,k->mn', self.P_dict, u))(selection_weights)
        permutation_matrices_column_softmax = jax.vmap(
            lambda u: jax.nn.softmax(u, axis=0)
        )(permutation_matrices)  # (L, N, N)

        # Project input into hidden space
        Bu_elements = jax.vmap(lambda u: B @ u)(x)  # (L, N) complex

        # Parallel associative scan over the sequence
        hidden_states = custom_function(
            permutation_matrices_column_softmax, diagonal_matrices, Bu_elements
        )  # (L, N) complex

        # Readout with skip connection
        y = jax.vmap(lambda h, u: (C @ h).real + self.D * u)(hidden_states, x)  # (L, H)
        return y


# ---------------------------------------------------------------------------
# PDSSMBlock: LayerNorm wrapper around PDSSMLayer
# ---------------------------------------------------------------------------

class PDSSMBlock(eqx.Module):
    norm: eqx.nn.LayerNorm
    pdssm: PDSSMLayer

    def __init__(self, N, H, K=6, *, key):
        self.norm = eqx.nn.LayerNorm(shape=H)
        self.pdssm = PDSSMLayer(N, H, K, key=key)

    def __call__(self, x):
        """x: (L, H) -> (L, H)"""
        return jax.vmap(self.norm)(self.pdssm(x))


# ---------------------------------------------------------------------------
# StateTrackingPDSSM: full model for discrete-token sequence tasks
# ---------------------------------------------------------------------------

class StateTrackingPDSSM(eqx.Module):
    embedding: eqx.nn.Embedding
    blocks: List[PDSSMBlock]
    readout: eqx.nn.Linear

    def __init__(self, vocab_size, label_dim, N, H, num_layers, K=6, *, key):
        """
        vocab_size: number of input token types (including padding token 0)
        label_dim:  number of output classes
        N:          hidden state dimension
        H:          model (embedding) dimension
        num_layers: number of stacked PDSSMBlocks
        K:          permutation dictionary size
        """
        keys = jr.split(key, num_layers + 2)
        self.embedding = eqx.nn.Embedding(vocab_size, H, key=keys[0])
        self.blocks = [PDSSMBlock(N, H, K, key=keys[i + 1]) for i in range(num_layers)]
        self.readout = eqx.nn.Linear(H, label_dim, key=keys[-1])

    def __call__(self, x):
        """
        x: (L,) int  — token IDs for a single sequence
        returns: (L, label_dim) float logits
        """
        x = jax.vmap(self.embedding)(x)   # (L, H)
        for block in self.blocks:
            x = block(x)
        return jax.vmap(self.readout)(x)   # (L, label_dim)
