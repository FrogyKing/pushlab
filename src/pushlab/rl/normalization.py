"""Welford running observation normalization for JAX and NumPy."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class RunningMeanStdState(NamedTuple):
    """Running statistics (mean, variance, count) for online observation normalization."""

    mean: jnp.ndarray
    var: jnp.ndarray
    count: jnp.ndarray


def init_running_mean_std(obs_dim: int) -> RunningMeanStdState:
    return RunningMeanStdState(
        mean=jnp.zeros((obs_dim,), dtype=jnp.float32),
        var=jnp.ones((obs_dim,), dtype=jnp.float32),
        count=jnp.array(1e-4, dtype=jnp.float32),
    )


def update_running_mean_std(
    state: RunningMeanStdState,
    batch_obs: jnp.ndarray,
) -> RunningMeanStdState:
    """Update running mean and variance with a batch of observations of shape (..., obs_dim)."""
    flat = jnp.reshape(batch_obs, (-1, batch_obs.shape[-1]))
    batch_mean = jnp.mean(flat, axis=0)
    batch_var = jnp.var(flat, axis=0)
    batch_count = jnp.array(flat.shape[0], dtype=jnp.float32)

    delta = batch_mean - state.mean
    tot_count = state.count + batch_count

    new_mean = state.mean + delta * (batch_count / tot_count)
    m_a = state.var * state.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + jnp.square(delta) * (state.count * batch_count / tot_count)
    new_var = m2 / tot_count

    return RunningMeanStdState(mean=new_mean, var=new_var, count=tot_count)


def normalize_obs_jax(
    obs: jnp.ndarray,
    state: RunningMeanStdState,
    clip: float = 10.0,
) -> jnp.ndarray:
    normed = (obs - state.mean) / jnp.sqrt(state.var + 1e-8)
    return jnp.clip(normed, -clip, clip)


def normalize_obs_numpy(
    obs: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    clip: float = 10.0,
) -> np.ndarray:
    normed = (obs - mean) / np.sqrt(var + 1e-8)
    return np.clip(normed, -clip, clip).astype(np.float32)
