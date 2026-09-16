"""Actor-Critic neural network architecture and inference engine in JAX and NumPy."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def orthogonal_init(
    rng: jax.Array,
    shape: tuple[int, int],
    scale: float = np.sqrt(2.0),
) -> jnp.ndarray:
    """Orthogonal matrix initialization for stable deep RL policy gradients."""
    rows, cols = shape
    flat_shape = (max(rows, cols), min(rows, cols))
    random_mat = jax.random.normal(rng, flat_shape, dtype=jnp.float32)
    q, r = jnp.linalg.qr(random_mat)
    d = jnp.diag(r)
    ph = jnp.sign(d)
    q = q * ph
    if rows < cols:
        q = q.T
    return (scale * q[:rows, :cols]).astype(jnp.float32)


def init_actor_critic_params(
    rng: jax.Array,
    obs_dim: int,
    action_dim: int,
    hidden_dims: tuple[int, ...] = (128, 128, 64),
    init_log_std: float = -0.65,
) -> dict[str, Any]:
    """Initialize separate Actor and Critic MLP parameters with orthogonal weights."""
    actor_layers = []
    critic_layers = []

    dims = [obs_dim, *hidden_dims]
    for i in range(len(hidden_dims)):
        rng, k_a, k_c = jax.random.split(rng, 3)
        in_d, out_d = dims[i], dims[i + 1]
        actor_layers.append(
            {
                "w": orthogonal_init(k_a, (in_d, out_d), scale=float(np.sqrt(2.0))),
                "b": jnp.zeros((out_d,), dtype=jnp.float32),
            }
        )
        critic_layers.append(
            {
                "w": orthogonal_init(k_c, (in_d, out_d), scale=float(np.sqrt(2.0))),
                "b": jnp.zeros((out_d,), dtype=jnp.float32),
            }
        )

    rng, k_ao, k_co = jax.random.split(rng, 3)
    last_dim = hidden_dims[-1]
    # Small output scale (0.01) ensures initial policy actions stay centered near 0
    actor_head = {
        "w": orthogonal_init(k_ao, (last_dim, action_dim), scale=0.01),
        "b": jnp.zeros((action_dim,), dtype=jnp.float32),
    }
    critic_head = {
        "w": orthogonal_init(k_co, (last_dim, 1), scale=1.0),
        "b": jnp.zeros((1,), dtype=jnp.float32),
    }

    return {
        "actor_layers": actor_layers,
        "actor_head": actor_head,
        "actor_log_std": jnp.full((action_dim,), init_log_std, dtype=jnp.float32),
        "critic_layers": critic_layers,
        "critic_head": critic_head,
    }


def forward_actor_critic_jax(
    params: dict[str, Any],
    norm_obs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate Actor mean, log_std, and Critic value V(s) for normalized observations."""
    x_a = norm_obs
    for layer in params["actor_layers"]:
        x_a = jax.nn.elu(jnp.dot(x_a, layer["w"]) + layer["b"])
    mean = jnp.dot(x_a, params["actor_head"]["w"]) + params["actor_head"]["b"]
    log_std = jnp.clip(params["actor_log_std"], -2.2, 0.3)

    x_c = norm_obs
    for layer in params["critic_layers"]:
        x_c = jax.nn.elu(jnp.dot(x_c, layer["w"]) + layer["b"])
    value = jnp.squeeze(
        jnp.dot(x_c, params["critic_head"]["w"]) + params["critic_head"]["b"], axis=-1
    )
    return mean, log_std, value


def gaussian_log_prob_jax(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    actions: jnp.ndarray,
) -> jnp.ndarray:
    """Compute diagonal Gaussian log-probability summed over action dimensions."""
    std = jnp.exp(log_std)
    var = jnp.square(std)
    log_scale = log_std + 0.5 * jnp.log(2.0 * jnp.pi)
    return jnp.sum(-0.5 * jnp.square(actions - mean) / var - log_scale, axis=-1)


def gaussian_entropy_jax(log_std: jnp.ndarray) -> jnp.ndarray:
    """Compute analytical differential entropy of diagonal Gaussian policy."""
    return jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1)


def sample_actions_jax(
    params: dict[str, Any],
    norm_obs: jnp.ndarray,
    rng: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample stochastic actions a_t ~ pi_theta(. | o_t), returning (action, log_prob, value, mean)."""
    mean, log_std, value = forward_actor_critic_jax(params, norm_obs)
    std = jnp.exp(log_std)
    noise = jax.random.normal(rng, mean.shape, dtype=jnp.float32)
    action = mean + std * noise
    log_prob = gaussian_log_prob_jax(mean, log_std, action)
    return action, log_prob, value, mean


def _elu_numpy(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, np.expm1(np.clip(x, -20.0, 0.0)))


class PolicyInferenceEngine:
    """Pure NumPy inference wrapper for trained Actor-Critic policies."""

    def __init__(
        self,
        params: dict[str, Any],
        obs_mean: np.ndarray,
        obs_var: np.ndarray,
        policy_name: str = "baseline_ppo",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.policy_name = policy_name
        self.metadata = metadata or {}
        self.obs_mean = np.asarray(obs_mean, dtype=np.float32)
        self.obs_var = np.asarray(obs_var, dtype=np.float32)

        self.actor_layers = [
            {
                "w": np.asarray(layer["w"], dtype=np.float32),
                "b": np.asarray(layer["b"], dtype=np.float32),
            }
            for layer in params["actor_layers"]
        ]
        self.actor_head = {
            "w": np.asarray(params["actor_head"]["w"], dtype=np.float32),
            "b": np.asarray(params["actor_head"]["b"], dtype=np.float32),
        }
        self.actor_log_std = np.clip(
            np.asarray(params["actor_log_std"], dtype=np.float32), -2.2, 0.3
        )
        self.critic_layers = [
            {
                "w": np.asarray(layer["w"], dtype=np.float32),
                "b": np.asarray(layer["b"], dtype=np.float32),
            }
            for layer in params["critic_layers"]
        ]
        self.critic_head = {
            "w": np.asarray(params["critic_head"]["w"], dtype=np.float32),
            "b": np.asarray(params["critic_head"]["b"], dtype=np.float32),
        }

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, float]:
        """Compute policy action and state-value estimate V(s) for a single observation."""
        norm_obs = np.clip(
            (np.asarray(obs, dtype=np.float32) - self.obs_mean)
            / np.sqrt(self.obs_var + 1e-8),
            -10.0,
            10.0,
        )
        x_a = norm_obs
        for layer in self.actor_layers:
            x_a = _elu_numpy(x_a @ layer["w"] + layer["b"])
        mean = x_a @ self.actor_head["w"] + self.actor_head["b"]

        x_c = norm_obs
        for layer in self.critic_layers:
            x_c = _elu_numpy(x_c @ layer["w"] + layer["b"])
        value = float(np.squeeze(x_c @ self.critic_head["w"] + self.critic_head["b"]))

        if deterministic:
            return mean.astype(np.float32), value
        gen = rng or np.random.default_rng()
        std = np.exp(self.actor_log_std)
        action = mean + std * gen.normal(size=mean.shape).astype(np.float32)
        return action.astype(np.float32), value
