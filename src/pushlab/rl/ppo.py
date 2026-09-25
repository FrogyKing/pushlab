"""Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE) in JAX."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from pushlab.config import PPOConfig
from pushlab.rl.networks import (
    forward_actor_critic_jax,
    gaussian_entropy_jax,
    gaussian_log_prob_jax,
)


class RolloutBatch(NamedTuple):
    """Trajectory batch collected across parallel environments over rollout_steps."""

    norm_obs: jnp.ndarray  # (T, N, obs_dim)
    actions: jnp.ndarray  # (T, N, action_dim)
    log_probs: jnp.ndarray  # (T, N)
    values: jnp.ndarray  # (T, N)
    rewards: jnp.ndarray  # (T, N)
    dones: jnp.ndarray  # (T, N)
    terminated: jnp.ndarray  # (T, N)


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_value: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Generalized Advantage Estimation (GAE) and value targets via reverse scan.

    Args:
        rewards: shape (T, N)
        values: shape (T, N)
        dones: shape (T, N) boolean or float mask
        last_value: bootstrap value V(s_T) of shape (N,)
        gamma: discount factor
        gae_lambda: GAE exponential weighting parameter

    Returns:
        advantages: shape (T, N)
        returns: shape (T, N) where returns = advantages + values
    """
    not_done = 1.0 - dones.astype(jnp.float32)

    def scan_step(
        carry: tuple[jnp.ndarray, jnp.ndarray],
        step_inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ) -> tuple[tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        gae, next_val = carry
        rew, val, mask = step_inputs
        delta = rew + gamma * next_val * mask - val
        gae = delta + gamma * gae_lambda * mask * gae
        return (gae, val), gae

    _, advantages = jax.lax.scan(
        scan_step,
        (jnp.zeros_like(last_value), last_value),
        (rewards, values, not_done),
        reverse=True,
    )
    returns = advantages + values
    return advantages, returns


def ppo_loss_fn(
    params: dict[str, Any],
    norm_obs: jnp.ndarray,
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    old_values: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    clip_eps: float = 0.20,
    value_loss_coef: float = 0.50,
    entropy_coef: float = 0.008,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Compute PPO clipped surrogate objective, clipped value loss, and entropy bonus."""
    mean, log_std, values = forward_actor_critic_jax(params, norm_obs)
    new_log_probs = gaussian_log_prob_jax(mean, log_std, actions)
    entropy = jnp.mean(gaussian_entropy_jax(log_std))

    # Normalize advantages across the minibatch
    norm_adv = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    log_ratio = new_log_probs - old_log_probs
    ratio = jnp.exp(log_ratio)

    surr1 = ratio * norm_adv
    surr2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * norm_adv
    policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

    # Value function loss with clipping
    val_pred_clipped = old_values + jnp.clip(values - old_values, -clip_eps, clip_eps)
    v_loss_unclipped = jnp.square(values - returns)
    v_loss_clipped = jnp.square(val_pred_clipped - returns)
    value_loss = 0.5 * jnp.mean(jnp.maximum(v_loss_unclipped, v_loss_clipped))

    total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

    # Diagnostics
    approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
    clip_fraction = jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32))
    var_y = jnp.var(returns)
    explained_var = jnp.where(
        var_y > 1e-6, 1.0 - jnp.var(returns - values) / (var_y + 1e-8), 0.0
    )

    metrics = {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "explained_variance": explained_var,
    }
    return total_loss, metrics


def create_ppo_optimizer(config: PPOConfig) -> optax.GradientTransformation:
    """Create Adam optimizer with global gradient norm clipping and linear LR decay."""
    total_grad_steps = config.num_updates * config.num_epochs * config.num_minibatches
    lr_schedule = optax.linear_schedule(
        init_value=config.learning_rate,
        end_value=config.learning_rate * 0.15,
        transition_steps=max(1, total_grad_steps),
    )
    return optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=lr_schedule, eps=1e-5),
    )


def make_ppo_update_fn(
    config: PPOConfig,
    optimizer: optax.GradientTransformation,
):
    """Build a JIT-compilable multi-epoch minibatch PPO update step."""
    grad_fn = jax.value_and_grad(ppo_loss_fn, has_aux=True)

    def update_step(
        params: dict[str, Any],
        opt_state: optax.OptState,
        batch: RolloutBatch,
        last_value: jnp.ndarray,
        rng: jax.Array,
    ) -> tuple[dict[str, Any], optax.OptState, dict[str, jnp.ndarray]]:
        advantages, returns = compute_gae(
            rewards=batch.rewards,
            values=batch.values,
            dones=batch.dones,
            last_value=last_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )

        batch_size = batch.norm_obs.shape[0] * batch.norm_obs.shape[1]
        flat_obs = jnp.reshape(batch.norm_obs, (batch_size, -1))
        flat_act = jnp.reshape(batch.actions, (batch_size, -1))
        flat_logp = jnp.reshape(batch.log_probs, (batch_size,))
        flat_val = jnp.reshape(batch.values, (batch_size,))
        flat_adv = jnp.reshape(advantages, (batch_size,))
        flat_ret = jnp.reshape(returns, (batch_size,))

        def epoch_body(
            carry: tuple[dict[str, Any], optax.OptState, jax.Array],
            _: None,
        ) -> tuple[tuple[dict[str, Any], optax.OptState, jax.Array], dict[str, jnp.ndarray]]:
            p, o_state, key = carry
            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, batch_size)

            mb_size = batch_size // config.num_minibatches

            def minibatch_body(
                mb_carry: tuple[dict[str, Any], optax.OptState],
                mb_idx: jnp.ndarray,
            ) -> tuple[tuple[dict[str, Any], optax.OptState], dict[str, jnp.ndarray]]:
                curr_p, curr_opt = mb_carry
                start = mb_idx * mb_size
                idx = jax.lax.dynamic_slice(perm, (start,), (mb_size,))

                (_, aux), grads = grad_fn(
                    curr_p,
                    flat_obs[idx],
                    flat_act[idx],
                    flat_logp[idx],
                    flat_val[idx],
                    flat_adv[idx],
                    flat_ret[idx],
                    config.clip_eps,
                    config.value_loss_coef,
                    config.entropy_coef,
                )
                grad_norm = optax.tree.norm(grads)
                updates, next_opt = optimizer.update(grads, curr_opt, curr_p)
                next_p = optax.apply_updates(curr_p, updates)
                aux_with_grad = {**aux, "grad_norm": grad_norm}
                return (next_p, next_opt), aux_with_grad

            (p_out, opt_out), mb_metrics = jax.lax.scan(
                minibatch_body,
                (p, o_state),
                jnp.arange(config.num_minibatches),
            )
            epoch_mean = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), mb_metrics)
            return (p_out, opt_out, key), epoch_mean

        (final_params, final_opt_state, _), all_epoch_metrics = jax.lax.scan(
            epoch_body,
            (params, opt_state, rng),
            None,
            length=config.num_epochs,
        )
        summary_metrics = jax.tree_util.tree_map(
            lambda x: jnp.mean(x, axis=0), all_epoch_metrics
        )
        return final_params, final_opt_state, summary_metrics

    return jax.jit(update_step)
