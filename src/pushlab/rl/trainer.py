"""End-to-end GPU-accelerated PPO training pipeline using JAX and MuJoCo MJX."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from pushlab.config import CHECKPOINTS_DIR, ExperimentConfig
from pushlab.rl.checkpoints import save_policy_checkpoint
from pushlab.rl.networks import (
    forward_actor_critic_jax,
    init_actor_critic_params,
    sample_actions_jax,
)
from pushlab.rl.normalization import (
    RunningMeanStdState,
    init_running_mean_std,
    normalize_obs_jax,
    update_running_mean_std,
)
from pushlab.rl.ppo import RolloutBatch, create_ppo_optimizer, make_ppo_update_fn
from pushlab.sim.mjx_env import MJXEnvState, MJXQuadrupedEnv

logger = logging.getLogger(__name__)


def train_ppo(
    config: ExperimentConfig,
    output_dir: Path = CHECKPOINTS_DIR,
    log_interval: int = 10,
) -> dict[str, Any]:
    """Train a quadruped locomotion policy with PPO on MJX and save artifacts."""
    ppo_cfg = config.ppo
    sim_cfg = config.sim

    devices = jax.devices()
    backend_platform = jax.default_backend()
    device_desc = str(devices[0]) if devices else "cpu"
    logger.info(
        "Starting PPO training for '%s' on backend=%s (%s): num_envs=%d, updates=%d, total_steps=%d",
        ppo_cfg.policy_name,
        backend_platform,
        device_desc,
        ppo_cfg.num_envs,
        ppo_cfg.num_updates,
        ppo_cfg.total_timesteps,
    )

    mjx_env = MJXQuadrupedEnv(config)
    rng = jax.random.PRNGKey(ppo_cfg.seed)
    rng, init_key, env_key = jax.random.split(rng, 3)

    params = init_actor_critic_params(
        init_key,
        obs_dim=mjx_env.obs_dim,
        action_dim=mjx_env.action_dim,
        hidden_dims=ppo_cfg.hidden_dims,
        init_log_std=ppo_cfg.init_log_std,
    )
    norm_state = init_running_mean_std(mjx_env.obs_dim)
    optimizer = create_ppo_optimizer(ppo_cfg)
    opt_state = optimizer.init(params)
    ppo_update_step = make_ppo_update_fn(ppo_cfg, optimizer)

    env_state = mjx_env.reset(env_key, num_envs=ppo_cfg.num_envs)

    @jax.jit
    def collect_rollout(
        state: MJXEnvState,
        net_params: dict[str, Any],
        obs_norm: RunningMeanStdState,
        rollout_rng: jax.Array,
    ) -> tuple[MJXEnvState, RolloutBatch, jnp.ndarray, jnp.ndarray]:
        def step_fn(
            carry: tuple[MJXEnvState, jax.Array],
            _: None,
        ) -> tuple[tuple[MJXEnvState, jax.Array], tuple[RolloutBatch, jnp.ndarray]]:
            curr_state, key = carry
            key, act_key = jax.random.split(key)

            raw_obs = curr_state.obs
            norm_obs = normalize_obs_jax(raw_obs, obs_norm)
            actions, log_probs, values, _ = sample_actions_jax(
                net_params, norm_obs, act_key
            )
            next_state = mjx_env.step(curr_state, actions)

            transition = RolloutBatch(
                norm_obs=norm_obs,
                actions=actions,
                log_probs=log_probs,
                values=values,
                rewards=next_state.reward,
                dones=next_state.done,
                terminated=next_state.terminated,
            )
            return (next_state, key), (transition, raw_obs)

        (final_state, _), (batch, raw_obs_traj) = jax.lax.scan(
            step_fn,
            (state, rollout_rng),
            None,
            length=ppo_cfg.rollout_steps,
        )
        last_norm_obs = normalize_obs_jax(final_state.obs, obs_norm)
        _, _, last_value = forward_actor_critic_jax(net_params, last_norm_obs)
        return final_state, batch, last_value, raw_obs_traj

    history: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for update_idx in range(1, ppo_cfg.num_updates + 1):
        rng, r_key, u_key = jax.random.split(rng, 3)

        env_state, batch, last_value, raw_obs_traj = collect_rollout(
            env_state, params, norm_state, r_key
        )
        norm_state = update_running_mean_std(norm_state, raw_obs_traj)
        params, opt_state, update_metrics = ppo_update_step(
            params, opt_state, batch, last_value, u_key
        )

        if update_idx == 1 or update_idx % log_interval == 0 or update_idx == ppo_cfg.num_updates:
            mean_step_rew = float(jnp.mean(batch.rewards))
            term_rate = float(jnp.mean(batch.terminated.astype(jnp.float32)))
            survival_rate = float(1.0 - jnp.clip(term_rate * sim_cfg.max_episode_steps * 0.35, 0.0, 1.0))
            completed_mask = env_state.completed_length > 0
            if bool(jnp.any(completed_mask)):
                mean_ep_ret = float(
                    jnp.sum(jnp.where(completed_mask, env_state.completed_return, 0.0))
                    / jnp.maximum(1, jnp.sum(completed_mask))
                )
                mean_ep_len = float(
                    jnp.sum(jnp.where(completed_mask, env_state.completed_length, 0))
                    / jnp.maximum(1, jnp.sum(completed_mask))
                )
            else:
                mean_ep_ret = mean_step_rew * sim_cfg.max_episode_steps
                mean_ep_len = float(jnp.mean(env_state.episode_length))

            elapsed_s = time.perf_counter() - start_time
            timesteps = update_idx * ppo_cfg.batch_size
            fps = timesteps / max(1e-6, elapsed_s)

            entry = {
                "update": update_idx,
                "timesteps": timesteps,
                "mean_step_reward": round(mean_step_rew, 4),
                "mean_episode_return": round(mean_ep_ret, 3),
                "mean_episode_length": round(mean_ep_len, 2),
                "step_termination_rate": round(term_rate, 5),
                "estimated_survival_rate": round(survival_rate, 4),
                "policy_loss": round(float(update_metrics["policy_loss"]), 5),
                "value_loss": round(float(update_metrics["value_loss"]), 5),
                "entropy": round(float(update_metrics["entropy"]), 4),
                "approx_kl": round(float(update_metrics["approx_kl"]), 5),
                "clip_fraction": round(float(update_metrics["clip_fraction"]), 4),
                "explained_variance": round(float(update_metrics["explained_variance"]), 4),
                "grad_norm": round(float(update_metrics["grad_norm"]), 4),
                "fps": round(fps, 1),
                "elapsed_s": round(elapsed_s, 2),
            }
            history.append(entry)
            logger.info(
                "[%s] Update %d/%d (%d steps) | ep_ret=%.2f | step_rew=%.3f | kl=%.4f | fps=%.0f",
                ppo_cfg.policy_name,
                update_idx,
                ppo_cfg.num_updates,
                timesteps,
                mean_ep_ret,
                mean_step_rew,
                entry["approx_kl"],
                fps,
            )

    total_elapsed = time.perf_counter() - start_time
    metadata = {
        "policy_name": ppo_cfg.policy_name,
        "seed": ppo_cfg.seed,
        "backend": backend_platform,
        "device": device_desc,
        "total_timesteps": ppo_cfg.total_timesteps,
        "num_envs": ppo_cfg.num_envs,
        "num_updates": ppo_cfg.num_updates,
        "training_seconds": round(total_elapsed, 2),
        "disturbance_enabled": config.disturbance.enabled,
        "force_magnitude_range": list(config.disturbance.force_magnitude_range),
        "final_metrics": history[-1] if history else {},
    }

    npz_path, json_path = save_policy_checkpoint(
        params=params,
        norm_state=norm_state,
        policy_name=ppo_cfg.policy_name,
        output_dir=output_dir,
        metadata=metadata,
    )

    metrics_path = output_dir / f"{ppo_cfg.policy_name}_metrics.json"
    metrics_payload = {
        "policy_name": ppo_cfg.policy_name,
        "config": config.model_dump(),
        "metadata": metadata,
        "history": history,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    return {
        "npz_path": str(npz_path),
        "json_path": str(json_path),
        "metrics_path": str(metrics_path),
        "metadata": metadata,
        "history": history,
    }
