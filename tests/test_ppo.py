"""Unit tests for PPO clipped objective, GAE, normalization, and checkpointing."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pushlab.config import PPOConfig, get_baseline_experiment_config
from pushlab.rl.checkpoints import load_policy_checkpoint, save_policy_checkpoint
from pushlab.rl.networks import (
    PolicyInferenceEngine,
    forward_actor_critic_jax,
    init_actor_critic_params,
)
from pushlab.rl.normalization import (
    init_running_mean_std,
    normalize_obs_jax,
    normalize_obs_numpy,
    update_running_mean_std,
)
from pushlab.rl.ppo import (
    RolloutBatch,
    compute_gae,
    create_ppo_optimizer,
    make_ppo_update_fn,
    ppo_loss_fn,
)
from pushlab.rl.trainer import train_ppo


def test_gae_hand_calculated() -> None:
    rewards = jnp.array([[1.0, 2.0], [0.5, 1.0], [2.0, -1.0]], dtype=jnp.float32)
    values = jnp.array([[0.8, 1.5], [0.4, 0.9], [1.2, 0.2]], dtype=jnp.float32)
    dones = jnp.array([[False, False], [False, True], [False, False]])
    last_val = jnp.array([1.0, 0.5], dtype=jnp.float32)
    gamma = 0.99
    lam = 0.95

    adv, ret = compute_gae(rewards, values, dones, last_val, gamma=gamma, gae_lambda=lam)
    assert adv.shape == (3, 2)
    assert ret.shape == (3, 2)

    # Check step t=1, env=1 where done=True: delta = r_1 + gamma * 0 - V_1 = 1.0 - 0.9 = 0.1
    assert np.isclose(float(adv[1, 1]), 0.1, atol=1e-5)
    np.testing.assert_allclose(np.asarray(ret), np.asarray(adv + values), atol=1e-5)


def test_ppo_clipped_loss_and_update() -> None:
    rng = jax.random.PRNGKey(42)
    params = init_actor_critic_params(rng, obs_dim=50, action_dim=12)
    obs = jax.random.normal(rng, (16, 50), dtype=jnp.float32)
    mean, log_std, values = forward_actor_critic_jax(params, obs)
    actions = mean + 0.05

    from pushlab.rl.networks import gaussian_log_prob_jax

    old_logp = gaussian_log_prob_jax(mean, log_std, actions)
    adv = jax.random.normal(rng, (16,), dtype=jnp.float32)
    ret = values + adv

    loss, metrics = ppo_loss_fn(params, obs, actions, old_logp, values, adv, ret)
    assert np.isfinite(float(loss))
    # Since params == old_params, approx_kl and clip_fraction must be 0
    assert np.isclose(float(metrics["approx_kl"]), 0.0, atol=1e-5)
    assert np.isclose(float(metrics["clip_fraction"]), 0.0, atol=1e-5)

    ppo_cfg = PPOConfig(
        num_envs=4,
        rollout_steps=4,
        num_updates=2,
        num_epochs=2,
        num_minibatches=2,
    )
    opt = create_ppo_optimizer(ppo_cfg)
    opt_state = opt.init(params)
    update_fn = make_ppo_update_fn(ppo_cfg, opt)

    batch = RolloutBatch(
        norm_obs=obs.reshape(4, 4, 50),
        actions=actions.reshape(4, 4, 12),
        log_probs=old_logp.reshape(4, 4),
        values=values.reshape(4, 4),
        rewards=jnp.ones((4, 4), dtype=jnp.float32),
        dones=jnp.zeros((4, 4), dtype=jnp.bool_),
        terminated=jnp.zeros((4, 4), dtype=jnp.bool_),
    )
    new_params, _, upd_metrics = update_fn(
        params, opt_state, batch, jnp.zeros((4,), dtype=jnp.float32), rng
    )
    assert np.isfinite(float(upd_metrics["policy_loss"]))
    assert np.isfinite(float(upd_metrics["grad_norm"]))
    assert not np.allclose(
        np.asarray(params["actor_head"]["b"]), np.asarray(new_params["actor_head"]["b"])
    )


def test_running_normalization_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    rng = jax.random.PRNGKey(99)
    params = init_actor_critic_params(rng, obs_dim=50, action_dim=12)
    norm_state = init_running_mean_std(50)
    sample_obs = jax.random.normal(rng, (32, 50), dtype=jnp.float32) * 2.5 + 1.0
    norm_state = update_running_mean_std(norm_state, sample_obs)

    single_obs = np.asarray(sample_obs[0], dtype=np.float32)
    norm_jax = np.asarray(normalize_obs_jax(jnp.asarray(single_obs), norm_state))
    norm_np = normalize_obs_numpy(
        single_obs, np.asarray(norm_state.mean), np.asarray(norm_state.var)
    )
    np.testing.assert_allclose(norm_jax, norm_np, atol=1e-5)

    npz_path, json_path = save_policy_checkpoint(
        params, norm_state, "test_policy", tmp_path, metadata={"seed": 99}
    )
    assert npz_path.exists()
    assert json_path.exists()

    eng_npz: PolicyInferenceEngine = load_policy_checkpoint(npz_path)
    eng_json: PolicyInferenceEngine = load_policy_checkpoint(json_path)

    act_npz, val_npz = eng_npz.predict(single_obs, deterministic=True)
    act_json, val_json = eng_json.predict(single_obs, deterministic=True)

    np.testing.assert_allclose(act_npz, act_json, atol=1e-4)
    assert np.isclose(val_npz, val_json, atol=1e-4)

    mean_jax, _, val_jax = forward_actor_critic_jax(params, jnp.asarray(norm_jax))
    np.testing.assert_allclose(act_npz, np.asarray(mean_jax), atol=1e-5)
    assert np.isclose(val_npz, float(val_jax), atol=1e-5)


def test_mini_train_ppo_execution(tmp_path: Path) -> None:
    cfg = get_baseline_experiment_config(seed=5, num_envs=4, num_updates=2)
    cfg.ppo.rollout_steps = 4
    cfg.ppo.num_epochs = 2
    cfg.ppo.num_minibatches = 2

    res = train_ppo(cfg, output_dir=tmp_path, log_interval=1)
    assert Path(res["npz_path"]).exists()
    assert Path(res["json_path"]).exists()
    assert Path(res["metrics_path"]).exists()
    assert len(res["history"]) == 2
