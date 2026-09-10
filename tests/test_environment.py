"""Unit tests for QuadrupedPushEnv, MJXQuadrupedEnv, reward parity, and disturbances."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from pushlab.config import (
    DisturbanceConfig,
    get_baseline_experiment_config,
    get_robust_experiment_config,
)
from pushlab.sim.disturbance import DisturbanceSchedule, sample_disturbance_wrench_jax
from pushlab.sim.env import QuadrupedPushEnv, quat_rotate_inverse_numpy
from pushlab.sim.mjx_env import MJXQuadrupedEnv, quat_rotate_inverse_jax
from pushlab.sim.rewards import (
    compute_gait_reference_jax,
    compute_gait_reference_numpy,
    compute_reward_jax,
    compute_reward_numpy,
)


def test_quat_rotate_inverse_parity() -> None:
    quat = np.array([0.9238795, 0.22, -0.18, 0.258], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    vec = np.array([0.4, -0.7, -9.81], dtype=np.float64)

    res_np = quat_rotate_inverse_numpy(quat, vec)
    res_jax = np.asarray(
        quat_rotate_inverse_jax(jnp.asarray(quat, dtype=jnp.float32), jnp.asarray(vec, dtype=jnp.float32))
    )
    np.testing.assert_allclose(res_np, res_jax, atol=1e-5)


def test_gait_and_reward_parity() -> None:
    cfg = get_baseline_experiment_config()
    vel_cmd = np.array([0.65, -0.1, 0.2], dtype=np.float64)
    phase = 1.23

    ref_np = compute_gait_reference_numpy(phase, vel_cmd, cfg.sim)
    ref_jax = np.asarray(
        compute_gait_reference_jax(
            jnp.array(phase, dtype=jnp.float32),
            jnp.asarray(vel_cmd, dtype=jnp.float32),
            cfg.sim,
        )
    )
    np.testing.assert_allclose(ref_np, ref_jax, atol=1e-5)

    base_pos = np.array([0.1, 0.0, 0.30], dtype=np.float64)
    grav = np.array([0.02, -0.03, -0.999], dtype=np.float64)
    lin_vel = np.array([0.60, -0.05, 0.01], dtype=np.float64)
    ang_vel = np.array([0.05, -0.04, 0.18], dtype=np.float64)
    q_pos = ref_np.copy()
    act = np.zeros(12, dtype=np.float64)
    prev_act = np.zeros(12, dtype=np.float64)
    torques = np.ones(12, dtype=np.float64) * 2.5

    r_np, terms_np = compute_reward_numpy(
        base_pos, grav, lin_vel, ang_vel, q_pos, ref_np, vel_cmd, act, prev_act, torques, False, cfg.sim, cfg.rewards
    )
    r_jax, terms_jax = compute_reward_jax(
        jnp.asarray(base_pos, dtype=jnp.float32),
        jnp.asarray(grav, dtype=jnp.float32),
        jnp.asarray(lin_vel, dtype=jnp.float32),
        jnp.asarray(ang_vel, dtype=jnp.float32),
        jnp.asarray(q_pos, dtype=jnp.float32),
        jnp.asarray(ref_np, dtype=jnp.float32),
        jnp.asarray(vel_cmd, dtype=jnp.float32),
        jnp.asarray(act, dtype=jnp.float32),
        jnp.asarray(prev_act, dtype=jnp.float32),
        jnp.asarray(torques, dtype=jnp.float32),
        jnp.array(False),
        cfg.sim,
        cfg.rewards,
    )
    assert np.isclose(r_np, float(r_jax), atol=1e-4)
    for k in terms_np:
        assert np.isclose(terms_np[k], float(terms_jax[k]), atol=1e-4)


def test_disturbance_schedule_and_jax_sampling() -> None:
    dist_cfg = DisturbanceConfig(
        enabled=True,
        push_interval_steps=20,
        push_duration_steps=5,
        force_magnitude_range=(30.0, 50.0),
        torque_magnitude_range=(2.0, 4.0),
    )
    sched = DisturbanceSchedule(dist_cfg, seed=123)
    ev = sched.trigger_manual_push(current_step=5, force_magnitude_n=40.0, azimuth_rad=np.pi / 2, duration_steps=4)
    assert np.isclose(ev.magnitude_n, 40.0)
    w_active = sched.get_wrench_at_step(6)
    assert np.isclose(w_active[1], 40.0, atol=1e-4)
    w_after = sched.get_wrench_at_step(10)
    assert np.allclose(w_after, 0.0)

    w_jax = sample_disturbance_wrench_jax(jax.random.PRNGKey(0), dist_cfg)
    assert w_jax.shape == (6,)
    horiz_mag = float(jnp.linalg.norm(w_jax[:2]))
    assert 30.0 <= horiz_mag <= 50.0


def test_cpu_quadruped_env_rollout() -> None:
    cfg = get_robust_experiment_config(seed=7)
    env = QuadrupedPushEnv(cfg, seed=7)
    obs = env.reset(seed=7, fixed_vel_cmd=(0.55, 0.0, 0.0))
    assert obs.shape == (50,)
    assert np.isfinite(obs).all()

    env.trigger_push(force_magnitude_n=35.0, azimuth_deg=90.0, duration_steps=5)
    for _ in range(15):
        obs, reward, terminated, truncated, info = env.step(np.zeros(12))
        assert obs.shape == (50,)
        assert np.isfinite(reward)
        assert "vel_tracking_error" in info
        if terminated or truncated:
            break

    render_state = env.get_render_state()
    assert len(render_state["joint_pos"]) == 12
    assert len(render_state["foot_positions"]) == 4


def test_mjx_vectorized_env() -> None:
    cfg = get_robust_experiment_config(seed=11, num_envs=4, num_updates=2)
    mjx_env = MJXQuadrupedEnv(cfg)
    state = mjx_env.reset(jax.random.PRNGKey(11), num_envs=4)
    assert state.obs.shape == (4, 50)

    actions = jnp.zeros((4, 12), dtype=jnp.float32)
    next_state = mjx_env.step(state, actions)
    assert next_state.obs.shape == (4, 50)
    assert next_state.reward.shape == (4,)
    assert jnp.all(jnp.isfinite(next_state.obs))
