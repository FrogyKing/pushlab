"""Principled locomotion and disturbance-rejection reward functions in JAX and NumPy."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from pushlab.config import RewardWeights, SimConfig


def compute_gait_reference_jax(
    phase_rad: jnp.ndarray,
    vel_cmd: jnp.ndarray,
    sim_cfg: SimConfig,
) -> jnp.ndarray:
    """Compute the 12-DOF diagonal trot reference joint angles in JAX."""
    q_def = jnp.array(sim_cfg.default_joint_pos, dtype=jnp.float32)
    speed_scale = jnp.clip(
        jnp.hypot(vel_cmd[0], vel_cmd[1]) / 0.55 + jnp.abs(vel_cmd[2]) * 0.35,
        0.25,
        1.35,
    )
    dir_sign = jnp.where(vel_cmd[0] >= 0.0, 1.0, -1.0)
    s1 = jnp.sin(phase_rad)
    c1 = jnp.cos(phase_rad) * dir_sign
    s2 = -s1
    c2 = -c1

    amp_t = sim_cfg.trot_swing_amp_thigh * speed_scale
    amp_c = sim_cfg.trot_swing_amp_calf * speed_scale

    # Lateral hip offset for commanded vy and yaw rate wz
    hip_lat = jnp.clip(0.18 * vel_cmd[1], -0.15, 0.15)
    hip_yaw = jnp.clip(0.10 * vel_cmd[2], -0.12, 0.12)

    delta = jnp.array(
        [
            hip_lat - hip_yaw * s1,
            -amp_t * c1,
            amp_c * jnp.maximum(0.0, s1),  # FL
            hip_lat + hip_yaw * s2,
            -amp_t * c2,
            amp_c * jnp.maximum(0.0, s2),  # FR
            hip_lat + hip_yaw * s2,
            -amp_t * c2,
            amp_c * jnp.maximum(0.0, s2),  # RL
            hip_lat - hip_yaw * s1,
            -amp_t * c1,
            amp_c * jnp.maximum(0.0, s1),  # RR
        ],
        dtype=jnp.float32,
    )
    return q_def + delta


def compute_gait_reference_numpy(
    phase_rad: float,
    vel_cmd: np.ndarray,
    sim_cfg: SimConfig,
) -> np.ndarray:
    """Compute the 12-DOF diagonal trot reference joint angles in NumPy."""
    q_def = np.array(sim_cfg.default_joint_pos, dtype=np.float64)
    speed_scale = float(
        np.clip(
            np.hypot(vel_cmd[0], vel_cmd[1]) / 0.55 + abs(vel_cmd[2]) * 0.35,
            0.25,
            1.35,
        )
    )
    dir_sign = 1.0 if vel_cmd[0] >= 0.0 else -1.0
    s1 = float(np.sin(phase_rad))
    c1 = float(np.cos(phase_rad)) * dir_sign
    s2 = -s1
    c2 = -c1

    amp_t = sim_cfg.trot_swing_amp_thigh * speed_scale
    amp_c = sim_cfg.trot_swing_amp_calf * speed_scale

    hip_lat = float(np.clip(0.18 * vel_cmd[1], -0.15, 0.15))
    hip_yaw = float(np.clip(0.10 * vel_cmd[2], -0.12, 0.12))

    delta = np.array(
        [
            hip_lat - hip_yaw * s1,
            -amp_t * c1,
            amp_c * max(0.0, s1),  # FL
            hip_lat + hip_yaw * s2,
            -amp_t * c2,
            amp_c * max(0.0, s2),  # FR
            hip_lat + hip_yaw * s2,
            -amp_t * c2,
            amp_c * max(0.0, s2),  # RL
            hip_lat - hip_yaw * s1,
            -amp_t * c1,
            amp_c * max(0.0, s1),  # RR
        ],
        dtype=np.float64,
    )
    return q_def + delta


def _deadzone_jax(x: jnp.ndarray, threshold: float) -> jnp.ndarray:
    return jnp.sign(x) * jnp.maximum(0.0, jnp.abs(x) - threshold)


def _deadzone_numpy(x: float, threshold: float) -> float:
    return float(np.sign(x) * max(0.0, abs(x) - threshold))


def compute_capture_point_offset_jax(
    base_lin_vel_body: jnp.ndarray,
    base_ang_vel_body: jnp.ndarray,
    vel_cmd: jnp.ndarray,
) -> jnp.ndarray:
    """Compute capture-point reactive joint offset for external perturbations beyond nominal sway."""
    vy_err = _deadzone_jax(base_lin_vel_body[1] - vel_cmd[1], 0.18)
    vx_err = _deadzone_jax(base_lin_vel_body[0] - vel_cmd[0], 0.22)
    wx = _deadzone_jax(base_ang_vel_body[0], 0.70)
    wy = _deadzone_jax(base_ang_vel_body[1], 0.70)

    hip_corr = jnp.clip(0.55 * vy_err - 0.15 * wx, -0.35, 0.35)
    knee_ext = jnp.clip(0.24 * vy_err - 0.08 * wx, -0.24, 0.24)
    thigh_pitch = jnp.clip(-0.20 * vx_err - 0.08 * wy, -0.20, 0.20)

    return jnp.array(
        [
            hip_corr,
            thigh_pitch,
            knee_ext,  # FL
            hip_corr,
            thigh_pitch,
            -knee_ext,  # FR
            hip_corr,
            thigh_pitch,
            knee_ext,  # RL
            hip_corr,
            thigh_pitch,
            -knee_ext,  # RR
        ],
        dtype=jnp.float32,
    )


def compute_capture_point_offset_numpy(
    base_lin_vel_body: np.ndarray,
    base_ang_vel_body: np.ndarray,
    vel_cmd: np.ndarray,
) -> np.ndarray:
    """Compute capture-point reactive joint offset in NumPy."""
    vy_err = _deadzone_numpy(float(base_lin_vel_body[1] - vel_cmd[1]), 0.18)
    vx_err = _deadzone_numpy(float(base_lin_vel_body[0] - vel_cmd[0]), 0.22)
    wx = _deadzone_numpy(float(base_ang_vel_body[0]), 0.70)
    wy = _deadzone_numpy(float(base_ang_vel_body[1]), 0.70)

    hip_corr = float(np.clip(0.55 * vy_err - 0.15 * wx, -0.35, 0.35))
    knee_ext = float(np.clip(0.24 * vy_err - 0.08 * wx, -0.24, 0.24))
    thigh_pitch = float(np.clip(-0.20 * vx_err - 0.08 * wy, -0.20, 0.20))

    return np.array(
        [
            hip_corr,
            thigh_pitch,
            knee_ext,  # FL
            hip_corr,
            thigh_pitch,
            -knee_ext,  # FR
            hip_corr,
            thigh_pitch,
            knee_ext,  # RL
            hip_corr,
            thigh_pitch,
            -knee_ext,  # RR
        ],
        dtype=np.float64,
    )


def compute_reward_jax(
    base_pos: jnp.ndarray,
    gravity_body: jnp.ndarray,
    base_lin_vel_body: jnp.ndarray,
    base_ang_vel_body: jnp.ndarray,
    joint_pos: jnp.ndarray,
    q_phase_ref: jnp.ndarray,
    vel_cmd: jnp.ndarray,
    action: jnp.ndarray,
    prev_action: jnp.ndarray,
    torques: jnp.ndarray,
    terminated: jnp.ndarray,
    sim_cfg: SimConfig,
    weights: RewardWeights,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Compute scalar MDP reward and decomposed diagnostic terms in JAX."""
    # 1. Velocity tracking (exponential kernel)
    lin_vel_err_sq = jnp.sum(jnp.square(base_lin_vel_body[:2] - vel_cmd[:2]))
    r_lin_vel = jnp.exp(-lin_vel_err_sq / 0.12)

    ang_vel_err_sq = jnp.square(base_ang_vel_body[2] - vel_cmd[2])
    r_ang_vel = jnp.exp(-ang_vel_err_sq / 0.18)

    # 2. Upright orientation and base height stability
    tilt_sq = jnp.sum(jnp.square(gravity_body[:2]))
    r_upright = jnp.exp(-tilt_sq / 0.08)

    height_err_sq = jnp.square(base_pos[2] - sim_cfg.nominal_base_height)
    r_height = jnp.exp(-height_err_sq / 0.0035)

    # 3. Gait phase + capture-point stabilization tracking
    cp_offset = compute_capture_point_offset_jax(
        base_lin_vel_body, base_ang_vel_body, vel_cmd
    )
    q_ideal = q_phase_ref + cp_offset
    action_offset = sim_cfg.action_scale * jnp.tanh(action)
    gait_err_sq = jnp.sum(jnp.square(joint_pos - q_ideal)) + 1.5 * jnp.sum(
        jnp.square(action_offset - cp_offset)
    )
    r_gait = jnp.exp(-gait_err_sq / 0.18)

    # 4. Stability & smoothness penalties
    p_vert_vel = jnp.square(base_lin_vel_body[2])
    p_ang_vel_xy = jnp.sum(jnp.square(base_ang_vel_body[:2]))
    q_def = jnp.array(sim_cfg.default_joint_pos, dtype=jnp.float32)
    p_joint_dev = jnp.sum(jnp.square(joint_pos - q_def))
    p_torque = jnp.sum(jnp.square(torques))
    p_action_rate = jnp.sum(jnp.square(action - prev_action))
    p_action_mag = jnp.sum(jnp.square(action - cp_offset / sim_cfg.action_scale))

    # 5. Survival & termination
    not_term = 1.0 - terminated.astype(jnp.float32)
    term_flag = terminated.astype(jnp.float32)

    total_reward = (
        weights.lin_vel_tracking * r_lin_vel
        + weights.ang_vel_tracking * r_ang_vel
        + weights.upright_orientation * r_upright
        + weights.base_height * r_height
        + weights.gait_phase_tracking * r_gait
        + weights.vertical_vel_penalty * p_vert_vel
        + weights.body_ang_vel_penalty * p_ang_vel_xy
        + weights.joint_deviation_penalty * p_joint_dev
        + weights.torque_energy_penalty * p_torque
        + weights.action_rate_penalty * p_action_rate
        + weights.action_magnitude_penalty * p_action_mag
        + weights.alive_bonus * not_term
        + weights.termination_penalty * term_flag
    ) * sim_cfg.ctrl_dt * 10.0

    terms = {
        "r_lin_vel": r_lin_vel,
        "r_ang_vel": r_ang_vel,
        "r_upright": r_upright,
        "r_height": r_height,
        "r_gait": r_gait,
        "p_torque": p_torque,
        "p_action_rate": p_action_rate,
    }
    return total_reward, terms


def compute_reward_numpy(
    base_pos: np.ndarray,
    gravity_body: np.ndarray,
    base_lin_vel_body: np.ndarray,
    base_ang_vel_body: np.ndarray,
    joint_pos: np.ndarray,
    q_phase_ref: np.ndarray,
    vel_cmd: np.ndarray,
    action: np.ndarray,
    prev_action: np.ndarray,
    torques: np.ndarray,
    terminated: bool,
    sim_cfg: SimConfig,
    weights: RewardWeights,
) -> tuple[float, dict[str, float]]:
    """Compute scalar MDP reward and decomposed diagnostic terms in NumPy."""
    lin_vel_err_sq = float(np.sum(np.square(base_lin_vel_body[:2] - vel_cmd[:2])))
    r_lin_vel = float(np.exp(-lin_vel_err_sq / 0.12))

    ang_vel_err_sq = float(np.square(base_ang_vel_body[2] - vel_cmd[2]))
    r_ang_vel = float(np.exp(-ang_vel_err_sq / 0.18))

    tilt_sq = float(np.sum(np.square(gravity_body[:2])))
    r_upright = float(np.exp(-tilt_sq / 0.08))

    height_err_sq = float(np.square(base_pos[2] - sim_cfg.nominal_base_height))
    r_height = float(np.exp(-height_err_sq / 0.0035))

    cp_offset = compute_capture_point_offset_numpy(
        base_lin_vel_body, base_ang_vel_body, vel_cmd
    )
    q_ideal = q_phase_ref + cp_offset
    action_offset = sim_cfg.action_scale * np.tanh(action)
    gait_err_sq = float(
        np.sum(np.square(joint_pos - q_ideal))
        + 1.5 * np.sum(np.square(action_offset - cp_offset))
    )
    r_gait = float(np.exp(-gait_err_sq / 0.18))

    p_vert_vel = float(np.square(base_lin_vel_body[2]))
    p_ang_vel_xy = float(np.sum(np.square(base_ang_vel_body[:2])))
    q_def = np.array(sim_cfg.default_joint_pos, dtype=np.float64)
    p_joint_dev = float(np.sum(np.square(joint_pos - q_def)))
    p_torque = float(np.sum(np.square(torques)))
    p_action_rate = float(np.sum(np.square(action - prev_action)))
    p_action_mag = float(
        np.sum(np.square(action - cp_offset / sim_cfg.action_scale))
    )

    not_term = 0.0 if terminated else 1.0
    term_flag = 1.0 if terminated else 0.0

    total_reward = (
        weights.lin_vel_tracking * r_lin_vel
        + weights.ang_vel_tracking * r_ang_vel
        + weights.upright_orientation * r_upright
        + weights.base_height * r_height
        + weights.gait_phase_tracking * r_gait
        + weights.vertical_vel_penalty * p_vert_vel
        + weights.body_ang_vel_penalty * p_ang_vel_xy
        + weights.joint_deviation_penalty * p_joint_dev
        + weights.torque_energy_penalty * p_torque
        + weights.action_rate_penalty * p_action_rate
        + weights.action_magnitude_penalty * p_action_mag
        + weights.alive_bonus * not_term
        + weights.termination_penalty * term_flag
    ) * sim_cfg.ctrl_dt * 10.0

    terms = {
        "r_lin_vel": r_lin_vel,
        "r_ang_vel": r_ang_vel,
        "r_upright": r_upright,
        "r_height": r_height,
        "r_gait": r_gait,
        "p_torque": p_torque,
        "p_action_rate": p_action_rate,
    }
    return float(total_reward), terms
