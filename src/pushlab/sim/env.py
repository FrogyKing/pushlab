"""MuJoCo quadruped locomotion and push-recovery MDP environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from pushlab.config import (
    DisturbanceConfig,
    ExperimentConfig,
    RewardWeights,
    SimConfig,
)
from pushlab.sim.disturbance import DisturbanceSchedule, WrenchEvent
from pushlab.sim.rewards import compute_gait_reference_numpy, compute_reward_numpy

MODEL_XML_PATH = Path(__file__).resolve().parents[1] / "assets" / "go1.xml"


def quat_rotate_inverse_numpy(quat_wxyz: np.ndarray, vec_xyz: np.ndarray) -> np.ndarray:
    """Rotate a 3D world vector into the body frame defined by unit quaternion (w, x, y, z)."""
    w, x, y, z = quat_wxyz
    q_vec = np.array([x, y, z], dtype=np.float64)
    a = vec_xyz * (2.0 * w * w - 1.0)
    b = 2.0 * w * np.cross(q_vec, vec_xyz)
    c = 2.0 * np.dot(q_vec, vec_xyz) * q_vec
    return a - b + c


def quat_to_rpy_numpy(quat_wxyz: np.ndarray) -> tuple[float, float, float]:
    """Convert unit quaternion (w, x, y, z) to (roll, pitch, yaw) in radians."""
    w, x, y, z = quat_wxyz
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))

    sinp = float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    pitch = float(np.arcsin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))
    return roll, pitch, yaw


class QuadrupedPushEnv:
    """High-precision MuJoCo MDP environment for Unitree Go1 disturbance rejection."""

    def __init__(
        self,
        config: ExperimentConfig | None = None,
        seed: int = 42,
    ) -> None:
        self.config = config or ExperimentConfig()
        self.sim_cfg: SimConfig = self.config.sim
        self.dist_cfg: DisturbanceConfig = self.config.disturbance
        self.reward_weights: RewardWeights = self.config.rewards

        self.mj_model = mujoco.MjModel.from_xml_path(str(MODEL_XML_PATH))
        self.mj_model.opt.timestep = self.sim_cfg.physics_dt
        self.mj_data = mujoco.MjData(self.mj_model)

        self.trunk_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "trunk"
        )
        self.foot_site_ids = [
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_foot_site")
            for leg in ("FL", "FR", "RL", "RR")
        ]

        self.rng = np.random.default_rng(seed)
        self.disturbance = DisturbanceSchedule(self.dist_cfg, seed=seed)

        self.step_count: int = 0
        self.phase_rad: float = 0.0
        self.vel_cmd = np.array([0.60, 0.0, 0.0], dtype=np.float64)
        self.prev_action = np.zeros(self.sim_cfg.action_dim, dtype=np.float64)
        self.last_torques = np.zeros(self.sim_cfg.action_dim, dtype=np.float64)
        self.last_wrench = np.zeros(6, dtype=np.float64)
        self.kp_scale: float = 1.0
        self._base_friction = self.mj_model.geom_friction.copy()

        # Recovery tracking telemetry
        self._post_push_perturbed: bool = False
        self._push_end_step: int = -1
        self.last_recovery_time_s: float | None = None

    @property
    def obs_dim(self) -> int:
        return self.sim_cfg.obs_dim

    @property
    def action_dim(self) -> int:
        return self.sim_cfg.action_dim

    def reset(
        self,
        seed: int | None = None,
        fixed_vel_cmd: tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        """Reset the simulation to nominal keyframe with slight initial state jitter."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.disturbance.reset(seed=seed + 1000)
        else:
            self.disturbance.reset()

        mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)

        # Small initial joint perturbation
        q_noise = self.rng.uniform(-0.025, 0.025, size=self.sim_cfg.action_dim)
        self.mj_data.qpos[7:] += q_noise
        self.mj_data.qvel[:] = self.rng.uniform(-0.03, 0.03, size=self.mj_model.nv)
        self.mj_data.xfrc_applied[:] = 0.0

        if self.dist_cfg.domain_randomization:
            f_scale = float(self.rng.uniform(*self.dist_cfg.friction_scale_range))
            self.mj_model.geom_friction[:, 0] = self._base_friction[:, 0] * f_scale
            self.kp_scale = float(self.rng.uniform(*self.dist_cfg.kp_scale_range))
        else:
            self.mj_model.geom_friction[:] = self._base_friction
            self.kp_scale = 1.0

        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.step_count = 0
        self.phase_rad = float(self.rng.uniform(0.0, 2.0 * np.pi))
        if fixed_vel_cmd is not None:
            self.vel_cmd = np.array(fixed_vel_cmd, dtype=np.float64)
        else:
            vx = float(self.rng.uniform(*self.sim_cfg.vx_cmd_range))
            vy = float(self.rng.uniform(*self.sim_cfg.vy_cmd_range))
            wz = float(self.rng.uniform(*self.sim_cfg.wz_cmd_range))
            self.vel_cmd = np.array([vx, vy, wz], dtype=np.float64)

        self.prev_action = np.zeros(self.sim_cfg.action_dim, dtype=np.float64)
        self.last_torques = np.zeros(self.sim_cfg.action_dim, dtype=np.float64)
        self.last_wrench = np.zeros(6, dtype=np.float64)
        self._post_push_perturbed = False
        self._push_end_step = -1
        self.last_recovery_time_s = None

        return self._get_obs(add_noise=self.dist_cfg.obs_noise_std > 0)

    def trigger_push(
        self,
        force_magnitude_n: float,
        azimuth_deg: float,
        duration_steps: int = 8,
        vertical_force_n: float = 0.0,
    ) -> WrenchEvent:
        """Apply an interactive external push wrench to the robot trunk."""
        azimuth_rad = float(np.deg2rad(azimuth_deg))
        event = self.disturbance.trigger_manual_push(
            current_step=self.step_count,
            force_magnitude_n=force_magnitude_n,
            azimuth_rad=azimuth_rad,
            duration_steps=duration_steps,
            vertical_force_n=vertical_force_n,
        )
        self._post_push_perturbed = True
        self._push_end_step = event.end_step
        return event

    def _get_body_kinematics(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        quat = self.mj_data.qpos[3:7]
        gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        gravity_body = quat_rotate_inverse_numpy(quat, gravity_world)
        lin_vel_world = self.mj_data.qvel[0:3]
        ang_vel_world = self.mj_data.qvel[3:6]
        base_lin_vel_body = quat_rotate_inverse_numpy(quat, lin_vel_world)
        base_ang_vel_body = quat_rotate_inverse_numpy(quat, ang_vel_world)
        return quat, gravity_body, base_lin_vel_body, base_ang_vel_body

    def _get_obs(self, add_noise: bool = False) -> np.ndarray:
        _, gravity_body, base_lin_vel_body, base_ang_vel_body = self._get_body_kinematics()
        q_def = np.array(self.sim_cfg.default_joint_pos, dtype=np.float64)
        joint_pos_err = self.mj_data.qpos[7:] - q_def
        joint_vel_scaled = self.mj_data.qvel[6:] * 0.05
        clock = np.array([np.sin(self.phase_rad), np.cos(self.phase_rad)], dtype=np.float64)

        obs = np.concatenate(
            [
                gravity_body,  # 3
                base_lin_vel_body,  # 3
                base_ang_vel_body * 0.25,  # 3
                self.vel_cmd,  # 3
                clock,  # 2
                joint_pos_err,  # 12
                joint_vel_scaled,  # 12
                self.prev_action,  # 12
            ]
        ).astype(np.float32)

        if add_noise and self.dist_cfg.obs_noise_std > 0:
            noise = self.rng.normal(0.0, self.dist_cfg.obs_noise_std, size=obs.shape).astype(
                np.float32
            )
            obs = obs + noise

        return np.clip(obs, -10.0, 10.0)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one control step (decimation physics substeps) and return MDP transition."""
        raw_action = np.asarray(action, dtype=np.float64).reshape(self.sim_cfg.action_dim)
        clipped_action = np.clip(raw_action, -2.5, 2.5)
        action_offset = self.sim_cfg.action_scale * np.tanh(clipped_action)

        q_phase_ref = compute_gait_reference_numpy(
            self.phase_rad, self.vel_cmd, self.sim_cfg
        )
        q_target = q_phase_ref + action_offset

        wrench = self.disturbance.get_wrench_at_step(self.step_count)
        self.last_wrench = wrench
        self.mj_data.xfrc_applied[:] = 0.0
        self.mj_data.xfrc_applied[self.trunk_body_id, :] = wrench

        if np.linalg.norm(wrench[:3]) > 1.0:
            self._post_push_perturbed = True
            if self.disturbance.active_event is not None:
                self._push_end_step = self.disturbance.active_event.end_step

        kp_eff = self.sim_cfg.kp * self.kp_scale
        kv_eff = self.sim_cfg.kv
        torque_accum = np.zeros(self.sim_cfg.action_dim, dtype=np.float64)

        for _ in range(self.sim_cfg.decimation):
            q_cur = self.mj_data.qpos[7:]
            dq_cur = self.mj_data.qvel[6:]
            tau = kp_eff * (q_target - q_cur) - kv_eff * dq_cur
            tau = np.clip(tau, -28.0, 28.0)
            self.mj_data.ctrl[:] = tau
            torque_accum += tau
            mujoco.mj_step(self.mj_model, self.mj_data)

        self.last_torques = torque_accum / float(self.sim_cfg.decimation)
        self.step_count += 1
        self.phase_rad = float(
            (self.phase_rad + 2.0 * np.pi * self.sim_cfg.gait_freq_hz * self.sim_cfg.ctrl_dt)
            % (2.0 * np.pi)
        )

        quat, gravity_body, base_lin_vel_body, base_ang_vel_body = self._get_body_kinematics()
        base_pos = self.mj_data.qpos[0:3].copy()
        upright_cos = float(-gravity_body[2])

        terminated = bool(
            base_pos[2] < self.sim_cfg.min_base_height
            or upright_cos < self.sim_cfg.min_upright_cos
            or not np.isfinite(self.mj_data.qpos).all()
        )
        truncated = bool(self.step_count >= self.sim_cfg.max_episode_steps)

        # Track recovery time after push ends
        if (
            self._post_push_perturbed
            and not terminated
            and self.step_count >= self._push_end_step
        ):
            vel_err = float(np.linalg.norm(base_lin_vel_body[:2] - self.vel_cmd[:2]))
            tilt_mag = float(np.linalg.norm(gravity_body[:2]))
            if vel_err < 0.25 and tilt_mag < 0.16:
                self.last_recovery_time_s = float(
                    (self.step_count - self._push_end_step + 1) * self.sim_cfg.ctrl_dt
                )
                self._post_push_perturbed = False

        reward, reward_terms = compute_reward_numpy(
            base_pos=base_pos,
            gravity_body=gravity_body,
            base_lin_vel_body=base_lin_vel_body,
            base_ang_vel_body=base_ang_vel_body,
            joint_pos=self.mj_data.qpos[7:],
            q_phase_ref=q_phase_ref,
            vel_cmd=self.vel_cmd,
            action=clipped_action,
            prev_action=self.prev_action,
            torques=self.last_torques,
            terminated=terminated,
            sim_cfg=self.sim_cfg,
            weights=self.reward_weights,
        )

        self.prev_action = clipped_action.copy()
        obs = self._get_obs(add_noise=self.dist_cfg.obs_noise_std > 0)

        roll, pitch, yaw = quat_to_rpy_numpy(quat)
        mechanical_power_w = float(np.sum(np.abs(self.last_torques * self.mj_data.qvel[6:])))
        vel_tracking_err = float(np.linalg.norm(base_lin_vel_body[:2] - self.vel_cmd[:2]))

        info: dict[str, Any] = {
            "step": self.step_count,
            "time_s": round(self.step_count * self.sim_cfg.ctrl_dt, 3),
            "base_pos": base_pos.tolist(),
            "base_lin_vel_body": base_lin_vel_body.tolist(),
            "base_ang_vel_body": base_ang_vel_body.tolist(),
            "vel_cmd": self.vel_cmd.tolist(),
            "roll_rad": roll,
            "pitch_rad": pitch,
            "yaw_rad": yaw,
            "upright_cos": upright_cos,
            "vel_tracking_error": vel_tracking_err,
            "mechanical_power_w": mechanical_power_w,
            "torque_norm": float(np.linalg.norm(self.last_torques)),
            "external_wrench": self.last_wrench.tolist(),
            "recovery_time_s": self.last_recovery_time_s,
            "terminated": terminated,
            "truncated": truncated,
            "reward_terms": reward_terms,
        }
        return obs, reward, terminated, truncated, info

    def get_render_state(self) -> dict[str, Any]:
        """Extract complete 3D kinematic and dynamic state for interactive WebGL visualization."""
        quat, gravity_body, base_lin_vel_body, base_ang_vel_body = self._get_body_kinematics()
        roll, pitch, yaw = quat_to_rpy_numpy(quat)

        foot_positions = []
        foot_contacts = []
        for site_id in self.foot_site_ids:
            pos = self.mj_data.site_xpos[site_id].tolist()
            foot_positions.append(pos)
            foot_contacts.append(bool(pos[2] <= 0.032))

        return {
            "step": self.step_count,
            "time_s": round(self.step_count * self.sim_cfg.ctrl_dt, 3),
            "base_pos": self.mj_data.qpos[0:3].tolist(),
            "base_quat_wxyz": quat.tolist(),
            "roll_deg": round(float(np.rad2deg(roll)), 2),
            "pitch_deg": round(float(np.rad2deg(pitch)), 2),
            "yaw_deg": round(float(np.rad2deg(yaw)), 2),
            "upright_cos": round(float(-gravity_body[2]), 4),
            "base_lin_vel": base_lin_vel_body.round(4).tolist(),
            "base_ang_vel": base_ang_vel_body.round(4).tolist(),
            "vel_cmd": self.vel_cmd.round(4).tolist(),
            "joint_pos": self.mj_data.qpos[7:].round(4).tolist(),
            "joint_vel": self.mj_data.qvel[6:].round(4).tolist(),
            "torques": self.last_torques.round(3).tolist(),
            "foot_positions": foot_positions,
            "foot_contacts": foot_contacts,
            "external_force": self.last_wrench[:3].round(2).tolist(),
            "external_torque": self.last_wrench[3:].round(2).tolist(),
            "mechanical_power_w": round(
                float(np.sum(np.abs(self.last_torques * self.mj_data.qvel[6:]))), 2
            ),
            "vel_tracking_error": round(
                float(np.linalg.norm(base_lin_vel_body[:2] - self.vel_cmd[:2])), 4
            ),
            "recovery_time_s": self.last_recovery_time_s,
        }
