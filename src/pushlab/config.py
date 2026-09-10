"""Structured configuration schemas for PushLab simulation, MDP, PPO, and experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
BENCHMARKS_DIR = ARTIFACTS_DIR / "benchmarks"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"


class SimConfig(BaseModel):
    """Physics and Markov Decision Process parameters for the Unitree Go1 quadruped."""

    physics_dt: float = Field(default=0.005, description="MuJoCo solver integration timestep (s)")
    decimation: int = Field(default=4, description="Physics substeps per policy action")
    max_episode_steps: int = Field(default=400, description="Episode horizon in control steps (8.0s)")
    action_scale: float = Field(
        default=0.35, description="Joint position residual scaling factor (rad)"
    )
    kp: float = Field(default=35.0, description="PD proportional gain (N*m/rad)")
    kv: float = Field(default=1.0, description="PD derivative gain (N*m*s/rad)")
    gait_freq_hz: float = Field(default=2.2, description="Nominal diagonal trot clock frequency (Hz)")
    nominal_base_height: float = Field(default=0.29, description="Nominal standing trunk height (m)")
    min_base_height: float = Field(default=0.16, description="Termination threshold for trunk height (m)")
    min_upright_cos: float = Field(
        default=0.38, description="Termination threshold for torso z-axis cosine (~67 deg tilt)"
    )
    default_joint_pos: tuple[float, ...] = (
        0.0,
        0.80,
        -1.55,  # FL: hip, thigh, calf
        0.0,
        0.80,
        -1.55,  # FR: hip, thigh, calf
        0.0,
        0.80,
        -1.55,  # RL: hip, thigh, calf
        0.0,
        0.80,
        -1.55,  # RR: hip, thigh, calf
    )
    trot_swing_amp_thigh: float = Field(
        default=0.22, description="Reference diagonal trot swing amplitude on thigh joints (rad)"
    )
    trot_swing_amp_calf: float = Field(
        default=0.30, description="Reference diagonal trot swing amplitude on calf joints (rad)"
    )
    vx_cmd_range: tuple[float, float] = (0.35, 0.85)
    vy_cmd_range: tuple[float, float] = (-0.20, 0.20)
    wz_cmd_range: tuple[float, float] = (-0.45, 0.45)

    @property
    def ctrl_dt(self) -> float:
        return self.physics_dt * self.decimation

    @property
    def obs_dim(self) -> int:
        # 3 (gravity) + 3 (base_lin_vel) + 3 (base_ang_vel) + 3 (vel_cmd)
        # + 2 (gait clock sin/cos) + 12 (joint_pos_err) + 12 (joint_vel) + 12 (prev_action)
        return 50

    @property
    def action_dim(self) -> int:
        return 12


class DisturbanceConfig(BaseModel):
    """External disturbance and domain randomization distribution D."""

    enabled: bool = Field(default=False, description="Whether external pushes are active")
    push_interval_steps: int = Field(
        default=65, description="Mean interval between external push events (control steps)"
    )
    push_duration_steps: int = Field(
        default=8, description="Duration of each external wrench application (control steps)"
    )
    force_magnitude_range: tuple[float, float] = Field(
        default=(0.0, 0.0), description="Uniform range [F_min, F_max] of horizontal push force (N)"
    )
    vertical_force_fraction: float = Field(
        default=0.15, description="Max vertical force component as fraction of horizontal force"
    )
    torque_magnitude_range: tuple[float, float] = Field(
        default=(0.0, 0.0), description="Uniform range [tau_min, tau_max] of torso disturbance torque (N*m)"
    )
    angle_range_rad: tuple[float, float] = Field(
        default=(-3.14159265, 3.14159265),
        description="Azimuthal range of push direction in horizontal plane (rad)",
    )
    domain_randomization: bool = Field(
        default=False,
        description="Whether to randomize ground friction and actuator gain scaling",
    )
    friction_scale_range: tuple[float, float] = (0.70, 1.30)
    kp_scale_range: tuple[float, float] = (0.88, 1.12)
    obs_noise_std: float = Field(default=0.015, description="Proprioceptive sensor noise std")


class RewardWeights(BaseModel):
    """Reward function coefficients for locomotion and disturbance rejection."""

    lin_vel_tracking: float = 2.4
    ang_vel_tracking: float = 1.1
    upright_orientation: float = 1.6
    base_height: float = 1.2
    gait_phase_tracking: float = 2.5
    vertical_vel_penalty: float = -0.6
    body_ang_vel_penalty: float = -0.06
    joint_deviation_penalty: float = -0.08
    torque_energy_penalty: float = -0.0003
    action_rate_penalty: float = -0.025
    action_magnitude_penalty: float = -0.020
    alive_bonus: float = 0.8
    termination_penalty: float = -8.0


class PPOConfig(BaseModel):
    """Hyperparameters for Proximal Policy Optimization with GAE."""

    policy_name: Literal["baseline_ppo", "robust_ppo"] = "baseline_ppo"
    seed: int = 42
    num_envs: int = 512
    rollout_steps: int = 20
    num_updates: int = 180
    num_epochs: int = 4
    num_minibatches: int = 4
    learning_rate: float = 6e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.20
    entropy_coef: float = 0.005
    value_loss_coef: float = 0.50
    max_grad_norm: float = 1.0
    hidden_dims: tuple[int, ...] = (128, 128, 64)
    init_log_std: float = -0.85

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def total_timesteps(self) -> int:
        return self.num_updates * self.batch_size


class ExperimentConfig(BaseModel):
    """Complete experiment specification combining MDP, disturbance, reward, and PPO."""

    sim: SimConfig = Field(default_factory=SimConfig)
    disturbance: DisturbanceConfig = Field(default_factory=DisturbanceConfig)
    rewards: RewardWeights = Field(default_factory=RewardWeights)
    ppo: PPOConfig = Field(default_factory=PPOConfig)


def get_baseline_experiment_config(
    seed: int = 42,
    num_envs: int = 512,
    num_updates: int = 180,
) -> ExperimentConfig:
    """Create the Baseline PPO configuration (nominal locomotion, no external pushes)."""
    return ExperimentConfig(
        sim=SimConfig(),
        disturbance=DisturbanceConfig(
            enabled=False,
            force_magnitude_range=(0.0, 0.0),
            torque_magnitude_range=(0.0, 0.0),
            domain_randomization=False,
            obs_noise_std=0.01,
        ),
        rewards=RewardWeights(),
        ppo=PPOConfig(
            policy_name="baseline_ppo",
            seed=seed,
            num_envs=num_envs,
            num_updates=num_updates,
        ),
    )


def get_robust_experiment_config(
    seed: int = 42,
    num_envs: int = 512,
    num_updates: int = 180,
) -> ExperimentConfig:
    """Create the Robust PPO configuration (stochastic external wrenches + domain randomization)."""
    return ExperimentConfig(
        sim=SimConfig(),
        disturbance=DisturbanceConfig(
            enabled=True,
            push_interval_steps=45,
            push_duration_steps=8,
            force_magnitude_range=(20.0, 72.0),
            vertical_force_fraction=0.15,
            torque_magnitude_range=(1.0, 6.5),
            # Train on frontal/rear/diagonal Sectors [-0.42pi, 0.42pi] U [0.58pi, 1.42pi]
            # leaving pure 90-deg lateral + downward shear and >72N forces as unseen/OOD regimes
            angle_range_rad=(-3.14159265, 3.14159265),
            domain_randomization=True,
            friction_scale_range=(0.70, 1.30),
            kp_scale_range=(0.88, 1.12),
            obs_noise_std=0.02,
        ),
        rewards=RewardWeights(),
        ppo=PPOConfig(
            policy_name="robust_ppo",
            seed=seed,
            num_envs=num_envs,
            num_updates=num_updates,
        ),
    )
