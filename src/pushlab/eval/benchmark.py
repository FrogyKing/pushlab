"""Rigorous multi-regime robustness benchmark suite and ablation experiments."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pushlab.config import BENCHMARKS_DIR, CHECKPOINTS_DIR, ExperimentConfig
from pushlab.eval.metrics import aggregate_episode_metrics
from pushlab.rl.checkpoints import load_policy_checkpoint
from pushlab.rl.networks import PolicyInferenceEngine
from pushlab.sim.env import QuadrupedPushEnv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkRegime:
    """Specification of an evaluation disturbance regime."""

    regime_id: str
    label: str
    description: str
    force_range_n: tuple[float, float]
    duration_steps: int
    push_steps: tuple[int, ...]
    fixed_azimuths_deg: tuple[float, ...] | None = None
    vertical_shear_ratio: float = 0.0


CANONICAL_REGIMES: tuple[BenchmarkRegime, ...] = (
    BenchmarkRegime(
        regime_id="nominal",
        label="No Disturbance",
        description="Nominal flat-ground locomotion without external pushes (0 N).",
        force_range_n=(0.0, 0.0),
        duration_steps=8,
        push_steps=(),
    ),
    BenchmarkRegime(
        regime_id="weak_pushes",
        label="Weak Pushes",
        description="Low-magnitude omnidirectional external pushes (20-32 N).",
        force_range_n=(20.0, 32.0),
        duration_steps=8,
        push_steps=(45, 110, 175),
    ),
    BenchmarkRegime(
        regime_id="moderate_pushes",
        label="Moderate Pushes",
        description="Moderate external pushes challenging lateral balance (38-48 N).",
        force_range_n=(38.0, 48.0),
        duration_steps=8,
        push_steps=(45, 110, 175),
    ),
    BenchmarkRegime(
        regime_id="strong_pushes",
        label="Strong Pushes",
        description="High-intensity external wrenches near upper training bound (52-64 N).",
        force_range_n=(52.0, 64.0),
        duration_steps=8,
        push_steps=(45, 110, 175),
    ),
    BenchmarkRegime(
        regime_id="unseen_directions",
        label="Unseen Shear Directions",
        description="Pure 90-deg lateral shocks coupled with -30% downward vertical shear (44-58 N).",
        force_range_n=(44.0, 58.0),
        duration_steps=8,
        push_steps=(45, 110, 175),
        fixed_azimuths_deg=(-90.0, 90.0),
        vertical_shear_ratio=-0.30,
    ),
    BenchmarkRegime(
        regime_id="ood_extreme",
        label="OOD Extreme Pushes",
        description="Out-of-distribution forces exceeding the 72 N training ceiling (74-86 N).",
        force_range_n=(74.0, 86.0),
        duration_steps=8,
        push_steps=(45, 110, 175),
        fixed_azimuths_deg=(-90.0, 90.0),
    ),
)


def evaluate_single_episode(
    policy: PolicyInferenceEngine,
    regime: BenchmarkRegime,
    seed: int,
    max_steps: int = 240,
    action_gain: float = 1.0,
) -> dict[str, Any]:
    """Run one deterministic evaluation episode under a specified disturbance regime."""
    cfg = ExperimentConfig()
    cfg.sim.max_episode_steps = max_steps
    cfg.disturbance.enabled = False
    cfg.disturbance.obs_noise_std = 0.0

    env = QuadrupedPushEnv(config=cfg, seed=seed)
    obs = env.reset(seed=seed, fixed_vel_cmd=(0.55, 0.0, 0.0))
    rng = np.random.default_rng(seed + 5000)

    total_return = 0.0
    vel_errors: list[float] = []
    powers: list[float] = []
    torques: list[float] = []
    tilts_deg: list[float] = []
    recovery_times: list[float] = []

    terminated = False
    step_idx = 0

    for step_idx in range(max_steps):
        if step_idx in regime.push_steps and regime.force_range_n[1] > 0.0:
            f_mag = float(rng.uniform(regime.force_range_n[0], regime.force_range_n[1]))
            if regime.fixed_azimuths_deg is not None:
                az_deg = float(
                    regime.fixed_azimuths_deg[
                        (step_idx + seed) % len(regime.fixed_azimuths_deg)
                    ]
                )
            else:
                # Sample across lateral-dominant angles where quadruped stability is critical
                choices = (-120.0, -90.0, -60.0, 60.0, 90.0, 120.0)
                az_deg = float(choices[(step_idx + seed) % len(choices)])
            fz = f_mag * regime.vertical_shear_ratio
            env.trigger_push(
                force_magnitude_n=f_mag,
                azimuth_deg=az_deg,
                duration_steps=regime.duration_steps,
                vertical_force_n=fz,
            )

        action, _ = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action * action_gain)
        total_return += reward
        vel_errors.append(float(info["vel_tracking_error"]))
        powers.append(float(info["mechanical_power_w"]))
        torques.append(float(info["torque_norm"]))
        tilt_deg = float(np.hypot(np.rad2deg(info["roll_rad"]), np.rad2deg(info["pitch_rad"])))
        tilts_deg.append(tilt_deg)

        if info["recovery_time_s"] is not None:
            recovery_times.append(float(info["recovery_time_s"]))

        if terminated or truncated:
            break

    return {
        "seed": seed,
        "survived": not terminated,
        "episode_return": round(total_return, 4),
        "episode_length": step_idx + 1,
        "mean_vel_tracking_error": float(np.mean(vel_errors)) if vel_errors else 0.0,
        "mean_mechanical_power_w": float(np.mean(powers)) if powers else 0.0,
        "mean_torque_norm": float(np.mean(torques)) if torques else 0.0,
        "max_tilt_deg": float(np.max(tilts_deg)) if tilts_deg else 0.0,
        "mean_recovery_time_s": (
            float(np.mean(recovery_times))
            if recovery_times
            else (0.0 if regime.force_range_n[1] == 0 else None)
        ),
    }


def run_comprehensive_benchmark(
    checkpoints_dir: Path = CHECKPOINTS_DIR,
    output_dir: Path = BENCHMARKS_DIR,
    num_seeds: int = 12,
    max_steps: int = 240,
) -> dict[str, Any]:
    """Execute full multi-regime, intensity-sweep, directional, and ablation benchmarks."""
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_policy = load_policy_checkpoint(checkpoints_dir / "baseline_ppo.npz")
    robust_policy = load_policy_checkpoint(checkpoints_dir / "robust_ppo.npz")
    policies = {
        "baseline_ppo": baseline_policy,
        "robust_ppo": robust_policy,
    }

    # 1. Canonical Regimes
    regime_results: dict[str, dict[str, Any]] = {}
    for regime in CANONICAL_REGIMES:
        regime_results[regime.regime_id] = {
            "label": regime.label,
            "description": regime.description,
            "force_range_n": list(regime.force_range_n),
            "policies": {},
        }
        for p_name, policy in policies.items():
            episodes = [
                evaluate_single_episode(policy, regime, seed=s, max_steps=max_steps)
                for s in range(num_seeds)
            ]
            summary = aggregate_episode_metrics(episodes)
            regime_results[regime.regime_id]["policies"][p_name] = summary
            logger.info(
                "Regime %-18s | %-12s | survival=%.1f%% | return=%.1f | rec=%.2fs | vel_err=%.3f",
                regime.regime_id,
                p_name,
                summary["survival_rate"] * 100.0,
                summary["mean_return"],
                summary["recovery_time_s"],
                summary["vel_tracking_error_ms"],
            )

    # 2. Force Intensity Sweep (0 N to 92 N)
    sweep_forces = [0.0, 20.0, 32.0, 42.0, 52.0, 62.0, 74.0, 84.0, 94.0]
    intensity_sweep: list[dict[str, Any]] = []
    for f_val in sweep_forces:
        sweep_reg = BenchmarkRegime(
            regime_id=f"force_{int(f_val)}N",
            label=f"{int(f_val)} N",
            description=f"Fixed {f_val} N lateral disturbance sweep",
            force_range_n=(f_val, f_val),
            duration_steps=8,
            push_steps=() if f_val == 0.0 else (45, 110, 175),
            fixed_azimuths_deg=(-90.0, 90.0),
        )
        point_entry: dict[str, Any] = {"force_n": f_val, "policies": {}}
        for p_name, policy in policies.items():
            episodes = [
                evaluate_single_episode(policy, sweep_reg, seed=s, max_steps=max_steps)
                for s in range(num_seeds)
            ]
            point_entry["policies"][p_name] = aggregate_episode_metrics(episodes)
        intensity_sweep.append(point_entry)

    # 3. Directional Polar Sweep at 56 N
    azimuth_angles = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    directional_sweep: list[dict[str, Any]] = []
    for angle in azimuth_angles:
        dir_reg = BenchmarkRegime(
            regime_id=f"angle_{int(angle)}deg",
            label=f"{int(angle)} deg",
            description=f"56 N push at azimuth {angle} deg",
            force_range_n=(56.0, 56.0),
            duration_steps=8,
            push_steps=(45, 110, 175),
            fixed_azimuths_deg=(angle,),
        )
        angle_entry: dict[str, Any] = {"azimuth_deg": angle, "policies": {}}
        for p_name, policy in policies.items():
            episodes = [
                evaluate_single_episode(policy, dir_reg, seed=s, max_steps=max_steps)
                for s in range(8)
            ]
            angle_entry["policies"][p_name] = aggregate_episode_metrics(episodes)
        directional_sweep.append(angle_entry)

    # 4. Ablation Study (Feedback Action Gain Scaling at 56 N)
    ablation_reg = BenchmarkRegime(
        regime_id="ablation_56N",
        label="56 N Lateral Push Ablation",
        description="Ablating closed-loop policy feedback gain alpha in {0.0, 0.5, 1.0}",
        force_range_n=(56.0, 56.0),
        duration_steps=8,
        push_steps=(45, 110, 175),
        fixed_azimuths_deg=(-90.0, 90.0),
    )
    ablation_results: list[dict[str, Any]] = []
    for variant_name, p_name, gain in [
        ("Open-Loop Gait (No Policy, alpha=0.0)", "baseline_ppo", 0.0),
        ("Baseline PPO (Full Policy, alpha=1.0)", "baseline_ppo", 1.0),
        ("Robust PPO (Half Gain, alpha=0.5)", "robust_ppo", 0.5),
        ("Robust PPO (Full Policy, alpha=1.0)", "robust_ppo", 1.0),
    ]:
        episodes = [
            evaluate_single_episode(
                policies[p_name],
                ablation_reg,
                seed=s,
                max_steps=max_steps,
                action_gain=gain,
            )
            for s in range(num_seeds)
        ]
        ablation_results.append(
            {
                "variant": variant_name,
                "base_policy": p_name,
                "action_gain": gain,
                "metrics": aggregate_episode_metrics(episodes),
            }
        )

    payload = {
        "num_seeds_per_regime": num_seeds,
        "episode_horizon_steps": max_steps,
        "training_force_ceiling_n": 72.0,
        "regimes": regime_results,
        "intensity_sweep": intensity_sweep,
        "directional_sweep": directional_sweep,
        "ablation_study": ablation_results,
    }

    out_path = output_dir / "benchmark_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
