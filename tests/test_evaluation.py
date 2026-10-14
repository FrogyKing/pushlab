"""Unit tests for robustness benchmark metrics, statistical intervals, and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pushlab.config import CHECKPOINTS_DIR
from pushlab.eval.benchmark import CANONICAL_REGIMES, evaluate_single_episode
from pushlab.eval.metrics import aggregate_episode_metrics, summarize_scalar_series
from pushlab.eval.plotting import generate_all_plots
from pushlab.rl.checkpoints import load_policy_checkpoint


def test_summarize_scalar_series_confidence_intervals() -> None:
    vals = [10.0, 12.0, 14.0, 16.0, 18.0]
    stats = summarize_scalar_series(vals)
    assert np.isclose(stats["mean"], 14.0)
    assert stats["std"] > 0.0
    assert stats["ci95"] > stats["se"] > 0.0
    assert stats["min"] == 10.0
    assert stats["max"] == 18.0


def test_aggregate_episode_metrics() -> None:
    episodes = [
        {
            "survived": True,
            "episode_return": 240.0,
            "episode_length": 240,
            "mean_vel_tracking_error": 0.14,
            "mean_mechanical_power_w": 18.5,
            "mean_torque_norm": 11.2,
            "max_tilt_deg": 6.5,
            "mean_recovery_time_s": 0.12,
        },
        {
            "survived": False,
            "episode_return": 110.0,
            "episode_length": 95,
            "mean_vel_tracking_error": 0.38,
            "mean_mechanical_power_w": 24.0,
            "mean_torque_norm": 15.0,
            "max_tilt_deg": 68.0,
            "mean_recovery_time_s": None,
        },
    ]
    agg = aggregate_episode_metrics(episodes)
    assert agg["num_episodes"] == 2
    assert np.isclose(agg["survival_rate"], 0.5)
    assert np.isclose(agg["fall_frequency"], 0.5)
    assert np.isclose(agg["mean_return"], 175.0)
    assert np.isclose(agg["recovery_time_s"], 0.12)


def test_trained_policies_benchmark_sanity(tmp_path: Path) -> None:
    b_pol = load_policy_checkpoint(CHECKPOINTS_DIR / "baseline_ppo.npz")
    r_pol = load_policy_checkpoint(CHECKPOINTS_DIR / "robust_ppo.npz")

    strong_regime = CANONICAL_REGIMES[3]  # strong_pushes
    ep_b = evaluate_single_episode(b_pol, strong_regime, seed=0, max_steps=120)
    ep_r = evaluate_single_episode(r_pol, strong_regime, seed=0, max_steps=120)

    assert ep_r["survived"] is True
    assert ep_r["episode_return"] > ep_b["episode_return"]

    plots = generate_all_plots(output_dir=tmp_path)
    assert len(plots) == 4
    for p in plots:
        assert p.exists() and p.stat().st_size > 10_000
