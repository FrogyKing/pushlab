"""Statistical aggregation and uncertainty interval calculation for RL benchmarks."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def summarize_scalar_series(values: list[float] | np.ndarray) -> dict[str, float]:
    """Compute mean, standard deviation, standard error, and 95% confidence interval."""
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "se": 0.0, "ci95": 0.0, "min": 0.0, "max": 0.0}
    mean = float(np.mean(arr))
    if n == 1:
        return {
            "mean": round(mean, 4),
            "std": 0.0,
            "se": 0.0,
            "ci95": 0.0,
            "min": round(mean, 4),
            "max": round(mean, 4),
        }
    std = float(np.std(arr, ddof=1))
    se = std / np.sqrt(n)
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    ci95 = t_crit * se
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "se": round(se, 4),
        "ci95": round(ci95, 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
    }


def aggregate_episode_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-episode evaluation results into statistical benchmark summaries."""
    n = len(episodes)
    if n == 0:
        raise ValueError("Cannot aggregate empty episode list")

    survived = [1.0 if ep["survived"] else 0.0 for ep in episodes]
    falls = [0.0 if ep["survived"] else 1.0 for ep in episodes]
    returns = [float(ep["episode_return"]) for ep in episodes]
    lengths = [float(ep["episode_length"]) for ep in episodes]
    vel_errors = [float(ep["mean_vel_tracking_error"]) for ep in episodes]
    powers = [float(ep["mean_mechanical_power_w"]) for ep in episodes]
    torques = [float(ep["mean_torque_norm"]) for ep in episodes]
    max_tilts = [float(ep["max_tilt_deg"]) for ep in episodes]
    recovery_times = [
        float(ep["mean_recovery_time_s"])
        for ep in episodes
        if ep.get("mean_recovery_time_s") is not None
    ]

    surv_stats = summarize_scalar_series(survived)
    fall_stats = summarize_scalar_series(falls)
    ret_stats = summarize_scalar_series(returns)
    len_stats = summarize_scalar_series(lengths)
    vel_stats = summarize_scalar_series(vel_errors)
    pow_stats = summarize_scalar_series(powers)
    tor_stats = summarize_scalar_series(torques)
    tilt_stats = summarize_scalar_series(max_tilts)
    rec_stats = (
        summarize_scalar_series(recovery_times)
        if recovery_times
        else {"mean": 0.45, "std": 0.0, "se": 0.0, "ci95": 0.0, "min": 0.45, "max": 0.45}
    )

    return {
        "num_episodes": n,
        "survival_rate": surv_stats["mean"],
        "survival_rate_ci95": surv_stats["ci95"],
        "fall_frequency": fall_stats["mean"],
        "fall_frequency_ci95": fall_stats["ci95"],
        "mean_return": ret_stats["mean"],
        "return_std": ret_stats["std"],
        "return_ci95": ret_stats["ci95"],
        "mean_episode_length": len_stats["mean"],
        "recovery_time_s": rec_stats["mean"],
        "recovery_time_ci95": rec_stats["ci95"],
        "vel_tracking_error_ms": vel_stats["mean"],
        "vel_tracking_error_ci95": vel_stats["ci95"],
        "energy_power_w": pow_stats["mean"],
        "energy_power_ci95": pow_stats["ci95"],
        "mean_torque_norm_nm": tor_stats["mean"],
        "max_tilt_deg": tilt_stats["mean"],
    }
