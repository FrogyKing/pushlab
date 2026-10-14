"""Publication-quality scientific visualization generator for PushLab benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pushlab.config import BENCHMARKS_DIR, CHECKPOINTS_DIR, FIGURES_DIR

BASELINE_COLOR = "#38bdf8"  # Cyan blue
ROBUST_COLOR = "#10b981"  # Emerald green
ACCENT_RED = "#f43f5e"
BG_DARK = "#0b0f19"
PANEL_DARK = "#111827"
GRID_COLOR = "#1f2937"
TEXT_MAIN = "#f3f4f6"
TEXT_MUTED = "#9ca3af"


def _apply_dark_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG_DARK,
            "axes.facecolor": PANEL_DARK,
            "axes.edgecolor": "#374151",
            "axes.labelcolor": TEXT_MAIN,
            "axes.titlecolor": TEXT_MAIN,
            "xtick.color": TEXT_MUTED,
            "ytick.color": TEXT_MUTED,
            "text.color": TEXT_MAIN,
            "grid.color": GRID_COLOR,
            "grid.linestyle": "--",
            "grid.alpha": 0.65,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.facecolor": "#1f2937",
            "legend.edgecolor": "#374151",
        }
    )


def generate_robustness_curve_plot(
    benchmark_data: dict[str, Any],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot survival rate and episode return across disturbance force intensities."""
    _apply_dark_publication_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep = benchmark_data["intensity_sweep"]
    forces = np.array([pt["force_n"] for pt in sweep], dtype=np.float64)

    b_surv = np.array([pt["policies"]["baseline_ppo"]["survival_rate"] * 100.0 for pt in sweep])
    b_surv_ci = np.array(
        [pt["policies"]["baseline_ppo"]["survival_rate_ci95"] * 100.0 for pt in sweep]
    )
    r_surv = np.array([pt["policies"]["robust_ppo"]["survival_rate"] * 100.0 for pt in sweep])
    r_surv_ci = np.array(
        [pt["policies"]["robust_ppo"]["survival_rate_ci95"] * 100.0 for pt in sweep]
    )

    b_ret = np.array([pt["policies"]["baseline_ppo"]["mean_return"] for pt in sweep])
    b_ret_ci = np.array([pt["policies"]["baseline_ppo"]["return_ci95"] for pt in sweep])
    r_ret = np.array([pt["policies"]["robust_ppo"]["mean_return"] for pt in sweep])
    r_ret_ci = np.array([pt["policies"]["robust_ppo"]["return_ci95"] for pt in sweep])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=200)

    # Panel 1: Survival Rate
    ax1.plot(
        forces,
        b_surv,
        marker="o",
        linewidth=2.4,
        color=BASELINE_COLOR,
        label="Baseline PPO (Nominal Training)",
    )
    ax1.fill_between(
        forces,
        np.clip(b_surv - b_surv_ci, 0, 100),
        np.clip(b_surv + b_surv_ci, 0, 100),
        color=BASELINE_COLOR,
        alpha=0.18,
    )
    ax1.plot(
        forces,
        r_surv,
        marker="s",
        linewidth=2.4,
        color=ROBUST_COLOR,
        label="Robust PPO (Disturbance-Aware)",
    )
    ax1.fill_between(
        forces,
        np.clip(r_surv - r_surv_ci, 0, 100),
        np.clip(r_surv + r_surv_ci, 0, 100),
        color=ROBUST_COLOR,
        alpha=0.18,
    )
    ax1.axvline(72.0, color=ACCENT_RED, linestyle=":", linewidth=1.8, label="Training Max (72 N)")
    ax1.axvspan(72.0, 96.0, color=ACCENT_RED, alpha=0.07, label="Out-of-Distribution (OOD)")
    ax1.set_title("Episode Survival Rate vs. External Push Force", fontweight="bold")
    ax1.set_xlabel("External Lateral Wrench Force Magnitude (N)")
    ax1.set_ylabel("Survival Rate (%)")
    ax1.set_ylim(-5, 106)
    ax1.set_xlim(-2, 96)
    ax1.grid(True)
    ax1.legend(loc="lower left", fontsize=8.5)

    # Panel 2: Mean Episode Return
    ax2.plot(
        forces,
        b_ret,
        marker="o",
        linewidth=2.4,
        color=BASELINE_COLOR,
        label="Baseline PPO",
    )
    ax2.fill_between(
        forces,
        b_ret - b_ret_ci,
        b_ret + b_ret_ci,
        color=BASELINE_COLOR,
        alpha=0.18,
    )
    ax2.plot(
        forces,
        r_ret,
        marker="s",
        linewidth=2.4,
        color=ROBUST_COLOR,
        label="Robust PPO",
    )
    ax2.fill_between(
        forces,
        r_ret - r_ret_ci,
        r_ret + r_ret_ci,
        color=ROBUST_COLOR,
        alpha=0.18,
    )
    ax2.axvline(72.0, color=ACCENT_RED, linestyle=":", linewidth=1.8, label="Training Ceiling (72 N)")
    ax2.axvspan(72.0, 96.0, color=ACCENT_RED, alpha=0.07)
    ax2.set_title("Mean Episode Return vs. External Push Force", fontweight="bold")
    ax2.set_xlabel("External Lateral Wrench Force Magnitude (N)")
    ax2.set_ylabel("Mean Cumulative Episode Return")
    ax2.set_xlim(-2, 96)
    ax2.grid(True)
    ax2.legend(loc="lower left", fontsize=8.5)

    fig.suptitle(
        "PushLab Robustness Curve: Disturbance-Aware vs. Nominal PPO (95% CI)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()
    out_path = output_dir / "robustness_curve.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_regime_comparison_plot(
    benchmark_data: dict[str, Any],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Generate 4-panel bar chart comparing policies across the 6 evaluation regimes."""
    _apply_dark_publication_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    regimes = benchmark_data["regimes"]
    regime_keys = list(regimes.keys())
    labels = [regimes[k]["label"].replace(" ", "\n") for k in regime_keys]
    x = np.arange(len(regime_keys))
    width = 0.36

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), dpi=200)
    specs = [
        ("survival_rate", "Survival Rate (%)", 100.0, axes[0, 0]),
        ("mean_return", "Mean Episode Return", 1.0, axes[0, 1]),
        ("recovery_time_s", "Mean Recovery Time (s) [Lower is Better]", 1.0, axes[1, 0]),
        (
            "vel_tracking_error_ms",
            "Velocity Tracking Error (m/s) [Lower is Better]",
            1.0,
            axes[1, 1],
        ),
    ]

    for metric_key, title, scale, ax in specs:
        b_vals = [regimes[k]["policies"]["baseline_ppo"][metric_key] * scale for k in regime_keys]
        r_vals = [regimes[k]["policies"]["robust_ppo"][metric_key] * scale for k in regime_keys]

        ax.bar(
            x - width / 2,
            b_vals,
            width,
            color=BASELINE_COLOR,
            alpha=0.88,
            label="Baseline PPO",
        )
        ax.bar(
            x + width / 2,
            r_vals,
            width,
            color=ROBUST_COLOR,
            alpha=0.88,
            label="Robust PPO",
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.grid(True, axis="y")
        ax.legend(fontsize=8.5)

    fig.suptitle(
        "PushLab Multi-Regime Benchmark Suite: Baseline PPO vs. Robust PPO",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout()
    out_path = output_dir / "benchmark_regimes.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_directional_and_ablation_plot(
    benchmark_data: dict[str, Any],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot polar directional robustness and feedback gain ablation study."""
    _apply_dark_publication_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    dir_sweep = benchmark_data["directional_sweep"]
    angles_deg = [d["azimuth_deg"] for d in dir_sweep]
    angles_rad = np.deg2rad([*angles_deg, angles_deg[0]])

    b_dir_ret = [d["policies"]["baseline_ppo"]["mean_return"] for d in dir_sweep]
    b_dir_ret.append(b_dir_ret[0])
    r_dir_ret = [d["policies"]["robust_ppo"]["mean_return"] for d in dir_sweep]
    r_dir_ret.append(r_dir_ret[0])

    fig = plt.figure(figsize=(13, 5.4), dpi=200)
    ax_polar = fig.add_subplot(1, 2, 1, projection="polar")
    ax_polar.set_facecolor(PANEL_DARK)
    ax_polar.plot(
        angles_rad,
        b_dir_ret,
        marker="o",
        linewidth=2.2,
        color=BASELINE_COLOR,
        label="Baseline PPO",
    )
    ax_polar.fill(angles_rad, b_dir_ret, color=BASELINE_COLOR, alpha=0.18)
    ax_polar.plot(
        angles_rad,
        r_dir_ret,
        marker="s",
        linewidth=2.2,
        color=ROBUST_COLOR,
        label="Robust PPO",
    )
    ax_polar.fill(angles_rad, r_dir_ret, color=ROBUST_COLOR, alpha=0.22)
    ax_polar.set_title(
        "Directional Push Robustness (56 N Wrench)\nEpisode Return by Azimuth Angle",
        fontweight="bold",
        pad=16,
    )
    ax_polar.legend(loc="lower right", bbox_to_anchor=(1.18, -0.08), fontsize=8.5)

    # Panel 2: Ablation study
    ax_abl = fig.add_subplot(1, 2, 2)
    abl = benchmark_data["ablation_study"]
    v_labels = [a["variant"].replace(" (", "\n(") for a in abl]
    v_surv = [a["metrics"]["survival_rate"] * 100.0 for a in abl]
    v_ret = [a["metrics"]["mean_return"] for a in abl]

    y_pos = np.arange(len(abl))
    colors = ["#6b7280", BASELINE_COLOR, "#34d399", ROBUST_COLOR]
    bars = ax_abl.barh(y_pos, v_surv, color=colors, height=0.55, alpha=0.9)
    ax_abl.set_yticks(y_pos)
    ax_abl.set_yticklabels(v_labels, fontsize=8.5)
    ax_abl.set_xlabel("Survival Rate (%) under 56 N Lateral Pushes")
    ax_abl.set_xlim(0, 115)
    ax_abl.set_title("Policy Feedback Ablation Study (56 N Lateral Pushes)", fontweight="bold")
    ax_abl.grid(True, axis="x")

    for bar, ret_val in zip(bars, v_ret, strict=True):
        w = bar.get_width()
        ax_abl.text(
            w + 2.0,
            bar.get_y() + bar.get_height() / 2.0,
            f"{w:.0f}% (Ret: {ret_val:.0f})",
            va="center",
            fontsize=8.5,
            color=TEXT_MAIN,
        )

    fig.tight_layout()
    out_path = output_dir / "directional_polar.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_training_curves_plot(
    checkpoints_dir: Path = CHECKPOINTS_DIR,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot Modal L4 GPU training progression for Baseline PPO and Robust PPO."""
    _apply_dark_publication_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    b_meta = json.loads(
        (checkpoints_dir / "baseline_ppo_metrics.json").read_text(encoding="utf-8")
    )
    r_meta = json.loads(
        (checkpoints_dir / "robust_ppo_metrics.json").read_text(encoding="utf-8")
    )

    b_hist = b_meta["history"]
    r_hist = r_meta["history"]

    b_steps = np.array([h["timesteps"] / 1e6 for h in b_hist])
    b_ret = np.array([h["mean_episode_return"] for h in b_hist])
    b_ev = np.array([h["explained_variance"] for h in b_hist])

    r_steps = np.array([h["timesteps"] / 1e6 for h in r_hist])
    r_ret = np.array([h["mean_episode_return"] for h in r_hist])
    r_ev = np.array([h["explained_variance"] for h in r_hist])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0), dpi=200)

    ax1.plot(
        b_steps,
        b_ret,
        marker="o",
        linewidth=2.2,
        color=BASELINE_COLOR,
        label="Baseline PPO (0 N Training)",
    )
    ax1.plot(
        r_steps,
        r_ret,
        marker="s",
        linewidth=2.2,
        color=ROBUST_COLOR,
        label="Robust PPO (20-72 N Stochastic Pushes)",
    )
    ax1.set_title("GPU Training Learning Curve (NVIDIA L4 + MuJoCo MJX)", fontweight="bold")
    ax1.set_xlabel("Environment Timesteps (Millions)")
    ax1.set_ylabel("Mean Training Episode Return")
    ax1.grid(True)
    ax1.legend(loc="lower right", fontsize=8.5)

    ax2.plot(
        b_steps,
        b_ev,
        marker="o",
        linewidth=2.2,
        color=BASELINE_COLOR,
        label="Baseline Critic Explained Variance",
    )
    ax2.plot(
        r_steps,
        r_ev,
        marker="s",
        linewidth=2.2,
        color=ROBUST_COLOR,
        label="Robust Critic Explained Variance",
    )
    ax2.set_title("Critic Value Function Fit (Explained Variance)", fontweight="bold")
    ax2.set_xlabel("Environment Timesteps (Millions)")
    ax2.set_ylabel("Explained Variance $1 - \\mathrm{Var}(R - V)/\\mathrm{Var}(R)$")
    ax2.set_ylim(-0.05, 1.0)
    ax2.grid(True)
    ax2.legend(loc="lower right", fontsize=8.5)

    fig.suptitle(
        "PushLab Modal Cloud GPU Training Diagnostics (512 Parallel MJX Environments)",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()
    out_path = output_dir / "training_curves.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_all_plots(
    benchmark_path: Path = BENCHMARKS_DIR / "benchmark_results.json",
    checkpoints_dir: Path = CHECKPOINTS_DIR,
    output_dir: Path = FIGURES_DIR,
) -> list[Path]:
    """Generate all four publication-quality figures from benchmark and training artifacts."""
    benchmark_data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    p1 = generate_robustness_curve_plot(benchmark_data, output_dir=output_dir)
    p2 = generate_regime_comparison_plot(benchmark_data, output_dir=output_dir)
    p3 = generate_directional_and_ablation_plot(benchmark_data, output_dir=output_dir)
    p4 = generate_training_curves_plot(checkpoints_dir=checkpoints_dir, output_dir=output_dir)
    return [p1, p2, p3, p4]
