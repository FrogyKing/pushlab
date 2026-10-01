"""Unit tests for CloudRunTracker and Modal training runtime estimator."""

from __future__ import annotations

from pathlib import Path

import pytest

from pushlab.cloud.modal_train import estimate_training_seconds
from pushlab.cloud.registry import CloudRunTracker
from pushlab.config import get_baseline_experiment_config


def test_gpu_selection_and_runtime_estimator() -> None:
    gpu = CloudRunTracker.select_gpu(prefer_l4=True)
    assert gpu == "L4"

    cfg = get_baseline_experiment_config(num_envs=512, num_updates=180)
    secs = estimate_training_seconds(cfg)
    assert 80.0 < secs < 300.0


def test_invalid_gpu_rejected(tmp_path: Path) -> None:
    tracker = CloudRunTracker(registry_path=tmp_path / "training_runs.json")
    with pytest.raises(ValueError):
        tracker.record_job_run(
            job_id="bad-1",
            job_name="invalid_gpu_job",
            gpu_type="UNKNOWN_GPU",
            runtime_seconds=10.0,
            total_timesteps=1000,
        )


def test_cloud_registry_persistence(tmp_path: Path) -> None:
    reg_path = tmp_path / "training_runs.json"
    tracker = CloudRunTracker(registry_path=reg_path)

    rec1 = tracker.record_job_run(
        job_id="job-1",
        job_name="smoke_test",
        gpu_type="L4",
        runtime_seconds=36.0,
        total_timesteps=4096,
        status="smoke_test",
    )
    assert rec1.throughput_fps > 0.0

    rec2 = tracker.record_job_run(
        job_id="job-2",
        job_name="train_baseline",
        gpu_type="L4",
        runtime_seconds=78.0,
        total_timesteps=1843200,
        status="completed",
    )
    registry = tracker.load_registry()
    assert len(registry.jobs) == 2
    assert registry.total_timesteps == 4096 + 1843200
    assert pytest.approx(registry.total_runtime_seconds, rel=1e-5) == (
        rec1.runtime_seconds + rec2.runtime_seconds
    )
