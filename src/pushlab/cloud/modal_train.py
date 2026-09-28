"""Modal GPU training entry points and artifact retrieval."""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal

from pushlab.cloud.registry import CloudRunTracker
from pushlab.config import (
    CHECKPOINTS_DIR,
    ExperimentConfig,
    get_baseline_experiment_config,
    get_robust_experiment_config,
)

logger = logging.getLogger(__name__)

GPU_TYPE = CloudRunTracker.select_gpu(prefer_l4=True)

modal_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda12]>=0.4.35",
        "mujoco>=3.2.0",
        "mujoco-mjx>=3.2.0",
        "optax>=0.2.4",
        "numpy>=2.0.0",
        "pydantic>=2.9.0",
        "scipy>=1.14.0",
    )
    .add_local_python_source("pushlab", ignore=["__pycache__", "*.pyc"])
)

app = modal.App("pushlab-gpu-rl")
artifacts_volume = modal.Volume.from_name("pushlab-artifacts", create_if_missing=True)


@app.function(
    image=modal_image,
    gpu=GPU_TYPE,
    timeout=900,
    volumes={"/vol/artifacts": artifacts_volume},
)
def run_remote_ppo_job(config_dict: dict[str, Any], log_interval: int = 10) -> dict[str, Any]:
    """Execute GPU-accelerated MJX PPO training inside a Modal GPU container."""
    import subprocess

    import jax

    from pushlab.config import ExperimentConfig
    from pushlab.rl.trainer import train_ppo

    t0 = time.perf_counter()
    started_utc = datetime.now(UTC).isoformat()

    gpu_name = "Unknown GPU"
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            gpu_name = smi.stdout.strip().splitlines()[0]
    except Exception:
        pass

    jax_devices = [str(d) for d in jax.devices()]
    jax_backend = jax.default_backend()

    exp_config = ExperimentConfig.model_validate(config_dict)
    remote_out_dir = Path("/vol/artifacts/checkpoints")
    remote_out_dir.mkdir(parents=True, exist_ok=True)

    result = train_ppo(
        config=exp_config,
        output_dir=remote_out_dir,
        log_interval=log_interval,
    )
    artifacts_volume.commit()

    npz_bytes = Path(result["npz_path"]).read_bytes()
    json_text = Path(result["json_path"]).read_text(encoding="utf-8")
    metrics_text = Path(result["metrics_path"]).read_text(encoding="utf-8")

    runtime_s = time.perf_counter() - t0
    finished_utc = datetime.now(UTC).isoformat()

    return {
        "policy_name": exp_config.ppo.policy_name,
        "gpu_hardware": gpu_name,
        "jax_backend": jax_backend,
        "jax_devices": jax_devices,
        "started_at_utc": started_utc,
        "finished_at_utc": finished_utc,
        "gpu_runtime_seconds": round(runtime_s, 3),
        "npz_bytes": npz_bytes,
        "json_text": json_text,
        "metrics_text": metrics_text,
        "metadata": result["metadata"],
        "history": result["history"],
    }


def estimate_training_seconds(config: ExperimentConfig) -> float:
    """Estimate Modal L4 GPU wall-clock runtime (including XLA JIT compilation)."""
    jit_compile_overhead_s = 38.0
    throughput_steps_per_sec = 16000.0
    step_time_s = config.ppo.total_timesteps / throughput_steps_per_sec
    return jit_compile_overhead_s + step_time_s


def execute_modal_training_job(
    config: ExperimentConfig,
    job_name: str,
    is_smoke_test: bool = False,
    output_dir: Path = CHECKPOINTS_DIR,
    run_tracker: CloudRunTracker | None = None,
    log_interval: int = 10,
) -> dict[str, Any]:
    """Run PPO training on Modal GPU, retrieve checkpoint artifacts, and record telemetry."""
    tracker = run_tracker or CloudRunTracker()
    job_id = f"modal-{uuid.uuid4().hex[:10]}"
    wall_t0 = time.perf_counter()
    started_utc = datetime.now(UTC).isoformat()

    try:
        with app.run():
            remote_res = run_remote_ppo_job.remote(
                config.model_dump(),
                log_interval=log_interval,
            )
    except Exception as exc:
        elapsed = time.perf_counter() - wall_t0
        tracker.record_job_run(
            job_id=job_id,
            job_name=job_name,
            gpu_type=GPU_TYPE,
            runtime_seconds=elapsed,
            total_timesteps=0,
            status="failed",
            started_at_utc=started_utc,
            notes=f"Failed: {exc}",
        )
        raise

    gpu_seconds = float(remote_res.get("gpu_runtime_seconds", time.perf_counter() - wall_t0))
    status = "smoke_test" if is_smoke_test else "completed"

    record = tracker.record_job_run(
        job_id=job_id,
        job_name=job_name,
        gpu_type=GPU_TYPE,
        runtime_seconds=gpu_seconds,
        total_timesteps=config.ppo.total_timesteps,
        status=status,
        started_at_utc=remote_res.get("started_at_utc", started_utc),
        finished_at_utc=remote_res.get("finished_at_utc"),
        notes=(
            f"GPU={remote_res.get('gpu_hardware')} | "
            f"backend={remote_res.get('jax_backend')} | "
            f"steps={config.ppo.total_timesteps}"
        ),
    )

    if not is_smoke_test:
        output_dir.mkdir(parents=True, exist_ok=True)
        policy_name = config.ppo.policy_name
        npz_path = output_dir / f"{policy_name}.npz"
        json_path = output_dir / f"{policy_name}.json"
        metrics_path = output_dir / f"{policy_name}_metrics.json"

        npz_path.write_bytes(remote_res["npz_bytes"])

        json_payload = json.loads(remote_res["json_text"])
        json_payload["metadata"]["gpu_hardware"] = remote_res.get("gpu_hardware")
        json_payload["metadata"]["modal_gpu_type"] = GPU_TYPE
        json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

        metrics_payload = json.loads(remote_res["metrics_text"])
        metrics_payload["metadata"]["gpu_hardware"] = remote_res.get("gpu_hardware")
        metrics_payload["metadata"]["modal_gpu_type"] = GPU_TYPE
        metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    return {
        "job_record": record.model_dump(),
        "gpu_hardware": remote_res.get("gpu_hardware"),
        "jax_backend": remote_res.get("jax_backend"),
        "metadata": remote_res.get("metadata"),
        "history": remote_res.get("history"),
    }


def run_modal_smoke_test() -> dict[str, Any]:
    """Execute a small GPU smoke test on Modal before full policy training."""
    cfg = get_baseline_experiment_config(seed=101, num_envs=128, num_updates=4)
    cfg.ppo.rollout_steps = 8
    return execute_modal_training_job(
        config=cfg,
        job_name="modal_gpu_smoke_test",
        is_smoke_test=True,
        log_interval=2,
    )


def run_full_modal_training_suite(
    seed: int = 42,
    num_envs: int = 512,
    num_updates: int = 180,
) -> dict[str, Any]:
    """Run smoke test followed by Baseline PPO and Robust PPO training on Modal GPU."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    tracker = CloudRunTracker()

    smoke_res = run_modal_smoke_test()
    logger.info(
        "Smoke test verified GPU backend=%s (%s)",
        smoke_res["jax_backend"],
        smoke_res["gpu_hardware"],
    )

    baseline_cfg = get_baseline_experiment_config(
        seed=seed, num_envs=num_envs, num_updates=num_updates
    )
    baseline_res = execute_modal_training_job(
        config=baseline_cfg,
        job_name="train_baseline_ppo",
        is_smoke_test=False,
        run_tracker=tracker,
        log_interval=15,
    )

    robust_cfg = get_robust_experiment_config(
        seed=seed, num_envs=num_envs, num_updates=num_updates
    )
    robust_res = execute_modal_training_job(
        config=robust_cfg,
        job_name="train_robust_ppo",
        is_smoke_test=False,
        run_tracker=tracker,
        log_interval=15,
    )

    registry_after = tracker.load_registry()
    return {
        "smoke_test": smoke_res,
        "baseline_ppo": baseline_res,
        "robust_ppo": robust_res,
        "registry": registry_after.model_dump(),
    }
