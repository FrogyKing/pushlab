"""Cloud GPU training run registry and runtime telemetry tracker for Modal compute."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from pushlab.config import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

SUPPORTED_MODAL_GPUS: tuple[str, ...] = (
    "T4",
    "L4",
    "A10G",
    "L40S",
    "A100-40GB",
    "A100-80GB",
    "H100",
)
DEFAULT_REGISTRY_PATH = ARTIFACTS_DIR / "training_runs.json"


class GPUTelemetryRecord(BaseModel):
    """Completed or failed Modal GPU training job entry in the experiment registry."""

    job_id: str
    job_name: str
    gpu_type: str
    started_at_utc: str
    finished_at_utc: str
    runtime_seconds: float
    total_timesteps: int
    throughput_fps: float
    status: Literal["completed", "failed", "smoke_test"]
    notes: str = ""


class TrainingRunRegistry(BaseModel):
    """Persistent registry tracking Modal GPU training runs and throughput."""

    total_runtime_seconds: float = Field(default=0.0)
    total_timesteps: int = Field(default=0)
    jobs: list[GPUTelemetryRecord] = Field(default_factory=list)


class CloudRunTracker:
    """Tracks GPU hardware selection, runtime projections, and completed training runs."""

    def __init__(self, registry_path: Path = DEFAULT_REGISTRY_PATH) -> None:
        self.registry_path = registry_path

    def load_registry(self) -> TrainingRunRegistry:
        if not self.registry_path.exists():
            return TrainingRunRegistry()
        raw = json.loads(self.registry_path.read_text(encoding="utf-8"))
        return TrainingRunRegistry.model_validate(raw)

    def save_registry(self, registry: TrainingRunRegistry) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(
            json.dumps(registry.model_dump(), indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def select_gpu(prefer_l4: bool = True) -> str:
        """Select optimal GPU for JAX/MJX parallel physics simulation."""
        return "L4" if prefer_l4 else "T4"

    def record_job_run(
        self,
        job_id: str,
        job_name: str,
        gpu_type: str,
        runtime_seconds: float,
        total_timesteps: int,
        status: Literal["completed", "failed", "smoke_test"] = "completed",
        started_at_utc: str | None = None,
        finished_at_utc: str | None = None,
        notes: str = "",
    ) -> GPUTelemetryRecord:
        normalized_gpu = gpu_type.upper()
        if normalized_gpu not in SUPPORTED_MODAL_GPUS:
            raise ValueError(
                f"Unsupported Modal GPU type '{gpu_type}'. Supported: {SUPPORTED_MODAL_GPUS}"
            )
        now_iso = datetime.now(UTC).isoformat()
        fps = round(total_timesteps / max(1e-3, runtime_seconds), 1)

        record = GPUTelemetryRecord(
            job_id=job_id,
            job_name=job_name,
            gpu_type=normalized_gpu,
            started_at_utc=started_at_utc or now_iso,
            finished_at_utc=finished_at_utc or now_iso,
            runtime_seconds=round(runtime_seconds, 3),
            total_timesteps=total_timesteps,
            throughput_fps=fps,
            status=status,
            notes=notes,
        )

        registry = self.load_registry()
        registry.jobs.append(record)
        registry.total_runtime_seconds = round(
            sum(j.runtime_seconds for j in registry.jobs), 3
        )
        registry.total_timesteps = sum(j.total_timesteps for j in registry.jobs)
        self.save_registry(registry)

        logger.info(
            "Recorded GPU job '%s' (%s): runtime=%.2fs, steps=%d, fps=%.1f",
            job_name,
            status,
            runtime_seconds,
            total_timesteps,
            fps,
        )
        return record
