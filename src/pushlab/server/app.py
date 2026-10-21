"""FastAPI backend providing live MuJoCo physics stepping, policy inference, and benchmark APIs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pushlab.cloud.registry import CloudRunTracker
from pushlab.config import (
    BENCHMARKS_DIR,
    CHECKPOINTS_DIR,
    FIGURES_DIR,
    PROJECT_ROOT,
    ExperimentConfig,
)
from pushlab.rl.checkpoints import load_policy_checkpoint
from pushlab.rl.networks import PolicyInferenceEngine
from pushlab.sim.env import QuadrupedPushEnv


class ResetRequest(BaseModel):
    policy_name: Literal["baseline_ppo", "robust_ppo"] = "robust_ppo"
    seed: int = 42
    vx_cmd: float = Field(default=0.55, ge=-0.6, le=1.2)
    vy_cmd: float = Field(default=0.0, ge=-0.5, le=0.5)
    wz_cmd: float = Field(default=0.0, ge=-1.0, le=1.0)


class StepRequest(BaseModel):
    policy_name: Literal["baseline_ppo", "robust_ppo"] = "robust_ppo"
    duel_mode: bool = False
    vx_cmd: float | None = None
    vy_cmd: float | None = None
    wz_cmd: float | None = None
    num_steps: int = Field(default=1, ge=1, le=10)


class PushRequest(BaseModel):
    force_magnitude_n: float = Field(default=52.0, ge=0.0, le=150.0)
    azimuth_deg: float = Field(default=90.0, ge=-360.0, le=360.0)
    duration_steps: int = Field(default=8, ge=1, le=30)
    vertical_force_n: float = Field(default=0.0, ge=-60.0, le=60.0)


class SimulationSession:
    """Manages synchronized MuJoCo environments for Baseline PPO and Robust PPO."""

    def __init__(self) -> None:
        cfg = ExperimentConfig()
        cfg.sim.max_episode_steps = 10_000
        cfg.disturbance.enabled = False
        cfg.disturbance.obs_noise_std = 0.0

        self.envs: dict[str, QuadrupedPushEnv] = {
            "baseline_ppo": QuadrupedPushEnv(config=cfg, seed=42),
            "robust_ppo": QuadrupedPushEnv(config=cfg, seed=42),
        }
        self.obs: dict[str, np.ndarray] = {
            "baseline_ppo": self.envs["baseline_ppo"].reset(seed=42, fixed_vel_cmd=(0.55, 0.0, 0.0)),
            "robust_ppo": self.envs["robust_ppo"].reset(seed=42, fixed_vel_cmd=(0.55, 0.0, 0.0)),
        }
        self.cumulative_return: dict[str, float] = {
            "baseline_ppo": 0.0,
            "robust_ppo": 0.0,
        }
        self.fall_count: dict[str, int] = {
            "baseline_ppo": 0,
            "robust_ppo": 0,
        }
        self.is_terminated: dict[str, bool] = {
            "baseline_ppo": False,
            "robust_ppo": False,
        }
        self.last_state_cache: dict[str, dict[str, Any]] = {}
        self.policies: dict[str, PolicyInferenceEngine] = {
            "baseline_ppo": load_policy_checkpoint(CHECKPOINTS_DIR / "baseline_ppo.npz"),
            "robust_ppo": load_policy_checkpoint(CHECKPOINTS_DIR / "robust_ppo.npz"),
        }

    def reset(
        self,
        seed: int = 42,
        vel_cmd: tuple[float, float, float] = (0.55, 0.0, 0.0),
    ) -> dict[str, Any]:
        for name, env in self.envs.items():
            self.obs[name] = env.reset(seed=seed, fixed_vel_cmd=vel_cmd)
            self.cumulative_return[name] = 0.0
            self.is_terminated[name] = False
            self.last_state_cache.pop(name, None)
        return self.get_full_snapshot("robust_ppo", duel_mode=True)

    def apply_push(
        self,
        force_magnitude_n: float,
        azimuth_deg: float,
        duration_steps: int = 8,
        vertical_force_n: float = 0.0,
    ) -> dict[str, Any]:
        for name, env in self.envs.items():
            if not self.is_terminated[name]:
                env.trigger_push(
                    force_magnitude_n=force_magnitude_n,
                    azimuth_deg=azimuth_deg,
                    duration_steps=duration_steps,
                    vertical_force_n=vertical_force_n,
                )
        return {
            "status": "pushed",
            "force_magnitude_n": force_magnitude_n,
            "azimuth_deg": azimuth_deg,
            "duration_steps": duration_steps,
        }

    def step_policy(self, name: str, num_steps: int = 1) -> dict[str, Any]:
        if self.is_terminated[name] and name in self.last_state_cache:
            cached = dict(self.last_state_cache[name])
            cached["external_force"] = [0.0, 0.0, 0.0]
            cached["external_torque"] = [0.0, 0.0, 0.0]
            return cached

        env = self.envs[name]
        policy = self.policies[name]

        last_reward = 0.0
        last_val = 0.0
        last_act = np.zeros(12, dtype=np.float32)
        terminated = False
        info: dict[str, Any] = {}

        for _ in range(num_steps):
            last_act, last_val = policy.predict(self.obs[name], deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(last_act)
            self.obs[name] = next_obs
            self.cumulative_return[name] += reward
            last_reward = reward
            if terminated:
                self.is_terminated[name] = True
                self.fall_count[name] += 1
                break
            if truncated:
                break

        state = env.get_render_state()
        state["policy_name"] = name
        state["step_reward"] = round(float(last_reward), 4)
        state["cumulative_return"] = round(float(self.cumulative_return[name]), 2)
        state["critic_value"] = round(float(last_val), 3)
        state["action"] = np.round(last_act, 4).tolist()
        state["terminated"] = self.is_terminated[name]
        state["fall_count"] = self.fall_count[name]
        state["reward_terms"] = info.get("reward_terms", {})
        self.last_state_cache[name] = state
        return state

    def get_full_snapshot(self, primary_policy: str, duel_mode: bool = False) -> dict[str, Any]:
        primary_env = self.envs[primary_policy]
        primary_pol = self.policies[primary_policy]
        act, val = primary_pol.predict(self.obs[primary_policy], deterministic=True)
        primary_state = primary_env.get_render_state()
        primary_state["policy_name"] = primary_policy
        primary_state["step_reward"] = 0.0
        primary_state["cumulative_return"] = round(
            float(self.cumulative_return[primary_policy]), 2
        )
        primary_state["critic_value"] = round(float(val), 3)
        primary_state["action"] = np.round(act, 4).tolist()
        primary_state["terminated"] = False
        primary_state["fall_count"] = self.fall_count[primary_policy]

        result: dict[str, Any] = {"primary": primary_state}
        if duel_mode:
            other = "baseline_ppo" if primary_policy == "robust_ppo" else "robust_ppo"
            other_env = self.envs[other]
            other_pol = self.policies[other]
            o_act, o_val = other_pol.predict(self.obs[other], deterministic=True)
            other_state = other_env.get_render_state()
            other_state["policy_name"] = other
            other_state["step_reward"] = 0.0
            other_state["cumulative_return"] = round(float(self.cumulative_return[other]), 2)
            other_state["critic_value"] = round(float(o_val), 3)
            other_state["action"] = np.round(o_act, 4).tolist()
            other_state["terminated"] = False
            other_state["fall_count"] = self.fall_count[other]
            result["secondary"] = other_state
        return result


def create_app() -> FastAPI:
    app = FastAPI(
        title="PushLab Robotics & RL API",
        version="0.1.0",
        description="Real-time MuJoCo quadruped disturbance simulation and policy evaluation API",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    session = SimulationSession()

    @app.get("/api/health")
    def get_health() -> dict[str, Any]:
        registry = CloudRunTracker().load_registry()
        return {
            "status": "ok",
            "mujoco_version": mujoco.__version__,
            "policies": list(session.policies.keys()),
            "training_runs": {
                "total_runtime_seconds": registry.total_runtime_seconds,
                "total_timesteps": registry.total_timesteps,
                "num_jobs": len(registry.jobs),
            },
        }

    @app.get("/api/benchmarks")
    def get_benchmarks() -> dict[str, Any]:
        bench_path = BENCHMARKS_DIR / "benchmark_results.json"
        if not bench_path.exists():
            raise HTTPException(status_code=404, detail="Benchmark results not found")

        benchmark_data = json.loads(bench_path.read_text(encoding="utf-8"))
        b_metrics = json.loads(
            (CHECKPOINTS_DIR / "baseline_ppo_metrics.json").read_text(encoding="utf-8")
        )
        r_metrics = json.loads(
            (CHECKPOINTS_DIR / "robust_ppo_metrics.json").read_text(encoding="utf-8")
        )
        registry = CloudRunTracker().load_registry().model_dump()

        return {
            "benchmarks": benchmark_data,
            "training": {
                "baseline_ppo": b_metrics,
                "robust_ppo": r_metrics,
            },
            "training_runs": registry,
        }

    @app.get("/api/checkpoints/{policy_name}")
    def get_checkpoint(policy_name: str) -> dict[str, Any]:
        if policy_name not in ("baseline_ppo", "robust_ppo"):
            raise HTTPException(status_code=404, detail=f"Unknown policy '{policy_name}'")
        json_path = CHECKPOINTS_DIR / f"{policy_name}.json"
        if not json_path.exists():
            raise HTTPException(status_code=404, detail="Checkpoint file missing")
        return json.loads(json_path.read_text(encoding="utf-8"))

    @app.post("/api/sim/reset")
    def reset_sim(req: ResetRequest) -> dict[str, Any]:
        return session.reset(
            seed=req.seed,
            vel_cmd=(req.vx_cmd, req.vy_cmd, req.wz_cmd),
        )

    @app.post("/api/sim/push")
    def push_sim(req: PushRequest) -> dict[str, Any]:
        return session.apply_push(
            force_magnitude_n=req.force_magnitude_n,
            azimuth_deg=req.azimuth_deg,
            duration_steps=req.duration_steps,
            vertical_force_n=req.vertical_force_n,
        )

    @app.post("/api/sim/step")
    def step_sim(req: StepRequest) -> dict[str, Any]:
        for env in session.envs.values():
            if req.vx_cmd is not None:
                env.vel_cmd[0] = req.vx_cmd
            if req.vy_cmd is not None:
                env.vel_cmd[1] = req.vy_cmd
            if req.wz_cmd is not None:
                env.vel_cmd[2] = req.wz_cmd

        primary_state = session.step_policy(req.policy_name, num_steps=req.num_steps)
        result: dict[str, Any] = {"primary": primary_state}
        if req.duel_mode:
            other = "baseline_ppo" if req.policy_name == "robust_ppo" else "robust_ppo"
            result["secondary"] = session.step_policy(other, num_steps=req.num_steps)
        return result

    @app.get("/api/figures/{filename}")
    def get_figure(filename: str) -> FileResponse:
        safe_name = Path(filename).name
        fig_path = FIGURES_DIR / safe_name
        if not fig_path.exists():
            raise HTTPException(status_code=404, detail=f"Figure {safe_name} not found")
        return FileResponse(fig_path, media_type="image/png")

    web_dist = PROJECT_ROOT / "web" / "dist"
    if web_dist.exists():
        app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="web")

    return app


app = create_app()
