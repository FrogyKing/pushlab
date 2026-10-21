"""Integration tests for FastAPI simulation endpoints and CLI commands."""

from __future__ import annotations

from fastapi.testclient import TestClient

from pushlab.cli import run_cli
from pushlab.server.app import create_app


def test_fastapi_simulation_and_benchmark_endpoints() -> None:
    app = create_app()
    client = TestClient(app)

    health = client.get("/api/health")
    assert health.status_code == 200
    h_data = health.json()
    assert h_data["status"] == "ok"
    assert "baseline_ppo" in h_data["policies"]
    assert "robust_ppo" in h_data["policies"]
    assert h_data["training_runs"]["total_timesteps"] > 0

    bench = client.get("/api/benchmarks")
    assert bench.status_code == 200
    b_data = bench.json()
    assert "regimes" in b_data["benchmarks"]
    assert "intensity_sweep" in b_data["benchmarks"]

    ckpt = client.get("/api/checkpoints/robust_ppo")
    assert ckpt.status_code == 200
    assert ckpt.json()["policy_name"] == "robust_ppo"

    reset_res = client.post(
        "/api/sim/reset",
        json={"policy_name": "robust_ppo", "seed": 42, "vx_cmd": 0.55, "vy_cmd": 0.0, "wz_cmd": 0.0},
    )
    assert reset_res.status_code == 200
    assert "primary" in reset_res.json()

    push_res = client.post(
        "/api/sim/push",
        json={"force_magnitude_n": 50.0, "azimuth_deg": 90.0, "duration_steps": 6},
    )
    assert push_res.status_code == 200
    assert push_res.json()["status"] == "pushed"

    step_res = client.post(
        "/api/sim/step",
        json={"policy_name": "robust_ppo", "duel_mode": True, "num_steps": 2},
    )
    assert step_res.status_code == 200
    s_data = step_res.json()
    assert s_data["primary"]["policy_name"] == "robust_ppo"
    assert s_data["secondary"]["policy_name"] == "baseline_ppo"
    assert len(s_data["primary"]["joint_pos"]) == 12


def test_cli_runs_command() -> None:
    code = run_cli(["runs"])
    assert code == 0
