# PushLab System Architecture

## 1. High-Level Architecture Diagram

```text
+-----------------------------------------------------------------------------------+
|                               PUSHLAB ARCHITECTURE                                |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  +-----------------------------+         +-------------------------------------+  |
|  |   Cloud GPU Training Tier   |         |   Local Simulation & Eval Tier      |  |
|  |  (Modal NVIDIA L4 + CUDA12) |         |  (Python 3.12 + MuJoCo 3.17 + JAX)  |  |
|  +--------------+--------------+         +------------------+------------------+  |
|                 |                                           |                     |
|   Run Registry  |  512 Parallel MJX Envs                    | Deterministic       |
|   & Telemetry   v  (jax.vmap + lax.scan)                    v Rollouts (12 Seeds) |
|  +--------------+--------------+         +------------------+------------------+  |
|  | src/pushlab/cloud/          |         | src/pushlab/eval/                   |  |
|  | - modal_train.py            |         | - benchmark.py (6 Regimes + Sweep)  |  |
|  | - registry.py (Run Tracker) |         | - metrics.py (95% CI Statistics)    |  |
|  +--------------+--------------+         | - plotting.py (Publication Figures) |  |
|                 |                        +------------------+------------------+  |
|                 | Retrieves .npz, .json, _metrics.json      |                     |
|                 +---------------------+---------------------+                     |
|                                       |                                           |
|                                       v                                           |
|                 +-------------------------------------------+                     |
|                 |       Model & Benchmark Artifact Store    |                     |
|                 |  - artifacts/checkpoints/baseline_ppo.*   |                     |
|                 |  - artifacts/checkpoints/robust_ppo.*     |                     |
|                 |  - artifacts/benchmarks/benchmark_*.json  |                     |
|                 |  - artifacts/training_runs.json           |                     |
|                 +---------------------+---------------------+                     |
|                                       |                                           |
|                  +--------------------+--------------------+                      |
|                  |                                         |                      |
|                  v                                         v                      |
|  +---------------+---------------+         +---------------+-------------------+  |
|  | FastAPI MuJoCo Backend Server |         | React + Three.js 3D Web Simulator |  |
|  | (src/pushlab/server/app.py)   |         | (web/src/)                        |  |
|  | - /api/sim/step, /push, /reset|<------->| - BrowserPolicyEngine (ELU MLP)   |  |
|  | - /api/benchmarks, /health    |         | - BrowserQuadrupedInstance (50Hz) |  |
|  +-------------------------------+         | - RobotViewport3D (Three.js Rig)  |  |
|                                            | - DisturbanceControls & Telemetry |  |
|                                            +-----------------------------------+  |
+-----------------------------------------------------------------------------------+
```

## 2. Module Boundaries & Responsibilities

| Module | Path | Responsibility |
| :--- | :--- | :--- |
| **Configuration** | `src/pushlab/config.py` | Typed Pydantic schemas (`SimConfig`, `DisturbanceConfig`, `RewardWeights`, `PPOConfig`, `ExperimentConfig`) and experiment builders. |
| **Robot MJCF** | `src/pushlab/assets/go1.xml` | 12-DOF Unitree Go1 quadruped MJCF model with analytical capsule/sphere/box collision primitives optimized for both CPU MuJoCo and GPU MJX. |
| **Disturbances** | `src/pushlab/sim/disturbance.py` | Stochastic 6-DOF wrench generator $d_t \sim \mathcal{D}$ in both NumPy and JAX, plus interactive manual push scheduling. |
| **Rewards** | `src/pushlab/sim/rewards.py` | Parity-verified JAX and NumPy reward computation with velocity tracking, posture, capture-point deadzone stabilization, and energy penalties. |
| **Environments** | `src/pushlab/sim/env.py`, `mjx_env.py` | Single-instance CPU `QuadrupedPushEnv` and GPU-vectorized `MJXQuadrupedEnv` (`jax.vmap` + auto-reset). |
| **RL & PPO** | `src/pushlab/rl/` | Orthogonal Actor-Critic MLP (`networks.py`), Welford observation normalizer (`normalization.py`), GAE & clipped PPO (`ppo.py`), JIT rollout trainer (`trainer.py`), and dual `.npz`/`.json` serializer (`checkpoints.py`). |
| **Modal Cloud** | `src/pushlab/cloud/` | Containerized CUDA 12 + JAX/MJX execution on Modal `NVIDIA L4` GPUs (`modal_train.py`) and persistent run telemetry registry (`registry.py`). |
| **Evaluation** | `src/pushlab/eval/` | 6-regime benchmark runner (`benchmark.py`), Student-t 95% confidence intervals (`metrics.py`), and dark-theme publication plots (`plotting.py`). |
| **Backend API** | `src/pushlab/server/app.py` | FastAPI service exposing live MuJoCo stepping, synchronized dual-policy comparison, and static web asset serving. |
| **Web Frontend** | `web/src/` | Interactive 3D Three.js quadruped viewport, browser-side neural network inference, polar wrench controls, live telemetry sparklines, and research dashboard. |
