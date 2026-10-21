"""Command-line interface for PushLab training, benchmarking, telemetry inspection, and web server."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence

import uvicorn

from pushlab.cloud.modal_train import (
    execute_modal_training_job,
    run_full_modal_training_suite,
    run_modal_smoke_test,
)
from pushlab.cloud.registry import CloudRunTracker
from pushlab.config import get_baseline_experiment_config, get_robust_experiment_config
from pushlab.eval.benchmark import run_comprehensive_benchmark
from pushlab.eval.plotting import generate_all_plots
from pushlab.rl.trainer import train_ppo


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pushlab",
        description="PushLab: GPU-Accelerated Robust Reinforcement Learning for Quadruped Locomotion",
    )
    sub = parser.add_subparsers(dest="command")

    # 1. serve
    serve_p = sub.add_parser("serve", help="Start the FastAPI MuJoCo backend and web UI server")
    serve_p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    serve_p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")

    # 2. train
    train_p = sub.add_parser("train", help="Train Baseline PPO and/or Robust PPO policies")
    train_p.add_argument(
        "--backend",
        choices=["modal", "local"],
        default="modal",
        help="Execution backend: Modal L4 GPU or local JAX",
    )
    train_p.add_argument(
        "--policy",
        choices=["both", "baseline_ppo", "robust_ppo", "smoke"],
        default="both",
    )
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--num-envs", type=int, default=512)
    train_p.add_argument("--num-updates", type=int, default=180)

    # 3. evaluate
    eval_p = sub.add_parser("evaluate", help="Run multi-regime robustness benchmarks and plots")
    eval_p.add_argument("--seeds", type=int, default=12, help="Number of evaluation seeds per regime")
    eval_p.add_argument("--max-steps", type=int, default=240, help="Episode horizon in steps")

    # 4. runs
    sub.add_parser("runs", help="Inspect Modal GPU training run registry and throughput telemetry")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.command == "serve":
        uvicorn.run("pushlab.server.app:app", host=args.host, port=args.port, reload=False)
        return 0

    if args.command == "train":
        if args.backend == "modal":
            if args.policy == "smoke":
                res = run_modal_smoke_test()
                print(json.dumps(res["job_record"], indent=2))
            elif args.policy == "both":
                run_full_modal_training_suite(
                    seed=args.seed,
                    num_envs=args.num_envs,
                    num_updates=args.num_updates,
                )
            elif args.policy == "baseline_ppo":
                cfg = get_baseline_experiment_config(
                    seed=args.seed, num_envs=args.num_envs, num_updates=args.num_updates
                )
                execute_modal_training_job(cfg, "train_baseline_ppo")
            elif args.policy == "robust_ppo":
                cfg = get_robust_experiment_config(
                    seed=args.seed, num_envs=args.num_envs, num_updates=args.num_updates
                )
                execute_modal_training_job(cfg, "train_robust_ppo")
        else:
            if args.policy in ("both", "baseline_ppo"):
                cfg = get_baseline_experiment_config(
                    seed=args.seed, num_envs=args.num_envs, num_updates=args.num_updates
                )
                train_ppo(cfg)
            if args.policy in ("both", "robust_ppo"):
                cfg = get_robust_experiment_config(
                    seed=args.seed, num_envs=args.num_envs, num_updates=args.num_updates
                )
                train_ppo(cfg)
        return 0

    if args.command == "evaluate":
        run_comprehensive_benchmark(num_seeds=args.seeds, max_steps=args.max_steps)
        plots = generate_all_plots()
        print("Generated figures:", [str(p) for p in plots])
        return 0

    if args.command == "runs":
        registry = CloudRunTracker().load_registry()
        print(json.dumps(registry.model_dump(), indent=2))
        return 0

    parser.print_help()
    return 0
