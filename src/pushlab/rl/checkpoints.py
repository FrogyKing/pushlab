"""Checkpoint serialization and loading in both .npz and portable .json formats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pushlab.rl.networks import PolicyInferenceEngine
from pushlab.rl.normalization import RunningMeanStdState


def save_policy_checkpoint(
    params: dict[str, Any],
    norm_state: RunningMeanStdState,
    policy_name: str,
    output_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Save trained Actor-Critic parameters and observation normalizer to .npz and .json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = metadata or {}

    npz_dict: dict[str, Any] = {
        "obs_mean": np.asarray(norm_state.mean, dtype=np.float32),
        "obs_var": np.asarray(norm_state.var, dtype=np.float32),
        "obs_count": np.asarray(norm_state.count, dtype=np.float32),
        "actor_log_std": np.asarray(params["actor_log_std"], dtype=np.float32),
        "actor_head_w": np.asarray(params["actor_head"]["w"], dtype=np.float32),
        "actor_head_b": np.asarray(params["actor_head"]["b"], dtype=np.float32),
        "critic_head_w": np.asarray(params["critic_head"]["w"], dtype=np.float32),
        "critic_head_b": np.asarray(params["critic_head"]["b"], dtype=np.float32),
        "num_layers": np.array(len(params["actor_layers"]), dtype=np.int32),
    }

    actor_layers_json = []
    critic_layers_json = []
    for idx, (a_layer, c_layer) in enumerate(
        zip(params["actor_layers"], params["critic_layers"], strict=True)
    ):
        aw = np.asarray(a_layer["w"], dtype=np.float32)
        ab = np.asarray(a_layer["b"], dtype=np.float32)
        cw = np.asarray(c_layer["w"], dtype=np.float32)
        cb = np.asarray(c_layer["b"], dtype=np.float32)
        npz_dict[f"actor_layer_{idx}_w"] = aw
        npz_dict[f"actor_layer_{idx}_b"] = ab
        npz_dict[f"critic_layer_{idx}_w"] = cw
        npz_dict[f"critic_layer_{idx}_b"] = cb
        actor_layers_json.append({"w": np.round(aw, 6).tolist(), "b": np.round(ab, 6).tolist()})
        critic_layers_json.append({"w": np.round(cw, 6).tolist(), "b": np.round(cb, 6).tolist()})

    npz_path = output_dir / f"{policy_name}.npz"
    np.savez_compressed(npz_path, **npz_dict)

    json_payload = {
        "policy_name": policy_name,
        "metadata": meta,
        "obs_mean": np.round(npz_dict["obs_mean"], 6).tolist(),
        "obs_var": np.round(npz_dict["obs_var"], 6).tolist(),
        "actor_layers": actor_layers_json,
        "actor_head": {
            "w": np.round(npz_dict["actor_head_w"], 6).tolist(),
            "b": np.round(npz_dict["actor_head_b"], 6).tolist(),
        },
        "actor_log_std": np.round(npz_dict["actor_log_std"], 6).tolist(),
        "critic_layers": critic_layers_json,
        "critic_head": {
            "w": np.round(npz_dict["critic_head_w"], 6).tolist(),
            "b": np.round(npz_dict["critic_head_b"], 6).tolist(),
        },
    }
    json_path = output_dir / f"{policy_name}.json"
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    return npz_path, json_path


def load_policy_checkpoint(checkpoint_path: Path) -> PolicyInferenceEngine:
    """Load a PolicyInferenceEngine from either a .npz or .json checkpoint file."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Policy checkpoint not found: {checkpoint_path}")

    if checkpoint_path.suffix == ".json":
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        params = {
            "actor_layers": payload["actor_layers"],
            "actor_head": payload["actor_head"],
            "actor_log_std": payload["actor_log_std"],
            "critic_layers": payload["critic_layers"],
            "critic_head": payload["critic_head"],
        }
        return PolicyInferenceEngine(
            params=params,
            obs_mean=np.asarray(payload["obs_mean"], dtype=np.float32),
            obs_var=np.asarray(payload["obs_var"], dtype=np.float32),
            policy_name=payload.get("policy_name", checkpoint_path.stem),
            metadata=payload.get("metadata", {}),
        )

    with np.load(checkpoint_path, allow_pickle=False) as data:
        num_layers = int(data["num_layers"])
        actor_layers = [
            {
                "w": data[f"actor_layer_{i}_w"],
                "b": data[f"actor_layer_{i}_b"],
            }
            for i in range(num_layers)
        ]
        critic_layers = [
            {
                "w": data[f"critic_layer_{i}_w"],
                "b": data[f"critic_layer_{i}_b"],
            }
            for i in range(num_layers)
        ]
        params = {
            "actor_layers": actor_layers,
            "actor_head": {"w": data["actor_head_w"], "b": data["actor_head_b"]},
            "actor_log_std": data["actor_log_std"],
            "critic_layers": critic_layers,
            "critic_head": {"w": data["critic_head_w"], "b": data["critic_head_b"]},
        }
        obs_mean = data["obs_mean"]
        obs_var = data["obs_var"]

    json_companion = checkpoint_path.with_suffix(".json")
    metadata: dict[str, Any] = {}
    if json_companion.exists():
        metadata = json.loads(json_companion.read_text(encoding="utf-8")).get("metadata", {})

    return PolicyInferenceEngine(
        params=params,
        obs_mean=obs_mean,
        obs_var=obs_var,
        policy_name=checkpoint_path.stem,
        metadata=metadata,
    )
