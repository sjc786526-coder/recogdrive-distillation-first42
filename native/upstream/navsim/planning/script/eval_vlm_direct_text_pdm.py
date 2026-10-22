#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import lzma
import os
import pickle
import re
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml
from transformers import AutoModel, AutoTokenizer, GenerationConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
INTERNVL_ROOT = REPO_ROOT / "internvl_chat"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(INTERNVL_ROOT) not in sys.path:
    sys.path.insert(0, str(INTERNVL_ROOT))

from navsim.agents.recogdrive.utils.internvl_preprocess import load_image  # noqa: E402
from navsim.common.dataloader import MetricCacheLoader, SceneLoader  # noqa: E402
from navsim.common.dataclasses import SceneFilter, SensorConfig, Trajectory  # noqa: E402
from navsim.evaluate.pdm_score import pdm_score  # noqa: E402
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (  # noqa: E402
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (  # noqa: E402
    PDMSimulator,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402


SYSTEM_MESSAGE = """
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
"""


OFFICIAL_PT_RE = re.compile(
    r"\[PT(?:, )?((?:\([-+]?\d*\.\d+, [-+]?\d*\.\d+, [-+]?\d*\.\d+\)(?:, )?){8})\]"
)
OFFICIAL_COORD_RE = re.compile(r"\(([-+]?\d*\.\d+), ([-+]?\d*\.\d+), ([-+]?\d*\.\d+)\)")
TOLERANT_COORD_RE = re.compile(
    r"\(\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate raw VLM direct text [PT,...] trajectory output with PDM.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--lora-adapter-dir", type=Path, default=None)
    parser.add_argument("--navsim-log-path", type=Path, default=Path("/mnt/data/navsim/test_navsim_logs/test"))
    parser.add_argument("--sensor-blobs-path", type=Path, default=Path("/mnt/data/navsim/test_sensor_blobs/test"))
    parser.add_argument("--scene-filter-yaml", type=Path, default=Path("navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"))
    parser.add_argument("--metric-cache-dir", type=Path, default=Path("navsim/exp/metric_cache"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--cam-type", choices=("single", "multi_view", "cont"), default="single")
    parser.add_argument("--prompt-type", choices=("base", "vel_and_acc"), default="base")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--allow-tolerant-parse", action="store_true")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def configure_env() -> None:
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    os.environ.setdefault("NUPLAN_MAPS_ROOT", "/mnt/data/navsim/maps")
    os.environ.setdefault("OPENSCENE_DATA_ROOT", "/mnt/data/navsim")


def torch_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def format_number(n: float, decimal_places: int = 2) -> str | float:
    if abs(round(float(n), decimal_places)) <= 1e-2:
        return 0.0
    return f"{float(n):+.{decimal_places}f}"


def load_scene_filter(path: Path) -> SceneFilter:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    return SceneFilter(**data)


def build_prompt(agent_input: Any, cam_type: str, prompt_type: str) -> Tuple[str, List[str]]:
    ego_statuses = agent_input.ego_statuses
    cameras = agent_input.cameras
    history_trajectory = []
    for idx in range(4):
        ego_status = ego_statuses[idx]
        history_trajectory.append(
            {
                "x": format_number(ego_status.ego_pose[0]),
                "y": format_number(ego_status.ego_pose[1]),
                "heading": format_number(ego_status.ego_pose[2]),
            }
        )

    high_command_one_hot = ego_statuses[-1].driving_command
    navigation_commands = ["turn left", "go straight", "turn right"]
    command_str = [
        navigation_commands[i]
        for i in range(len(high_command_one_hot))
        if high_command_one_hot[i] == 1
    ]
    command_str = command_str[0] if command_str else "unknown"

    image_paths: List[str] = []
    image_prompt_lines: List[str] = []
    image_prompt_desc = ""
    if cam_type == "single":
        image_paths.append(str(cameras[-1].cam_f0.image))
        image_prompt_lines.append("<FRONT VIEW>:\n<image>\n")
        image_prompt_desc = "1. Visual perception from front camera view\n"
    elif cam_type == "multi_view":
        image_paths.extend(
            [
                str(cameras[-1].cam_f0.image),
                str(cameras[-1].cam_l0.image),
                str(cameras[-1].cam_r0.image),
                str(cameras[-1].cam_l2.image),
                str(cameras[-1].cam_r2.image),
                str(cameras[-1].cam_b0.image),
            ]
        )
        image_prompt_lines.append(
            "<FRONT VIEW>:\n<image>\n<FRONT LEFT VIEW>:\n<image>\n<FRONT RIGHT VIEW>:\n<image>\n"
            "<BACK LEFT VIEW>:\n<image>\n<BACK RIGHT VIEW>:\n<image>\n<BACK VIEW>:\n<image>\n"
        )
        image_prompt_desc = "1. Visual perception from the six surrounding camera views\n"
    else:
        for idx in range(4):
            image_paths.append(str(cameras[idx].cam_f0.image))
            image_prompt_lines.append(f"<FRONT VIEW>Frame-{idx + 1}: <image>\n")
        image_prompt_desc = "1. Visual perception from continuous front camera views of the last 4 timesteps\n"

    common_prompt = (
        "As an autonomous driving system, predict the vehicle's trajectory based on:\n"
        f"{image_prompt_desc}"
        "2. Historical motion context (last 4 timesteps):"
        + " ".join(
            [
                f'   - t-{3 - idx}: ({item["x"]}, {item["y"]}, {item["heading"]})'
                for idx, item in enumerate(history_trajectory)
            ]
        )
        + f"\n3. Active navigation command: [{command_str.upper()}]"
    )
    output_requirements = (
        "\nOutput requirements:\n- Predict 8 future trajectory points\n"
        "- Each point format: (x:float, y:float, heading:float)\n"
        "- Use [PT, ...] to encapsulate the trajectory\n"
        "- Maintain numerical precision to 2 decimal places"
    )
    if prompt_type == "vel_and_acc":
        current = ego_statuses[-1]
        vel_acc_info = (
            f"\n4. Current velocity: ({format_number(current.ego_velocity[0])}, {format_number(current.ego_velocity[1])})"
            f"\n5. Current acceleration: ({format_number(current.ego_acceleration[0])}, {format_number(current.ego_acceleration[1])})"
        )
        question = f"{''.join(image_prompt_lines)}\n{common_prompt}{vel_acc_info}{output_requirements}"
    else:
        question = f"{''.join(['<image>' for _ in image_paths])}\n{common_prompt}{output_requirements}"
    return question, image_paths


def parse_pt_trajectory(text: str, allow_tolerant: bool) -> Tuple[np.ndarray, bool, str]:
    full_match = OFFICIAL_PT_RE.search(text)
    if full_match:
        coords_matches = OFFICIAL_COORD_RE.findall(full_match.group(1))
        if len(coords_matches) == 8:
            coords = np.asarray([tuple(map(float, coord)) for coord in coords_matches], dtype=np.float32)
            return coords, True, "official"
    if allow_tolerant:
        bracket_match = re.search(r"\[PT,?\s*(.*?)\]", text, flags=re.S)
        search_area = bracket_match.group(1) if bracket_match else text
        coords_matches = TOLERANT_COORD_RE.findall(search_area)
        if len(coords_matches) >= 8:
            coords = np.asarray([tuple(map(float, coord)) for coord in coords_matches[:8]], dtype=np.float32)
            return coords, True, "tolerant"
    return np.zeros((8, 3), dtype=np.float32), False, "failed_zero"


class ScannedMetricCacheLoader:
    def __init__(self, cache_path: Path) -> None:
        self.metric_cache_paths = {
            path.parent.name: path
            for path in cache_path.rglob("metric_cache.pkl")
            if path.is_file()
        }
        if not self.metric_cache_paths:
            raise FileNotFoundError(f"No metric_cache.pkl files found under {cache_path}")

    def get_from_token(self, token: str) -> Any:
        with lzma.open(self.metric_cache_paths[token], "rb") as f:
            return pickle.load(f)


def build_metric_cache_loader(cache_path: Path) -> Any:
    try:
        metadata_loader = MetricCacheLoader(cache_path)
    except Exception as exc:
        warnings.warn(f"Could not load metric cache metadata from {cache_path}: {exc!r}; scanning files.")
        return ScannedMetricCacheLoader(cache_path)
    try:
        scanned_loader = ScannedMetricCacheLoader(cache_path)
    except Exception:
        return metadata_loader
    if len(scanned_loader.metric_cache_paths) > len(metadata_loader.metric_cache_paths):
        return scanned_loader
    return metadata_loader


def build_pdm_tools() -> Tuple[TrajectorySampling, PDMSimulator, PDMScorer]:
    proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = PDMSimulator(proposal_sampling=proposal_sampling)
    scorer = PDMScorer(
        proposal_sampling=proposal_sampling,
        config=PDMScorerConfig(
            progress_weight=5.0,
            ttc_weight=5.0,
            comfortable_weight=2.0,
            driving_direction_weight=0.0,
            driving_direction_horizon=1.0,
            driving_direction_compliance_threshold=2.0,
            driving_direction_violation_threshold=6.0,
            stopped_speed_threshold=5e-3,
            progress_distance_threshold=5.0,
        ),
    )
    return proposal_sampling, simulator, scorer


def mean_or_none(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def build_metric_summary(
    args: argparse.Namespace,
    num_samples: int,
    num_parse_ok: int,
    num_pdm_failed: int,
    pdm_metric_values: Dict[str, List[float]],
    partial: bool,
) -> Dict[str, Any]:
    pdms = mean_or_none(pdm_metric_values["score"])
    return {
        "partial": partial,
        "trajectory_output_key": "direct_text_pt",
        "model_path": str(args.model_path),
        "lora_adapter_dir": str(args.lora_adapter_dir) if args.lora_adapter_dir is not None else None,
        "prompt_type": args.prompt_type,
        "cam_type": args.cam_type,
        "precision": args.precision,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "num_samples": num_samples,
        "num_parse_ok": num_parse_ok,
        "parse_success_rate": num_parse_ok / num_samples if num_samples else None,
        "num_pdm_valid": len(pdm_metric_values["score"]),
        "num_pdm_failed": num_pdm_failed,
        "PDMS": pdms,
        "pdm_score": pdms,
        "NC": mean_or_none(pdm_metric_values["no_at_fault_collisions"]),
        "DAC": mean_or_none(pdm_metric_values["drivable_area_compliance"]),
        "TTC": mean_or_none(pdm_metric_values["time_to_collision_within_bound"]),
        "comfort": mean_or_none(pdm_metric_values["comfort"]),
        "EP": mean_or_none(pdm_metric_values["ego_progress"]),
        "DDC": mean_or_none(pdm_metric_values["driving_direction_compliance"]),
    }


def write_outputs(output_dir: Path, predictions: List[Dict[str, Any]], pdm_rows: List[Dict[str, Any]], metrics: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions.json").write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    fields = [
        "sample_token",
        "scene_token",
        "valid",
        "parse_ok",
        "parse_mode",
        "score",
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "comfort",
        "driving_direction_compliance",
        "error",
    ]
    with (output_dir / "pdm_results.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in pdm_rows:
            writer.writerow({key: row.get(key) for key in fields})


def iter_shard(tokens: List[str], num_shards: int, shard_index: int, max_samples: Optional[int]) -> Iterable[str]:
    selected = [token for idx, token in enumerate(tokens) if idx % num_shards == shard_index]
    if max_samples is not None:
        selected = selected[:max_samples]
    yield from selected


def load_model(args: argparse.Namespace, device: torch.device) -> Tuple[Any, Any]:
    dtype = torch_dtype(args.precision) if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
    )
    model = model.eval().to(device)
    if args.lora_adapter_dir is not None:
        from peft import LoraConfig, get_peft_model

        from navsim.agents.recogdrive.vlm_lora_utils import (
            load_lora_adapter_metadata,
            peft_lora_config_kwargs_supported,
        )

        lora_config = load_lora_adapter_metadata(args.lora_adapter_dir)
        supported_kwargs = peft_lora_config_kwargs_supported()
        lora_kwargs: Dict[str, Any] = {
            "r": int(lora_config["r"]),
            "lora_alpha": int(lora_config["alpha"]),
            "lora_dropout": float(lora_config["dropout"]),
            "target_modules": lora_config["target_modules"],
            "bias": str(lora_config["bias"]),
        }
        if "use_rslora" in supported_kwargs:
            lora_kwargs["use_rslora"] = bool(lora_config.get("use_rslora", False))
        if "use_dora" in supported_kwargs:
            lora_kwargs["use_dora"] = bool(lora_config.get("use_dora", False))
        model = get_peft_model(model, LoraConfig(**lora_kwargs))
        state_path = args.lora_adapter_dir / "adapter_model.bin"
        state = torch.load(state_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model_state = model.state_dict()
        filtered: Dict[str, torch.Tensor] = {}
        skipped: List[str] = []
        for key, value in state.items():
            mapped_key = key[len("agent."):] if isinstance(key, str) and key.startswith("agent.") else key
            if isinstance(mapped_key, str) and mapped_key.startswith("backbone.model."):
                mapped_key = mapped_key[len("backbone.model."):]
            elif isinstance(mapped_key, str) and mapped_key.startswith("backbone."):
                mapped_key = mapped_key[len("backbone."):]
            if (
                isinstance(mapped_key, str)
                and "lora_" in mapped_key
                and isinstance(value, torch.Tensor)
                and mapped_key in model_state
                and tuple(model_state[mapped_key].shape) == tuple(value.shape)
            ):
                filtered[mapped_key] = value
            else:
                skipped.append(str(key))
        if not filtered:
            raise RuntimeError(f"No compatible VLM LoRA tensors found in {state_path}")
        model.load_state_dict(filtered, strict=False)
        print(f"Loaded {len(filtered)} VLM LoRA tensors from {state_path}.", flush=True)
        if skipped:
            print(f"Skipped {len(skipped)} incompatible/non-LoRA adapter tensors.", flush=True)
        model = model.merge_and_unload().eval().to(device)
    if hasattr(model, "system_message"):
        model.system_message = SYSTEM_MESSAGE
    return model, tokenizer


def main() -> int:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards).")
    configure_env()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args, device)
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
    }
    generation_config = {key: value for key, value in generation_config.items() if value is not None}

    scene_filter = load_scene_filter(args.scene_filter_yaml)
    scene_loader = SceneLoader(
        sensor_blobs_path=args.sensor_blobs_path,
        data_path=args.navsim_log_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_all_sensors(include=[0, 1, 2, 3]),
        load_image_path=True,
    )
    metric_cache_loader = build_metric_cache_loader(args.metric_cache_dir)
    tokens = sorted(set(scene_loader.tokens) & set(metric_cache_loader.metric_cache_paths))
    tokens_to_evaluate = list(iter_shard(tokens, args.num_shards, args.shard_index, args.max_samples))

    future_sampling, simulator, scorer = build_pdm_tools()
    predictions: List[Dict[str, Any]] = []
    pdm_rows: List[Dict[str, Any]] = []
    pdm_metric_values: Dict[str, List[float]] = {
        "score": [],
        "no_at_fault_collisions": [],
        "drivable_area_compliance": [],
        "ego_progress": [],
        "time_to_collision_within_bound": [],
        "comfort": [],
        "driving_direction_compliance": [],
    }
    num_parse_ok = 0
    num_pdm_failed = 0
    write_json_atomic(
        args.output_dir / "progress.json",
        build_metric_summary(args, 0, num_parse_ok, num_pdm_failed, pdm_metric_values, partial=True),
    )

    for idx, token in enumerate(tokens_to_evaluate):
        record: Dict[str, Any] = {
            "sample_token": token,
            "scene_token": token,
            "trajectory_output_key": "direct_text_pt",
            "model_path": str(args.model_path),
            "lora_adapter_dir": str(args.lora_adapter_dir) if args.lora_adapter_dir is not None else None,
        }
        row: Dict[str, Any] = {"sample_token": token, "scene_token": token, "valid": False}
        try:
            agent_input = scene_loader.get_agent_input_from_token(token)
            question, image_paths = build_prompt(agent_input, args.cam_type, args.prompt_type)
            pixel_values = [load_image(path, max_num=12) for path in image_paths]
            num_patches_list = [item.shape[0] for item in pixel_values]
            pixel_values_tensor = torch.cat(pixel_values, dim=0).to(device=device, dtype=torch_dtype(args.precision))
            with torch.no_grad():
                response = model.chat(
                    tokenizer,
                    pixel_values_tensor,
                    question,
                    generation_config=dict(generation_config),
                    num_patches_list=num_patches_list,
                )
            pred, parse_ok, parse_mode = parse_pt_trajectory(response, args.allow_tolerant_parse)
            num_parse_ok += int(parse_ok)
            metric_cache = metric_cache_loader.get_from_token(token)
            result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=Trajectory(poses=pred),
                future_sampling=future_sampling,
                simulator=simulator,
                scorer=scorer,
            )
            result_dict = asdict(result)
            for key, value in result_dict.items():
                pdm_metric_values[key].append(float(value))
            record.update(
                {
                    "response": response,
                    "parse_ok": parse_ok,
                    "parse_mode": parse_mode,
                    "pred_traj": pred.tolist(),
                    "pdm": result_dict,
                }
            )
            row.update({"valid": True, "parse_ok": parse_ok, "parse_mode": parse_mode, **result_dict})
        except Exception as exc:
            num_pdm_failed += 1
            record.update({"error": repr(exc), "parse_ok": False, "parse_mode": "exception"})
            row.update({"valid": False, "parse_ok": False, "parse_mode": "exception", "error": repr(exc)})
        predictions.append(record)
        pdm_rows.append(row)
        if args.progress_every > 0 and (idx + 1) % args.progress_every == 0:
            progress_metrics = build_metric_summary(
                args,
                len(predictions),
                num_parse_ok,
                num_pdm_failed,
                pdm_metric_values,
                partial=True,
            )
            progress_metrics["last_sample_token"] = token
            write_json_atomic(args.output_dir / "progress.json", progress_metrics)
        if args.save_every > 0 and (idx + 1) % args.save_every == 0:
            metrics = build_metric_summary(
                args,
                len(predictions),
                num_parse_ok,
                num_pdm_failed,
                pdm_metric_values,
                partial=True,
            )
            write_outputs(args.output_dir, predictions, pdm_rows, metrics)
            print(json.dumps(metrics, sort_keys=True), flush=True)

    metrics = build_metric_summary(
        args,
        len(predictions),
        num_parse_ok,
        num_pdm_failed,
        pdm_metric_values,
        partial=False,
    )
    metrics.update(
        {
            "generation_config": generation_config,
            "allow_tolerant_parse": args.allow_tolerant_parse,
            "navsim_log_path": str(args.navsim_log_path),
            "sensor_blobs_path": str(args.sensor_blobs_path),
            "metric_cache_dir": str(args.metric_cache_dir),
        }
    )
    write_outputs(args.output_dir, predictions, pdm_rows, metrics)
    write_json_atomic(args.output_dir / "progress.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
