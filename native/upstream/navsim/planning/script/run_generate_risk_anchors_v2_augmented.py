from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch.distributed as dist
from shapely import affinity
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters

from navsim.common.dataloader import MetricCacheLoader

LABEL_SAFE = np.uint8(0)
LABEL_RISK = np.uint8(1)
LABEL_UNKNOWN = np.uint8(2)

EVENT_NONE = np.uint8(0)
EVENT_COLLISION = np.uint8(1)
EVENT_OFFROAD = np.uint8(2)

SOURCE_PAD = np.uint8(0)
SOURCE_DYNAMIC_AGENT_COLLISION = np.uint8(2)
SOURCE_DRIVABLE_BOUNDARY_OFFROAD = np.uint8(3)

ANCHOR_NUM_STEPS = 8
SOURCE_SUFFIX = "risk_anchors_v1.npz"
OUTPUT_SUFFIX = "risk_anchors_v2_augmented.npz"


def _distributed_info() -> Tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    return rank, world_size


def _teardown_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y"}


def _normalize_angle(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def _origin_from_metric_cache(metric_cache: Any) -> np.ndarray:
    return np.asarray(metric_cache.ego_state.rear_axle.serialize(), dtype=np.float64)


def _to_initial_ego_geom(geom: Any, origin: np.ndarray) -> Any:
    translated = affinity.affine_transform(geom, [1, 0, 0, 1, -origin[0], -origin[1]])
    c = math.cos(float(origin[2]))
    s = math.sin(float(origin[2]))
    return affinity.affine_transform(translated, [c, s, -s, c, 0, 0])


def _ego_footprint(rear_axle_pose: Sequence[float]) -> Polygon:
    params = get_pacifica_parameters()
    front = float(getattr(params, "front_length", 4.0))
    rear = float(getattr(params, "rear_length", 1.0))
    width = float(getattr(params, "width", 2.2))
    x, y, yaw = float(rear_axle_pose[0]), float(rear_axle_pose[1]), float(rear_axle_pose[2])
    corners = np.asarray(
        [
            [front, width / 2.0],
            [front, -width / 2.0],
            [-rear, -width / 2.0],
            [-rear, width / 2.0],
        ],
        dtype=np.float64,
    )
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    pts = corners @ rot.T + np.asarray([x, y], dtype=np.float64)
    return Polygon(pts)


def _rect_from_polygon(poly: Any) -> Optional[Tuple[np.ndarray, float, float, float]]:
    if poly.is_empty:
        return None
    rect = poly.minimum_rotated_rectangle
    if rect.is_empty or not hasattr(rect, "exterior"):
        return None
    coords = np.asarray(rect.exterior.coords[:-1], dtype=np.float64)
    if coords.shape[0] != 4:
        return None
    edges = np.roll(coords, -1, axis=0) - coords
    lengths = np.linalg.norm(edges, axis=1)
    long_idx = int(np.argmax(lengths))
    length = float(lengths[long_idx])
    width = float(lengths[(long_idx + 1) % 4])
    vec = edges[long_idx]
    yaw = math.atan2(float(vec[1]), float(vec[0]))
    center = coords.mean(axis=0)
    if length < width:
        length, width = width, length
        yaw = _normalize_angle(yaw + math.pi / 2.0)
    return center, length, width, _normalize_angle(yaw)


def _near_field_xy(x: float, y: float, args: argparse.Namespace) -> bool:
    return (args.x_min_m <= x <= args.x_max_m) and (abs(y) <= args.y_abs_max_m)


def _dedup_key(pose: Sequence[float], xy_res: float, yaw_res_deg: float) -> Tuple[int, int, int]:
    yaw_res = math.radians(yaw_res_deg)
    return (
        int(round(float(pose[0]) / xy_res)),
        int(round(float(pose[1]) / xy_res)),
        int(round(_normalize_angle(float(pose[2])) / yaw_res)),
    )


def _append_anchor(
    anchors: List[Tuple[float, float, float, int, int]],
    dedup: set,
    pose: Sequence[float],
    event_type: int,
    source_type: int,
    args: argparse.Namespace,
) -> None:
    x, y, yaw = float(pose[0]), float(pose[1]), _normalize_angle(float(pose[2]))
    if not _near_field_xy(x, y, args):
        return
    key = (source_type, event_type, _dedup_key((x, y, yaw), args.dedup_xy_resolution_m, args.dedup_yaw_resolution_deg))
    if key in dedup:
        return
    dedup.add(key)
    anchors.append((x, y, yaw, int(event_type), int(source_type)))


def _agent_geometries(metric_cache: Any, origin: np.ndarray, step_10hz: int, args: argparse.Namespace) -> List[Any]:
    try:
        occ = metric_cache.observation[step_10hz]
    except Exception:
        return []
    geometries = []
    for geom in getattr(occ, "_geometries", []):
        try:
            local = _to_initial_ego_geom(geom, origin)
            if local.is_empty:
                continue
            minx, miny, maxx, maxy = local.bounds
            if maxx < args.x_min_m or minx > args.x_max_m or maxy < -args.y_abs_max_m or miny > args.y_abs_max_m:
                continue
            geometries.append(local)
        except Exception:
            continue
    return geometries


def _sample_agent_risk_anchors(metric_cache: Any, origin: np.ndarray, window_idx: int, args: argparse.Namespace) -> List[Tuple[float, float, float, int, int]]:
    step = (window_idx + 1) * 5
    agent_polys = _agent_geometries(metric_cache, origin, step, args)
    params = get_pacifica_parameters()
    ego_front = float(getattr(params, "front_length", 4.0))
    ego_rear = float(getattr(params, "rear_length", 1.0))
    ego_width = float(getattr(params, "width", 2.2))
    ego_center_offset = (ego_front - ego_rear) / 2.0
    anchors: List[Tuple[float, float, float, int, int]] = []
    dedup = set()

    # Process near agents first when the per-window cap is hit.
    agent_infos = []
    for poly in agent_polys:
        rect = _rect_from_polygon(poly)
        if rect is None:
            continue
        center, length, width, yaw = rect
        dist = float(np.linalg.norm(center))
        agent_infos.append((dist, poly, center, length, width, yaw))
    agent_infos.sort(key=lambda item: item[0])

    for _, agent_poly, center, length, width, agent_yaw in agent_infos:
        h = np.asarray([math.cos(agent_yaw), math.sin(agent_yaw)], dtype=np.float64)
        l = np.asarray([-math.sin(agent_yaw), math.cos(agent_yaw)], dtype=np.float64)
        yaw_values = [_normalize_angle(agent_yaw + float(offset)) for offset in args.agent_yaw_offsets_rad]
        side_specs = [
            # name, normal, tangent, agent half extent along normal, ego half extent along normal, side length
            ("front", h, l, length / 2.0, (ego_front + ego_rear) / 2.0, width),
            ("back", -h, l, length / 2.0, (ego_front + ego_rear) / 2.0, width),
            ("left", l, h, width / 2.0, ego_width / 2.0, length),
            ("right", -l, h, width / 2.0, ego_width / 2.0, length),
        ]
        for yaw in yaw_values:
            yaw_vec = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
            for _, normal, tangent, agent_half, ego_half, side_len in side_specs:
                if args.agent_side_sample_points <= 1:
                    offsets = [0.0]
                else:
                    half = max(side_len / 2.0, 0.0)
                    offsets = np.linspace(-half, half, args.agent_side_sample_points).tolist()
                # Place ego footprint just inside the safety margin around the agent.
                center_distance = agent_half + ego_half + args.safety_margin_m * 0.5
                for offset in offsets:
                    ego_center = center + normal * center_distance + tangent * float(offset)
                    rear_axle = ego_center - yaw_vec * ego_center_offset
                    pose = (float(rear_axle[0]), float(rear_axle[1]), yaw)
                    ego_poly = _ego_footprint(pose)
                    if ego_poly.distance(agent_poly) <= args.safety_margin_m:
                        _append_anchor(anchors, dedup, pose, EVENT_COLLISION, SOURCE_DYNAMIC_AGENT_COLLISION, args)
                        if len(anchors) >= args.max_agent_risk_per_window:
                            return anchors
    return anchors[: args.max_agent_risk_per_window]


def _flatten_lines(geom: Any) -> Iterable[LineString]:
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return list(geom.geoms)
    if isinstance(geom, Polygon):
        return [LineString(geom.exterior.coords)]
    if isinstance(geom, MultiPolygon):
        lines: List[LineString] = []
        for poly in geom.geoms:
            lines.append(LineString(poly.exterior.coords))
        return lines
    if isinstance(geom, GeometryCollection):
        lines = []
        for part in geom.geoms:
            lines.extend(_flatten_lines(part))
        return lines
    return []


def _drivable_union_local(metric_cache: Any, origin: np.ndarray, args: argparse.Namespace) -> Optional[Any]:
    geometries = []
    drivable_map = metric_cache.drivable_area_map
    for geom in getattr(drivable_map, "_geometries", []):
        try:
            local = _to_initial_ego_geom(geom, origin)
            if local.is_empty:
                continue
            minx, miny, maxx, maxy = local.bounds
            pad = 15.0
            if maxx < args.x_min_m - pad or minx > args.x_max_m + pad or maxy < -args.y_abs_max_m - pad or miny > args.y_abs_max_m + pad:
                continue
            geometries.append(local)
        except Exception:
            continue
    if not geometries:
        return None
    try:
        return unary_union(geometries)
    except Exception:
        return None


def _line_tangent_yaw(line: LineString, distance: float) -> float:
    eps = min(0.5, max(line.length * 0.01, 1e-3))
    d0 = max(0.0, distance - eps)
    d1 = min(line.length, distance + eps)
    p0 = line.interpolate(d0)
    p1 = line.interpolate(d1)
    dx = float(p1.x - p0.x)
    dy = float(p1.y - p0.y)
    if abs(dx) + abs(dy) < 1e-6:
        return 0.0
    return _normalize_angle(math.atan2(dy, dx))


def _sample_boundary_risk_anchors(metric_cache: Any, origin: np.ndarray, args: argparse.Namespace) -> List[Tuple[float, float, float, int, int]]:
    drivable = _drivable_union_local(metric_cache, origin, args)
    if drivable is None or drivable.is_empty:
        return []
    boundary = drivable.boundary
    lines = list(_flatten_lines(boundary))
    anchors: List[Tuple[float, float, float, int, int]] = []
    dedup = set()

    sample_records: List[Tuple[float, LineString, float]] = []
    for line in lines:
        if line.is_empty or line.length <= 0:
            continue
        count = max(1, int(math.floor(line.length / args.edge_sample_interval_m)))
        for i in range(count + 1):
            d = min(line.length, i * args.edge_sample_interval_m)
            p = line.interpolate(d)
            x, y = float(p.x), float(p.y)
            if not _near_field_xy(x, y, args):
                continue
            sample_records.append((math.hypot(x, y), line, d))
    sample_records.sort(key=lambda item: item[0])

    for _, line, d in sample_records:
        p = line.interpolate(d)
        tangent_yaw = _line_tangent_yaw(line, d)
        tangent = np.asarray([math.cos(tangent_yaw), math.sin(tangent_yaw)], dtype=np.float64)
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        base = np.asarray([float(p.x), float(p.y)], dtype=np.float64)
        for yaw_offset in args.edge_yaw_offsets_rad:
            yaw = _normalize_angle(tangent_yaw + float(yaw_offset))
            for offset in args.edge_offsets_m:
                for sign in (-1.0, 1.0):
                    xy = base + normal * (float(offset) * sign)
                    pose = (float(xy[0]), float(xy[1]), yaw)
                    ego_poly = _ego_footprint(pose)
                    is_offroad = not drivable.contains(ego_poly)
                    near_boundary = ego_poly.distance(boundary) <= args.safety_margin_m
                    if is_offroad or near_boundary:
                        _append_anchor(anchors, dedup, pose, EVENT_OFFROAD, SOURCE_DRIVABLE_BOUNDARY_OFFROAD, args)
                        if len(anchors) >= args.max_boundary_risk_per_window:
                            return anchors
    return anchors[: args.max_boundary_risk_per_window]


def _empty_aug_arrays(num_scenes: int, max_aug_per_window: int) -> Dict[str, np.ndarray]:
    shape = (num_scenes, ANCHOR_NUM_STEPS, max_aug_per_window)
    return {
        "aug_anchors_2hz": np.zeros(shape + (3,), dtype=np.float32),
        "aug_risk_label_2hz": np.full(shape, LABEL_UNKNOWN, dtype=np.uint8),
        "aug_valid_mask_2hz": np.zeros(shape, dtype=bool),
        "aug_event_type_2hz": np.zeros(shape, dtype=np.uint8),
        "aug_source_type_2hz": np.zeros(shape, dtype=np.uint8),
    }


def _write_scene_aug(outputs: Dict[str, np.ndarray], scene_idx: int, window_idx: int, anchors: List[Tuple[float, float, float, int, int]], max_aug_per_window: int) -> int:
    count = min(len(anchors), max_aug_per_window)
    if count <= 0:
        return 0
    arr = np.asarray([[a[0], a[1], a[2]] for a in anchors[:count]], dtype=np.float32)
    events = np.asarray([a[3] for a in anchors[:count]], dtype=np.uint8)
    sources = np.asarray([a[4] for a in anchors[:count]], dtype=np.uint8)
    outputs["aug_anchors_2hz"][scene_idx, window_idx, :count] = arr
    outputs["aug_risk_label_2hz"][scene_idx, window_idx, :count] = LABEL_RISK
    outputs["aug_valid_mask_2hz"][scene_idx, window_idx, :count] = True
    outputs["aug_event_type_2hz"][scene_idx, window_idx, :count] = events
    outputs["aug_source_type_2hz"][scene_idx, window_idx, :count] = sources
    return count


def _process_file(source_path: Path, output_path: Path, metric_loader: MetricCacheLoader, args: argparse.Namespace) -> Dict[str, Any]:
    with np.load(source_path, allow_pickle=False) as z:
        source = {key: z[key] for key in z.files}
    original_num_scenes = int(source["tokens"].shape[0])
    if args.limit_scenes_per_file > 0:
        scene_limit = min(args.limit_scenes_per_file, original_num_scenes)
        for key, value in list(source.items()):
            if hasattr(value, "shape") and value.shape and value.shape[0] == original_num_scenes:
                source[key] = value[:scene_limit]
    tokens = source["tokens"]
    num_scenes = int(tokens.shape[0])
    max_aug = args.max_agent_risk_per_window + args.max_boundary_risk_per_window
    outputs = _empty_aug_arrays(num_scenes, max_aug)

    missing_metric_cache = 0
    failed_scenes = 0
    agent_anchor_count = 0
    boundary_anchor_count = 0
    scenes_with_aug = 0

    for scene_idx, token_value in enumerate(tokens):
        token = str(token_value)
        if token not in metric_loader.metric_cache_paths:
            missing_metric_cache += 1
            continue
        try:
            metric_cache = metric_loader.get_from_token(token)
            origin = _origin_from_metric_cache(metric_cache)
            boundary_anchors = _sample_boundary_risk_anchors(metric_cache, origin, args)
            scene_count = 0
            for window_idx in range(ANCHOR_NUM_STEPS):
                window_anchors: List[Tuple[float, float, float, int, int]] = []
                agent_anchors = _sample_agent_risk_anchors(metric_cache, origin, window_idx, args)
                window_anchors.extend(agent_anchors[: args.max_agent_risk_per_window])
                window_anchors.extend(boundary_anchors[: args.max_boundary_risk_per_window])
                written = _write_scene_aug(outputs, scene_idx, window_idx, window_anchors, max_aug)
                scene_count += written
                agent_anchor_count += min(len(agent_anchors), args.max_agent_risk_per_window)
                boundary_anchor_count += min(len(boundary_anchors), args.max_boundary_risk_per_window)
            if scene_count > 0:
                scenes_with_aug += 1
        except Exception:
            failed_scenes += 1
            continue

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict: Dict[str, Any] = {
        "source_v1_file": np.asarray(source_path.name),
        "tokens": tokens,
        **outputs,
    }
    if args.copy_v1_fields:
        for key, value in source.items():
            if key not in save_dict:
                save_dict[key] = value
    if args.compress:
        np.savez_compressed(output_path, **save_dict)
    else:
        tmp_path = output_path.with_suffix(".tmp.npz")
        np.savez(tmp_path, **save_dict)
        tmp_path.replace(output_path)

    return {
        "source_file": source_path.name,
        "output_file": output_path.name,
        "num_scenes": num_scenes,
        "max_aug_per_window": max_aug,
        "missing_metric_cache_scenes": missing_metric_cache,
        "failed_scenes": failed_scenes,
        "scenes_with_aug": scenes_with_aug,
        "agent_aug_anchors": int(agent_anchor_count),
        "boundary_aug_anchors": int(boundary_anchor_count),
        "total_aug_anchors": int(agent_anchor_count + boundary_anchor_count),
    }


def _source_to_output_name(source_name: str) -> str:
    if source_name.endswith(SOURCE_SUFFIX):
        return source_name[: -len(SOURCE_SUFFIX)] + OUTPUT_SUFFIX
    return source_name.replace(".npz", f"_{OUTPUT_SUFFIX}")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _build_manifest(args: argparse.Namespace, rank: int, world_size: int, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "version": "v2_augmented",
        "rank": rank,
        "world_size": world_size,
        "source_v1_dir": str(Path(args.source_v1_dir).expanduser()),
        "output_dir": str(Path(args.output_dir).expanduser()),
        "metric_cache_path": str(Path(args.metric_cache_path).expanduser()),
        "copy_v1_fields": bool(args.copy_v1_fields),
        "compress": bool(args.compress),
        "label_mapping": {"0": "safe", "1": "risk", "2": "unknown"},
        "event_type_mapping": {"0": "none", "1": "collision", "2": "offroad"},
        "source_type_mapping": {
            "0": "pad_none",
            "2": "dynamic_agent_collision",
            "3": "drivable_boundary_offroad",
        },
        "safety_margin_m": args.safety_margin_m,
        "agent_risk_rule": "ego footprint distance to agent footprint <= safety_margin_m",
        "boundary_risk_rule": "ego footprint outside drivable area or distance to drivable boundary <= safety_margin_m",
        "agent_yaw_reference": "agent minimum-rotated-rectangle heading plus configured offsets",
        "boundary_yaw_reference": "drivable boundary tangent plus configured offsets",
        "params": {
            "max_agent_risk_per_window": args.max_agent_risk_per_window,
            "max_boundary_risk_per_window": args.max_boundary_risk_per_window,
            "agent_side_sample_points": args.agent_side_sample_points,
            "agent_yaw_offsets_rad": list(args.agent_yaw_offsets_rad),
            "edge_sample_interval_m": args.edge_sample_interval_m,
            "edge_offsets_m": list(args.edge_offsets_m),
            "edge_yaw_offsets_rad": list(args.edge_yaw_offsets_rad),
            "x_min_m": args.x_min_m,
            "x_max_m": args.x_max_m,
            "y_abs_max_m": args.y_abs_max_m,
            "dedup_xy_resolution_m": args.dedup_xy_resolution_m,
            "dedup_yaw_resolution_deg": args.dedup_yaw_resolution_deg,
        },
        "chunks": chunks,
        "summary": {
            "num_chunks": len(chunks),
            "num_scenes": int(sum(item["num_scenes"] for item in chunks)),
            "missing_metric_cache_scenes": int(sum(item["missing_metric_cache_scenes"] for item in chunks)),
            "failed_scenes": int(sum(item["failed_scenes"] for item in chunks)),
            "scenes_with_aug": int(sum(item["scenes_with_aug"] for item in chunks)),
            "agent_aug_anchors": int(sum(item["agent_aug_anchors"] for item in chunks)),
            "boundary_aug_anchors": int(sum(item["boundary_aug_anchors"] for item in chunks)),
            "total_aug_anchors": int(sum(item["total_aug_anchors"] for item in chunks)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate V2 augmented sparse risk anchors from V1 anchors and NAVSIM metric cache.")
    parser.add_argument("--source-v1-dir", default="/mnt/workspace/DrivoR/navsim/exp/drivor_navtrain_risk_anchors_v1")
    parser.add_argument("--output-dir", default="navsim/exp/drivor_navtrain_risk_anchors_v2_augmented")
    parser.add_argument("--metric-cache-path", default="/mnt/workspace/ReCogDrive/ReCogDrive/recogdrive/navsim/exp/metric_cache_train")
    parser.add_argument("--overwrite", type=_as_bool, default=False)
    parser.add_argument("--copy-v1-fields", type=_as_bool, default=True)
    parser.add_argument("--compress", type=_as_bool, default=False)
    parser.add_argument("--limit-files", type=int, default=0)
    parser.add_argument("--limit-scenes-per-file", type=int, default=0)

    parser.add_argument("--safety-margin-m", type=float, default=0.5)
    parser.add_argument("--max-agent-risk-per-window", type=int, default=64)
    parser.add_argument("--max-boundary-risk-per-window", type=int, default=64)
    parser.add_argument("--agent-side-sample-points", type=int, default=5)
    parser.add_argument("--agent-yaw-offsets-rad", type=float, nargs="*", default=[0.0])
    parser.add_argument("--edge-sample-interval-m", type=float, default=2.0)
    parser.add_argument("--edge-offsets-m", type=float, nargs="*", default=[-1.0, -0.5, 0.0, 0.5, 1.0])
    parser.add_argument("--edge-yaw-offsets-rad", type=float, nargs="*", default=[0.0])
    parser.add_argument("--x-min-m", type=float, default=-10.0)
    parser.add_argument("--x-max-m", type=float, default=80.0)
    parser.add_argument("--y-abs-max-m", type=float, default=40.0)
    parser.add_argument("--dedup-xy-resolution-m", type=float, default=0.5)
    parser.add_argument("--dedup-yaw-resolution-deg", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank, world_size = _distributed_info()
    source_dir = Path(args.source_v1_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source_v1_dir not found: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(source_dir.glob(f"*{SOURCE_SUFFIX}"))
    if args.limit_files > 0:
        source_files = source_files[: args.limit_files]
    shard_files = [path for idx, path in enumerate(source_files) if idx % world_size == rank]
    metric_loader = MetricCacheLoader(Path(args.metric_cache_path).expanduser())

    chunks: List[Dict[str, Any]] = []
    for source_path in shard_files:
        output_path = output_dir / _source_to_output_name(source_path.name)
        if output_path.exists() and not args.overwrite:
            chunks.append({
                "source_file": source_path.name,
                "output_file": output_path.name,
                "num_scenes": 0,
                "max_aug_per_window": args.max_agent_risk_per_window + args.max_boundary_risk_per_window,
                "missing_metric_cache_scenes": 0,
                "failed_scenes": 0,
                "scenes_with_aug": 0,
                "agent_aug_anchors": 0,
                "boundary_aug_anchors": 0,
                "total_aug_anchors": 0,
                "skipped_existing": True,
            })
            continue
        print(f"[rank {rank}/{world_size}] processing {source_path.name} -> {output_path.name}", flush=True)
        chunks.append(_process_file(source_path, output_path, metric_loader, args))

    rank_manifest = _build_manifest(args, rank, world_size, chunks)
    _write_json(output_dir / f"manifest_rank{rank:03d}.json", rank_manifest)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if rank == 0:
        manifests = []
        for manifest_path in sorted(output_dir.glob("manifest_rank*.json")):
            manifests.append(json.loads(manifest_path.read_text()))
        all_chunks = []
        for manifest in manifests:
            all_chunks.extend(manifest.get("chunks", []))
        merged_manifest = _build_manifest(args, rank=0, world_size=world_size, chunks=all_chunks)
        merged_manifest["rank_manifests"] = [f"manifest_rank{idx:03d}.json" for idx in range(world_size)]
        _write_json(output_dir / "manifest.json", merged_manifest)
        print(json.dumps(merged_manifest["summary"], indent=2, sort_keys=True), flush=True)
    _teardown_distributed()


if __name__ == "__main__":
    main()
