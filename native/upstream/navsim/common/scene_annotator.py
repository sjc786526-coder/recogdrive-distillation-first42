"""Structured information extractor for navsim scenes to assist VLM annotation."""

from typing import Any, Dict, List, Optional, Tuple
import math
import logging

import cv2
import numpy as np
import numpy.typing as npt

from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.common.actor_state.state_representation import StateSE2
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from navsim.common.dataclasses import Scene, Annotations, Frame, Camera
from navsim.common.enums import BoundingBoxIndex
from navsim.visualization.camera import (
    _transform_annotations_to_camera,
    _transform_points_to_image,
    _rotation_3d_in_axis,
)

logger = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────────────

def _normalize_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _speed_2d(vec: npt.NDArray) -> float:
    return float(math.sqrt(float(vec[0]) ** 2 + float(vec[1]) ** 2))


def _xy_in_range(x: float, y: float, x_range: Tuple[float, float], y_range: Tuple[float, float]) -> bool:
    return (x_range[0] <= x <= x_range[1]) and (y_range[0] <= y <= y_range[1])


def _compute_curvature_along_linestring(ls: LineString, n_pts: int = 20) -> Dict[str, float]:
    """Compute curvature statistics along a shapely LineString (rad/m)."""
    coords = np.array(ls.coords)
    if len(coords) < 3:
        return {"max": 0.0, "mean": 0.0, "min": 0.0, "length_m": ls.length}
    # resample uniformly
    distances = np.linspace(0, ls.length, min(n_pts, len(coords)))
    pts = np.array([ls.interpolate(d).coords[0] for d in distances])
    dx_dt = np.gradient(pts[:, 0])
    dy_dt = np.gradient(pts[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    denom = (dx_dt ** 2 + dy_dt ** 2) ** 1.5
    denom[denom < 1e-6] = 1e-6
    curvature = (dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / denom
    return {
        "max": float(np.max(np.abs(curvature))),
        "mean": float(np.mean(np.abs(curvature))),
        "min": float(np.min(np.abs(curvature))),
        "length_m": float(ls.length),
    }


def _classify_curvature(
    max_curvature: float,
    length_m: float = 0.0,
    total_heading_rad: float = 0.0,
) -> str:
    """Map maximum curvature to a human-readable description.

    A sanity check using *total_heading_rad* prevents short lanes from being
    misclassified: when the end-to-end heading change is tiny the lane is
    forced to "直道" regardless of the per-meter curvature (which is
    unreliable for very short baselines due to numerical noise).
    """
    if max_curvature < 0.005:
        curve_type = "直道"
    elif max_curvature < 0.02:
        curve_type = "长缓弯"
    elif max_curvature < 0.05:
        curve_type = "中等弯道"
    else:
        curve_type = "急弯"

    # Sanity check: tiny end-to-end heading → really straight
    if curve_type != "直道" and abs(total_heading_rad) < math.radians(2):
        curve_type = "直道"

    return curve_type


def _end_to_end_heading_change(ls: LineString) -> float:
    """Total heading change from the first to the last segment of *ls* (rad)."""
    coords = np.array(ls.coords)
    if len(coords) < 2:
        return 0.0
    # heading of the first segment
    d0 = coords[min(1, len(coords) - 1)] - coords[0]
    h0 = math.atan2(float(d0[1]), float(d0[0]))
    # heading of the last segment
    d1 = coords[-1] - coords[-2]
    h1 = math.atan2(float(d1[1]), float(d1[0]))
    return float(_normalize_angle(h1 - h0))


def _driving_command_str(driving_command: npt.NDArray) -> str:
    cmd_idx = int(np.argmax(driving_command))
    return {0: "左转", 1: "直行", 2: "右转", 3: "停止"}.get(cmd_idx, "未知")


def _classify_position(x: float, y: float) -> str:
    """Classify a target's position relative to ego."""
    angle = math.atan2(y, x)
    if abs(angle) < math.radians(10):
        return "正前方"
    elif abs(angle) < math.radians(45):
        return "右前方" if angle < 0 else "左前方"
    elif abs(angle) < math.radians(135):
        return "右侧" if angle < 0 else "左侧"
    else:
        return "后方"


def _classify_lateral_side(y: float) -> str:
    if y > 1.0:
        return "左侧"
    if y < -1.0:
        return "右侧"
    return "本车道附近"


def _format_optional(value: Optional[float], fmt: str = ".1f") -> str:
    return "不可用" if value is None else format(value, fmt)


def _clean_float(value: float, eps: float = 5e-4) -> float:
    return 0.0 if abs(value) < eps else float(value)


def _camera_image_size(camera: Camera) -> Tuple[int, int]:
    """Return image size as (width, height), falling back to intrinsics if pixels are unavailable."""
    image = camera.image
    if isinstance(image, np.ndarray) and image.ndim >= 2:
        h, w = image.shape[:2]
        return int(w), int(h)
    if camera.intrinsics is not None:
        cx = float(camera.intrinsics[0, 2])
        cy = float(camera.intrinsics[1, 2])
        if cx > 0 and cy > 0:
            return int(round(cx * 2)), int(round(cy * 2))
    return 2000, 1100


def _project_boxes_to_2d(
    annotations: Annotations, camera: Camera
) -> List[Optional[Tuple[float, float, float, float]]]:
    """Project 3D boxes to 2D image-space bounding boxes [xmin, ymin, xmax, ymax]."""
    image_w, image_h = _camera_image_size(camera)
    boxes_cam = _transform_annotations_to_camera(
        annotations.boxes,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
    )
    box_positions = boxes_cam[:, BoundingBoxIndex.POSITION]
    box_dimensions = boxes_cam[:, BoundingBoxIndex.DIMENSION]
    box_heading = boxes_cam[:, BoundingBoxIndex.HEADING]

    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = box_dimensions.reshape([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])
    corners = _rotation_3d_in_axis(corners, box_heading, axis=1)
    corners += box_positions.reshape(-1, 1, 3)

    box_corners, corners_pc_in_fov = _transform_points_to_image(
        corners.reshape(-1, 3), camera.intrinsics
    )
    box_corners = box_corners.reshape(-1, 8, 2)
    corners_pc_in_fov = corners_pc_in_fov.reshape(-1, 8)
    valid = corners_pc_in_fov.any(-1)

    results: List[Optional[Tuple[float, float, float, float]]] = []
    for i in range(len(annotations.boxes)):
        if valid[i]:
            visible_corner_count = int(np.sum(corners_pc_in_fov[i]))
            cs = box_corners[i]
            xmin, ymin = float(np.min(cs[:, 0])), float(np.min(cs[:, 1]))
            xmax, ymax = float(np.max(cs[:, 0])), float(np.max(cs[:, 1]))
            if xmax < 0 or ymax < 0 or xmin > image_w or ymin > image_h:
                results.append(None)  # box fully outside image
            else:
                clipped_xmin = max(0.0, xmin)
                clipped_ymin = max(0.0, ymin)
                clipped_xmax = min(float(image_w), xmax)
                clipped_ymax = min(float(image_h), ymax)
                clipped_area = max(0.0, clipped_xmax - clipped_xmin) * max(0.0, clipped_ymax - clipped_ymin)
                image_area = max(float(image_w * image_h), 1.0)

                # Near side/rear boxes can straddle the camera plane. Projecting
                # corners behind the camera often expands to a full-image box
                # after clipping, which is not a useful visible 2D grounding.
                if visible_corner_count < 8 and clipped_area / image_area > 0.85:
                    results.append(None)
                else:
                    results.append((clipped_xmin, clipped_ymin, clipped_xmax, clipped_ymax))
        else:
            results.append(None)
    return results


class SceneAnnotator:
    """Extracts structured information from a navsim Scene to serve as VLM context."""

    def __init__(self, scene: Scene):
        self.scene = scene
        meta = scene.scene_metadata
        self._cur_idx = meta.num_history_frames - 1
        self._num_hist = meta.num_history_frames
        self._num_fut = meta.num_future_frames
        self._cur_frame = scene.frames[self._cur_idx]
        self._ego = self._cur_frame.ego_status
        self._anns = self._cur_frame.annotations
        self._map_api = scene.map_api
        self._ego_pose_se2 = StateSE2(*self._ego.ego_pose)

    @property
    def _dt(self) -> float:
        return 0.5

    def _to_ego_local(self, gx: float, gy: float) -> Tuple[float, float]:
        dx = gx - self._ego_pose_se2.x
        dy = gy - self._ego_pose_se2.y
        cos_h = math.cos(-self._ego_pose_se2.heading)
        sin_h = math.sin(-self._ego_pose_se2.heading)
        lx = dx * cos_h - dy * sin_h
        ly = dx * sin_h + dy * cos_h
        return lx, ly

    def _to_global(self, lx: float, ly: float) -> Tuple[float, float]:
        cos_h = math.cos(self._ego_pose_se2.heading)
        sin_h = math.sin(self._ego_pose_se2.heading)
        gx = self._ego_pose_se2.x + lx * cos_h - ly * sin_h
        gy = self._ego_pose_se2.y + lx * sin_h + ly * cos_h
        return gx, gy

    def _linestring_local_coords(self, ls: LineString) -> npt.NDArray[np.float64]:
        return np.array([self._to_ego_local(float(x), float(y)) for x, y in ls.coords], dtype=np.float64)

    @staticmethod
    def _measure_lateral_from_current_lane(
        current_lane_ls: LineString,
        candidate_local: npt.NDArray[np.float64],
        n_samples: int = 8,
    ) -> Tuple[float, float]:
        """Measure lateral offsets at regular intervals along *current_lane_ls*.

        At each sample point a perpendicular ray is cast; its intersection
        with the candidate lane gives the signed lateral gap (+ = left,
        − = right).

        Returns ``(mean_offset_m, std_offset_m)``.  A low *std* indicates the
        candidate lane is strictly parallel to the current lane throughout.
        """
        candidate_ls = LineString(candidate_local)
        s_vals = np.linspace(0.0, current_lane_ls.length, n_samples)
        gaps: List[float] = []
        for s in s_vals:
            cp = current_lane_ls.interpolate(float(s))
            # perpendicular direction at cp
            eps = 0.5
            nxt = current_lane_ls.interpolate(min(float(s) + eps, current_lane_ls.length))
            prv = current_lane_ls.interpolate(max(float(s) - eps, 0.0))
            dx = float(nxt.x - prv.x)
            dy = float(nxt.y - prv.y)
            norm = math.sqrt(dx * dx + dy * dy) or 1e-6
            pdx = -dy / norm  # rotate CCW → left
            pdy = dx / norm

            # long perpendicular segment
            L = 30.0
            perp = LineString([
                (float(cp.x) + pdx * L, float(cp.y) + pdy * L),
                (float(cp.x) - pdx * L, float(cp.y) - pdy * L),
            ])

            inter = perp.intersection(candidate_ls)
            if inter.is_empty:
                continue
            if inter.geom_type == "Point":
                ip = inter
            elif inter.geom_type == "MultiPoint":
                pts = list(inter.geoms)
                ip = min(pts, key=lambda p: p.distance(cp))
            else:
                ip = inter.centroid

            # signed gap: project (ip − cp) onto perpendicular direction
            gap = (float(ip.x) - float(cp.x)) * pdx + (float(ip.y) - float(cp.y)) * pdy
            gaps.append(gap)

        if len(gaps) < 2:
            return (0.0, float("inf"))

        return (float(np.mean(gaps)), float(np.std(gaps)))

    # ── public entry ─────────────────────────────────────────────────────────

    def annotate(self) -> Dict[str, Any]:
        return {
            "scene_meta": self._scene_meta(),
            "environment_understanding": self._annotate_environment(),
            "agent_grounding": self._annotate_agent_grounding(),
            "planning": self._annotate_planning(),
            "acting": self._annotate_acting(),
        }

    def to_context_text(self, data: Optional[Dict[str, Any]] = None) -> str:
        """Render all structured annotations as human-readable text for VLM prompting."""
        if data is None:
            data = self.annotate()
        parts: List[str] = []

        # ── scene meta ──
        sm = data["scene_meta"]
        parts.append(f"场景信息: 地图={sm['map_name']}, 日志={sm['log_name']}")

        # ── 1. 环境理解 ──
        env = data["environment_understanding"]

        lane = env["lane_structure"]
        parts.append("\n[车道结构]")
        cl = lane["current_lane"]
        parts.append(f"  当前车道: 类型={cl['type']}, 曲率={cl['curvature_type']} (max={cl['max_curvature']:.4f} rad/m)")
        if cl["trajectory_points"]:
            traj_str = " | ".join(
                f"({p['x']:.2f}, {p['y']:.2f}, {p['heading']:.3f})" for p in cl["trajectory_points"]
            )
            parts.append(f"  车道轨迹点(自车坐标系,等间距采样): {traj_str}")
        for side, cn in (("left", "左侧"), ("right", "右侧")):
            adj = lane["adjacent_lanes"][side]
            parts.append(f"  {cn}同向车道: {adj['summary']}")
            for i, traj in enumerate(adj.get("trajectories", [])):
                if traj:
                    traj_str = " | ".join(
                        f"({p['x']:.2f}, {p['y']:.2f}, {p['heading']:.3f})" for p in traj
                    )
                    parts.append(f"    {cn}同向车道{i+1}轨迹点(自车坐标系,等间距采样): {traj_str}")
        for side, cn in (("left", "左侧"), ("right", "右侧")):
            onc = lane["oncoming_lanes"][side]
            parts.append(f"  {cn}对向车道: {onc['summary']}")
            for i, traj in enumerate(onc.get("trajectories", [])):
                if traj:
                    traj_str = " | ".join(
                        f"({p['x']:.2f}, {p['y']:.2f}, {p['heading']:.3f})" for p in traj
                    )
                    parts.append(f"    {cn}对向车道{i+1}轨迹点(自车坐标系,等间距采样): {traj_str}")
        parts.append(f"  语义: {lane['semantic_description']}")

        flow = env["traffic_flow"]
        parts.append("\n[车流状态]")
        if flow["lead_vehicle"] is not None:
            lv = flow["lead_vehicle"]
            parts.append(f"  前车: 距离={lv['distance_m']:.1f}m, 速度={lv['speed_mps']:.1f}m/s, "
                         f"车头时距={_format_optional(lv['headway_s'])}s")
        else:
            parts.append("  前车: 无 (视野开阔)")
        parts.append(f"  前方车辆数: {flow['forward_vehicle_count']} (50m范围内), 周围车辆数: {flow['nearby_vehicle_count']}")
        parts.append(f"  车流状态: {flow['flow_state']}")

        # ── 2. 目标定位 ──
        grd = data["agent_grounding"]
        parts.append("\n[周围交通参与者]")
        parts.append(f"  感知目标总数: {grd['agent_count']}, 重要目标数: {len(grd['important_agents'])}")
        for i, ag in enumerate(grd["important_agents"]):
            parts.append(f"  [{i}] {ag['class']} | 位置={ag['position']} | "
                         f"距离={ag['distance_m']:.1f}m (纵向={ag['longitudinal_m']:.1f}m, 横向={ag['lateral_m']:.1f}m) | "
                         f"速度={ag['speed_mps']:.1f}m/s | 尺寸={ag['length_m']:.1f}x{ag['width_m']:.1f}m")
            if ag["bbox_2d"] is not None:
                b2 = ag["bbox_2d"]
                parts.append(f"      2D框: ({b2['xmin']:.0f}, {b2['ymin']:.0f}, {b2['xmax']:.0f}, {b2['ymax']:.0f}), 图像区域={ag['bbox_image_region']}")
            if ag["acceleration_mps2"] is not None:
                parts.append(f"      加速度: {ag['acceleration_mps2']:.2f} m/s²")
            parts.append(f"      状态: {ag['state_description']}；距离语义: {ag['distance_description']}")

        # ── 3. 行为决策 ──
        plan = data["planning"]
        parts.append("\n[行为决策]")
        parts.append(f"  导航指令: {plan['nav_command']}")
        if plan.get("lane_change") is not None:
            lc = plan["lane_change"]
            parts.append(f"  变道检测: {lc['description']}")

        # ── 4. 运动控制 ──
        act = data["acting"]
        parts.append("\n[历史运动]")
        parts.append(f"  描述: {act['history_motion_description']}")
        parts.append(f"  历史轨迹: {act['history_trajectory_str']}")

        parts.append("\n[当前/未来运动]")
        parts.append(f"  当前速度: {act['current_speed_mps']:.1f} m/s")
        parts.append(f"  当前加速度: ({act['current_accel_x']:.2f}, {act['current_accel_y']:.2f}) m/s²")
        parts.append(f"  轨迹描述: {act['future_trajectory_description']}")
        parts.append(f"  未来轨迹控制量: 横向偏移={act['future_total_lateral_m']:.2f}m, 航向变化={act['future_heading_change_rad']:.3f}rad")
        parts.append(f"  未来轨迹: {act['future_trajectory_str']}")

        return "\n".join(parts)

    def draw_bboxes_on_image(
        self, output_path: Optional[str] = None,
    ) -> np.ndarray:
        """Draw 2D bounding boxes of important agents on the front camera image.

        Colors:
          - Red   → pedestrian
          - Cyan  → bicycle
          - Green → vehicle (important / same-lane)
          - Gray  → vehicle (other)

        :param output_path: if provided, save the annotated image to this path (PNG).
        :return: annotated image as an RGB numpy array (suitable for plt.imshow).
        """
        cam_f0 = self._cur_frame.cameras.cam_f0
        if cam_f0.intrinsics is None or cam_f0.image is None:
            raise ValueError("Front camera image or intrinsics not available")
        if not isinstance(cam_f0.image, np.ndarray):
            raise ValueError("Front camera image pixels are not loaded")

        image = cam_f0.image.copy()
        h, w = image.shape[:2]
        boxes_2d = _project_boxes_to_2d(self._anns, cam_f0)

        for i, (box, name, b2d) in enumerate(
            zip(self._anns.boxes, self._anns.names, boxes_2d)
        ):
            if b2d is None:
                continue

            xmin, ymin, xmax, ymax = b2d
            xmin = max(0.0, min(float(xmin), float(w - 1)))
            ymin = max(0.0, min(float(ymin), float(h - 1)))
            xmax = max(0.0, min(float(xmax), float(w - 1)))
            ymax = max(0.0, min(float(ymax), float(h - 1)))

            # Choose color by class
            if name == "pedestrian":
                color = (255, 0, 0)    # red (RGB)
                thickness = 3
            elif name == "bicycle":
                color = (255, 255, 0)  # cyan (RGB)
                thickness = 2
            elif name == "vehicle":
                # Highlight important vehicles (in front, close) in green,
                # others in gray.
                x = float(box[0])
                y = float(box[1])
                dist = math.sqrt(x * x + y * y)
                if x > -5 and abs(y) < 6.0 and dist < 50:
                    color = (0, 255, 0)   # green (RGB)
                    thickness = 2
                else:
                    color = (160, 160, 160)  # gray (RGB)
                    thickness = 1
            else:
                color = (160, 160, 160)
                thickness = 1

            pt1 = (int(xmin), int(ymin))
            pt2 = (int(xmax), int(ymax))
            cv2.rectangle(image, pt1, pt2, color, thickness)

            # Label
            label = f"{name}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(image, (pt1[0], pt1[1] - th - 4),
                          (pt1[0] + tw + 2, pt1[1]), color, -1)
            cv2.putText(image, label, (pt1[0] + 1, pt1[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        if output_path is not None:
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        return image

    def get_oncoming_lane_ids(self) -> set:
        """Return the set of oncoming lane IDs selected by the annotator.

        These are the strictly-parallel opposite-direction lanes that passed
        all filters.  The returned IDs can be passed directly to
        ``add_map_to_bev_ax(oncoming_lane_ids=...)`` or
        ``add_configured_bev_on_ax(oncoming_lane_ids=...)``.
        """
        lane = self._extract_lane_structure()
        ids: set = set()
        for side in ("left", "right"):
            ids.update(lane["oncoming_lanes"][side].get("ids", []))
        return ids

    # ── scene meta ───────────────────────────────────────────────────────────

    def _scene_meta(self) -> Dict[str, Any]:
        sm = self.scene.scene_metadata
        return {
            "log_name": sm.log_name,
            "scene_token": sm.scene_token,
            "map_name": sm.map_name,
            "initial_token": sm.initial_token,
            "num_history_frames": sm.num_history_frames,
            "num_future_frames": sm.num_future_frames,
        }

    # ── 1. environment understanding ─────────────────────────────────────────

    def _annotate_environment(self) -> Dict[str, Any]:
        return {
            "lane_structure": self._extract_lane_structure(),
            "traffic_flow": self._extract_traffic_flow(),
        }

    def _extract_lane_structure(self) -> Dict[str, Any]:
        """Extract ego's current lane info, same-direction adjacent lanes, and oncoming lanes."""
        result: Dict[str, Any] = {
            "current_lane": {
                "ids": [],
                "type": "未知",
                "curvature_type": "未知",
                "max_curvature": 0.0,
                "heading_delta_rad": 0.0,
                "trajectory_points": [],
            },
            "adjacent_lanes": {
                "left": {"exists": False, "direction_compatible": False, "ids": [], "trajectories": [], "curvature_type": "未知", "count": 0, "summary": "左侧无同向平行车道"},
                "right": {"exists": False, "direction_compatible": False, "ids": [], "trajectories": [], "curvature_type": "未知", "count": 0, "summary": "右侧无同向平行车道"},
            },
            "oncoming_lanes": {
                "left": {"exists": False, "ids": [], "trajectories": [], "count": 0, "summary": "左侧无对向车道"},
                "right": {"exists": False, "ids": [], "trajectories": [], "count": 0, "summary": "右侧无对向车道"},
            },
            "semantic_description": "车道结构不可用",
        }

        ego_point = Point(self._ego_pose_se2.x, self._ego_pose_se2.y)
        lookahead = 80.0

        try:
            map_objects = self._map_api.get_proximal_map_objects(
                point=ego_point,
                radius=lookahead,
                layers=[SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR],
            )
        except Exception:
            logger.warning("map_api query failed for lane structure", exc_info=True)
            return result

        all_lanes = list(map_objects.get(SemanticMapLayer.LANE, []))
        all_connectors = list(map_objects.get(SemanticMapLayer.LANE_CONNECTOR, []))

        forward_lanes = []
        forward_connectors = []
        for obj in all_lanes:
            ls = obj.baseline_path.linestring
            local = self._linestring_local_coords(ls)
            mask = (local[:, 0] > -5) & (local[:, 0] < lookahead) & (np.abs(local[:, 1]) < 18)
            if bool(mask.any()):
                forward_lanes.append(obj)
        for obj in all_connectors:
            ls = obj.baseline_path.linestring
            local = self._linestring_local_coords(ls)
            mask = (local[:, 0] > -5) & (local[:, 0] < lookahead) & (np.abs(local[:, 1]) < 18)
            if bool(mask.any()):
                forward_connectors.append(obj)

        all_forward = forward_lanes + forward_connectors
        local_drivable_area = self._build_local_drivable_area(ego_point, lookahead)

        # ── 1. current lane ──
        current_lane_candidates = self._find_current_lane_candidates(all_lanes, all_connectors)
        current_ids_set: set = set()
        nearest_ls: Optional[LineString] = None
        if current_lane_candidates:
            result["current_lane"]["ids"] = [str(getattr(obj, "id", "")) for obj in current_lane_candidates]
            current_ids_set = set(result["current_lane"]["ids"])
            result["current_lane"]["type"] = "lane_connector" if current_lane_candidates[0] in all_connectors else "lane"
            nearest_ls = current_lane_candidates[0].baseline_path.linestring
            if nearest_ls is not None:
                nc = _compute_curvature_along_linestring(nearest_ls)
                result["current_lane"]["max_curvature"] = nc["max"]
                result["current_lane"]["curvature_type"] = _classify_curvature(
                    nc["max"], nc["length_m"], _end_to_end_heading_change(nearest_ls),
                )
                local = self._linestring_local_coords(nearest_ls)
                result["current_lane"]["heading_delta_rad"] = round(float(self._local_linestring_heading_delta(local)), 4)
        else:
            nearest_ls = self._find_nearest_linestring(all_forward)
            if nearest_ls is not None:
                nc = _compute_curvature_along_linestring(nearest_ls)
                result["current_lane"]["max_curvature"] = nc["max"]
                result["current_lane"]["curvature_type"] = _classify_curvature(
                    nc["max"], nc["length_m"], _end_to_end_heading_change(nearest_ls),
                )

        # sample 8 trajectory points at equal intervals along current lane baseline
        if nearest_ls is not None:
            current_traj_points = self._sample_lane_trajectory(nearest_ls)
            if current_lane_candidates and not self._has_forward_lane_samples(current_traj_points):
                future_overlap_points = self._sample_future_overlap_trajectory(current_lane_candidates[0])
                if future_overlap_points:
                    current_traj_points = future_overlap_points
            result["current_lane"]["trajectory_points"] = current_traj_points

        # Build current lane baseline for adjacent lane search
        current_lane_ls: Optional[LineString] = None
        if nearest_ls is not None:
            current_lane_local = self._linestring_local_coords(nearest_ls)
            current_lane_ls = LineString(current_lane_local)

        # ── 2. same-direction parallel adjacent lanes ──
        # Search for lanes adjacent to the current lane baseline (lane-centric)
        left_lanes: List[Any] = []
        right_lanes: List[Any] = []
        if current_lane_ls is not None:
            left_lanes, right_lanes = self._search_lateral_lanes(
                current_lane_ls, all_forward, forward_lanes, current_ids_set,
                local_drivable_area=local_drivable_area, oncoming=False,
            )
            # Remove candidates that are longitudinal extensions of the current lane
            left_lanes = self._filter_current_lane_extensions(
                left_lanes, current_lane_ls, current_lane_candidates,
            )
            right_lanes = self._filter_current_lane_extensions(
                right_lanes, current_lane_ls, current_lane_candidates,
            )

        # Pre-compute the perpendicular line from ego's projection on the
        # current lane — used to find sampling start points on adjacent lanes.
        perp_line: Optional[LineString] = None
        ego_proj_pt: Optional[Point] = None
        if current_lane_ls is not None:
            ego_proj_s = current_lane_ls.project(Point(0, 0))
            ego_proj_pt = current_lane_ls.interpolate(ego_proj_s)
            # Lane direction at projection point
            eps = 0.5
            nxt = current_lane_ls.interpolate(min(ego_proj_s + eps, current_lane_ls.length))
            prv = current_lane_ls.interpolate(max(ego_proj_s - eps, 0.0))
            dx = float(nxt.x - prv.x)
            dy = float(nxt.y - prv.y)
            perp_norm = math.sqrt(dx * dx + dy * dy) or 1e-6
            # perpendicular direction (rotate lane direction 90° CCW → left side)
            perp_dx = -dy / perp_norm
            perp_dy = dx / perp_norm
            extend_len = 30.0
            perp_line = LineString([
                (float(ego_proj_pt.x) - perp_dx * extend_len,
                 float(ego_proj_pt.y) - perp_dy * extend_len),
                (float(ego_proj_pt.x) + perp_dx * extend_len,
                 float(ego_proj_pt.y) + perp_dy * extend_len),
            ])

        for side, lanes in (("left", left_lanes), ("right", right_lanes)):
            if not lanes:
                continue
            side_info = result["adjacent_lanes"][side]
            side_info["exists"] = True
            side_info["direction_compatible"] = True
            side_info["ids"] = [str(getattr(obj, "id", "")) for obj in lanes[:3]]
            side_info["count"] = len(side_info["ids"])
            side_info["trajectories"] = []
            for obj in lanes[:3]:
                adj_local = self._linestring_local_coords(obj.baseline_path.linestring)
                adj_ls = LineString(adj_local)
                if perp_line is not None and ego_proj_pt is not None:
                    start_s = self._find_perp_intersection_s(perp_line, adj_ls, ego_proj_pt)
                else:
                    start_s = adj_ls.project(Point(0, 0))
                side_info["trajectories"].append(
                    self._sample_lane_trajectory(obj.baseline_path.linestring, start_s=start_s)
                )
            nearest = self._find_nearest_linestring(lanes)
            if nearest is not None:
                nc = _compute_curvature_along_linestring(nearest)
                side_info["curvature_type"] = _classify_curvature(
                    nc["max"], nc["length_m"], _end_to_end_heading_change(nearest),
                )
            cn = "左侧" if side == "left" else "右侧"
            side_info["summary"] = f"{cn}存在{side_info['count']}条同向平行车道"

        # ── 3. oncoming (opposite-direction) lanes ──
        oncoming_left: List[Any] = []
        oncoming_right: List[Any] = []
        if current_lane_ls is not None:
            oncoming_left, oncoming_right = self._search_lateral_lanes(
                current_lane_ls, all_forward, forward_lanes, current_ids_set,
                local_drivable_area=local_drivable_area, oncoming=True,
            )
            # Remove candidates that are longitudinal extensions of the current lane
            oncoming_left = self._filter_current_lane_extensions(
                oncoming_left, current_lane_ls, current_lane_candidates,
            )
            oncoming_right = self._filter_current_lane_extensions(
                oncoming_right, current_lane_ls, current_lane_candidates,
            )

        for side, lanes in (("left", oncoming_left), ("right", oncoming_right)):
            if not lanes:
                continue
            side_info = result["oncoming_lanes"][side]
            side_info["exists"] = True
            side_info["ids"] = [str(getattr(obj, "id", "")) for obj in lanes[:3]]
            side_info["count"] = len(side_info["ids"])
            side_info["trajectories"] = []
            for obj in lanes[:3]:
                adj_local = self._linestring_local_coords(obj.baseline_path.linestring)
                adj_ls = LineString(adj_local)
                if perp_line is not None and ego_proj_pt is not None:
                    start_s = self._find_perp_intersection_s(perp_line, adj_ls, ego_proj_pt)
                else:
                    start_s = adj_ls.project(Point(0, 0))
                side_info["trajectories"].append(
                    self._sample_lane_trajectory(obj.baseline_path.linestring, start_s=start_s)
                )
            cn = "左侧" if side == "left" else "右侧"
            side_info["summary"] = f"{cn}存在{side_info['count']}条对向车道"

        # ── semantic description ──
        cl = result["current_lane"]
        if cl["ids"]:
            lane_type_str = "连接段" if cl["type"] == "lane_connector" else "车道"
            current_info = f"自车当前在{lane_type_str}上，前方道路为{cl['curvature_type']}"
        else:
            current_info = "未识别到自车当前车道"
        al = result["adjacent_lanes"]
        adjacent_info = f"{al['left']['summary']}，{al['right']['summary']}"
        ol = result["oncoming_lanes"]
        oncoming_info = f"{ol['left']['summary']}，{ol['right']['summary']}"
        result["semantic_description"] = f"{current_info}。{adjacent_info}。{oncoming_info}"

        return result

    def _sample_lane_trajectory(
        self, ls: LineString, start_s: Optional[float] = None, n_pts: int = 8,
    ) -> List[Dict[str, float]]:
        """Sample *n_pts* equally spaced along the lane from *start_s* to lane end.

        All returned points lie strictly on the lane baseline (ego-local coords).

        - When *start_s* is None (current lane): the projection of ego (origin)
          onto the lane is used as the sampling start.
        - When *start_s* is provided (adjacent lane): that s-coordinate is used.
        The lane end is always used as the sampling end.
        """
        local_coords = self._linestring_local_coords(ls)
        local_ls = LineString(local_coords)

        if start_s is None:
            start_s = local_ls.project(Point(0, 0))

        lane_length = float(local_ls.length)
        if lane_length <= 1e-6:
            return []

        start_s = min(max(float(start_s), 0.0), lane_length)
        eps = min(0.5, lane_length * 0.25)
        s_nxt = min(start_s + eps, lane_length)
        s_prv = max(start_s - eps, 0.0)
        nxt_pt = local_ls.interpolate(s_nxt)
        prv_pt = local_ls.interpolate(s_prv)
        tangent_x = float(nxt_pt.x - prv_pt.x)

        # Some nuPlan baselines are ordered opposite to the ego travel
        # direction. Sample toward ego-forward (+x) instead of blindly toward
        # increasing arc length.
        forward_remaining = lane_length - start_s
        backward_remaining = start_s
        sample_increasing = tangent_x >= 0.0
        if forward_remaining < 1e-3 and backward_remaining > forward_remaining:
            sample_increasing = False
        elif backward_remaining < 1e-3 and forward_remaining > backward_remaining:
            sample_increasing = True

        end_s = lane_length if sample_increasing else 0.0
        if abs(end_s - start_s) < 1e-3:
            return []

        s_vals = np.linspace(start_s, end_s, n_pts)
        points: List[Dict[str, float]] = []
        for s in s_vals:
            pt = local_ls.interpolate(s)
            # heading from local tangent
            eps = 0.5
            s_nxt = min(s + eps, lane_length)
            s_prv = max(s - eps, 0.0)
            nxt_pt = local_ls.interpolate(s_nxt)
            prv_pt = local_ls.interpolate(s_prv)
            if sample_increasing:
                dx = float(nxt_pt.x - prv_pt.x)
                dy = float(nxt_pt.y - prv_pt.y)
            else:
                dx = float(prv_pt.x - nxt_pt.x)
                dy = float(prv_pt.y - nxt_pt.y)
            h = math.atan2(dy, dx)
            points.append({
                "x": round(float(pt.x), 3),
                "y": round(float(pt.y), 3),
                "heading": round(h, 4),
            })
        return points

    @staticmethod
    def _has_forward_lane_samples(points: List[Dict[str, float]], min_forward_x: float = 0.0) -> bool:
        return any(float(point.get("x", 0.0)) >= min_forward_x for point in points)

    def _sample_future_overlap_trajectory(self, lane_obj: Any, n_pts: int = 8) -> List[Dict[str, float]]:
        """Use future trajectory points inside a lane polygon when its baseline is unusable."""
        try:
            future_poses = self.scene.get_future_trajectory(self._num_fut).poses
        except Exception:
            logger.debug("future trajectory unavailable for lane overlap samples", exc_info=True)
            return []
        if len(future_poses) == 0:
            return []

        polygon = getattr(lane_obj, "polygon", None)
        if polygon is None or polygon.is_empty:
            return []
        buffered_polygon = polygon.buffer(0.35)

        points: List[Dict[str, float]] = []
        for pose in future_poses:
            local_x = float(pose[0])
            local_y = float(pose[1])
            global_point = Point(*self._to_global(local_x, local_y))
            if not buffered_polygon.covers(global_point):
                if points:
                    break
                continue
            points.append({
                "x": round(local_x, 3),
                "y": round(local_y, 3),
                "heading": round(float(pose[2]), 4),
            })
            if len(points) >= n_pts:
                break

        return points

    @staticmethod
    def _find_perp_intersection_s(
        perp_line: LineString, target_ls: LineString, fallback_pt: Point,
    ) -> float:
        """Return the s coordinate on *target_ls* where *perp_line* intersects it.

        If there is no intersection, fall back to the projection of *fallback_pt*.
        When multiple intersection points exist, the one closest to *fallback_pt*
        (projected onto *perp_line*) is chosen.
        """
        intersection = perp_line.intersection(target_ls)
        if intersection.is_empty:
            return target_ls.project(fallback_pt)
        if intersection.geom_type == "Point":
            int_pt = intersection
        elif intersection.geom_type == "MultiPoint":
            pts = list(intersection.geoms)
            ref_on_perp = perp_line.interpolate(perp_line.project(fallback_pt))
            int_pt = min(pts, key=lambda p: p.distance(ref_on_perp))
        else:
            int_pt = intersection.centroid
        return target_ls.project(int_pt)

    def _build_local_drivable_area(self, ego_point: Point, radius: float) -> Optional[Any]:
        """Build a local-coordinate union of lane-level polygons.

        Do not include broad DRIVABLE_AREA/CARPARK_AREA layers here: those can
        cover shoulders or other non-lane surfaces and would hide separators
        between the current lane and a candidate lane.
        """
        layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        intersection_layer = getattr(SemanticMapLayer, "INTERSECTION", None)
        if intersection_layer is not None:
            layers.append(intersection_layer)

        try:
            map_objects = self._map_api.get_proximal_map_objects(point=ego_point, radius=radius, layers=layers)
        except Exception:
            logger.debug("map_api query failed for drivable area", exc_info=True)
            return None

        local_polygons = []
        for layer in layers:
            for obj in map_objects.get(layer, []):
                polygon = getattr(obj, "polygon", None)
                if polygon is None or polygon.is_empty:
                    continue
                try:
                    exterior = [self._to_ego_local(float(x), float(y)) for x, y in polygon.exterior.coords]
                    interiors = [
                        [self._to_ego_local(float(x), float(y)) for x, y in interior.coords]
                        for interior in polygon.interiors
                    ]
                    local_polygons.append(type(polygon)(exterior, interiors))
                except Exception:
                    logger.debug("failed to transform drivable polygon", exc_info=True)
                    continue

        if not local_polygons:
            return None

        try:
            return unary_union(local_polygons).buffer(0.01)
        except Exception:
            logger.debug("failed to union drivable polygons", exc_info=True)
            return None

    @staticmethod
    def _has_drivable_lateral_connection(
        current_lane_ls: LineString,
        candidate_ls: LineString,
        local_drivable_area: Optional[Any],
        n_samples: int = 11,
    ) -> bool:
        """Return True when lateral segments from current lane to candidate are drivable."""
        if local_drivable_area is None or local_drivable_area.is_empty:
            return True

        margin = min(1.0, current_lane_ls.length * 0.1)
        start_s = margin
        end_s = max(margin, current_lane_ls.length - margin)
        s_vals = np.linspace(start_s, end_s, n_samples)
        checked_segments = 0
        drivable = local_drivable_area.buffer(0.01)
        max_non_drivable_length = 0.01

        for s in s_vals:
            cp = current_lane_ls.interpolate(float(s))
            eps = 0.5
            nxt = current_lane_ls.interpolate(min(float(s) + eps, current_lane_ls.length))
            prv = current_lane_ls.interpolate(max(float(s) - eps, 0.0))
            dx = float(nxt.x - prv.x)
            dy = float(nxt.y - prv.y)
            norm = math.sqrt(dx * dx + dy * dy) or 1e-6
            pdx = -dy / norm
            pdy = dx / norm

            ray_len = 30.0
            perp = LineString([
                (float(cp.x) + pdx * ray_len, float(cp.y) + pdy * ray_len),
                (float(cp.x) - pdx * ray_len, float(cp.y) - pdy * ray_len),
            ])
            inter = perp.intersection(candidate_ls)
            if inter.is_empty:
                return False
            if inter.geom_type == "Point":
                ip = inter
            elif inter.geom_type == "MultiPoint":
                ip = min(list(inter.geoms), key=lambda p: p.distance(cp))
            else:
                ip = inter.centroid

            lateral_segment = LineString([(float(cp.x), float(cp.y)), (float(ip.x), float(ip.y))])
            if lateral_segment.length < 1e-3:
                return False
            non_drivable = lateral_segment.difference(drivable)
            if not non_drivable.is_empty and float(non_drivable.length) > max_non_drivable_length:
                return False
            checked_segments += 1

        return checked_segments == n_samples

    def _find_current_lane_candidates(self, lanes: List[Any], connectors: List[Any]) -> List[Any]:
        future_ranked = self._rank_current_lane_candidates_by_future_overlap(lanes + connectors)
        if future_ranked:
            return future_ranked[:1]

        ego_point = Point(self._ego_pose_se2.x, self._ego_pose_se2.y)
        containing = []
        for obj in lanes + connectors:
            try:
                if obj.polygon.contains(ego_point) or obj.polygon.distance(ego_point) < 0.8:
                    local = self._linestring_local_coords(obj.baseline_path.linestring)
                    if abs(self._local_linestring_heading_delta(local)) <= math.radians(45):
                        containing.append(obj)
            except Exception:
                continue
        if containing:
            containing.sort(key=lambda obj: 0 if obj in lanes else 1)
            return containing[:3]

        nearest = []
        for obj in lanes + connectors:
            local = self._linestring_local_coords(obj.baseline_path.linestring)
            near = local[(local[:, 0] > -8.0) & (local[:, 0] < 12.0)]
            if len(near) == 0:
                near = local
            if len(near) == 0:
                continue
            dist = float(np.min(np.linalg.norm(near[:, :2], axis=1)))
            lateral = float(np.min(np.abs(near[:, 1])))
            if lateral < 2.5 and abs(self._local_linestring_heading_delta(local)) <= math.radians(45):
                nearest.append((dist, obj))
        nearest.sort(key=lambda item: item[0])
        return [obj for _, obj in nearest[:2]]

    def _rank_current_lane_candidates_by_future_overlap(self, candidates: List[Any]) -> List[Any]:
        """Return lanes/connectors touched by future trajectory, ranked by weighted overlap.

        Future points closer to the current frame carry more voting weight so
        the selected current lane reflects the near-term lane assignment rather
        than being dominated by far-horizon route changes.
        """
        if not candidates:
            return []

        try:
            future_poses = self.scene.get_future_trajectory(self._num_fut).poses
        except Exception:
            logger.debug("future trajectory unavailable for current lane ranking", exc_info=True)
            return []
        if len(future_poses) == 0:
            return []

        future_points = [Point(*self._to_global(float(p[0]), float(p[1]))) for p in future_poses]
        n_future = len(future_points)
        if n_future == 1:
            point_weights = [1.0]
        else:
            # Linear decay: first future point has 1.0 vote, last has 0.25.
            point_weights = [
                1.0 - 0.75 * (idx / float(n_future - 1))
                for idx in range(n_future)
            ]
        line_coords = [(self._ego_pose_se2.x, self._ego_pose_se2.y)] + [(p.x, p.y) for p in future_points]
        future_line = LineString(line_coords) if len(line_coords) >= 2 else None

        scored = []
        for original_idx, obj in enumerate(candidates):
            try:
                polygon = obj.polygon
                # Small buffer tolerates map boundary noise when the trajectory rides near lane edges.
                buffered_polygon = polygon.buffer(0.35)
                overlap_length = 0.0
                if future_line is not None and future_line.length > 1e-6:
                    overlap_length = float(future_line.intersection(buffered_polygon).length)
                point_hits = 0
                weighted_hits = 0.0
                weighted_closeness = 0.0
                hit_indices: List[int] = []
                near_window_weighted_hits = 0.0
                near_window_closeness = 0.0
                near_window_size = min(5, len(future_points))
                for point_idx, (point, weight) in enumerate(zip(future_points, point_weights)):
                    closeness = max(0.0, 1.0 - float(polygon.distance(point)) / 1.5)
                    if buffered_polygon.covers(point):
                        point_hits += 1
                        weighted_hits += weight
                        hit_indices.append(point_idx)
                        if point_idx < near_window_size:
                            near_window_weighted_hits += weight
                    if point_idx < near_window_size:
                        near_window_closeness += weight * closeness
                    weighted_closeness += weight * closeness
                min_distance = min(float(polygon.distance(point)) for point in future_points)
                if point_hits <= 0 and overlap_length < 0.5:
                    continue
                earliest_hit_idx = hit_indices[0] if hit_indices else len(future_points)
                mean_hit_idx = float(np.mean(hit_indices)) if hit_indices else float(len(future_points))
                scored.append((
                    -near_window_weighted_hits,
                    -near_window_closeness,
                    -weighted_hits,
                    earliest_hit_idx,
                    mean_hit_idx,
                    -point_hits,
                    -overlap_length,
                    -weighted_closeness,
                    min_distance,
                    original_idx,
                    obj,
                ))
            except Exception:
                continue

        if not scored:
            return []

        scored.sort(key=lambda item: item[:10])
        return [obj for *_, obj in scored]

    def _extract_traffic_flow(self) -> Dict[str, Any]:
        """Analyze surrounding vehicles for traffic flow state."""
        vehicles = []
        for box, name, vel in zip(self._anns.boxes, self._anns.names, self._anns.velocity_3d):
            if name != "vehicle":
                continue
            x, y = box[BoundingBoxIndex.X], box[BoundingBoxIndex.Y]
            if not _xy_in_range(x, y, (-50, 100), (-20, 20)):
                continue
            dist = math.sqrt(x ** 2 + y ** 2)
            speed = _speed_2d(vel)
            vehicles.append({
                "x": float(x), "y": float(y),
                "distance": dist, "speed": speed,
            })

        vehicles.sort(key=lambda v: v["distance"])
        forward_vehicles = [v for v in vehicles if 0 < v["x"] < 50 and abs(v["y"]) < 10]

        # lead vehicle: closest vehicle ahead (x > 0) with |y| < 3m
        lead = None
        for v in forward_vehicles:
            if v["x"] > 0 and abs(v["y"]) < 3.0:
                lead = v
                break

        # flow classification
        ego_speed = _speed_2d(self._ego.ego_velocity)
        avg_forward_speed = float(np.mean([v["speed"] for v in forward_vehicles])) if forward_vehicles else None
        stopped_or_slow = sum(1 for v in forward_vehicles if v["speed"] < 4.2)
        density = len(forward_vehicles) / 50.0

        if lead is None:
            flow_state = "视野开阔，无前车干扰"
        elif (lead["speed"] < 4.2 and len(forward_vehicles) >= 3) or stopped_or_slow >= 3:
            flow_state = "前方车流拥堵（低速车辆较多）"
        elif lead["distance"] < 15 or (ego_speed > 0.5 and lead["distance"] / ego_speed < 1.5):
            flow_state = "跟车密集（车距较近）"
        elif avg_forward_speed is not None and avg_forward_speed > 8.0 and density < 0.12:
            flow_state = "车流顺畅"
        else:
            flow_state = "车流平稳"

        lead_info = None
        if lead is not None:
            headway = None if ego_speed < 0.5 else lead["distance"] / ego_speed
            lead_info = {
                "distance_m": round(lead["distance"], 1),
                "speed_mps": round(lead["speed"], 1),
                "headway_s": None if headway is None else round(headway, 1),
                "position": _classify_position(lead["x"], lead["y"]),
            }

        return {
            "lead_vehicle": lead_info,
            "nearby_vehicle_count": len(vehicles),
            "forward_vehicle_count": len(forward_vehicles),
            "forward_vehicle_density_per_m": round(density, 3),
            "avg_forward_vehicle_speed_mps": None if avg_forward_speed is None else round(avg_forward_speed, 1),
            "ego_speed_mps": round(ego_speed, 1),
            "flow_state": flow_state,
        }

    # ── 2. agent grounding ───────────────────────────────────────────────────

    def _annotate_agent_grounding(self) -> Dict[str, Any]:
        cam_f0 = self._cur_frame.cameras.cam_f0
        image_size = _camera_image_size(cam_f0)
        boxes_2d: List = []
        if cam_f0.intrinsics is not None:
            boxes_2d = _project_boxes_to_2d(self._anns, cam_f0)
        else:
            boxes_2d = [None] * len(self._anns.boxes)  # cameras not loaded

        # compute accelerations from previous frame via track_token
        prev_accel = {}
        prev_speed = {}
        if self._cur_idx > 0:
            prev_frame = self.scene.frames[self._cur_idx - 1]
            for i, track_id in enumerate(prev_frame.annotations.track_tokens):
                pv = prev_frame.annotations.velocity_3d[i]
                prev_speed[track_id] = _speed_2d(pv)
            for i, track_id in enumerate(self._anns.track_tokens):
                cv = self._anns.velocity_3d[i]
                cur_spd = _speed_2d(cv)
                if track_id in prev_speed:
                    prev_accel[track_id] = (cur_spd - prev_speed[track_id]) / self._dt

        agents = []
        for i, (box, name, vel, track_id) in enumerate(
            zip(self._anns.boxes, self._anns.names, self._anns.velocity_3d, self._anns.track_tokens)
        ):
            x, y = box[BoundingBoxIndex.X], box[BoundingBoxIndex.Y]
            distance = math.sqrt(x ** 2 + y ** 2)

            # filter to relevant classes within 80m
            if name not in ("vehicle", "pedestrian", "bicycle"):
                continue
            if distance > 80:
                continue

            length = float(box[BoundingBoxIndex.LENGTH])
            width = float(box[BoundingBoxIndex.WIDTH])
            heading = float(box[BoundingBoxIndex.HEADING])
            speed = _speed_2d(vel)
            accel = prev_accel.get(track_id)

            b2d = boxes_2d[i]
            bbox_2d = None
            bbox_image_region = "不可见"
            if b2d is not None:
                bbox_2d = {
                    "xmin": round(b2d[0], 1),
                    "ymin": round(b2d[1], 1),
                    "xmax": round(b2d[2], 1),
                    "ymax": round(b2d[3], 1),
                }
                bbox_image_region = self._classify_bbox_image_region(b2d, image_size)

            importance_score = self._agent_importance_score(name, float(x), float(y), distance, speed, bbox_2d, image_size)
            visible_in_front = bbox_2d is not None
            state_description = self._agent_state_description(name, speed, accel, heading, float(x), float(y))
            distance_description = self._agent_distance_description(float(x), float(y), distance)

            agents.append({
                "class": name,
                "track_token": track_id,
                "position": _classify_position(x, y),
                "distance_m": round(distance, 1),
                "longitudinal_m": round(float(x), 1),
                "lateral_m": round(float(y), 1),
                "lateral_abs_m": round(abs(float(y)), 1),
                "lateral_side": _classify_lateral_side(float(y)),
                "speed_mps": round(speed, 1),
                "heading_rad": round(heading, 3),
                "length_m": round(length, 2),
                "width_m": round(width, 2),
                "bbox_2d": bbox_2d,
                "bbox_image_region": bbox_image_region,
                "acceleration_mps2": round(accel, 2) if accel is not None else None,
                "state_description": state_description,
                "distance_description": distance_description,
                "importance_score": round(importance_score, 3),
                "is_important": importance_score >= 0.35,
                "visible_in_front": visible_in_front,
            })

        # Sort all agents by semantic priority, importance, and distance.
        agents.sort(key=lambda a: (
            self._agent_priority_group(
                a["class"], a["longitudinal_m"], a["lateral_m"],
                a["speed_mps"], a["heading_rad"],
            ),
            -a["importance_score"],
            not a["visible_in_front"],
            a["distance_m"],
        ))
        important_agents = [a for a in agents if a["is_important"]][:15]
        if not important_agents:
            important_agents = [a for a in agents if a["bbox_2d"] is not None or a["distance_m"] < 30][:8]
        return {
            "agents": agents,
            "important_agents": important_agents,
            "agent_count": len(agents),
        }

    @staticmethod
    def _classify_bbox_image_region(
        bbox: Tuple[float, float, float, float],
        image_size: Tuple[int, int],
    ) -> str:
        xmin, ymin, xmax, ymax = bbox
        image_w, image_h = image_size
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        horizontal = "左侧" if cx < image_w / 3 else "右侧" if cx > image_w * 2 / 3 else "中央"
        vertical = "上方" if cy < image_h / 3 else "下方" if cy > image_h * 2 / 3 else "中部"
        return f"图像{vertical}{horizontal}"

    @staticmethod
    def _agent_priority_group(
        class_name: str,
        longitudinal_m: float,
        lateral_m: float,
        speed_mps: float = 0.0,
        heading_rad: float = 0.0,
    ) -> int:
        """Return priority group: 0=pedestrian, 1=moving lead (current+oncoming), 2=moving, 3=other.

        Lower number = higher priority. Within each group, agents are sorted by distance.
        """
        MOVING_SPEED = 0.3  # m/s — threshold above which an agent is considered "moving"

        # Priority 0: pedestrian (highest)
        if class_name == "pedestrian":
            return 0

        is_moving = speed_mps > MOVING_SPEED

        # Priority 1: moving lead vehicle in current lane or oncoming lane
        if class_name == "vehicle" and is_moving and longitudinal_m > 0:
            # current lane: |lateral| < 3 m
            if abs(lateral_m) < 3.0:
                return 1
            # oncoming lane: heading roughly opposite to ego (|heading| > ~115°)
            if abs(heading_rad) > 2.0:
                return 1

        # Priority 2: other moving vehicles (vehicle, bicycle)
        if class_name in ("vehicle", "bicycle") and is_moving:
            return 2

        # Priority 3: everything else (static, behind ego, etc.)
        return 3

    @staticmethod
    def _agent_importance_score(
        name: str,
        x: float,
        y: float,
        distance: float,
        speed: float,
        bbox_2d: Optional[Dict[str, float]],
        image_size: Tuple[int, int],
    ) -> float:
        score = 0.0
        if x > -5 and abs(y) < 3.0:
            score += 0.45
        elif x > -10 and abs(y) < 6.0:
            score += 0.25
        if distance < 15:
            score += 0.35
        elif distance < 30:
            score += 0.2
        elif distance < 50:
            score += 0.08
        if name in ("pedestrian", "bicycle") and x > -5 and distance < 40:
            score += 0.25
        if speed < 0.5 and x > 0 and abs(y) < 4.0:
            score += 0.15
        if bbox_2d is not None:
            area = max(0.0, bbox_2d["xmax"] - bbox_2d["xmin"]) * max(0.0, bbox_2d["ymax"] - bbox_2d["ymin"])
            image_w, image_h = image_size
            score += min(area / max(float(image_w * image_h), 1.0), 0.2)
        return score

    @staticmethod
    def _agent_state_description(
        name: str, speed: float, accel: Optional[float], heading: float, x: float, y: float
    ) -> str:
        if speed < 0.3:
            motion = "静止或近似静止"
        elif accel is not None and accel < -0.8:
            motion = "明显减速"
        elif accel is not None and accel < -0.25:
            motion = "轻微减速"
        elif accel is not None and accel > 0.5:
            motion = "加速"
        else:
            motion = "稳定运动"

        rel_heading = _normalize_angle(heading)
        if abs(rel_heading) > 0.35 and x > -5:
            orient = "存在横向朝向/转向趋势"
        else:
            orient = "朝向与道路基本一致"
        return f"{name}{motion}，{orient}，位于{_classify_position(x, y)}"

    @staticmethod
    def _agent_distance_description(x: float, y: float, distance: float) -> str:
        if x >= 0:
            longitudinal = f"前方{x:.1f}米"
        else:
            longitudinal = f"后方{abs(x):.1f}米"
        side = _classify_lateral_side(y)
        if side == "本车道附近":
            return f"位于{longitudinal}，横向接近本车道，欧氏距离{distance:.1f}米"
        return f"位于{longitudinal}、{side}横向{abs(y):.1f}米，欧氏距离{distance:.1f}米"

    # ── 3. planning ──────────────────────────────────────────────────────────

    def _annotate_planning(self) -> Dict[str, Any]:
        cmd_str = _driving_command_str(self._ego.driving_command)
        lane = self._extract_lane_structure()
        flow = self._extract_traffic_flow()
        fut_traj = self.scene.get_future_trajectory(self._num_fut)
        lane_change = self._detect_lane_change(lane, fut_traj.poses)
        reasoning_context = self._build_reasoning_context(cmd_str, flow, lane, lane_change)

        result: Dict[str, Any] = {
            "nav_command": cmd_str,
            "reasoning_context": reasoning_context,
        }
        if lane_change is not None:
            result["lane_change"] = lane_change
        return result

    @staticmethod
    def _local_linestring_heading_delta(local: npt.NDArray[np.float64]) -> float:
        """Estimate lane direction relative to ego while preserving baseline point order."""
        if len(local) < 2:
            return math.pi

        segment_vectors = []
        for start, end in zip(local[:-1], local[1:]):
            midpoint = (start + end) / 2.0
            if -5.0 <= midpoint[0] <= 35.0 and abs(midpoint[1]) <= 8.0:
                segment_vectors.append(end - start)

        if segment_vectors:
            direction = np.sum(np.asarray(segment_vectors), axis=0)
        else:
            direction = local[-1] - local[0]

        if float(np.linalg.norm(direction[:2])) < 1e-3:
            return math.pi

        heading = math.atan2(float(direction[1]), float(direction[0]))
        return _normalize_angle(heading)

    @staticmethod
    def _global_linestring_heading(local: npt.NDArray[np.float64]) -> float:
        """Compute lane heading from ALL segments (no ego-centric window).

        Unlike ``_local_linestring_heading_delta``, this uses every segment so
        that the direction reflects the laneʼs *global* course — required for
        the strict parallelism check in lateral lane search.
        """
        if len(local) < 2:
            return math.pi
        direction = np.sum(local[1:] - local[:-1], axis=0)
        norm = float(np.linalg.norm(direction[:2]))
        if norm < 1e-3:
            return math.pi
        return math.atan2(float(direction[1]), float(direction[0]))

    def _build_reasoning_context(
        self,
        nav_cmd: str,
        flow: Dict[str, Any],
        lane: Dict[str, Any],
        lane_change: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Provide structured context to help VLM reason about the decision."""
        reasons: List[str] = []

        # Lane change takes priority in reasoning
        if lane_change is not None:
            reasons.append(lane_change["description"])

        if nav_cmd in ("左转", "右转"):
            reasons.append(f"导航指令要求{nav_cmd}")
            reasons.append(f"前方道路{lane['current_lane']['curvature_type']}")
        elif nav_cmd == "停止":
            if flow["lead_vehicle"] is not None and flow["lead_vehicle"]["distance_m"] < 20:
                reasons.append("前方存在近距离车辆")
            else:
                reasons.append("导航指令要求停止或低速等待")
        else:
            if lane_change is None:
                if flow["lead_vehicle"] is not None:
                    reasons.append("跟随前车行驶")
                else:
                    reasons.append("车道畅通，持续直行")

        return "; ".join(reasons) if reasons else "无特殊原因"

    def _detect_lane_change(
        self,
        lane_structure: Dict[str, Any],
        future_poses: npt.NDArray[np.float64],
    ) -> Optional[Dict[str, Any]]:
        """Detect whether the future trajectory changes from the current lane to an adjacent lane.

        Compares the first 1/3 and last 1/3 of *future_poses* against the
        currently occupied lane and the future/adjacent lane trajectories. A
        lane change is reported when the closest lane shifts from one lane to
        a different one. Lateral shift is reported as diagnostic information,
        but it is not a hard gate.

        Returns None when no lane change is detected.
        """
        current_traj = lane_structure["current_lane"].get("trajectory_points", [])
        current_ids = lane_structure["current_lane"].get("ids", [])
        left_trajs = lane_structure["adjacent_lanes"]["left"].get("trajectories", [])
        right_trajs = lane_structure["adjacent_lanes"]["right"].get("trajectories", [])

        if len(future_poses) < 3:
            return None

        # Build lane lookup: name → trajectory points
        lanes: Dict[str, List[Dict[str, float]]] = {}
        lane_ids: Dict[str, Optional[str]] = {}

        ego_lane = self._find_ego_occupied_lane_for_lane_change()
        ego_lane_id = None
        if ego_lane is not None and ego_lane.get("trajectory"):
            ego_lane_id = ego_lane.get("id")
            lanes["自车当前占用车道"] = ego_lane["trajectory"]
            lane_ids["自车当前占用车道"] = ego_lane_id

        selected_current_id = str(current_ids[0]) if current_ids else None
        if current_traj and selected_current_id != ego_lane_id:
            current_name = "未来轨迹重叠车道" if ego_lane_id is not None else "当前车道"
            lanes[current_name] = current_traj
            lane_ids[current_name] = selected_current_id

        left_ids = lane_structure["adjacent_lanes"]["left"].get("ids", [])
        right_ids = lane_structure["adjacent_lanes"]["right"].get("ids", [])
        for i, traj in enumerate(left_trajs):
            if traj:
                lanes[f"左侧车道{i + 1}"] = traj
                lane_ids[f"左侧车道{i + 1}"] = str(left_ids[i]) if i < len(left_ids) else None
        for i, traj in enumerate(right_trajs):
            if traj:
                lanes[f"右侧车道{i + 1}"] = traj
                lane_ids[f"右侧车道{i + 1}"] = str(right_ids[i]) if i < len(right_ids) else None

        if len(lanes) < 2:
            return None  # no other lanes to switch to

        # Compare first 1/3 and last 1/3 of the future trajectory
        n = len(future_poses)
        n_seg = max(1, n // 3)
        start_pts = future_poses[:n_seg]
        end_pts = future_poses[-n_seg:]

        if "自车当前占用车道" in lanes:
            start_lane = "自车当前占用车道"
        else:
            start_lane = self._find_closest_lane(start_pts, lanes)
        end_lane = self._find_closest_lane(end_pts, lanes)

        if start_lane is None or end_lane is None or start_lane == end_lane:
            return None

        lateral_shift = float(future_poses[-1, 1] - future_poses[0, 1])

        # Determine direction from lane name first, then lane relative offset.
        direction: Optional[str] = None
        if "左" in end_lane:
            direction = "左"
        elif "右" in end_lane:
            direction = "右"
        if direction is None and start_lane in lanes and end_lane in lanes:
            lane_delta = self._lane_traj_lateral_delta(lanes[start_lane], lanes[end_lane])
            if lane_delta > 0.2:
                direction = "左"
            elif lane_delta < -0.2:
                direction = "右"
        if direction is None:
            if lateral_shift > 0.0:
                direction = "左"
            elif lateral_shift < 0.0:
                direction = "右"
        if direction is None:
            return None

        return {
            "is_lane_change": True,
            "direction": direction,
            "from_lane": start_lane,
            "to_lane": end_lane,
            "from_lane_id": lane_ids.get(start_lane),
            "to_lane_id": lane_ids.get(end_lane),
            "lateral_shift_m": round(abs(lateral_shift), 3),
            "criterion": "future_trajectory_start_end_lane_change",
            "description": (
                f"规划轨迹从{start_lane}向{end_lane}变道"
                f"（横向偏移{abs(lateral_shift):.2f}m）"
            ),
        }

    def _find_ego_occupied_lane_for_lane_change(self) -> Optional[Dict[str, Any]]:
        """Find the lane currently occupied by ego for lane-change start state."""
        ego_point = Point(self._ego_pose_se2.x, self._ego_pose_se2.y)
        layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        try:
            map_objects = self._map_api.get_proximal_map_objects(
                point=ego_point,
                radius=20.0,
                layers=layers,
            )
        except Exception:
            logger.debug("map_api query failed for ego occupied lane", exc_info=True)
            return None

        candidates = []
        for layer_idx, layer in enumerate(layers):
            for obj in map_objects.get(layer, []):
                polygon = getattr(obj, "polygon", None)
                if polygon is None or polygon.is_empty:
                    continue
                try:
                    contains_ego = polygon.contains(ego_point)
                    polygon_dist = float(polygon.distance(ego_point))
                    if not contains_ego and polygon_dist >= 0.8:
                        continue
                    local = self._linestring_local_coords(obj.baseline_path.linestring)
                    if abs(self._local_linestring_heading_delta(local)) > math.radians(45):
                        continue
                    near = local[(local[:, 0] > -8.0) & (local[:, 0] < 12.0)]
                    if len(near) == 0:
                        near = local
                    baseline_dist = float(np.min(np.linalg.norm(near[:, :2], axis=1))) if len(near) else float("inf")
                    candidates.append((0 if contains_ego else 1, polygon_dist, baseline_dist, layer_idx, obj))
                except Exception:
                    continue

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[:4])
        obj = candidates[0][4]
        return {
            "id": str(getattr(obj, "id", "")),
            "type": "lane_connector" if obj in map_objects.get(SemanticMapLayer.LANE_CONNECTOR, []) else "lane",
            "trajectory": self._sample_lane_trajectory(obj.baseline_path.linestring),
        }

    @staticmethod
    def _lane_traj_lateral_delta(
        from_traj: List[Dict[str, float]],
        to_traj: List[Dict[str, float]],
    ) -> float:
        """Estimate target lane lateral offset relative to source lane in ego frame."""
        n = min(len(from_traj), len(to_traj))
        if n == 0:
            return 0.0
        return float(np.mean([to_traj[i]["y"] - from_traj[i]["y"] for i in range(n)]))

    @staticmethod
    def _find_closest_lane(
        points: npt.NDArray[np.float64],
        lanes: Dict[str, List[Dict[str, float]]],
    ) -> Optional[str]:
        """Return the name of the lane whose trajectory has the smallest average
        Euclidean distance to *points*.
        """
        best_lane: Optional[str] = None
        best_dist = float("inf")
        for lane_name, traj in lanes.items():
            total_dist = 0.0
            count = 0
            for i in range(len(points)):
                px = float(points[i, 0])
                py = float(points[i, 1])
                min_d = float("inf")
                for tp in traj:
                    d = math.sqrt((px - tp["x"]) ** 2 + (py - tp["y"]) ** 2)
                    if d < min_d:
                        min_d = d
                total_dist += min_d
                count += 1
            avg_dist = total_dist / count if count > 0 else float("inf")
            if avg_dist < best_dist:
                best_dist = avg_dist
                best_lane = lane_name
        return best_lane

    # ── 4. acting ────────────────────────────────────────────────────────────

    def _annotate_acting(self) -> Dict[str, Any]:
        hist_traj = self.scene.get_history_trajectory(self._num_hist)
        fut_traj = self.scene.get_future_trajectory(self._num_fut)

        hist_desc, hist_str = self._describe_history(hist_traj.poses)
        fut_desc, fut_str = self._describe_future(fut_traj.poses)

        vx, vy = self._ego.ego_velocity
        ax, ay = self._ego.ego_acceleration

        return {
            "history_motion_description": hist_desc,
            "history_trajectory_str": hist_str,
            "current_speed_mps": round(math.sqrt(vx ** 2 + vy ** 2), 2),
            "current_accel_x": round(float(ax), 3),
            "current_accel_y": round(float(ay), 3),
            "future_trajectory_description": fut_desc,
            "future_trajectory_str": fut_str,
            "future_trajectory_points": [
                [round(_clean_float(p[0]), 3), round(_clean_float(p[1]), 3), round(_clean_float(p[2]), 4)] for p in fut_traj.poses
            ],
            "future_total_lateral_m": round(_clean_float(fut_traj.poses[-1, 1]), 3) if len(fut_traj.poses) else 0.0,
            "future_heading_change_rad": round(_clean_float(_normalize_angle(fut_traj.poses[-1, 2])), 4) if len(fut_traj.poses) else 0.0,
        }

    def _describe_history(self, poses: npt.NDArray) -> Tuple[str, str]:
        """Generate natural-language description of past ego motion."""
        parts = []
        for i in range(len(poses)):
            t = (len(poses) - 1 - i) * self._dt
            label = "t当前" if t < 1e-6 else f"t-{t:.1f}s"
            parts.append(
                f"{label}: ({_clean_float(poses[i, 0]):.2f}, {_clean_float(poses[i, 1]):.2f}, "
                f"{_clean_float(poses[i, 2]):.3f})"
            )

        if len(poses) >= 2:
            dx = poses[-1, 0] - poses[0, 0]
            dy = poses[-1, 1] - poses[0, 1]
            dh = _normalize_angle(poses[-1, 2] - poses[0, 2])
            duration = max((len(poses) - 1) * self._dt, self._dt)
            avg_speed = math.sqrt(dx ** 2 + dy ** 2) / duration
            if avg_speed < 0.2 and abs(dh) < 0.01:
                desc = f"此前{duration:.1f}秒内，车辆保持静止或近似静止，位置和航向基本不变"
            elif abs(dh) < 0.01 and abs(dy) < 0.1:
                desc = f"此前{duration:.1f}秒内，车辆基本直行，平均速度约{avg_speed:.1f}m/s，横摆角速度接近0"
            elif abs(dh) < 0.05:
                desc = f"此前{duration:.1f}秒内，车辆略微修正方向（航向变化={dh:.3f}rad，横向位移={dy:.2f}m）"
            else:
                desc = f"此前{duration:.1f}秒内，车辆转向中（航向变化={dh:.3f}rad，横向位移={dy:.2f}m）"
        else:
            desc = "历史轨迹数据不足"

        return desc, " | ".join(parts)

    def _describe_future(self, poses: npt.NDArray) -> Tuple[str, str]:
        """Generate natural-language description of future trajectory."""
        parts = []
        for i in range(len(poses)):
            parts.append(
                f"t+{(i + 1) * self._dt:.1f}s: ({_clean_float(poses[i, 0]):.2f}, "
                f"{_clean_float(poses[i, 1]):.2f}, {_clean_float(poses[i, 2]):.3f})"
            )

        if len(poses) == 0:
            return "未来轨迹数据不足", ""

        total_lateral = _clean_float(float(poses[-1, 1]))
        duration = max(len(poses) * self._dt, self._dt)
        heading_change = _clean_float(float(_normalize_angle(poses[-1, 2])))
        heading_rate = heading_change / duration
        total_forward = _clean_float(float(poses[-1, 0]))
        path_points = np.vstack([np.zeros((1, 2), dtype=poses.dtype), poses[:, :2]])
        path_len = float(np.sum(np.linalg.norm(np.diff(path_points, axis=0), axis=1)))

        if path_len < 0.5 and abs(heading_change) < 0.01:
            desc = f"未来{duration:.1f}秒轨迹保持静止或近似静止，车辆位置和航向基本不变"
        elif abs(heading_rate) < 0.005 and abs(total_lateral) < 0.2:
            desc = f"未来{duration:.1f}秒轨迹近乎直线，保持当前车道，前进约{total_forward:.1f}m"
        elif abs(heading_rate) < 0.02:
            direction = '左' if total_lateral > 0 else '右'
            desc = f"未来轨迹为极缓{direction}向弧线，横向偏移{abs(total_lateral):.2f}m，路径长度约{path_len:.1f}m"
        elif abs(heading_rate) < 0.1:
            direction = '左' if total_lateral > 0 else '右'
            desc = f"未来轨迹为中等{direction}向弯道，横向偏移{abs(total_lateral):.2f}m，航向变化{heading_change:.3f}rad"
        else:
            desc = f"急弯，航向变化率{abs(heading_rate):.3f}rad/s"

        return desc, " | ".join(parts)

    # ── map helpers ──────────────────────────────────────────────────────────

    def _find_nearest_linestring(self, map_objects_list) -> Optional[LineString]:
        """Return the baseline linestring closest to ego origin from a list of map objects."""
        candidates = []
        for obj in map_objects_list:
            ls = obj.baseline_path.linestring
            local = self._linestring_local_coords(ls)
            forward = local[(local[:, 0] > -5) & (local[:, 0] < 30)]
            if len(forward) == 0:
                forward = local
            dist = float(np.min(np.linalg.norm(forward[:, :2], axis=1)))
            candidates.append((dist, ls))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    # ── map-graph helpers ────────────────────────────────────────────────────

    @staticmethod
    def _are_map_objects_connected(obj_a: Any, obj_b: Any) -> bool:
        """True when *obj_a* and *obj_b* are directly connected in the map graph."""
        b_id = str(obj_b.id)
        for edge in obj_a.outgoing_edges:
            if str(edge.id) == b_id:
                return True
        for edge in obj_a.incoming_edges:
            if str(edge.id) == b_id:
                return True
        return False

    @staticmethod
    def _collect_reachable_ids(root_objs: List[Any], steps: int = 1) -> set:
        """Return IDs reachable from *root_objs* within *steps* graph hops."""
        reachable: set = set()
        frontier = list(root_objs)
        for _ in range(steps):
            next_frontier: List[Any] = []
            for obj in frontier:
                for edge in list(obj.outgoing_edges) + list(obj.incoming_edges):
                    eid = str(edge.id)
                    if eid not in reachable:
                        reachable.add(eid)
                        next_frontier.append(edge)
            frontier = next_frontier
        return reachable

    @staticmethod
    def _merge_connected(items: List[Tuple[float, bool, Any]]) -> List[Any]:
        """Merge items that are connected in the map graph, keeping the closest."""
        if len(items) <= 1:
            return [obj for _, _, obj in items]

        items = sorted(items, key=lambda x: x[0])  # by lateral distance
        skip: set = set()
        for i in range(len(items)):
            if i in skip:
                continue
            for j in range(i + 1, len(items)):
                if j in skip:
                    continue
                if SceneAnnotator._are_map_objects_connected(items[i][2], items[j][2]):
                    skip.add(j)  # keep items[i] (closer to current lane)
        return [obj for idx, (_, _, obj) in enumerate(items) if idx not in skip]

    # ── adjacent lane search ─────────────────────────────────────────────────

    @staticmethod
    def _cluster_by_lane_width(
        items: List[Tuple[float, bool, Any]], lane_width: float = 3.5,
    ) -> List[List[Tuple[float, bool, Any]]]:
        """Group candidates into lane-width buckets by lateral offset |d|.

        Items within ~0.7 × *lane_width* of the first item in a cluster are
        considered the same lane.  Returns clusters sorted nearest-first.
        """
        if not items:
            return []
        items = sorted(items, key=lambda x: x[0])
        clusters: List[List[Tuple[float, bool, Any]]] = []
        cluster = [items[0]]
        center = items[0][0]
        for it in items[1:]:
            if it[0] - center < lane_width * 0.7:
                cluster.append(it)
            else:
                clusters.append(cluster)
                cluster = [it]
                center = it[0]
        clusters.append(cluster)
        return clusters

    def _search_lateral_lanes(
        self,
        current_lane_ls: LineString,
        all_forward: List[Any],
        forward_lanes: List[Any],
        current_ids_set: set,
        local_drivable_area: Optional[Any] = None,
        oncoming: bool = False,
    ) -> Tuple[List[Any], List[Any]]:
        """Search laterally from the current lane baseline for adjacent lanes.

        Origin    = ego projected onto current lane.
        X-axis    = lane forward direction.
        Y-axis    = perpendicular (left positive).
        Clustering = lane-width buckets → arbitrary number of lanes per side.
        Side       = ``is_left_of`` / ``is_right_of`` (map API), fallback to Frenet d.

        When *oncoming* is False, keeps lanes whose heading delta is ≤ 45°
        (same-direction).  When True, keeps lanes with |heading_delta| > 135°
        (opposite-direction).
        """
        MIN_LATERAL = 1.5  # |d| below this → same lane, skip

        left_candidates: List[Tuple[float, bool, Any]] = []
        right_candidates: List[Tuple[float, bool, Any]] = []
        current_objs = [o for o in all_forward if str(getattr(o, "id", "")) in current_ids_set]

        for obj in all_forward:
            obj_id = str(getattr(obj, "id", ""))
            if obj_id in current_ids_set:
                continue

            local = self._linestring_local_coords(obj.baseline_path.linestring)
            if len(local) < 2:
                continue

            # ── global direction check (entire lane, not just a window) ──
            heading_delta = self._global_linestring_heading(local)
            if oncoming:
                if abs(heading_delta) <= math.radians(135):
                    continue
            else:
                if abs(heading_delta) > math.radians(45):
                    continue

            # ── lateral offset & parallelism (current-lane-centric) ──
            avg_d, d_std = self._measure_lateral_from_current_lane(
                current_lane_ls, local,
            )
            if abs(avg_d) < MIN_LATERAL:
                continue  # same lane (or noise)
            if d_std > 1.0:   # > 1 m std → lane diverges / converges
                continue
            candidate_ls = LineString(local)
            if not self._has_drivable_lateral_connection(
                current_lane_ls, candidate_ls, local_drivable_area,
            ):
                continue

            is_lane = obj in forward_lanes

            # Determine side: prefer map API, fall back to Frenet sign
            is_left = False
            is_right = False
            if current_objs:
                try:
                    is_left = any(obj.is_left_of(c) for c in current_objs)
                    is_right = any(obj.is_right_of(c) for c in current_objs)
                except (AssertionError, NotImplementedError):
                    is_left = avg_d > 0
                    is_right = avg_d < 0
            if not is_left and not is_right:
                is_left = avg_d > 0
                is_right = avg_d < 0

            if is_left:
                left_candidates.append((abs(avg_d), is_lane, obj))
            elif is_right:
                right_candidates.append((abs(avg_d), is_lane, obj))
            # If neither left nor right (e.g. directly ahead), skip

        # Cluster into lane-width groups (each group = one adjacent lane)
        left_clusters = self._cluster_by_lane_width(left_candidates)
        right_clusters = self._cluster_by_lane_width(right_candidates)

        # Within each cluster: merge graph-connected objects, pick the closest
        left_result: List[Any] = []
        for cluster in left_clusters:
            merged = self._merge_connected(cluster)
            if merged:
                left_result.append(merged[0])

        right_result: List[Any] = []
        for cluster in right_clusters:
            merged = self._merge_connected(cluster)
            if merged:
                right_result.append(merged[0])

        return left_result, right_result

    def _filter_current_lane_extensions(
        self,
        candidates: List[Any],
        current_lane_ls: LineString,
        current_lane_objs: List[Any],
    ) -> List[Any]:
        """Remove candidates that are graph-connected to the current lane."""
        if not candidates or not current_lane_objs:
            return candidates

        # IDs directly reachable from current lane (extensions / predecessors)
        extension_ids = self._collect_reachable_ids(current_lane_objs, steps=1)

        return [c for c in candidates if str(c.id) not in extension_ids]
