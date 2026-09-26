"""No-API frontier exploration baselines with local target stopping.

The default policy receives the explicit target category, explores from
walkable geometry and visit history, then switches to an active last-mile
controller after target evidence appears.  The controller approaches a
reachable standoff pose, faces the candidate, and requires spatially
consistent multi-view evidence before STOP.  ``--legacy-full-vocabulary``
retains the original pure-frontier policy for paired ablations.

Neither policy reads the intent when choosing navigation actions or calls a
hosted model.  They are navigation-only calibrations, not implicit-intent
baselines.

Example (requires the goodnav + Isaac Sim environment):
  python agents/agent_fpe.py --style formal --limit 1
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent / "simulator"))
sys.path.insert(0, str(THIS_DIR.parent))

import dino_detector  # noqa: E402
from common import (  # noqa: E402
    EPISODES_FILE,
    REPO,
    STYLES,
    USD_ROOT,
    episode_path,
    intent_for_style,
    load_items,
    now_iso,
    save_atomic,
)
from walkable_map import WalkableMap  # noqa: E402


PROTOCOL_VERSION = "strict_stop_2026_08"
ACTIVE_PROTOCOL_VERSION = "strict_stop_2026_08_active_perception"
LEGACY_MODEL_NAME = "fpe_dino_explicit"
ACTIVE_MODEL_NAME = "fpe_active_dino_explicit"
SYNONYM_FILE = THIS_DIR.parent / "vocab" / "category_synonyms.yaml"


def load_detector_synonyms(path: Path = SYNONYM_FILE) -> dict[str, list[str]]:
    """Load optional visual aliases without making PyYAML a hard import."""
    if not path.exists():
        return {}
    try:
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return {
        normalize_detector_label(category): [
            normalize_detector_label(alias)
            for alias in aliases or []
            if normalize_detector_label(alias)
        ]
        for category, aliases in raw.items()
    }


def build_target_queries(
    target_category: str,
    synonym_map: dict[str, list[str]],
    canonical_categories: list[str],
    max_aliases: int = 2,
) -> list[str]:
    """Build a compact, collision-aware target-only detector prompt.

    Full benchmark vocabularies cause unrelated phrases to compete in a
    single GroundingDINO prompt.  The active policy instead queries the
    explicit goal plus at most two unambiguous visual aliases.  Aliases that
    are themselves another benchmark category, or that belong to multiple
    categories, are excluded so that detector recall is not bought by
    silently changing the goal semantics.
    """
    target = normalize_detector_label(target_category)
    canonicals = {normalize_detector_label(value) for value in canonical_categories}
    alias_owners: dict[str, set[str]] = {}
    for owner, aliases in synonym_map.items():
        for alias in aliases:
            alias_owners.setdefault(alias, set()).add(owner)

    queries = [target]
    for alias in synonym_map.get(target, []):
        if alias == target or alias in queries:
            continue
        if alias in canonicals and alias != target:
            continue
        if len(alias_owners.get(alias, {target})) > 1:
            continue
        queries.append(alias)
        if len(queries) >= 1 + max(0, max_aliases):
            break
    return queries


def load_episodes() -> dict[str, dict]:
    episodes = {}
    with EPISODES_FILE.open(encoding="utf-8") as f:
        for line in f:
            episode = json.loads(line)
            episodes[episode["selection_id"]] = episode
    return episodes


def normalize_detector_label(label: str) -> str:
    """Normalize a DINO text span for strict vocabulary matching."""
    normalized = str(label or "").strip().lower().replace("_", " ")
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return re.sub(r"^(?:a|an|the) ", "", normalized)


def bbox_depth_m(
    depth: np.ndarray | None,
    bbox: list[float],
    inner_fraction: float = 0.5,
) -> float | None:
    """Robust median depth from the central part of a detection box."""
    if depth is None or depth.ndim < 2 or len(bbox) != 4:
        return None
    height, width = depth.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in bbox)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w = max(1.0, (x2 - x1) * inner_fraction / 2.0)
    half_h = max(1.0, (y2 - y1) * inner_fraction / 2.0)
    left = max(0, min(width - 1, int(math.floor(cx - half_w))))
    right = max(left + 1, min(width, int(math.ceil(cx + half_w))))
    top = max(0, min(height - 1, int(math.floor(cy - half_h))))
    bottom = max(top + 1, min(height, int(math.ceil(cy + half_h))))
    crop = depth[top:bottom, left:right]
    valid = crop[np.isfinite(crop) & (crop > 0.05)]
    return float(np.median(valid)) if valid.size else None


def stop_box_quality(
    bbox: list[float],
    image_shape: tuple[int, ...],
    edge_margin_px: float = 2.0,
    max_area_fraction: float = 0.5,
) -> tuple[bool, dict]:
    """Reject broad or clipped boxes that are unsafe for terminal STOP."""
    height, width = image_shape[:2]
    x1, y1, x2, y2 = (float(value) for value in bbox)
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    area_fraction = box_width * box_height / max(1.0, float(width * height))
    fully_contained = (
        x1 >= edge_margin_px
        and y1 >= edge_margin_px
        and x2 <= width - edge_margin_px
        and y2 <= height - edge_margin_px
    )
    valid_extent = box_width >= 2.0 and box_height >= 2.0
    accepted = (
        fully_contained
        and valid_extent
        and area_fraction <= max_area_fraction
    )
    return accepted, {
        "fully_contained": fully_contained,
        "valid_extent": valid_extent,
        "area_fraction": round(area_fraction, 4),
        "max_area_fraction": max_area_fraction,
    }


def detection_world_xy(
    bbox: list[float],
    depth_m: float | None,
    image_shape: tuple[int, ...],
    pose: dict,
    hfov_rad: float = math.pi / 2.0,
) -> tuple[float, float] | None:
    """Back-project a detection-box centre to a horizontal world point."""
    if depth_m is None or not math.isfinite(depth_m) or depth_m <= 0.05:
        return None
    _, width = image_shape[:2]
    if width <= 0 or len(bbox) != 4:
        return None
    x1, _, x2, _ = (float(value) for value in bbox)
    px_norm = ((x1 + x2) / 2.0) / float(width)
    u_norm = 2.0 * px_norm - 1.0
    # Isaac's camera-right direction is negative yaw in the horizontal plane.
    bearing = float(pose["yaw"]) - u_norm * hfov_rad / 2.0
    return (
        float(pose["position"][0]) + depth_m * math.cos(bearing),
        float(pose["position"][1]) + depth_m * math.sin(bearing),
    )


def circular_yaw_spread(yaws: list[float]) -> float:
    """Maximum pairwise circular yaw separation in radians."""
    spread = 0.0
    for index, first in enumerate(yaws):
        for second in yaws[index + 1:]:
            delta = abs(float(first) - float(second)) % (2.0 * math.pi)
            spread = max(spread, min(delta, 2.0 * math.pi - delta))
    return spread


def admit_target_evidence(
    memory: list[dict],
    xy: tuple[float, float],
    pose: dict,
    score: float,
    step: int,
    cluster_radius_m: float = 0.9,
) -> dict:
    """Fuse one detector observation into a persistent 2-D target cluster."""
    nearest = min(
        memory,
        key=lambda cluster: math.hypot(
            float(cluster["xy"][0]) - xy[0],
            float(cluster["xy"][1]) - xy[1],
        ),
        default=None,
    )
    if nearest is None or math.hypot(
        float(nearest["xy"][0]) - xy[0],
        float(nearest["xy"][1]) - xy[1],
    ) > cluster_radius_m:
        nearest = {
            "xy": [float(xy[0]), float(xy[1])],
            "observations": 0,
            "best_score": 0.0,
            "last_seen_step": step,
            "misses": 0,
            "projected_points": [],
            "viewpoints": [],
            "view_yaws": [],
        }
        memory.append(nearest)

    nearest["projected_points"].append([float(xy[0]), float(xy[1])])
    nearest["viewpoints"].append([
        float(pose["position"][0]),
        float(pose["position"][1]),
    ])
    nearest["view_yaws"].append(float(pose["yaw"]))
    nearest["observations"] = len(nearest["projected_points"])
    nearest["xy"] = [
        float(np.mean([point[0] for point in nearest["projected_points"]])),
        float(np.mean([point[1] for point in nearest["projected_points"]])),
    ]
    nearest["best_score"] = max(float(nearest["best_score"]), float(score))
    nearest["last_seen_step"] = step
    nearest["misses"] = 0
    return nearest


def evidence_confirmation(cluster: dict) -> dict:
    """Summarize whether evidence comes from two meaningfully different views."""
    viewpoints = cluster.get("viewpoints") or []
    baseline_m = 0.0
    for index, first in enumerate(viewpoints):
        for second in viewpoints[index + 1:]:
            baseline_m = max(
                baseline_m,
                math.hypot(
                    float(first[0]) - float(second[0]),
                    float(first[1]) - float(second[1]),
                ),
            )
    yaw_spread = circular_yaw_spread(cluster.get("view_yaws") or [])
    observations = int(cluster.get("observations", 0))
    confirmed = observations >= 2 and (
        baseline_m >= 0.20 or yaw_spread >= math.radians(5.0)
    )
    return {
        "confirmed": confirmed,
        "observations": observations,
        "viewpoint_baseline_m": round(baseline_m, 3),
        "yaw_spread_deg": round(math.degrees(yaw_spread), 2),
    }


def _path_step(
    wm: WalkableMap,
    current_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    max_step_m: float,
) -> tuple[float, float] | None:
    """Select the furthest in-budget point along a walkable A* path."""
    path = wm.shortest_path_2d(current_xy, goal_xy)
    if not path:
        return None
    travelled = 0.0
    previous = path[0]
    chosen = None
    for point in path[1:]:
        segment = math.hypot(point[0] - previous[0], point[1] - previous[1])
        if travelled + segment > max_step_m + 1e-6:
            break
        travelled += segment
        previous = point
        chosen = point
    if chosen is None or math.hypot(
        chosen[0] - current_xy[0], chosen[1] - current_xy[1]
    ) < 0.10:
        return None
    return float(chosen[0]), float(chosen[1])


def choose_evidence_waypoint(
    wm: WalkableMap,
    current_xy: tuple[float, float],
    target_xy: tuple[float, float],
    step: int,
    max_step_m: float = 1.7,
    standoff_m: float = 0.9,
) -> tuple[float, float] | None:
    """Choose an approach or short orbit waypoint around visual evidence."""
    dx = target_xy[0] - current_xy[0]
    dy = target_xy[1] - current_xy[1]
    distance = math.hypot(dx, dy)
    if distance <= 1e-6:
        return None
    unit_x, unit_y = dx / distance, dy / distance

    if distance > standoff_m + 0.20:
        raw_goal = (
            target_xy[0] - standoff_m * unit_x,
            target_xy[1] - standoff_m * unit_y,
        )
        goal = raw_goal if wm.is_walkable(*raw_goal) else wm.nearby_walkable(
            *raw_goal, radius_m=0.8
        )
        if goal is not None:
            waypoint = _path_step(wm, current_xy, goal, max_step_m)
            if waypoint is not None:
                return waypoint

    # Already close, or the direct standoff is blocked: take a modest orbit
    # step to create parallax for verification instead of repeatedly querying
    # the exact same view.
    base_angle = math.atan2(current_xy[1] - target_xy[1],
                            current_xy[0] - target_xy[0])
    radius = max(0.65, min(max(distance, standoff_m), 1.25))
    signs = (1.0, -1.0) if step % 2 else (-1.0, 1.0)
    for sign in signs:
        angle = base_angle + sign * math.radians(25.0)
        candidate = (
            target_xy[0] + radius * math.cos(angle),
            target_xy[1] + radius * math.sin(angle),
        )
        if not wm.is_walkable(*candidate):
            nearby = wm.nearby_walkable(*candidate, radius_m=0.45)
            if nearby is None:
                continue
            candidate = nearby
        displacement = math.hypot(
            candidate[0] - current_xy[0], candidate[1] - current_xy[1]
        )
        if 0.20 <= displacement <= max_step_m and wm.line_of_sight(
            current_xy[0], current_xy[1], candidate[0], candidate[1]
        ):
            return float(candidate[0]), float(candidate[1])
    return None


def score_waypoint(
    waypoint: tuple[float, float, float],
    wm: WalkableMap,
    history: list[tuple[float, float]],
    current_xy: tuple[float, float],
) -> float:
    """Geometry-only frontier score with deterministic revisit avoidance."""
    x, y, angle_offset = waypoint
    explored_density = wm.explored_density_at(x, y, radius_m=0.8)
    novelty = 1.0 - explored_density
    history_clearance = min(
        (math.hypot(x - hx, y - hy) for hx, hy in history),
        default=2.0,
    )
    history_clearance = min(history_clearance, 2.0) / 2.0
    travel = min(math.hypot(x - current_xy[0], y - current_xy[1]), 1.7) / 1.7
    turn_cost = min(abs(float(angle_offset)), math.pi) / math.pi
    return novelty + 0.25 * history_clearance + 0.10 * travel - 0.05 * turn_cost


def run_episode(
    env,
    wm: WalkableMap,
    item: dict,
    episode: dict,
    style: str,
    step_cap: int,
    num_waypoints: int,
    detector_threshold: float,
    stop_score_threshold: float,
    stop_depth_m: float,
    detection_vocabulary: list[str],
    target_queries: list[str],
    active_perception: bool,
    model_name: str,
    seed: int,
    out_path: Path,
) -> dict:
    target_category = str(item["target_category"])
    target_terms = {
        normalize_detector_label(query)
        for query in (target_queries if active_perception else [target_category])
    }
    detector_queries = target_queries if active_perception else detection_vocabulary
    env.place_agent(
        episode["start_position"],
        episode["start_rotation_quat_wxyz"],
    )
    wm.reset_explored()

    pose = env.get_pose()
    current_xy = (float(pose["position"][0]), float(pose["position"][1]))
    wm.mark_explored(*current_xy, radius_m=0.8)
    history = [current_xy]
    start_room = wm.room_at(*current_xy)
    rooms_visited = [start_room] if start_room else []
    trajectory = [{
        "step": 0,
        "action": "START",
        "position": pose["position"],
        "yaw": pose["yaw"],
        "room": start_room,
    }]
    stop_reason = "step_cap"
    final_rgb = None
    detector_calls = 0
    target_memory: list[dict] = []
    active_moves = 0
    verification_looks = 0

    for step in range(1, step_cap + 1):
        rgb = env.render_rgb()
        depth = env.render_depth()
        final_rgb = rgb
        detections = dino_detector.detect(
            rgb,
            detector_queries,
            threshold=detector_threshold,
            text_threshold=0.20,
        )
        detector_calls += 1
        target_detections = [
            detection for detection in detections
            if normalize_detector_label(detection[0]) in target_terms
        ]
        best = target_detections[0] if target_detections else None
        competitors = [
            {
                "label": label,
                "score": round(float(score), 4),
            }
            for label, score, _ in detections[:3]
            if normalize_detector_label(label) not in target_terms
        ]
        detection_meta = None
        current_cluster = None
        if best is not None:
            label, score, bbox = best
            depth_m = bbox_depth_m(depth, bbox)
            box_accepted, box_quality = stop_box_quality(bbox, rgb.shape)
            pose = env.get_pose()
            projected_xy = (
                detection_world_xy(bbox, depth_m, rgb.shape, pose)
                if active_perception else None
            )
            if projected_xy is not None:
                current_cluster = admit_target_evidence(
                    target_memory,
                    projected_xy,
                    pose,
                    float(score),
                    step,
                )
            confirmation = (
                evidence_confirmation(current_cluster)
                if current_cluster is not None
                else {
                    "confirmed": not active_perception,
                    "observations": 0,
                    "viewpoint_baseline_m": 0.0,
                    "yaw_spread_deg": 0.0,
                }
            )
            stop_gate = {
                "score_ok": float(score) >= stop_score_threshold,
                "depth_ok": depth_m is not None and depth_m <= stop_depth_m,
                "box_ok": box_accepted,
                "multiview_ok": bool(confirmation["confirmed"]),
                "box_quality": box_quality,
            }
            detection_meta = {
                "label": label,
                "score": round(float(score), 4),
                "bbox": [round(float(value), 2) for value in bbox],
                "depth_m": round(depth_m, 3) if depth_m is not None else None,
                "projected_xy": (
                    [round(projected_xy[0], 3), round(projected_xy[1], 3)]
                    if projected_xy is not None else None
                ),
                "confirmation": confirmation,
                "stop_gate": stop_gate,
            }
            if all((
                stop_gate["score_ok"],
                stop_gate["depth_ok"],
                stop_gate["box_ok"],
                stop_gate["multiview_ok"],
            )):
                pose = env.get_pose()
                trajectory.append({
                    "step": step,
                    "action": "STOP",
                    "position": pose["position"],
                    "yaw": pose["yaw"],
                    "room": wm.room_at(
                        pose["position"][0], pose["position"][1]
                    ),
                    "dino": detection_meta,
                    "dino_competitors": competitors,
                })
                stop_reason = (
                    "dino_multiview_target_within_depth"
                    if active_perception else "dino_target_within_depth"
                )
                break

        # VLFM/APRR-style phase switch: once visual evidence exists, spend
        # the next action approaching or re-centering it instead of throwing
        # the observation away and returning to an unrelated frontier.
        if active_perception and current_cluster is not None:
            pose = env.get_pose()
            current_xy = (
                float(pose["position"][0]),
                float(pose["position"][1]),
            )
            evidence_xy = (
                float(current_cluster["xy"][0]),
                float(current_cluster["xy"][1]),
            )
            waypoint = choose_evidence_waypoint(
                wm, current_xy, evidence_xy, step
            )
            if waypoint is not None:
                env.teleport_to(waypoint, face_direction_xy=evidence_xy)
                action = "MOVE_TO_EVIDENCE"
                active_moves += 1
            else:
                yaw = math.atan2(
                    evidence_xy[1] - current_xy[1],
                    evidence_xy[0] - current_xy[0],
                )
                env.look_at_yaw(yaw)
                action = "LOOK_AT_EVIDENCE"
                verification_looks += 1

            pose = env.get_pose()
            current_xy = (
                float(pose["position"][0]),
                float(pose["position"][1]),
            )
            history.append(current_xy)
            wm.mark_explored(*current_xy, radius_m=0.8)
            room = wm.room_at(*current_xy)
            if room and room not in rooms_visited:
                rooms_visited.append(room)
            trajectory.append({
                "step": step,
                "action": action,
                "position": pose["position"],
                "yaw": pose["yaw"],
                "waypoint": list(waypoint) if waypoint is not None else None,
                "evidence_xy": list(evidence_xy),
                "room": room,
                "dino": detection_meta,
                "dino_competitors": competitors,
            })
            continue

        # A brief ±12° scan gives a recently observed candidate two chances
        # to reappear.  Persistent misses expire naturally after two steps.
        if active_perception and target_memory:
            recent = max(
                target_memory,
                key=lambda cluster: (
                    int(cluster.get("last_seen_step", -1)),
                    float(cluster.get("best_score", 0.0)),
                ),
            )
            if step - int(recent.get("last_seen_step", -99)) <= 2:
                recent["misses"] = int(recent.get("misses", 0)) + 1
                pose = env.get_pose()
                current_xy = (
                    float(pose["position"][0]),
                    float(pose["position"][1]),
                )
                evidence_xy = (
                    float(recent["xy"][0]), float(recent["xy"][1])
                )
                base_yaw = math.atan2(
                    evidence_xy[1] - current_xy[1],
                    evidence_xy[0] - current_xy[0],
                )
                sweep = math.radians(12.0) * (
                    -1.0 if int(recent["misses"]) % 2 else 1.0
                )
                env.look_at_yaw(base_yaw + sweep)
                verification_looks += 1
                pose = env.get_pose()
                room = wm.room_at(*current_xy)
                trajectory.append({
                    "step": step,
                    "action": "REACQUIRE_EVIDENCE",
                    "position": pose["position"],
                    "yaw": pose["yaw"],
                    "evidence_xy": list(evidence_xy),
                    "evidence_misses": int(recent["misses"]),
                    "room": room,
                    "dino": None,
                    "dino_competitors": competitors,
                })
                continue

        waypoints = env.sample_frontier_waypoints(
            wm,
            K=num_waypoints,
            seed=seed + step,
            force_angular_spread_rad=2.0 * math.pi,
        )
        if not waypoints:
            stop_reason = "no_waypoints"
            break

        pose = env.get_pose()
        current_xy = (
            float(pose["position"][0]),
            float(pose["position"][1]),
        )
        scores = [
            score_waypoint(waypoint, wm, history, current_xy)
            for waypoint in waypoints
        ]
        best_index = max(range(len(waypoints)), key=lambda idx: (scores[idx], -idx))
        waypoint = waypoints[best_index]
        env.teleport_to((waypoint[0], waypoint[1]))
        pose = env.get_pose()
        current_xy = (
            float(pose["position"][0]),
            float(pose["position"][1]),
        )
        history.append(current_xy)
        wm.mark_explored(*current_xy, radius_m=0.8)
        room = wm.room_at(*current_xy)
        if room and room not in rooms_visited:
            rooms_visited.append(room)
        trajectory.append({
            "step": step,
            "action": "MOVE",
            "position": pose["position"],
            "yaw": pose["yaw"],
            "waypoint": [float(waypoint[0]), float(waypoint[1])],
            "frontier_score": round(float(scores[best_index]), 4),
            "n_waypoints": len(waypoints),
            "room": room,
            "dino": detection_meta,
            "dino_competitors": competitors,
        })

    if not trajectory or trajectory[-1].get("action") != "STOP":
        final_rgb = env.render_rgb()
    if final_rgb is None:
        final_rgb = env.render_rgb()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_frame_path = out_path.parent / "final.png"
    env.save_frame(final_rgb, final_frame_path)

    record = {
        "selection_id": item["selection_id"],
        "tier": "fpe",
        "model": model_name,
        "style": style,
        "scene_id": item["scene_id"],
        "target_category": target_category,
        "intent": intent_for_style(item, style),
        "photo": item.get("photo"),
        "final_frame": str(final_frame_path.relative_to(REPO)),
        "prediction": {"target": target_category},
        "trajectory": trajectory,
        "rooms_visited": rooms_visited,
        "stop_reason": stop_reason,
        "step_cap": step_cap,
        "episode_meta": episode,
        "evaluation_protocol": {
            "version": (
                ACTIVE_PROTOCOL_VERSION if active_perception else PROTOCOL_VERSION
            ),
            "goal_input": "explicit_target",
            "metric_scope": "navigation_only",
            "stop_is_terminal": True,
            "post_budget_actions": False,
            "max_step_m": 1.7,
            "privileged_target_position": False,
        },
        "run_config": {
            "command": [sys.executable, *sys.argv],
            "seed": seed,
            "step_cap": step_cap,
            "num_waypoints": num_waypoints,
            "detector_threshold": detector_threshold,
            "stop_score_threshold": stop_score_threshold,
            "stop_depth_m": stop_depth_m,
            "active_perception": active_perception,
            "target_queries": target_queries,
        },
        "model_meta": {
            "provider": "local",
            "model": "IDEA-Research/grounding-dino-tiny",
            "api_calls": 0,
            "detector_calls": detector_calls,
            "detector_query_mode": (
                "target_with_collision_safe_aliases"
                if active_perception else "full_benchmark_vocabulary"
            ),
            "detector_vocabulary_size": len(detector_queries),
            "detector_queries": detector_queries,
            "detector_threshold": detector_threshold,
            "stop_score_threshold": stop_score_threshold,
            "stop_depth_m": stop_depth_m,
            "navigation_policy": (
                "frontier_then_active_target_approach"
                if active_perception else "frontier_pure_exploration"
            ),
            "active_target_moves": active_moves,
            "verification_looks": verification_looks,
        },
        "target_memory": target_memory,
        "timestamp": now_iso(),
    }
    save_atomic(record, out_path)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--style",
        default="formal",
        choices=[*STYLES, "all"],
    )
    parser.add_argument("--step-cap", type=int, default=30)
    parser.add_argument("--num-waypoints", type=int, default=8)
    parser.add_argument("--detector-threshold", type=float, default=0.30)
    parser.add_argument("--stop-score-threshold", type=float, default=0.50)
    parser.add_argument("--stop-depth-m", type=float, default=2.0)
    parser.add_argument(
        "--legacy-full-vocabulary",
        action="store_true",
        help=(
            "Reproduce the original full-vocabulary pure-frontier baseline; "
            "the default is target-focused active last-mile navigation."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Print '<completed> <expected>' for this scene shard and exit.",
    )
    parser.add_argument("--headless", action="store_true", default=True)
    args = parser.parse_args()
    if args.stop_score_threshold < args.detector_threshold:
        parser.error("--stop-score-threshold must be >= --detector-threshold")
    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")

    items = load_items()
    active_perception = not args.legacy_full_vocabulary
    model_name = ACTIVE_MODEL_NAME if active_perception else LEGACY_MODEL_NAME
    detection_vocabulary = sorted({
        str(item["target_category"]).replace("_", " ")
        for item in items
    })
    synonym_map = load_detector_synonyms()
    canonical_scenes = sorted({item["scene_id"] for item in items})
    scene_shard = {
        scene_id: index % args.num_shards
        for index, scene_id in enumerate(canonical_scenes)
    }
    if args.only:
        wanted = {value.strip() for value in args.only.split(",") if value.strip()}
        items = [item for item in items if item["selection_id"] in wanted]
    if args.limit is not None:
        items = items[:args.limit]
    items = [
        item for item in items
        if scene_shard[item["scene_id"]] == args.shard_index
    ]
    items.sort(key=lambda item: (item["scene_id"], item["selection_id"]))

    styles = STYLES if args.style == "all" else (args.style,)
    if args.count_only:
        expected = len(items) * len(styles)
        completed = 0
        for item in items:
            for style in styles:
                path = episode_path(
                    item["scene_id"],
                    "fpe",
                    model_name,
                    style,
                    item["selection_id"],
                )
                completed += int(path.exists())
        print(f"{completed} {expected}")
        return

    # Isaac Sim consumes argv during SimulationApp construction.
    sys.argv = sys.argv[:1]
    from simulator.iss_env import IsaacSimEnv

    episodes = load_episodes()
    env = IsaacSimEnv(headless=args.headless)
    wm_cache: dict[str, WalkableMap | None] = {}
    loaded_scenes: set[str] = set()
    completed = skipped = failed = 0

    try:
        for item in items:
            selection_id = item["selection_id"]
            episode = episodes.get(selection_id)
            if episode is None:
                failed += 1
                continue
            scene_id = episode["scene_id"]
            pending = []
            for style in styles:
                out_path = episode_path(
                    scene_id, "fpe", model_name, style, selection_id
                )
                if out_path.exists() and not args.force:
                    skipped += 1
                else:
                    pending.append((style, out_path))
            if not pending:
                continue

            if (
                args.max_scenes is not None
                and scene_id not in loaded_scenes
                and len(loaded_scenes) >= args.max_scenes
            ):
                print(
                    f"[fpe] recycle requested after {len(loaded_scenes)} scenes"
                )
                break

            if scene_id not in wm_cache:
                wm_cache[scene_id] = WalkableMap.load(scene_id)
            wm = wm_cache[scene_id]
            if wm is None:
                failed += len(pending)
                continue

            usd_path = USD_ROOT / scene_id / "start_result_navigation.usd"
            env.load_scene(scene_id, str(usd_path))
            loaded_scenes.add(scene_id)
            numeric_id = int("".join(ch for ch in selection_id if ch.isdigit()) or "0")
            target_queries = build_target_queries(
                str(item["target_category"]),
                synonym_map,
                detection_vocabulary,
            )

            for style, out_path in pending:
                try:
                    run_episode(
                        env,
                        wm,
                        item,
                        episode,
                        style,
                        step_cap=args.step_cap,
                        num_waypoints=args.num_waypoints,
                        detector_threshold=args.detector_threshold,
                        stop_score_threshold=args.stop_score_threshold,
                        stop_depth_m=args.stop_depth_m,
                        detection_vocabulary=detection_vocabulary,
                        target_queries=target_queries,
                        active_perception=active_perception,
                        model_name=model_name,
                        seed=args.seed + numeric_id * 1000,
                        out_path=out_path,
                    )
                    completed += 1
                except Exception as exc:
                    import traceback

                    failed += 1
                    print(f"[fpe] {selection_id}/{style}: {exc}", file=sys.stderr)
                    traceback.print_exc()

            if (completed + skipped) % 10 == 0:
                print(
                    f"[fpe] completed={completed} skipped={skipped} "
                    f"failed={failed} scenes={len(loaded_scenes)}"
                )
    finally:
        env.close()

    print(
        f"[fpe] completed={completed} skipped={skipped} failed={failed} "
        f"scenes={len(loaded_scenes)}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
