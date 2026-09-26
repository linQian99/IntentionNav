"""Pose-anchored multi-view target re-observation.

The first target-conditioned detector proposal remains anchored to the camera
pose and image ray that produced it.  It may request one lateral, reachable
view, but its noisy depth projection is never used as a navigation goal.
Only a later observation accepted by the ordinary detector/verifier path can
be fused into target memory.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable


def _circular_distance(a: float, b: float) -> float:
    delta = abs(float(a) - float(b)) % (2.0 * math.pi)
    return min(delta, 2.0 * math.pi - delta)


def _cross_2d(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(a[0]) * float(b[1]) - float(a[1]) * float(b[0])


def observation_ray_yaw(
    observation: dict,
    *,
    image_width: int = 768,
    horizontal_fov_rad: float = math.pi / 2.0,
) -> float | None:
    """Return the world yaw of a detector box centre.

    The sign and linear image-angle convention deliberately match the engine's
    existing RGB-D back-projection so an R042 arm changes association and view
    selection, not the camera model.
    """
    bbox = observation.get("bbox")
    yaw = observation.get("yaw")
    if not bbox or len(bbox) != 4 or yaw is None or image_width <= 0:
        return None
    try:
        centre_x = 0.5 * (float(bbox[0]) + float(bbox[2]))
        u_norm = (centre_x / float(image_width)) * 2.0 - 1.0
        if not math.isfinite(u_norm):
            return None
        return float(yaw) - u_norm * float(horizontal_fov_rad) / 2.0
    except (TypeError, ValueError):
        return None


def latest_pose_observation(cluster: dict | None) -> dict | None:
    observations = (cluster or {}).get("observations") or []
    for observation in reversed(observations):
        if observation.get("observer_xy") and observation.get("bbox"):
            return observation
    return None


def _semantic_shortlist_rank(observation: dict) -> int:
    verification = (
        observation.get("semantic_verification")
        or observation.get("clip_verification")
        or {}
    )
    try:
        return int(verification.get("target_rank", 10_000))
    except (TypeError, ValueError):
        return 10_000


def best_pose_graph_candidate(
    memory: Iterable[dict],
    *,
    step: int,
    current_xy: tuple[float, float],
    action_budget_remaining: int,
    max_age_steps: int = 0,
    max_attempts_per_cluster: int = 1,
    max_semantic_rank: int = 5,
    anchor_tolerance_m: float = 0.25,
    observer_allowed: Callable[[float, float], bool] | None = None,
) -> dict | None:
    """Select a fresh proposal at the current known-reachable camera anchor."""
    if int(action_budget_remaining) <= 0:
        return None
    candidates = []
    for cluster in memory:
        if bool(cluster.get("promoted", False)):
            continue
        if int(cluster.get("verification_attempts", 0)) >= int(
            max_attempts_per_cluster
        ):
            continue
        if int(step) - int(cluster.get("step", -10_000)) > int(max_age_steps):
            continue
        observation = latest_pose_observation(cluster)
        if observation is None:
            continue
        observer = observation.get("observer_xy")
        if not observer or len(observer) < 2:
            continue
        ox, oy = float(observer[0]), float(observer[1])
        if math.hypot(ox - current_xy[0], oy - current_xy[1]) > float(
            anchor_tolerance_m
        ):
            continue
        rank = _semantic_shortlist_rank(observation)
        if not 1 <= rank <= int(max_semantic_rank):
            continue
        if observer_allowed is not None and not observer_allowed(ox, oy):
            continue
        if observation_ray_yaw(observation) is None:
            continue
        candidates.append((rank, cluster, observation))
    if not candidates:
        return None
    _, cluster, _ = min(
        candidates,
        key=lambda item: (
            item[0],
            -float(item[1].get("detector_score_max", 0.0)),
            -int(item[1].get("step", 0)),
        ),
    )
    return cluster


def _path_limited_waypoint(
    path: list[tuple[float, float]],
    current_xy: tuple[float, float],
    max_move_m: float,
) -> tuple[float, float] | None:
    if not path:
        return None
    eligible = [
        (float(point[0]), float(point[1]))
        for point in path
        if math.hypot(
            float(point[0]) - current_xy[0],
            float(point[1]) - current_xy[1],
        ) <= float(max_move_m) + 1e-6
    ]
    return eligible[-1] if eligible else None


def pose_anchored_reobservation_waypoint(
    wm,
    cluster: dict,
    *,
    current_xy: tuple[float, float],
    lateral_offset_m: float = 0.8,
    min_baseline_m: float = 0.4,
    max_move_m: float = 1.0,
    min_predicted_crossing_rad: float = math.radians(8.0),
    image_width: int = 768,
    horizontal_fov_rad: float = math.pi / 2.0,
) -> tuple[tuple[float, float] | None, dict]:
    """Choose a reachable lateral view while retaining the source image ray."""
    observation = latest_pose_observation(cluster)
    if observation is None:
        return None, {"mode": "pose_graph_reobserve", "failure": "no_anchor"}
    observer = observation.get("observer_xy")
    ray_yaw = observation_ray_yaw(
        observation,
        image_width=image_width,
        horizontal_fov_rad=horizontal_fov_rad,
    )
    if not observer or ray_yaw is None:
        return None, {"mode": "pose_graph_reobserve", "failure": "bad_anchor"}
    source_xy = float(observer[0]), float(observer[1])
    if math.hypot(
        source_xy[0] - current_xy[0], source_xy[1] - current_xy[1]
    ) > 0.25:
        return None, {"mode": "pose_graph_reobserve", "failure": "stale_anchor"}

    depth = observation.get("depth_m")
    if depth is None and observation.get("xy"):
        depth = math.hypot(
            float(observation["xy"][0]) - source_xy[0],
            float(observation["xy"][1]) - source_xy[1],
        )
    try:
        look_range = min(8.0, max(1.5, float(depth)))
    except (TypeError, ValueError):
        look_range = 4.0
    ray_direction = math.cos(ray_yaw), math.sin(ray_yaw)
    face_xy = (
        source_xy[0] + look_range * ray_direction[0],
        source_xy[1] + look_range * ray_direction[1],
    )

    candidates = []
    for distance in (
        float(lateral_offset_m),
        max(float(min_baseline_m), 0.75 * float(lateral_offset_m)),
        float(min_baseline_m),
    ):
        for side in (-1.0, 1.0):
            lateral_yaw = ray_yaw + side * math.pi / 2.0
            desired = (
                source_xy[0] + distance * math.cos(lateral_yaw),
                source_xy[1] + distance * math.sin(lateral_yaw),
            )
            try:
                if not wm.is_walkable(*desired):
                    desired = wm.nearby_walkable(
                        *desired, radius_m=min(0.3, 0.5 * distance)
                    )
                if desired is None:
                    continue
                path = wm.shortest_path_2d(
                    current_xy, desired, snap_radius_m=0.3
                )
            except Exception:
                continue
            waypoint = _path_limited_waypoint(path or [], current_xy, max_move_m)
            if waypoint is None:
                continue
            baseline = math.hypot(
                waypoint[0] - source_xy[0], waypoint[1] - source_xy[1]
            )
            displacement = math.hypot(
                waypoint[0] - current_xy[0], waypoint[1] - current_xy[1]
            )
            if baseline + 1e-6 < float(min_baseline_m) or displacement <= 0.1:
                continue
            new_yaw = math.atan2(
                face_xy[1] - waypoint[1], face_xy[0] - waypoint[0]
            )
            crossing = _circular_distance(ray_yaw, new_yaw)
            if crossing + 1e-6 < float(min_predicted_crossing_rad):
                continue
            candidates.append((
                -crossing,
                displacement,
                waypoint,
                baseline,
                new_yaw,
                crossing,
            ))
    if not candidates:
        return None, {
            "mode": "pose_graph_reobserve",
            "failure": "no_reachable_lateral_view",
            "source_anchor_xy": list(source_xy),
            "source_ray_yaw": float(ray_yaw),
        }
    _, displacement, waypoint, baseline, new_yaw, crossing = min(candidates)
    return waypoint, {
        "mode": "pose_graph_reobserve",
        "source_anchor_xy": [float(source_xy[0]), float(source_xy[1])],
        "source_ray_yaw": float(ray_yaw),
        "source_observation_step": int(observation.get("step", -1)),
        "face_direction_xy": [float(face_xy[0]), float(face_xy[1])],
        "waypoint_displacement_m": round(float(displacement), 6),
        "view_baseline_m": round(float(baseline), 6),
        "predicted_view_yaw": float(new_yaw),
        "predicted_crossing_angle_rad": round(float(crossing), 6),
        "uncertain_world_xy_used_as_goal": False,
    }


def triangulate_pose_observations(
    first: dict,
    second: dict,
    *,
    min_baseline_m: float = 0.4,
    min_crossing_angle_rad: float = math.radians(8.0),
    max_crossing_angle_rad: float = math.pi / 2.0,
    max_range_m: float = 12.0,
    max_accepted_projection_residual_m: float = 1.5,
    image_width: int = 768,
    horizontal_fov_rad: float = math.pi / 2.0,
) -> dict:
    """Triangulate two forward image rays and return a fail-closed ledger."""
    first_observer = first.get("observer_xy")
    second_observer = second.get("observer_xy")
    first_yaw = observation_ray_yaw(
        first,
        image_width=image_width,
        horizontal_fov_rad=horizontal_fov_rad,
    )
    second_yaw = observation_ray_yaw(
        second,
        image_width=image_width,
        horizontal_fov_rad=horizontal_fov_rad,
    )
    result = {
        "valid": False,
        "baseline_m": None,
        "crossing_angle_rad": None,
        "intersection_xy": None,
        "first_forward_range_m": None,
        "second_forward_range_m": None,
        "accepted_projection_residual_m": None,
    }
    if (
        not first_observer or not second_observer
        or first_yaw is None or second_yaw is None
    ):
        result["failure"] = "missing_ray"
        return result
    p = float(first_observer[0]), float(first_observer[1])
    q = float(second_observer[0]), float(second_observer[1])
    baseline = math.hypot(q[0] - p[0], q[1] - p[1])
    crossing = _circular_distance(first_yaw, second_yaw)
    result["baseline_m"] = round(float(baseline), 6)
    result["crossing_angle_rad"] = round(float(crossing), 6)
    if baseline + 1e-6 < float(min_baseline_m):
        result["failure"] = "insufficient_baseline"
        return result
    if (
        crossing + 1e-6 < float(min_crossing_angle_rad)
        or crossing > float(max_crossing_angle_rad) + 1e-6
    ):
        result["failure"] = "degenerate_crossing"
        return result
    d1 = math.cos(first_yaw), math.sin(first_yaw)
    d2 = math.cos(second_yaw), math.sin(second_yaw)
    denominator = _cross_2d(d1, d2)
    if abs(denominator) < 1e-6:
        result["failure"] = "parallel_rays"
        return result
    offset = q[0] - p[0], q[1] - p[1]
    first_range = _cross_2d(offset, d2) / denominator
    second_range = _cross_2d(offset, d1) / denominator
    result["first_forward_range_m"] = round(float(first_range), 6)
    result["second_forward_range_m"] = round(float(second_range), 6)
    if not (
        0.25 <= first_range <= float(max_range_m)
        and 0.25 <= second_range <= float(max_range_m)
    ):
        result["failure"] = "intersection_outside_forward_range"
        return result
    intersection = (
        p[0] + first_range * d1[0],
        p[1] + first_range * d1[1],
    )
    if not all(math.isfinite(value) for value in intersection):
        result["failure"] = "nonfinite_intersection"
        return result
    result["intersection_xy"] = [
        round(float(intersection[0]), 6),
        round(float(intersection[1]), 6),
    ]
    accepted_xy = second.get("xy")
    if accepted_xy and len(accepted_xy) >= 2:
        residual = math.hypot(
            intersection[0] - float(accepted_xy[0]),
            intersection[1] - float(accepted_xy[1]),
        )
        result["accepted_projection_residual_m"] = round(float(residual), 6)
        if residual > float(max_accepted_projection_residual_m):
            result["failure"] = "accepted_projection_inconsistent"
            return result
    else:
        result["failure"] = "missing_accepted_projection"
        return result
    result["valid"] = True
    return result


def pose_graph_support_observation(
    proposal_observation: dict,
    triangulation: dict,
    *,
    step: int,
) -> dict:
    """Create a non-terminal support observation for an accepted cluster."""
    if not triangulation.get("valid") or not triangulation.get(
        "intersection_xy"
    ):
        raise ValueError("pose graph support requires valid triangulation")
    support = dict(proposal_observation)
    support.update({
        "xy": [float(value) for value in triangulation["intersection_xy"]],
        "step": int(step),
        "source": "pose_graph_support",
        "pose_graph_support": True,
        "pose_graph_triangulation": dict(triangulation),
    })
    return support
