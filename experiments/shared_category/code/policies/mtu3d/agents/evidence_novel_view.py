"""Buy up to two physically novel observations of one accepted target cluster.

The first action follows a missed close approach. A second action is possible
only immediately after the first one reacquires the same cluster. Planning
uses policy memory and the walkable map; ordinary perception and STOP retain
sole responsibility for admitting evidence and confirming the target.
"""
from __future__ import annotations

import math
from typing import Any


def _novel_waypoint(wm: Any, current: tuple, target: tuple, history: list) -> tuple:
    """Deterministic local search; no fallback to an old or rotation-only view."""
    candidates = []
    counts = {"sampled": 0, "novel": 0, "walkable": 0, "visible": 0}
    for distance in (0.35, 0.50, 0.65, 0.75):
        for index in range(32):
            angle = index * math.tau / 32
            point = (current[0] + distance * math.cos(angle),
                     current[1] + distance * math.sin(angle))
            counts["sampled"] += 1
            radius = math.dist(point, target)
            baseline = min(math.dist(point, old) for old in history)
            bearing = math.atan2(point[1] - target[1], point[0] - target[0])
            novelty = min(abs(math.atan2(math.sin(bearing - math.atan2(old[1] - target[1], old[0] - target[0])),
                                         math.cos(bearing - math.atan2(old[1] - target[1], old[0] - target[0]))))
                          for old in history)
            if not (0.75 <= radius <= 2.0 and baseline >= 0.35
                    and novelty >= math.radians(15)):
                continue
            counts["novel"] += 1
            samples = math.ceil(distance / 0.025)
            if not all(wm.is_walkable(current[0] + (point[0] - current[0]) * i / samples,
                                      current[1] + (point[1] - current[1]) * i / samples)
                       for i in range(samples + 1)):
                continue
            counts["walkable"] += 1
            if not wm.line_of_sight(*point, *target):
                continue
            counts["visible"] += 1
            cost = (abs(radius - 1.2) + 0.5 * distance
                    - 0.4 * min(novelty / math.radians(30), 1.0))
            candidates.append((cost, point, baseline, novelty))
    return (min(candidates) if candidates else None), counts


def evidence_view_recovery_action(
    *, memory: list[dict], current_xy: tuple[float, float],
    current_evidence: dict | None, previous_action: dict,
    step: int, step_cap: int, actions_used: int,
    used_clusters: set[str], walkable_map: Any,
    diagnostics: dict | None = None,
) -> tuple[dict | None, str]:
    """Return a safe novel pose without modifying memory or confirmation state."""
    uid = previous_action.get("selected_navigation_cluster_uid")
    continuation = bool(actions_used == 1 and uid in used_clusters
                        and used_clusters == {uid}
                        and (previous_action.get("nav_meta") or {}).get("protocol")
                        == "r071_novel_evidence_view_v1"
                        and previous_action.get("step") == step - 1)
    checks = {
        "episode_budget_used": actions_used < 2,
        "no_followup_observation_budget": step < step_cap,
        "current_evidence_incompatible": (
            (current_evidence or {}).get("cluster_uid") == uid
            if continuation else current_evidence is None),
        "previous_action_not_move": previous_action.get("action") == "MOVE",
        "not_approaching_admitted_target": bool(uid and (
            continuation or previous_action.get("policy_candidate_kind") == "target")),
        "session_already_used": continuation or (actions_used == 0 and not used_clusters),
    }
    facts = {"cluster_uid": uid, "step": step, "current_xy": list(current_xy),
             "actions_used": actions_used, "continuation": continuation,
             "confirmation_required_for_action": False}

    def finish(action: dict | None = None) -> tuple[dict | None, str]:
        failed = [key for key, value in checks.items() if value is False]
        if diagnostics is not None:
            diagnostics.update(schema="r071_novel_view_gates_v1", checks=checks,
                               failed_checks=failed, facts=facts)
        return (None if failed else action), (failed[0] if failed else "eligible")

    if not checks["not_approaching_admitted_target"]:
        return finish()
    cluster = next((c for c in memory if c.get("cluster_uid") == uid), None)
    checks["cluster_missing"] = cluster is not None
    if cluster is None:
        return finish()
    source_step = step if continuation else step - 1
    checks["stale_cluster"] = cluster.get("step") == source_step
    checks["cluster_suppressed_or_failed"] = (
        int(cluster.get("suppressed_until_step", -1)) <= step
        and int(cluster.get("verification_failures", 0)) < 2)
    dispersion = float(cluster.get("position_dispersion_m", math.inf))
    checks["unstable_target_position"] = math.isfinite(dispersion) and dispersion <= 0.8
    observations = cluster.get("observations") or []
    checks["no_observation_history"] = bool(observations)
    if not observations:
        return finish()
    latest = observations[-1]
    verification = latest.get("clip_verification") or {}
    sources = set(verification.get("support_sources") or [])
    checks["stale_source_observation"] = latest.get("step") == source_step
    checks["latest_view_not_accepted"] = bool(
        verification.get("available") and verification.get("accepted")
        and sources.intersection({"clip_b32", "siglip2"}))
    depth = latest.get("depth_m")
    checks["latest_view_not_close"] = bool(
        depth is not None and math.isfinite(float(depth)) and 0.05 < float(depth) <= 2.0)
    checks["latest_box_invalid"] = bool((latest.get("bbox_quality") or {}).get("valid_extent"))
    observer, target = latest.get("observer_xy"), cluster.get("xy")
    history = [obs.get("observer_xy") for obs in observations]
    checks["invalid_geometry"] = all(
        point is not None and len(point) == 2 and all(math.isfinite(float(v)) for v in point)
        for point in [current_xy, target, *history])
    facts.update(cluster_step=cluster.get("step"), cluster_xy=target,
                 position_dispersion_m=dispersion if math.isfinite(dispersion) else None,
                 verification_failures=cluster.get("verification_failures", 0),
                 source_observation_step=latest.get("step"), observer_history_xy=history,
                 observation_count=len(observations), distinct_viewpoints=cluster.get("distinct_viewpoints"),
                 source_support=sorted(sources))
    if not checks["invalid_geometry"]:
        return finish()
    source_move = math.dist(current_xy, observer)
    checks["target_not_nearby"] = math.dist(current_xy, target) <= 2.0 and math.dist(observer, target) <= 2.5
    checks["source_pose_incompatible"] = (source_move <= 1e-6 if continuation
                                           else 0.15 <= source_move <= 0.75)
    if any(value is False for value in checks.values()):
        return finish()
    selected, counts = _novel_waypoint(walkable_map, current_xy, target, history)
    checks["no_safe_novel_view"] = selected is not None
    facts["planner_candidates"] = counts
    if selected is None:
        return finish()
    _, point, baseline, novelty = selected
    yaw = math.atan2(target[1] - point[1], target[0] - point[0])
    return finish({
        "mode": "evidence_view_recovery", "protocol": "r071_novel_evidence_view_v1",
        "cluster_uid": str(uid), "source_observation_step": source_step,
        "action_step": step, "expected_observation_step": step + 1,
        "return_xy": list(point), "return_yaw": yaw, "face_direction_xy": list(target),
        "displacement_m": math.dist(current_xy, point),
        "current_cluster_distance_m": math.dist(current_xy, target),
        "source_support": sorted(sources), "source_accepted": True,
        "repeated_observation_pose": False, "confirmation_granted": False,
        "continuation": continuation, "action_index": actions_used + 1,
        "min_history_baseline_m": baseline, "min_target_bearing_change_rad": novelty,
        "observer_history_xy": [list(old) for old in history],
        "distinct_viewpoints_before": cluster.get("distinct_viewpoints", 0),
    })
