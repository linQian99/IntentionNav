"""One observation recheck, with admission and terminal confirmation separate.

An ordinary accepted observation can justify a bounded return for re-observation.
It does not need to have already acquired two encoders or two physical views.
This action never admits a rejected detection or changes any STOP criterion.
"""
from __future__ import annotations

import math
from typing import Any


def evidence_view_recovery_action(
    *, memory: list[dict], current_xy: tuple[float, float],
    current_evidence: dict | None, previous_action: dict,
    step: int, step_cap: int, actions_used: int,
    used_clusters: set[str], walkable_map: Any,
    diagnostics: dict | None = None,
) -> tuple[dict | None, str]:
    """Propose one exact-pose MOVE and report every evaluable failed gate.

    The first-miss, proximity, suppression, complete-segment and episode-budget
    bounds remain those of R069. Dual support and translated history are STOP
    evidence, not prerequisites for buying a single additional observation.
    """
    uid = previous_action.get("selected_navigation_cluster_uid")
    checks = {
        "episode_budget_used": actions_used < 1,
        "no_followup_observation_budget": step < step_cap,
        "current_evidence_present": current_evidence is None,
        "previous_action_not_move": previous_action.get("action") == "MOVE",
        "not_approaching_admitted_target": bool(
            uid and previous_action.get("policy_candidate_kind") == "target"),
        "cluster_already_revisited": uid not in used_clusters,
    }
    facts: dict = {"cluster_uid": uid, "step": int(step),
                  "current_xy": list(current_xy), "actions_used": int(actions_used),
                  "confirmation_required_for_action": False}

    def finish(action: dict | None = None) -> tuple[dict | None, str]:
        failed = [reason for reason, passed in checks.items() if passed is False]
        reason = failed[0] if failed else "eligible"
        if diagnostics is not None:
            diagnostics.update(schema="r070_recheck_gates_v1", checks=checks,
                               failed_checks=failed, facts=facts)
        return (None if failed else action), reason

    if not checks["not_approaching_admitted_target"]:
        return finish()
    cluster = next((c for c in memory if c.get("cluster_uid") == uid), None)
    checks["cluster_missing"] = cluster is not None
    if cluster is None:
        return finish()
    checks["not_first_missed_frame"] = int(cluster.get("step", -1)) == step - 1
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
    checks["latest_observation_not_previous_frame"] = int(latest.get("step", -1)) == step - 1
    checks["latest_view_not_accepted"] = bool(
        verification.get("available") and verification.get("accepted")
        and sources.intersection({"clip_b32", "siglip2"}))
    depth = latest.get("depth_m")
    checks["latest_view_not_close"] = bool(
        depth is not None and math.isfinite(float(depth)) and 0.05 < float(depth) <= 2.0)
    checks["latest_box_invalid"] = bool((latest.get("bbox_quality") or {}).get("valid_extent"))
    observer, target, yaw = latest.get("observer_xy"), cluster.get("xy"), latest.get("yaw")
    checks["invalid_geometry"] = bool(
        observer is not None and target is not None and yaw is not None
        and len(observer) == len(target) == len(current_xy) == 2
        and all(math.isfinite(float(v)) for v in (*observer, *target, *current_xy, yaw)))
    # This compact snapshot uses only live policy state and supports exact
    # gate replay without borrowing the final episode's future memory.
    facts.update(cluster_step=cluster.get("step"), cluster_xy=target,
                 position_dispersion_m=dispersion if math.isfinite(dispersion) else None,
                 verification_failures=cluster.get("verification_failures", 0),
                 suppressed_until_step=cluster.get("suppressed_until_step", -1),
                 observation_count=len(observations), source_observation={
                     "step": latest.get("step"), "observer_xy": observer, "yaw": yaw,
                     "depth_m": depth, "bbox_quality": latest.get("bbox_quality"),
                     "clip_verification": {k: verification.get(k) for k in
                                           ("available", "accepted", "support_sources")},
                 })
    if not checks["invalid_geometry"]:
        return finish()
    displacement = math.dist(current_xy, observer)
    checks["target_not_nearby"] = (
        math.dist(current_xy, target) <= 2.0 and math.dist(observer, target) <= 2.5)
    checks["return_distance_outside_bounds"] = 0.15 <= displacement <= 0.75
    facts.update(displacement_m=displacement,
                 current_cluster_distance_m=math.dist(current_xy, target),
                 source_support=sorted(sources))
    checks["return_segment_not_walkable"] = None
    checks["return_segment_occluded"] = None
    if checks["return_distance_outside_bounds"]:
        samples = max(1, math.ceil(displacement / 0.025))
        checks["return_segment_not_walkable"] = all(walkable_map.is_walkable(
            current_xy[0] + (observer[0] - current_xy[0]) * i / samples,
            current_xy[1] + (observer[1] - current_xy[1]) * i / samples,
        ) for i in range(samples + 1))
        checks["return_segment_occluded"] = bool(walkable_map.line_of_sight(*current_xy, *observer))
    return finish({
        "mode": "evidence_view_recovery", "protocol": "r070_observation_recheck_v1",
        "cluster_uid": str(uid), "source_observation_step": int(latest["step"]),
        "action_step": int(step), "expected_observation_step": int(step + 1),
        "return_xy": [float(v) for v in observer], "return_yaw": float(yaw),
        "face_direction_xy": [observer[0] + math.cos(yaw), observer[1] + math.sin(yaw)],
        "displacement_m": displacement, "current_cluster_distance_m": math.dist(current_xy, target),
        "source_support": sorted(sources), "source_accepted": True,
        "repeated_observation_pose": True, "confirmation_granted": False,
    })
