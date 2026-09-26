"""One bounded return to a recently verified observation pose.

Uses only policy memory, executed actions, and the walkable map. Returning to
the recorded position AND yaw does not manufacture an independent yaw view.
This helper neither admits detections nor changes the terminal STOP gates.
"""
from __future__ import annotations

import math
from typing import Any


def evidence_view_recovery_action(
    *, memory: list[dict], current_xy: tuple[float, float],
    current_evidence: dict | None, previous_action: dict,
    step: int, step_cap: int, actions_used: int,
    used_clusters: set[str], walkable_map: Any,
) -> tuple[dict | None, str]:
    """Return a certified MOVE proposal or an explicit ineligibility reason."""
    if actions_used >= 1:
        return None, "episode_budget_used"
    if step >= step_cap:
        return None, "no_followup_observation_budget"
    if current_evidence is not None:
        return None, "current_evidence_present"
    if previous_action.get("action") != "MOVE":
        return None, "previous_action_not_move"
    uid = previous_action.get("selected_navigation_cluster_uid")
    if not uid or previous_action.get("policy_candidate_kind") != "target":
        return None, "not_approaching_admitted_target"
    if uid in used_clusters:
        return None, "cluster_already_revisited"
    cluster = next((item for item in memory if item.get("cluster_uid") == uid), None)
    if cluster is None:
        return None, "cluster_missing"
    if int(cluster.get("step", -1)) != step - 1:
        return None, "not_first_missed_frame"
    if (int(cluster.get("suppressed_until_step", -1)) > step
            or int(cluster.get("verification_failures", 0)) >= 2):
        return None, "cluster_suppressed_or_failed"
    if float(cluster.get("position_dispersion_m", math.inf)) > 0.8:
        return None, "unstable_target_position"
    observations = cluster.get("observations") or []
    if not observations:
        return None, "no_observation_history"
    latest = observations[-1]
    if int(latest.get("step", -1)) != step - 1:
        return None, "latest_observation_not_previous_frame"
    verification = latest.get("clip_verification") or {}
    if not {"clip_b32", "siglip2"}.issubset(verification.get("support_sources") or []):
        return None, "latest_view_lacks_dual_support"
    depth = latest.get("depth_m")
    if depth is None or not math.isfinite(depth) or not 0.05 < depth <= 2.0:
        return None, "latest_view_not_close"
    if not (latest.get("bbox_quality") or {}).get("valid_extent", False):
        return None, "latest_box_invalid"
    observer = latest.get("observer_xy")
    target = cluster.get("xy")
    yaw = latest.get("yaw")
    if (observer is None or target is None or yaw is None
            or not all(math.isfinite(float(v)) for v in (*observer, *target, *current_xy, yaw))):
        return None, "invalid_geometry"
    if not any(
        item.get("observer_xy") and math.dist(item["observer_xy"], observer) >= 0.35
        for item in observations[:-1]
    ):
        return None, "no_translated_supporting_view"
    if math.dist(current_xy, target) > 2.0 or math.dist(observer, target) > 2.5:
        return None, "target_not_nearby"
    displacement = math.dist(current_xy, observer)
    if not 0.15 <= displacement <= 0.75:
        return None, "return_distance_outside_bounds"
    # Check the complete return segment rather than only the destination.
    samples = max(1, math.ceil(displacement / 0.025))
    if any(not walkable_map.is_walkable(
        current_xy[0] + (observer[0] - current_xy[0]) * i / samples,
        current_xy[1] + (observer[1] - current_xy[1]) * i / samples,
    ) for i in range(samples + 1)):
        return None, "return_segment_not_walkable"
    if not walkable_map.line_of_sight(*current_xy, *observer):
        return None, "return_segment_occluded"
    return {
        "mode": "evidence_view_recovery", "cluster_uid": str(uid),
        "source_observation_step": int(latest["step"]),
        "action_step": int(step), "expected_observation_step": int(step + 1),
        "return_xy": [float(v) for v in observer], "return_yaw": float(yaw),
        "face_direction_xy": [observer[0] + math.cos(yaw), observer[1] + math.sin(yaw)],
        "displacement_m": displacement, "current_cluster_distance_m": math.dist(current_xy, target),
        "source_support": sorted(verification["support_sources"]),
        "repeated_observation_pose": True,
    }, "eligible"
