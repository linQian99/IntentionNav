"""One prospective range-to-fit view for an ambiguous floor-supported object.

Only category knowledge, current model outputs, measured depth and a walkable
map enter planning. A floor contact point is a hypothesis, not verified object
geometry. The action never inserts evidence or changes semantic/STOP gates.
"""
from __future__ import annotations

import math
from typing import Any


FLOOR_SUPPORTED = frozenset({
    "bed", "bench", "cabinet", "chair", "desk", "dining table", "floor lamp",
    "fridge", "night stand", "piano", "room divider", "sofa", "stool",
    "storage", "table", "washing machine", "water cooler",
})
BOTTOM_MARGIN_FRACTION = 0.10
MAX_MOVE_M = 1.50


def required_floor_range(camera_height_m: float, vertical_fov_rad: float) -> float:
    """Range putting a hypothetical ground contact at normalized image y=.90."""
    return camera_height_m / ((1 - 2 * BOTTOM_MARGIN_FRACTION)
                              * math.tan(vertical_fov_rad / 2))


def framing_action(
    *, target: str, candidates: list[dict], current_evidence: dict | None,
    pose: dict, image_shape: tuple[int, ...], walkable_map: Any,
    step: int, step_cap: int, actions_used: int,
    camera_height_m: float = 1.5, horizontal_fov_deg: float = 90.0,
    diagnostics: dict | None = None,
) -> tuple[dict | None, str]:
    """Choose at most one map-reachable view; no inputs are mutated."""
    target = " ".join(target.lower().replace("_", " ").split())
    facts: dict = {"schema": "r074_vertical_framing_gates_v1", "candidates": []}

    def finish(action: dict | None, reason: str):
        if diagnostics is not None:
            diagnostics.update(facts, reason=reason, proposal=action)
        return action, reason

    if actions_used >= 1:
        return finish(None, "episode_budget_used")
    if step >= step_cap:
        return finish(None, "no_followup_observation")
    if current_evidence is not None:
        return finish(None, "current_target_already_admitted")
    if target not in FLOOR_SUPPORTED:
        return finish(None, "no_floor_support_prior")
    if walkable_map is None:
        return finish(None, "no_walkable_map")
    height, width = image_shape[:2]
    current = tuple(float(x) for x in pose["position"][:2])
    if (height <= 0 or width <= 0 or not all(math.isfinite(x) for x in current)
            or not 0 < camera_height_m < 3 or not 0 < horizontal_fov_deg < 180):
        return finish(None, "invalid_camera_geometry")
    vfov = 2 * math.atan(height / width * math.tan(math.radians(horizontal_fov_deg) / 2))
    radius = required_floor_range(camera_height_m, vfov)
    facts.update(required_range_m=radius, vertical_fov_rad=vfov)
    proposals = []
    for candidate in candidates:
        index = candidate["candidate_index"]
        v = candidate.get("verification") or {}
        primary, secondary = v.get("clip_b32") or {}, v.get("siglip2") or {}
        box = candidate.get("bbox") or []
        sample = candidate.get("projection_sample") or {}
        reason = None
        if (not candidate.get("strict_label_match") or candidate.get("outcome") != "semantic_rejected"
                or not v.get("available") or v.get("accepted") or v.get("support_sources")):
            reason = "not_unadmitted_semantic_candidate"
        elif (not math.isfinite(float(candidate["score"])) or float(candidate["score"]) < .30 or not primary.get("available")
              or not secondary.get("available") or primary.get("target_rank", 10000) > 3
              or secondary.get("target_rank", 10000) > 3):
            reason = "insufficient_cross_encoder_candidate_support"
        elif (len(box) != 4 or not all(math.isfinite(float(x)) for x in box)
              or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
            reason = "invalid_box"
        elif box[3] < .98 * height:
            reason = "not_bottom_clipped"
        elif not sample.get("valid"):
            reason = "no_measured_depth"
        if reason:
            facts["candidates"].append({"candidate_index": index, "reason": reason})
            continue
        depth = float(sample["axial_depth_m"])
        anchor = tuple(float(x) for x in sample["pinhole_xy"])
        spread = float(sample["sample_max_m"]) - float(sample["sample_min_m"])
        if (len(anchor) != 2 or not all(math.isfinite(x) for x in (*anchor, depth, spread))
                or not .2 <= depth <= 2 or sample["valid_pixel_count"] < 9
                or spread < 0 or spread > .25 * depth):
            facts["candidates"].append({"candidate_index": index, "reason": "uncertain_depth_anchor"})
            continue
        distance = math.dist(current, anchor)
        if not .2 <= distance < radius - .10:
            facts["candidates"].append({"candidate_index": index, "reason": "no_floor_range_deficit"})
            continue
        bearing = math.atan2(current[1] - anchor[1], current[0] - anchor[0])
        safe = []
        for offset in (0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90):
            angle = bearing + math.radians(offset)
            point = (anchor[0] + radius * math.cos(angle), anchor[1] + radius * math.sin(angle))
            move = math.dist(current, point)
            if not .20 <= move <= MAX_MOVE_M:
                continue
            n = math.ceil(move / .025)
            if not all(walkable_map.is_walkable(current[0] + (point[0] - current[0]) * i / n,
                                               current[1] + (point[1] - current[1]) * i / n)
                       for i in range(n + 1)):
                continue
            if not walkable_map.line_of_sight(*point, *anchor):
                continue
            safe.append((move, abs(offset), offset, point))
        facts["candidates"].append({"candidate_index": index, "reason": "eligible" if safe else "no_safe_framing_view",
                                    "anchor_xy": list(anchor), "range_before_m": distance, "safe_views": len(safe)})
        if safe:
            move, _, offset, point = min(safe)
            proposals.append((-float(candidate["score"]), move, index, {
                "mode": "vertical_framing", "protocol": "r074_floor_range_view_v1",
                "candidate_index": index, "action_step": step, "expected_observation_step": step + 1,
                "return_xy": list(point), "face_direction_xy": list(anchor),
                "return_yaw": math.atan2(anchor[1] - point[1], anchor[0] - point[0]),
                "displacement_m": move, "range_before_m": distance, "range_after_m": radius,
                "camera_height_m": camera_height_m, "vertical_fov_rad": vfov,
                "predicted_ground_row_fraction": .5 + camera_height_m / (2 * radius * math.tan(vfov / 2)),
                "ground_support_is_prior": True, "ground_geometry_verified": False,
                "source_accepted": False, "confirmation_granted": False,
                "source_bbox": list(box), "target_bearing_change_deg": offset,
                "clip_rank": primary["target_rank"], "siglip_rank": secondary["target_rank"],
            }))
    return finish(min(proposals, key=lambda p: p[:3])[3] if proposals else None,
                  "eligible" if proposals else "no_eligible_framing_candidate")
