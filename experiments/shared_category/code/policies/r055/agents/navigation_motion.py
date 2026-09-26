"""Shared MOVE execution and audit contract on the permitted navigation grid.

Use the same full-segment and no-corner-cutting primitives as the MTU adapter.
Connectivity alone does not authorize a straight teleport between two cells.
No target labels, renderer, observations or model calls enter this module.
"""
from __future__ import annotations

import math

from mtu3d_strict_grid import strict_path_action, strict_segment


def _xy(value) -> tuple[float, float]:
    if len(value) < 2:
        raise ValueError("Missing navigation coordinates")
    point = (float(value[0]), float(value[1]))
    if not all(math.isfinite(v) for v in point):
        raise ValueError("Nonfinite navigation coordinates")
    return point


def resolve_navigation_move(wm, current_xy, requested_xy, maximum: float = 1.7):
    """Return one clear, bounded step toward a request, or a paid hold.

    An unobstructed in-budget request remains byte-for-byte the same float
    coordinates. Otherwise use MTU's strict A* prefix. Invalid/unreachable
    destinations hold the valid current pose; never snap an invalid request
    across a wall. The caller must charge the resulting MOVE as one action.
    """
    current = _xy(current_xy)
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("Invalid MOVE distance budget")
    if not strict_segment(wm, current, current):
        raise ValueError("MOVE starts outside the permitted free grid")
    try:
        requested = _xy(requested_xy)
    except (TypeError, ValueError, IndexError, OverflowError) as exc:
        raise ValueError("Invalid proposed MOVE") from exc
    distance = math.dist(current, requested)
    if not wm.is_walkable(*requested):
        chosen, reason = current, "hold_invalid_endpoint"
    elif distance <= maximum + 1e-8 and strict_segment(wm, current, requested):
        chosen, reason = requested, "direct"
    else:
        prefix = strict_path_action(wm, current, requested, maximum=maximum)
        chosen = tuple(prefix) if prefix is not None else current
        reason = "strict_path_prefix" if prefix is not None else "hold_no_route"
    if (not strict_segment(wm, current, chosen)
            or math.dist(current, chosen) > maximum + 1e-8):
        raise ValueError("Resolved MOVE violates the execution contract")
    return chosen, {
        "version": "full_segment_move_v1",
        "requested_xy": list(requested),
        "executed_xy": list(chosen),
        "adjusted": chosen != requested,
        "reason": reason,
        "maximum_m": maximum,
        "distance_m": math.dist(current, chosen),
    }


def audit_navigation_motion(record: dict, wm, maximum: float = 1.7) -> dict:
    """Reject invalid poses, displaced rotations/STOPs and blocked MOVE segments."""
    trajectory = record["trajectory"]
    cap = record["step_cap"]
    if (not isinstance(cap, int) or cap < 1 or not 2 <= len(trajectory) <= cap + 1
            or [p["step"] for p in trajectory] != list(range(len(trajectory)))
            or trajectory[0].get("action", "START") not in {"START", ""}):
        raise ValueError("Invalid action sequence or budget")
    moves = 0
    previous = None
    for index, event in enumerate(trajectory):
        position = event["position"]
        if len(position) != 3 or not all(math.isfinite(float(v)) for v in position):
            raise ValueError(f"step {index}: invalid position")
        xy = _xy(position)
        yaw = float(event["yaw"])
        if not math.isfinite(yaw) or not strict_segment(wm, xy, xy):
            raise ValueError(f"step {index}: invalid yaw or occupied pose")
        if previous is not None:
            action = event["action"]
            if abs(float(position[2]) - float(previous["position"][2])) > 1e-5:
                raise ValueError(f"step {index}: uncharged height change")
            distance = math.dist(previous["position"][:2], xy)
            if action == "MOVE":
                if distance > maximum + 1e-8 or not strict_segment(wm, previous["position"][:2], xy):
                    raise ValueError(f"step {index}: blocked or over-budget MOVE")
                moves += 1
            elif action in {"STOP", "ROTATE_SCAN", "ROTATE_TARGET_RECENTER", "ROTATE_TARGET"}:
                if distance > 1e-5:
                    raise ValueError(f"step {index}: translation during {action}")
                if action == "STOP":
                    angle = math.atan2(math.sin(yaw - previous["yaw"]), math.cos(yaw - previous["yaw"]))
                    if index != len(trajectory) - 1 or abs(angle) > 1e-6:
                        raise ValueError(f"step {index}: nonterminal or rotated STOP")
            else:
                raise ValueError(f"step {index}: unsupported action {action}")
        previous = event
    if trajectory[-1]["action"] != "STOP" and len(trajectory) != cap + 1:
        raise ValueError("Trajectory ended before STOP or the action budget")
    return {"passed": True, "actions": len(trajectory) - 1, "moves": moves}
