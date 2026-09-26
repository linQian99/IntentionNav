"""Read-only candidate diagnostics using outputs already computed by policy."""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np


def projection_sample(depth: np.ndarray | None, bbox: list,
                      pose: dict, image_width: int, image_height: int,
                      y_fraction: float, half_window: int) -> dict:
    """Describe the exact R068 sample and both formulas, without model calls."""
    if depth is None:
        return {"valid": False, "reason": "no_depth"}
    height, width = depth.shape[:2]
    px = int(round(((bbox[0] + bbox[2]) / 2 / image_width) * width))
    py = int(round(((bbox[1] * (1 - y_fraction) + bbox[3] * y_fraction) / image_height) * height))
    result = {"pixel_xy": [px, py], "depth_shape": [height, width],
              "half_window": half_window, "diagnostic_only": True}
    if not (0 <= px < width and 0 <= py < height):
        return {**result, "valid": False, "reason": "pixel_out_of_bounds"}
    window = depth[max(0, py-half_window):min(height, py+half_window+1),
                   max(0, px-half_window):min(width, px+half_window+1)]
    valid = window[(window > 0.05) & np.isfinite(window)]
    if not valid.size:
        return {**result, "valid": False, "reason": "no_valid_sample_depth"}
    d = float(np.median(valid))
    yaw = float(pose["yaw"])
    origin = pose["position"]
    bearing = yaw - (2 * px / width - 1) * math.pi / 4
    left = -(px - width / 2) * d / (width / 2)
    return {**result, "valid": True, "axial_depth_m": d,
            "valid_pixel_count": int(valid.size),
            "sample_min_m": float(valid.min()), "sample_max_m": float(valid.max()),
            "legacy_xy": [origin[0] + d * math.cos(bearing), origin[1] + d * math.sin(bearing)],
            "pinhole_xy": [origin[0] + d * math.cos(yaw) - left * math.sin(yaw),
                           origin[1] + d * math.sin(yaw) + left * math.cos(yaw)]}


def build_perception_trace(
    *, step: int, pose: dict, detections: list, strict_detections: list,
    verifications: list, selected: tuple | None, evidence: dict | None,
    admission_enabled: bool, depth: np.ndarray | None, image_shape: tuple,
    y_fraction: float, half_window: int, error: str | None,
    room_rejections: int, wall_rejections: int,
) -> dict:
    """Log all proposals, including rejection before projection and selection."""
    strict_indices = {id(detection): index for index, detection in enumerate(strict_detections)}
    candidates = []
    for index, detection in enumerate(detections):
        label, score, bbox = detection
        strict_index = strict_indices.get(id(detection))
        verification = (verifications[strict_index] if strict_index is not None
                        and strict_index < len(verifications) else None)
        is_selected = detection is selected
        if not admission_enabled:
            reason = "admission_disabled"
        elif strict_index is None:
            reason = "label_filtered"
        elif verification is None or not verification.get("available", False):
            reason = "verification_unavailable"
        elif not verification.get("support_sources"):
            reason = "semantic_rejected"
        elif not is_selected:
            reason = "verified_not_selected"
        elif evidence is not None:
            reason = "admitted"
        elif room_rejections:
            reason = "room_rejected"
        elif wall_rejections:
            reason = "wall_rejected"
        else:
            reason = "projection_unavailable"
        if error and not (is_selected and evidence is not None):
            reason = "perception_error"
        candidates.append({"candidate_index": index, "label": label, "score": float(score),
                           "bbox": [float(v) for v in bbox], "strict_label_match": strict_index is not None,
                           "verification": verification, "selected": is_selected, "outcome": reason,
                           "projection_sample": projection_sample(
                               depth, bbox, pose, image_shape[1], image_shape[0], y_fraction, half_window)})
    return {"schema": "r069_candidate_trace_v1", "step": int(step),
            "projection_config": {"bbox_y_fraction": y_fraction, "half_window": half_window,
                                  "horizontal_fov_deg": 90.0},
            "pre_action_pose": {"position": list(pose["position"]), "yaw": float(pose["yaw"])},
            "raw_proposal_count": len(candidates), "candidates": candidates,
            "outcome_counts": dict(Counter(c["outcome"] for c in candidates)),
            "admitted_cluster_uid": (evidence or {}).get("cluster_uid"), "error": error}


def save_replay_frame(directory: Path, step: int, rgb: np.ndarray,
                      depth: np.ndarray | None) -> str:
    """Save lossless policy input arrays; no extra rendering or perception."""
    path = directory / f"step_{step:02d}_policy_input.npz"
    arrays = {"rgb": rgb}
    if depth is not None:
        arrays["depth"] = depth
    # Per-episode immutable output directory; replacing is never permitted.
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
    return path.name
