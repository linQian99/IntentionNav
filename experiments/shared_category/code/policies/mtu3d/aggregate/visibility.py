"""Occlusion-aware visibility checks for active IntentionNav trajectories.

This module intentionally duplicates the pure geometry used by
`capture/capture_surfaces.py` instead of importing it: that capture script
initializes Isaac Sim at module import time. The functions here stay lightweight
so metric aggregation can run in a normal Python process.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parents[1]
DEFAULT_DATASET_ROOT = REPO / "results/dataset"
DEFAULT_DATASET_JSONL = DEFAULT_DATASET_ROOT / "selected_500_intents.jsonl"
DEFAULT_SCENE_SUMMARY = Path(os.environ.get(
    "INTENTIONNAV_SCENE_SUMMARY",
    str(REPO / "data/SceneSummary/kujiale_scene_summary"),
)
)

IMAGE_SIZE = 1024
CAMERA_HEIGHT = 1.5
VISIBLE_RADIUS_M = 3.0
MIN_EFFECTIVE_VISIBLE_PX = 1.0   # "looks visible at all" — any non-zero coverage
MAX_OCCLUSION_RATIO = 0.95       # only fully-occluded targets count as not visible


class VisibilityDataError(RuntimeError):
    """Raised when an occlusion-aware metric lacks its required scene data."""


def _project_bbox_to_2d_rect(camera_pos, aim_pos, bbox_min, bbox_max,
                             image_size: int = IMAGE_SIZE):
    """Project a 3D bbox to a 2D image rectangle.

    Returns (x_min, y_min, x_max, y_max), or None when all bbox corners are
    behind the camera. This matches the capture-time pinhole approximation.
    """
    cam = np.array(camera_pos, dtype=np.float64)
    aim = np.array(aim_pos, dtype=np.float64)
    direction = np.array([aim[0] - cam[0], aim[1] - cam[1], 0.0])
    dn = np.linalg.norm(direction)
    if dn < 1e-6:
        return None
    fwd = direction / dn
    right = np.cross(np.array([0.0, 0.0, 1.0]), fwd)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        return None
    right = right / rn
    up = np.cross(fwd, right)
    r_inv = np.column_stack((fwd, right, up)).T

    bmin = np.array(bbox_min, dtype=np.float64)
    bmax = np.array(bbox_max, dtype=np.float64)
    corners = np.array([
        [bmin[0], bmin[1], bmin[2]], [bmin[0], bmin[1], bmax[2]],
        [bmin[0], bmax[1], bmin[2]], [bmin[0], bmax[1], bmax[2]],
        [bmax[0], bmin[1], bmin[2]], [bmax[0], bmin[1], bmax[2]],
        [bmax[0], bmax[1], bmin[2]], [bmax[0], bmax[1], bmax[2]],
    ])
    local = (r_inv @ (corners - cam).T).T
    in_front = local[:, 0] > 0.01
    if not np.any(in_front):
        return None

    half = image_size / 2.0
    pxs, pys = [], []
    for i in range(8):
        if not in_front[i]:
            continue
        px = local[i, 1] / local[i, 0] * half + half
        py = -local[i, 2] / local[i, 0] * half + half
        pxs.append(px)
        pys.append(py)
    if not pxs:
        return None
    return (min(pxs), min(pys), max(pxs), max(pys))


def _clipped_rect_area(rect, image_size: int = IMAGE_SIZE) -> float:
    if rect is None:
        return 0.0
    x0 = max(0.0, rect[0])
    y0 = max(0.0, rect[1])
    x1 = min(float(image_size), rect[2])
    y1 = min(float(image_size), rect[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def compute_occlusion_ratio(camera_pos, aim_pos, target_bbox_min, target_bbox_max,
                            scene_objects, image_size: int = IMAGE_SIZE) -> float:
    """Approximate target-projection occlusion by nearer scene-object bboxes."""
    cam = np.array(camera_pos[:2], dtype=np.float64)
    tgt_center = np.array([
        (target_bbox_min[0] + target_bbox_max[0]) / 2.0,
        (target_bbox_min[1] + target_bbox_max[1]) / 2.0,
    ])
    dist_to_target = float(np.linalg.norm(tgt_center - cam))

    t_rect = _project_bbox_to_2d_rect(
        camera_pos, aim_pos, target_bbox_min, target_bbox_max, image_size
    )
    if t_rect is None:
        return 1.0
    tx0 = max(0.0, t_rect[0])
    ty0 = max(0.0, t_rect[1])
    tx1 = min(float(image_size), t_rect[2])
    ty1 = min(float(image_size), t_rect[3])
    t_area = max(0.0, tx1 - tx0) * max(0.0, ty1 - ty0)
    if t_area < 1.0:
        return 1.0

    total_occluded = 0.0
    for obj in scene_objects:
        pos = obj.get("position")
        bmin = obj.get("min_points")
        bmax = obj.get("max_points")
        if not pos or not bmin or not bmax:
            continue
        if len(bmin) < 3 or len(bmax) < 3:
            continue

        obj_center = np.array(pos[:2], dtype=np.float64)
        dist_to_obj = float(np.linalg.norm(obj_center - cam))
        if dist_to_obj >= dist_to_target * 0.95:
            continue
        to_obj = obj_center - cam
        to_tgt = tgt_center - cam
        if float(np.dot(to_obj, to_tgt)) < 0:
            continue

        o_rect = _project_bbox_to_2d_rect(
            camera_pos, aim_pos, bmin, bmax, image_size
        )
        if o_rect is None:
            continue
        ix0 = max(tx0, o_rect[0])
        iy0 = max(ty0, o_rect[1])
        ix1 = min(tx1, o_rect[2])
        iy1 = min(ty1, o_rect[3])
        total_occluded += max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)

    return min(1.0, total_occluded / t_area)


@lru_cache(maxsize=8)
def _items_by_selection_id(dataset_jsonl: str = str(DEFAULT_DATASET_JSONL)) -> dict[str, dict]:
    out = {}
    path = Path(dataset_jsonl)
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            out[item["selection_id"]] = item
    return out


@lru_cache(maxsize=512)
def _scene_manifest(
    scene_id: str,
    dataset_root: str = str(DEFAULT_DATASET_ROOT),
) -> list[dict]:
    path = Path(dataset_root) / scene_id / "manifest.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


@lru_cache(maxsize=512)
def _scene_objects(scene_id: str, require_occluders: bool = True) -> list[dict]:
    """Load scene-object AABBs used by the occlusion approximation.

    GSR used to silently turn into a no-occluder projection metric when
    ``object_dict.json`` was unavailable. That changes the metric while keeping
    the same name, so benchmark aggregation now fails closed by default.
    ``require_occluders=False`` is retained only for explicitly labelled
    diagnostics and legacy reproduction.
    """
    root = Path(os.environ.get(
        "INTENTIONNAV_SCENE_SUMMARY",
        str(DEFAULT_SCENE_SUMMARY),
    ))
    path = root / scene_id / "object_dict.json"
    if not path.exists():
        if require_occluders:
            raise VisibilityDataError(
                "occlusion-aware visibility requires "
                f"{path}; set INTENTIONNAV_SCENE_SUMMARY to the frozen "
                "scene-summary root or pass --allow-missing-occluders for a "
                "clearly labelled legacy diagnostic"
            )
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    objects = [
        {
            "position": v.get("position"),
            "min_points": v.get("min_points"),
            "max_points": v.get("max_points"),
        }
        for v in raw.values()
        if v.get("position") and v.get("min_points") and v.get("max_points")
    ]
    if require_occluders and not objects:
        raise VisibilityDataError(
            f"scene summary contains no usable occluder AABBs: {path}"
        )
    return objects


def _target_entry(
    record: dict,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    dataset_jsonl: str | Path = DEFAULT_DATASET_JSONL,
) -> dict | None:
    scene_id = record.get("scene_id")
    if not scene_id:
        return None
    manifest = _scene_manifest(scene_id, str(Path(dataset_root).resolve()))
    if not manifest:
        return None

    item = _items_by_selection_id(
        str(Path(dataset_jsonl).resolve())
    ).get(record.get("selection_id", ""), {})
    representative = item.get("target_representative")
    if representative:
        for entry in manifest:
            if entry.get("target_representative") == representative:
                return entry

    photo_name = Path(str(item.get("photo") or record.get("photo") or "")).name
    if photo_name:
        for entry in manifest:
            for photo in entry.get("photos") or []:
                if Path(str(photo.get("path") or "")).name == photo_name:
                    return entry

    target_category = record.get("target_category")
    matches = [e for e in manifest if e.get("target_category") == target_category]
    return matches[0] if len(matches) == 1 else None


def _combined_target_bbox(
    record: dict,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    dataset_jsonl: str | Path = DEFAULT_DATASET_JSONL,
):
    entry = _target_entry(record, dataset_root, dataset_jsonl)
    if not entry:
        return None, None
    bmin_out = None
    bmax_out = None
    for detail in entry.get("target_details") or []:
        bbox = detail.get("bbox") or {}
        bmin = bbox.get("min")
        bmax = bbox.get("max")
        if not bmin or not bmax or len(bmin) < 3 or len(bmax) < 3:
            continue
        if bmin_out is None:
            bmin_out = [float(v) for v in bmin[:3]]
            bmax_out = [float(v) for v in bmax[:3]]
        else:
            for i in range(3):
                bmin_out[i] = min(bmin_out[i], float(bmin[i]))
                bmax_out[i] = max(bmax_out[i], float(bmax[i]))
    return bmin_out, bmax_out


def _target_center(bbox_min, bbox_max) -> tuple[float, float, float]:
    return (
        (bbox_min[0] + bbox_max[0]) / 2.0,
        (bbox_min[1] + bbox_max[1]) / 2.0,
        (bbox_min[2] + bbox_max[2]) / 2.0,
    )


def trajectory_visibility(record: dict,
                          radius_m: float = VISIBLE_RADIUS_M,
                          min_effective_visible_px: float = MIN_EFFECTIVE_VISIBLE_PX,
                          max_occlusion_ratio: float = MAX_OCCLUSION_RATIO,
                          require_occluders: bool = True,
                          dataset_root: str | Path = DEFAULT_DATASET_ROOT,
                          dataset_jsonl: str | Path = DEFAULT_DATASET_JSONL) -> dict:
    """Return target visibility for the agent's FINAL frame only.

    G measures whether the agent ACTUALLY STOPPED with target in view —
    not just "did target appear in any frame across the whole trajectory".
    Old version (G_seen across all steps) was loose: agent could pass by
    target on step 5, look elsewhere by step 30, and still get G=1.
    Tightened: only the final-step camera pose counts.

    Returns G_seen=1 iff at the final step the target's effective visible
    bbox area >= min_effective_visible_px and occlusion <= max ratio.
    """
    traj = record.get("trajectory") or []
    if not traj:
        return {}
    bbox_min, bbox_max = _combined_target_bbox(
        record,
        dataset_root=dataset_root,
        dataset_jsonl=dataset_jsonl,
    )
    if not bbox_min or not bbox_max:
        return {}

    target_center = _target_center(bbox_min, bbox_max)
    scene_objects = _scene_objects(
        record.get("scene_id", ""),
        require_occluders=require_occluders,
    )
    best = {
        "G_seen": 0,
        "max_effective_visible_px": 0.0,
        "best_occlusion_ratio": None,
        "best_visible_step": None,
    }

    # Only the FINAL trajectory entry counts (the frame agent stopped at).
    last = None
    for step in reversed(traj):
        if step.get("position") and step.get("yaw") is not None:
            last = step
            break
    if last is None:
        return best

    pos = last.get("position")
    yaw = last.get("yaw")
    cam = [
        float(pos[0]),
        float(pos[1]),
        float(pos[2]) if len(pos) >= 3 else CAMERA_HEIGHT,
    ]
    dist_xy = math.hypot(cam[0] - target_center[0], cam[1] - target_center[1])
    if dist_xy > radius_m:
        return best
    look = [math.cos(float(yaw)), math.sin(float(yaw)), 0.0]
    aim = [cam[0] + look[0], cam[1] + look[1], cam[2]]
    rect = _project_bbox_to_2d_rect(cam, aim, bbox_min, bbox_max, IMAGE_SIZE)
    proj_area = _clipped_rect_area(rect, IMAGE_SIZE)
    if proj_area < 1.0:
        return best
    occ = compute_occlusion_ratio(
        cam, aim, bbox_min, bbox_max, scene_objects, image_size=IMAGE_SIZE
    )
    effective_visible = proj_area * (1.0 - occ)
    best["max_effective_visible_px"] = round(float(effective_visible), 3)
    best["best_occlusion_ratio"] = round(float(occ), 4)
    best["best_visible_step"] = last.get("step")
    if effective_visible >= min_effective_visible_px and occ <= max_occlusion_ratio:
        best["G_seen"] = 1
    return best
