"""Optional MobileSAM box-to-mask refinement for IntentionNav.

GroundingDINO supplies the open-vocabulary semantics and a box prompt.  This
module converts that box into an instance mask and projects only masked depth
pixels into the navigation plane.  It is lazy and fail-closed: experiments
without the pinned local checkpoint retain the existing box projection.
"""
from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import Iterable

import numpy as np


MODEL_COMMIT = "f706ad9c4eb7f219c00d9050e46328518ffb65d2"
CHECKPOINT_SHA256 = (
    "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f"
)
DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "INAV_MOBILE_SAM_CHECKPOINT",
        "/path/to/workspace/models/MobileSAM/mobile_sam.pt",
    )
)

_lock = threading.Lock()
_predictor = None
_device = None


def configured_checkpoint() -> str:
    return str(DEFAULT_CHECKPOINT)


def _ensure_loaded() -> None:
    global _predictor, _device
    if _predictor is not None:
        return
    with _lock:
        if _predictor is not None:
            return
        if not DEFAULT_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"MobileSAM checkpoint not found: {DEFAULT_CHECKPOINT}"
            )
        import torch
        from mobile_sam import SamPredictor, sam_model_registry

        _device = os.environ.get(
            "INAV_MASK_DEVICE",
            os.environ.get(
                "DINO_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"
            ),
        )
        model = sam_model_registry["vit_t"](
            checkpoint=str(DEFAULT_CHECKPOINT)
        )
        model.to(device=_device).eval()
        _predictor = SamPredictor(model)


def refine_bbox(rgb: np.ndarray, bbox: Iterable[float]) -> dict:
    """Return the best MobileSAM mask for a DINO box, never raising."""
    values = np.asarray(list(bbox), dtype=np.float32)
    if rgb.ndim != 3 or values.shape != (4,):
        return {"available": False, "error": "invalid_input"}
    height, width = rgb.shape[:2]
    values[[0, 2]] = np.clip(values[[0, 2]], 0, max(0, width - 1))
    values[[1, 3]] = np.clip(values[[1, 3]], 0, max(0, height - 1))
    if values[2] <= values[0] or values[3] <= values[1]:
        return {"available": False, "error": "invalid_bbox"}
    try:
        _ensure_loaded()
        _predictor.set_image(np.ascontiguousarray(rgb))
        masks, scores, _ = _predictor.predict(
            box=values,
            multimask_output=True,
        )
        if len(scores) == 0:
            return {"available": False, "error": "no_mask"}
        index = int(np.argmax(scores))
        mask = np.asarray(masks[index], dtype=bool)
        return {
            "available": True,
            "mask": mask,
            "predicted_iou": round(float(scores[index]), 6),
            "mask_area_fraction": round(float(mask.mean()), 6),
            "checkpoint": str(DEFAULT_CHECKPOINT),
        }
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def mask_depth_world_xy(
    mask: np.ndarray | None,
    depth: np.ndarray | None,
    pose: dict,
    hfov_rad: float = math.pi / 2,
) -> tuple[tuple[float, float] | None, dict]:
    """Project masked depth pixels and return their robust XY median.

    Unlike a box-center ray, this representation excludes background pixels
    and does not assume that the visible object's center lies on the floor.
    The returned point is a visible object-surface anchor, matching the
    surface-based ObjectNav diagnostics used by the benchmark.
    """
    if mask is None or depth is None or mask.shape != depth.shape[:2]:
        return None, {"available": False, "error": "invalid_mask_or_depth"}
    valid = mask & np.isfinite(depth) & (depth > 0.05)
    ys, xs = np.nonzero(valid)
    if xs.size == 0:
        return None, {"available": False, "error": "no_valid_mask_depth"}
    ranges = depth[ys, xs].astype(np.float64)
    width = depth.shape[1]
    # Use pixel centres; otherwise even a symmetric two-pixel mask acquires a
    # half-pixel lateral bias at low resolutions.
    u_norm = ((xs.astype(np.float64) + 0.5) / max(1, width)) * 2.0 - 1.0
    bearings = float(pose["yaw"]) - u_norm * (hfov_rad / 2.0)
    world_x = float(pose["position"][0]) + ranges * np.cos(bearings)
    world_y = float(pose["position"][1]) + ranges * np.sin(bearings)
    xy = (float(np.median(world_x)), float(np.median(world_y)))
    return xy, {
        "available": True,
        "valid_depth_pixels": int(xs.size),
        "depth_median": round(float(np.median(ranges)), 4),
        "depth_q10": round(float(np.quantile(ranges, 0.1)), 4),
        "depth_q90": round(float(np.quantile(ranges, 0.9)), 4),
        "pixel_x_median": round(float(np.median(xs)), 2),
    }
