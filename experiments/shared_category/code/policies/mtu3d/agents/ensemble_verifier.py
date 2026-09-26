"""Conservative CLIP-B/32 + SigLIP2 semantic verifier ensemble."""
from __future__ import annotations

import os

import numpy as np

import clip_verifier
import siglip_verifier


ENABLED = os.environ.get("INAV_SIGLIP_ENSEMBLE", "0") == "1"
MODEL_ID = (
    f"{clip_verifier.MODEL_ID}+{siglip_verifier.MODEL_ID}"
    if ENABLED else clip_verifier.MODEL_ID
)
MODEL_REVISION = (
    f"{clip_verifier.MODEL_REVISION}+{siglip_verifier.MODEL_REVISION}"
    if ENABLED else clip_verifier.MODEL_REVISION
)
BACKGROUND_CALIBRATION = clip_verifier.BACKGROUND_CALIBRATION


def _merged(primary: dict, secondary: dict, target: str, component_max_rank: int) -> dict:
    primary_ok = clip_verifier.supports_target(
        primary, target, component_max_rank=component_max_rank
    )
    secondary_ok = siglip_verifier.supports_target(secondary, target)
    candidates = [item for item in (primary, secondary) if item.get("available")]
    chosen = min(
        candidates,
        key=lambda item: int(item.get("target_rank", 10_000)),
        default={"available": False, "accepted": False},
    )
    result = dict(chosen)
    result.update({
        "available": bool(candidates),
        "accepted": primary_ok or secondary_ok,
        "support_sources": [
            name for name, ok in (
                ("clip_b32", primary_ok), ("siglip2", secondary_ok)
            ) if ok
        ],
        "clip_b32": primary,
        "siglip2": secondary,
        "model": MODEL_ID,
    })
    return result


def verify_detections(
    rgb: np.ndarray,
    detections: list[tuple[str, float, list[float]]],
    target: str,
    max_detections: int = 5,
    max_rank: int = 3,
) -> list[dict]:
    primary = clip_verifier.verify_detections(
        rgb, detections, target,
        max_detections=max_detections,
        max_rank=max_rank,
    )
    if not ENABLED:
        return primary
    secondary = siglip_verifier.verify_detections(
        rgb, detections, target,
        max_detections=max_detections,
        max_rank=max_rank,
    )
    return [
        _merged(old, new, target, component_max_rank=max_rank)
        for old, new in zip(primary, secondary)
    ]


def supports_target(
    verification: dict,
    target: str,
    component_max_rank: int = 3,
) -> bool:
    if "clip_b32" not in verification:
        return clip_verifier.supports_target(
            verification, target, component_max_rank=component_max_rank
        )
    return (
        clip_verifier.supports_target(
            verification["clip_b32"],
            target,
            component_max_rank=component_max_rank,
        )
        or siglip_verifier.supports_target(verification["siglip2"], target)
    )


def score_scene_context(
    rgb: np.ndarray,
    target: str,
    likely_rooms=(),
    support_labels=(),
) -> dict:
    """Use the pinned primary CLIP model for non-terminal scene retrieval."""
    return clip_verifier.score_scene_context(
        rgb,
        target,
        likely_rooms=likely_rooms,
        support_labels=support_labels,
    )


def score_scene_context_prompt_ensemble(
    rgb: np.ndarray,
    target: str,
) -> dict:
    """Use the calibrated primary CLIP prompt ensemble for frontier belief.

    The engine historically imports this module as ``clip_verifier`` so that
    terminal verification can optionally fuse CLIP and SigLIP.  R040's
    probabilistic frontier signal was calibrated with the primary CLIP model,
    therefore this public wrapper deliberately delegates without adding a
    second model or changing the offline-tested score distribution.
    """
    return clip_verifier.score_scene_context_prompt_ensemble(rgb, target)
