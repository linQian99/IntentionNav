"""Full-frame contrastive verifier for target-conditioned proposals.

GroundingDINO is intentionally queried with the target for recall.  A proposal
is not semantic proof: YOLO-World sees the same full frame with the complete
IntentionNav vocabulary, allowing visually competing labels to explain the
same region.  This module is optional and lazy-loaded so the established
reference agent remains unchanged unless explicitly enabled.
"""
from __future__ import annotations

import importlib.metadata
import os
import threading
from pathlib import Path
from typing import Iterable

import numpy as np

from evidence_nav import target_detector_queries, target_query_match
from object_vocabulary import BENCHMARK_VOCABULARY


MODEL_PATH = Path(os.environ.get(
    "INAV_YOLO_WORLD_CHECKPOINT",
    "/path/to/workspace/models/YOLOWorld/yolov8s-worldv2.pt",
))
MODEL_SHA256 = "9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792"
DEFAULT_CONFIDENCE = float(os.environ.get("INAV_YOLO_WORLD_CONFIDENCE", "0.05"))
DEFAULT_IMAGE_SIZE = int(os.environ.get("INAV_YOLO_WORLD_IMAGE_SIZE", "640"))
DEFAULT_OVERLAP = float(os.environ.get("INAV_YOLO_WORLD_OVERLAP", "0.5"))

_lock = threading.Lock()
_model = None
_device = None


def package_version() -> str | None:
    try:
        return importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        return None


def configured_model() -> dict:
    return {
        "checkpoint": str(MODEL_PATH),
        "checkpoint_sha256": MODEL_SHA256,
        "ultralytics_version": package_version(),
        "vocabulary_size": len(BENCHMARK_VOCABULARY),
    }


def _ensure_loaded() -> None:
    global _model, _device
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"YOLO-World checkpoint not found: {MODEL_PATH}")
        from ultralytics import YOLOWorld

        _device = os.environ.get(
            "INAV_YOLO_WORLD_DEVICE",
            os.environ.get("DINO_DEVICE", "cpu"),
        )
        model = YOLOWorld(str(MODEL_PATH))
        model.set_classes(list(BENCHMARK_VOCABULARY))
        _model = model


def detect(
    rgb: np.ndarray,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> list[dict]:
    """Detect the fixed benchmark vocabulary in one full RGB frame."""
    _ensure_loaded()
    result = _model.predict(
        source=rgb,
        conf=float(confidence),
        imgsz=int(image_size),
        device=_device,
        verbose=False,
    )[0]
    detections = []
    names = result.names
    for box in result.boxes:
        index = int(box.cls.item())
        label = names[index] if isinstance(names, dict) else names[index]
        detections.append({
            "label": str(label),
            "score": float(box.conf.item()),
            "bbox": [float(value) for value in box.xyxy[0].tolist()],
        })
    detections.sort(key=lambda item: (-item["score"], item["label"]))
    return detections


def target_proposals(
    rgb: np.ndarray,
    target: str,
    *,
    confidence: float = 0.25,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> tuple[list[tuple[str, float, list[float]]], list[dict]]:
    """Return direct target boxes from a full-vocabulary detector pass.

    This is intentionally stricter than target-conditioned detection: the
    model must select the requested category while competing against the
    complete benchmark vocabulary.  Component labels are not promoted here;
    doing so would make a common object such as ``cup`` sufficient evidence
    for the composite ``tea set`` category.
    """
    queries = target_detector_queries(target)
    rows = [
        detection for detection in detect(
            rgb,
            confidence=float(confidence),
            image_size=int(image_size),
        )
        if target_query_match(detection["label"], queries)
    ]
    proposals = [
        (
            str(target),
            float(detection["score"]),
            [float(value) for value in detection["bbox"]],
        )
        for detection in rows
    ]
    metadata = [
        {
            "available": True,
            "accepted": True,
            "decision": "target",
            "best_label": detection["label"],
            "score": round(float(detection["score"]), 6),
            "full_vocabulary_competition": True,
            "model": configured_model(),
        }
        for detection in rows
    ]
    return proposals, metadata


def _intersection_over_min_area(
    first: Iterable[float], second: Iterable[float]
) -> float:
    """Overlap coefficient robust to detector boxes at different scales."""
    a = list(first)
    b = list(second)
    if len(a) != 4 or len(b) != 4:
        return 0.0
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(a[2]) - float(a[0])) * max(
        0.0, float(a[3]) - float(a[1])
    )
    area_b = max(0.0, float(b[2]) - float(b[0])) * max(
        0.0, float(b[3]) - float(b[1])
    )
    denominator = min(area_a, area_b)
    return float(intersection / denominator) if denominator > 0.0 else 0.0


def contrast_proposal(
    proposal: tuple[str, float, list[float]],
    detections: list[dict],
    target: str,
    *,
    min_overlap: float = DEFAULT_OVERLAP,
) -> dict:
    """Classify a DINO proposal using spatially competing YOLO labels.

    The winning label maximizes detector confidence times spatial agreement.
    A target is positive only when it wins this local competition.  A sibling
    winner is explicit negative evidence; no associated detection is unknown.
    """
    _, proposal_score, proposal_bbox = proposal
    queries = target_detector_queries(target)
    associated = []
    for detection in detections:
        overlap = _intersection_over_min_area(proposal_bbox, detection["bbox"])
        if overlap < float(min_overlap):
            continue
        associated.append({
            **detection,
            "overlap": round(overlap, 6),
            "association_score": round(
                float(detection["score"]) * overlap, 6
            ),
        })
    associated.sort(
        key=lambda item: (
            -item["association_score"], -item["score"], item["label"]
        )
    )
    winner = associated[0] if associated else None
    positive = bool(
        winner and target_query_match(winner["label"], queries)
    )
    return {
        "available": True,
        "accepted": positive,
        "decision": (
            "target" if positive else "sibling" if winner else "unknown"
        ),
        "target": target,
        "proposal_score": round(float(proposal_score), 6),
        "proposal_bbox": [round(float(value), 2) for value in proposal_bbox],
        "winning_label": winner["label"] if winner else None,
        "winning_score": round(float(winner["score"]), 6) if winner else None,
        "winning_overlap": winner["overlap"] if winner else None,
        "associated": associated[:5],
        "model": configured_model(),
    }


def supports_target(verification: dict | None) -> bool:
    """Return whether YOLO-World independently supports a DINO proposal.

    Keep this predicate deliberately exact.  The target-conditioned detector
    supplies recall, while the full-vocabulary detector must both overlap the
    same image region and choose the requested target over every competing
    benchmark label.  ``unknown`` is not negative evidence and a sibling
    winner is never promoted.
    """
    return bool(
        verification
        and verification.get("available") is True
        and verification.get("accepted") is True
        and verification.get("decision") == "target"
        and verification.get("winning_label")
    )


def verify_proposals(
    rgb: np.ndarray,
    proposals: list[tuple[str, float, list[float]]],
    target: str,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    min_overlap: float = DEFAULT_OVERLAP,
) -> list[dict]:
    """Run one full-frame pass and contrast every DINO proposal."""
    if not proposals:
        return []
    try:
        detections = detect(rgb, confidence=float(confidence))
        return [
            contrast_proposal(
                proposal,
                detections,
                target,
                min_overlap=float(min_overlap),
            )
            for proposal in proposals
        ]
    except Exception as exc:
        return [{
            "available": False,
            "accepted": False,
            "decision": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "model": configured_model(),
        } for _ in proposals]
