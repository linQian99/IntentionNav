"""Frozen COCO-category specialist used for VLFM-style detector routing."""
from __future__ import annotations

import importlib.metadata
import hashlib
import os
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

from evidence_nav import normalize_label


MODEL_PATH = Path(os.environ.get(
    "INAV_COCO_CHECKPOINT",
    "/path/to/workspace/models/YOLOWorld/yolo11m.pt",
))
MODEL_SHA256 = "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95"
DEFAULT_CONFIDENCE = float(os.environ.get("INAV_COCO_CONFIDENCE", "0.25"))
DEFAULT_IMAGE_SIZE = int(os.environ.get("INAV_COCO_IMAGE_SIZE", "640"))
CONFIGURED_DEVICE = os.environ.get(
    "INAV_COCO_DEVICE", os.environ.get("DINO_DEVICE", "cpu")
)

# Only exact, auditable category correspondences are routed. Generic `table`,
# `toy`, `plate`, and composite sets are intentionally excluded.
TARGET_TO_COCO = {
    "basin": {"sink"},
    "bench": {"bench"},
    "book": {"book"},
    "bowl": {"bowl"},
    "builtin oven": {"oven"},
    "chair": {"chair"},
    "clock": {"clock"},
    "cup": {"cup"},
    "dining table": {"dining table"},
    "fork": {"fork"},
    "fridge": {"refrigerator"},
    "knife": {"knife"},
    "microwave": {"microwave"},
    "remote control": {"remote"},
    "sofa": {"couch"},
    "spoon": {"spoon"},
    "television": {"tv"},
    "toilet": {"toilet"},
    "vase": {"vase"},
}

_lock = threading.Lock()
_model = None
_device = None


def coco_labels(target: str) -> set[str]:
    return set(TARGET_TO_COCO.get(normalize_label(target), set()))


def supports_target() -> set[str]:
    return set(TARGET_TO_COCO)


@lru_cache(maxsize=4)
def checkpoint_sha256(path: str = str(MODEL_PATH)) -> str | None:
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configured_model() -> dict:
    try:
        version = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        version = None
    actual_sha256 = checkpoint_sha256(str(MODEL_PATH))
    return {
        "checkpoint": str(MODEL_PATH),
        "checkpoint_expected_sha256": MODEL_SHA256,
        "checkpoint_sha256": actual_sha256,
        "checkpoint_hash_matches": actual_sha256 == MODEL_SHA256,
        "ultralytics_version": version,
        "mapped_target_categories": len(TARGET_TO_COCO),
        "confidence": DEFAULT_CONFIDENCE,
        "image_size": DEFAULT_IMAGE_SIZE,
        "device": CONFIGURED_DEVICE,
    }


def _ensure_loaded() -> None:
    global _model, _device
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"COCO checkpoint not found: {MODEL_PATH}")
        actual_sha256 = checkpoint_sha256(str(MODEL_PATH))
        if actual_sha256 != MODEL_SHA256:
            raise RuntimeError(
                "COCO checkpoint hash mismatch: "
                f"expected {MODEL_SHA256}, got {actual_sha256}"
            )
        from ultralytics import YOLO

        _device = CONFIGURED_DEVICE
        _model = YOLO(str(MODEL_PATH))


def ultralytics_bgr_input(rgb: np.ndarray) -> np.ndarray:
    """Convert simulator RGB to the BGR NumPy contract used by Ultralytics."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected RGB HxWx3 image, got {image.shape}")
    return np.ascontiguousarray(image[..., ::-1])


def detect(
    rgb: np.ndarray,
    *,
    confidence: float = DEFAULT_CONFIDENCE,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> list[dict]:
    _ensure_loaded()
    result = _model.predict(
        source=ultralytics_bgr_input(rgb),
        conf=float(confidence),
        imgsz=int(image_size),
        device=_device,
        verbose=False,
    )[0]
    names = result.names
    detections = []
    for box in result.boxes:
        index = int(box.cls.item())
        label = names[index] if isinstance(names, dict) else names[index]
        detections.append({
            "label": normalize_label(label),
            "score": float(box.conf.item()),
            "bbox": [float(value) for value in box.xyxy[0].tolist()],
        })
    detections.sort(key=lambda item: (-item["score"], item["label"]))
    return detections


def target_detections(rgb: np.ndarray, target: str) -> list[dict]:
    labels = coco_labels(target)
    if not labels:
        return []
    return [item for item in detect(rgb) if item["label"] in labels]


def target_proposals(
    rgb: np.ndarray, target: str
) -> list[tuple[str, float, list[float]]]:
    """Return agent-compatible proposals labeled with the benchmark target."""
    target_label = normalize_label(target)
    return [
        (target_label, float(item["score"]), list(item["bbox"]))
        for item in target_detections(rgb, target_label)
    ]


def semantic_verification(target: str, score: float) -> dict:
    """Represent an exact COCO-class detection as independent semantics."""
    target_label = normalize_label(target)
    return {
        "available": True,
        "accepted": True,
        "target_rank": 1,
        "target_similarity": round(float(score), 4),
        "margin_to_best_other": 1.0,
        "best_label": target_label,
        "top5": [{"label": target_label, "score": round(float(score), 4)}],
        "model": str(MODEL_PATH),
        "source": "coco_specialist",
    }
