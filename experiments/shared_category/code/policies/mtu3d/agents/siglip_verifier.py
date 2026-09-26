"""Pinned SigLIP2 crop verifier for fine-grained local object semantics."""
from __future__ import annotations

import os
import threading
from typing import Iterable

import numpy as np
from PIL import Image

from object_vocabulary import BENCHMARK_VOCABULARY


MODEL_ID = os.environ.get(
    "INAV_SIGLIP_MODEL", "google/siglip2-base-patch16-384"
)
MODEL_REVISION = os.environ.get(
    "INAV_SIGLIP_REVISION", "f775b65a79762255128c981547af89addcfe0f88"
)
MIN_MARGIN = float(os.environ.get("INAV_SIGLIP_MIN_MARGIN", "0.01"))

_lock = threading.Lock()
_processor = None
_model = None
_device = None
_text_cache: dict[tuple[str, ...], object] = {}


def _normalize_label(label: str) -> str:
    return " ".join(str(label or "").lower().replace("_", " ").split())


def supports_target(verification: dict, target: str) -> bool:
    if not verification.get("available", False):
        return False
    return (
        _normalize_label(verification.get("best_label", ""))
        == _normalize_label(target)
        and int(verification.get("target_rank", 10_000)) == 1
        and float(verification.get("margin_to_best_other", float("-inf")))
        >= MIN_MARGIN
    )


def _ensure_loaded() -> None:
    global _processor, _model, _device
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        _device = os.environ.get(
            "SIGLIP_DEVICE",
            os.environ.get(
                "CLIP_DEVICE",
                os.environ.get(
                    "DINO_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"
                ),
            ),
        )
        load_args = {
            "revision": MODEL_REVISION,
            "local_files_only": os.environ.get("HF_HUB_OFFLINE", "0") == "1",
        }
        _processor = AutoProcessor.from_pretrained(MODEL_ID, **load_args)
        _model = AutoModel.from_pretrained(
            MODEL_ID, use_safetensors=True, **load_args
        ).to(_device).eval()


def _crop(
    rgb: np.ndarray,
    bbox: Iterable[float],
    padding_fraction: float = 0.08,
) -> Image.Image | None:
    values = list(bbox)
    if len(values) != 4:
        return None
    height, width = rgb.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in values)
    pad_x = max(2.0, (x2 - x1) * padding_fraction)
    pad_y = max(2.0, (y2 - y1) * padding_fraction)
    left = max(0, min(width - 1, int(x1 - pad_x)))
    right = max(left + 1, min(width, int(x2 + pad_x)))
    top = max(0, min(height - 1, int(y1 - pad_y)))
    bottom = max(top + 1, min(height, int(y2 + pad_y)))
    if right - left < 3 or bottom - top < 3:
        return None
    return Image.fromarray(rgb[top:bottom, left:right])


def _text_features(labels: tuple[str, ...]):
    import torch

    cached = _text_cache.get(labels)
    if cached is not None:
        return cached
    prompts = [f"This is a photo of {label}." for label in labels]
    inputs = _processor(
        text=prompts,
        padding="max_length",
        return_tensors="pt",
    ).to(_device)
    with torch.no_grad():
        features = _model.get_text_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        features = features / features.norm(dim=-1, keepdim=True)
    _text_cache[labels] = features
    return features


def verify_detections(
    rgb: np.ndarray,
    detections: list[tuple[str, float, list[float]]],
    target: str,
    max_detections: int = 5,
    max_rank: int = 3,
) -> list[dict]:
    """Rank detector crops against the complete fixed benchmark vocabulary."""
    del max_rank  # SigLIP support is deliberately top-1 plus a global margin.
    if not detections:
        return []
    try:
        _ensure_loaded()
        import torch

        target_label = _normalize_label(target)
        labels = list(BENCHMARK_VOCABULARY)
        if target_label not in labels:
            labels.append(target_label)
        labels_tuple = tuple(labels)
        target_index = labels.index(target_label)
        selected = detections[:max_detections]
        crops = [_crop(rgb, detection[2]) for detection in selected]
        valid_indices = [index for index, crop in enumerate(crops) if crop is not None]
        output = [{"available": False, "accepted": False} for _ in selected]
        if not valid_indices:
            return output
        image_inputs = _processor(
            images=[crops[index] for index in valid_indices],
            return_tensors="pt",
        ).to(_device)
        with torch.no_grad():
            image_features = _model.get_image_features(**image_inputs)
            if not isinstance(image_features, torch.Tensor):
                image_features = image_features.pooler_output
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = image_features @ _text_features(labels_tuple).T

        for row, detection_index in enumerate(valid_indices):
            scores = similarities[row]
            order = torch.argsort(scores, descending=True)
            rank = int((order == target_index).nonzero(as_tuple=False)[0].item()) + 1
            target_score = float(scores[target_index].item())
            best_index = int(order[0].item())
            best_other = max(
                float(scores[index].item())
                for index in range(len(labels)) if index != target_index
            )
            result = {
                "available": True,
                "target_rank": rank,
                "target_similarity": round(target_score, 4),
                "margin_to_best_other": round(target_score - best_other, 4),
                "best_label": labels[best_index],
                "top5": [
                    {
                        "label": labels[int(index)],
                        "score": round(float(scores[int(index)].item()), 4),
                    }
                    for index in order[:5]
                ],
                "model": MODEL_ID,
                "revision": MODEL_REVISION,
            }
            result["accepted"] = supports_target(result, target_label)
            output[detection_index] = result
        return output
    except Exception as exc:
        return [
            {
                "available": False,
                "accepted": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            for _ in detections[:max_detections]
        ]
