"""Local CLIP crop verifier used as an independent semantic STOP signal.

GroundingDINO answers *where* a text phrase might be, but target-only prompts
can assign that phrase to many visually unrelated boxes.  CLIP re-ranks each
crop against the complete fixed IntentionNav vocabulary, providing a second
model family and explicit negative classes without a hosted API.
"""
from __future__ import annotations

import os
import threading
from typing import Iterable

import numpy as np
from PIL import Image

from object_vocabulary import BENCHMARK_VOCABULARY


MODEL_ID = os.environ.get(
    "INAV_CLIP_MODEL", "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
)
MODEL_REVISION = os.environ.get(
    "INAV_CLIP_REVISION", "1a25a446712ba5ee05982a381eed697ef9b435cf"
)
MAX_RANK = int(os.environ.get("INAV_CLIP_MAX_RANK", "5"))

# The benchmark vocabulary contains only valid goal categories.  Without an
# explicit reject option, CLIP must name every crop as one of those categories,
# even when GroundingDINO has proposed a wall texture or a clipped furniture
# fragment.  These generic structural/background concepts calibrate that
# closed world without encoding any scene, episode, or target-specific rule.
BACKGROUND_VOCABULARY = (
    "background",
    "empty scene",
    "wall surface",
    "floor surface",
    "ceiling surface",
    "door frame",
    "window frame",
    "furniture part",
    "object fragment",
    "unrecognizable object",
)
BACKGROUND_CALIBRATION = (
    os.environ.get("INAV_CLIP_BACKGROUND_CALIBRATION", "0") == "1"
)

# A crop of a composite object may be dominated by one of its visible parts.
# Keep this deliberately small and auditable: these are synonym/component
# relations, not scene- or episode-specific exceptions.
SEMANTIC_COMPONENTS = {
    "dining table": {"table"},
    "tea set": {"cup", "kettle"},
    "wine set": {"cup"},
}

_lock = threading.Lock()
_processor = None
_model = None
_device = None
_text_cache: dict[tuple[str, ...], object] = {}
_raw_text_cache: dict[tuple[str, ...], object] = {}

# Closed competition set for whole-frame scene-context scoring.  These are
# generic room types, not simulator room labels.  They are combined with the
# released benchmark vocabulary and fixed category-to-room/support priors so
# the policy can ask whether the current RGB is useful for finding a target
# without reading scene metadata or the target pose.
SCENE_CONTEXT_ROOMS = (
    "bathroom", "bedroom", "dining room", "kitchen", "living room",
    "study room", "balcony", "hallway",
)

# Fixed prompt ensemble reported by UIAP-OGN (ECMR 2025).  Every template is
# applied to every competing category, so the resulting target rank remains a
# closed-set, calibration-free score rather than a raw prompt-dependent cosine.
SCENE_CONTEXT_PROMPT_TEMPLATES = (
    "Seems like there is a {label} ahead",
    "A place where {label} can be found",
    "A {label} can be in the vicinity",
    "Seems like a {label} is ahead",
    "A {label} is in the vicinity",
    "{label} likely ahead",
    "{label}",
)


def _normalize_label(label: str) -> str:
    return " ".join(str(label or "").lower().replace("_", " ").split())


def verification_labels(target: str) -> tuple[str, ...]:
    """Return the fixed category competition set for a target crop."""
    labels = list(BENCHMARK_VOCABULARY)
    target_label = _normalize_label(target)
    if target_label and target_label not in labels:
        labels.append(target_label)
    if BACKGROUND_CALIBRATION:
        labels.extend(
            label for label in BACKGROUND_VOCABULARY if label not in labels
        )
    return tuple(labels)


def supports_target(
    verification: dict,
    target: str,
    component_max_rank: int = 3,
) -> bool:
    """Return whether CLIP positively supports the requested target.

    Exact targets must be CLIP's top-ranked benchmark class.  A small set of
    composite/synonym categories may instead be represented by an explicitly
    listed component, but the requested category must still rank near the top.
    """
    if not verification.get("available", False):
        return False
    target_label = _normalize_label(target)
    best_label = _normalize_label(verification.get("best_label", ""))
    target_rank = int(verification.get("target_rank", 10_000))
    if best_label == target_label and target_rank == 1:
        return True
    return (
        best_label in SEMANTIC_COMPONENTS.get(target_label, set())
        and target_rank <= int(component_max_rank)
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
            "CLIP_DEVICE",
            os.environ.get("DINO_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"),
        )
        _processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION
        )
        # Safetensors avoids torch.load compatibility/security restrictions
        # in the pinned Isaac/PyTorch environment.
        _model = AutoModel.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, use_safetensors=True
        ).to(_device).eval()


def _crop(rgb: np.ndarray, bbox: Iterable[float], padding_fraction: float = 0.08) -> Image.Image | None:
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
    is_siglip = "siglip" in MODEL_ID.lower()
    prompts = (
        [f"This is a photo of {label}." for label in labels]
        if is_siglip else
        [f"a photo of a {label}" for label in labels]
    )
    inputs = _processor(
        text=prompts,
        padding="max_length" if is_siglip else True,
        return_tensors="pt",
    ).to(_device)
    with torch.no_grad():
        features = _model.get_text_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        features = features / features.norm(dim=-1, keepdim=True)
    _text_cache[labels] = features
    return features


def _raw_text_features(prompts: tuple[str, ...]):
    """Encode already-formed prompt strings with a dedicated cache."""
    import torch

    cached = _raw_text_cache.get(prompts)
    if cached is not None:
        return cached
    inputs = _processor(
        text=list(prompts),
        padding="max_length" if "siglip" in MODEL_ID.lower() else True,
        return_tensors="pt",
    ).to(_device)
    with torch.no_grad():
        features = _model.get_text_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        features = features / features.norm(dim=-1, keepdim=True)
    _raw_text_cache[prompts] = features
    return features


def scene_context_labels(
    target: str,
    likely_rooms: Iterable[str] = (),
    support_labels: Iterable[str] = (),
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Build a fixed closed-set competition for scene-level relevance.

    The positive set contains only the requested category and episode-
    independent common-sense relations supplied by the caller.  All other
    benchmark categories and generic room types remain negatives.  Returning
    the source groups separately keeps target, support, and room contributions
    auditable in rollout records.
    """
    target_label = _normalize_label(target)
    support = tuple(dict.fromkeys(
        label for label in (_normalize_label(item) for item in support_labels)
        if label and label != target_label
    ))
    rooms = tuple(dict.fromkeys(
        label for label in (_normalize_label(item) for item in likely_rooms)
        if label
    ))
    positives = tuple(dict.fromkeys(
        label for label in (target_label, *support, *rooms) if label
    ))
    labels = tuple(dict.fromkeys((
        *BENCHMARK_VOCABULARY,
        *SCENE_CONTEXT_ROOMS,
        *positives,
        *BACKGROUND_VOCABULARY,
    )))
    groups = {
        "target": (target_label,) if target_label else (),
        "support": support,
        "room": rooms,
    }
    return labels, positives, groups


def _rank_relevance(
    order: list[int],
    labels: tuple[str, ...],
    positives: tuple[str, ...],
) -> tuple[float, int | None, str | None]:
    """Map the best positive rank to [0, 1] without a fitted threshold."""
    positive_set = set(positives)
    for zero_rank, index in enumerate(order):
        if labels[index] in positive_set:
            denominator = max(1, len(labels) - 1)
            return 1.0 - zero_rank / denominator, zero_rank + 1, labels[index]
    return 0.0, None, None


def score_scene_context(
    rgb: np.ndarray,
    target: str,
    likely_rooms: Iterable[str] = (),
    support_labels: Iterable[str] = (),
) -> dict:
    """Score whether a full RGB frame is useful for target search.

    This is a policy-side retrieval score, never target evidence and never a
    STOP signal.  It ranks the target and its fixed room/support relations in
    a closed set containing every benchmark category, generic room types, and
    background concepts.  Rank percentile is more stable across scenes than a
    raw cosine threshold and needs no benchmark-specific calibration.
    """
    try:
        _ensure_loaded()
        import torch

        labels, positives, groups = scene_context_labels(
            target, likely_rooms=likely_rooms, support_labels=support_labels
        )
        if not positives:
            return {"available": False, "reason": "no_positive_labels"}
        image = Image.fromarray(rgb)
        image_inputs = _processor(images=[image], return_tensors="pt").to(
            _device
        )
        with torch.no_grad():
            image_features = _model.get_image_features(**image_inputs)
            if not isinstance(image_features, torch.Tensor):
                image_features = image_features.pooler_output
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )
            scores = (image_features @ _text_features(labels).T)[0]
        order_tensor = torch.argsort(scores, descending=True)
        order = [int(index) for index in order_tensor.tolist()]
        relevance, best_rank, best_positive = _rank_relevance(
            order, labels, positives
        )
        group_results = {}
        for name, group in groups.items():
            group_relevance, group_rank, group_label = _rank_relevance(
                order, labels, group
            )
            group_results[name] = {
                "relevance": round(float(group_relevance), 6),
                "best_rank": group_rank,
                "best_label": group_label,
            }
        return {
            "available": True,
            "relevance": round(float(relevance), 6),
            "best_positive_rank": best_rank,
            "best_positive_label": best_positive,
            "positive_labels": list(positives),
            "groups": group_results,
            "top5": [
                {
                    "label": labels[index],
                    "score": round(float(scores[index].item()), 4),
                }
                for index in order[:5]
            ],
            "competition_size": len(labels),
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def score_scene_context_prompt_ensemble(
    rgb: np.ndarray,
    target: str,
) -> dict:
    """Return a prompt-ensemble distribution of target rank relevance.

    The seven fixed prompt forms are applied symmetrically to every benchmark,
    room, and background label.  The output is a distribution over target rank
    percentiles in [0, 1], suitable for a probabilistic semantic map.  It is a
    search-only signal and cannot authorize target memory or STOP.
    """
    try:
        _ensure_loaded()
        import torch

        labels, positives, _ = scene_context_labels(target)
        if len(positives) != 1:
            return {"available": False, "reason": "target_not_unique"}
        target_label = positives[0]
        target_index = labels.index(target_label)
        prompts = tuple(
            template.format(label=label)
            for template in SCENE_CONTEXT_PROMPT_TEMPLATES
            for label in labels
        )
        image_inputs = _processor(
            images=[Image.fromarray(rgb)], return_tensors="pt"
        ).to(_device)
        with torch.no_grad():
            image_features = _model.get_image_features(**image_inputs)
            if not isinstance(image_features, torch.Tensor):
                image_features = image_features.pooler_output
            image_features = image_features / image_features.norm(
                dim=-1, keepdim=True
            )
            scores = (image_features @ _raw_text_features(prompts).T)[0]
        scores = scores.reshape(len(SCENE_CONTEXT_PROMPT_TEMPLATES), len(labels))
        denominator = max(1, len(labels) - 1)
        samples = []
        for template, row in zip(SCENE_CONTEXT_PROMPT_TEMPLATES, scores):
            order = torch.argsort(row, descending=True)
            zero_rank = int(
                (order == target_index).nonzero(as_tuple=False)[0].item()
            )
            samples.append({
                "template": template,
                "rank": zero_rank + 1,
                "relevance": 1.0 - zero_rank / denominator,
                "similarity": float(row[target_index].item()),
            })
        relevance = np.asarray(
            [sample["relevance"] for sample in samples], dtype=np.float64
        )
        mean_scores = scores.mean(dim=0)
        mean_order = torch.argsort(mean_scores, descending=True)
        return {
            "available": True,
            "relevance": round(float(relevance.mean()), 6),
            "relevance_mean": round(float(relevance.mean()), 6),
            "relevance_variance": round(float(relevance.var()), 8),
            "relevance_std": round(float(relevance.std()), 6),
            "samples": [
                {
                    **sample,
                    "relevance": round(float(sample["relevance"]), 6),
                    "similarity": round(float(sample["similarity"]), 6),
                }
                for sample in samples
            ],
            "ensemble_size": len(samples),
            "competition_size": len(labels),
            "top5_mean": [
                {
                    "label": labels[int(index)],
                    "score": round(float(mean_scores[int(index)].item()), 4),
                }
                for index in mean_order[:5]
            ],
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def verify_detections(
    rgb: np.ndarray,
    detections: list[tuple[str, float, list[float]]],
    target: str,
    max_detections: int = 5,
    max_rank: int = MAX_RANK,
) -> list[dict]:
    """CLIP-rank up to ``max_detections`` boxes; never raises to the agent."""
    if not detections:
        return []
    try:
        _ensure_loaded()
        import torch

        target_label = _normalize_label(target)
        labels_tuple = verification_labels(target_label)
        labels = list(labels_tuple)
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
            top5 = [
                {"label": labels[int(index)],
                 "score": round(float(scores[int(index)].item()), 4)}
                for index in order[:5]
            ]
            output[detection_index] = {
                "available": True,
                "accepted": rank <= max_rank,
                "target_rank": rank,
                "target_similarity": round(target_score, 4),
                "margin_to_best_other": round(target_score - best_other, 4),
                "best_label": labels[best_index],
                "top5": top5,
                "model": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "background_calibration": BACKGROUND_CALIBRATION,
                "background_winner": (
                    labels[best_index] in BACKGROUND_VOCABULARY
                ),
            }
        return output
    except Exception as exc:
        return [
            {"available": False, "accepted": False,
             "error": f"{type(exc).__name__}: {exc}"}
            for _ in detections[:max_detections]
        ]
