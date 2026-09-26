"""Refine rejected DINO regions with OWLv2 and fixed SigLIP crop stability.

The DINO score remains the original region's score, not an OWLv2 probability
or a newly calibrated score for the refined box. Full provenance is retained.
"""
from __future__ import annotations

import os
import threading
from typing import Callable

import numpy as np
from PIL import Image

import ensemble_verifier
import siglip_verifier
from object_vocabulary import BENCHMARK_VOCABULARY


MODEL_ID = "google/owlv2-base-patch16-ensemble"
MODEL_REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"
MIN_ANCHOR_IOU = 0.5
VARIANTS = {
    "original": (0., 0., 1.), "left2": (-2., 0., 1.), "right2": (2., 0., 1.),
    "up2": (0., -2., 1.), "down2": (0., 2., 1.),
    "shrink5": (0., 0., .95), "expand5": (0., 0., 1.05),
}
Detection = tuple[str, float, list[float]]
_model = None
_processor = None
_device = None
_lock = threading.Lock()


def box_iou(left: list[float], right: list[float]) -> float:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    intersection = float(np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod())
    union = float(np.maximum(0, a[2:] - a[:2]).prod() + np.maximum(0, b[2:] - b[:2]).prod() - intersection)
    return intersection / union if union > 0 else 0.


def perturbed_box(box: list[float], variant: str, height: int, width: int) -> list[float]:
    if variant == "original":
        return list(box)
    dx, dy, scale = VARIANTS[variant]
    value = np.asarray(box, dtype=float)
    center = (value[:2] + value[2:]) / 2 + [dx, dy]
    half = (value[2:] - value[:2]) * scale / 2
    changed = np.clip(np.r_[center - half, center + half], 0, [width, height, width, height])
    if not np.isfinite(changed).all() or np.any(changed[2:] <= changed[:2]):
        raise ValueError("Invalid refinement crop geometry")
    return changed.tolist()


def proposals_from_raw(scores: np.ndarray, boxes: np.ndarray, target: str) -> list[dict]:
    """Same 64-way winner, >.1 score, stable .5 NMS and five-box cap as R078."""
    category = " ".join(target.lower().replace("_", " ").split())
    index = list(BENCHMARK_VOCABULARY).index(category)
    if scores.ndim != 2 or scores.shape[1] != 64 or boxes.shape != (len(scores), 4):
        raise ValueError("Unexpected OWLv2 prediction shapes")
    if not np.isfinite(scores).all() or not np.isfinite(boxes).all():
        raise ValueError("Nonfinite OWLv2 prediction")
    valid = (scores[:, index] > .1) & (scores.argmax(axis=1) == index)
    valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    order = sorted(np.flatnonzero(valid).tolist(), key=lambda i: (-float(scores[i, index]), i))
    selected = []
    for i in order:
        if all(box_iou(boxes[i], boxes[j]) <= .5 for j in selected):
            selected.append(i)
        if len(selected) == 5:
            break
    return [{"label": target, "score": float(scores[i, index]), "bbox": boxes[i].tolist(),
             "patch_index": i} for i in selected]


def detect(rgb: np.ndarray, target: str, *, raw_callback: Callable | None = None) -> list[dict]:
    """Run the pinned public detector on RGB only; no simulator or GT access."""
    global _model, _processor, _device
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    if rgb.shape != (768, 768, 3) or rgb.dtype != np.uint8:
        raise ValueError("R080 expects the fixed 768-square RGB policy input")
    if _model is None:
        with _lock:
            if _model is None:
                _device = os.environ.get("DINO_DEVICE", "cuda:0")
                _processor = Owlv2Processor.from_pretrained(
                    MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
                _model = Owlv2ForObjectDetection.from_pretrained(
                    MODEL_ID, revision=MODEL_REVISION, local_files_only=True,
                    use_safetensors=True, dtype=torch.float32,
                    attn_implementation="sdpa").eval().to(_device)
    prompts = [f"a photo of a {category}" for category in BENCHMARK_VOCABULARY]
    inputs = _processor(text=[prompts], images=Image.fromarray(rgb), return_tensors="pt").to(_device)
    with torch.cuda.device(_device), torch.inference_mode():
        prediction = _model(**inputs)
    scores = prediction.logits[0].sigmoid().cpu().numpy()
    centers = prediction.pred_boxes[0].cpu().numpy()
    boxes = np.concatenate((centers[:, :2] - centers[:, 2:] / 2,
                            centers[:, :2] + centers[:, 2:] / 2), axis=1)
    boxes = (boxes * np.array([768] * 4, dtype=boxes.dtype)).clip(0, 768)
    if raw_callback is not None:
        raw_callback({"class_scores": scores, "normalized_center_boxes": centers,
                      "image_xyxy": boxes,
                      "objectness": prediction.objectness_logits[0].sigmoid().cpu().numpy()})
    return proposals_from_raw(scores, boxes, target)


def select_refinements(
    ordinary: list[Detection], proposals: list[dict],
    verifications: dict[str, list[dict]], target: str,
) -> tuple[list[tuple[Detection, dict]], list[dict]]:
    """Associate stable refined boxes to original regions without score fusion."""
    if list(verifications) != list(VARIANTS):
        raise ValueError("Every prescribed stability variant is required")
    if any(len(values) != len(proposals) for values in verifications.values()):
        raise ValueError("Incomplete refinement verification batch")
    accepted, records = [], []
    for i, proposal in enumerate(proposals):
        variants = {name: values[i] for name, values in verifications.items()}
        if any(not value.get("available") or any(not value.get(name, {}).get("available")
               or value.get(name, {}).get("error") for name in ("clip_b32", "siglip2"))
               for value in variants.values()):
            raise RuntimeError("Required refinement verification failed")
        support = {name: siglip_verifier.supports_target(value["siglip2"], target)
                   for name, value in variants.items()}
        overlaps = [box_iou(proposal["bbox"], d[2]) for d in ordinary]
        anchor_index = max(range(len(ordinary)), key=lambda j: (overlaps[j], -j)) if ordinary else None
        overlap = overlaps[anchor_index] if anchor_index is not None else 0.
        stable, anchored = all(support.values()), overlap >= MIN_ANCHOR_IOU
        metadata = {"owl_index": i, "owl_proposal": proposal, "anchor_index": anchor_index,
                    "anchor_iou": overlap, "anchor_iou_required": MIN_ANCHOR_IOU,
                    "siglip_support_by_variant": support, "stable_siglip_support": stable,
                    "accepted": stable and anchored, "verifications": variants,
                    "reason": "accepted" if stable and anchored else "unstable_semantics" if not stable else "no_dino_anchor"}
        if anchor_index is not None:
            label, score, bbox = ordinary[anchor_index]
            metadata["anchor"] = {"label": label, "score": float(score), "bbox": list(bbox)}
        if stable and anchored:
            provenance = {key: value for key, value in metadata.items() if key != "verifications"}
            provenance["score_semantics"] = "original_dino_region_score_not_refined_box_probability"
            refined = (label, float(score), list(proposal["bbox"]))
            value = {**variants["original"], "proposal_refinement": provenance}
            if not ensemble_verifier.supports_target(value, target):
                raise RuntimeError("Stable SigLIP support did not reach the ordinary semantic gate")
            accepted.append((refined, value))
        records.append(metadata)
    return accepted, records


def refine_rejected(
    rgb: np.ndarray, ordinary: list[Detection], ordinary_verifications: list[dict],
    target: str, *, detector: Callable = detect, verifier: Callable | None = None,
) -> tuple[list[tuple[Detection, dict]], dict]:
    """Preserve ordinary successes; refine only already-verified rejected regions."""
    if any(ensemble_verifier.supports_target(value, target) for value in ordinary_verifications):
        return [], {"called": False, "reason": "ordinary_supported", "proposals": []}
    # The baseline verifies at most five candidates. Never treat an unverified
    # later candidate as an eligible anchor or change that baseline batching.
    anchors = ordinary[:len(ordinary_verifications)]
    if not anchors:
        return [], {"called": False, "reason": "no_verified_dino_regions", "proposals": []}
    if any(not value.get("available") for value in ordinary_verifications):
        return [], {"called": False, "reason": "ordinary_verification_unavailable", "proposals": []}
    if not ensemble_verifier.ENABLED or siglip_verifier.MIN_MARGIN != .01:
        raise RuntimeError("R080 requires the unchanged base SigLIP ensemble gate")
    proposals = detector(rgb, target)
    verifier = verifier or ensemble_verifier.verify_detections
    values = {}
    for variant in VARIANTS:
        detections = [(d["label"], d["score"], perturbed_box(d["bbox"], variant, *rgb.shape[:2]))
                      for d in proposals]
        values[variant] = verifier(rgb, detections, target, max_detections=5) if detections else []
    accepted, records = select_refinements(anchors, proposals, values, target)
    return accepted, {"called": True, "reason": "refined" if accepted else "no_accepted_refinement",
                      "proposals": records, "accepted_count": len(accepted),
                      "anchor_count": len(anchors), "variant_count": len(VARIANTS)}
