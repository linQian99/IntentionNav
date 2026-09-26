"""GroundingDINO open-vocab detector for last-mile target verification.

Wraps HuggingFace `IDEA-Research/grounding-dino-tiny`. Lazy-loaded singleton:
the first call pays ~13s model load + ~0.6s inference; subsequent calls
~0.1-0.3s on a 3090.

Used in agent_vlm.py to detect target_guess + candidate_objects in the
current RGB frame, get bbox center, look up depth at that pixel for
target-grounded Force-STOP. Replaces VLM's `target_loc` 9-grid pointer
(unreliable on small objects).
"""
from __future__ import annotations

import threading
from typing import List, Tuple

_lock = threading.Lock()
_proc = None
_model = None
_device = None


def _ensure_loaded():
    """Lazy-load model on first call. Cheap on subsequent calls."""
    global _proc, _model, _device
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        model_id = "IDEA-Research/grounding-dino-tiny"
        _proc = AutoProcessor.from_pretrained(model_id)
        # Pin to a specific cuda index, not the bare string "cuda".
        # Isaac Sim's replicator/RTX path can change torch.cuda's *current*
        # device between detect() calls. If `_device == "cuda"`, then on
        # the next call `inputs.to(_device)` lands on whatever cuda is
        # current, while the model parameters still live on the device
        # we initially loaded onto. Result: cross-device tensor errors
        # mid-episode. Pinning to cuda:0 (or DINO_DEVICE override) keeps
        # both sides consistent regardless of Isaac Sim activity.
        if torch.cuda.is_available():
            import os as _os
            _device = _os.environ.get("DINO_DEVICE", "cuda:0")
        else:
            _device = "cpu"
        _model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(_device).eval()


def detect(rgb_np, query_phrases: List[str],
           threshold: float = 0.30,
           text_threshold: float = 0.20
           ) -> List[Tuple[str, float, List[float]]]:
    """Run zero-shot detection on RGB frame.

    Thresholds match OpenFMNav defaults (box 0.30 / text 0.20). Text
    format follows GroundingDINO original convention: bare nouns,
    period-joined, no "a " prefix (e.g. "plate.bed.sofa.").

    Args:
      rgb_np: (H, W, 3) uint8 numpy array.
      query_phrases: list of candidate labels.
      threshold: bbox score threshold.
      text_threshold: text-image alignment threshold.

    Returns:
      List of (label_str, score, [x1, y1, x2, y2]) sorted by score desc.
      Empty list if no detections or on error (caller should not crash).
    """
    if not query_phrases:
        return []
    try:
        _ensure_loaded()
        import torch
        from PIL import Image
        img = Image.fromarray(rgb_np)
        # Strip leading "a " / "an " if caller included it; tokenize as
        # bare lowercase nouns joined with periods (OpenFMNav-compatible).
        cleaned = []
        for p in query_phrases:
            s = p.strip().lower()
            for prefix in ("a ", "an "):
                if s.startswith(prefix):
                    s = s[len(prefix):]
            s = s.replace("_", " ").strip(" .")
            if s and s not in cleaned:
                cleaned.append(s)
        if not cleaned:
            return []
        text = ".".join(cleaned) + "."
        # Belt-and-suspenders: pin torch's CURRENT cuda device to _device
        # for the entire forward pass. Without this, model internal ops
        # that allocate transient tensors via the current-device default
        # (e.g. index buffers inside attention) can land on whatever cuda
        # Isaac Sim last set, while parameter tensors stay on _device.
        if isinstance(_device, str) and _device.startswith("cuda"):
            cuda_ctx = torch.cuda.device(_device)
        else:
            from contextlib import nullcontext
            cuda_ctx = nullcontext()
        with cuda_ctx, torch.no_grad():
            inputs = _proc(images=img, text=text, return_tensors="pt").to(_device)
            out = _model(**inputs)
        res = _proc.post_process_grounded_object_detection(
            out, threshold=threshold, text_threshold=text_threshold,
            target_sizes=[img.size[::-1]],
        )[0]
        detections = []
        for box, score, label in zip(res["boxes"], res["scores"], res["text_labels"]):
            b = box.tolist()
            detections.append((str(label), float(score.item()), b))
        detections.sort(key=lambda x: -x[1])
        return detections
    except Exception as e:
        import sys
        print(f"[dino] detection error: {e}", file=sys.stderr)
        return []


def find_best_match(detections: List[Tuple[str, float, List[float]]],
                     target_terms: List[str]
                     ) -> Tuple[str, float, List[float]] | None:
    """Pick the top detection whose label contains any of target_terms.
    target_terms are case-insensitive substrings to match label against.
    Returns None if no match."""
    if not detections or not target_terms:
        return None
    terms = [t.strip().lower() for t in target_terms if t and t.strip()]
    for label, score, bbox in detections:  # sorted by score desc
        ll = label.lower()
        if any(t in ll for t in terms):
            return (label, score, bbox)
    return None
