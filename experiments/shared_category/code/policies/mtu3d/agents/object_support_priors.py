"""Fixed category-to-support priors for small-object search.

These are episode-independent commonsense relations over categories already in
the released benchmark vocabulary.  They never encode a scene, room instance,
target pose, or benchmark episode.
"""
from __future__ import annotations

import math


SUPPORT_PRIORS = {
    "book": ("shelf", "desk", "night stand", "table"),
    "bowl": ("dining table", "table", "cabinet", "shelf"),
    "chopstick": ("dining table", "table", "cabinet"),
    "cosmetic": ("cabinet", "shelf", "night stand", "table", "basin"),
    "cup": ("dining table", "table", "cabinet", "shelf", "desk"),
    "flower": ("dining table", "table", "shelf", "cabinet", "desk"),
    "fork": ("dining table", "table", "cabinet"),
    "fruit": ("dining table", "table", "cabinet"),
    "kettle": ("table", "cabinet", "shelf"),
    "knife": ("dining table", "table", "cabinet", "shelf", "basin"),
    "menorah": ("dining table", "table", "shelf", "cabinet"),
    "plate": ("dining table", "table", "cabinet", "shelf"),
    "reed diffuser": ("shelf", "cabinet", "night stand", "table"),
    "remote control": ("sofa", "table", "night stand"),
    "sculpture": ("shelf", "cabinet", "table", "desk"),
    "spoon": ("dining table", "table", "cabinet"),
    "table lamp": ("night stand", "table", "desk", "cabinet"),
    "tea set": ("dining table", "table", "cabinet", "shelf"),
    "toy": ("shelf", "cabinet", "table", "sofa"),
    "vase": ("dining table", "table", "shelf", "cabinet", "desk"),
    "wine set": ("dining table", "table", "cabinet", "shelf"),
}


def support_labels_for_object(category: str) -> tuple[str, ...]:
    """Return deterministic support labels for a canonical target category."""
    normalized = " ".join(
        str(category or "").strip().lower().replace("_", " ").split()
    )
    return SUPPORT_PRIORS.get(normalized, ())


def support_detection_candidates(
    detections: list[dict],
    support_labels: tuple[str, ...],
) -> list[dict]:
    """Rank fixed-prior supports by relation rank and then confidence.

    The ordering is deliberately category-level and episode-independent.  A
    high-confidence generic support cannot displace a more specific support
    listed earlier in the prior, and detections outside the fixed list are
    ignored.
    """
    priority = {label: index for index, label in enumerate(support_labels)}
    eligible = []
    for detection in detections:
        label = " ".join(
            str(detection.get("label") or "")
            .strip().lower().replace("_", " ").split()
        )
        if label not in priority:
            continue
        try:
            score = float(detection.get("score", 0.0))
            bbox = [float(value) for value in detection.get("bbox", [])]
        except (TypeError, ValueError):
            continue
        if (
            len(bbox) != 4
            or not math.isfinite(score)
            or not all(math.isfinite(value) for value in bbox)
            or bbox[2] <= bbox[0]
            or bbox[3] <= bbox[1]
        ):
            continue
        eligible.append((priority[label], -score, label, bbox, detection))
    ranked = []
    ordered = sorted(
        eligible,
        key=lambda item: (
            item[0], item[1], item[2], tuple(item[3])
        ),
    )
    for rank, negative_score, label, bbox, original in ordered:
        ranked.append({
            **original,
            "label": label,
            "score": -negative_score,
            "bbox": bbox,
            "support_priority": int(rank),
        })
    return ranked


def select_support_detection(
    detections: list[dict],
    support_labels: tuple[str, ...],
) -> dict | None:
    """Select the highest-ranked valid support detection, if any."""
    ranked = support_detection_candidates(detections, support_labels)
    return ranked[0] if ranked else None


def support_candidate_is_new(
    candidate_xy: tuple[float, float] | list[float],
    rejected_xy: list[tuple[float, float] | list[float]],
    *,
    min_separation_m: float = 1.0,
) -> bool:
    """Reject a previously inspected physical support instance."""
    try:
        x, y = float(candidate_xy[0]), float(candidate_xy[1])
    except (TypeError, ValueError, IndexError):
        return False
    return all(
        math.hypot(x - float(old[0]), y - float(old[1]))
        >= float(min_separation_m)
        for old in rejected_xy
        if old is not None and len(old) >= 2
    )


def carrier_observation_outcome(
    *,
    target_hit: bool,
    actions_taken: int,
    action_budget: int,
    observation_step: int,
    step_cap: int,
    session_complete: bool = False,
) -> str:
    """Return the bounded carrier-session transition after an observation."""
    if target_hit:
        return "target_hit"
    if (
        bool(session_complete)
        or int(actions_taken) >= int(action_budget)
        or int(observation_step) >= int(step_cap)
    ):
        return "exhausted"
    return "continue"
