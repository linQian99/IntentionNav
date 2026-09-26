"""Uncertainty-aware target belief for active ObjectNav verification.

GroundingDINO detections are target-conditioned proposals, not terminal proof.
This module keeps those proposals in a persistent 3-D memory and combines
detector, crop-semantic, viewpoint, and spatial-consistency evidence.  It is
simulator-agnostic and never reads evaluator target state.
"""
from __future__ import annotations

import math
from typing import Iterable


BELIEF_PRIOR = 0.20
BELIEF_APPROACH_THRESHOLD = 0.28
BELIEF_CONFIRM_THRESHOLD = 0.72
BELIEF_STOP_THRESHOLD = 0.88
BELIEF_MAX_DISPERSION_M = 0.75
BELIEF_MIN_VIEW_BASELINE_M = 0.40
BELIEF_MIN_VIEW_ANGLE_RAD = math.radians(25.0)


def belief_controls_viewpoint(
    cluster_kind: str | None,
    cluster: dict | None,
) -> bool:
    """Restrict the new view controller to belief-owned hypotheses.

    Strict detector-plus-semantic target clusters already have an established
    approach controller.  Applying a second controller to those clusters
    bundles an unrelated intervention and can regress otherwise solved cases.
    """
    if cluster_kind == "belief":
        return True
    return bool(
        cluster_kind == "target"
        and cluster
        and cluster.get("semantic_confirmation")
            == "target_belief_multiview"
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _logit(probability: float) -> float:
    p = _clamp(probability, 1e-6, 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    if value >= 0:
        term = math.exp(-value)
        return 1.0 / (1.0 + term)
    term = math.exp(value)
    return term / (1.0 + term)


def _circular_distance(a: float, b: float) -> float:
    delta = abs(float(a) - float(b)) % (2.0 * math.pi)
    return min(delta, 2.0 * math.pi - delta)


def _semantic_components(verification: dict | None) -> list[dict]:
    """Return independent encoder records from plain or ensemble metadata."""
    if not verification:
        return []
    nested = [
        verification.get("clip_b32"),
        verification.get("siglip2"),
    ]
    components = [
        component for component in nested
        if isinstance(component, dict) and component.get("available", False)
    ]
    if components:
        return components
    if verification.get("available", False):
        return [verification]
    return []


def _rank_log_likelihood(component: dict) -> float:
    """Map a full-vocabulary semantic rank to bounded evidence.

    The bins are intentionally coarse.  They express ordinal evidence rather
    than pretending that similarities from different encoders are calibrated
    probabilities.
    """
    rank = int(component.get("target_rank", 10_000))
    if rank == 1:
        evidence = 1.00
    elif rank == 2:
        evidence = 0.65
    elif rank == 3:
        evidence = 0.42
    elif rank <= 5:
        evidence = 0.15
    elif rank <= 10:
        evidence = -0.30
    elif rank <= 20:
        evidence = -0.65
    else:
        evidence = -1.00
    margin = float(component.get("margin_to_best_other", 0.0) or 0.0)
    if rank == 1:
        evidence += _clamp(margin / 0.04, -0.20, 0.25)
    return _clamp(evidence, -1.10, 1.20)


def observation_belief_evidence(observation: dict) -> dict:
    """Summarize one grounded detector observation as bounded log evidence."""
    score = float(observation.get("score", 0.0) or 0.0)
    detector_llr = _clamp(0.20 + 2.0 * (score - 0.30), -0.10, 1.00)
    verification = (
        observation.get("semantic_verification")
        or observation.get("clip_verification")
        or {}
    )
    components = _semantic_components(verification)
    ranks = [int(item.get("target_rank", 10_000)) for item in components]
    if components:
        semantic_llr = sum(
            _rank_log_likelihood(item) for item in components
        ) / len(components)
    else:
        # Missing semantics is unknown, not positive evidence.
        semantic_llr = -0.15

    semantic_support = bool(ranks and min(ranks) <= 5)
    strong_support = bool(
        ranks
        and (
            max(ranks) <= 5
            or (min(ranks) <= 3 and max(ranks) <= 10)
        )
    )
    encoder_disagreement = bool(
        len(ranks) >= 2 and min(ranks) <= 5 and max(ranks) > 20
    )
    disagreement_penalty = 0.40 if encoder_disagreement else 0.0

    bbox_quality = observation.get("bbox_quality") or {}
    edge_clipped = bool(bbox_quality.get("edge_clipped", False))
    projection_penalty = 0.15 if edge_clipped else 0.0
    log_likelihood_ratio = _clamp(
        detector_llr + semantic_llr
        - disagreement_penalty - projection_penalty,
        -1.25,
        1.45,
    )
    return {
        "log_likelihood_ratio": round(log_likelihood_ratio, 6),
        "detector_llr": round(detector_llr, 6),
        "semantic_llr": round(semantic_llr, 6),
        "semantic_ranks": ranks,
        "semantic_support": semantic_support,
        "strong_semantic_support": strong_support,
        "encoder_disagreement": encoder_disagreement,
        "edge_clipped": edge_clipped,
    }


def _view_groups(observations: Iterable[dict], target_xy: list[float]) -> list[list[dict]]:
    """Group correlated frames by physical target-centric viewpoint."""
    groups: list[list[dict]] = []
    representatives: list[tuple[tuple[float, float], float]] = []
    for observation in observations:
        observer = observation.get("observer_xy")
        if not observer or len(observer) < 2:
            continue
        point = float(observer[0]), float(observer[1])
        bearing = math.atan2(
            point[1] - float(target_xy[1]),
            point[0] - float(target_xy[0]),
        )
        group_index = None
        for index, (previous, previous_bearing) in enumerate(representatives):
            baseline = math.hypot(
                point[0] - previous[0], point[1] - previous[1]
            )
            if (
                baseline < BELIEF_MIN_VIEW_BASELINE_M
                and _circular_distance(bearing, previous_bearing)
                    < BELIEF_MIN_VIEW_ANGLE_RAD
            ):
                group_index = index
                break
        if group_index is None:
            representatives.append((point, bearing))
            groups.append([observation])
        else:
            groups[group_index].append(observation)
    return groups


def update_target_belief(cluster: dict) -> dict:
    """Recompute the posterior and auditable belief diagnostics in-place."""
    observations = cluster.get("observations") or []
    target_xy = cluster.get("xy") or [0.0, 0.0]
    groups = _view_groups(observations, target_xy)
    view_evidence = []
    for group in groups:
        summarized = [observation_belief_evidence(item) for item in group]
        # Several frames at one pose are correlated; retain only the strongest
        # measurement from that physical viewpoint.
        best = max(
            summarized,
            key=lambda item: float(item["log_likelihood_ratio"]),
        )
        view_evidence.append(best)

    # Bound memory influence to the six most recent physical viewpoints.
    view_evidence = view_evidence[-6:]
    semantic_support_views = sum(
        int(item["semantic_support"]) for item in view_evidence
    )
    strong_support_views = sum(
        int(item["strong_semantic_support"]) for item in view_evidence
    )
    log_odds = _logit(BELIEF_PRIOR) + sum(
        float(item["log_likelihood_ratio"]) for item in view_evidence
    )
    # Cross-view semantic recurrence is stronger than repeated detector score,
    # but the bonus is capped to avoid runaway confidence.
    log_odds += 0.45 * min(max(semantic_support_views - 1, 0), 3)
    log_odds += 0.15 * min(strong_support_views, 3)
    log_odds -= 0.70 * int(cluster.get("belief_active_misses", 0))
    dispersion = float(cluster.get("position_dispersion_m", 0.0) or 0.0)
    if dispersion > 0.50:
        log_odds -= min(1.0, (dispersion - 0.50) / 0.35)
    posterior = _sigmoid(log_odds)
    entropy = 0.0
    if 0.0 < posterior < 1.0:
        entropy = -posterior * math.log(posterior) \
            - (1.0 - posterior) * math.log(1.0 - posterior)
        entropy /= math.log(2.0)

    cluster.update({
        "belief_posterior": round(float(posterior), 6),
        "belief_entropy": round(float(entropy), 6),
        "belief_log_odds": round(float(log_odds), 6),
        "belief_viewpoints": len(view_evidence),
        "belief_semantic_support_viewpoints": semantic_support_views,
        "belief_strong_support_viewpoints": strong_support_views,
        "belief_view_evidence": view_evidence,
    })
    return cluster


def target_belief_confirmed(cluster: dict | None) -> bool:
    """Return whether a proposal may enter navigable target memory."""
    if not cluster:
        return False
    update_target_belief(cluster)
    return (
        float(cluster.get("belief_posterior", 0.0))
            >= BELIEF_CONFIRM_THRESHOLD
        and int(cluster.get("belief_viewpoints", 0)) >= 2
        and int(cluster.get("belief_semantic_support_viewpoints", 0)) >= 2
        and float(cluster.get("position_dispersion_m", float("inf")))
            <= BELIEF_MAX_DISPERSION_M
    )


def target_belief_stop_supported(
    cluster: dict | None,
    current_verification: dict | None,
) -> bool:
    """Conservative terminal gate for a belief-promoted current detection."""
    if not cluster:
        return False
    update_target_belief(cluster)
    current = observation_belief_evidence({
        "score": float(cluster.get("detector_score_max", 0.0)),
        "semantic_verification": current_verification or {},
    })
    return (
        float(cluster.get("belief_posterior", 0.0)) >= BELIEF_STOP_THRESHOLD
        and int(cluster.get("belief_viewpoints", 0)) >= 3
        and int(cluster.get("belief_semantic_support_viewpoints", 0)) >= 3
        and bool(current["semantic_support"])
        and not bool(current["encoder_disagreement"])
        and float(cluster.get("position_dispersion_m", float("inf")))
            <= BELIEF_MAX_DISPERSION_M
    )


def record_belief_reperception(
    cluster: dict,
    *,
    step: int,
    observed: bool,
) -> None:
    """Record the outcome of one action-counted active re-observation."""
    cluster["belief_verification_attempts"] = int(
        cluster.get("belief_verification_attempts", 0)
    ) + 1
    cluster["belief_last_verification_step"] = int(step)
    if observed:
        cluster["belief_active_hits"] = int(
            cluster.get("belief_active_hits", 0)
        ) + 1
        cluster["belief_active_misses"] = max(
            0, int(cluster.get("belief_active_misses", 0)) - 1
        )
    else:
        cluster["belief_active_misses"] = int(
            cluster.get("belief_active_misses", 0)
        ) + 1
    update_target_belief(cluster)


def best_target_belief_candidate(
    memory: list[dict],
    *,
    step: int,
    current_xy: tuple[float, float],
    max_age_steps: int = 8,
    max_attempts: int = 3,
    max_distance_m: float | None = None,
) -> dict | None:
    """Select a plausible hypothesis by expected information gain per travel."""
    candidates = []
    for cluster in memory:
        if bool(cluster.get("promoted", False)):
            continue
        if step - int(cluster.get("step", -10_000)) > int(max_age_steps):
            continue
        if int(cluster.get("belief_verification_attempts", 0)) >= int(max_attempts):
            continue
        update_target_belief(cluster)
        posterior = float(cluster.get("belief_posterior", 0.0))
        supports = int(cluster.get("belief_semantic_support_viewpoints", 0))
        if posterior < BELIEF_APPROACH_THRESHOLD or supports < 1:
            continue
        point = cluster.get("xy")
        if not point or len(point) < 2:
            continue
        distance = math.hypot(
            float(point[0]) - current_xy[0],
            float(point[1]) - current_xy[1],
        )
        if max_distance_m is not None and distance > float(max_distance_m):
            continue
        entropy = float(cluster.get("belief_entropy", 0.0))
        information_gain = entropy * (0.65 + 0.35 * posterior)
        score = information_gain * (1.0 + 0.12 * min(supports, 3)) \
            / (1.0 + 0.22 * distance)
        candidates.append((score, posterior, -distance, cluster))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[:3])[3]


def estimated_target_radius_m(
    *,
    bbox: Iterable[float] | None,
    image_width: int,
    depth_m: float | None,
    horizontal_fov_deg: float = 90.0,
) -> float | None:
    """Estimate half-width in metres from policy-visible box geometry."""
    values = list(bbox or [])
    if len(values) != 4 or image_width <= 0 or depth_m is None or depth_m <= 0:
        return None
    fraction = _clamp((float(values[2]) - float(values[0])) / image_width, 0.0, 1.0)
    angular_half_width = math.radians(horizontal_fov_deg) * fraction / 2.0
    radius = float(depth_m) * math.tan(angular_half_width)
    return round(_clamp(radius, 0.10, 0.60), 4)
