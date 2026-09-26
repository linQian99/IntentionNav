"""Evidence-aware target approach utilities for IntentionNav.

The module is deliberately simulator-light: target observations are fused into
serializable dictionaries, while approach waypoints use only ``WalkableMap``
queries.  This keeps the policy shared by hosted-VLM agents and the local
explicit-ObjectNav calibration agent.
"""
from __future__ import annotations

import math
import re
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


SYNONYM_FILE = Path(__file__).resolve().parent.parent / "vocab" / \
    "category_synonyms.yaml"


def normalize_label(label: str) -> str:
    """Normalize an open-vocabulary detector label for exact matching."""
    value = str(label or "").strip().lower().replace("_", " ")
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return re.sub(r"^(?:a|an|the) ", "", value)


def target_label_match(label: str, target: str) -> bool:
    """Allow harmless modifiers (``wall mirror``) but not sibling labels."""
    observed = normalize_label(label)
    wanted = normalize_label(target)
    if not observed or not wanted:
        return False
    return observed == wanted or observed.endswith(f" {wanted}")


@lru_cache(maxsize=4)
def load_detector_synonyms(path: str = str(SYNONYM_FILE)) -> dict[str, list[str]]:
    """Load normalized aliases used to make compact detector prompts."""
    source = Path(path)
    if not source.exists():
        return {}
    try:
        import yaml

        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return {
        normalize_label(category): [
            normalize_label(alias)
            for alias in aliases or []
            if normalize_label(alias)
        ]
        for category, aliases in raw.items()
    }


def target_detector_queries(target: str, max_aliases: int = 2) -> list[str]:
    """Return the target plus collision-safe visual aliases.

    An alias is excluded when it is another benchmark category (for example
    ``couch`` versus ``sofa``) or is owned by multiple categories.  This keeps
    the prompt compact and improves rare-label recall without turning a
    category-level navigation task into an any-related-object task.
    """
    wanted = normalize_label(target)
    if not wanted:
        return []
    synonyms = load_detector_synonyms()
    canonical_categories = set(synonyms)
    alias_owners: dict[str, set[str]] = {}
    for owner, aliases in synonyms.items():
        for alias in aliases:
            alias_owners.setdefault(alias, set()).add(owner)

    queries = [wanted]
    for alias in synonyms.get(wanted, []):
        if alias == wanted or alias in queries:
            continue
        if alias in canonical_categories and alias != wanted:
            continue
        if len(alias_owners.get(alias, {wanted})) > 1:
            continue
        queries.append(alias)
        if len(queries) >= 1 + max(0, int(max_aliases)):
            break
    return queries


def target_query_match(label: str, queries: Iterable[str]) -> bool:
    """Match a DINO label against any phrase in a target-only query set."""
    return any(target_label_match(label, query) for query in queries)


def target_bbox_recenter_action(
    detections: Iterable[dict],
    target_queries: Iterable[str],
    *,
    current_yaw_rad: float,
    image_width: int,
    horizontal_fov_rad: float,
    score_min: float = 0.30,
    edge_fraction_min: float = 0.15,
    max_rotation_rad: float = math.radians(45.0),
) -> tuple[float | None, dict]:
    """Return a bounded yaw that centers a target-conditioned edge box.

    This is an information-gathering action, not target admission.  It uses
    only the current detector label/box and camera intrinsics; callers retain
    the normal semantic verifier and STOP gate on the next observation.
    ``edge_fraction`` follows the audit convention ``abs(cx / width - .5)``.
    """
    queries = tuple(target_queries)
    width = float(image_width)
    hfov = float(horizontal_fov_rad)
    if width <= 0.0 or not 0.0 < hfov < math.pi:
        return None, {"eligible": False, "reason": "invalid_camera"}
    candidates: list[dict] = []
    for detection in detections:
        label = str(detection.get("label") or "")
        score = float(detection.get("score", 0.0))
        bbox = detection.get("bbox") or []
        if (
            score < float(score_min)
            or len(bbox) != 4
            or not target_query_match(label, queries)
        ):
            continue
        x1, _, x2, _ = (float(value) for value in bbox)
        center_x = 0.5 * (x1 + x2)
        edge_fraction = abs(center_x / width - 0.5)
        if edge_fraction < float(edge_fraction_min):
            continue
        candidates.append({
            "label": label,
            "score": score,
            "bbox": [float(value) for value in bbox],
            "center_x": center_x,
            "edge_fraction": edge_fraction,
        })
    if not candidates:
        return None, {"eligible": False, "reason": "no_edge_target_box"}

    # Prefer detector confidence, using larger displacement only as a stable
    # tie-break.  Selecting the edge-most weak box would amplify false boxes.
    selected = max(
        candidates,
        key=lambda item: (item["score"], item["edge_fraction"]),
    )
    focal_px = width / (2.0 * math.tan(hfov / 2.0))
    pixel_offset = float(selected["center_x"]) - width / 2.0
    raw_rotation = math.atan2(pixel_offset, focal_px)
    bounded_rotation = max(
        -float(max_rotation_rad),
        min(float(max_rotation_rad), raw_rotation),
    )
    # Isaac Sim's image-right direction corresponds to decreasing policy yaw.
    target_yaw = float(current_yaw_rad) - bounded_rotation
    return target_yaw, {
        "eligible": True,
        "reason": "target_edge_box",
        "label": selected["label"],
        "score": round(float(selected["score"]), 6),
        "bbox": selected["bbox"],
        "center_x": round(float(selected["center_x"]), 4),
        "edge_fraction": round(float(selected["edge_fraction"]), 6),
        "raw_rotation_rad": round(float(raw_rotation), 6),
        "applied_rotation_rad": round(float(bounded_rotation), 6),
        "yaw_before_rad": float(current_yaw_rad),
        "target_yaw_rad": float(target_yaw),
        "uses_evaluator_target": False,
        "can_admit_target": False,
        "can_stop": False,
    }


def persistent_single_encoder_stop_ok(
    support_sources: Iterable[str],
    n_observations: int,
    *,
    contradiction_observations: int = 4,
) -> bool:
    """Reject persistent evidence that never gains cross-encoder support.

    Repetition and semantic independence are different evidence axes. Once a
    cluster has accumulated four observations, continued support from only one
    of the two frozen image encoders is treated as persistent disagreement,
    not as confirmation. Earlier stops retain the established policy.
    """
    independent_sources = {
        str(source).strip() for source in support_sources if str(source).strip()
    }
    return bool(
        int(n_observations) < int(contradiction_observations)
        or len(independent_sources) >= 2
    )


def terminal_evidence_quorum(
    support_sources: Iterable[str],
    n_observations: int,
    clip_verification: dict | None,
    contextual_vote_count: int,
    *,
    persistent_observations: int = 4,
    weak_rank_max: int = 5,
    contextual_min_votes: int = 2,
) -> tuple[bool, dict]:
    """Require independent corroboration before a target cluster can STOP.

    Two accepting image encoders form a direct quorum.  A young cluster with
    one accepting encoder may instead use either weak compatibility from both
    encoders (the target appears in both top-k lists) or two independent
    contextual votes.  Repeated observations never upgrade persistent
    one-encoder disagreement into terminal evidence.
    """
    independent_sources = sorted({
        str(source).strip()
        for source in support_sources
        if str(source).strip()
    })
    verification = clip_verification or {}
    encoder_ranks: dict[str, int] = {}
    for encoder in ("clip_b32", "siglip2"):
        result = verification.get(encoder) or {}
        rank = result.get("target_rank")
        if result.get("available") and rank is not None:
            encoder_ranks[encoder] = int(rank)

    direct_cross_encoder = len(independent_sources) >= 2
    persistent_disagreement = bool(
        len(independent_sources) < 2
        and int(n_observations) >= int(persistent_observations)
    )
    weak_cross_encoder = bool(
        len(encoder_ranks) >= 2
        and max(encoder_ranks.values()) <= int(weak_rank_max)
    )
    contextual_corroboration = bool(
        int(contextual_vote_count) >= int(contextual_min_votes)
    )
    accepted = bool(
        direct_cross_encoder
        or (
            len(independent_sources) == 1
            and not persistent_disagreement
            and (weak_cross_encoder or contextual_corroboration)
        )
    )
    return accepted, {
        "accepted": accepted,
        "support_sources": independent_sources,
        "n_observations": int(n_observations),
        "direct_cross_encoder": direct_cross_encoder,
        "persistent_disagreement": persistent_disagreement,
        "encoder_ranks": encoder_ranks,
        "weak_cross_encoder": weak_cross_encoder,
        "weak_rank_max": int(weak_rank_max),
        "contextual_vote_count": int(contextual_vote_count),
        "contextual_corroboration": contextual_corroboration,
        "contextual_min_votes": int(contextual_min_votes),
        "uses_evaluator_target": False,
    }


def global_geometry_frontier_waypoint(
    *,
    enabled: bool,
    navigation_cluster: dict | None,
    value_map,
    current_xy: tuple[float, float],
    agent_path: list[tuple[float, float]],
) -> tuple[tuple[float, float] | None, dict]:
    """Return a geometry-only global frontier when no candidate is committed.

    Disabled treatments and committed target/tentative clusters cannot call
    the global planner. Enabled calls receive empty target memory, no direction
    hint, and a zero room-prior scale. A planner fallback is surfaced as
    ``None`` so the engine can retain its established local-frontier fallback.
    """
    if not enabled or navigation_cluster is not None:
        return None, {}
    waypoint, meta = value_map.next_waypoint(
        current_xy,
        target_memory=[],
        agent_path=agent_path,
        direction_hint=None,
        room_prior_scale=0.0,
    )
    meta = dict(meta)
    meta["mode"] = "global_geometry_frontier"
    if meta.get("fallback_reason") is not None:
        return None, meta
    return waypoint, meta


def policy_waypoint_is_reachable(
    wm,
    current_xy: tuple[float, float],
    waypoint_xy: tuple[float, float],
) -> bool:
    """Fail closed unless a MOVE endpoint is finite, walkable, and connected.

    `world_to_cell` clamps out-of-range coordinates to an edge cell, so the
    explicit `is_walkable` call is required before asking for a geodesic path.
    This helper uses only the policy-side freemap and never evaluator target
    state.
    """
    try:
        x, y = float(waypoint_xy[0]), float(waypoint_xy[1])
        current_x, current_y = float(current_xy[0]), float(current_xy[1])
    except (TypeError, ValueError, IndexError):
        return False
    if not all(math.isfinite(value) for value in (x, y, current_x, current_y)):
        return False
    try:
        if not wm.is_walkable(x, y):
            return False
        return wm.geodesic_distance_2d(
            (current_x, current_y), (x, y)
        ) is not None
    except Exception:
        return False


def target_lock_activation_allowed(
    *,
    current_xy: tuple[float, float],
    target_xy: tuple[float, float],
    activation_radius_m: float,
) -> bool:
    """Gate local target-lock actions to an already-entered verify region.

    The gate depends only on policy-side target memory and agent pose.  Invalid
    coordinates fail closed so they cannot consume the bounded local-action
    budget or bypass the established frontier fallback.
    """
    try:
        current_x, current_y = map(float, current_xy)
        target_x, target_y = map(float, target_xy)
        radius = float(activation_radius_m)
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (
        current_x, current_y, target_x, target_y, radius
    )) or radius < 0.0:
        return False
    return math.hypot(current_x - target_x, current_y - target_y) <= radius + 1e-9


def target_visual_memory_hint(
    *,
    enabled: bool,
    context_result: dict | None,
    agent_xy: tuple[float, float],
    yaw: float,
    depth_image=None,
    hfov_rad: float = math.pi / 2.0,
) -> tuple[dict | None, bool]:
    """Convert a closed-set whole-frame rank into a VLFM-style map hint.

    The existing value-map cone treats scores below 5/10 as irrelevant.  We
    preserve that frozen boundary and simply map the calibration-free rank
    percentile to 0..10.  This function never creates target evidence and
    cannot authorize STOP.
    """
    if not enabled or not (context_result or {}).get("available", False):
        return None, False
    try:
        relevance = min(1.0, max(0.0, float(context_result["relevance"])))
    except (KeyError, TypeError, ValueError):
        return None, False
    relevance_score = int(round(10.0 * relevance))
    informative = relevance_score >= 5
    return {
        "agent_xy": (float(agent_xy[0]), float(agent_xy[1])),
        "yaw": float(yaw),
        "relevance_score": relevance_score,
        "relevance_value": relevance,
        "explore_direction": "forward" if informative else "no_clue",
        "depth_image": depth_image,
        "hfov_rad": float(hfov_rad),
        "source": "target_visual_memory",
    }, informative


def target_visual_probabilistic_hint(
    *,
    enabled: bool,
    ensemble_result: dict | None,
    agent_xy: tuple[float, float],
    yaw: float,
    depth_image=None,
    hfov_rad: float = math.pi / 2.0,
) -> dict | None:
    """Validate a prompt-ensemble observation for probabilistic mapping."""
    result = ensemble_result or {}
    if not enabled or not result.get("available", False):
        return None
    try:
        mean = min(1.0, max(0.0, float(result["relevance_mean"])))
        variance = max(0.0, float(result["relevance_variance"]))
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(mean) or not math.isfinite(variance):
        return None
    return {
        "agent_xy": (float(agent_xy[0]), float(agent_xy[1])),
        "yaw": float(yaw),
        "relevance_mean": mean,
        "relevance_variance": variance,
        "depth_image": depth_image,
        "hfov_rad": float(hfov_rad),
        "source": "target_visual_prompt_ensemble",
    }


def target_visual_frontier_waypoint(
    *,
    enabled: bool,
    memory_ready: bool,
    navigation_cluster: dict | None,
    value_map,
    current_xy: tuple[float, float],
    agent_path: list[tuple[float, float]],
    direction_hint: dict | None,
) -> tuple[tuple[float, float] | None, dict]:
    """Select a target-conditioned reachable frontier with no GT room prior."""
    if not enabled or not memory_ready or navigation_cluster is not None:
        return None, {}
    waypoint, meta = value_map.next_waypoint(
        current_xy,
        target_memory=[],
        agent_path=agent_path,
        direction_hint=direction_hint,
        room_prior_scale=0.0,
    )
    meta = dict(meta)
    meta["mode"] = "target_visual_memory_frontier"
    meta["target_visual_memory_ready"] = True
    if meta.get("fallback_reason") is not None:
        return None, meta
    return waypoint, meta


def should_use_global_frontier(
    *,
    enabled: bool,
    navigation_cluster: dict | None,
    local_frontier_meta: dict,
    score_threshold: float,
) -> bool:
    """Escalate from local to global search only after local stagnation.

    ``frontier_score`` is dominated by unexplored-area novelty (range 0..1),
    with small clearance/travel bonuses. A low score therefore means every
    sampled local option is mostly explored. Missing/failed local candidates
    also justify global escalation. No semantic or evaluator state is used.
    """
    if not enabled or navigation_cluster is not None:
        return False
    if local_frontier_meta.get("fallback_reason") is not None:
        return True
    score = local_frontier_meta.get("frontier_score")
    return score is None or float(score) <= float(score_threshold)


def bbox_depth_m(
    depth: np.ndarray | None,
    bbox: Iterable[float],
    inner_fraction: float = 0.5,
    foreground_quantile: float = 0.5,
) -> float | None:
    """Return a foreground-biased depth from the centre of a detection box.

    Callers may select a low quantile for thin objects that occupy less than
    half their detector box.  The median remains the safe default for large
    furniture whose detector box is mostly foreground.
    """
    values = list(bbox)
    if depth is None or depth.ndim < 2 or len(values) != 4:
        return None
    height, width = depth.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in values)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w = max(1.0, (x2 - x1) * inner_fraction / 2.0)
    half_h = max(1.0, (y2 - y1) * inner_fraction / 2.0)
    left = max(0, min(width - 1, int(math.floor(cx - half_w))))
    right = max(left + 1, min(width, int(math.ceil(cx + half_w))))
    top = max(0, min(height - 1, int(math.floor(cy - half_h))))
    bottom = max(top + 1, min(height, int(math.ceil(cy + half_h))))
    crop = depth[top:bottom, left:right]
    valid = crop[np.isfinite(crop) & (crop > 0.05)]
    if valid.size == 0:
        return None
    quantile = max(0.05, min(0.5, float(foreground_quantile)))
    return float(np.quantile(valid, quantile))


def stop_box_quality(
    bbox: Iterable[float],
    image_shape: tuple[int, ...],
    edge_margin_px: float = 2.0,
    min_area_fraction: float = 0.0002,
    max_area_fraction: float = 0.5,
) -> tuple[bool, dict]:
    """Reject degenerate/tiny/full-frame boxes while allowing edge clipping.

    Close targets such as beds, fridges, and tables naturally extend beyond
    the RGB frame. Requiring all four box edges to be visible prevented STOP
    even after detector, CLIP, depth, distance, and multi-view confirmation.
    """
    values = list(bbox)
    if len(values) != 4:
        return False, {"reason": "invalid_bbox"}
    height, width = image_shape[:2]
    x1, y1, x2, y2 = (float(value) for value in values)
    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    area_fraction = box_width * box_height / max(1.0, float(width * height))
    fully_contained = (
        x1 >= edge_margin_px
        and y1 >= edge_margin_px
        and x2 <= width - edge_margin_px
        and y2 <= height - edge_margin_px
    )
    valid_extent = box_width >= 2.0 and box_height >= 2.0
    touches_edges = sum((
        x1 < edge_margin_px,
        y1 < edge_margin_px,
        x2 > width - edge_margin_px,
        y2 > height - edge_margin_px,
    ))
    accepted = (
        valid_extent
        and min_area_fraction <= area_fraction <= max_area_fraction
    )
    return accepted, {
        "fully_contained": fully_contained,
        "touches_edges": touches_edges,
        "valid_extent": valid_extent,
        "area_fraction": round(area_fraction, 4),
        "min_area_fraction": min_area_fraction,
        "max_area_fraction": max_area_fraction,
    }


def adaptive_stop_gates(
    *,
    detector_score: float,
    depth_m: float | None,
    bbox_ok: bool,
    bbox_quality: dict | None,
    cluster_distance_m: float,
    confirmation_ok: bool,
    semantic_ok: bool,
    strong_semantic_ok: bool = False,
    require_strong_semantics_for_relaxation: bool = False,
    score_threshold: float = 0.35,
    verified_score_floor: float = 0.30,
    cluster_distance_threshold_m: float = 1.5,
    verified_cluster_distance_m: float = 1.8,
    max_depth_m: float = 2.0,
    large_bbox_max_depth_m: float = 1.5,
    large_bbox_max_area_fraction: float = 0.99,
) -> tuple[dict[str, bool], dict]:
    """Evaluate strict current-frame STOP evidence with bounded relaxation.

    The base thresholds remain unchanged.  A weak detector score, a slightly
    noisy cluster centroid, or a close large-object box may pass only when two
    independent viewpoints have confirmed the cluster *and* the crop verifier
    supports the requested target.  This avoids making any one relaxed scalar
    threshold sufficient for STOP.
    """
    # Relaxing a geometric/detector scalar requires stronger evidence than a
    # normal STOP: a merely top-ranked crop with a near-zero CLIP margin is
    # insufficient.  The base gates below can still accept persistent weak
    # semantics without relaxing any scalar threshold.
    verified = bool(
        confirmation_ok
        and semantic_ok
        and (
            strong_semantic_ok
            or not require_strong_semantics_for_relaxation
        )
    )
    score = float(detector_score)
    distance = float(cluster_distance_m)
    area_fraction = float((bbox_quality or {}).get("area_fraction", 0.0) or 0.0)
    valid_extent = bool((bbox_quality or {}).get("valid_extent", False))
    depth_ok = depth_m is not None and float(depth_m) <= max_depth_m

    relaxed_score = verified and score >= verified_score_floor
    # A centroid-distance relaxation is only meant to compensate for the
    # same projection noise already evidenced by a weak detector score or a
    # close, clipped large-object box.  Treating distance as an independent
    # relaxation caused otherwise confident detections of a wrong instance to
    # STOP early in multi-instance scenes.
    distance_relaxation_supported = score < score_threshold or not bbox_ok
    relaxed_distance = (
        verified
        and distance_relaxation_supported
        and distance <= verified_cluster_distance_m
    )
    relaxed_large_bbox = (
        verified
        and depth_m is not None
        and float(depth_m) <= large_bbox_max_depth_m
        and valid_extent
        and 0.5 < area_fraction <= large_bbox_max_area_fraction
    )
    gates = {
        "current_frame": True,
        "score_ok": score >= score_threshold or relaxed_score,
        "depth_ok": depth_ok,
        "bbox_ok": bool(bbox_ok) or relaxed_large_bbox,
        "cluster_distance_ok": (
            distance <= cluster_distance_threshold_m or relaxed_distance
        ),
        "confirmation_ok": bool(confirmation_ok),
        "clip_semantic_ok": bool(semantic_ok),
    }
    return gates, {
        "verified_relaxation_eligible": verified,
        "relaxed_score_used": bool(score < score_threshold and relaxed_score),
        "relaxed_distance_used": bool(
            distance > cluster_distance_threshold_m and relaxed_distance
        ),
        "relaxed_large_bbox_used": bool(not bbox_ok and relaxed_large_bbox),
    }


def _normalize_room_label(room: str | None) -> str:
    """Normalize simulator room IDs without consulting episode ground truth."""
    value = normalize_label(room or "")
    value = re.sub(r"(?:^| )\d+(?: |$)", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def contextual_stop_consensus(
    *,
    enabled: bool,
    supported_target: bool,
    clip_verification: dict | None,
    current_room: str | None,
    likely_rooms: Iterable[str],
    distinct_viewpoints: int,
    bbox_area_fraction: float,
    strong_margin: float = 0.01,
    persistent_min_viewpoints: int = 3,
    persistent_min_area_fraction: float = 0.01,
    min_votes: int = 2,
) -> tuple[bool, dict]:
    """Require a quorum of semantic, contextual, and persistent evidence.

    This gate is deliberately downstream of the existing evidence STOP.  It
    neither creates target memory nor changes exploration.  For a safely
    mapped target, two of three policy-visible signals must agree:

    1. CLIP ranks the target first with a positive calibrated margin;
    2. the current simulator room type matches the fixed object-room prior;
    3. a positive-margin crop is spatially persistent and large enough to be
       more than a tiny texture fragment.

    No target pose, target room annotation, instance ID, or segmentation is
    accepted as input.  Unsupported categories are an exact policy no-op.
    """
    required = bool(enabled and supported_target)
    verification = clip_verification or {}
    available = bool(verification.get("available", False))
    target_rank = int(verification.get("target_rank", 10_000))
    margin = float(
        verification.get("margin_to_best_other", float("-inf"))
    )
    target_rank_one = available and target_rank == 1

    normalized_room = _normalize_room_label(current_room)
    normalized_likely = [
        _normalize_room_label(room) for room in likely_rooms
        if _normalize_room_label(room)
    ]
    room_vote = bool(
        normalized_room
        and any(
            normalized_room == likely
            or normalized_room in likely
            or likely in normalized_room
            for likely in normalized_likely
        )
    )
    semantic_vote = bool(target_rank_one and margin >= float(strong_margin))
    persistence_vote = bool(
        target_rank_one
        and margin > 0.0
        and int(distinct_viewpoints) >= int(persistent_min_viewpoints)
        and float(bbox_area_fraction)
        >= float(persistent_min_area_fraction)
    )
    votes = {
        "semantic_margin": semantic_vote,
        "room_context": room_vote,
        "visual_persistence": persistence_vote,
    }
    vote_count = sum(bool(value) for value in votes.values())
    accepted = not required or vote_count >= int(min_votes)
    return accepted, {
        "required": required,
        "supported_target": bool(supported_target),
        "accepted": accepted,
        "vote_count": vote_count,
        "min_votes": int(min_votes),
        "votes": votes,
        "clip_available": available,
        "clip_target_rank": target_rank if available else None,
        "clip_margin": round(margin, 6) if math.isfinite(margin) else None,
        "strong_margin": float(strong_margin),
        "current_room": normalized_room or None,
        "likely_rooms": normalized_likely,
        "distinct_viewpoints": int(distinct_viewpoints),
        "persistent_min_viewpoints": int(persistent_min_viewpoints),
        "bbox_area_fraction": round(float(bbox_area_fraction), 6),
        "persistent_min_area_fraction": float(
            persistent_min_area_fraction
        ),
    }


def budgeted_scan_site_decision(
    *,
    enabled: bool,
    step: int,
    step_cap: int,
    current_xy: tuple[float, float],
    current_room: str | None,
    anchors_by_room: dict[str, list[tuple[float, float]]],
    total_sites: int,
    has_navigation_candidate: bool,
    likely_rooms: Iterable[str] = (),
    require_likely_room: bool = False,
    max_sites: int = 3,
    max_sites_per_room: int = 2,
    min_anchor_distance_m: float = 2.5,
    min_step: int = 2,
    min_budget_fraction: float = 0.0,
    required_action_slots: int = 1,
) -> tuple[bool, dict]:
    """Select a room-coverage site for an explicitly budgeted yaw sweep.

    This pure policy helper sees only the current pose/room and prior scan
    anchors.  It never receives the target pose, target room, instance ID, or
    evaluator state.  Rendering and rotation are deliberately absent: the
    engine records each accepted rotation as an action and consumes its image
    on the next normal loop iteration.
    """
    room_key = normalize_label(current_room or "")
    room_base = re.sub(r"\s+\d+$", "", room_key)
    normalized_likely_rooms = [
        re.sub(r"\s+\d+$", "", normalize_label(room))
        for room in likely_rooms
        if normalize_label(room)
    ]
    likely_room_match = bool(
        room_base
        and any(
            room_base == likely
            or room_base in likely
            or likely in room_base
            for likely in normalized_likely_rooms
        )
    )
    reserved_translation_actions = int(math.ceil(
        float(step_cap) * float(min_budget_fraction)
    ))
    earliest_fractional_scan_step = (
        reserved_translation_actions + 1
        if reserved_translation_actions > 0 else 0
    )
    effective_min_step = max(
        int(min_step),
        earliest_fractional_scan_step,
    )
    room_usable = bool(
        room_key
        and not room_key.startswith("unknown")
        and room_base not in {"none", "other", "corridor", "hallway"}
    )
    room_anchors = anchors_by_room.get(room_key, []) if room_usable else []
    all_anchors = [
        anchor
        for anchors in anchors_by_room.values()
        for anchor in anchors
    ]
    nearest_anchor_m = (
        min(
            math.hypot(
                float(current_xy[0]) - float(anchor[0]),
                float(current_xy[1]) - float(anchor[1]),
            )
            for anchor in all_anchors
        )
        if all_anchors else None
    )
    checks = {
        "enabled": bool(enabled),
        "within_observation_budget": (
            effective_min_step <= int(step) < int(step_cap)
        ),
        "enough_action_slots": (
            int(step) + max(1, int(required_action_slots)) - 1
            <= int(step_cap)
        ),
        "room_usable": room_usable,
        "likely_room": (
            likely_room_match if require_likely_room else True
        ),
        "no_navigation_candidate": not bool(has_navigation_candidate),
        "total_site_budget": int(total_sites) < int(max_sites),
        "room_site_budget": len(room_anchors) < int(max_sites_per_room),
        "spatially_novel_site": (
            nearest_anchor_m is None
            or nearest_anchor_m >= float(min_anchor_distance_m)
        ),
    }
    accepted = all(checks.values())
    return accepted, {
        "accepted": accepted,
        "room_key": room_key or None,
        "room_base": room_base or None,
        "checks": checks,
        "step": int(step),
        "step_cap": int(step_cap),
        "effective_min_step": effective_min_step,
        "reserved_translation_actions": reserved_translation_actions,
        "earliest_scan_action_step": effective_min_step,
        "min_budget_fraction": float(min_budget_fraction),
        "required_action_slots": max(1, int(required_action_slots)),
        "likely_rooms": normalized_likely_rooms,
        "likely_room_match": likely_room_match,
        "require_likely_room": bool(require_likely_room),
        "total_sites": int(total_sites),
        "room_sites": len(room_anchors),
        "max_sites": int(max_sites),
        "max_sites_per_room": int(max_sites_per_room),
        "nearest_anchor_m": (
            round(float(nearest_anchor_m), 6)
            if nearest_anchor_m is not None else None
        ),
        "min_anchor_distance_m": float(min_anchor_distance_m),
    }


def circular_distance(a: float, b: float) -> float:
    delta = abs(float(a) - float(b)) % (2.0 * math.pi)
    return min(delta, 2.0 * math.pi - delta)


def _distinct_view_count(
    observations: list[dict],
    baseline_m: float = 0.35,
    yaw_delta_rad: float = math.radians(20.0),
) -> int:
    representatives: list[dict] = []
    for observation in observations:
        observer = observation.get("observer_xy")
        yaw = float(observation.get("yaw", 0.0))
        if not observer:
            continue
        is_new = True
        for previous in representatives:
            prev_xy = previous["observer_xy"]
            spatial = math.hypot(observer[0] - prev_xy[0], observer[1] - prev_xy[1])
            angular = circular_distance(yaw, float(previous.get("yaw", 0.0)))
            if spatial < baseline_m and angular < yaw_delta_rad:
                is_new = False
                break
        if is_new:
            representatives.append(observation)
    return len(representatives)


def refresh_cluster(cluster: dict) -> dict:
    """Recompute a bounded evidence quality score and diagnostics in-place."""
    observations = cluster.get("observations") or []
    scores = sorted(
        (float(obs.get("score", 0.0)) for obs in observations), reverse=True
    )
    max_score = scores[0] if scores else float(cluster.get("detector_score_max", 0.0))
    mean_top = sum(scores[:3]) / max(1, len(scores[:3]))
    view_count = _distinct_view_count(observations)
    source_count = len({str(obs.get("source", "")) for obs in observations})

    points = [obs.get("xy") for obs in observations if obs.get("xy")]
    dispersion = 0.0
    if points:
        cx, cy = cluster["xy"]
        dispersion = math.sqrt(
            sum((p[0] - cx) ** 2 + (p[1] - cy) ** 2 for p in points)
            / len(points)
        )
    quality = (
        0.50 * max_score
        + 0.15 * mean_top
        + 0.15 * min(len(observations), 3) / 3.0
        + 0.15 * min(view_count, 2) / 2.0
        + 0.05 * min(source_count, 2) / 2.0
        - 0.15 * min(dispersion / 0.75, 1.0)
    )
    failures = int(cluster.get("verification_failures", 0))
    # Suppression, not a one-miss score collapse, controls candidate rejection.
    # The old 0.5**failures factor made the first missed re-detection drop most
    # candidates below the approach threshold, even though the documented
    # policy requires two misses before blacklisting.
    quality *= 0.8 ** failures
    cluster.update({
        "score": round(float(max(0.0, min(1.0, quality))), 6),
        "detector_score_max": round(float(max_score), 6),
        "n_observations": len(observations),
        "distinct_viewpoints": view_count,
        "source_types": sorted({str(obs.get("source", "")) for obs in observations}),
        "position_dispersion_m": round(float(dispersion), 4),
    })
    return cluster


def admit_observation(
    memory: list[dict],
    xy: tuple[float, float],
    score: float,
    label: str,
    step: int,
    source: str,
    observer_xy: tuple[float, float],
    yaw: float,
    depth_m: float | None = None,
    bbox: Iterable[float] | None = None,
    cluster_radius_m: float = 0.75,
) -> dict:
    """Robustly merge one grounded observation into the nearest cluster."""
    tx, ty = float(xy[0]), float(xy[1])
    observation = {
        "xy": [tx, ty],
        "score": round(float(score), 6),
        "label": str(label),
        "step": int(step),
        "source": str(source),
        "observer_xy": [float(observer_xy[0]), float(observer_xy[1])],
        "yaw": float(yaw),
        "depth_m": round(float(depth_m), 4) if depth_m is not None else None,
        "bbox": [round(float(v), 2) for v in bbox] if bbox is not None else None,
    }
    cluster = None
    best_distance = float("inf")
    for candidate in memory:
        distance = math.hypot(candidate["xy"][0] - tx, candidate["xy"][1] - ty)
        if distance < cluster_radius_m and distance < best_distance:
            cluster = candidate
            best_distance = distance
    if cluster is None:
        cluster = {
            "cluster_uid": (
                f"{normalize_label(source)}:{len(memory)}:{int(step)}"
            ),
            "xy": [tx, ty],
            "label": str(label),
            "step": int(step),
            "observations": [],
            "verification_failures": 0,
            "suppressed_until_step": -1,
        }
        memory.append(cluster)

    observations = cluster.setdefault("observations", [])
    observations.append(observation)
    if len(observations) > 16:
        del observations[:-16]
    weights = [max(0.05, float(obs.get("score", 0.0))) for obs in observations]
    weight_sum = sum(weights)
    cluster["xy"] = [
        sum(obs["xy"][0] * w for obs, w in zip(observations, weights)) / weight_sum,
        sum(obs["xy"][1] * w for obs, w in zip(observations, weights)) / weight_sum,
    ]
    cluster["step"] = int(step)
    if float(score) >= float(cluster.get("detector_score_max", 0.0)):
        cluster["label"] = str(label)
    # Repeated output from the same detector is not independent evidence and
    # must not erase a close-range verification failure.  Suppression expires
    # by time in ``best_cluster``; a future independent verifier can explicitly
    # rehabilitate a candidate if one is added.
    return refresh_cluster(cluster)


def cluster_confirmed(
    cluster: dict | None,
    min_observations: int = 2,
    min_viewpoints: int = 2,
    max_dispersion_m: float = 0.8,
) -> bool:
    if not cluster:
        return False
    return (
        int(cluster.get("n_observations", 0)) >= min_observations
        and int(cluster.get("distinct_viewpoints", 0)) >= min_viewpoints
        and float(cluster.get("position_dispersion_m", float("inf"))) <= max_dispersion_m
    )


def stop_confirmation_ok(
    cluster: dict | None,
    strong_semantic_current: bool,
    weak_semantic_min_viewpoints: int = 3,
    max_weak_semantic_failures: int = 1,
) -> bool:
    """Require stronger persistence when crop semantics are ambiguous.

    Two independent views are enough when CLIP has a clear target margin.
    Otherwise the cluster needs at least three views and must not have already
    failed close-range verification twice.  This separates detector recall
    (weak candidates may still be approached) from terminal precision.
    """
    if not cluster_confirmed(cluster):
        return False
    if strong_semantic_current:
        return True
    return (
        cluster_confirmed(
            cluster,
            min_observations=weak_semantic_min_viewpoints,
            min_viewpoints=weak_semantic_min_viewpoints,
        )
        and int(cluster.get("verification_failures", 0))
        <= max_weak_semantic_failures
    )


def semantic_uncertainty_candidate(
    verification: dict | None,
    max_rank: int = 5,
) -> bool:
    """Return whether CLIP is uncertain rather than confidently negative.

    GroundingDINO has already supplied positive target-conditioned evidence.
    Active verification is warranted only when the independent vocabulary-wide
    model still places the requested category in its short candidate list.  A
    low-ranked target is negative evidence and should not consume motion.
    """
    if not verification or not verification.get("available", False):
        return False
    rank = int(verification.get("target_rank", 10_000))
    return 1 <= rank <= int(max_rank)


def persistent_proposal_confirmed(
    cluster: dict | None,
    *,
    min_viewpoints: int = 3,
    min_semantic_supports: int = 2,
    max_clip_rank: int = 5,
    max_dispersion_m: float = 0.5,
) -> bool:
    """Confirm a passive proposal using independent spatial observations.

    A target-conditioned detector hit is only a proposal.  It becomes usable
    target memory when (1) its 3-D projections agree from at least three
    distinct viewpoints and (2) the independent vocabulary-wide crop model
    repeatedly keeps the requested class in its short list. This promotes a
    *relation between observations*, not a single score crossing a threshold.
    Repeated model outputs may still share a systematic error, so this is an
    experimental diagnostic and is disabled by default.
    The promoted cluster still needs a current-frame detection and the normal
    geometric gates before STOP.
    """
    if not cluster_confirmed(
        cluster,
        min_observations=min_viewpoints,
        min_viewpoints=min_viewpoints,
        max_dispersion_m=max_dispersion_m,
    ):
        return False
    observations = (cluster or {}).get("observations") or []
    semantic_supports = 0
    for observation in observations:
        verification = observation.get("semantic_verification") or {}
        if not verification.get("available", False):
            continue
        rank = int(verification.get("target_rank", 10_000))
        semantic_supports += int(1 <= rank <= int(max_clip_rank))
    return semantic_supports >= int(min_semantic_supports)


def best_cluster(
    memory: list[dict],
    step: int,
    min_quality: float = 0.45,
    max_age_steps: int = 8,
    current_xy: tuple[float, float] | None = None,
) -> dict | None:
    candidates = [
        cluster for cluster in memory
        if float(cluster.get("score", 0.0)) >= min_quality
        and step - int(cluster.get("step", -10_000)) <= max_age_steps
        and step >= int(cluster.get("suppressed_until_step", -1))
    ]
    if not candidates:
        return None
    def priority(cluster: dict) -> tuple[float, int, int]:
        quality = float(cluster.get("score", 0.0))
        confirmed_bonus = 0.08 if cluster_confirmed(cluster) else 0.0
        proximity_bonus = 0.0
        if current_xy is not None and cluster.get("xy"):
            distance = math.hypot(
                float(cluster["xy"][0]) - current_xy[0],
                float(cluster["xy"][1]) - current_xy[1],
            )
            # A nearby current hypothesis is cheap to verify and should not be
            # hidden by a slightly higher-scoring stale hypothesis across the
            # room.  The bonus is bounded so detector quality still dominates.
            proximity_bonus = 0.06 * max(0.0, 1.0 - min(distance, 3.0) / 3.0)
        return (
            quality + confirmed_bonus + proximity_bonus,
            int(cluster.get("distinct_viewpoints", 0)),
            int(cluster.get("step", 0)),
        )

    return max(candidates, key=priority)


def bounded_commitment_candidate(
    memory: list[dict],
    *,
    enabled: bool,
    step: int,
    step_cap: int,
    min_quality: float,
    max_age_steps: int = 3,
    current_xy: tuple[float, float] | None = None,
    xy_allowed: Callable[[float, float], bool] | None = None,
) -> dict | None:
    """Recover one recently admitted target after its first missed re-detect.

    The ordinary executor intentionally drops a cluster whose post-failure
    quality falls below ``min_quality``.  That makes a single approach miss a
    hard switch back to unrelated frontier exploration, even though the
    detector and independent crop verifier already admitted the hypothesis.
    This helper permits exactly one additional, action-counted verification
    move when the hypothesis was approach-quality before that miss.

    It never admits evidence or relaxes STOP.  ``xy_allowed`` is a policy-side
    context predicate (the engine uses its fixed object-room prior), not an
    evaluator target-room check.
    """
    if not enabled or int(step) >= int(step_cap):
        return None
    candidates = []
    for cluster in memory:
        xy = cluster.get("xy")
        if not xy or len(xy) < 2:
            continue
        if int(cluster.get("verification_failures", 0)) != 1:
            continue
        if int(cluster.get("bounded_commitment_attempts", 0)) >= 1:
            continue
        if int(step) - int(cluster.get("step", -10_000)) > int(max_age_steps):
            continue
        if int(step) < int(cluster.get("suppressed_until_step", -1)):
            continue
        quality_before_failure = float(
            cluster.get("quality_before_last_failure", 0.0)
        )
        if quality_before_failure < float(min_quality):
            continue
        if xy_allowed is not None and not xy_allowed(
            float(xy[0]), float(xy[1])
        ):
            continue
        candidates.append(cluster)
    if not candidates:
        return None

    def priority(cluster: dict) -> tuple[float, float, int]:
        distance = 0.0
        if current_xy is not None:
            distance = math.hypot(
                float(cluster["xy"][0]) - float(current_xy[0]),
                float(cluster["xy"][1]) - float(current_xy[1]),
            )
        return (
            float(cluster.get("quality_before_last_failure", 0.0)),
            -distance,
            int(cluster.get("step", 0)),
        )

    return max(candidates, key=priority)


def projected_room_matches_prior(
    wm,
    xy: tuple[float, float],
    likely_rooms: Iterable[str],
) -> bool:
    """Match an object projection to a fixed policy-side room prior.

    Object projections commonly land on non-walkable furniture cells, so this
    deliberately uses the scene room polygon/type lookup rather than a
    walkable-cell admission mask.  Missing room metadata or an empty prior
    fails closed: bounded commitment is optional and must not turn an unknown
    context into permission to spend its extra action.
    """
    try:
        wx, wy = float(xy[0]), float(xy[1])
    except (TypeError, ValueError, IndexError):
        return False
    if not (math.isfinite(wx) and math.isfinite(wy)):
        return False
    x_coords = getattr(wm, "x_coords", None)
    y_coords = getattr(wm, "y_coords", None)
    if x_coords is not None and y_coords is not None:
        try:
            if len(x_coords) == 0 or len(y_coords) == 0:
                return False
            x_half = (
                abs(float(x_coords[1]) - float(x_coords[0])) / 2.0
                if len(x_coords) >= 2 else 0.0
            )
            y_half = (
                abs(float(y_coords[1]) - float(y_coords[0])) / 2.0
                if len(y_coords) >= 2 else 0.0
            )
            if not (
                float(np.min(x_coords)) - x_half
                <= wx <= float(np.max(x_coords)) + x_half
                and float(np.min(y_coords)) - y_half
                <= wy <= float(np.max(y_coords)) + y_half
            ):
                return False
        except (TypeError, ValueError, IndexError):
            return False
    else:
        return False
    priors = {
        _normalize_room_label(str(room))
        for room in likely_rooms
        if _normalize_room_label(str(room))
    }
    if not priors:
        return False
    try:
        room = wm.room_at(wx, wy)
    except Exception:
        return False
    room_label = _normalize_room_label(room)
    if not room_label:
        return False
    return any(
        prior in room_label or room_label in prior
        for prior in priors
    )


def source_backed_clusters(
    memory: Iterable[dict],
    *,
    source: str,
    step: int,
    max_age_steps: int,
) -> list[dict]:
    """Return clusters with a recent observation from one exact source."""
    selected = []
    for cluster in memory:
        source_steps = [
            int(observation.get("step", -10_000))
            for observation in cluster.get("observations", [])
            if observation.get("source") == source
        ]
        if source_steps and int(step) - max(source_steps) <= int(max_age_steps):
            selected.append(cluster)
    return selected


def best_verification_proposal(
    memory: list[dict],
    step: int,
    max_age_steps: int = 0,
    max_attempts_per_cluster: int = 1,
    current_xy: tuple[float, float] | None = None,
    require_semantic_shortlist: bool = False,
    max_clip_rank: int = 5,
    xy_allowed: Callable[[float, float], bool] | None = None,
) -> dict | None:
    """Select a fresh, not-yet-probed proposal without a score cutoff.

    Proposal memory is deliberately separate from target memory.  A proposal
    is a geometrically grounded GroundingDINO box that the independent crop
    verifier did not accept.  It may justify one information-gathering action,
    but it must never justify STOP.  This turns a hard semantic rejection into
    a bounded active-perception state instead of tuning the CLIP threshold on
    benchmark episodes.
    """
    candidates = [
        cluster for cluster in memory
        if not bool(cluster.get("promoted", False))
        and int(cluster.get("verification_attempts", 0))
        < int(max_attempts_per_cluster)
        and step - int(cluster.get("step", -10_000)) <= int(max_age_steps)
        and step >= int(cluster.get("suppressed_until_step", -1))
        and (
            xy_allowed is None
            or (
                cluster.get("xy")
                and len(cluster["xy"]) >= 2
                and xy_allowed(
                    float(cluster["xy"][0]),
                    float(cluster["xy"][1]),
                )
            )
        )
        and (
            not require_semantic_shortlist
            or semantic_uncertainty_candidate(
                (cluster.get("observations") or [{}])[-1].get(
                    "semantic_verification"
                ),
                max_rank=max_clip_rank,
            )
        )
    ]
    if not candidates:
        return None

    def priority(cluster: dict) -> tuple[int, float, float, int]:
        distance = 0.0
        if current_xy is not None and cluster.get("xy"):
            distance = math.hypot(
                float(cluster["xy"][0]) - current_xy[0],
                float(cluster["xy"][1]) - current_xy[1],
            )
        return (
            int(cluster.get("distinct_viewpoints", 0)),
            float(cluster.get("detector_score_max", 0.0)),
            -distance,
            int(cluster.get("step", 0)),
        )

    return max(candidates, key=priority)


def mark_verification_attempt(
    cluster: dict,
    step: int,
    observer_xy: tuple[float, float] | None = None,
    target_xy: tuple[float, float] | None = None,
    min_angle_rad: float = math.radians(30.0),
) -> None:
    """Record that the policy spent an action to obtain a better view."""
    cluster["verification_attempts"] = (
        int(cluster.get("verification_attempts", 0)) + 1
    )
    cluster["last_verification_step"] = int(step)
    if observer_xy is None or target_xy is None:
        return
    bearing = math.atan2(
        float(observer_xy[1]) - float(target_xy[1]),
        float(observer_xy[0]) - float(target_xy[0]),
    )
    attempts = cluster.setdefault("verification_observers", [])
    attempts.append({
        "observer_xy": [float(observer_xy[0]), float(observer_xy[1])],
        "bearing_rad": float(bearing),
        "step": int(step),
    })
    if len(attempts) > 8:
        del attempts[:-8]
    representative_bearings: list[float] = []
    for attempt in attempts:
        attempt_bearing = float(attempt["bearing_rad"])
        if all(
            circular_distance(attempt_bearing, previous)
            >= float(min_angle_rad)
            for previous in representative_bearings
        ):
            representative_bearings.append(attempt_bearing)
    cluster["distinct_verification_attempts"] = len(representative_bearings)


def mark_verification_success(cluster: dict, step: int) -> None:
    """Record a positive re-detection after a deliberate verification move.

    A successful observation from the selected standoff is more informative
    than repeatedly rendering at the same pose.  It therefore rehabilitates
    one earlier miss, while retaining the rest of the cluster history.
    """
    cluster["verification_successes"] = (
        int(cluster.get("verification_successes", 0)) + 1
    )
    cluster["last_verified_step"] = int(step)
    cluster["verification_failures"] = max(
        0, int(cluster.get("verification_failures", 0)) - 1
    )
    if int(cluster.get("suppressed_until_step", -1)) > int(step):
        cluster["suppressed_until_step"] = -1
    refresh_cluster(cluster)


def mark_proposal_promoted(cluster: dict, step: int) -> None:
    """Mark a tentative proposal that was accepted from the active view."""
    cluster["promoted"] = True
    cluster["promoted_step"] = int(step)


def mark_verification_failure(
    cluster: dict,
    step: int,
    suppress_after: int = 2,
    suppress_steps: int = 8,
    min_distinct_attempts_for_suppression: int | None = None,
) -> None:
    cluster["quality_before_last_failure"] = float(
        cluster.get("score", 0.0)
    )
    cluster["verification_failures"] = int(cluster.get("verification_failures", 0)) + 1
    enough_distinct_attempts = (
        min_distinct_attempts_for_suppression is None
        or int(cluster.get("distinct_verification_attempts", 0))
        >= int(min_distinct_attempts_for_suppression)
    )
    if (cluster["verification_failures"] >= suppress_after
            and enough_distinct_attempts):
        cluster["suppressed_until_step"] = int(step) + int(suppress_steps)
    refresh_cluster(cluster)


def frontier_waypoint_score(
    waypoint: tuple[float, float, float],
    wm,
    history: list[tuple[float, float]],
    current_xy: tuple[float, float],
) -> float:
    """Score a local frontier proposal by coverage and revisit avoidance.

    The proposal itself is generated from the walkable map, so this helper
    deliberately stays category-agnostic.  Semantic evidence is handled by
    the separate target-approach phase rather than being allowed to pull the
    explorer toward an unverified detector cluster.
    """
    x, y, angle_offset = waypoint
    explored_density = wm.explored_density_at(x, y, radius_m=0.8)
    novelty = 1.0 - explored_density
    history_clearance = min(
        (math.hypot(x - hx, y - hy) for hx, hy in history),
        default=2.0,
    )
    history_clearance = min(history_clearance, 2.0) / 2.0
    travel = min(math.hypot(x - current_xy[0], y - current_xy[1]), 1.7) / 1.7
    turn_cost = min(abs(float(angle_offset)), math.pi) / math.pi
    return novelty + 0.25 * history_clearance + 0.10 * travel - 0.05 * turn_cost


def select_frontier_waypoint(
    waypoints: Iterable[tuple[float, float, float]],
    wm,
    history: list[tuple[float, float]],
    current_xy: tuple[float, float],
) -> tuple[tuple[float, float] | None, dict]:
    """Choose the best deterministic coverage waypoint from local proposals."""
    candidates = list(waypoints)
    if not candidates:
        return None, {
            "mode": "hybrid_frontier",
            "fallback_reason": "no_frontier_waypoints",
            "n_waypoints": 0,
        }
    scores = [
        frontier_waypoint_score(candidate, wm, history, current_xy)
        for candidate in candidates
    ]
    best_index = max(range(len(candidates)), key=lambda index: (scores[index], -index))
    best = candidates[best_index]
    return (float(best[0]), float(best[1])), {
        "mode": "hybrid_frontier",
        "frontier_score": round(float(scores[best_index]), 4),
        "n_waypoints": len(candidates),
        "angle_offset_rad": round(float(best[2]), 4),
        "fallback_reason": None,
    }


def select_coverage_constrained_semantic_frontier(
    waypoints: Iterable[tuple[float, float, float]],
    wm,
    history: list[tuple[float, float]],
    current_xy: tuple[float, float],
    semantic_value_at: Callable[[float, float], float],
    min_geometry_fraction: float = 0.9,
) -> tuple[tuple[float, float] | None, dict]:
    """Use visual memory only inside a near-best coverage candidate set.

    Whole-frame target relevance is a noisy search cue, not target evidence.
    It may therefore break ties among local frontier proposals but may not
    replace geometric exploration or select an arbitrary global map cell.  A
    selected proposal is guaranteed to retain at least
    ``min_geometry_fraction`` of the best available coverage score.
    """
    candidates = list(waypoints)
    if not candidates:
        return None, {
            "mode": "coverage_constrained_semantic_frontier",
            "fallback_reason": "no_frontier_waypoints",
            "n_waypoints": 0,
            "semantic_rerank_applied": False,
        }
    geometry_scores = [
        frontier_waypoint_score(candidate, wm, history, current_xy)
        for candidate in candidates
    ]
    baseline_index = max(
        range(len(candidates)),
        key=lambda index: (geometry_scores[index], -index),
    )
    best_geometry = float(geometry_scores[baseline_index])
    fraction = min(1.0, max(0.0, float(min_geometry_fraction)))
    geometry_floor = (
        best_geometry * fraction
        if best_geometry >= 0.0
        else best_geometry - (1.0 - fraction) * abs(best_geometry)
    )
    admissible = [
        index for index, score in enumerate(geometry_scores)
        if float(score) + 1e-12 >= geometry_floor
    ]
    semantic_scores = [
        max(0.0, float(semantic_value_at(candidate[0], candidate[1])))
        for candidate in candidates
    ]
    informative = bool(
        admissible
        and max(semantic_scores[index] for index in admissible) > 0.0
        and (
            max(semantic_scores[index] for index in admissible)
            - min(semantic_scores[index] for index in admissible)
        ) > 1e-8
    )
    if informative:
        best_index = max(
            admissible,
            key=lambda index: (
                semantic_scores[index], geometry_scores[index], -index
            ),
        )
    else:
        best_index = baseline_index
    best = candidates[best_index]
    changed = best_index != baseline_index
    return (float(best[0]), float(best[1])), {
        "mode": "coverage_constrained_semantic_frontier",
        "frontier_score": round(float(geometry_scores[best_index]), 4),
        "best_geometry_score": round(best_geometry, 4),
        "geometry_floor": round(float(geometry_floor), 4),
        "min_geometry_fraction": fraction,
        "n_waypoints": len(candidates),
        "n_geometry_admissible": len(admissible),
        "angle_offset_rad": round(float(best[2]), 4),
        "semantic_score": round(float(semantic_scores[best_index]), 6),
        "baseline_semantic_score": round(
            float(semantic_scores[baseline_index]), 6
        ),
        "semantic_rerank_informative": informative,
        "semantic_rerank_applied": changed,
        "baseline_candidate_index": int(baseline_index),
        "selected_candidate_index": int(best_index),
        "fallback_reason": None,
    }


def _normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def select_coverage_constrained_expected_improvement_frontier(
    waypoints: Iterable[tuple[float, float, float]],
    wm,
    history: list[tuple[float, float]],
    current_xy: tuple[float, float],
    semantic_belief_at: Callable[[float, float], tuple[float, float]],
    min_geometry_fraction: float = 0.9,
) -> tuple[tuple[float, float] | None, dict]:
    """Select a safe local frontier by parameter-free expected improvement.

    This mirrors UIAP-OGN's I-FBE1 decision rule while retaining IntentionNav's
    local geometric safety envelope.  Unknown cells have high posterior
    variance, so low-relevance observations encourage exploration; confident
    high-relevance cells encourage exploitation.  Semantic belief cannot make
    a candidate below the fixed geometry floor admissible.
    """
    candidates = list(waypoints)
    if not candidates:
        return None, {
            "mode": "coverage_constrained_expected_improvement_frontier",
            "fallback_reason": "no_frontier_waypoints",
            "n_waypoints": 0,
            "semantic_rerank_applied": False,
        }
    geometry_scores = [
        frontier_waypoint_score(candidate, wm, history, current_xy)
        for candidate in candidates
    ]
    baseline_index = max(
        range(len(candidates)),
        key=lambda index: (geometry_scores[index], -index),
    )
    best_geometry = float(geometry_scores[baseline_index])
    fraction = min(1.0, max(0.0, float(min_geometry_fraction)))
    geometry_floor = (
        best_geometry * fraction
        if best_geometry >= 0.0
        else best_geometry - (1.0 - fraction) * abs(best_geometry)
    )
    admissible = [
        index for index, score in enumerate(geometry_scores)
        if float(score) + 1e-12 >= geometry_floor
    ]
    beliefs = []
    for candidate in candidates:
        mean, variance = semantic_belief_at(candidate[0], candidate[1])
        beliefs.append((
            min(1.0, max(0.0, float(mean))),
            max(1e-8, float(variance)),
        ))
    best_mean = max(beliefs[index][0] for index in admissible)
    expected_improvements = []
    for mean, variance in beliefs:
        sigma = math.sqrt(variance)
        z_value = (mean - best_mean) / sigma
        improvement = (
            (mean - best_mean) * _normal_cdf(z_value)
            + sigma * _normal_pdf(z_value)
        )
        expected_improvements.append(max(0.0, improvement))
    best_index = max(
        admissible,
        key=lambda index: (
            expected_improvements[index],
            geometry_scores[index],
            -index,
        ),
    )
    best = candidates[best_index]
    changed = best_index != baseline_index
    selected_mean, selected_variance = beliefs[best_index]
    baseline_mean, baseline_variance = beliefs[baseline_index]
    return (float(best[0]), float(best[1])), {
        "mode": "coverage_constrained_expected_improvement_frontier",
        "frontier_score": round(float(geometry_scores[best_index]), 4),
        "best_geometry_score": round(best_geometry, 4),
        "geometry_floor": round(float(geometry_floor), 4),
        "min_geometry_fraction": fraction,
        "n_waypoints": len(candidates),
        "n_geometry_admissible": len(admissible),
        "angle_offset_rad": round(float(best[2]), 4),
        "semantic_mean": round(selected_mean, 6),
        "semantic_variance": round(selected_variance, 8),
        "expected_improvement": round(
            float(expected_improvements[best_index]), 8
        ),
        "baseline_semantic_mean": round(baseline_mean, 6),
        "baseline_semantic_variance": round(baseline_variance, 8),
        "baseline_expected_improvement": round(
            float(expected_improvements[baseline_index]), 8
        ),
        "semantic_rerank_applied": changed,
        "baseline_candidate_index": int(baseline_index),
        "selected_candidate_index": int(best_index),
        "fallback_reason": None,
    }


def _path_step(
    path: list[tuple[float, float]],
    step_m: float,
) -> tuple[float, float]:
    if not path:
        raise ValueError("path must not be empty")
    distance = 0.0
    previous = path[0]
    for point in path[1:]:
        distance += math.hypot(point[0] - previous[0], point[1] - previous[1])
        previous = point
        if distance >= step_m:
            return point
    return path[-1]


def _view_angle_novelty(
    observer_xy: tuple[float, float],
    target_xy: tuple[float, float],
    observer_history: list[tuple[float, float]],
) -> float:
    """Smallest target-centric bearing difference from prior observations."""
    if not observer_history:
        return math.pi
    bearing = math.atan2(
        observer_xy[1] - target_xy[1],
        observer_xy[0] - target_xy[0],
    )
    previous_bearings = [
        math.atan2(py - target_xy[1], px - target_xy[0])
        for px, py in observer_history
        if math.hypot(px - target_xy[0], py - target_xy[1]) > 1e-6
    ]
    if not previous_bearings:
        return math.pi
    return min(circular_distance(bearing, previous) for previous in previous_bearings)


def target_lock_waypoint(
    wm,
    current_xy: tuple[float, float],
    target_xy: tuple[float, float],
    *,
    current_frame_detected: bool,
    observer_history: list[tuple[float, float]] | None = None,
    max_step_m: float = 0.45,
    terminal_radius_m: float = 1.5,
    min_progress_m: float = 0.05,
    min_translation_m: float = 0.25,
    max_distance_increase_m: float = 0.05,
) -> tuple[tuple[float, float], dict]:
    """Retain bounded local control after strict target approach fails.

    A stale target cluster may only rotate the camera: it cannot cause
    translation.  With a fresh detection, the controller either makes a
    connected monotonic step toward the cluster or a distance-preserving
    tangential step for another view.  Evaluator target state is deliberately
    absent from the interface.
    """
    current = float(current_xy[0]), float(current_xy[1])
    target = float(target_xy[0]), float(target_xy[1])
    values = (
        *current,
        *target,
        float(max_step_m),
        float(terminal_radius_m),
        float(min_progress_m),
        float(min_translation_m),
        float(max_distance_increase_m),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("target-lock inputs must be finite")
    if max_step_m <= 0.0 or terminal_radius_m <= 0.0:
        raise ValueError("target-lock radii must be positive")
    if min_progress_m < 0.0 or min_translation_m < 0.0:
        raise ValueError("target-lock minimum distances must be non-negative")
    if max_distance_increase_m < 0.0:
        raise ValueError("target-lock distance tolerance must be non-negative")

    history = observer_history or []
    current_distance = math.hypot(
        current[0] - target[0], current[1] - target[1]
    )
    base_meta = {
        "candidate_kind": "target",
        "target_xy": [round(target[0], 3), round(target[1], 3)],
        "current_frame_detected": bool(current_frame_detected),
        "target_distance_before_m": round(current_distance, 4),
        "max_step_m": float(max_step_m),
        "terminal_radius_m": float(terminal_radius_m),
        "uses_evaluator_target": False,
    }

    def hold(reason: str) -> tuple[tuple[float, float], dict]:
        return current, {
            **base_meta,
            "mode": "target_lock_reobserve",
            "fallback_reason": reason,
            "waypoint_displacement_m": 0.0,
            "target_distance_after_m": round(current_distance, 4),
            "target_distance_delta_m": 0.0,
            "translation_allowed": False,
        }

    if not current_frame_detected:
        return hold("stale_evidence_hold")

    try:
        if not wm.is_walkable(*current):
            return hold("current_pose_not_walkable")
        cell_m = float(abs(wm.x_coords[1] - wm.x_coords[0]))
        center_y, center_x = wm.world_to_cell(*current)
        radius_cells = max(1, int(math.ceil(float(max_step_m) / cell_m)))
        height, width = wm.grid.shape
    except Exception:
        return hold("invalid_policy_map")

    candidates: list[dict] = []
    for yi in range(
        max(0, center_y - radius_cells),
        min(height, center_y + radius_cells + 1),
    ):
        for xi in range(
            max(0, center_x - radius_cells),
            min(width, center_x + radius_cells + 1),
        ):
            if not wm.grid[yi, xi]:
                continue
            point = wm.cell_to_world(yi, xi)
            displacement = math.hypot(
                point[0] - current[0], point[1] - current[1]
            )
            if displacement <= 1e-6 or displacement > float(max_step_m) + 1e-9:
                continue
            path = wm.shortest_path_2d(current, point, snap_radius_m=0.25)
            if not path:
                continue
            path_length = sum(
                math.hypot(b[0] - a[0], b[1] - a[1])
                for a, b in zip(path, path[1:])
            )
            if path_length > float(max_step_m) + 1e-9:
                continue
            target_distance = math.hypot(
                point[0] - target[0], point[1] - target[1]
            )
            novelty = _view_angle_novelty(point, target, history)
            try:
                line_of_sight = bool(wm.line_of_sight(
                    point[0], point[1], target[0], target[1]
                ))
            except Exception:
                line_of_sight = False
            candidates.append({
                "point": (float(point[0]), float(point[1])),
                "displacement": float(displacement),
                "path_length": float(path_length),
                "target_distance": float(target_distance),
                "progress": float(current_distance - target_distance),
                "novelty": float(novelty),
                "line_of_sight": line_of_sight,
            })

    if current_distance > float(terminal_radius_m):
        eligible = [
            candidate for candidate in candidates
            if candidate["progress"] >= float(min_progress_m) - 1e-9
        ]
        if not eligible:
            return hold("no_monotonic_local_step")
        selected = max(
            eligible,
            key=lambda candidate: (
                candidate["progress"],
                candidate["line_of_sight"],
                candidate["novelty"],
                -candidate["path_length"],
            ),
        )
        mode = "target_lock_monotonic"
    else:
        eligible = [
            candidate for candidate in candidates
            if candidate["displacement"] >= float(min_translation_m) - 1e-9
            and candidate["target_distance"]
                <= current_distance + float(max_distance_increase_m) + 1e-9
        ]
        if not eligible:
            return hold("no_safe_tangential_view")
        selected = max(
            eligible,
            key=lambda candidate: (
                candidate["novelty"],
                candidate["line_of_sight"],
                candidate["displacement"],
                -abs(candidate["target_distance"] - current_distance),
                -candidate["path_length"],
            ),
        )
        mode = "target_lock_tangential"

    return selected["point"], {
        **base_meta,
        "mode": mode,
        "fallback_reason": None,
        "waypoint_displacement_m": round(selected["displacement"], 4),
        "path_length_m": round(selected["path_length"], 4),
        "target_distance_after_m": round(selected["target_distance"], 4),
        "target_distance_delta_m": round(
            selected["target_distance"] - current_distance, 4
        ),
        "target_progress_m": round(selected["progress"], 4),
        "view_angle_novelty_rad": round(selected["novelty"], 4),
        "line_of_sight": bool(selected["line_of_sight"]),
        "translation_allowed": True,
    }


def in_reachable_target_region(
    distance_m: float,
    *,
    strict_max_standoff_m: float,
    reachable_region_max_standoff_m: float,
) -> bool:
    """Return membership in the relaxed ``(strict_max, region_max]`` ring."""
    distance = float(distance_m)
    return (
        distance > float(strict_max_standoff_m)
        and distance <= float(reachable_region_max_standoff_m)
    )


def _reachable_walkable_cells(
    wm,
    current_xy: tuple[float, float],
    *,
    snap_radius_m: float = 1.5,
) -> np.ndarray:
    """Return the 8-connected component reachable from the current pose."""
    start_xy = current_xy
    if not wm.is_walkable(*start_xy):
        snapped = wm.nearby_walkable(
            *start_xy, radius_m=float(snap_radius_m)
        )
        if snapped is None:
            return np.zeros_like(wm.grid, dtype=bool)
        start_xy = snapped
    start = wm.world_to_cell(*start_xy)
    reachable = np.zeros_like(wm.grid, dtype=bool)
    if not (
        0 <= start[0] < reachable.shape[0]
        and 0 <= start[1] < reachable.shape[1]
        and wm.grid[start]
    ):
        return reachable
    queue = deque([start])
    reachable[start] = True
    neighbors = (
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    )
    height, width = reachable.shape
    while queue:
        yi, xi = queue.popleft()
        for dyi, dxi in neighbors:
            neighbor = yi + dyi, xi + dxi
            if not (
                0 <= neighbor[0] < height
                and 0 <= neighbor[1] < width
            ):
                continue
            if reachable[neighbor] or not wm.grid[neighbor]:
                continue
            reachable[neighbor] = True
            queue.append(neighbor)
    return reachable


def _target_visibility_fraction(
    wm,
    observer_xy: tuple[float, float],
    target_xy: tuple[float, float],
    target_extent_radius_m: float | None,
) -> float:
    """Approximate target visibility from policy-visible map geometry."""
    radius = max(0.0, float(target_extent_radius_m or 0.0))
    if radius < 0.05:
        return float(wm.line_of_sight(
            observer_xy[0], observer_xy[1], target_xy[0], target_xy[1]
        ))
    dx = float(target_xy[0]) - float(observer_xy[0])
    dy = float(target_xy[1]) - float(observer_xy[1])
    norm = max(math.hypot(dx, dy), 1e-6)
    perpendicular = -dy / norm, dx / norm
    offsets = (-1.0, -0.5, 0.0, 0.5, 1.0)
    visible = 0
    for scale in offsets:
        sample = (
            float(target_xy[0]) + scale * radius * perpendicular[0],
            float(target_xy[1]) + scale * radius * perpendicular[1],
        )
        visible += int(wm.line_of_sight(
            observer_xy[0], observer_xy[1], sample[0], sample[1]
        ))
    return visible / len(offsets)


def approach_waypoint(
    wm,
    current_xy: tuple[float, float],
    target_xy: tuple[float, float],
    observer_history: list[tuple[float, float]] | None = None,
    desired_standoff_m: float = 1.0,
    min_standoff_m: float = 0.75,
    max_standoff_m: float = 1.45,
    step_m: float = 1.2,
    prefer_novel_view: bool = False,
    min_view_angle_rad: float = math.radians(30.0),
    arrival_tolerance_m: float = 0.25,
    allow_reachable_region_fallback: bool = False,
    reachable_region_max_standoff_m: float = 2.0,
    visibility_aware: bool = False,
    target_extent_radius_m: float | None = None,
) -> tuple[tuple[float, float] | None, dict]:
    """Pick a reachable, visible ring pose and one path step toward it.

    When ``prefer_novel_view`` is enabled, reachable standoffs that observe the
    target from a genuinely different target-centric bearing are preferred.
    This avoids counting several correlated renders at essentially the same
    pose as candidate verification.  The function still falls back to the
    ordinary reachable ring when scene geometry offers no novel view.
    """
    observer_history = observer_history or []
    cell_m = float(abs(wm.x_coords[1] - wm.x_coords[0]))
    radius_cells = max(1, int(math.ceil(max_standoff_m / cell_m)))
    center_y, center_x = wm.world_to_cell(*target_xy)
    height, width = wm.grid.shape
    candidates: list[
        tuple[float, tuple[float, float], float, float]
    ] = []
    for yi in range(max(0, center_y - radius_cells), min(height, center_y + radius_cells + 1)):
        for xi in range(max(0, center_x - radius_cells), min(width, center_x + radius_cells + 1)):
            if not wm.grid[yi, xi]:
                continue
            point = wm.cell_to_world(yi, xi)
            target_distance = math.hypot(point[0] - target_xy[0], point[1] - target_xy[1])
            if not min_standoff_m <= target_distance <= max_standoff_m:
                continue
            visibility_fraction = _target_visibility_fraction(
                wm,
                point,
                target_xy,
                target_extent_radius_m if visibility_aware else None,
            )
            if visibility_fraction <= 0.0:
                continue
            nearest_observer = min(
                (math.hypot(point[0] - px, point[1] - py) for px, py in observer_history),
                default=1.0,
            )
            revisit_cost = max(0.0, 0.5 - nearest_observer)
            explored_cost = wm.explored_density_at(*point, radius_m=0.5)
            travel_cost = math.hypot(point[0] - current_xy[0], point[1] - current_xy[1])
            view_novelty = _view_angle_novelty(
                point, target_xy, observer_history
            )
            cost = (
                abs(target_distance - desired_standoff_m)
                + 0.18 * travel_cost
                + 0.45 * revisit_cost
                + 0.10 * explored_cost
            )
            if prefer_novel_view:
                cost -= 0.35 * min(view_novelty, math.pi) / math.pi
            if visibility_aware:
                cost -= 0.65 * visibility_fraction
            candidates.append(
                (cost, point, view_novelty, visibility_fraction)
            )

    novel_candidates = [
        candidate for candidate in candidates
        if candidate[2] >= float(min_view_angle_rad)
    ]
    novel_view_available = bool(novel_candidates)
    candidate_groups = []
    if prefer_novel_view and novel_view_available:
        candidate_groups.append((novel_candidates, True))
        ordinary_candidates = [
            candidate for candidate in candidates
            if candidate[2] < float(min_view_angle_rad)
        ]
        if ordinary_candidates:
            candidate_groups.append((ordinary_candidates, False))
    else:
        candidate_groups.append((candidates, False))
    reachable_cells = (
        _reachable_walkable_cells(wm, current_xy)
        if allow_reachable_region_fallback else None
    )
    for ranked_candidates, novel_view_used in candidate_groups:
        # Both arms execute exactly the frozen legacy top-32 strict search.
        # The treatment must not improve the strict planner itself; its only
        # intervention is the wider goal-region fallback below.
        ranked_pool = sorted(
            ranked_candidates, key=lambda item: item[0]
        )[:32]
        for _, candidate, view_novelty, visibility_fraction in sorted(
            ranked_pool, key=lambda item: item[0]
        ):
            path = wm.shortest_path_2d(current_xy, candidate, snap_radius_m=1.5)
            if path:
                waypoint = _path_step(path, step_m=step_m)
                arrived_standoff = (
                    math.hypot(
                        waypoint[0] - candidate[0], waypoint[1] - candidate[1]
                    ) <= float(arrival_tolerance_m)
                )
                waypoint_view_novelty = _view_angle_novelty(
                    waypoint, target_xy, observer_history
                )
                waypoint_displacement = math.hypot(
                    waypoint[0] - current_xy[0],
                    waypoint[1] - current_xy[1],
                )
                return waypoint, {
                    "mode": "target_approach",
                    "target_xy": [round(target_xy[0], 3), round(target_xy[1], 3)],
                    "standoff_xy": [round(candidate[0], 3), round(candidate[1], 3)],
                    "path_len_cells": len(path),
                    "prefer_novel_view": bool(prefer_novel_view),
                    "novel_view_available": novel_view_available,
                    "novel_view_used": bool(novel_view_used),
                    "view_angle_novelty_rad": round(float(view_novelty), 4),
                    "waypoint_view_angle_novelty_rad": round(
                        float(waypoint_view_novelty), 4
                    ),
                    "waypoint_displacement_m": round(
                        float(waypoint_displacement), 4
                    ),
                    "min_view_angle_rad": round(float(min_view_angle_rad), 4),
                    "arrived_standoff": arrived_standoff,
                    "visibility_aware": bool(visibility_aware),
                    "estimated_visibility_fraction": round(
                        float(visibility_fraction), 4
                    ),
                    "target_extent_radius_m": (
                        round(float(target_extent_radius_m), 4)
                        if target_extent_radius_m is not None else None
                    ),
                }
    strict_failure = {
        "mode": "target_approach",
        "target_xy": [round(target_xy[0], 3), round(target_xy[1], 3)],
        "prefer_novel_view": bool(prefer_novel_view),
        "novel_view_available": novel_view_available,
        "visibility_aware": bool(visibility_aware),
        "fallback_reason": "no_reachable_standoff",
    }
    if not allow_reachable_region_fallback:
        return None, strict_failure

    # A strict-ring cell outside the legacy top-32 may still be connected to
    # the agent.  In that case the frozen planner would return ``None`` and
    # hand control back to the unchanged exploration policy.  Preserve that
    # behavior instead of either executing the lower-ranked strict candidate
    # or entering the relaxed ring.  Reachability over all strict candidates
    # is used only as a treatment veto, never as a new strict planner.
    if reachable_cells is not None and any(
        reachable_cells[wm.world_to_cell(*candidate)]
        for _, candidate, _, _ in candidates
    ):
        return None, strict_failure

    # A depth-projected target point often lies on furniture or immediately
    # behind its occupied footprint.  Requiring both a narrow 0.75--1.45 m
    # ring and map line-of-sight can therefore reject every otherwise useful
    # goal pose.  Falling back to an unrelated exploration frontier after the
    # target has already been semantically admitted makes the agent walk away
    # from its best evidence.  Instead, search the benchmark's 2 m goal-region
    # scale for the nearest reachable cell.  This uses only the policy's
    # estimated target cluster and the navigation map; no target pose, target
    # room, or evaluator visibility is consulted.
    relaxed_max = float(reachable_region_max_standoff_m)
    if relaxed_max <= float(max_standoff_m):
        return None, {
            **strict_failure,
            "fallback_reason": "invalid_reachable_region_radius",
        }
    relaxed_radius_cells = max(1, int(math.ceil(relaxed_max / cell_m)))
    relaxed_candidates: list[
        tuple[float, tuple[float, float], float, bool]
    ] = []
    for yi in range(
        max(0, center_y - relaxed_radius_cells),
        min(height, center_y + relaxed_radius_cells + 1),
    ):
        for xi in range(
            max(0, center_x - relaxed_radius_cells),
            min(width, center_x + relaxed_radius_cells + 1),
        ):
            if not wm.grid[yi, xi]:
                continue
            point = wm.cell_to_world(yi, xi)
            target_distance = math.hypot(
                point[0] - target_xy[0], point[1] - target_xy[1]
            )
            if not in_reachable_target_region(
                target_distance,
                strict_max_standoff_m=max_standoff_m,
                reachable_region_max_standoff_m=relaxed_max,
            ):
                continue
            has_line_of_sight = bool(
                wm.line_of_sight(
                    point[0], point[1], target_xy[0], target_xy[1]
                )
            )
            travel_cost = math.hypot(
                point[0] - current_xy[0], point[1] - current_xy[1]
            )
            explored_cost = wm.explored_density_at(
                *point, radius_m=0.5
            )
            # Prefer the closest safe goal-region boundary.  Line of sight is
            # a soft preference because the cluster may sit inside the
            # obstacle footprint that generated the depth observation.
            cost = (
                target_distance
                + 0.18 * travel_cost
                + 0.10 * explored_cost
                + (0.0 if has_line_of_sight else 0.20)
            )
            relaxed_candidates.append(
                (cost, point, target_distance, has_line_of_sight)
            )

    for _, candidate, target_distance, has_line_of_sight in sorted(
        relaxed_candidates, key=lambda item: item[0]
    ):
        candidate_cell = wm.world_to_cell(*candidate)
        if reachable_cells is not None and not reachable_cells[candidate_cell]:
            continue
        path = wm.shortest_path_2d(
            current_xy, candidate, snap_radius_m=1.5
        )
        if not path:
            continue
        waypoint = _path_step(path, step_m=step_m)
        arrived_standoff = math.hypot(
            waypoint[0] - candidate[0], waypoint[1] - candidate[1]
        ) <= float(arrival_tolerance_m)
        return waypoint, {
            "mode": "target_region_fallback",
            "target_xy": [
                round(target_xy[0], 3), round(target_xy[1], 3)
            ],
            "standoff_xy": [
                round(candidate[0], 3), round(candidate[1], 3)
            ],
            "target_distance_m": round(float(target_distance), 4),
            "path_len_cells": len(path),
            "line_of_sight": has_line_of_sight,
            "strict_fallback_reason": "no_reachable_standoff",
            "arrived_standoff": arrived_standoff,
        }
    return None, {
        **strict_failure,
        "fallback_reason": "no_reachable_target_region",
        "reachable_region_max_standoff_m": relaxed_max,
    }


def committed_standoff_waypoint(
    wm,
    current_xy: tuple[float, float],
    standoff_xy: tuple[float, float],
    face_direction_xy: tuple[float, float],
    step_m: float = 1.2,
    arrival_tolerance_m: float = 0.25,
) -> tuple[tuple[float, float] | None, dict]:
    """Follow a previously selected target-observation pose.

    ObjectNav policies should not replace a reachable high-level observation
    goal merely because one later detector frame is absent or shifts the
    projected target by a few centimetres. This helper replans the low-level
    path on the current walkable map while keeping the selected standoff and
    look-at point fixed. It has no terminal authority and returns ``None``
    when the commitment is reached or no longer reachable.
    """
    remaining_m = math.hypot(
        float(standoff_xy[0]) - float(current_xy[0]),
        float(standoff_xy[1]) - float(current_xy[1]),
    )
    base_meta = {
        "mode": "target_standoff_commitment",
        "standoff_xy": [
            round(float(standoff_xy[0]), 3),
            round(float(standoff_xy[1]), 3),
        ],
        "face_direction_xy": [
            round(float(face_direction_xy[0]), 3),
            round(float(face_direction_xy[1]), 3),
        ],
        "remaining_to_standoff_m": round(float(remaining_m), 4),
        "arrived_standoff": remaining_m <= float(arrival_tolerance_m),
    }
    if base_meta["arrived_standoff"]:
        return None, {**base_meta, "release_reason": "arrived_standoff"}

    path = wm.shortest_path_2d(
        current_xy, standoff_xy, snap_radius_m=1.5
    )
    if not path:
        return None, {**base_meta, "release_reason": "standoff_unreachable"}
    waypoint = _path_step(path, step_m=step_m)
    displacement_m = math.hypot(
        float(waypoint[0]) - float(current_xy[0]),
        float(waypoint[1]) - float(current_xy[1]),
    )
    return waypoint, {
        **base_meta,
        "path_len_cells": len(path),
        "waypoint_displacement_m": round(float(displacement_m), 4),
        "release_reason": None,
    }
