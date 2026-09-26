"""T_engine: engine-driven nav agent (Track B).

Replaces VLM-as-letter-picker with engine value map + A*. VLM is confined
to the cognitive layer per STATUS §10.3 #3:
  - episode start: VLM emits plan {target_guess, candidate_objects,
    likely_rooms} (reused from agent_vlm.make_episode_plan)
  - per step:
      1. Render RGB + depth.
      2. DINO scans for plan.target_guess and candidate_objects.
         Detected bbox → back-projected to world → target_memory cluster.
      3. ONE VLM call asks "do you see <target_guess>? if yes, pixel
         center?" Single-frame, single answer. NO action selection.
         Affirmative + finite-depth pixel → second target_memory cluster.
      4. ValueMap (target_memory + likely_rooms + frontier + visited)
         is updated. A* picks next waypoint.
      5. STOP fires iff (engine_should_stop AND vlm_sees_target).

Records are schema-compatible with eval/judge/judge.py and
eval/aggregate/compute_metrics.py — same `prediction.target`,
`trajectory[].position`, `episode_meta.target_position`,
`final_frame` path. Tier label is "vlm_engine".
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import time
from collections import deque
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent / "simulator"))
sys.path.insert(0, str(THIS_DIR.parent))

from clients import call_vlm, MODEL_CATALOG  # noqa: E402
from common import (
    EPISODES_FILE, REPO, USD_ROOT, DATASET_ROOT, STYLES, load_prompt, load_items,
    tolerant_json_parse, intent_for_style, episode_path, save_atomic, now_iso,
    scene_manifest_path,
    ROBUSTNESS_ARG_CHOICES, make_target_absent_robustness,
    normalize_category, normalize_robustness_mode, robustness_output_model_name,
)  # noqa: E402
from walkable_map import WalkableMap  # noqa: E402
import dino_detector  # noqa: E402
import ensemble_verifier as clip_verifier  # noqa: E402
import coco_detector  # noqa: E402
import mask_refiner  # noqa: E402
from value_map import ValueMap, STOP_RADIUS_M  # noqa: E402
from evidence_nav import (  # noqa: E402
    adaptive_stop_gates,
    admit_observation,
    approach_waypoint,
    bbox_depth_m,
    best_cluster,
    best_verification_proposal,
    bounded_commitment_candidate,
    budgeted_scan_site_decision,
    cluster_confirmed,
    committed_standoff_waypoint,
    contextual_stop_consensus,
    global_geometry_frontier_waypoint,
    policy_waypoint_is_reachable,
    mark_proposal_promoted,
    mark_verification_attempt,
    mark_verification_success,
    projected_room_matches_prior,
    refresh_cluster,
    persistent_proposal_confirmed,
    persistent_single_encoder_stop_ok,
    terminal_evidence_quorum,
    stop_confirmation_ok,
    select_frontier_waypoint,
    select_coverage_constrained_semantic_frontier,
    select_coverage_constrained_expected_improvement_frontier,
    mark_verification_failure,
    semantic_uncertainty_candidate,
    should_use_global_frontier,
    source_backed_clusters,
    stop_box_quality,
    target_bbox_recenter_action,
    target_detector_queries,
    target_lock_activation_allowed,
    target_lock_waypoint,
    target_query_match,
    target_visual_frontier_waypoint,
    target_visual_memory_hint,
    target_visual_probabilistic_hint,
)
from target_belief import (  # noqa: E402
    BELIEF_CONFIRM_THRESHOLD,
    BELIEF_STOP_THRESHOLD,
    belief_controls_viewpoint,
    best_target_belief_candidate,
    estimated_target_radius_m,
    record_belief_reperception,
    target_belief_confirmed,
    target_belief_stop_supported,
    update_target_belief,
)
from pose_evidence_graph import (  # noqa: E402
    best_pose_graph_candidate,
    latest_pose_observation,
    pose_anchored_reobservation_waypoint,
    pose_graph_support_observation,
    triangulate_pose_observations,
)
# Reuse episode-start plan from agent_vlm so plan format is identical.
from agent_vlm import make_episode_plan, pil_to_jpeg_bytes  # noqa: E402
from object_room_priors import likely_rooms_for_object  # noqa: E402
from object_vocabulary import BENCHMARK_VOCABULARY  # noqa: E402
from object_support_priors import (  # noqa: E402
    carrier_observation_outcome,
    support_candidate_is_new,
    support_detection_candidates,
    support_labels_for_object,
)
from visual_target_descriptors import visual_detector_queries  # noqa: E402


MODEL_CHOICES = list(MODEL_CATALOG.keys())
OPEN_WEIGHT_INTENT_PROTOCOL_V1 = "open_weight_intent_inference_v1"
OPEN_WEIGHT_INTENT_PROTOCOL_V2 = "open_weight_intent_inference_v2"
OPEN_WEIGHT_INTENT_PROTOCOL_V3 = "open_weight_intent_compat_fusion_v3"
OPEN_WEIGHT_INTENT_PROTOCOLS = {
    OPEN_WEIGHT_INTENT_PROTOCOL_V1,
    OPEN_WEIGHT_INTENT_PROTOCOL_V2,
    OPEN_WEIGHT_INTENT_PROTOCOL_V3,
}
OPEN_WEIGHT_ROOM_LABELS = {
    "bathroom", "bedroom", "dining room", "kitchen", "living room",
    "study room", "balcony", "hallway",
}
OPEN_WEIGHT_SUPPORT_LABELS = {
    "basin", "bed", "cabinet", "desk", "dining table", "night stand",
    "shelf", "sofa", "table",
}
OPEN_WEIGHT_ROOM_EVIDENCE_CUES = {
    "bathroom": ("bathroom", "bath", "bathtub", "shower", "toilet"),
    "bedroom": ("bedroom", "bed", "bedside", "night stand", "nightstand"),
    "dining room": ("dining room", "dining table", "dinner table"),
    "kitchen": ("kitchen", "oven", "stove", "fridge", "kitchen counter"),
    "living room": ("living room", "sofa", "couch", "television", "tv"),
    "study room": ("study room", "study", "office", "desk", "computer"),
    "balcony": ("balcony",),
    "hallway": ("hallway", "corridor"),
}
OPEN_WEIGHT_SUPPORT_EVIDENCE_CUES = {
    "basin": ("basin", "sink"),
    "bed": ("bed", "bedside"),
    "cabinet": ("cabinet", "cupboard"),
    "desk": ("desk",),
    "dining table": ("dining table", "dinner table"),
    "night stand": ("night stand", "nightstand", "bedside table"),
    "shelf": ("shelf", "bookshelf"),
    "sofa": ("sofa", "couch"),
    "table": ("table",),
}
OPEN_WEIGHT_SUPPORT_ROOM_PRIORS = {
    "bed": "bedroom",
    "night stand": "bedroom",
    "desk": "study room",
    "dining table": "dining room",
    "sofa": "living room",
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _model_slug(model_id: str) -> str:
    leaf = str(model_id).strip().split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]+", "_", leaf).strip("_") or "open_weight"


@lru_cache(maxsize=8)
def _open_weight_summary(predictions_dir: Path) -> dict:
    summary_path = predictions_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing intent summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol_version") not in OPEN_WEIGHT_INTENT_PROTOCOLS:
        raise ValueError(f"intent summary protocol mismatch: {summary_path}")
    model_id = str(summary.get("model_id") or "").strip()
    revision = str(summary.get("model_revision") or "").strip()
    if not model_id or not revision:
        raise ValueError(f"intent summary lacks pinned model provenance: {summary_path}")
    manifest = summary.get("policy_inputs_manifest") or {}
    if int(manifest.get("files", 0)) <= 0 or not manifest.get("sha256"):
        raise ValueError(f"intent summary lacks policy-input manifest: {summary_path}")
    return summary


def open_weight_output_model_name(predictions_dir: Path) -> str:
    """Stable result-cell name for a frozen open-weight policy input set."""
    summary = _open_weight_summary(predictions_dir)
    model_id = str(summary["model_id"])
    protocol = summary.get("protocol_version")
    suffix = (
        "r057_fusion_local"
        if protocol == OPEN_WEIGHT_INTENT_PROTOCOL_V3
        else "r057_context_local"
        if protocol == OPEN_WEIGHT_INTENT_PROTOCOL_V2
        else "r055_local"
    )
    return f"{_model_slug(model_id)}_{suffix}"


def _normalized_context_text(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def _load_context_constraints(
    prediction: dict,
    field: str,
    *,
    intent: str,
    allowed_labels: set[str],
    evidence_cues: dict[str, tuple[str, ...]],
) -> list[dict[str, str]]:
    """Validate that every policy context constraint quotes the input intent."""
    values = prediction.get(field) or []
    if not isinstance(values, list) or len(values) > 3:
        raise ValueError(f"invalid {field} in open-weight policy input")
    intent_text = _normalized_context_text(intent)
    constraints = []
    for value in values:
        if not isinstance(value, dict) or set(value) != {"label", "evidence"}:
            raise ValueError(f"invalid {field} entry in open-weight policy input")
        label = _normalized_context_text(value.get("label"))
        evidence = str(value.get("evidence") or "").strip()
        evidence_text = _normalized_context_text(evidence)
        cues = evidence_cues.get(label, ())
        if (
            label not in allowed_labels
            or not evidence
            or evidence_text not in intent_text
            or not any(
                _normalized_context_text(cue) in evidence_text for cue in cues
            )
        ):
            raise ValueError(f"ungrounded {field} entry in open-weight policy input")
        constraints.append({"label": label, "evidence": evidence})
    return constraints


def load_open_weight_policy_plan(
    predictions_dir: Path,
    item: dict,
    style: str,
    intent: str,
) -> tuple[dict, dict]:
    """Load a target prediction from a ground-truth-free policy record."""
    path = predictions_dir / "policy_records" / style / f"{item['selection_id']}.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing open-weight policy input: {path}")
    row = json.loads(path.read_text(encoding="utf-8"))
    allowed = {
        "protocol_version",
        "selection_id",
        "scene_id",
        "style",
        "intent_sha256",
        "model",
        "prediction",
        "source_record_sha256",
    }
    unexpected = sorted(set(row) - allowed)
    if unexpected:
        raise ValueError(f"policy input contains unexpected fields: {unexpected}")
    forbidden = {"target_category", "IM_hit", "intent_mode", "raw_response"}
    leaked = sorted(forbidden.intersection(row))
    if leaked:
        raise ValueError(f"policy input contains forbidden ground-truth fields: {leaked}")
    protocol_version = str(row.get("protocol_version") or "")
    expected = {
        "protocol_version": protocol_version,
        "selection_id": item["selection_id"],
        "scene_id": item["scene_id"],
        "style": style,
        "intent_sha256": _sha256_text(intent),
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(
                f"open-weight policy input mismatch for {item['selection_id']}/{style}: "
                f"{key}={row.get(key)!r}, expected {value!r}"
            )
    if protocol_version not in OPEN_WEIGHT_INTENT_PROTOCOLS:
        raise ValueError(
            f"unsupported open-weight protocol for {item['selection_id']}/{style}: "
            f"{protocol_version!r}"
        )
    model = row.get("model") or {}
    model_id = str(model.get("id") or "").strip()
    revision = str(model.get("revision") or "").strip()
    prediction = row.get("prediction") or {}
    allowed_prediction_keys = {"target"}
    if protocol_version in {
        OPEN_WEIGHT_INTENT_PROTOCOL_V2,
        OPEN_WEIGHT_INTENT_PROTOCOL_V3,
    }:
        allowed_prediction_keys.update({
            "room_constraints", "support_constraints",
        })
    unexpected_prediction_keys = sorted(
        set(prediction) - allowed_prediction_keys
    )
    if unexpected_prediction_keys:
        raise ValueError(
            f"unexpected prediction keys for {item['selection_id']}/{style}: "
            f"{unexpected_prediction_keys}"
        )
    target_guess = str(prediction.get("target") or "").strip().lower()
    if not model_id or not revision or not target_guess:
        raise ValueError(
            f"incomplete open-weight policy input for {item['selection_id']}/{style}"
        )
    summary = _open_weight_summary(predictions_dir)
    if model_id != summary.get("model_id") or revision != summary.get(
        "model_revision"
    ):
        raise ValueError(
            f"policy input model differs from frozen summary for "
            f"{item['selection_id']}/{style}"
        )
    if protocol_version != summary.get("protocol_version"):
        raise ValueError(
            f"policy input protocol differs from frozen summary for "
            f"{item['selection_id']}/{style}"
        )
    room_constraints = []
    support_constraints = []
    if protocol_version in {
        OPEN_WEIGHT_INTENT_PROTOCOL_V2,
        OPEN_WEIGHT_INTENT_PROTOCOL_V3,
    }:
        canonical_targets = {
            _normalized_context_text(label) for label in BENCHMARK_VOCABULARY
        }
        if _normalized_context_text(target_guess) not in canonical_targets:
            raise ValueError(
                f"noncanonical structured target for {item['selection_id']}/{style}: "
                f"{target_guess!r}"
            )
        room_constraints = _load_context_constraints(
            prediction,
            "room_constraints",
            intent=intent,
            allowed_labels=OPEN_WEIGHT_ROOM_LABELS,
            evidence_cues=OPEN_WEIGHT_ROOM_EVIDENCE_CUES,
        )
        support_constraints = _load_context_constraints(
            prediction,
            "support_constraints",
            intent=intent,
            allowed_labels=OPEN_WEIGHT_SUPPORT_LABELS,
            evidence_cues=OPEN_WEIGHT_SUPPORT_EVIDENCE_CUES,
        )
    explicit_rooms = [value["label"] for value in room_constraints]
    support_rooms = []
    for value in support_constraints:
        room = OPEN_WEIGHT_SUPPORT_ROOM_PRIORS.get(value["label"])
        if room and room not in support_rooms:
            support_rooms.append(room)
    plan = {
        "target_guess": target_guess,
        "candidate_objects": [target_guess],
        "likely_rooms": (
            explicit_rooms
            or support_rooms
            or likely_rooms_for_object(target_guess)
        ),
        "room_constraints": room_constraints,
        "support_constraints": support_constraints,
        "support_derived_rooms": support_rooms,
        "strategy": f"Search likely rooms for a {target_guess} and stop next to it.",
        "action_plan": [],
    }
    provenance = {
        "protocol_version": protocol_version,
        "model_id": model_id,
        "model_revision": revision,
        "policy_record": str(path.resolve()),
        "policy_record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_record_sha256": row.get("source_record_sha256"),
        "ground_truth_fields_loaded": False,
        "room_constraints": room_constraints,
        "support_constraints": support_constraints,
        "support_derived_rooms": support_rooms,
    }
    return plan, provenance


def _load_yolo_world_verifier():
    """Lazy-load the scan-only detector so the OFF baseline is a no-op."""
    import yolo_world_verifier

    return yolo_world_verifier

# Step-level retry on VLM exhaustion (mirrors agent_vlm A2).
FALLBACK_RETRY_ENABLED = os.environ.get("FALLBACK_RETRY", "1") == "1"
FALLBACK_RETRY_SLEEP_S = float(os.environ.get("FALLBACK_RETRY_SLEEP_S", "10"))
# When enabled, any VLM call that remains failed after all client-level
# retries aborts the episode before record.json is written. The queue keeps
# the item in inflight so run_distributed's retry round can rerun it.
STRICT_VLM_FAILURE = os.environ.get("STRICT_VLM_FAILURE", "0") == "1"

# ---- Track B engine tuning knobs ----
# DINO false-positive control (default DINO threshold is 0.30 — too lax for
# Kujiale where many lamp/sconce/sculpture objects fire on "candelabra"
# style queries). Raise to require higher confidence before writing to
# target_memory.
VM_DINO_THRESHOLD = float(os.environ.get("VM_DINO_THRESHOLD", "0.30"))
# Fix F: only admit DINO clusters where DINO's chosen label EXACTLY matches
# target_guess (case-insensitive, stripped). Catches the sofa/bed/cabinet
# category-confusion failures where DINO returns a candidate label
# (e.g. "armchair") instead of target_guess ("sofa") — under the old
# `find_best_match` logic, these still got admitted as the target. With
# strict label match, an "armchair" detection no longer commits the agent
# to chasing a sofa cluster.
VM_DINO_LABEL_STRICT = os.environ.get("VM_DINO_LABEL_STRICT", "1") == "1"
# Fix G: back-project DINO bbox using bottom-anchored Y instead of center
# Y. For furniture sitting on the floor (bed/sofa/desk/dining_table/
# cabinet), bbox-bottom is the object's ground-contact line, whose depth
# equals the XY-projection of the object onto the floor — i.e. ~target
# geometric center. bbox-center back-projects to the visible surface
# (top/front face) which is 0.5-2m off in XY. Easy-24 V13 showed bed
# cluster_d=1.94m, bathtub=2.08m, mirror=1.79m — all just over 2m SR
# threshold; this systematic offset is the cause.
# Y_FRACTION=0.85 means: 85% down the bbox toward bottom edge.
VM_DINO_BBOX_Y_FRAC = float(os.environ.get("VM_DINO_BBOX_Y_FRAC", "0.5"))
# When ON, DINO admits (both pano-init and per-step) are rejected if the
# back-projected XY falls outside plan.likely_rooms. Easy-24 diag found
# bed-target episodes had ~12.8 false-positive DINO clusters per ep, all
# in the wrong starting room (living room), capturing the value-map argmax
# and blocking exploration to bedroom. VLM admits are NOT gated — VLM is
# the trusted primary signal; DINO is auxiliary detection.
VM_DINO_ADMIT_ROOMS_ONLY = os.environ.get("VM_DINO_ADMIT_ROOMS_ONLY", "0") == "1"
# Fix D1: reject admits whose back-projected XY is on a non-walkable cell
# AND has no walkable cell within VM_WALL_PIERCE_TOLERANCE_M. Catches the
# "depth measures wall, not the object behind" failure mode (bed SEL_007:
# VLM saw bed through wall, depth = wall depth → cluster on wall).
VM_WALL_PIERCE_REJECT = os.environ.get("VM_WALL_PIERCE_REJECT", "0") == "1"
VM_WALL_PIERCE_TOLERANCE_M = float(os.environ.get("VM_WALL_PIERCE_TOLERANCE_M", "0.4"))
# Fix D2: strong_stop additionally requires multi-angle confirmation OR
# recent VLM corroboration. Suppresses single-frame DINO false positives
# from triggering geometric STOP. Cluster's admit_yaws must span
# ≥ VM_STRONG_STOP_YAW_SPREAD_RAD, OR recent VLM see_target=True
# (medium+) within last VM_STRONG_STOP_VLM_FRESHNESS_STEPS, OR
# n_observations ≥ VM_STRONG_STOP_NOBS_BYPASS.
VM_STRONG_STOP_REQUIRE_CONFIRM = os.environ.get("VM_STRONG_STOP_REQUIRE_CONFIRM", "0") == "1"
VM_STRONG_STOP_YAW_SPREAD_RAD = float(os.environ.get("VM_STRONG_STOP_YAW_SPREAD_RAD", "0.52"))
VM_STRONG_STOP_VLM_FRESHNESS_STEPS = int(os.environ.get("VM_STRONG_STOP_VLM_FRESHNESS_STEPS", "2"))
VM_STRONG_STOP_NOBS_BYPASS = int(os.environ.get("VM_STRONG_STOP_NOBS_BYPASS", "3"))
# VM_NO_ADMIT=1: disable BOTH VLM+DINO target_memory writes. Pure
# room_prior + frontier + visited-penalty value map. Use to isolate "is
# memory the bug" vs "is the value map fundamentally pointing wrong way".
VM_NO_ADMIT = os.environ.get("VM_NO_ADMIT", "0") == "1"
# VM_NO_VLM_ADMIT=1: disable only the VLM see_target → memory write.
# DINO admits still happen. Test whether VLM hallucinations are the source
# of bad clusters (DINO might be more reliable since it grounds to bbox).
VM_NO_VLM_ADMIT = os.environ.get("VM_NO_VLM_ADMIT", "1") == "1"
# Engine STOP requires VLM to see target in N consecutive steps before
# triggering. Default 2 = "saw target last step AND this step". Default 1
# = old behavior (single-frame trigger).
VM_STOP_CONSECUTIVE = int(os.environ.get("VM_STOP_CONSECUTIVE", "2"))

# v2: WMNav-style strong geometric STOP. Fires when pose is within
# VM_STRONG_STOP_RADIUS_M of any cluster with score ≥ VM_STRONG_STOP_SCORE_MIN.
# Pure geometry — no VLM confirmation needed. Standard STOP (engine_close
# AND VLM see_target consecutive) remains as the soft fallback.
VM_STRONG_STOP_RADIUS_M = float(os.environ.get("VM_STRONG_STOP_RADIUS_M", "0.8"))
VM_STRONG_STOP_SCORE_MIN = float(os.environ.get("VM_STRONG_STOP_SCORE_MIN", "0.65"))
# Minimum number of source admits a cluster must have to count for
# strong_stop. 1 = any (fragile, original v4 behavior). 2 = require
# multi-source / multi-step reinforcement (filters single-shot DINO
# false positives which dominate score band 0.45-0.55).
VM_STRONG_STOP_MIN_SOURCES = int(
    os.environ.get("VM_STRONG_STOP_MIN_SOURCES", "2")
)
# Q2.1: standard_stop sanity check. VLM see_target=True alone is unreliable
# (Gemini-3.1-flash-lite hallucinates "I see a mirror" when target is at
# yaw -128° behind the agent). Require DINO to also detect target on the
# same frame before honoring VLM's stop signal. If only VLM sees target
# but DINO doesn't, demote — agent keeps navigating; no spurious stop.
# Filters ~4/11 SR=1-but-GSR=0 standard_stop cases observed in V22 +
# the 12/23 standard_stop fires that didn't even reach SR=1.
VM_STANDARD_STOP_REQUIRE_DINO = os.environ.get(
    "VM_STANDARD_STOP_REQUIRE_DINO", "1") == "1"
# X1: detection-driven STOP (VLFM/WMNav-style). When DINO detects target
# on this frame with high confidence AND back-projected world XY is
# within VM_DINO_STOP_RADIUS_M of agent → STOP. Replaces VLM see_target
# as the primary STOP trigger. VLM see_target was the largest source of
# false positives (Gemini-flash hallucinated 'I see X' when X was at
# yaw -128° behind agent). DINO+depth gives a real geometric anchor.
# Standard_stop / strong_stop still exist as fallbacks.
VM_DINO_STOP_ENABLED = os.environ.get("VM_DINO_STOP_ENABLED", "1") == "1"
VM_DINO_STOP_SCORE_MIN = float(os.environ.get("VM_DINO_STOP_SCORE_MIN", "0.5"))
VM_DINO_STOP_RADIUS_M = float(os.environ.get("VM_DINO_STOP_RADIUS_M", "1.5"))
# Episode-start panoramic DINO scan: 4 yaw rotations × DINO render.
# Seeds target_memory before the agent moves. Zero VLM cost (DINO only,
# local GPU). ~1s wall per episode.
VM_PANO_INIT = os.environ.get("VM_PANO_INIT", "1") == "1"
# Fix H: mid-episode pano scan when agent enters a new likely_room.
# Each room gets re-scanned at most once. Agent stays in place, rotates
# 90/180/270, runs DINO on each frame, admits any matching cluster.
# Catches the "agent walks past target but DINO missed it on the single
# forward-facing frame" failure (75% of OSR=1 SR=0 cases in V14).
# Cost: 4 renders + 4 DINO calls per scan (≈1s, $0 since no VLM).
VM_MID_PANO_SCAN = os.environ.get("VM_MID_PANO_SCAN", "1") == "1"
INAV_MID_PANO_ALL_ROOMS = os.environ.get(
    "INAV_MID_PANO_ALL_ROOMS", "0"
) == "1"
VM_MID_PANO_DINO_THRESHOLD = float(
    os.environ.get("VM_MID_PANO_DINO_THRESHOLD", "0.45")
)
# Pano init uses a STRICTER DINO threshold than in-loop. Pano admits at
# step 0 with no validation — false positives become permanent value-map
# bias. In-loop DINO can self-correct (newer admits override).
VM_PANO_INIT_DINO_THRESHOLD = float(
    os.environ.get("VM_PANO_INIT_DINO_THRESHOLD", "0.6")
)
# Strong_stop requires the cluster to have been admitted within the last
# N steps. Filters out stale step-0 pano-init clusters that turn out
# wrong (agent walks toward them, hits 0.8m, false STOP). Set to 999
# to disable freshness check.
VM_STRONG_STOP_FRESHNESS_STEPS = int(
    os.environ.get("VM_STRONG_STOP_FRESHNESS_STEPS", "5")
)

# IntentionNav evidence-aware navigation.  This is the default shared
# navigation policy for all three VLM backends.  Set INAV_EVIDENCE_NAV=0 only
# to reproduce the legacy collection of independent STOP heuristics.
INAV_EVIDENCE_NAV = os.environ.get("INAV_EVIDENCE_NAV", "1") == "1"
INAV_TARGET_ONLY_DINO = os.environ.get("INAV_TARGET_ONLY_DINO", "1") == "1"
INAV_APPROACH_QUALITY = float(os.environ.get("INAV_APPROACH_QUALITY", "0.35"))
INAV_EVIDENCE_MAX_AGE = int(os.environ.get("INAV_EVIDENCE_MAX_AGE", "8"))
INAV_STOP_DINO_SCORE = float(os.environ.get("INAV_STOP_DINO_SCORE", "0.35"))
INAV_STOP_DEPTH_M = float(os.environ.get("INAV_STOP_DEPTH_M", "2.0"))
INAV_STOP_CLUSTER_DISTANCE_M = float(
    # Keep a 0.5 m safety margin inside the benchmark's 2 m success radius.
    # Back-projected object centroids are noisy near cabinet-sized targets.
    os.environ.get("INAV_STOP_CLUSTER_DISTANCE_M", "1.5")
)
INAV_ADAPTIVE_STOP = os.environ.get("INAV_ADAPTIVE_STOP", "1") == "1"
INAV_VERIFIED_STOP_DINO_SCORE = float(
    os.environ.get("INAV_VERIFIED_STOP_DINO_SCORE", "0.30")
)
INAV_VERIFIED_STOP_CLUSTER_DISTANCE_M = float(
    os.environ.get("INAV_VERIFIED_STOP_CLUSTER_DISTANCE_M", "1.8")
)
INAV_LARGE_BBOX_MAX_DEPTH_M = float(
    os.environ.get("INAV_LARGE_BBOX_MAX_DEPTH_M", "1.5")
)
INAV_LARGE_BBOX_MAX_AREA_FRACTION = float(
    os.environ.get("INAV_LARGE_BBOX_MAX_AREA_FRACTION", "0.99")
)
INAV_VERIFY_DISTANCE_M = float(os.environ.get("INAV_VERIFY_DISTANCE_M", "1.8"))
INAV_STANDOFF_M = float(os.environ.get("INAV_STANDOFF_M", "1.0"))
INAV_APPROACH_STEP_M = float(os.environ.get("INAV_APPROACH_STEP_M", "0.6"))
# Opt-in semantic goal-region fallback.  When the strict visible standoff ring
# around an admitted target cluster is unreachable, retain target-directed
# planning toward a reachable cell within the 2 m goal-region scale instead
# of handing control back to an unrelated exploration frontier.
INAV_REACHABLE_TARGET_REGION = (
    os.environ.get("INAV_REACHABLE_TARGET_REGION", "0") == "1"
)
if INAV_REACHABLE_TARGET_REGION:
    INAV_REACHABLE_TARGET_REGION_MAX_M = float(
        os.environ.get("INAV_REACHABLE_TARGET_REGION_MAX_M", "2.0")
    )
    if not 1.45 < INAV_REACHABLE_TARGET_REGION_MAX_M <= 2.0:
        raise RuntimeError(
            "reachable target-region radius must be in (1.45, 2.0] m"
        )
else:
    # Preserve a strict OFF arm even when a parent shell contains malformed
    # treatment-only settings.
    INAV_REACHABLE_TARGET_REGION_MAX_M = 2.0
# One bounded exploitation retry after a semantically admitted target misses
# its first action-counted re-detection.  This preserves the normal STOP gates
# and uses only the fixed policy-side room prior to reject obvious wrong-room
# commitments.  It is opt-in until the paired mechanism screen passes.
INAV_BOUNDED_COMMITMENT = (
    os.environ.get("INAV_BOUNDED_COMMITMENT", "0") == "1"
)
# Structured last-mile verification is an opt-in treatment until it passes the
# frozen hard18/dev40 gates.  It does not alter frontier search or an already
# valid STOP.  Only a semantically admitted target cluster within the local
# verification radius is routed toward a target-visible ring pose whose
# target-centric bearing differs from prior observations.
INAV_STRUCTURED_VERIFICATION = (
    os.environ.get("INAV_STRUCTURED_VERIFICATION", "0") == "1"
)
INAV_STRUCTURED_VERIFY_RADIUS_M = float(
    os.environ.get("INAV_STRUCTURED_VERIFY_RADIUS_M", "2.4")
)
INAV_STRUCTURED_VERIFY_MIN_ANGLE_RAD = math.radians(float(
    os.environ.get("INAV_STRUCTURED_VERIFY_MIN_ANGLE_DEG", "30.0")
))
INAV_STRUCTURED_VERIFY_ARRIVAL_TOLERANCE_M = float(
    os.environ.get("INAV_STRUCTURED_VERIFY_ARRIVAL_TOLERANCE_M", "0.25")
)
INAV_CLIP_VERIFY = os.environ.get("INAV_CLIP_VERIFY", "1") == "1"
# R035: independent full-frame semantic evidence for target-conditioned DINO
# proposals.  DINO retains responsibility for recall/localization; YOLO-World
# sees the complete benchmark vocabulary and may corroborate only an
# overlapping proposal whose winning label is the requested target.  This is
# an opt-in union with the established CLIP/SigLIP admission path, not a
# replacement, so the OFF arm remains byte-for-byte reproducible.
INAV_DUAL_DETECTOR_FUSION = (
    os.environ.get("INAV_DUAL_DETECTOR_FUSION", "0") == "1"
)
INAV_DUAL_DETECTOR_CONFIDENCE = float(
    os.environ.get("INAV_DUAL_DETECTOR_CONFIDENCE", "0.05")
)
INAV_DUAL_DETECTOR_OVERLAP = float(
    os.environ.get("INAV_DUAL_DETECTOR_OVERLAP", "0.5")
)
INAV_DUAL_DETECTOR_STOP_SUPPORT = (
    os.environ.get("INAV_DUAL_DETECTOR_STOP_SUPPORT", "0") == "1"
)
if INAV_DUAL_DETECTOR_FUSION:
    if not INAV_CLIP_VERIFY:
        raise RuntimeError(
            "dual-detector fusion requires the established CLIP path"
        )
    if not 0.0 < INAV_DUAL_DETECTOR_CONFIDENCE <= 1.0:
        raise RuntimeError("dual-detector confidence must be in (0, 1]")
    if not 0.0 < INAV_DUAL_DETECTOR_OVERLAP <= 1.0:
        raise RuntimeError("dual-detector overlap must be in (0, 1]")


def _dual_detector_consensus_accepted(clip_meta: dict | None) -> bool:
    """Return a fail-closed consensus decision for optional verifier data."""
    dual_meta = (clip_meta or {}).get("dual_detector_verification") or {}
    return dual_meta.get("accepted") is True


# VLFM-style category routing: for exact IntentionNav categories with a safe
# COCO mapping, query a frozen closed-set specialist first.  Its box may guide
# approach.  The optional STOP gate decouples high-recall search from terminal
# precision: DINO may still create/approach memory after a specialist miss,
# but a mapped category cannot terminate without a current specialist hit.
# Unsupported categories preserve the reference policy.  Both switches remain
# opt-in until the navigation A/B gate passes; a render audit is not a
# substitute for embodied evaluation.
INAV_COCO_SPECIALIST = os.environ.get("INAV_COCO_SPECIALIST", "0") == "1"
INAV_COCO_STOP_GATE = os.environ.get("INAV_COCO_STOP_GATE", "0") == "1"
INAV_CONTEXTUAL_STOP_CONSENSUS = (
    os.environ.get("INAV_CONTEXTUAL_STOP_CONSENSUS", "0") == "1"
)
INAV_CONTEXTUAL_STOP_STRONG_MARGIN = float(
    os.environ.get("INAV_CONTEXTUAL_STOP_STRONG_MARGIN", "0.01")
)
INAV_CONTEXTUAL_STOP_MIN_VIEWPOINTS = int(
    os.environ.get("INAV_CONTEXTUAL_STOP_MIN_VIEWPOINTS", "3")
)
INAV_CONTEXTUAL_STOP_MIN_AREA_FRACTION = float(
    os.environ.get("INAV_CONTEXTUAL_STOP_MIN_AREA_FRACTION", "0.01")
)
INAV_CONTEXTUAL_STOP_MIN_VOTES = int(
    os.environ.get("INAV_CONTEXTUAL_STOP_MIN_VOTES", "2")
)
INAV_BUDGETED_VIEW_SCAN = (
    os.environ.get("INAV_BUDGETED_VIEW_SCAN", "0") == "1"
)
if INAV_BUDGETED_VIEW_SCAN:
    INAV_BUDGETED_SCAN_ROTATION_DEG = float(
        os.environ.get("INAV_BUDGETED_SCAN_ROTATION_DEG", "120.0")
    )
    INAV_BUDGETED_SCAN_ROTATIONS_PER_SITE = int(
        os.environ.get("INAV_BUDGETED_SCAN_ROTATIONS_PER_SITE", "2")
    )
    INAV_BUDGETED_SCAN_MAX_SITES = int(
        os.environ.get("INAV_BUDGETED_SCAN_MAX_SITES", "3")
    )
    INAV_BUDGETED_SCAN_MAX_SITES_PER_ROOM = int(
        os.environ.get("INAV_BUDGETED_SCAN_MAX_SITES_PER_ROOM", "2")
    )
    INAV_BUDGETED_SCAN_MIN_ANCHOR_DISTANCE_M = float(
        os.environ.get("INAV_BUDGETED_SCAN_MIN_ANCHOR_DISTANCE_M", "2.5")
    )
    INAV_BUDGETED_SCAN_MIN_STEP = int(
        os.environ.get("INAV_BUDGETED_SCAN_MIN_STEP", "2")
    )
    INAV_BUDGETED_SCAN_MIN_BUDGET_FRACTION = float(
        os.environ.get("INAV_BUDGETED_SCAN_MIN_BUDGET_FRACTION", "0.0")
    )
    INAV_BUDGETED_SCAN_LIKELY_ROOM_ONLY = (
        os.environ.get("INAV_BUDGETED_SCAN_LIKELY_ROOM_ONLY", "0") == "1"
    )
    INAV_BUDGETED_SCAN_YOLO_WORLD = (
        os.environ.get("INAV_BUDGETED_SCAN_YOLO_WORLD", "0") == "1"
    )
    INAV_BUDGETED_SCAN_YOLO_CONFIDENCE = float(
        os.environ.get("INAV_BUDGETED_SCAN_YOLO_CONFIDENCE", "0.25")
    )
    INAV_BUDGETED_SCAN_APPROACH_QUALITY = float(
        os.environ.get("INAV_BUDGETED_SCAN_APPROACH_QUALITY", "0.30")
    )
    if not 0.0 < INAV_BUDGETED_SCAN_ROTATION_DEG < 180.0:
        raise RuntimeError(
            "budgeted scan rotation must be in (0, 180) degrees"
        )
    if INAV_BUDGETED_SCAN_ROTATIONS_PER_SITE < 1:
        raise RuntimeError("budgeted scan needs at least one rotation per site")
    if INAV_BUDGETED_SCAN_MAX_SITES < 1:
        raise RuntimeError("budgeted scan needs at least one site")
    if INAV_BUDGETED_SCAN_MAX_SITES_PER_ROOM < 1:
        raise RuntimeError("budgeted scan needs at least one site per room")
    if not 0.0 <= INAV_BUDGETED_SCAN_MIN_BUDGET_FRACTION < 1.0:
        raise RuntimeError(
            "budgeted scan minimum budget fraction must be in [0, 1)"
        )
    if not 0.0 < INAV_BUDGETED_SCAN_YOLO_CONFIDENCE <= 1.0:
        raise RuntimeError(
            "budgeted scan YOLO-World confidence must be in (0, 1]"
        )
    if not 0.0 <= INAV_BUDGETED_SCAN_APPROACH_QUALITY <= 1.0:
        raise RuntimeError(
            "budgeted scan approach quality must be in [0, 1]"
        )
else:
    # A disabled treatment is a strict no-op: malformed treatment-only
    # environment variables must not make the reference baseline fail import.
    INAV_BUDGETED_SCAN_ROTATION_DEG = 120.0
    INAV_BUDGETED_SCAN_ROTATIONS_PER_SITE = 2
    INAV_BUDGETED_SCAN_MAX_SITES = 3
    INAV_BUDGETED_SCAN_MAX_SITES_PER_ROOM = 2
    INAV_BUDGETED_SCAN_MIN_ANCHOR_DISTANCE_M = 2.5
    INAV_BUDGETED_SCAN_MIN_STEP = 2
    INAV_BUDGETED_SCAN_MIN_BUDGET_FRACTION = 0.0
    INAV_BUDGETED_SCAN_LIKELY_ROOM_ONLY = False
    INAV_BUDGETED_SCAN_YOLO_WORLD = False
    INAV_BUDGETED_SCAN_YOLO_CONFIDENCE = 0.25
    INAV_BUDGETED_SCAN_APPROACH_QUALITY = 0.30
if INAV_COCO_STOP_GATE and not INAV_COCO_SPECIALIST:
    raise RuntimeError(
        "INAV_COCO_STOP_GATE=1 requires INAV_COCO_SPECIALIST=1"
    )
if INAV_CONTEXTUAL_STOP_CONSENSUS and not INAV_COCO_SPECIALIST:
    raise RuntimeError(
        "INAV_CONTEXTUAL_STOP_CONSENSUS=1 requires "
        "INAV_COCO_SPECIALIST=1"
    )
if INAV_CONTEXTUAL_STOP_CONSENSUS and INAV_COCO_STOP_GATE:
    raise RuntimeError(
        "contextual consensus and the rejected specialist-only STOP gate "
        "are mutually exclusive"
    )
INAV_CLIP_ADMIT_MAX_RANK = int(os.environ.get("INAV_CLIP_ADMIT_MAX_RANK", "3"))
INAV_CLIP_STOP_MAX_RANK = int(os.environ.get("INAV_CLIP_STOP_MAX_RANK", "3"))
INAV_CLIP_STRONG_MARGIN = float(os.environ.get("INAV_CLIP_STRONG_MARGIN", "0.015"))
INAV_MASK_REFINEMENT = os.environ.get("INAV_MASK_REFINEMENT", "0") == "1"
INAV_STOP_WEAK_SEMANTIC_MIN_VIEWPOINTS = int(
    os.environ.get("INAV_STOP_WEAK_SEMANTIC_MIN_VIEWPOINTS", "3")
)
INAV_STOP_MAX_WEAK_SEMANTIC_FAILURES = int(
    os.environ.get("INAV_STOP_MAX_WEAK_SEMANTIC_FAILURES", "1")
)
INAV_PERSISTENT_STOP_CONFIRMATION = (
    os.environ.get("INAV_PERSISTENT_STOP_CONFIRMATION", "0") == "1"
)
INAV_STRONG_SEMANTIC_RELAXATION = (
    os.environ.get("INAV_STRONG_SEMANTIC_RELAXATION", "0") == "1"
)
# R053: repeated observations do not substitute for independent semantic
# evidence. After four observations, a cluster supported by only one frozen
# image encoder remains navigable but cannot trigger terminal STOP.
INAV_PERSISTENT_SINGLE_ENCODER_VETO = (
    os.environ.get("INAV_PERSISTENT_SINGLE_ENCODER_VETO", "0") == "1"
)
INAV_PERSISTENT_SINGLE_ENCODER_VETO_OBSERVATIONS = int(
    os.environ.get(
        "INAV_PERSISTENT_SINGLE_ENCODER_VETO_OBSERVATIONS", "4"
    )
)
# R054: terminal evidence is a quorum, not a repetition count. Two accepting
# encoders authorize STOP directly. Before persistent disagreement, one
# accepting encoder needs either weak top-k compatibility from both encoders
# or two contextual votes. Search, mapping, and approach remain unchanged.
INAV_TERMINAL_EVIDENCE_QUORUM = (
    os.environ.get("INAV_TERMINAL_EVIDENCE_QUORUM", "0") == "1"
)
INAV_TERMINAL_QUORUM_PERSISTENT_OBSERVATIONS = int(os.environ.get(
    "INAV_TERMINAL_QUORUM_PERSISTENT_OBSERVATIONS", "4"
))
INAV_TERMINAL_QUORUM_WEAK_RANK_MAX = int(os.environ.get(
    "INAV_TERMINAL_QUORUM_WEAK_RANK_MAX", "5"
))
INAV_TERMINAL_QUORUM_CONTEXTUAL_MIN_VOTES = int(os.environ.get(
    "INAV_TERMINAL_QUORUM_CONTEXTUAL_MIN_VOTES", "2"
))
# Hybrid exploration keeps unverified detector clusters out of the navigation
# objective.  Until evidence is strong enough for the last-mile controller,
# the agent follows local geometry frontiers with explicit revisit avoidance.
# This preserves the high-precision evidence/STOP path while addressing the
# dominant no-evidence failure mode of the value-map-only explorer.
INAV_HYBRID_FRONTIER = os.environ.get("INAV_HYBRID_FRONTIER", "1") == "1"
# Geometry-only global frontier treatment.  With no verified target evidence,
# select a reachable global explored/unexplored boundary by information gain
# and A* travel cost.  It receives neither target memory nor a room prior, so
# it cannot exploit a benchmark target room or a category-to-room heuristic.
# Disabled by default until the paired navigation screen passes.
INAV_GLOBAL_FRONTIER = os.environ.get("INAV_GLOBAL_FRONTIER", "0") == "1"
INAV_GLOBAL_FRONTIER_SCORE_THRESHOLD = float(
    os.environ.get("INAV_GLOBAL_FRONTIER_SCORE_THRESHOLD", "0.5")
)
# Geometry foundation repair.  Every proposed MOVE must remain on the
# policy-side connected walkable map.  Invalid proposals are replaced by one
# step of the existing geometry-only global frontier planner; target evidence,
# semantic scoring, STOP, and the action budget are unchanged.
INAV_GEOMETRY_SAFE_FRONTIER = (
    os.environ.get("INAV_GEOMETRY_SAFE_FRONTIER", "0") == "1"
)
# R040: no-action, no-API uncertainty-aware target visual memory. Seven fixed
# prompt forms rank the requested target against the same complete competition
# set. Their mean/variance update a probabilistic semantic map with an
# uninformed N(0.5, 0.5) prior. Parameter-free expected improvement may only
# re-rank local candidates retaining at least 90% of the best geometric
# coverage score. Detector admission, target approach, and STOP are unchanged.
INAV_TARGET_VISUAL_MEMORY_FRONTIER = (
    os.environ.get("INAV_TARGET_VISUAL_MEMORY_FRONTIER", "0") == "1"
)
if INAV_TARGET_VISUAL_MEMORY_FRONTIER and INAV_GLOBAL_FRONTIER:
    raise RuntimeError(
        "target visual memory and geometry-only global frontier are isolated arms"
    )
INAV_FRONTIER_WAYPOINTS = int(os.environ.get("INAV_FRONTIER_WAYPOINTS", "8"))
INAV_FRONTIER_SEED = int(os.environ.get("INAV_FRONTIER_SEED", "42"))
INAV_EXPLORED_RADIUS_M = float(os.environ.get("INAV_EXPLORED_RADIUS_M", "0.8"))
# A second, non-terminal memory keeps geometrically valid target-like boxes
# that the independent crop verifier rejects.  Such a proposal can spend one
# bounded action to acquire a better view, but it cannot affect STOP.  This is
# the local, API-free counterpart of candidate-verification / last-mile active
# perception states used by recent ObjectNav systems.
INAV_TENTATIVE_MEMORY = os.environ.get("INAV_TENTATIVE_MEMORY", "0") == "1"
INAV_TENTATIVE_PROBE_BUDGET = int(
    os.environ.get("INAV_TENTATIVE_PROBE_BUDGET", "2")
)
INAV_TENTATIVE_PROBE_STEP_M = float(
    os.environ.get("INAV_TENTATIVE_PROBE_STEP_M", "0.6")
)
INAV_TENTATIVE_MAX_RANK = int(
    os.environ.get("INAV_TENTATIVE_MAX_RANK", "5")
)
# R042: a target-like box is retained at its known-reachable camera pose and
# image ray.  One bounded lateral view may turn it into ordinary admitted
# target memory, but the uncertain depth projection is never a motion goal and
# never has terminal authority.
INAV_POSE_EVIDENCE_GRAPH = (
    os.environ.get("INAV_POSE_EVIDENCE_GRAPH", "0") == "1"
)
INAV_POSE_GRAPH_ACTION_BUDGET = int(
    os.environ.get("INAV_POSE_GRAPH_ACTION_BUDGET", "2")
)
INAV_POSE_GRAPH_LATERAL_M = float(
    os.environ.get("INAV_POSE_GRAPH_LATERAL_M", "0.8")
)
INAV_POSE_GRAPH_MIN_BASELINE_M = float(
    os.environ.get("INAV_POSE_GRAPH_MIN_BASELINE_M", "0.4")
)
INAV_POSE_GRAPH_MAX_MOVE_M = float(
    os.environ.get("INAV_POSE_GRAPH_MAX_MOVE_M", "1.0")
)
INAV_POSE_GRAPH_MIN_CROSSING_RAD = math.radians(float(
    os.environ.get("INAV_POSE_GRAPH_MIN_CROSSING_DEG", "8.0")
))
INAV_POSE_GRAPH_MAX_CROSSING_RAD = math.radians(float(
    os.environ.get("INAV_POSE_GRAPH_MAX_CROSSING_DEG", "90.0")
))
INAV_POSE_GRAPH_MAX_RANGE_M = float(
    os.environ.get("INAV_POSE_GRAPH_MAX_RANGE_M", "12.0")
)
INAV_POSE_GRAPH_ACCEPTED_RESIDUAL_M = float(
    os.environ.get("INAV_POSE_GRAPH_ACCEPTED_RESIDUAL_M", "1.5")
)
# R044: once an ordinary admitted target is already inside the established
# verification radius and has no reachable strict standoff, keep target
# ownership for a few local actions instead of returning to an unrelated
# exploration frontier. Stale memory can only request a same-position
# re-observation; translation requires current-frame evidence.
INAV_TARGET_LOCK_CONTROLLER = (
    os.environ.get("INAV_TARGET_LOCK_CONTROLLER", "0") == "1"
)
INAV_TARGET_LOCK_ACTION_BUDGET = int(
    os.environ.get("INAV_TARGET_LOCK_ACTION_BUDGET", "3")
)
INAV_TARGET_LOCK_MAX_STEP_M = float(
    os.environ.get("INAV_TARGET_LOCK_MAX_STEP_M", "0.45")
)
INAV_TARGET_LOCK_TERMINAL_RADIUS_M = float(
    os.environ.get("INAV_TARGET_LOCK_TERMINAL_RADIUS_M", "1.5")
)
INAV_TARGET_LOCK_ACTIVATION_RADIUS_M = float(
    os.environ.get(
        "INAV_TARGET_LOCK_ACTIVATION_RADIUS_M",
        str(INAV_VERIFY_DISTANCE_M),
    )
)
INAV_TARGET_LOCK_MIN_PROGRESS_M = float(
    os.environ.get("INAV_TARGET_LOCK_MIN_PROGRESS_M", "0.05")
)
INAV_TARGET_LOCK_MIN_TRANSLATION_M = float(
    os.environ.get("INAV_TARGET_LOCK_MIN_TRANSLATION_M", "0.25")
)
INAV_TARGET_LOCK_MAX_DISTANCE_INCREASE_M = float(
    os.environ.get("INAV_TARGET_LOCK_MAX_DISTANCE_INCREASE_M", "0.05")
)
# R051: keep a high-level target observation pose for a bounded number of
# actions.  Low-level A* is still refreshed every step, while intermittent
# detections and centimetre-scale depth projection changes cannot replace the
# selected standoff immediately.  Admission and STOP remain unchanged.
INAV_TARGET_STANDOFF_COMMITMENT = (
    os.environ.get("INAV_TARGET_STANDOFF_COMMITMENT", "0") == "1"
)
INAV_TARGET_STANDOFF_COMMITMENT_MAX_ACTIONS = int(
    os.environ.get("INAV_TARGET_STANDOFF_COMMITMENT_MAX_ACTIONS", "6")
)
INAV_TARGET_STANDOFF_COMMITMENT_EPISODE_ACTION_BUDGET = int(
    os.environ.get(
        "INAV_TARGET_STANDOFF_COMMITMENT_EPISODE_ACTION_BUDGET", "12"
    )
)
INAV_TARGET_STANDOFF_COMMITMENT_MAX_DRIFT_M = float(
    os.environ.get("INAV_TARGET_STANDOFF_COMMITMENT_MAX_DRIFT_M", "0.75")
)
# R052: when an already-grounded target is proposed at the image edge but has
# not passed the unchanged terminal gate, spend a bounded rotation to center
# the crop and rerun ordinary perception on the next action.  This covers both
# rejected and admitted-but-unconfirmed boxes. It is view control only: the
# rotation cannot itself enter evidence or trigger STOP.
INAV_TARGET_BBOX_RECENTER = (
    os.environ.get("INAV_TARGET_BBOX_RECENTER", "0") == "1"
)
INAV_TARGET_BBOX_RECENTER_ACTION_BUDGET = int(
    os.environ.get("INAV_TARGET_BBOX_RECENTER_ACTION_BUDGET", "2")
)
INAV_TARGET_BBOX_RECENTER_SCORE_MIN = float(
    os.environ.get("INAV_TARGET_BBOX_RECENTER_SCORE_MIN", "0.30")
)
INAV_TARGET_BBOX_RECENTER_EDGE_FRACTION_MIN = float(
    os.environ.get("INAV_TARGET_BBOX_RECENTER_EDGE_FRACTION_MIN", "0.15")
)
INAV_TARGET_BBOX_RECENTER_MAX_ROTATION_DEG = float(
    os.environ.get("INAV_TARGET_BBOX_RECENTER_MAX_ROTATION_DEG", "45.0")
)
INAV_TARGET_BBOX_RECENTER_MEMORY_RADIUS_M = float(
    os.environ.get("INAV_TARGET_BBOX_RECENTER_MEMORY_RADIUS_M", "2.5")
)
# Structural treatment: replace single-frame rejection with a persistent,
# uncertainty-aware 3-D target belief and action-counted re-perception.  This
# mode remains opt-in so the established baseline is exactly reproducible.
INAV_BELIEF_REPERCEPTION = (
    os.environ.get("INAV_BELIEF_REPERCEPTION", "0") == "1"
)
INAV_BELIEF_MAX_AGE = int(os.environ.get("INAV_BELIEF_MAX_AGE", "10"))
INAV_BELIEF_MAX_ATTEMPTS = int(
    os.environ.get("INAV_BELIEF_MAX_ATTEMPTS", "3")
)
INAV_BELIEF_STEP_M = float(os.environ.get("INAV_BELIEF_STEP_M", "1.2"))
# R034: soft target beliefs may request a new view only when the normal
# coverage policy has already brought the agent close enough that one bounded
# action reaches an independent verification standoff.  This prevents an
# uncertain proposal from becoming a long-range navigation goal and preserves
# the frontier controller as the owner of global coverage.
INAV_BELIEF_COVERAGE_PRESERVING = (
    os.environ.get("INAV_BELIEF_COVERAGE_PRESERVING", "0") == "1"
)
INAV_BELIEF_OPPORTUNISTIC_RADIUS_M = float(
    os.environ.get("INAV_BELIEF_OPPORTUNISTIC_RADIUS_M", "2.5")
)
INAV_BELIEF_OPPORTUNISTIC_MAX_AGE = int(
    os.environ.get("INAV_BELIEF_OPPORTUNISTIC_MAX_AGE", "1")
)
INAV_BELIEF_OPPORTUNISTIC_EPISODE_BUDGET = int(
    os.environ.get("INAV_BELIEF_OPPORTUNISTIC_EPISODE_BUDGET", "2")
)
INAV_BELIEF_OPPORTUNISTIC_MAX_MOVE_M = float(
    os.environ.get("INAV_BELIEF_OPPORTUNISTIC_MAX_MOVE_M", "1.1")
)
INAV_CAMERA_HFOV_DEG = float(
    os.environ.get("INAV_CAMERA_HFOV_DEG", "90.0")
)
# A small-object search may spend one action inspecting a visible support
# surface (for example, a table likely to hold a tea set).  Support detections
# never enter target memory and never satisfy STOP; only the normal target
# detector+verifier on the following observation can affect the policy.
INAV_SUPPORT_INSPECTION = (
    os.environ.get("INAV_SUPPORT_INSPECTION", "0") == "1"
)
INAV_SUPPORT_INSPECTION_BUDGET = int(
    os.environ.get("INAV_SUPPORT_INSPECTION_BUDGET", "1")
)
INAV_SUPPORT_INSPECTION_CONFIDENCE = float(
    os.environ.get("INAV_SUPPORT_INSPECTION_CONFIDENCE", "0.25")
)
INAV_SUPPORT_INSPECTION_STEP_M = float(
    os.environ.get("INAV_SUPPORT_INSPECTION_STEP_M", "0.6")
)
if INAV_SUPPORT_INSPECTION:
    if INAV_SUPPORT_INSPECTION_BUDGET != 1:
        raise RuntimeError("support inspection budget is frozen at one action")
    if not math.isclose(
        INAV_SUPPORT_INSPECTION_CONFIDENCE, 0.25,
        rel_tol=0.0, abs_tol=1e-9,
    ):
        raise RuntimeError("support inspection confidence is frozen at 0.25")
    if not math.isclose(
        INAV_SUPPORT_INSPECTION_STEP_M, 0.6,
        rel_tol=0.0, abs_tol=1e-9,
    ):
        raise RuntimeError("support inspection step is frozen at 0.6 m")
# R031: a support is a bounded search region, not a one-step target proxy.
# The controller persists across observations, approaches a reachable standoff,
# requests spatially distinct views, and blacklists an exhausted instance.
# Support evidence itself remains non-terminal and never enters target memory.
INAV_CARRIER_ACTIVE_VERIFY = (
    os.environ.get("INAV_CARRIER_ACTIVE_VERIFY", "0") == "1"
)
INAV_CARRIER_SESSION_BUDGET = int(
    os.environ.get("INAV_CARRIER_SESSION_BUDGET", "2")
)
INAV_CARRIER_ACTION_BUDGET = int(
    os.environ.get("INAV_CARRIER_ACTION_BUDGET", "8")
)
INAV_CARRIER_APPROACH_BUDGET = int(
    os.environ.get("INAV_CARRIER_APPROACH_BUDGET", "5")
)
INAV_CARRIER_ROTATION_BUDGET = int(
    os.environ.get("INAV_CARRIER_ROTATION_BUDGET", "3")
)
INAV_CARRIER_ROTATION_DEG = float(
    os.environ.get("INAV_CARRIER_ROTATION_DEG", "90.0")
)
INAV_CARRIER_STEP_M = float(os.environ.get("INAV_CARRIER_STEP_M", "1.2"))
INAV_CARRIER_BLACKLIST_RADIUS_M = float(
    os.environ.get("INAV_CARRIER_BLACKLIST_RADIUS_M", "1.0")
)
INAV_CARRIER_MIN_VIEW_ANGLE_RAD = math.radians(float(
    os.environ.get("INAV_CARRIER_MIN_VIEW_ANGLE_DEG", "30.0")
))
INAV_VISUAL_TARGET_DESCRIPTORS = (
    os.environ.get("INAV_VISUAL_TARGET_DESCRIPTORS", "0") == "1"
)
if INAV_CARRIER_ACTIVE_VERIFY:
    if INAV_SUPPORT_INSPECTION:
        raise RuntimeError(
            "carrier active verification and legacy support inspection are "
            "mutually exclusive"
        )
    if INAV_CARRIER_SESSION_BUDGET < 1:
        raise RuntimeError("carrier verification needs at least one session")
    if not 2 <= INAV_CARRIER_ACTION_BUDGET <= 10:
        raise RuntimeError("carrier action budget must be in [2, 10]")
    if not 1 <= INAV_CARRIER_APPROACH_BUDGET < INAV_CARRIER_ACTION_BUDGET:
        raise RuntimeError(
            "carrier approach budget must be positive and below total budget"
        )
    if not 1 <= INAV_CARRIER_ROTATION_BUDGET <= 3:
        raise RuntimeError("carrier rotation budget must be in [1, 3]")
    if not 0.0 < INAV_CARRIER_ROTATION_DEG < 180.0:
        raise RuntimeError("carrier rotation must be in (0, 180) degrees")
    if not 0.5 <= INAV_CARRIER_STEP_M <= 1.5:
        raise RuntimeError("carrier step must be in [0.5, 1.5] m")
    if INAV_CARRIER_BLACKLIST_RADIUS_M <= 0.0:
        raise RuntimeError("carrier blacklist radius must be positive")
# Passive multi-view promotion keeps uncertain boxes out of the policy until
# their 3-D projections agree across views.  Unlike active tentative probing,
# it cannot divert exploration before that confirmation exists.
INAV_PASSIVE_MULTI_VIEW = os.environ.get("INAV_PASSIVE_MULTI_VIEW", "0") == "1"
INAV_PASSIVE_MIN_VIEWPOINTS = int(
    os.environ.get("INAV_PASSIVE_MIN_VIEWPOINTS", "3")
)
INAV_PASSIVE_MIN_SEMANTIC_SUPPORTS = int(
    os.environ.get("INAV_PASSIVE_MIN_SEMANTIC_SUPPORTS", "2")
)
INAV_PASSIVE_MAX_DISPERSION_M = float(
    os.environ.get("INAV_PASSIVE_MAX_DISPERSION_M", "0.5")
)
if (INAV_SUPPORT_INSPECTION or INAV_CARRIER_ACTIVE_VERIFY) and any((
    INAV_TENTATIVE_MEMORY,
    INAV_POSE_EVIDENCE_GRAPH,
    INAV_PASSIVE_MULTI_VIEW,
    INAV_BOUNDED_COMMITMENT,
    INAV_BUDGETED_VIEW_SCAN,
    INAV_REACHABLE_TARGET_REGION,
)):
    raise RuntimeError(
        "support/carrier inspection must be evaluated as an isolated treatment"
    )
if INAV_BELIEF_REPERCEPTION and any((
    INAV_TENTATIVE_MEMORY,
    INAV_POSE_EVIDENCE_GRAPH,
    INAV_PASSIVE_MULTI_VIEW,
    INAV_SUPPORT_INSPECTION,
    INAV_CARRIER_ACTIVE_VERIFY,
    INAV_BOUNDED_COMMITMENT,
    INAV_BUDGETED_VIEW_SCAN,
    INAV_REACHABLE_TARGET_REGION,
)):
    raise RuntimeError(
        "belief re-perception must be evaluated as an isolated treatment"
    )
if INAV_POSE_EVIDENCE_GRAPH:
    if any((
        INAV_TENTATIVE_MEMORY,
        INAV_PASSIVE_MULTI_VIEW,
        INAV_BELIEF_REPERCEPTION,
        INAV_SUPPORT_INSPECTION,
        INAV_CARRIER_ACTIVE_VERIFY,
        INAV_BOUNDED_COMMITMENT,
        INAV_BUDGETED_VIEW_SCAN,
        INAV_REACHABLE_TARGET_REGION,
        INAV_TARGET_VISUAL_MEMORY_FRONTIER,
        INAV_GLOBAL_FRONTIER,
        INAV_DUAL_DETECTOR_FUSION,
    )):
        raise RuntimeError(
            "pose evidence graph must be evaluated as an isolated treatment"
        )
    if not 1 <= INAV_POSE_GRAPH_ACTION_BUDGET <= 2:
        raise RuntimeError("pose graph action budget must be one or two")
    if not math.isclose(
        INAV_POSE_GRAPH_LATERAL_M, 0.8, rel_tol=0.0, abs_tol=1e-9
    ):
        raise RuntimeError("R042 lateral offset is frozen at 0.8 m")
    if not math.isclose(
        INAV_POSE_GRAPH_MIN_BASELINE_M, 0.4,
        rel_tol=0.0, abs_tol=1e-9,
    ):
        raise RuntimeError("R042 minimum baseline is frozen at 0.4 m")
    if not math.isclose(
        INAV_POSE_GRAPH_MAX_MOVE_M, 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise RuntimeError("R042 maximum move is frozen at 1.0 m")
    if not math.isclose(
        INAV_POSE_GRAPH_MIN_CROSSING_RAD, math.radians(8.0),
        rel_tol=0.0, abs_tol=1e-9,
    ):
        raise RuntimeError("R042 minimum crossing is frozen at 8 degrees")
    if not math.isclose(
        INAV_POSE_GRAPH_MAX_CROSSING_RAD, math.pi / 2.0,
        rel_tol=0.0, abs_tol=1e-9,
    ):
        raise RuntimeError("R042 maximum crossing is frozen at 90 degrees")
    if not math.isclose(
        INAV_POSE_GRAPH_MAX_RANGE_M, 12.0,
        rel_tol=0.0, abs_tol=1e-9,
    ):
        raise RuntimeError("R042 maximum triangulation range is frozen at 12 m")
    if not math.isclose(
        INAV_POSE_GRAPH_ACCEPTED_RESIDUAL_M, 1.5,
        rel_tol=0.0, abs_tol=1e-9,
    ):
        raise RuntimeError("R042 accepted-projection residual is frozen at 1.5 m")
if INAV_TARGET_LOCK_CONTROLLER:
    if not INAV_GEOMETRY_SAFE_FRONTIER:
        raise RuntimeError("R044 target lock requires R041 geometry safety")
    if any((
        INAV_TENTATIVE_MEMORY,
        INAV_POSE_EVIDENCE_GRAPH,
        INAV_PASSIVE_MULTI_VIEW,
        INAV_BELIEF_REPERCEPTION,
        INAV_SUPPORT_INSPECTION,
        INAV_CARRIER_ACTIVE_VERIFY,
        INAV_BOUNDED_COMMITMENT,
        INAV_BUDGETED_VIEW_SCAN,
        INAV_REACHABLE_TARGET_REGION,
        INAV_TARGET_VISUAL_MEMORY_FRONTIER,
        INAV_GLOBAL_FRONTIER,
        INAV_DUAL_DETECTOR_FUSION,
    )):
        raise RuntimeError(
            "R044 target lock must be evaluated as an isolated treatment"
        )
    if INAV_TARGET_LOCK_ACTION_BUDGET != 3:
        raise RuntimeError("R044 action budget is frozen at three")
    frozen_target_lock_values = (
        (INAV_TARGET_LOCK_MAX_STEP_M, 0.45, "maximum step"),
        (INAV_TARGET_LOCK_TERMINAL_RADIUS_M, 1.5, "terminal radius"),
        (INAV_TARGET_LOCK_ACTIVATION_RADIUS_M, 1.8, "activation radius"),
        (INAV_TARGET_LOCK_MIN_PROGRESS_M, 0.05, "minimum progress"),
        (INAV_TARGET_LOCK_MIN_TRANSLATION_M, 0.25, "minimum translation"),
        (
            INAV_TARGET_LOCK_MAX_DISTANCE_INCREASE_M,
            0.05,
            "maximum distance increase",
        ),
    )
    for actual, expected, label in frozen_target_lock_values:
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(
                f"R044 {label} is frozen at {expected} m"
            )
    if not math.isclose(
        INAV_TARGET_LOCK_ACTIVATION_RADIUS_M,
        INAV_VERIFY_DISTANCE_M,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            "R044 activation radius must equal the existing verifier radius"
        )
if INAV_TARGET_STANDOFF_COMMITMENT:
    if not INAV_GEOMETRY_SAFE_FRONTIER:
        raise RuntimeError(
            "R051 target standoff commitment requires R041 geometry safety"
        )
    if INAV_TARGET_STANDOFF_COMMITMENT_MAX_ACTIONS != 6:
        raise RuntimeError("R051 standoff commitment is frozen at six actions")
    if INAV_TARGET_STANDOFF_COMMITMENT_EPISODE_ACTION_BUDGET != 12:
        raise RuntimeError(
            "R051 standoff commitment episode budget is frozen at 12 actions"
        )
    if not math.isclose(
        INAV_TARGET_STANDOFF_COMMITMENT_MAX_DRIFT_M,
        0.75,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("R051 target drift release is frozen at 0.75 m")
if INAV_TARGET_BBOX_RECENTER:
    if any((
        INAV_TARGET_LOCK_CONTROLLER,
        INAV_TARGET_STANDOFF_COMMITMENT,
        INAV_TENTATIVE_MEMORY,
        INAV_POSE_EVIDENCE_GRAPH,
        INAV_PASSIVE_MULTI_VIEW,
        INAV_BELIEF_REPERCEPTION,
        INAV_SUPPORT_INSPECTION,
        INAV_CARRIER_ACTIVE_VERIFY,
        INAV_BOUNDED_COMMITMENT,
        INAV_BUDGETED_VIEW_SCAN,
        INAV_REACHABLE_TARGET_REGION,
        INAV_TARGET_VISUAL_MEMORY_FRONTIER,
        INAV_GLOBAL_FRONTIER,
        INAV_DUAL_DETECTOR_FUSION,
    )):
        raise RuntimeError(
            "R052 target bbox recenter must be evaluated as an isolated "
            "treatment"
        )
    if INAV_TARGET_BBOX_RECENTER_ACTION_BUDGET != 2:
        raise RuntimeError("R052 recenter budget is frozen at two actions")
    frozen_recenter_values = (
        (INAV_TARGET_BBOX_RECENTER_SCORE_MIN, 0.30, "score threshold"),
        (
            INAV_TARGET_BBOX_RECENTER_EDGE_FRACTION_MIN,
            0.15,
            "edge fraction",
        ),
        (
            INAV_TARGET_BBOX_RECENTER_MAX_ROTATION_DEG,
            45.0,
            "maximum rotation",
        ),
        (
            INAV_TARGET_BBOX_RECENTER_MEMORY_RADIUS_M,
            2.5,
            "memory radius",
        ),
    )
    for actual, expected, label in frozen_recenter_values:
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(
                f"R052 {label} is frozen at {expected}"
            )
if INAV_PERSISTENT_SINGLE_ENCODER_VETO:
    if any((
        INAV_TARGET_LOCK_CONTROLLER,
        INAV_TARGET_STANDOFF_COMMITMENT,
        INAV_TARGET_BBOX_RECENTER,
        INAV_TENTATIVE_MEMORY,
        INAV_POSE_EVIDENCE_GRAPH,
        INAV_PASSIVE_MULTI_VIEW,
        INAV_BELIEF_REPERCEPTION,
        INAV_SUPPORT_INSPECTION,
        INAV_CARRIER_ACTIVE_VERIFY,
        INAV_BOUNDED_COMMITMENT,
        INAV_BUDGETED_VIEW_SCAN,
        INAV_REACHABLE_TARGET_REGION,
        INAV_TARGET_VISUAL_MEMORY_FRONTIER,
        INAV_GLOBAL_FRONTIER,
        INAV_DUAL_DETECTOR_FUSION,
        INAV_TERMINAL_EVIDENCE_QUORUM,
    )):
        raise RuntimeError(
            "R053 persistent single-encoder veto must be evaluated as an "
            "isolated treatment"
        )
    if INAV_PERSISTENT_SINGLE_ENCODER_VETO_OBSERVATIONS != 4:
        raise RuntimeError(
            "R053 contradiction threshold is frozen at four observations"
        )
if INAV_TERMINAL_EVIDENCE_QUORUM:
    if any((
        INAV_PERSISTENT_SINGLE_ENCODER_VETO,
        INAV_TARGET_LOCK_CONTROLLER,
        INAV_TARGET_STANDOFF_COMMITMENT,
        INAV_TARGET_BBOX_RECENTER,
        INAV_TENTATIVE_MEMORY,
        INAV_POSE_EVIDENCE_GRAPH,
        INAV_PASSIVE_MULTI_VIEW,
        INAV_BELIEF_REPERCEPTION,
        INAV_SUPPORT_INSPECTION,
        INAV_CARRIER_ACTIVE_VERIFY,
        INAV_BOUNDED_COMMITMENT,
        INAV_BUDGETED_VIEW_SCAN,
        INAV_REACHABLE_TARGET_REGION,
        INAV_TARGET_VISUAL_MEMORY_FRONTIER,
        INAV_GLOBAL_FRONTIER,
        INAV_DUAL_DETECTOR_FUSION,
    )):
        raise RuntimeError(
            "R054 terminal evidence quorum must be evaluated as an isolated "
            "treatment"
        )
    if (
        INAV_TERMINAL_QUORUM_PERSISTENT_OBSERVATIONS != 4
        or INAV_TERMINAL_QUORUM_WEAK_RANK_MAX != 5
        or INAV_TERMINAL_QUORUM_CONTEXTUAL_MIN_VOTES != 2
    ):
        raise RuntimeError("R054 terminal quorum parameters are frozen")
if INAV_BELIEF_REPERCEPTION:
    if not 1 <= INAV_BELIEF_MAX_ATTEMPTS <= 4:
        raise RuntimeError("belief re-perception attempts must be in [1, 4]")
    if not 0.5 <= INAV_BELIEF_STEP_M <= 1.5:
        raise RuntimeError("belief re-perception step must be in [0.5, 1.5] m")
if INAV_BELIEF_COVERAGE_PRESERVING:
    if not INAV_BELIEF_REPERCEPTION:
        raise RuntimeError(
            "coverage-preserving belief requires belief re-perception"
        )
    if INAV_BELIEF_OPPORTUNISTIC_MAX_AGE != 1:
        raise RuntimeError(
            "R034 requires a current-or-next-frame belief (max age 1)"
        )
    if not 1.5 <= INAV_BELIEF_OPPORTUNISTIC_RADIUS_M <= 3.0:
        raise RuntimeError(
            "opportunistic belief radius must be in [1.5, 3.0] m"
        )
    if not 1 <= INAV_BELIEF_OPPORTUNISTIC_EPISODE_BUDGET <= 2:
        raise RuntimeError(
            "opportunistic belief episode budget must be one or two actions"
        )
    if not 0.5 <= INAV_BELIEF_OPPORTUNISTIC_MAX_MOVE_M <= 1.1:
        raise RuntimeError(
            "opportunistic belief move must be in [0.5, 1.1] m"
        )
# Best-effort per-episode seed.  Isaac rendering and some CUDA kernels may
# still be nondeterministic, so this is recorded as a seed rather than claimed
# as bit-for-bit determinism.
INAV_EXECUTION_SEED = int(os.environ.get("INAV_EXECUTION_SEED", "20260825"))
INAV_RENDER_ANTIALIASING_MODE = int(
    os.environ.get("INAV_RENDER_ANTIALIASING_MODE", "3")
)
if INAV_RENDER_ANTIALIASING_MODE not in {0, 1, 2, 3, 4}:
    raise RuntimeError(
        "INAV_RENDER_ANTIALIASING_MODE must be one of 0, 1, 2, 3, 4"
    )
INAV_RENDER_TICKS = int(os.environ.get("ISS_RENDER_TICKS", "8"))
if not 1 <= INAV_RENDER_TICKS <= 64:
    raise RuntimeError("ISS_RENDER_TICKS must be in [1, 64]")
INAV_INSTANCE_VISIBILITY_DIAGNOSTIC = (
    os.environ.get("INAV_INSTANCE_VISIBILITY_DIAGNOSTIC", "0") == "1"
)


def _seed_episode_execution(selection_id: str, style: str) -> int:
    numeric_id = int("".join(ch for ch in selection_id if ch.isdigit()) or "0")
    style_index = {"formal": 0, "natural": 1, "casual": 2,
                   "emotional": 3}.get(style, 0)
    seed = INAV_EXECUTION_SEED + numeric_id * 1009 + style_index * 97
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except Exception:
        pass
    return seed


def load_episodes() -> dict[str, dict]:
    out = {}
    with EPISODES_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            out[r["selection_id"]] = r
    return out


# ---------------- Scene-candidate planning baseline ----------------
SCENE_CANDIDATE_PLAN_SYSTEM = (
    "You are an embodied-navigation planner. Given an indirect human intent "
    "and a compact inventory of object categories that exist in the current "
    "scene, infer which object in THIS scene most likely satisfies the intent. "
    "Choose target_guess from scene_candidate_categories whenever possible. "
    "Use room_object_inventory only as scene context; it is not a target label. "
    "Output JSON only with target_guess, candidate_objects, likely_rooms, "
    "strategy, and action_plan."
)

_scene_plan_cache: dict[tuple[str, str, str], dict] = {}
_BG_SCENE_LABELS = {
    "wall", "floor", "ceiling", "window", "door", "entrance", "room",
    "living_room", "bedroom", "bathroom", "kitchen", "dining_room",
    "study_room", "balcony", "hallway", "corridor",
}


def _norm_label(x) -> str:
    lab = str(x or "").strip().lower().replace(" ", "_")
    return "" if lab in _BG_SCENE_LABELS else lab


def _load_manifest_entries(scene_id: str) -> list[dict]:
    path = scene_manifest_path(scene_id)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("entries", list(raw.values()))
    return raw if isinstance(raw, list) else []


def _scene_candidate_context(scene_id: str) -> dict:
    """Build a compact scene inventory from the dataset manifest.

    This is a diagnostic metadata-assisted baseline: it exposes object
    categories present in the scene, but does not mark the ground-truth target.
    """
    entries = _load_manifest_entries(scene_id)
    categories: set[str] = set()
    rooms: set[str] = set()
    by_room: dict[str, set[str]] = {}
    for entry in entries:
        room = str(entry.get("room") or "").strip()
        if room:
            rooms.add(room)
            by_room.setdefault(room, set())
        for key in ("target_category", "surface_category"):
            lab = _norm_label(entry.get(key))
            if lab:
                categories.add(lab)
                if room:
                    by_room[room].add(lab)
        for lab0 in entry.get("room_objects") or []:
            lab = _norm_label(lab0)
            if lab:
                categories.add(lab)
                if room:
                    by_room[room].add(lab)
        for td in entry.get("target_details") or []:
            lab = _norm_label(td.get("category"))
            td_room = str(td.get("room") or room or "").strip()
            if lab:
                categories.add(lab)
                if td_room:
                    rooms.add(td_room)
                    by_room.setdefault(td_room, set()).add(lab)
            nearby = td.get("nearby_objects") or {}
            if isinstance(nearby, dict):
                for meta in nearby.values():
                    lab = _norm_label((meta or {}).get("category"))
                    if lab:
                        categories.add(lab)
                        if td_room:
                            by_room.setdefault(td_room, set()).add(lab)

    # Keep the prompt bounded and deterministic.
    cats = sorted(categories)[:120]
    room_lines = []
    for room in sorted(rooms)[:24]:
        objs = sorted(by_room.get(room, set()))[:30]
        if objs:
            room_lines.append(f"{room}: {', '.join(objs)}")
    return {
        "scene_candidate_categories": cats,
        "scene_rooms": sorted(rooms)[:40],
        "room_object_inventory": room_lines[:24],
    }


def make_scene_candidate_plan(intent: str, item: dict, model_key: str,
                              *, sel_id: str, style: str
                              ) -> tuple[dict, dict | None]:
    """Intent planner with manifest-derived scene candidates.

    The navigation engine remains unchanged; only the episode-start target
    inference gets scene inventory context inspired by scene-graph/RAG methods.
    """
    cache_key = (sel_id, style, model_key)
    cached = _scene_plan_cache.get(cache_key)
    if cached is not None:
        return {**cached, "_cached": True}, None

    ctx = _scene_candidate_context(item["scene_id"])
    user_text = (
        f"User intent: \"{intent}\"\n\n"
        "scene_candidate_categories:\n"
        f"{json.dumps(ctx['scene_candidate_categories'], ensure_ascii=False)}\n\n"
        "scene_rooms:\n"
        f"{json.dumps(ctx['scene_rooms'], ensure_ascii=False)}\n\n"
        "room_object_inventory:\n"
        + "\n".join(f"- {line}" for line in ctx["room_object_inventory"])
        + "\n\n"
        'Output JSON exactly: {"target_guess":"<one category from the scene list>",'
        '"candidate_objects":["<3-7 scene categories, include target_guess>"],'
        '"likely_rooms":["<1-3 scene rooms or room types>"],'
        '"strategy":"<10-25 words imperative>",'
        '"action_plan":["step 1 <=15 words","step 2 <=15 words"]}'
    )
    out, err, usage = call_vlm(
        model_key, SCENE_CANDIDATE_PLAN_SYSTEM, user_text,
        image_bytes=None, temperature=0.0, json_mode=True, max_retries=3,
    )
    plan = {"target_guess": "", "candidate_objects": [], "likely_rooms": [],
            "strategy": "", "action_plan": [],
            "scene_candidate_categories": ctx["scene_candidate_categories"],
            "scene_rooms": ctx["scene_rooms"]}
    if out:
        parsed, _perr = tolerant_json_parse(out)
        if isinstance(parsed, dict):
            plan["target_guess"] = _norm_label(parsed.get("target_guess"))
            co = parsed.get("candidate_objects") or []
            plan["candidate_objects"] = [_norm_label(x) for x in co if _norm_label(x)][:7]
            lr = parsed.get("likely_rooms") or []
            plan["likely_rooms"] = [str(x).strip() for x in lr if str(x).strip()][:3]
            plan["strategy"] = str(parsed.get("strategy", "") or "").strip()[:300]
            ap = parsed.get("action_plan") or []
            if isinstance(ap, list):
                plan["action_plan"] = [str(x).strip() for x in ap if str(x).strip()][:6]
    if err:
        plan["error"] = err

    scene_set = set(ctx["scene_candidate_categories"])
    if plan["target_guess"] and plan["target_guess"] not in scene_set:
        plan["target_guess_out_of_scene_list"] = plan["target_guess"]
    if not plan["candidate_objects"] and plan["target_guess"]:
        plan["candidate_objects"] = [plan["target_guess"]]
    if not err:
        _scene_plan_cache[cache_key] = dict(plan)
    return plan, usage


# ---------------- VLM prompts ----------------
SEE_TARGET_SYSTEM = (
    "You are a visual target-detection module for a navigation agent. "
    "Look at the first-person RGB and answer questions about whether the "
    "target object is visible AND how relevant the current view is to "
    "finding it. Output JSON only.\n\n"
    'Schema: {"see_target": true|false, '
    '"confidence": "low"|"medium"|"high", '
    '"pixel_x_norm": <float in [0,1] OR null>, '
    '"pixel_y_norm": <float in [0,1] OR null>, '
    '"observed_objects": ["<obj1>", ...], '
    '"relevance_score": <int 0..10>, '
    '"explore_direction": "forward" | "left" | "right" | "behind" | "no_clue", '
    '"reason": "<=20 words"}\n\n'
    "Rules:\n"
    "- pixel_x_norm/pixel_y_norm: image-space center of the target ONLY "
    "if see_target=true. Use null if you only see partial/distant.\n"
    "- observed_objects: 4-8 salient objects in the scene (helps the engine "
    "later judge progress).\n"
    "- relevance_score: 0 = wrong room/area, 5 = related items visible but "
    "no target, 10 = target directly in view. Be discriminating — 5 is the "
    "default for 'might be near'.\n"
    "- explore_direction: where would you go to find <target> if not in view? "
    "'no_clue' if you can't tell from current view. Use 'forward' when target "
    "is visible.\n"
    "- Be conservative on see_target: medium/high confidence only when the "
    "target is clearly visible and identifiable. Low when ambiguous.\n"
    "- DO NOT propose actions or waypoint choices. The engine handles "
    "navigation; you provide cognitive signal only."
)


CLOSE_VERIFY_SYSTEM = (
    "You are a strict object verifier for a navigation agent that just "
    "stopped near a candidate object. Look at the first-person RGB and "
    "decide whether the navigation TARGET is the dominant object directly "
    "in front of the camera (within 2 meters), distinct from visually "
    "similar furniture.\n\n"
    'Schema: {"is_target": true|false, "reason": "<=15 words"}\n\n'
    "Rules:\n"
    "- Be CONSERVATIVE. Say true ONLY if the target is clearly the main "
    "object centered in the lower-middle of the frame and visually matches "
    "the target category (not a similar item).\n"
    "- For 'sofa', reject armchair / loveseat / cushion / bench.\n"
    "- For 'bed', reject sofa / mattress-on-floor / large cushion.\n"
    "- For 'toilet', reject bidet / bathroom_sink / urinal / random fixture.\n"
    "- For 'desk', reject dining_table / coffee_table / countertop.\n"
    "- For 'cabinet', reject nightstand / dresser / wardrobe / bookshelf.\n"
    "- If the target is NOT clearly visible in the lower-half of the image, "
    "say false.\n"
    "- JSON only."
)


def call_close_verify(model_key: str, rgb_bytes: bytes, target_name: str,
                      *, sel_id: str, style: str, step: int):
    """Strict yes/no: is `target_name` the dominant object directly in front
    of the camera RIGHT NOW? Used to gate strong_stop and reject
    category-confusion clusters (e.g. armchair admitted as sofa).
    Returns (verdict_dict, usage, err)."""
    user_text = (
        f"Target object: {target_name}\n"
        "Is this the target, directly in front, right now? JSON only."
    )
    resp, err, usage = call_vlm(model_key, CLOSE_VERIFY_SYSTEM, user_text,
                                  rgb_bytes, temperature=0.0, json_mode=True)
    if resp is None:
        return None, usage, err
    return resp, usage, err


def call_see_target(model_key: str, rgb_bytes: bytes, target_name: str,
                    intent: str, *, sel_id: str, style: str, step: int,
                    robustness_note: str | None = None):
    """Single VLM call: see target? Returns dict with see_target/confidence/
    pixel_x_norm/pixel_y_norm/observed_objects/reason; usage; err."""
    user_text = (
        f"Intent: {intent}\n"
        f"Target object: {target_name}\n"
        "Look at the image and answer the schema. JSON only."
    )
    if robustness_note:
        user_text = (
            f"Intent: {intent}\n"
            f"Target object: {target_name}\n"
            f"Robustness note: {robustness_note}\n"
            "Look at the image and answer the schema. JSON only."
        )
    # Inline retry mirroring call_vlm_step in agent_vlm
    def _one():
        return call_vlm(model_key, SEE_TARGET_SYSTEM, user_text, rgb_bytes,
                         temperature=0.0, json_mode=True)
    resp, err, usage = _one()
    retried = False
    if resp is None and FALLBACK_RETRY_ENABLED:
        print(f"[see_target/retry] {sel_id}/{style} step {step} "
              f"first attempt {err!r}; sleeping {FALLBACK_RETRY_SLEEP_S}s",
              file=sys.stderr)
        time.sleep(FALLBACK_RETRY_SLEEP_S)
        resp2, err2, usage2 = _one()
        if resp2 is not None:
            resp, err, usage = resp2, None, usage2
            retried = True
    if resp is None:
        return None, usage, err, retried
    parsed, perr = tolerant_json_parse(resp)
    if parsed is None:
        return None, usage, f"parse: {perr}", retried
    # relevance_score safe-parse to int 0..10
    try:
        rel = int(parsed.get("relevance_score", 0))
    except (TypeError, ValueError):
        rel = 0
    rel = max(0, min(10, rel))
    explore_dir = str(parsed.get("explore_direction") or "no_clue").lower()
    if explore_dir not in ("forward", "left", "right", "behind", "no_clue"):
        explore_dir = "no_clue"
    out = {
        "see_target": bool(parsed.get("see_target")),
        "confidence": str(parsed.get("confidence") or "").lower(),
        "pixel_x_norm": parsed.get("pixel_x_norm"),
        "pixel_y_norm": parsed.get("pixel_y_norm"),
        "observed_objects": [str(o).strip().lower()
                              for o in (parsed.get("observed_objects") or [])
                              if str(o).strip()][:8],
        "relevance_score": rel,
        "explore_direction": explore_dir,
        "reason": str(parsed.get("reason") or "")[:120],
        "raw": resp,
    }
    return out, usage, None, retried


def _backproject_pixel_to_world(px_norm: float, py_norm: float,
                                  depth: np.ndarray | None,
                                  pose: dict, hfov_rad: float = math.pi / 2,
                                  ) -> tuple[float, float] | None:
    """Pixel-norm + depth + pose → world (x, y). Returns None if no valid
    depth at the pixel. Mirrors the back-projection in agent_vlm.py around
    the VLM target_marker handler."""
    if depth is None or px_norm is None or py_norm is None:
        return None
    H_d, W_d = depth.shape[:2]
    px = int(round(float(px_norm) * W_d))
    py = int(round(float(py_norm) * H_d))
    if not (0 <= py < H_d and 0 <= px < W_d):
        return None
    # Median over a small window rejects isolated edge-depth noise without
    # changing the detector anchor to a different surface in a large box.
    half = int(os.environ.get("VM_BACKPROJECT_DEPTH_HALF", "3"))
    y0, y1 = max(0, py - half), min(H_d, py + half + 1)
    x0, x1 = max(0, px - half), min(W_d, px + half + 1)
    win = depth[y0:y1, x0:x1]
    valid = win[(win > 0.05) & np.isfinite(win)]
    if valid.size == 0:
        return None
    d = float(np.median(valid))
    u_norm = (px / W_d) * 2.0 - 1.0   # [-1, 1]
    bearing_offset = -u_norm * (hfov_rad / 2.0)
    world_yaw = pose["yaw"] + bearing_offset
    return (pose["position"][0] + d * math.cos(world_yaw),
            pose["position"][1] + d * math.sin(world_yaw))


def _dino_projection(
    rgb: np.ndarray,
    depth: np.ndarray | None,
    bbox: list[float],
    pose: dict,
) -> tuple[tuple[float, float] | None, float | None, dict]:
    """Project one DINO box, optionally replacing its ray with a mask."""
    robust_depth = bbox_depth_m(depth, bbox)
    if depth is None:
        return None, robust_depth, {"enabled": INAV_MASK_REFINEMENT,
                                    "used": False, "error": "no_depth"}
    height, width = depth.shape[:2]
    cx_px = (bbox[0] + bbox[2]) / 2.0
    cy_px = (
        bbox[1] * (1 - VM_DINO_BBOX_Y_FRAC)
        + bbox[3] * VM_DINO_BBOX_Y_FRAC
    )
    fallback_xy = _backproject_pixel_to_world(
        cx_px / width if width > 0 else 0.5,
        cy_px / height if height > 0 else 0.5,
        depth,
        pose,
    )
    if not INAV_MASK_REFINEMENT:
        return fallback_xy, robust_depth, {"enabled": False, "used": False}

    refinement = mask_refiner.refine_bbox(rgb, bbox)
    public_meta = {key: value for key, value in refinement.items()
                   if key != "mask"}
    if not refinement.get("available", False):
        return fallback_xy, robust_depth, {
            "enabled": True, "used": False, **public_meta,
        }
    mask_xy, depth_meta = mask_refiner.mask_depth_world_xy(
        refinement.get("mask"), depth, pose
    )
    if mask_xy is None:
        return fallback_xy, robust_depth, {
            "enabled": True, "used": False, **public_meta,
            "projection": depth_meta,
        }
    # Keep the legacy box anchor for global association and approach.  A
    # visible mask surface changes across viewpoints and is therefore a poor
    # object-map centroid, but it is exactly the geometry needed by STOP.
    return fallback_xy, float(depth_meta["depth_median"]), {
        "enabled": True,
        "used": True,
        **public_meta,
        "surface_xy": [round(float(mask_xy[0]), 6),
                       round(float(mask_xy[1]), 6)],
        "projection": depth_meta,
    }


def _is_admit_xy_on_wall(wm, xy: tuple[float, float]) -> bool:
    """Fix D1: True iff back-projected XY is on a non-walkable cell AND
    no walkable cell exists within VM_WALL_PIERCE_TOLERANCE_M. Used to
    reject admits where depth measurement returned the wall, not the
    actual object visible behind it. Tolerance handles rasterization
    edges (object center near wall, but bbox center backprojects onto
    wall by 1-2 cells)."""
    if wm.is_walkable(xy[0], xy[1]):
        return False
    near = wm.nearby_walkable(xy[0], xy[1], radius_m=VM_WALL_PIERCE_TOLERANCE_M)
    return near is None


def _yaw_spread(yaws: list[float]) -> float:
    """Maximum pairwise circular angular distance among yaws (radians).
    Wraps around 2π. Returns 0 for ≤1 yaw."""
    if not yaws or len(yaws) < 2:
        return 0.0
    spread = 0.0
    for i in range(len(yaws)):
        for j in range(i + 1, len(yaws)):
            d = abs(yaws[i] - yaws[j]) % (2 * math.pi)
            d = min(d, 2 * math.pi - d)
            if d > spread:
                spread = d
    return spread


def _local_target_proposals(
    rgb: np.ndarray,
    target_guess: str,
    phrases: list[str],
    dino_threshold: float,
    counters: dict,
    *,
    budgeted_scan_frame: bool = False,
) -> tuple[list[tuple[str, float, list[float]]], list[dict | None], str]:
    """Run the category specialist first and fall back on DINO after a miss.

    The returned metadata is aligned with the proposal list.  COCO proposals
    carry an already-verified closed-set semantic record; DINO proposals carry
    ``None`` and retain the independent CLIP verification stage downstream.
    """
    if budgeted_scan_frame and INAV_BUDGETED_SCAN_YOLO_WORLD:
        counters["yolo_world_scan_calls"] = (
            counters.get("yolo_world_scan_calls", 0) + 1
        )
        try:
            yolo_world = _load_yolo_world_verifier()
            proposals, metadata = yolo_world.target_proposals(
                rgb,
                target_guess,
                confidence=INAV_BUDGETED_SCAN_YOLO_CONFIDENCE,
            )
        except Exception as exc:
            counters["yolo_world_scan_errors"] = (
                counters.get("yolo_world_scan_errors", 0) + 1
            )
            counters["yolo_world_scan_last_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            raise RuntimeError(
                "required budgeted-scan YOLO-World detector failed"
            ) from exc
        if proposals:
            counters["yolo_world_scan_hits"] = (
                counters.get("yolo_world_scan_hits", 0) + 1
            )
        return proposals, metadata, "yolo_world_scan"

    if INAV_COCO_SPECIALIST and coco_detector.coco_labels(target_guess):
        counters["coco_specialist_calls"] = (
            counters.get("coco_specialist_calls", 0) + 1
        )
        try:
            proposals = coco_detector.target_proposals(rgb, target_guess)
        except Exception as exc:
            counters["coco_specialist_errors"] = (
                counters.get("coco_specialist_errors", 0) + 1
            )
            counters["coco_specialist_last_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            if INAV_COCO_STOP_GATE or INAV_CONTEXTUAL_STOP_CONSENSUS:
                raise RuntimeError(
                    "COCO specialist failed while the terminal gate was active"
                ) from exc
            proposals = []
        if proposals:
            counters["coco_specialist_hits"] = (
                counters.get("coco_specialist_hits", 0) + 1
            )
            metadata = [
                coco_detector.semantic_verification(target_guess, proposal[1])
                for proposal in proposals
            ]
            return proposals, metadata, "coco_specialist"
        counters["coco_specialist_fallbacks"] = (
            counters.get("coco_specialist_fallbacks", 0) + 1
        )

    proposals = dino_detector.detect(
        rgb, phrases, threshold=float(dino_threshold)
    )
    return proposals, [None for _ in proposals], "dino"


def _specialist_stop_gate(
    target_guess: str,
    detector_source: str,
    semantic_meta: dict | None,
    *,
    enabled: bool | None = None,
) -> tuple[bool, dict]:
    """Require current closed-set evidence only for exactly mapped targets.

    The gate never changes exploration or memory admission.  A specialist
    miss may therefore fall back to DINO for navigation, but correlated
    target-only DINO/CLIP evidence cannot by itself terminate a mapped COCO
    category.  No target pose, target room, or simulator label is consulted.
    """
    gate_enabled = INAV_COCO_STOP_GATE if enabled is None else bool(enabled)
    supported = bool(coco_detector.coco_labels(target_guess))
    required = bool(INAV_COCO_SPECIALIST and gate_enabled and supported)
    verification = (semantic_meta or {}).get("specialist_verification") or {}
    observed_label = str(verification.get("best_label") or "")
    current_positive = bool(
        detector_source == "coco_specialist"
        and verification.get("available") is True
        and verification.get("accepted") is True
        and normalize_category(observed_label) == normalize_category(target_guess)
    )
    accepted = not required or current_positive
    return accepted, {
        "required": required,
        "supported_target": supported,
        "current_positive": current_positive,
        "detector_source": detector_source,
        "observed_label": observed_label or None,
    }


def _sensor_protocol(
    *, auto_reorient: bool, pano_init: bool, mid_pano_scan: bool,
    budgeted_view_scan: bool = False,
) -> str:
    """Name whether the policy receives any unbudgeted camera scan."""
    if auto_reorient or pano_init or mid_pano_scan:
        return "legacy_free_camera_scans"
    if budgeted_view_scan:
        return "budgeted_rotation_actions_no_free_scans"
    return "forward_view_only_no_free_scans"


def _use_reachable_target_region(navigation_cluster_kind: str | None) -> bool:
    """Route the relaxed goal region only for admitted target memory.

    Tentative proposals are deliberately non-terminal and may receive only
    their separately budgeted information action.  They must never inherit
    the target-region controller merely because both paths share
    ``approach_waypoint``.
    """
    return bool(
        INAV_REACHABLE_TARGET_REGION
        and navigation_cluster_kind == "target"
    )


def _do_pano_scan(env, wm, vmap, target_memory, target_guess: str,
                   candidates: list[str], step_for_admit: int,
                   source_tag: str, dino_threshold: float,
                   counters: dict, sel_id: str, style: str) -> int:
    """Generic 4-yaw panoramic DINO scan.

    Rotates camera to 3 additional yaws (current + 90, 180, 270), renders
    RGB+depth at each, runs DINO with `candidates` phrases, admits any
    detection that survives the room+wall+label gates. Restores original
    yaw at end.

    Used by both episode-start pano-init AND mid-episode re-scan when
    agent enters a new likely_room.

    Returns: number of clusters admitted.
    """
    if not target_guess:
        return 0
    initial_yaw = env.get_pose()["yaw"]
    target_queries = target_detector_queries(target_guess)
    if INAV_VISUAL_TARGET_DESCRIPTORS:
        target_queries = visual_detector_queries(
            target_guess, target_queries, max_queries=5
        )
    phrases = list(target_queries)
    if not INAV_TARGET_ONLY_DINO:
        for c in candidates:
            if c and c not in phrases:
                phrases.append(c)
    phrases = phrases[:5]
    admitted = 0
    try:
        for offset_deg in (90, 180, 270):
            env.look_at_yaw(initial_yaw + math.radians(offset_deg))
            rgb_p = env.render_rgb()
            depth_p = env.render_depth() if hasattr(env, "render_depth") else None
            pose_p = env.get_pose()
            try:
                dets, proposal_metadata, detector_source = (
                    _local_target_proposals(
                        rgb_p,
                        target_guess,
                        phrases,
                        dino_threshold,
                        counters,
                    )
                )
            except Exception as e:
                print(f"[{source_tag}/detector] {sel_id}/{style} "
                      f"yaw+{offset_deg}: {e}",
                      file=sys.stderr)
                if ((INAV_COCO_STOP_GATE
                     or INAV_CONTEXTUAL_STOP_CONSENSUS)
                        and coco_detector.coco_labels(target_guess)):
                    raise
                dets = []
                proposal_metadata = []
                detector_source = "dino"
            if depth_p is None:
                continue
            # Filter to strict-label match (same gate as per-step DINO admit)
            non_target_cands = {
                c.lower().strip().replace('_', ' ')
                for c in candidates if c.lower().strip() != target_guess.lower().strip()
            }
            target_pairs = []
            for detection, semantic_meta in zip(dets, proposal_metadata):
                label, _, _ = detection
                lab = str(label).lower().strip().replace('_', ' ')
                if lab in non_target_cands:
                    continue
                if not target_query_match(lab, target_queries):
                    continue
                target_pairs.append((detection, semantic_meta))
            target_dets = [pair[0] for pair in target_pairs]
            if detector_source == "coco_specialist":
                independent_clip = (
                    clip_verifier.verify_detections(
                        rgb_p, target_dets, target_guess,
                        max_rank=INAV_CLIP_ADMIT_MAX_RANK,
                    )
                    if INAV_CLIP_VERIFY else
                    [{"available": False, "accepted": True}
                     for _ in target_dets]
                )
                clip_results = [
                    {**clip_meta, "specialist_verification": specialist_meta}
                    for clip_meta, (_, specialist_meta) in zip(
                        independent_clip, target_pairs
                    )
                ]
            else:
                clip_results = (
                    clip_verifier.verify_detections(
                        rgb_p, target_dets, target_guess,
                        max_rank=INAV_CLIP_ADMIT_MAX_RANK,
                    )
                    if INAV_CLIP_VERIFY else
                    [{"available": False, "accepted": True}
                     for _ in target_dets]
                )
            verified_dets = [
                (detection, clip_meta)
                for detection, clip_meta in zip(target_dets, clip_results)
                if (
                    detector_source == "coco_specialist"
                    or not INAV_CLIP_VERIFY
                    or clip_verifier.supports_target(
                        clip_meta,
                        target_guess,
                        component_max_rank=INAV_CLIP_ADMIT_MAX_RANK,
                    )
                )
            ]
            # One independent instance hypothesis per view.  Admitting every
            # DINO box inflated a single false-positive frame into many
            # pseudo-observations of one cluster.
            if verified_dets:
                verified_dets = [max(verified_dets, key=lambda pair: pair[0][1])]
            for (label, score, bbox), clip_meta in verified_dets:
                box_ok, _ = stop_box_quality(bbox, rgb_p.shape)
                if INAV_EVIDENCE_NAV and not box_ok:
                    continue
                xy, robust_depth, mask_meta = _dino_projection(
                    rgb_p, depth_p, bbox, pose_p
                )
                if xy is None:
                    continue
                if VM_DINO_ADMIT_ROOMS_ONLY \
                        and not vmap.is_in_likely_room(xy[0], xy[1]):
                    counters["dino_admit_room_rejected"] = \
                        counters.get("dino_admit_room_rejected", 0) + 1
                    continue
                if VM_WALL_PIERCE_REJECT and _is_admit_xy_on_wall(wm, xy):
                    counters["wall_pierce_rejected"] = \
                        counters.get("wall_pierce_rejected", 0) + 1
                    continue
                admitted_cluster = _admit_cluster(
                    target_memory, xy,
                    float(min(score, 0.9)),
                    label, step_for_admit,
                    (
                        source_tag
                        if detector_source == "dino"
                        else f"{source_tag}_{detector_source}"
                    ),
                    admit_yaw=pose_p["yaw"],
                    observer_xy=(pose_p["position"][0],
                                 pose_p["position"][1]),
                    depth_m=robust_depth, bbox=bbox,
                )
                if INAV_EVIDENCE_NAV:
                    admitted_cluster["observations"][-1][
                        "clip_verification"
                    ] = clip_meta
                    admitted_cluster["observations"][-1][
                        "mask_refinement"
                    ] = mask_meta
                admitted += 1
        env.look_at_yaw(initial_yaw)
    except Exception as e:
        print(f"[{source_tag}] {sel_id}/{style} EXC {e}", file=sys.stderr)
        if ((INAV_COCO_STOP_GATE or INAV_CONTEXTUAL_STOP_CONSENSUS)
                and coco_detector.coco_labels(target_guess)):
            raise
    return admitted


def _admit_cluster(target_memory: list[dict], xy: tuple[float, float],
                   score: float, label: str, step: int, source: str,
                   admit_yaw: float = 0.0,
                   cluster_radius_m: float = 0.75,
                   observer_xy: tuple[float, float] | None = None,
                   depth_m: float | None = None,
                   bbox: list[float] | None = None):
    """Merge new admit into existing cluster within radius, else append.

    XY estimation uses score-weighted EMA across observations (not score-max
    overwrite), so multi-frame admits of the same object converge to a
    stable centroid. Single-frame back-projection noise is ~0.3-0.6m;
    averaging 3-5 admits cuts that noise by 1/sqrt(N). Strong_stop fires
    on cluster XY directly, so XY stability translates to SR (cluster off
    by 0.5m → agent stops 0.5m past target → SR=0 if real-target distance
    crosses 2m boundary)."""
    if INAV_EVIDENCE_NAV:
        return admit_observation(
            target_memory,
            xy=xy,
            score=score,
            label=label,
            step=step,
            source=source,
            observer_xy=observer_xy or xy,
            yaw=admit_yaw,
            depth_m=depth_m,
            bbox=bbox,
            cluster_radius_m=cluster_radius_m,
        )

    tx, ty = xy
    for mem in target_memory:
        if math.hypot(mem["xy"][0] - tx, mem["xy"][1] - ty) < cluster_radius_m:
            mem["step"] = step
            # Score-weighted running mean: each admit contributes weight ∝ score
            old_w = mem.get("xy_weight", float(mem.get("score", 0.5)))
            new_w = float(score)
            total = old_w + new_w
            if total > 0:
                mem["xy"] = [(mem["xy"][0]*old_w + tx*new_w) / total,
                              (mem["xy"][1]*old_w + ty*new_w) / total]
                mem["xy_weight"] = total
            mem["score"] = max(mem.get("score", 0), score)  # keep max score
            if score >= mem.get("best_label_score", 0):
                mem["label"] = label
                mem["best_label_score"] = score
            mem.setdefault("sources", []).append(source)
            mem.setdefault("admit_yaws", []).append(float(admit_yaw))
            mem["n_observations"] = mem.get("n_observations", 1) + 1
            return mem
    new = {"xy": [tx, ty], "score": score, "step": step,
           "label": label, "sources": [source],
           "admit_yaws": [float(admit_yaw)],
           "n_observations": 1, "xy_weight": float(score),
           "best_label_score": float(score)}
    target_memory.append(new)
    return new


# ---------------- Episode runner ----------------
def run_episode_engine(env, wm, item, episode_meta, style, model_key,
                       step_cap, out_path, objectnav=False,
                       scene_candidates=False,
                       robustness_mode: str = "none",
                       robustness_seed: int = 0,
                       local_perception_only: bool = False,
                       intent_predictions_dir: Path | None = None):
    """Engine-driven episode loop. See module docstring."""
    sel_id = item["selection_id"]
    epi_dir = out_path.parent
    execution_seed = _seed_episode_execution(sel_id, style)

    # ---- Plan ----
    robustness_mode = normalize_robustness_mode(robustness_mode)
    robustness_meta = make_target_absent_robustness(
        robustness_mode,
        item=item,
        episode_meta=episode_meta,
        seed=robustness_seed,
        style=style,
    )
    episode_meta_for_record = (
        robustness_meta.get("episode_meta") if robustness_meta else episode_meta
    )
    robustness_note = (
        robustness_meta.get("prompt_note") if robustness_meta else None
    )
    target_absent = bool(
        robustness_meta and robustness_meta.get("mode") == "target_absent"
    )

    open_weight_provenance = None
    if target_absent:
        tier_name = "vlm_engine"
        tgt_cat = robustness_meta["absent_target_category"]
        intent = (
            f"Navigate to a {tgt_cat}. The object may be absent; search "
            "thoroughly and only stop if it is clearly visible."
        )
        plan = {"target_guess": tgt_cat, "candidate_objects": [tgt_cat],
                "likely_rooms": [],
                "strategy": f"Search for a {tgt_cat}; do not STOP unless visible.",
                "action_plan": []}
        plan_usage = None
    elif objectnav:
        tier_name = "explicit_objectnav"
        # The versioned dataset is authoritative for taxonomy corrections;
        # episode geometry may retain the original simulator asset label.
        tgt_cat = item.get("target_category", "") or (episode_meta or {}).get(
            "target_category"
        )
        if not tgt_cat:
            raise ValueError(f"objectnav mode but no target_category for {sel_id}")
        intent = f"Navigate to a {tgt_cat}."
        plan = {"target_guess": tgt_cat, "candidate_objects": [tgt_cat],
                # Never read episode_meta.target_room here: ObjectNav gives
                # the category, not the ground-truth room instance.
                "likely_rooms": likely_rooms_for_object(tgt_cat),
                "strategy": f"Walk toward and STOP next to a {tgt_cat}.",
                "action_plan": []}
        plan_usage = None
    elif intent_predictions_dir is not None:
        tier_name = "open_weight_intentionnav"
        intent = intent_for_style(item, style)
        if not intent:
            raise ValueError(f"empty intent for {sel_id}/{style}")
        plan, open_weight_provenance = load_open_weight_policy_plan(
            intent_predictions_dir,
            item,
            style,
            intent,
        )
        plan_usage = None
    else:
        intent = intent_for_style(item, style)
        if not intent:
            raise ValueError(f"empty intent for {sel_id}/{style}")
        if scene_candidates:
            tier_name = "scene_candidate"
            plan, plan_usage = make_scene_candidate_plan(
                intent, item, model_key, sel_id=sel_id, style=style,
            )
        else:
            tier_name = "vlm_engine"
            plan, plan_usage = make_episode_plan(
                intent, model_key, sel_id=sel_id, style=style,
            )
    if open_weight_provenance:
        protocol = open_weight_provenance["protocol_version"]
        suffix = (
            "r057_fusion_local"
            if protocol == OPEN_WEIGHT_INTENT_PROTOCOL_V3
            else "r057_context_local"
            if protocol == OPEN_WEIGHT_INTENT_PROTOCOL_V2
            else "r055_local"
        )
        record_model = (
            f"{_model_slug(open_weight_provenance['model_id'])}_{suffix}"
        )
    elif local_perception_only:
        record_model = "intentionnav_local"
    else:
        record_model = robustness_output_model_name(model_key, robustness_mode)
    target_guess = (plan.get("target_guess") or "").strip().lower()
    if STRICT_VLM_FAILURE and plan.get("error"):
        raise RuntimeError(f"VLM_PLAN_FAILED {sel_id}/{style}/{model_key}: "
                           f"{plan.get('error')}")
    candidates = [str(c).strip().lower()
                   for c in (plan.get("candidate_objects") or [])
                   if str(c).strip()]
    if target_guess and target_guess not in candidates:
        candidates = [target_guess] + candidates
    target_queries = target_detector_queries(target_guess)
    if INAV_VISUAL_TARGET_DESCRIPTORS:
        target_queries = visual_detector_queries(
            target_guess, target_queries, max_queries=5
        )

    # ---- Episode init ----
    env.place_agent(episode_meta["start_position"],
                    episode_meta["start_rotation_quat_wxyz"])
    if hasattr(wm, "reset_explored"):
        wm.reset_explored()
    AUTO_REORIENT = os.environ.get("AUTO_REORIENT", "1") == "1"
    AUTO_REORIENT_THRESHOLD_M = float(os.environ.get("AUTO_REORIENT_THRESHOLD_M", "1.0"))
    if INAV_BUDGETED_VIEW_SCAN and (
        AUTO_REORIENT or VM_PANO_INIT or VM_MID_PANO_SCAN
    ):
        raise RuntimeError(
            "budgeted view scan requires AUTO_REORIENT=0, VM_PANO_INIT=0, "
            "and VM_MID_PANO_SCAN=0 so every extra view is action-counted"
        )
    reorient_log = None
    if AUTO_REORIENT:
        reorient_log = env.find_open_view(walkable_map=wm,
                                            threshold_m=AUTO_REORIENT_THRESHOLD_M)

    pose = env.get_pose()
    if hasattr(wm, "mark_explored"):
        wm.mark_explored(
            pose["position"][0], pose["position"][1],
            radius_m=INAV_EXPLORED_RADIUS_M,
        )
    rooms_visited: list[str] = []
    r0 = wm.room_at(pose["position"][0], pose["position"][1])
    if r0:
        rooms_visited.append(r0)
    traj = [{"step": 0, "position": pose["position"], "yaw": pose["yaw"],
             "room": r0, "reorient": reorient_log}]

    # State
    target_memory: list[dict] = []
    tentative_memory: list[dict] = []
    agent_path: list[tuple[float, float]] = [(pose["position"][0], pose["position"][1])]
    seen_objects: set[str] = set()
    target: str = "" if target_absent else target_guess
    final_target: str = target
    usage_calls: list[dict] = []
    if plan_usage:
        usage_calls.append({**plan_usage, "step": 0, "kind": "plan"})

    vmap = ValueMap(wm, plan)
    stop_reason = "step_cap"
    final_rgb = None
    # Track consecutive VLM-see-target hits for STOP gate (suppresses
    # single-frame hallucination triggers).
    vlm_see_streak = 0
    # Diagnostic: which STOP path fired this episode.
    stop_path = None
    terminal_frame_source = None
    pano_init_admitted = 0
    dino_admit_room_rejected = 0  # DINO admits dropped by VM_DINO_ADMIT_ROOMS_ONLY
    wall_pierce_rejected = 0  # admits dropped by VM_WALL_PIERCE_REJECT (D1)
    strong_stop_demoted = 0  # strong_stops suppressed by D2 confirmation gate
    standard_stop_demoted = 0  # standard_stops suppressed by Q2.1 DINO-confirm gate
    dino_stop_fired = 0  # X1: detection-driven STOP triggered (DINO + depth → distance gate)
    evidence_stop_fired = 0
    evidence_verification_failures = 0
    evidence_verification_attempts = 0
    evidence_verification_successes = 0
    evidence_clusters_suppressed = 0
    dual_detector_calls = 0
    dual_detector_consensus_hits = 0
    dual_detector_unknowns = 0
    dual_detector_sibling_vetoes = 0
    bounded_commitment_attempts = 0
    bounded_commitment_redetections = 0
    bounded_commitment_misses = 0
    adaptive_stop_relaxations = {
        "score": 0,
        "distance": 0,
        "large_bbox": 0,
    }
    hybrid_frontier_moves = 0
    hybrid_frontier_fallbacks = 0
    global_frontier_moves = 0
    global_frontier_fallbacks = 0
    global_frontier_triggers = 0
    global_frontier_high_novelty_skips = 0
    geometry_safe_waypoint_checks = 0
    geometry_safe_waypoint_rejections = 0
    geometry_safe_global_recoveries = 0
    geometry_safe_hold_position_failures = 0
    geometry_safe_ledger: list[dict] = []
    target_lock_actions = 0
    target_lock_monotonic_moves = 0
    target_lock_tangential_moves = 0
    target_lock_reobservations = 0
    target_lock_outside_activation_radius = 0
    target_lock_actions_by_cluster: dict[str, int] = {}
    target_lock_action_ledger: list[dict] = []
    target_standoff_commitment: dict | None = None
    target_standoff_commitment_initializations = 0
    target_standoff_commitment_actions = 0
    target_standoff_commitment_releases = 0
    target_standoff_commitment_ledger: list[dict] = []
    target_bbox_recenter_actions = 0
    target_bbox_recenter_hits = 0
    target_bbox_recenter_misses = 0
    target_bbox_recenter_stop_vetoes = 0
    target_bbox_recenter_ledger: list[dict] = []
    pending_target_bbox_recenter_event = None
    persistent_single_encoder_stop_vetoes = 0
    persistent_single_encoder_stop_veto_ledger: list[dict] = []
    terminal_evidence_quorum_vetoes = 0
    terminal_evidence_quorum_ledger: list[dict] = []
    target_visual_memory_observations = 0
    target_visual_memory_informative = 0
    target_visual_frontier_calls = 0
    target_visual_frontier_moves = 0
    target_visual_frontier_fallbacks = 0
    target_visual_rerank_changes = 0
    target_visual_memory_ready = False
    target_visual_memory_ledger: list[dict] = []
    verification_candidate = None
    verification_candidate_from_bounded_commitment = False
    tentative_verification_candidate = None
    tentative_verification_action_step = None
    tentative_probes = 0
    tentative_probe_promotions = 0
    tentative_probe_failures = 0
    tentative_probe_resolutions: list[dict] = []
    pending_pose_graph_candidate = None
    pending_pose_graph_action_step = None
    pose_graph_actions = 0
    pose_graph_promotions = 0
    pose_graph_misses = 0
    pose_graph_unreachable_candidates = 0
    pose_graph_resolutions: list[dict] = []
    pose_graph_action_ledger: list[dict] = []
    pending_belief_candidate = None
    pending_belief_action_step = None
    belief_reperception_actions = 0
    belief_reperception_hits = 0
    belief_reperception_misses = 0
    belief_promotions = 0
    belief_reperception_resolutions: list[dict] = []
    belief_opportunistic_rejections: list[dict] = []
    support_inspection_detector_calls = 0
    support_inspection_candidates = 0
    support_inspections = 0
    support_inspection_target_hits = 0
    support_inspection_failures = 0
    support_inspection_errors = 0
    support_inspection_resolutions: list[dict] = []
    pending_support_candidate = None
    pending_support_action_step = None
    carrier_detector_calls = 0
    carrier_candidates = 0
    carrier_sessions_started = 0
    carrier_actions = 0
    carrier_target_hits = 0
    carrier_failures = 0
    carrier_resolutions: list[dict] = []
    carrier_rejected_xy: list[list[float]] = []
    active_carrier_candidate = None
    pending_carrier_action = None
    budgeted_scan_sites = 0
    budgeted_scan_actions = 0
    budgeted_scan_observation_frames = 0
    budgeted_scan_evidence_hits = 0
    budgeted_scan_tentative_hits = 0
    budgeted_scan_policy_evidence_hits = 0
    budgeted_scan_terminal_stops = 0
    budgeted_scan_anchors: dict[str, list[tuple[float, float]]] = {}
    budgeted_scan_site_ledger: list[dict] = []
    budgeted_scan_events: list[dict] = []
    active_budgeted_scan = None
    pending_budgeted_scan_observation = None
    mask_refinement_calls = 0
    mask_refinement_successes = 0
    perception_counters = {
        "coco_specialist_calls": 0,
        "coco_specialist_hits": 0,
        "coco_specialist_fallbacks": 0,
        "coco_specialist_errors": 0,
        "coco_stop_gate_checks": 0,
        "coco_stop_gate_positives": 0,
        "coco_stop_gate_vetoes": 0,
        "contextual_stop_checks": 0,
        "contextual_stop_positives": 0,
        "contextual_stop_vetoes": 0,
    }
    # Track per-step VLM see_target verdicts for D2 freshness check.
    recent_vlm_sees: list[tuple[int, bool, str]] = []  # (step, see, conf)
    # Fix H: mid-episode pano scan tracking — each likely-room scanned at most once
    # CRITICAL: pre-populate with the STARTING room so we don't trigger mid-pano
    # on step 1. Otherwise: when starting room is also in likely_rooms (e.g.
    # bed SEL with likely_rooms=[bedroom, living_room] and agent spawns in
    # living room), mid-pano admits bed-shaped objects (sofa/cushion) in
    # living room → agent commits to fake cluster, never goes to bedroom.
    # V15 cell showed bed OSR regression 0.75 → 0.5 from this.
    rooms_pano_scanned: set = set()
    if hasattr(wm, "room_at"):
        start_room = wm.room_at(pose["position"][0], pose["position"][1])
        if start_room:
            rooms_pano_scanned.add(start_room)
    mid_pano_scans = 0
    mid_pano_admitted = 0

    # ---- v2: Episode-start panoramic DINO scan (zero VLM cost) ----
    # Rotate camera 90/180/270° from initial yaw, render + run DINO,
    # admit any high-score detection to target_memory. Mirrors WMNav's
    # 360° goal-detection pass but DINO-only. Restores original yaw at
    # the end. ~4 renders + 4 DINO calls per episode (~1s wall).
    if VM_PANO_INIT and target_guess:
        counters = perception_counters
        counters["dino_admit_room_rejected"] = dino_admit_room_rejected
        counters["wall_pierce_rejected"] = wall_pierce_rejected
        pano_init_admitted = _do_pano_scan(
            env, wm, vmap, target_memory, target_guess, candidates,
            step_for_admit=0, source_tag="dino_pano_init",
            dino_threshold=VM_PANO_INIT_DINO_THRESHOLD,
            counters=counters, sel_id=sel_id, style=style)
        dino_admit_room_rejected = counters["dino_admit_room_rejected"]
        wall_pierce_rejected = counters["wall_pierce_rejected"]
        pose = env.get_pose()

    # ---- Per-step loop ----
    for step in range(1, step_cap + 1):
        # A prior ROTATE_SCAN action exposes exactly one image here, through
        # the same render/detector path as every other policy observation.
        # Clearing the pending marker now prevents any frame from being
        # double-counted as active-view evidence.
        scan_frame_meta = pending_budgeted_scan_observation
        pending_budgeted_scan_observation = None
        target_bbox_recenter_frame_meta = (
            dict(pending_target_bbox_recenter_event)
            if pending_target_bbox_recenter_event is not None else None
        )
        if hasattr(wm, "mark_explored"):
            wm.mark_explored(
                pose["position"][0], pose["position"][1],
                radius_m=INAV_EXPLORED_RADIUS_M,
            )

        # Fix H (Option C — rescue mode): mid-episode pano scan only when
        # target_memory has NO high-confidence cluster (score >= 0.5) AND
        # agent is in a new matched likely_room. Idea: if DINO already
        # found a plausible target, trust it; only pano-scan as rescue
        # when we're in a target room with no signal.
        #
        # V15 had broken trigger (fired in starting room, admitted fakes).
        # V16 fixed starting-room exclusion but still over-triggered on
        # rooms where DINO already found something (desk/dining_table
        # regression). C makes mid-pano fire only as last resort.
        if VM_MID_PANO_SCAN and target_guess:
            has_strong_cluster = any(
                float(c.get("score", 0)) >= 0.5 for c in target_memory
            )
            cur_room_full = wm.room_at(pose["position"][0], pose["position"][1]) \
                            if hasattr(wm, "room_at") else None
            if (not has_strong_cluster
                    and cur_room_full
                    and cur_room_full not in rooms_pano_scanned):
                cur_room_norm = wm.normalize_room_type(cur_room_full).lower() \
                                 if hasattr(wm, "normalize_room_type") else cur_room_full
                matched = vmap.matched_rooms_diagnostic if hasattr(vmap, "matched_rooms_diagnostic") else []
                room_prior_match = any(
                    m == cur_room_norm
                    or m in cur_room_norm
                    or cur_room_norm in m
                    for m in matched
                )
                # Optional robustness ablation: broaden the one-time scan to
                # every newly entered room. It is off by default because the
                # targeted calibration added compute without new admits.
                if INAV_MID_PANO_ALL_ROOMS or room_prior_match:
                    rooms_pano_scanned.add(cur_room_full)
                    counters_mid = perception_counters
                    counters_mid["dino_admit_room_rejected"] = (
                        dino_admit_room_rejected
                    )
                    counters_mid["wall_pierce_rejected"] = wall_pierce_rejected
                    n_admit = _do_pano_scan(
                        env, wm, vmap, target_memory, target_guess, candidates,
                        step_for_admit=step, source_tag="dino_mid_pano",
                        dino_threshold=VM_MID_PANO_DINO_THRESHOLD,
                        counters=counters_mid, sel_id=sel_id, style=style)
                    dino_admit_room_rejected = counters_mid["dino_admit_room_rejected"]
                    wall_pierce_rejected = counters_mid["wall_pierce_rejected"]
                    mid_pano_scans += 1
                    mid_pano_admitted += n_admit
                    pose = env.get_pose()  # restore pose after look_at_yaw

        rgb = env.render_rgb()
        depth = env.render_depth() if hasattr(env, "render_depth") else None

        # ---- VLM see-target call ----
        rgb_bytes = pil_to_jpeg_bytes(Image.fromarray(rgb))
        if local_perception_only:
            verdict, vlm_usage, vlm_err, retried = None, None, None, False
        else:
            verdict, vlm_usage, vlm_err, retried = call_see_target(
                model_key, rgb_bytes, target_guess, intent,
                sel_id=sel_id, style=style, step=step,
                robustness_note=robustness_note,
            )
        if vlm_usage:
            usage_calls.append({**vlm_usage, "step": step, "kind": "see_target"})
        if STRICT_VLM_FAILURE and not local_perception_only and verdict is None:
            raise RuntimeError(f"VLM_SEE_TARGET_FAILED {sel_id}/{style}/{model_key} "
                               f"step={step}: {vlm_err}")
        # Trace
        try:
            tag = f"step_{step:02d}"
            (epi_dir / f"{tag}_rgb.jpg").parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(epi_dir / f"{tag}_rgb.jpg",
                                       "JPEG", quality=70, optimize=True)
            (epi_dir / f"{tag}_response.txt").write_text(
                ("[LOCAL PERCEPTION ONLY]" if local_perception_only else
                 json.dumps(verdict, indent=2) if verdict else f"[ERROR] {vlm_err}"),
                encoding="utf-8")
        except Exception:
            pass

        if verdict:
            for o in verdict.get("observed_objects", []):
                seen_objects.add(o)
        # ---- VLM-side memory admit (if see_target with finite depth) ----
        vlm_admitted = None
        if not VM_NO_ADMIT and not VM_NO_VLM_ADMIT \
                and verdict and verdict.get("see_target"):
            # Score by VLM confidence: low=0.4, medium=0.6, high=0.8
            conf_score = {"low": 0.4, "medium": 0.6, "high": 0.8}.get(
                verdict.get("confidence", ""), 0.5)
            xy = _backproject_pixel_to_world(verdict.get("pixel_x_norm"),
                                              verdict.get("pixel_y_norm"),
                                              depth, pose)
            if xy is not None:
                if VM_WALL_PIERCE_REJECT and _is_admit_xy_on_wall(wm, xy):
                    wall_pierce_rejected += 1
                else:
                    vlm_admitted = _admit_cluster(target_memory, xy, conf_score,
                                                   target_guess, step, "vlm",
                                                   admit_yaw=pose["yaw"],
                                                   observer_xy=(pose["position"][0],
                                                                pose["position"][1]))
            else:
                # No valid depth at pixel — fall back to coarse 0.8 m ahead
                xy = (pose["position"][0] + 0.8 * math.cos(pose["yaw"]),
                      pose["position"][1] + 0.8 * math.sin(pose["yaw"]))
                if VM_WALL_PIERCE_REJECT and _is_admit_xy_on_wall(wm, xy):
                    wall_pierce_rejected += 1
                else:
                    vlm_admitted = _admit_cluster(target_memory, xy,
                                                   conf_score * 0.5,  # half-score
                                                   target_guess, step, "vlm_coarse",
                                                   admit_yaw=pose["yaw"],
                                                   observer_xy=(pose["position"][0],
                                                                pose["position"][1]))

        # ---- DINO admit (always runs, supplements VLM) ----
        # Query the target plus collision-safe visual aliases.  This avoids
        # the long full-vocabulary prompt used by the original FPE while still
        # helping rare simulator labels (for example "closestool").  The
        # optional legacy mode additionally includes plan candidates.
        dino_dets = []
        dino_stop_xy = None      # X1: best DINO target XY for detection-driven STOP
        dino_stop_score = 0.0    # X1: matched DINO score for diagnostic
        current_dino_evidence = None
        current_tentative_evidence = None
        try:
            phrases = list(target_queries)
            if not INAV_TARGET_ONLY_DINO:
                for c in candidates:
                    if c and c not in phrases: phrases.append(c)
            phrases = phrases[:5]  # cap at 5 to keep DINO call cheap
            if phrases:
                dets, proposal_metadata, detector_source = (
                    _local_target_proposals(
                        rgb,
                        target_guess,
                        phrases,
                        VM_DINO_THRESHOLD,
                        perception_counters,
                        budgeted_scan_frame=scan_frame_meta is not None,
                    )
                )
                for label, score, bbox in dets:
                    dino_dets.append({"label": label, "score": float(score),
                                      "bbox": bbox,
                                      "detector_source": detector_source})
                # Admit ANY DINO detection of any candidate (not just
                # target_guess). Score is full DINO score capped at 0.9.
                if (not VM_NO_ADMIT) and dets:
                    # Fix F (relaxed): normalize-substring label match with
                    # candidate-rejection.
                    #
                    # Easy-24 OptB cell showed: strict equality (label.lower()
                    # == target_guess.lower()) was TOO strict — DINO often
                    # returns "wall mirror" / "bathroom mirror" / "dining
                    # table" / "kitchen sink" while target_guess is the bare
                    # word. Strict mode rejected ALL admits → 73/96 episodes
                    # hit step_cap. Relaxing to substring (after normalizing
                    # underscores → spaces) re-enables those legitimate
                    # admits. We still reject DINO labels that match an
                    # OTHER plan-candidate exactly — that catches the
                    # "DINO labeled it 'armchair' from the candidates list,
                    # don't admit it as 'sofa' target" failure mode.
                    best = None
                    best_clip_meta = None
                    proposal = None
                    proposal_clip_meta = None
                    if target_guess and VM_DINO_LABEL_STRICT:
                        non_target_cands = {
                            c.lower().strip().replace('_', ' ')
                            for c in candidates
                            if c.lower().strip() != target_guess.lower().strip()
                        }
                        strict_pairs = []
                        for d, semantic_meta in zip(dets, proposal_metadata):
                            lab = str(d[0]).lower().strip().replace('_', ' ')
                            if lab in non_target_cands:
                                continue  # DINO chose a sibling candidate
                            if target_query_match(lab, target_queries):
                                strict_pairs.append((d, semantic_meta))
                        strict_dets = [pair[0] for pair in strict_pairs]
                        if strict_dets and detector_source in {
                            "coco_specialist", "yolo_world_scan"
                        }:
                            independent_clip = (
                                clip_verifier.verify_detections(
                                    rgb, strict_dets, target_guess,
                                    max_rank=INAV_CLIP_ADMIT_MAX_RANK,
                                )
                                if INAV_CLIP_VERIFY else
                                [{"available": False, "accepted": True}
                                 for _ in strict_dets]
                            )
                            verified = [
                                (
                                    detection,
                                    {
                                        **clip_meta,
                                        "specialist_verification": semantic_meta,
                                    },
                                )
                                for (detection, semantic_meta), clip_meta
                                in zip(strict_pairs, independent_clip)
                            ]
                            if verified:
                                best, best_clip_meta = max(
                                    verified, key=lambda pair: pair[0][1]
                                )
                        elif strict_dets and INAV_CLIP_VERIFY:
                            clip_results = clip_verifier.verify_detections(
                                rgb, strict_dets, target_guess,
                                max_rank=INAV_CLIP_ADMIT_MAX_RANK,
                            )
                            dual_results = [None for _ in strict_dets]
                            if INAV_DUAL_DETECTOR_FUSION:
                                dual_detector_calls += 1
                                yolo_world = _load_yolo_world_verifier()
                                dual_results = yolo_world.verify_proposals(
                                    rgb, strict_dets, target_guess,
                                    confidence=(
                                        INAV_DUAL_DETECTOR_CONFIDENCE
                                    ),
                                    min_overlap=INAV_DUAL_DETECTOR_OVERLAP,
                                )
                                errors = [
                                    item for item in dual_results
                                    if item and not item.get("available", False)
                                    and item.get("decision") == "error"
                                ]
                                if errors:
                                    raise RuntimeError(
                                        "required dual-detector verifier "
                                        f"failed: {errors[0].get('error')}"
                                    )
                                dual_detector_consensus_hits += sum(
                                    int(yolo_world.supports_target(item))
                                    for item in dual_results
                                )
                                dual_detector_unknowns += sum(
                                    int(
                                        item is not None
                                        and item.get("decision") == "unknown"
                                    )
                                    for item in dual_results
                                )
                                dual_detector_sibling_vetoes += sum(
                                    int(
                                        item is not None
                                        and item.get("decision") == "sibling"
                                    )
                                    for item in dual_results
                                )
                            verified = [
                                (
                                    detection,
                                    {
                                        **clip_meta,
                                        "dual_detector_verification": dual_meta,
                                    },
                                )
                                for detection, clip_meta, dual_meta in zip(
                                    strict_dets, clip_results, dual_results
                                )
                                if (
                                    clip_verifier.supports_target(
                                        clip_meta,
                                        target_guess,
                                        component_max_rank=(
                                            INAV_CLIP_ADMIT_MAX_RANK
                                        ),
                                    )
                                    or (
                                        INAV_DUAL_DETECTOR_FUSION
                                        and _load_yolo_world_verifier(
                                        ).supports_target(dual_meta)
                                    )
                                )
                            ]
                            if verified:
                                best, best_clip_meta = max(
                                    verified, key=lambda pair: pair[0][1]
                                )
                            elif (
                                INAV_TENTATIVE_MEMORY
                                or INAV_POSE_EVIDENCE_GRAPH
                                or INAV_PASSIVE_MULTI_VIEW
                                or INAV_BELIEF_REPERCEPTION
                            ):
                                # Preserve the strongest geometrically
                                # groundable target-like box for one bounded
                                # information-gathering action.  It remains
                                # outside target_memory and cannot trigger
                                # STOP unless a later active view passes CLIP.
                                uncertain = [
                                    (detection, clip_meta)
                                    for detection, clip_meta in zip(
                                        strict_dets, clip_results
                                    )
                                    if (
                                        INAV_BELIEF_REPERCEPTION
                                        or semantic_uncertainty_candidate(
                                            clip_meta,
                                            max_rank=INAV_TENTATIVE_MAX_RANK,
                                        )
                                    )
                                ]
                                if uncertain:
                                    proposal, proposal_clip_meta = max(
                                        uncertain,
                                        key=lambda pair: pair[0][1],
                                    )
                        elif strict_dets:
                            best = max(strict_dets, key=lambda d: d[1])
                    elif target_guess:
                        best = dino_detector.find_best_match(dets, target_guess)
                        if best is None:
                            best = max(dets, key=lambda d: d[1])
                    selected = best if best is not None else proposal
                    if selected is not None:
                        label, score, bbox = selected
                        xy, robust_depth, mask_meta = _dino_projection(
                            rgb, depth, bbox, pose
                        )
                        if INAV_MASK_REFINEMENT:
                            mask_refinement_calls += 1
                            if mask_meta.get("used", False):
                                mask_refinement_successes += 1
                        box_ok, box_meta = stop_box_quality(bbox, rgb.shape)
                        if xy is not None:
                                if VM_DINO_ADMIT_ROOMS_ONLY \
                                        and not vmap.is_in_likely_room(xy[0], xy[1]):
                                    dino_admit_room_rejected += 1
                                elif VM_WALL_PIERCE_REJECT \
                                        and _is_admit_xy_on_wall(wm, xy):
                                    wall_pierce_rejected += 1
                                else:
                                    if best is not None:
                                        dual_consensus_accepted = (
                                            _dual_detector_consensus_accepted(
                                                best_clip_meta
                                            )
                                        )
                                        admitted_cluster = _admit_cluster(
                                            target_memory, xy,
                                            float(min(score, 0.9)),
                                            label, step,
                                            (
                                                "dino_yolo_consensus"
                                                if (
                                                    detector_source == "dino"
                                                    and dual_consensus_accepted
                                                )
                                                else "dino"
                                                if detector_source == "dino"
                                                else detector_source
                                            ),
                                            admit_yaw=pose["yaw"],
                                            observer_xy=(pose["position"][0],
                                                         pose["position"][1]),
                                            depth_m=robust_depth, bbox=bbox,
                                        )
                                        admitted_cluster["observations"][-1][
                                            "clip_verification"
                                        ] = best_clip_meta
                                        admitted_cluster["observations"][-1][
                                            "mask_refinement"
                                        ] = mask_meta
                                        admitted_cluster["observations"][-1][
                                            "bbox_quality"
                                        ] = box_meta
                                        extent_radius_m = estimated_target_radius_m(
                                            bbox=bbox,
                                            image_width=int(rgb.shape[1]),
                                            depth_m=robust_depth,
                                            horizontal_fov_deg=(
                                                INAV_CAMERA_HFOV_DEG
                                            ),
                                        )
                                        admitted_cluster["observations"][-1][
                                            "target_extent_radius_m"
                                        ] = extent_radius_m
                                        if INAV_BELIEF_REPERCEPTION:
                                            update_target_belief(
                                                admitted_cluster
                                            )
                                            admitted_cluster[
                                                "target_extent_radius_m"
                                            ] = extent_radius_m
                                        current_dino_evidence = {
                                            "cluster": admitted_cluster,
                                            "cluster_uid": admitted_cluster.get(
                                                "cluster_uid"
                                            ),
                                            "xy": [float(xy[0]), float(xy[1])],
                                            "score": float(score),
                                            "depth_m": robust_depth,
                                            "bbox_ok": bool(box_ok),
                                            "bbox_quality": box_meta,
                                            "clip_verification": best_clip_meta,
                                            "mask_refinement": mask_meta,
                                            "label": label,
                                            "detector_source": detector_source,
                                            "semantic_confirmation": (
                                                "dino_yolo_consensus"
                                                if dual_consensus_accepted
                                                else None
                                            ),
                                        }
                                        # Only semantically admitted evidence
                                        # may become a terminal candidate.
                                        if float(score) > dino_stop_score:
                                            dino_stop_xy = (
                                                float(xy[0]), float(xy[1])
                                            )
                                            dino_stop_score = float(score)
                                    else:
                                        proposal_cluster = admit_observation(
                                            tentative_memory,
                                            xy=xy,
                                            score=float(min(score, 0.9)),
                                            label=label,
                                            step=step,
                                            source="dino_tentative",
                                            observer_xy=(
                                                pose["position"][0],
                                                pose["position"][1],
                                            ),
                                            yaw=pose["yaw"],
                                            depth_m=robust_depth,
                                            bbox=bbox,
                                        )
                                        proposal_cluster["observations"][-1][
                                            "semantic_verification"
                                        ] = proposal_clip_meta
                                        proposal_cluster["observations"][-1][
                                            "mask_refinement"
                                        ] = mask_meta
                                        proposal_cluster["observations"][-1][
                                            "bbox_quality"
                                        ] = box_meta
                                        extent_radius_m = estimated_target_radius_m(
                                            bbox=bbox,
                                            image_width=int(rgb.shape[1]),
                                            depth_m=robust_depth,
                                            horizontal_fov_deg=(
                                                INAV_CAMERA_HFOV_DEG
                                            ),
                                        )
                                        proposal_cluster["observations"][-1][
                                            "target_extent_radius_m"
                                        ] = extent_radius_m
                                        proposal_cluster[
                                            "target_extent_radius_m"
                                        ] = extent_radius_m
                                        if INAV_BELIEF_REPERCEPTION:
                                            update_target_belief(
                                                proposal_cluster
                                            )
                                        promote_passively = (
                                            INAV_PASSIVE_MULTI_VIEW
                                            and persistent_proposal_confirmed(
                                                proposal_cluster,
                                                min_viewpoints=(
                                                    INAV_PASSIVE_MIN_VIEWPOINTS
                                                ),
                                                min_semantic_supports=(
                                                    INAV_PASSIVE_MIN_SEMANTIC_SUPPORTS
                                                ),
                                                max_clip_rank=(
                                                    INAV_TENTATIVE_MAX_RANK
                                                ),
                                                max_dispersion_m=(
                                                    INAV_PASSIVE_MAX_DISPERSION_M
                                                ),
                                            )
                                        )
                                        promote_belief = (
                                            INAV_BELIEF_REPERCEPTION
                                            and target_belief_confirmed(
                                                proposal_cluster
                                            )
                                        )
                                        if promote_passively or promote_belief:
                                            if not proposal_cluster.get(
                                                "promoted", False
                                            ):
                                                mark_proposal_promoted(
                                                    proposal_cluster, step=step
                                                )
                                                target_memory.append(
                                                    proposal_cluster
                                                )
                                                if promote_belief:
                                                    belief_promotions += 1
                                            proposal_cluster[
                                                "semantic_confirmation"
                                            ] = (
                                                "target_belief_multiview"
                                                if promote_belief
                                                else "dino_clip_multiview"
                                            )
                                            current_dino_evidence = {
                                                "cluster": proposal_cluster,
                                                "cluster_uid": (
                                                    proposal_cluster.get(
                                                        "cluster_uid"
                                                    )
                                                ),
                                                "xy": [
                                                    float(xy[0]), float(xy[1])
                                                ],
                                                "score": float(score),
                                                "depth_m": robust_depth,
                                                "bbox_ok": bool(box_ok),
                                                "bbox_quality": box_meta,
                                                "clip_verification": (
                                                    proposal_clip_meta
                                                ),
                                                "mask_refinement": mask_meta,
                                                "semantic_confirmation": (
                                                    proposal_cluster[
                                                        "semantic_confirmation"
                                                    ]
                                                ),
                                                "label": label,
                                                "detector_source": (
                                                    detector_source
                                                ),
                                                "belief_posterior": (
                                                    proposal_cluster.get(
                                                        "belief_posterior"
                                                    )
                                                ),
                                            }
                                        else:
                                            current_tentative_evidence = {
                                                "cluster": proposal_cluster,
                                                "cluster_uid": (
                                                    proposal_cluster.get(
                                                        "cluster_uid"
                                                    )
                                                ),
                                                "xy": [
                                                    float(xy[0]), float(xy[1])
                                                ],
                                                "score": float(score),
                                                "depth_m": robust_depth,
                                                "bbox_ok": bool(box_ok),
                                                "bbox_quality": box_meta,
                                                "clip_verification": (
                                                    proposal_clip_meta
                                                ),
                                                "mask_refinement": mask_meta,
                                                "label": label,
                                                "detector_source": (
                                                    detector_source
                                                ),
                                                "belief_posterior": (
                                                    proposal_cluster.get(
                                                        "belief_posterior"
                                                    )
                                                ),
                                            }
        except Exception as e:
            print(f"[dino] {sel_id}/{style} step {step}: {e}", file=sys.stderr)
            if (INAV_DUAL_DETECTOR_FUSION
                    or (
                        (
                            INAV_COCO_STOP_GATE
                            or INAV_CONTEXTUAL_STOP_CONSENSUS
                        )
                        and coco_detector.coco_labels(target_guess)
                    )):
                raise

        if pending_target_bbox_recenter_event is not None:
            recenter_event_index = int(
                pending_target_bbox_recenter_event["event_index"]
            )
            if not 0 <= recenter_event_index < len(
                target_bbox_recenter_ledger
            ):
                raise RuntimeError(
                    "target bbox recenter observation references an unknown "
                    f"event: {recenter_event_index}"
                )
            recenter_event = target_bbox_recenter_ledger[
                recenter_event_index
            ]
            if recenter_event.get("observation_consumed"):
                raise RuntimeError(
                    "target bbox recenter event consumed more than once: "
                    f"{recenter_event_index}"
                )
            admitted_after_recenter = current_dino_evidence is not None
            recenter_event.update({
                "observation_consumed": True,
                "observation_step": int(step),
                "target_evidence_admitted": bool(admitted_after_recenter),
                "admitted_cluster_uid": (
                    current_dino_evidence.get("cluster_uid")
                    if current_dino_evidence is not None else None
                ),
            })
            if admitted_after_recenter:
                target_bbox_recenter_hits += 1
            else:
                target_bbox_recenter_misses += 1
            pending_target_bbox_recenter_event = None

        if scan_frame_meta is not None:
            event_index = int(scan_frame_meta["event_index"])
            if not 0 <= event_index < len(budgeted_scan_events):
                raise RuntimeError(
                    "budgeted scan observation references an unknown event: "
                    f"{event_index}"
                )
            scan_event = budgeted_scan_events[event_index]
            if scan_event.get("observation_consumed"):
                raise RuntimeError(
                    "budgeted scan event consumed more than once: "
                    f"{event_index}"
                )
            scan_event.update({
                "observation_consumed": True,
                "observation_step": int(step),
                "target_evidence_admitted": (
                    current_dino_evidence is not None
                ),
                "tentative_evidence_observed": (
                    current_tentative_evidence is not None
                ),
            })
            if current_dino_evidence is not None:
                scan_event["target_evidence"] = {
                    "label": current_dino_evidence.get("label"),
                    "xy": current_dino_evidence.get("xy"),
                    "score": current_dino_evidence.get("score"),
                    "detector_source": current_dino_evidence.get(
                        "detector_source"
                    ),
                    "semantic_confirmation": current_dino_evidence.get(
                        "semantic_confirmation"
                    ),
                    "cluster_uid": current_dino_evidence.get("cluster_uid"),
                }
            elif current_tentative_evidence is not None:
                scan_event["tentative_evidence"] = {
                    "label": current_tentative_evidence.get("label"),
                    "xy": current_tentative_evidence.get("xy"),
                    "score": current_tentative_evidence.get("score"),
                    "cluster_uid": current_tentative_evidence.get(
                        "cluster_uid"
                    ),
                }
            budgeted_scan_observation_frames += 1
            if current_dino_evidence is not None:
                budgeted_scan_evidence_hits += 1
                current_dino_evidence["budgeted_scan_observation"] = dict(
                    scan_frame_meta
                )
            elif current_tentative_evidence is not None:
                budgeted_scan_tentative_hits += 1
                current_tentative_evidence[
                    "budgeted_scan_observation"
                ] = dict(scan_frame_meta)

        # Resolve one action-counted belief re-observation.  A nearby ordinary
        # admitted detection also counts as a hit because it is stronger than
        # the uncertain proposal; a miss supplies explicit negative evidence.
        if INAV_BELIEF_REPERCEPTION and pending_belief_candidate is not None:
            candidate_xy = pending_belief_candidate.get("xy")
            observed_evidence = current_tentative_evidence
            if observed_evidence is None:
                observed_evidence = current_dino_evidence
            observed_xy = (
                observed_evidence.get("xy")
                if observed_evidence is not None else None
            )
            same_cluster = bool(
                observed_evidence is not None
                and observed_evidence.get("cluster")
                    is pending_belief_candidate
            )
            spatial_match = bool(
                candidate_xy and observed_xy
                and math.hypot(
                    float(candidate_xy[0]) - float(observed_xy[0]),
                    float(candidate_xy[1]) - float(observed_xy[1]),
                ) <= 0.75
            )
            observed = same_cluster or spatial_match
            record_belief_reperception(
                pending_belief_candidate,
                step=step,
                observed=observed,
            )
            if observed:
                belief_reperception_hits += 1
                if current_dino_evidence is not None and not same_cluster:
                    mark_proposal_promoted(
                        pending_belief_candidate, step=step
                    )
            else:
                belief_reperception_misses += 1
            belief_reperception_resolutions.append({
                "proposal_cluster_uid": pending_belief_candidate.get(
                    "cluster_uid"
                ),
                "action_step": int(pending_belief_action_step or -1),
                "observation_step": int(step),
                "observed": bool(observed),
                "same_cluster": bool(same_cluster),
                "spatial_match": bool(spatial_match),
                "posterior_after": pending_belief_candidate.get(
                    "belief_posterior"
                ),
                "semantic_support_viewpoints": (
                    pending_belief_candidate.get(
                        "belief_semantic_support_viewpoints"
                    )
                ),
            })
            pending_belief_candidate = None
            pending_belief_action_step = None

        # R042 resolves exactly one action-counted, pose-anchored lateral view.
        # The second frame must pass the unchanged ordinary admission path;
        # then two forward image rays must agree geometrically.  A failed or
        # absent second detection never enters target memory and never affects
        # STOP.
        if INAV_POSE_EVIDENCE_GRAPH and pending_pose_graph_candidate is not None:
            proposal_observation = latest_pose_observation(
                pending_pose_graph_candidate
            )
            accepted_cluster = (
                current_dino_evidence.get("cluster")
                if current_dino_evidence is not None else None
            )
            accepted_observation = (
                (accepted_cluster.get("observations") or [None])[-1]
                if accepted_cluster is not None else None
            )
            triangulation = (
                triangulate_pose_observations(
                    proposal_observation,
                    accepted_observation,
                    min_baseline_m=INAV_POSE_GRAPH_MIN_BASELINE_M,
                    min_crossing_angle_rad=(
                        INAV_POSE_GRAPH_MIN_CROSSING_RAD
                    ),
                    max_crossing_angle_rad=(
                        INAV_POSE_GRAPH_MAX_CROSSING_RAD
                    ),
                    max_range_m=INAV_POSE_GRAPH_MAX_RANGE_M,
                    max_accepted_projection_residual_m=(
                        INAV_POSE_GRAPH_ACCEPTED_RESIDUAL_M
                    ),
                    image_width=int(rgb.shape[1]),
                    horizontal_fov_rad=math.radians(
                        INAV_CAMERA_HFOV_DEG
                    ),
                )
                if proposal_observation is not None
                and accepted_observation is not None
                else {"valid": False, "failure": "no_ordinary_admission"}
            )
            fused = bool(triangulation.get("valid", False))
            if fused:
                support = pose_graph_support_observation(
                    proposal_observation,
                    triangulation,
                    step=step,
                )
                accepted_cluster.setdefault("observations", []).append(support)
                if len(accepted_cluster["observations"]) > 16:
                    del accepted_cluster["observations"][:-16]
                refresh_cluster(accepted_cluster)
                accepted_cluster["pose_graph_fused_xy"] = list(
                    triangulation["intersection_xy"]
                )
                accepted_cluster["pose_graph_confirmed"] = True
                accepted_cluster["pose_graph_confirmation_step"] = int(step)
                current_dino_evidence["semantic_confirmation"] = (
                    "pose_graph_multiview"
                )
                current_dino_evidence["pose_graph_triangulation"] = dict(
                    triangulation
                )
                mark_proposal_promoted(
                    pending_pose_graph_candidate, step=step
                )
                pose_graph_promotions += 1
            else:
                mark_verification_failure(
                    pending_pose_graph_candidate, step=step,
                    suppress_after=1,
                )
                pose_graph_misses += 1
            pose_graph_resolutions.append({
                "proposal_cluster_uid": pending_pose_graph_candidate.get(
                    "cluster_uid"
                ),
                "action_step": int(pending_pose_graph_action_step or -1),
                "observation_step": int(step),
                "ordinary_target_admission": bool(
                    current_dino_evidence is not None
                ),
                "accepted_cluster_uid": (
                    current_dino_evidence.get("cluster_uid")
                    if current_dino_evidence is not None else None
                ),
                "fused": fused,
                "triangulation": triangulation,
            })
            pending_pose_graph_candidate = None
            pending_pose_graph_action_step = None

        # A tentative proposal gets exactly one bounded active view.  Promotion
        # requires the normal DINO+CLIP admission path from that new view;
        # proposal persistence alone is never terminal semantic evidence.
        if (INAV_EVIDENCE_NAV
                and tentative_verification_candidate is not None):
            proposal_xy = tentative_verification_candidate.get("xy")
            accepted_xy = (
                current_dino_evidence.get("xy")
                if current_dino_evidence is not None else None
            )
            promoted = bool(
                proposal_xy and accepted_xy
                and math.hypot(
                    float(proposal_xy[0]) - float(accepted_xy[0]),
                    float(proposal_xy[1]) - float(accepted_xy[1]),
                ) <= 0.75
            )
            proposal_uid = tentative_verification_candidate.get("cluster_uid")
            accepted_uid = (
                current_dino_evidence.get("cluster_uid")
                if current_dino_evidence is not None else None
            )
            projection_distance_m = (
                math.hypot(
                    float(proposal_xy[0]) - float(accepted_xy[0]),
                    float(proposal_xy[1]) - float(accepted_xy[1]),
                )
                if proposal_xy and accepted_xy else None
            )
            tentative_probe_resolutions.append({
                "proposal_cluster_uid": proposal_uid,
                "action_step": int(tentative_verification_action_step or -1),
                "observation_step": int(step),
                "promoted": promoted,
                "proposal_xy": (
                    [float(proposal_xy[0]), float(proposal_xy[1])]
                    if proposal_xy else None
                ),
                "accepted_cluster_uid": accepted_uid,
                "accepted_xy": (
                    [float(accepted_xy[0]), float(accepted_xy[1])]
                    if accepted_xy else None
                ),
                "projection_distance_m": (
                    round(float(projection_distance_m), 6)
                    if projection_distance_m is not None else None
                ),
            })
            if promoted:
                mark_proposal_promoted(
                    tentative_verification_candidate, step=step
                )
                tentative_probe_promotions += 1
            else:
                mark_verification_failure(
                    tentative_verification_candidate, step=step
                )
                tentative_probe_failures += 1
            tentative_verification_candidate = None
            tentative_verification_action_step = None

        # A support inspection is an information-gathering action, not target
        # evidence.  Attribute its outcome only to the immediately following
        # normal target-perception pass; the support box itself never enters
        # target_memory and cannot trigger STOP.
        if INAV_SUPPORT_INSPECTION and pending_support_candidate is not None:
            target_hit = current_dino_evidence is not None
            support_inspection_resolutions.append({
                "support_candidate_uid": pending_support_candidate.get(
                    "cluster_uid"
                ),
                "support_label": pending_support_candidate.get("label"),
                "support_xy": pending_support_candidate.get("xy"),
                "action_step": int(pending_support_action_step or -1),
                "observation_step": int(step),
                "target_evidence_admitted": bool(target_hit),
                "accepted_target_cluster_uid": (
                    current_dino_evidence.get("cluster_uid")
                    if current_dino_evidence is not None else None
                ),
                "accepted_target_xy": (
                    current_dino_evidence.get("xy")
                    if current_dino_evidence is not None else None
                ),
            })
            if target_hit:
                support_inspection_target_hits += 1
                current_dino_evidence["support_inspection_observation"] = {
                    "support_candidate_uid": pending_support_candidate.get(
                        "cluster_uid"
                    ),
                    "support_label": pending_support_candidate.get("label"),
                    "action_step": int(pending_support_action_step or -1),
                    "observation_step": int(step),
                }
            else:
                support_inspection_failures += 1
            pending_support_candidate = None
            pending_support_action_step = None

        # R031 carrier verification persists for several action-counted views.
        # Only the ordinary target DINO+crop-verifier path above can resolve a
        # session as a hit.  A miss keeps the same carrier active until its
        # bounded action budget is exhausted, after which that physical support
        # instance is blacklisted and global search resumes.
        if INAV_CARRIER_ACTIVE_VERIFY and pending_carrier_action is not None:
            target_hit = current_dino_evidence is not None
            candidate = active_carrier_candidate
            if candidate is None:
                raise RuntimeError(
                    "carrier action pending without an active carrier candidate"
                )
            observation = {
                "action_step": int(pending_carrier_action["action_step"]),
                "observation_step": int(step),
                "observer_xy": [
                    float(pose["position"][0]),
                    float(pose["position"][1]),
                ],
                "arrived_standoff": bool(
                    pending_carrier_action.get("arrived_standoff", False)
                ),
                "view_angle_novelty_rad": pending_carrier_action.get(
                    "view_angle_novelty_rad"
                ),
                "target_evidence_admitted": bool(target_hit),
            }
            candidate.setdefault("session_observations", []).append(observation)
            if observation["arrived_standoff"]:
                candidate["standoff_reached"] = True
                candidate.setdefault("verification_observers", []).append(
                    {
                        "observer_xy": list(observation["observer_xy"]),
                        "action_step": observation["action_step"],
                    }
                )
            outcome = carrier_observation_outcome(
                target_hit=target_hit,
                actions_taken=int(candidate.get("actions_taken", 0)),
                action_budget=INAV_CARRIER_ACTION_BUDGET,
                observation_step=step,
                step_cap=step_cap,
                session_complete=bool(
                    int(candidate.get("rotations_taken", 0))
                        >= INAV_CARRIER_ROTATION_BUDGET
                    or (
                        not candidate.get("standoff_reached", False)
                        and int(candidate.get("approach_actions", 0))
                            >= INAV_CARRIER_APPROACH_BUDGET
                    )
                ),
            )
            if outcome == "target_hit":
                carrier_target_hits += 1
                current_dino_evidence["carrier_verification_observation"] = {
                    "carrier_candidate_uid": candidate.get("cluster_uid"),
                    "support_label": candidate.get("label"),
                    "action_step": observation["action_step"],
                    "observation_step": observation["observation_step"],
                }
                carrier_resolutions.append({
                    "carrier_candidate_uid": candidate.get("cluster_uid"),
                    "support_label": candidate.get("label"),
                    "support_xy": candidate.get("xy"),
                    "started_step": candidate.get("started_step"),
                    "resolved_step": int(step),
                    "actions_taken": int(candidate.get("actions_taken", 0)),
                    "outcome": outcome,
                    "accepted_target_cluster_uid": (
                        current_dino_evidence.get("cluster_uid")
                    ),
                    "session_observations": list(
                        candidate.get("session_observations", [])
                    ),
                })
                active_carrier_candidate = None
            elif outcome == "exhausted":
                carrier_failures += 1
                carrier_rejected_xy.append([
                    float(candidate["xy"][0]), float(candidate["xy"][1])
                ])
                carrier_resolutions.append({
                    "carrier_candidate_uid": candidate.get("cluster_uid"),
                    "support_label": candidate.get("label"),
                    "support_xy": candidate.get("xy"),
                    "started_step": candidate.get("started_step"),
                    "resolved_step": int(step),
                    "actions_taken": int(candidate.get("actions_taken", 0)),
                    "outcome": outcome,
                    "accepted_target_cluster_uid": None,
                    "session_observations": list(
                        candidate.get("session_observations", [])
                    ),
                })
                active_carrier_candidate = None
            pending_carrier_action = None

        # Candidate verification happens only after an explicit approach
        # action faced the camera toward that candidate.  A missed re-detect
        # then counts as negative evidence; two misses temporarily blacklist
        # the cluster instead of letting a stale maximum dominate forever.
        if INAV_EVIDENCE_NAV and verification_candidate is not None:
            bounded_verification = bool(
                verification_candidate_from_bounded_commitment
            )
            same_cluster = (
                current_dino_evidence is not None
                and current_dino_evidence.get("cluster") is verification_candidate
            )
            candidate_xy = verification_candidate.get("xy")
            candidate_dist = (
                math.hypot(pose["position"][0] - candidate_xy[0],
                           pose["position"][1] - candidate_xy[1])
                if candidate_xy else float("inf")
            )
            if same_cluster:
                if INAV_STRUCTURED_VERIFICATION or bounded_verification:
                    mark_verification_success(verification_candidate, step)
                    evidence_verification_successes += 1
                if bounded_verification:
                    bounded_commitment_redetections += 1
            elif candidate_dist <= INAV_VERIFY_DISTANCE_M:
                verification_candidate["last_failed_verification_observer_xy"] = [
                    float(pose["position"][0]),
                    float(pose["position"][1]),
                ]
                was_suppressed = int(
                    verification_candidate.get("suppressed_until_step", -1)
                ) > step
                mark_verification_failure(
                    verification_candidate,
                    step,
                    min_distinct_attempts_for_suppression=(
                        2 if INAV_STRUCTURED_VERIFICATION else None
                    ),
                )
                evidence_verification_failures += 1
                if bounded_verification:
                    bounded_commitment_misses += 1
                is_suppressed = int(
                    verification_candidate.get("suppressed_until_step", -1)
                ) > step
                if is_suppressed and not was_suppressed:
                    evidence_clusters_suppressed += 1
            verification_candidate = None
            verification_candidate_from_bounded_commitment = False

        evidence_stop = False
        evidence_stop_meta = None
        if INAV_EVIDENCE_NAV and current_dino_evidence is not None:
            cluster = current_dino_evidence["cluster"]
            cluster_xy = cluster.get("xy")
            cluster_distance = (
                math.hypot(pose["position"][0] - cluster_xy[0],
                           pose["position"][1] - cluster_xy[1])
                if cluster_xy else float("inf")
            )
            mask_meta = current_dino_evidence.get("mask_refinement") or {}
            surface_xy = mask_meta.get("surface_xy")
            surface_distance = (
                math.hypot(pose["position"][0] - float(surface_xy[0]),
                           pose["position"][1] - float(surface_xy[1]))
                if mask_meta.get("used", False) and surface_xy
                else None
            )
            terminal_distance = (
                surface_distance
                if surface_distance is not None else cluster_distance
            )
            depth_m = current_dino_evidence.get("depth_m")
            clip_meta = current_dino_evidence.get("clip_verification") or {}
            multi_view_ok = cluster_confirmed(cluster)
            specialist_stop_ok, specialist_stop_meta = _specialist_stop_gate(
                target_guess,
                str(current_dino_evidence.get("detector_source") or ""),
                clip_meta,
            )
            strong_semantic_current = (
                bool(clip_meta.get("available", False))
                and int(clip_meta.get("target_rank", 10_000)) == 1
                and float(clip_meta.get("margin_to_best_other", float("-inf")))
                    >= INAV_CLIP_STRONG_MARGIN
            )
            belief_stop_ok = bool(
                INAV_BELIEF_REPERCEPTION
                and current_dino_evidence.get("semantic_confirmation")
                    == "target_belief_multiview"
                and target_belief_stop_supported(cluster, clip_meta)
            )
            semantic_ok = (
                (
                    specialist_stop_meta["required"]
                    and specialist_stop_meta["current_positive"]
                )
                or not INAV_CLIP_VERIFY
                or clip_verifier.supports_target(
                    clip_meta,
                    target_guess,
                    component_max_rank=INAV_CLIP_STOP_MAX_RANK,
                )
                or belief_stop_ok
                or (
                    INAV_DUAL_DETECTOR_FUSION
                    and INAV_DUAL_DETECTOR_STOP_SUPPORT
                    and current_dino_evidence.get(
                        "semantic_confirmation"
                    ) == "dino_yolo_consensus"
                    and _load_yolo_world_verifier().supports_target(
                        clip_meta.get("dual_detector_verification")
                    )
                )
                or (
                    current_dino_evidence.get("semantic_confirmation")
                        == "dino_clip_multiview"
                    and semantic_uncertainty_candidate(
                        clip_meta,
                        max_rank=INAV_TENTATIVE_MAX_RANK,
                    )
                )
            )
            stop_confirmation = multi_view_ok
            if INAV_PERSISTENT_STOP_CONFIRMATION:
                stop_confirmation = stop_confirmation_ok(
                    cluster,
                    strong_semantic_current=strong_semantic_current,
                    weak_semantic_min_viewpoints=(
                        INAV_STOP_WEAK_SEMANTIC_MIN_VIEWPOINTS
                    ),
                    max_weak_semantic_failures=(
                        INAV_STOP_MAX_WEAK_SEMANTIC_FAILURES
                    ),
                )
            if INAV_ADAPTIVE_STOP:
                gates, adaptive_meta = adaptive_stop_gates(
                    detector_score=float(current_dino_evidence.get("score", 0.0)),
                    depth_m=depth_m,
                    bbox_ok=bool(current_dino_evidence.get("bbox_ok")),
                    bbox_quality=current_dino_evidence.get("bbox_quality"),
                    cluster_distance_m=terminal_distance,
                    confirmation_ok=stop_confirmation,
                    semantic_ok=semantic_ok,
                    strong_semantic_ok=strong_semantic_current,
                    require_strong_semantics_for_relaxation=(
                        INAV_STRONG_SEMANTIC_RELAXATION
                    ),
                    score_threshold=INAV_STOP_DINO_SCORE,
                    verified_score_floor=INAV_VERIFIED_STOP_DINO_SCORE,
                    cluster_distance_threshold_m=INAV_STOP_CLUSTER_DISTANCE_M,
                    verified_cluster_distance_m=(
                        INAV_VERIFIED_STOP_CLUSTER_DISTANCE_M
                    ),
                    max_depth_m=INAV_STOP_DEPTH_M,
                    large_bbox_max_depth_m=INAV_LARGE_BBOX_MAX_DEPTH_M,
                    large_bbox_max_area_fraction=(
                        INAV_LARGE_BBOX_MAX_AREA_FRACTION
                    ),
                )
            else:
                gates = {
                    "current_frame": True,
                    "score_ok": float(current_dino_evidence.get("score", 0.0))
                                >= INAV_STOP_DINO_SCORE,
                    "depth_ok": depth_m is not None and depth_m <= INAV_STOP_DEPTH_M,
                    "bbox_ok": bool(current_dino_evidence.get("bbox_ok")),
                    "cluster_distance_ok": terminal_distance
                                           <= INAV_STOP_CLUSTER_DISTANCE_M,
                    "confirmation_ok": stop_confirmation,
                    "clip_semantic_ok": semantic_ok,
                }
                adaptive_meta = {
                    "verified_relaxation_eligible": False,
                    "relaxed_score_used": False,
                    "relaxed_distance_used": False,
                    "relaxed_large_bbox_used": False,
                }
            gates["specialist_stop_ok"] = specialist_stop_ok
            if specialist_stop_meta["required"]:
                perception_counters["coco_stop_gate_checks"] = (
                    perception_counters.get("coco_stop_gate_checks", 0) + 1
                )
                if specialist_stop_meta["current_positive"]:
                    perception_counters["coco_stop_gate_positives"] = (
                        perception_counters.get(
                            "coco_stop_gate_positives", 0
                        ) + 1
                    )
                counterfactual_stop = all(
                    value for key, value in gates.items()
                    if key != "specialist_stop_ok"
                )
                if not specialist_stop_ok and counterfactual_stop:
                    perception_counters["coco_stop_gate_vetoes"] = (
                        perception_counters.get("coco_stop_gate_vetoes", 0) + 1
                    )
            current_room = (
                wm.room_at(pose["position"][0], pose["position"][1])
                if hasattr(wm, "room_at") else None
            )
            contextual_stop_ok, contextual_stop_meta = (
                contextual_stop_consensus(
                    enabled=INAV_CONTEXTUAL_STOP_CONSENSUS,
                    supported_target=bool(
                        coco_detector.coco_labels(target_guess)
                    ),
                    clip_verification=clip_meta,
                    current_room=current_room,
                    likely_rooms=plan.get("likely_rooms") or [],
                    distinct_viewpoints=int(
                        cluster.get("distinct_viewpoints", 0)
                    ),
                    bbox_area_fraction=float(
                        (current_dino_evidence.get("bbox_quality") or {}).get(
                            "area_fraction", 0.0
                        ) or 0.0
                    ),
                    strong_margin=INAV_CONTEXTUAL_STOP_STRONG_MARGIN,
                    persistent_min_viewpoints=(
                        INAV_CONTEXTUAL_STOP_MIN_VIEWPOINTS
                    ),
                    persistent_min_area_fraction=(
                        INAV_CONTEXTUAL_STOP_MIN_AREA_FRACTION
                    ),
                    min_votes=INAV_CONTEXTUAL_STOP_MIN_VOTES,
                )
            )
            gates["contextual_consensus_ok"] = contextual_stop_ok
            if contextual_stop_meta["required"]:
                perception_counters["contextual_stop_checks"] = (
                    perception_counters.get("contextual_stop_checks", 0) + 1
                )
                if contextual_stop_ok:
                    perception_counters["contextual_stop_positives"] = (
                        perception_counters.get(
                            "contextual_stop_positives", 0
                        ) + 1
                    )
                counterfactual_stop = all(
                    value for key, value in gates.items()
                    if key != "contextual_consensus_ok"
                )
                if not contextual_stop_ok and counterfactual_stop:
                    perception_counters["contextual_stop_vetoes"] = (
                        perception_counters.get(
                            "contextual_stop_vetoes", 0
                        ) + 1
                    )
            persistent_encoder_ok = persistent_single_encoder_stop_ok(
                clip_meta.get("support_sources") or [],
                int(cluster.get("n_observations", 0)),
                contradiction_observations=(
                    INAV_PERSISTENT_SINGLE_ENCODER_VETO_OBSERVATIONS
                ),
            ) if INAV_PERSISTENT_SINGLE_ENCODER_VETO else True
            gates["persistent_cross_encoder_ok"] = persistent_encoder_ok
            if (
                INAV_PERSISTENT_SINGLE_ENCODER_VETO
                and not persistent_encoder_ok
                and all(
                    value for key, value in gates.items()
                    if key != "persistent_cross_encoder_ok"
                )
            ):
                persistent_single_encoder_stop_vetoes += 1
                persistent_single_encoder_stop_veto_ledger.append({
                    "step": int(step),
                    "cluster_uid": current_dino_evidence.get("cluster_uid"),
                    "n_observations": int(
                        cluster.get("n_observations", 0)
                    ),
                    "support_sources": sorted(
                        str(source)
                        for source in (clip_meta.get("support_sources") or [])
                        if str(source)
                    ),
                    "counterfactual_stop_without_veto": True,
                    "uses_evaluator_target": False,
                })
            terminal_quorum_ok, terminal_quorum_meta = (
                terminal_evidence_quorum(
                    clip_meta.get("support_sources") or [],
                    int(cluster.get("n_observations", 0)),
                    clip_meta,
                    int(contextual_stop_meta.get("vote_count", 0)),
                    persistent_observations=(
                        INAV_TERMINAL_QUORUM_PERSISTENT_OBSERVATIONS
                    ),
                    weak_rank_max=INAV_TERMINAL_QUORUM_WEAK_RANK_MAX,
                    contextual_min_votes=(
                        INAV_TERMINAL_QUORUM_CONTEXTUAL_MIN_VOTES
                    ),
                )
                if INAV_TERMINAL_EVIDENCE_QUORUM
                else (True, {"accepted": True, "enabled": False})
            )
            gates["terminal_evidence_quorum_ok"] = terminal_quorum_ok
            if (
                INAV_TERMINAL_EVIDENCE_QUORUM
                and not terminal_quorum_ok
                and all(
                    value for key, value in gates.items()
                    if key != "terminal_evidence_quorum_ok"
                )
            ):
                terminal_evidence_quorum_vetoes += 1
                terminal_evidence_quorum_ledger.append({
                    "step": int(step),
                    "cluster_uid": current_dino_evidence.get("cluster_uid"),
                    "counterfactual_stop_without_quorum": True,
                    **terminal_quorum_meta,
                })
            # A yaw-only re-centering action does not create spatial parallax.
            # Its follow-up frame may therefore terminate only when both
            # independent image encoders support the target. This prevents a
            # same-position angular view from turning one model's persistent
            # false positive into multi-view confirmation.
            recenter_support_sources = set(
                str(source)
                for source in (clip_meta.get("support_sources") or [])
                if str(source)
            )
            recenter_followup_consensus_ok = bool(
                target_bbox_recenter_frame_meta is None
                or len(recenter_support_sources) >= 2
            )
            gates["recenter_followup_consensus_ok"] = (
                recenter_followup_consensus_ok
            )
            if (
                target_bbox_recenter_frame_meta is not None
                and not recenter_followup_consensus_ok
                and all(
                    value for key, value in gates.items()
                    if key != "recenter_followup_consensus_ok"
                )
            ):
                target_bbox_recenter_stop_vetoes += 1
            evidence_stop = all(gates.values())
            if evidence_stop:
                adaptive_stop_relaxations["score"] += int(
                    adaptive_meta["relaxed_score_used"]
                )
                adaptive_stop_relaxations["distance"] += int(
                    adaptive_meta["relaxed_distance_used"]
                )
                adaptive_stop_relaxations["large_bbox"] += int(
                    adaptive_meta["relaxed_large_bbox_used"]
                )
            evidence_stop_meta = {
                **gates,
                **adaptive_meta,
                "multi_view_confirmed": multi_view_ok,
                "stop_confirmation_ok": stop_confirmation,
                "strong_semantic_current": strong_semantic_current,
                "belief_stop_ok": belief_stop_ok,
                "belief_posterior": cluster.get("belief_posterior"),
                "depth_m": round(float(depth_m), 3) if depth_m is not None else None,
                "cluster_distance_m": round(float(cluster_distance), 3),
                "surface_distance_m": (
                    round(float(surface_distance), 3)
                    if surface_distance is not None else None
                ),
                "terminal_distance_source": (
                    "mask_surface" if surface_distance is not None
                    else "object_cluster"
                ),
                "cluster_score": round(float(cluster.get("score", 0.0)), 4),
                "n_observations": int(cluster.get("n_observations", 0)),
                "distinct_viewpoints": int(cluster.get("distinct_viewpoints", 0)),
                "bbox_quality": current_dino_evidence.get("bbox_quality"),
                "clip_verification": clip_meta,
                "semantic_confirmation": current_dino_evidence.get(
                    "semantic_confirmation"
                ),
                "specialist_stop_gate": specialist_stop_meta,
                "contextual_stop_consensus": contextual_stop_meta,
                "terminal_evidence_quorum": terminal_quorum_meta,
            }

        # ---- X1: detection-driven STOP (VLFM/WMNav-style, primary) ----
        # Two paths:
        #  (a) THIS-frame DINO match: best detection score this step ≥
        #      VM_DINO_STOP_SCORE_MIN AND back-projected XY within
        #      VM_DINO_STOP_RADIUS_M of agent.
        #  (b) HISTORICAL cluster: any target_memory cluster with
        #      score ≥ VM_DINO_STOP_SCORE_MIN whose XY is within
        #      VM_DINO_STOP_RADIUS_M of agent. Path (b) covers the
        #      common case where DINO matched target via a synonym
        #      label on an earlier frame ("kitchen sink" ≈ "basin")
        #      and admitted the cluster, but THIS frame's DINO
        #      returned a different label not satisfying the strict
        #      substring check. The cluster's match already happened
        #      via the admit gate's logic; we just use its XY here.
        # Either path bypasses VLM see_target (hallucinated by small
        # VLMs). Geometric anchor in both cases.
        dino_stop = False
        dino_stop_anchor_xy = None
        if VM_DINO_STOP_ENABLED:
            cand_xy = None
            cand_score = 0.0
            # Path (a): this-frame admit
            if dino_stop_xy is not None and dino_stop_score >= VM_DINO_STOP_SCORE_MIN:
                d = math.hypot(pose["position"][0] - dino_stop_xy[0],
                                pose["position"][1] - dino_stop_xy[1])
                if d <= VM_DINO_STOP_RADIUS_M:
                    cand_xy = dino_stop_xy
                    cand_score = dino_stop_score
            # Path (b): historical cluster within reach
            for c in target_memory:
                if float(c.get("score", 0)) < VM_DINO_STOP_SCORE_MIN:
                    continue
                cxy = c.get("xy")
                if not cxy:
                    continue
                d = math.hypot(pose["position"][0] - cxy[0],
                                pose["position"][1] - cxy[1])
                if d <= VM_DINO_STOP_RADIUS_M and float(c["score"]) > cand_score:
                    cand_xy = (float(cxy[0]), float(cxy[1]))
                    cand_score = float(c["score"])
            if cand_xy is not None:
                dino_stop = True
                dino_stop_anchor_xy = cand_xy

        # ---- v2 STOP gate: WMNav-style geometric primary + soft fallback ----
        # strong_stop: pose within VM_STRONG_STOP_RADIUS_M (default 0.8m) of
        #   any cluster with score >= VM_STRONG_STOP_SCORE_MIN (default 0.5).
        #   Pure geometry, NO VLM confirmation required. Solves the rare-
        #   object case (menorah) where VLM never says see_target=True even
        #   when agent is right on top of cluster.
        # standard_stop: engine_close (1.5m + 0.5 score) AND VLM see_target
        #   medium+ for VM_STOP_CONSECUTIVE steps. Original strict gate.
        engine_stop, stop_meta = vmap.should_stop(
            (pose["position"][0], pose["position"][1]), target_memory)
        vlm_sees_now = bool(verdict and verdict.get("see_target")
                            and verdict.get("confidence") in ("medium", "high"))
        vlm_see_streak = (vlm_see_streak + 1) if vlm_sees_now else 0
        vlm_sees = (vlm_see_streak >= VM_STOP_CONSECUTIVE)
        # D2: track recent VLM see_target (any confidence) for STOP confirmation
        recent_vlm_sees.append((step, bool(verdict and verdict.get("see_target")),
                                  str(verdict.get("confidence", "")) if verdict else ""))
        if len(recent_vlm_sees) > 10:
            recent_vlm_sees = recent_vlm_sees[-10:]

        # Strong-stop gate (v2 + v4.2 fixes):
        #   - freshness: cluster must be admitted within last N steps
        #     (filters stale step-0 pano-init clusters)
        #   - score min: ≥ VM_STRONG_STOP_SCORE_MIN (raised to 0.65 to
        #     drop most single-shot DINO false positives in the 0.45-0.55
        #     band)
        #   - sources: cluster must have ≥ VM_STRONG_STOP_MIN_SOURCES
        #     admit events (multi-step or multi-source reinforcement)
        fresh_memory = [
            c for c in target_memory
            if (step - int(c.get("step", 0))) <= VM_STRONG_STOP_FRESHNESS_STEPS
            and float(c.get("score", 0)) >= VM_STRONG_STOP_SCORE_MIN
            and len(c.get("sources") or []) >= VM_STRONG_STOP_MIN_SOURCES
        ]
        fresh_engine_stop, fresh_meta = vmap.should_stop(
            (pose["position"][0], pose["position"][1]), fresh_memory)
        fresh_closest_dist = fresh_meta.get("closest_dist", 1e9) or 1e9
        strong_stop_geom = fresh_closest_dist < VM_STRONG_STOP_RADIUS_M
        strong_stop = strong_stop_geom

        # Fix D2: require confirmation BEFORE strong_stop fires.
        # Any of: (a) cluster admit_yaws spread ≥ threshold, OR
        #         (b) VLM see_target=True (medium+) within last N steps, OR
        #         (c) cluster has ≥ NOBS_BYPASS observations
        if strong_stop_geom and VM_STRONG_STOP_REQUIRE_CONFIRM:
            triggered_cluster = fresh_meta.get("closest_cluster")
            yaw_spread = _yaw_spread(triggered_cluster.get("admit_yaws", [])) \
                          if triggered_cluster else 0.0
            n_obs = (triggered_cluster.get("n_observations", 0)
                     if triggered_cluster else 0)
            recent_vlm_ok = any(
                s == True and c in ("medium", "high")
                for (st, s, c) in recent_vlm_sees[-VM_STRONG_STOP_VLM_FRESHNESS_STEPS:]
            )
            confirmed = (yaw_spread >= VM_STRONG_STOP_YAW_SPREAD_RAD
                         or recent_vlm_ok
                         or n_obs >= VM_STRONG_STOP_NOBS_BYPASS)
            if not confirmed:
                strong_stop = False
                strong_stop_demoted += 1
        standard_stop = engine_stop and vlm_sees

        # Q2.1: DINO confirm gate. Standard_stop is gated by VLM see_target,
        # which Gemini-class small VLMs hallucinate (claim "I see X" when
        # X is at yaw -128° behind agent). Require DINO to also detect
        # target on this frame's RGB before honoring the stop signal.
        # If only VLM saw target → demote (skip stop, keep navigating).
        if standard_stop and VM_STANDARD_STOP_REQUIRE_DINO:
            dino_saw_target = False
            if dino_dets and target_guess:
                non_target_cands = {
                    c.lower().strip().replace('_', ' ')
                    for c in candidates
                    if c.lower().strip() != target_guess.lower().strip()
                } if VM_DINO_LABEL_STRICT else set()
                for d in dino_dets:
                    lab = str(d.get("label", "")).lower().strip().replace('_', ' ')
                    if lab in non_target_cands:
                        continue
                    if target_query_match(lab, target_queries):
                        dino_saw_target = True; break
            if not dino_saw_target:
                standard_stop = False
                standard_stop_demoted += 1

        terminal_stop = (
            evidence_stop if INAV_EVIDENCE_NAV
            else (dino_stop or strong_stop or standard_stop)
        )
        if terminal_stop:
            if INAV_EVIDENCE_NAV:
                stop_path = "evidence_verified_stop"
                evidence_stop_fired += 1
            elif dino_stop:
                stop_path = "dino_stop"
                dino_stop_fired += 1
            elif strong_stop:
                stop_path = "strong_stop"
            else:
                stop_path = "standard_stop"
            if scan_frame_meta is not None:
                budgeted_scan_terminal_stops += 1
                terminal_scan_event = budgeted_scan_events[
                    int(scan_frame_meta["event_index"])
                ]
                terminal_scan_event.update({
                    "policy_eligible_evidence": bool(
                        current_dino_evidence is not None
                    ),
                    "policy_candidate_kind": (
                        "target_terminal"
                        if current_dino_evidence is not None else None
                    ),
                    "policy_cluster_uid": (
                        current_dino_evidence.get("cluster_uid")
                        if current_dino_evidence is not None else None
                    ),
                })
                if current_dino_evidence is not None:
                    budgeted_scan_policy_evidence_hits += 1
            stop_reason = stop_path
            # STOP is terminal: preserve the current observation and camera
            # pose that justified the decision.  Do not translate or rotate
            # after STOP, because either would be an uncounted action and make
            # GSR incomparable with standard ObjectNav semantics.
            final_rgb = rgb
            terminal_frame_source = "current_stop_observation"
            pose = env.get_pose()
            final_target = target_guess
            traj.append({"step": step, "action": "STOP",
                         "target": final_target,
                         "see_target_verdict": verdict,
                         "engine_stop_meta": stop_meta,
                         "evidence_stop_meta": evidence_stop_meta,
                         "stop_path": stop_path,
                         "terminal_frame_source": terminal_frame_source,
                         "observation_from_budgeted_scan": scan_frame_meta,
                         "policy_candidate_kind": "target_terminal",
                         "selected_navigation_cluster_uid": (
                             current_dino_evidence.get("cluster_uid")
                             if current_dino_evidence is not None else None
                         ),
                         "current_dino_evidence": (
                             {
                                 key: value
                                 for key, value in current_dino_evidence.items()
                                 if key != "cluster"
                             }
                             if current_dino_evidence is not None else None
                         ),
                         "current_tentative_evidence": (
                             {
                                 key: value
                                 for key, value
                                 in current_tentative_evidence.items()
                                 if key != "cluster"
                             }
                             if current_tentative_evidence is not None else None
                         ),
                         "dino": dino_dets,
                         "position": pose["position"], "yaw": pose["yaw"]})
            break

        # R052: a target-conditioned edge box that has not passed the
        # unchanged terminal gate may request only a camera rotation. The box
        # may already be admitted (but lack multi-view/terminal confirmation)
        # or may be rejected while a recent nearby target memory persists.
        # The next loop iteration reruns ordinary DINO+CLIP and STOP.
        if (
            INAV_TARGET_BBOX_RECENTER
            and target_bbox_recenter_actions
                < INAV_TARGET_BBOX_RECENTER_ACTION_BUDGET
            and step < step_cap
            and target_memory
        ):
            recenter_current_xy = (
                float(pose["position"][0]),
                float(pose["position"][1]),
            )
            nearby_recent_memory = [
                cluster for cluster in target_memory
                if (
                    step - int(cluster.get("step", -10_000))
                        <= INAV_EVIDENCE_MAX_AGE
                    and math.hypot(
                        recenter_current_xy[0] - float(cluster["xy"][0]),
                        recenter_current_xy[1] - float(cluster["xy"][1]),
                    ) <= INAV_TARGET_BBOX_RECENTER_MEMORY_RADIUS_M
                )
            ]
            recenter_yaw, recenter_meta = target_bbox_recenter_action(
                dino_dets,
                target_queries,
                current_yaw_rad=float(pose["yaw"]),
                image_width=int(rgb.shape[1]),
                horizontal_fov_rad=math.radians(INAV_CAMERA_HFOV_DEG),
                score_min=INAV_TARGET_BBOX_RECENTER_SCORE_MIN,
                edge_fraction_min=(
                    INAV_TARGET_BBOX_RECENTER_EDGE_FRACTION_MIN
                ),
                max_rotation_rad=math.radians(
                    INAV_TARGET_BBOX_RECENTER_MAX_ROTATION_DEG
                ),
            )
            if nearby_recent_memory and recenter_yaw is not None:
                memory_cluster = min(
                    nearby_recent_memory,
                    key=lambda cluster: math.hypot(
                        recenter_current_xy[0] - float(cluster["xy"][0]),
                        recenter_current_xy[1] - float(cluster["xy"][1]),
                    ),
                )
                memory_distance_m = math.hypot(
                    recenter_current_xy[0] - float(memory_cluster["xy"][0]),
                    recenter_current_xy[1] - float(memory_cluster["xy"][1]),
                )
                env.look_at_yaw(float(recenter_yaw))
                pose = env.get_pose()
                target_bbox_recenter_actions += 1
                recenter_event = {
                    "event_index": int(len(target_bbox_recenter_ledger)),
                    "action_step": int(step),
                    "expected_observation_step": int(step + 1),
                    "observation_step": None,
                    "observation_consumed": False,
                    "target_evidence_admitted": False,
                    "admitted_cluster_uid": None,
                    "memory_cluster_uid": memory_cluster.get("cluster_uid"),
                    "memory_distance_m": round(float(memory_distance_m), 4),
                    "trigger_evidence_admitted": bool(
                        current_dino_evidence is not None
                    ),
                    "trigger_cluster_uid": (
                        current_dino_evidence.get("cluster_uid")
                        if current_dino_evidence is not None else None
                    ),
                    **recenter_meta,
                    "yaw_after_rad": float(pose["yaw"]),
                }
                target_bbox_recenter_ledger.append(recenter_event)
                pending_target_bbox_recenter_event = {
                    "event_index": int(recenter_event["event_index"]),
                }
                traj.append({
                    "step": step,
                    "action": "ROTATE_TARGET_RECENTER",
                    "position": pose["position"],
                    "yaw": pose["yaw"],
                    "room": wm.room_at(
                        pose["position"][0], pose["position"][1]
                    ),
                    "see_target_verdict": verdict,
                    "engine_stop_meta": stop_meta,
                    "evidence_stop_meta": evidence_stop_meta,
                    "policy_candidate_kind": "target_bbox_recenter",
                    "selected_navigation_cluster_uid": (
                        memory_cluster.get("cluster_uid")
                    ),
                    "current_dino_evidence": (
                        {
                            key: value
                            for key, value in current_dino_evidence.items()
                            if key != "cluster"
                        }
                        if current_dino_evidence is not None else None
                    ),
                    "dino": dino_dets,
                    "target_bbox_recenter_meta": dict(recenter_event),
                })
                continue

        # ---- Engine picks next waypoint ----
        # Build direction_hint for VLFM-style cone source if VLM gave us
        # a non-trivial relevance/explore_direction this step.
        direction_hint = None
        if verdict and verdict.get("relevance_score", 0) >= 5 \
                and verdict.get("explore_direction", "no_clue") != "no_clue":
            direction_hint = {
                "agent_xy": (pose["position"][0], pose["position"][1]),
                "yaw": pose["yaw"],
                "relevance_score": verdict["relevance_score"],
                "explore_direction": verdict["explore_direction"],
                # X3.1 depth-mask + X3.2 confidence fusion inputs
                "depth_image": depth,
                "hfov_rad": math.pi / 2,   # IsaacSim camera default
            }
        current_xy = (pose["position"][0], pose["position"][1])
        target_visual_context = None
        if INAV_TARGET_VISUAL_MEMORY_FRONTIER:
            # R040 uses the fixed UIAP-OGN prompt ensemble symmetrically over
            # the full target/background competition set.  No scene relation,
            # target room, or support prior enters this observation.
            target_visual_context = (
                clip_verifier.score_scene_context_prompt_ensemble(
                    rgb, target_guess
                )
            )
            target_visual_memory_observations += int(
                target_visual_context.get("available", False)
            )
            visual_hint = target_visual_probabilistic_hint(
                enabled=True,
                ensemble_result=target_visual_context,
                agent_xy=current_xy,
                yaw=float(pose["yaw"]),
                depth_image=depth,
                hfov_rad=math.pi / 2,
            )
            informative = visual_hint is not None
            memory_ready_before = target_visual_memory_ready
            visual_memory_update = (
                vmap.observe_probabilistic_visual_hint(visual_hint)
                if visual_hint is not None else {
                    "updated": False,
                    "updated_cells": 0,
                }
            )
            if informative:
                target_visual_memory_ready = True
                target_visual_memory_informative += 1
            target_visual_memory_ledger.append({
                "step": int(step),
                "available": bool(
                    target_visual_context.get("available", False)
                ),
                "relevance": target_visual_context.get("relevance"),
                "relevance_mean": target_visual_context.get(
                    "relevance_mean"
                ),
                "relevance_variance": target_visual_context.get(
                    "relevance_variance"
                ),
                "ensemble_size": target_visual_context.get("ensemble_size"),
                "informative": bool(informative),
                "memory_ready_before": bool(memory_ready_before),
                "memory_ready_after": bool(target_visual_memory_ready),
                "visual_memory_update": visual_memory_update,
                "frontier_called": False,
                "frontier_selected": False,
            })
        support_candidate = (
            active_carrier_candidate
            if INAV_CARRIER_ACTIVE_VERIFY and not target_memory else None
        )
        if (
            support_candidate is not None
            and INAV_CARRIER_ACTIVE_VERIFY
            and step >= step_cap
        ):
            carrier_failures += 1
            carrier_rejected_xy.append([
                float(support_candidate["xy"][0]),
                float(support_candidate["xy"][1]),
            ])
            carrier_resolutions.append({
                "carrier_candidate_uid": support_candidate.get("cluster_uid"),
                "support_label": support_candidate.get("label"),
                "support_xy": support_candidate.get("xy"),
                "started_step": support_candidate.get("started_step"),
                "resolved_step": int(step),
                "actions_taken": int(
                    support_candidate.get("actions_taken", 0)
                ),
                "outcome": "budget_guard",
                "accepted_target_cluster_uid": None,
                "session_observations": list(
                    support_candidate.get("session_observations", [])
                ),
            })
            active_carrier_candidate = None
            support_candidate = None
        support_labels = support_labels_for_object(target_guess)
        support_current_room = (
            wm.room_at(current_xy[0], current_xy[1])
            if hasattr(wm, "room_at") else None
        )
        support_search_enabled = bool(
            INAV_SUPPORT_INSPECTION
            or (
                INAV_CARRIER_ACTIVE_VERIFY
                and active_carrier_candidate is None
                and carrier_sessions_started < INAV_CARRIER_SESSION_BUDGET
            )
        )
        if (
            support_search_enabled
            and INAV_EVIDENCE_NAV
            and not target_memory
            and (
                not INAV_SUPPORT_INSPECTION
                or support_inspections < INAV_SUPPORT_INSPECTION_BUDGET
            )
            and step < step_cap
            and support_labels
            and projected_room_matches_prior(
                wm, current_xy, plan.get("likely_rooms") or []
            )
        ):
            if INAV_CARRIER_ACTIVE_VERIFY:
                carrier_detector_calls += 1
            else:
                support_inspection_detector_calls += 1
            try:
                yolo_world = _load_yolo_world_verifier()
                support_detections = yolo_world.detect(
                    rgb,
                    confidence=INAV_SUPPORT_INSPECTION_CONFIDENCE,
                )
                for support_detection in support_detection_candidates(
                    support_detections, support_labels
                ):
                    support_xy, support_depth, support_projection_meta = (
                        _dino_projection(
                            rgb,
                            depth,
                            support_detection["bbox"],
                            pose,
                        )
                    )
                    if support_xy is None or not projected_room_matches_prior(
                        wm,
                        support_xy,
                        plan.get("likely_rooms") or [],
                    ):
                        continue
                    if (
                        INAV_CARRIER_ACTIVE_VERIFY
                        and not support_candidate_is_new(
                            support_xy,
                            carrier_rejected_xy,
                            min_separation_m=(
                                INAV_CARRIER_BLACKLIST_RADIUS_M
                            ),
                        )
                    ):
                        continue
                    support_candidate = {
                        "cluster_uid": (
                            f"{'carrier' if INAV_CARRIER_ACTIVE_VERIFY else 'support'}:"
                            f"{sel_id}:{step}:"
                            f"{support_detection['label']}"
                        ),
                        "xy": [float(support_xy[0]), float(support_xy[1])],
                        "label": support_detection["label"],
                        "score": float(support_detection["score"]),
                        "bbox": [
                            float(value)
                            for value in support_detection["bbox"]
                        ],
                        "support_priority": int(
                            support_detection["support_priority"]
                        ),
                        "depth_m": support_depth,
                        "projection": support_projection_meta,
                        "target_category": target_guess,
                        "room_context_match": True,
                        "current_room": support_current_room,
                        "projected_room": wm.room_at(
                            support_xy[0], support_xy[1]
                        ) if hasattr(wm, "room_at") else None,
                        "observations": [],
                    }
                    if INAV_CARRIER_ACTIVE_VERIFY:
                        support_candidate.update({
                            "started_step": int(step),
                            "actions_taken": 0,
                            "approach_actions": 0,
                            "rotations_taken": 0,
                            "standoff_reached": False,
                            "verification_observers": [],
                            "session_observations": [],
                        })
                        active_carrier_candidate = support_candidate
                        carrier_candidates += 1
                        carrier_sessions_started += 1
                    else:
                        support_inspection_candidates += 1
                    break
            except Exception as exc:
                if INAV_CARRIER_ACTIVE_VERIFY:
                    perception_counters["carrier_verification_errors"] = (
                        perception_counters.get(
                            "carrier_verification_errors", 0
                        ) + 1
                    )
                else:
                    support_inspection_errors += 1
                perception_counters["support_inspection_last_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                raise RuntimeError(
                    "required support-inspection detector failed"
                ) from exc
        approach_cluster = (
            best_cluster(
                target_memory,
                step=step,
                min_quality=INAV_APPROACH_QUALITY,
                max_age_steps=INAV_EVIDENCE_MAX_AGE,
                current_xy=current_xy,
            )
            if INAV_EVIDENCE_NAV else None
        )
        bounded_commitment_active = False
        if (
            approach_cluster is None
            and INAV_EVIDENCE_NAV
            and INAV_BOUNDED_COMMITMENT
            and bounded_commitment_attempts == 0
        ):
            approach_cluster = bounded_commitment_candidate(
                target_memory,
                enabled=True,
                step=step,
                step_cap=step_cap,
                min_quality=INAV_APPROACH_QUALITY,
                current_xy=current_xy,
                xy_allowed=lambda x, y: projected_room_matches_prior(
                    wm,
                    (x, y),
                    plan.get("likely_rooms") or [],
                ),
            )
            bounded_commitment_active = approach_cluster is not None
        if (
            approach_cluster is None
            and INAV_EVIDENCE_NAV
            and INAV_BUDGETED_SCAN_YOLO_WORLD
        ):
            # YOLO-World and DINO confidence/quality scales are not directly
            # comparable.  A direct-label, full-vocabulary rescue detection
            # has already passed its own 0.25 gate, so retain a separately
            # calibrated candidate floor without lowering the DINO threshold.
            # Under ``refresh_cluster``, one 0.25 observation has quality
            # 0.65*0.25 + 0.15 = 0.3125; the frozen 0.30 floor therefore
            # admits every detector-passing rescue rather than fitting an
            # episode-specific score.
            scan_clusters = source_backed_clusters(
                target_memory,
                source="yolo_world_scan",
                step=step,
                max_age_steps=INAV_EVIDENCE_MAX_AGE,
            )
            approach_cluster = best_cluster(
                scan_clusters,
                step=step,
                min_quality=INAV_BUDGETED_SCAN_APPROACH_QUALITY,
                max_age_steps=INAV_EVIDENCE_MAX_AGE,
                current_xy=current_xy,
            )
        pose_graph_cluster = None
        if (
            approach_cluster is None
            and INAV_EVIDENCE_NAV
            and INAV_POSE_EVIDENCE_GRAPH
            and not target_memory
            and pose_graph_actions < INAV_POSE_GRAPH_ACTION_BUDGET
            and step < step_cap
        ):
            likely_rooms = plan.get("likely_rooms") or []
            pose_graph_cluster = best_pose_graph_candidate(
                tentative_memory,
                step=step,
                current_xy=current_xy,
                action_budget_remaining=(
                    INAV_POSE_GRAPH_ACTION_BUDGET - pose_graph_actions
                ),
                max_age_steps=0,
                max_attempts_per_cluster=1,
                max_semantic_rank=INAV_TENTATIVE_MAX_RANK,
                observer_allowed=lambda x, y: (
                    not likely_rooms
                    or projected_room_matches_prior(
                        wm, (x, y), likely_rooms
                    )
                ),
            )
        belief_cluster = None
        if (
            approach_cluster is None
            and pose_graph_cluster is None
            and INAV_EVIDENCE_NAV
            and INAV_BELIEF_REPERCEPTION
            and (
                not INAV_BELIEF_COVERAGE_PRESERVING
                or belief_reperception_actions
                    < INAV_BELIEF_OPPORTUNISTIC_EPISODE_BUDGET
            )
            and step < step_cap
        ):
            belief_cluster = best_target_belief_candidate(
                tentative_memory,
                step=step,
                current_xy=current_xy,
                max_age_steps=(
                    INAV_BELIEF_OPPORTUNISTIC_MAX_AGE
                    if INAV_BELIEF_COVERAGE_PRESERVING
                    else INAV_BELIEF_MAX_AGE
                ),
                max_attempts=(
                    1 if INAV_BELIEF_COVERAGE_PRESERVING
                    else INAV_BELIEF_MAX_ATTEMPTS
                ),
                max_distance_m=(
                    INAV_BELIEF_OPPORTUNISTIC_RADIUS_M
                    if INAV_BELIEF_COVERAGE_PRESERVING else None
                ),
            )
        tentative_cluster = None
        if (approach_cluster is None
                and pose_graph_cluster is None
                and belief_cluster is None
                and INAV_EVIDENCE_NAV
                and INAV_TENTATIVE_MEMORY
                # A semantically admitted target hypothesis always outranks a
                # rejected proposal, even before its quality matures enough
                # for approach.  The tentative state exists only to recover
                # the hard no-memory branch of the perception funnel.
                and not target_memory
                and tentative_probes < INAV_TENTATIVE_PROBE_BUDGET
                and step < step_cap):
            tentative_cluster = best_verification_proposal(
                tentative_memory,
                step=step,
                # Only a proposal visible in the current frame can divert the
                # explorer.  Stale rejected boxes remain diagnostic memory.
                max_age_steps=0,
                max_attempts_per_cluster=1,
                current_xy=current_xy,
                require_semantic_shortlist=True,
                max_clip_rank=INAV_TENTATIVE_MAX_RANK,
                xy_allowed=lambda x, y: projected_room_matches_prior(
                    wm,
                    (x, y),
                    plan.get("likely_rooms") or [],
                ),
            )
        navigation_cluster = (
            approach_cluster or pose_graph_cluster or belief_cluster
            or tentative_cluster or support_candidate
        )
        navigation_cluster_kind = (
            "target" if approach_cluster is not None
            else "pose_graph" if pose_graph_cluster is not None
            else "belief" if belief_cluster is not None
            else "tentative" if tentative_cluster is not None
            else "carrier" if (
                support_candidate is not None and INAV_CARRIER_ACTIVE_VERIFY
            )
            else "support" if support_candidate is not None
            else None
        )
        if scan_frame_meta is not None:
            scan_event_for_policy = budgeted_scan_events[
                int(scan_frame_meta["event_index"])
            ]
            current_policy_evidence = bool(
                (
                    current_dino_evidence is not None
                    and navigation_cluster
                        is current_dino_evidence.get("cluster")
                )
                or (
                    current_tentative_evidence is not None
                    and navigation_cluster
                        is current_tentative_evidence.get("cluster")
                )
            )
            policy_cluster_uid = None
            if current_policy_evidence and current_dino_evidence is not None:
                policy_cluster_uid = current_dino_evidence.get("cluster_uid")
            elif (current_policy_evidence
                    and current_tentative_evidence is not None):
                policy_cluster_uid = current_tentative_evidence.get(
                    "cluster_uid"
                )
            scan_event_for_policy.update({
                "policy_eligible_evidence": current_policy_evidence,
                "policy_candidate_kind": (
                    navigation_cluster_kind if current_policy_evidence else None
                ),
                "policy_cluster_uid": policy_cluster_uid,
            })
            if current_policy_evidence:
                budgeted_scan_policy_evidence_hits += 1
        if navigation_cluster is not None:
            # A policy-eligible target/tentative candidate preempts any
            # unfinished coverage sweep and hands control to approach.
            active_budgeted_scan = None
        if (
            navigation_cluster_kind == "carrier"
            and support_candidate is active_carrier_candidate
            and support_candidate.get("standoff_reached", False)
            and int(support_candidate.get("rotations_taken", 0))
                < INAV_CARRIER_ROTATION_BUDGET
            and step < step_cap
        ):
            prior_yaw = float(pose["yaw"])
            env.look_at_yaw(
                prior_yaw + math.radians(INAV_CARRIER_ROTATION_DEG)
            )
            pose = env.get_pose()
            support_candidate["rotations_taken"] = int(
                support_candidate.get("rotations_taken", 0)
            ) + 1
            support_candidate["actions_taken"] = int(
                support_candidate.get("actions_taken", 0)
            ) + 1
            carrier_actions += 1
            pending_carrier_action = {
                "carrier_candidate_uid": support_candidate.get("cluster_uid"),
                "action_step": int(step),
                "arrived_standoff": True,
                "view_angle_novelty_rad": math.radians(
                    INAV_CARRIER_ROTATION_DEG
                ),
            }
            traj.append({
                "step": step,
                "action": "ROTATE_CARRIER",
                "position": pose["position"],
                "yaw": pose["yaw"],
                "room": wm.room_at(
                    pose["position"][0], pose["position"][1]
                ),
                "see_target_verdict": verdict,
                "engine_stop_meta": stop_meta,
                "evidence_stop_meta": evidence_stop_meta,
                "current_dino_evidence": (
                    {
                        key: value
                        for key, value in current_dino_evidence.items()
                        if key != "cluster"
                    }
                    if current_dino_evidence is not None else None
                ),
                "dino": dino_dets,
                "policy_candidate_kind": "carrier",
                "selected_navigation_cluster_uid": (
                    support_candidate.get("cluster_uid")
                ),
                "carrier_action_meta": {
                    "rotation_index": int(
                        support_candidate["rotations_taken"]
                    ),
                    "rotation_budget": INAV_CARRIER_ROTATION_BUDGET,
                    "rotation_deg": INAV_CARRIER_ROTATION_DEG,
                    "yaw_before_rad": prior_yaw,
                    "yaw_after_rad": float(pose["yaw"]),
                },
            })
            continue
        if active_budgeted_scan is not None and int(
            active_budgeted_scan.get("remaining_rotations", 0)
        ) <= 0:
            active_budgeted_scan = None
        scan_site_meta = None
        if (
            INAV_BUDGETED_VIEW_SCAN
            and navigation_cluster is None
            and active_budgeted_scan is None
        ):
            current_scan_room = (
                wm.room_at(current_xy[0], current_xy[1])
                if hasattr(wm, "room_at") else None
            )
            use_scan_site, scan_site_meta = budgeted_scan_site_decision(
                enabled=True,
                step=step,
                step_cap=step_cap,
                current_xy=current_xy,
                current_room=current_scan_room,
                anchors_by_room=budgeted_scan_anchors,
                total_sites=budgeted_scan_sites,
                has_navigation_candidate=False,
                max_sites=INAV_BUDGETED_SCAN_MAX_SITES,
                max_sites_per_room=(
                    INAV_BUDGETED_SCAN_MAX_SITES_PER_ROOM
                ),
                min_anchor_distance_m=(
                    INAV_BUDGETED_SCAN_MIN_ANCHOR_DISTANCE_M
                ),
                min_step=INAV_BUDGETED_SCAN_MIN_STEP,
                min_budget_fraction=(
                    INAV_BUDGETED_SCAN_MIN_BUDGET_FRACTION
                ),
                required_action_slots=(
                    INAV_BUDGETED_SCAN_ROTATIONS_PER_SITE + 1
                ),
                likely_rooms=plan.get("likely_rooms") or [],
                require_likely_room=(
                    INAV_BUDGETED_SCAN_LIKELY_ROOM_ONLY
                ),
            )
            if use_scan_site:
                room_key = str(scan_site_meta["room_key"])
                budgeted_scan_anchors.setdefault(room_key, []).append(
                    (float(current_xy[0]), float(current_xy[1]))
                )
                budgeted_scan_sites += 1
                budgeted_scan_site_ledger.append({
                    "site_index": int(budgeted_scan_sites),
                    "room_key": room_key,
                    "anchor_xy": [
                        float(current_xy[0]), float(current_xy[1])
                    ],
                    "selected_at_step": int(step),
                    "site_decision": dict(scan_site_meta),
                })
                active_budgeted_scan = {
                    "site_index": budgeted_scan_sites,
                    "room_key": room_key,
                    "anchor_xy": [
                        float(current_xy[0]), float(current_xy[1])
                    ],
                    "rotation_index": 0,
                    "remaining_rotations": (
                        INAV_BUDGETED_SCAN_ROTATIONS_PER_SITE
                    ),
                    "site_decision": scan_site_meta,
                }
        if (
            INAV_BUDGETED_VIEW_SCAN
            and navigation_cluster is None
            and active_budgeted_scan is not None
            and int(active_budgeted_scan.get("remaining_rotations", 0)) > 0
            and step < step_cap
        ):
            prior_yaw = float(pose["yaw"])
            target_yaw = prior_yaw + math.radians(
                INAV_BUDGETED_SCAN_ROTATION_DEG
            )
            env.look_at_yaw(target_yaw)
            pose = env.get_pose()
            active_budgeted_scan["rotation_index"] = int(
                active_budgeted_scan.get("rotation_index", 0)
            ) + 1
            active_budgeted_scan["remaining_rotations"] = int(
                active_budgeted_scan.get("remaining_rotations", 0)
            ) - 1
            budgeted_scan_actions += 1
            scan_event = {
                "event_index": int(len(budgeted_scan_events)),
                "site_index": int(active_budgeted_scan["site_index"]),
                "rotation_index": int(
                    active_budgeted_scan["rotation_index"]
                ),
                "action_step": int(step),
                "room_key": active_budgeted_scan["room_key"],
                "anchor_xy": list(active_budgeted_scan["anchor_xy"]),
                "rotation_deg": float(INAV_BUDGETED_SCAN_ROTATION_DEG),
                "yaw_before_rad": prior_yaw,
                "yaw_after_rad": float(pose["yaw"]),
                "expected_observation_step": int(step + 1),
                "observation_step": None,
                "observation_consumed": False,
                "target_evidence_admitted": False,
                "tentative_evidence_observed": False,
                "policy_eligible_evidence": False,
                "policy_candidate_kind": None,
                "policy_cluster_uid": None,
            }
            budgeted_scan_events.append(scan_event)
            pending_budgeted_scan_observation = {
                key: value for key, value in scan_event.items()
                if key not in {
                    "observation_step", "observation_consumed",
                    "target_evidence_admitted",
                    "tentative_evidence_observed",
                    "policy_eligible_evidence",
                    "policy_candidate_kind",
                    "policy_cluster_uid",
                }
            }
            traj.append({
                "step": step,
                "action": "ROTATE_SCAN",
                "position": pose["position"],
                "yaw": pose["yaw"],
                "room": wm.room_at(
                    pose["position"][0], pose["position"][1]
                ),
                "see_target_verdict": verdict,
                "engine_stop_meta": stop_meta,
                "evidence_stop_meta": evidence_stop_meta,
                "observation_from_budgeted_scan": scan_frame_meta,
                "current_dino_evidence": (
                    {
                        key: value
                        for key, value in current_dino_evidence.items()
                        if key != "cluster"
                    }
                    if current_dino_evidence is not None else None
                ),
                "current_tentative_evidence": (
                    {
                        key: value
                        for key, value in current_tentative_evidence.items()
                        if key != "cluster"
                    }
                    if current_tentative_evidence is not None else None
                ),
                "dino": dino_dets,
                "scan_action_meta": dict(
                    pending_budgeted_scan_observation
                ),
            })
            # ROTATE_SCAN replaces this step's MOVE.  Its single observation
            # is rendered and evaluated at the start of the next loop step.
            continue
        approach_target_xy = None
        approach_result, nav_meta = None, {}
        if (
            navigation_cluster is not None
            and navigation_cluster_kind == "pose_graph"
        ):
            approach_result, nav_meta = pose_anchored_reobservation_waypoint(
                wm,
                navigation_cluster,
                current_xy=current_xy,
                lateral_offset_m=INAV_POSE_GRAPH_LATERAL_M,
                min_baseline_m=INAV_POSE_GRAPH_MIN_BASELINE_M,
                max_move_m=INAV_POSE_GRAPH_MAX_MOVE_M,
                min_predicted_crossing_rad=(
                    INAV_POSE_GRAPH_MIN_CROSSING_RAD
                ),
                image_width=int(rgb.shape[1]),
                horizontal_fov_rad=math.radians(INAV_CAMERA_HFOV_DEG),
            )
            nav_meta["candidate_kind"] = "pose_graph"
            nav_meta["proposal_cluster_uid"] = navigation_cluster.get(
                "cluster_uid"
            )
            face_xy = nav_meta.get("face_direction_xy")
            if approach_result is not None and face_xy:
                approach_target_xy = (
                    float(face_xy[0]), float(face_xy[1])
                )
            else:
                pose_graph_unreachable_candidates += 1
                navigation_cluster = None
                navigation_cluster_kind = None
                pose_graph_cluster = None
                approach_result = None
                approach_target_xy = None
        if (
            navigation_cluster is not None
            and navigation_cluster_kind != "pose_graph"
        ):
            approach_target_xy = (
                float(navigation_cluster["xy"][0]),
                float(navigation_cluster["xy"][1]),
            )
            observer_history = [
                (float(obs["observer_xy"][0]), float(obs["observer_xy"][1]))
                for obs in navigation_cluster.get("observations", [])
                if obs.get("observer_xy")
            ]
            observer_history.extend(
                (float(attempt["observer_xy"][0]),
                 float(attempt["observer_xy"][1]))
                for attempt in navigation_cluster.get(
                    "verification_observers", []
                )
                if attempt.get("observer_xy")
            )
            if bounded_commitment_active:
                failed_observer = navigation_cluster.get(
                    "last_failed_verification_observer_xy"
                )
                if failed_observer and len(failed_observer) >= 2:
                    observer_history.append((
                        float(failed_observer[0]),
                        float(failed_observer[1]),
                    ))
            structured_last_mile = bool(
                INAV_STRUCTURED_VERIFICATION
                and navigation_cluster_kind == "target"
                and math.hypot(
                    current_xy[0] - approach_target_xy[0],
                    current_xy[1] - approach_target_xy[1],
                ) <= INAV_STRUCTURED_VERIFY_RADIUS_M
            )
            target_extent_radius_m = navigation_cluster.get(
                "target_extent_radius_m"
            )
            if target_extent_radius_m is None:
                target_extent_radius_m = next((
                    observation.get("target_extent_radius_m")
                    for observation in reversed(
                        navigation_cluster.get("observations", [])
                    )
                    if observation.get("target_extent_radius_m") is not None
                ), None)
            belief_view_control = belief_controls_viewpoint(
                navigation_cluster_kind, navigation_cluster
            )
            navigation_cluster_uid = str(
                navigation_cluster.get("cluster_uid") or ""
            )
            commitment_used = False
            if (
                INAV_TARGET_STANDOFF_COMMITMENT
                and navigation_cluster_kind == "target"
                and target_standoff_commitment is not None
            ):
                committed_uid = str(
                    target_standoff_commitment.get("cluster_uid") or ""
                )
                committed_face = tuple(
                    target_standoff_commitment["face_direction_xy"]
                )
                target_drift_m = math.hypot(
                    approach_target_xy[0] - float(committed_face[0]),
                    approach_target_xy[1] - float(committed_face[1]),
                )
                release_reason = None
                if navigation_cluster_uid != committed_uid:
                    release_reason = "cluster_changed"
                elif int(target_standoff_commitment["actions"]) >= (
                    INAV_TARGET_STANDOFF_COMMITMENT_MAX_ACTIONS
                ):
                    release_reason = "action_budget_exhausted"
                elif target_standoff_commitment_actions >= (
                    INAV_TARGET_STANDOFF_COMMITMENT_EPISODE_ACTION_BUDGET
                ):
                    release_reason = "episode_action_budget_exhausted"
                elif math.hypot(
                    current_xy[0] - float(committed_face[0]),
                    current_xy[1] - float(committed_face[1]),
                ) <= INAV_VERIFY_DISTANCE_M:
                    release_reason = "entered_verification_radius"
                elif target_drift_m > (
                    INAV_TARGET_STANDOFF_COMMITMENT_MAX_DRIFT_M
                ):
                    release_reason = "target_drift"
                if release_reason is None:
                    committed_waypoint, committed_meta = (
                        committed_standoff_waypoint(
                            wm,
                            current_xy=current_xy,
                            standoff_xy=tuple(
                                target_standoff_commitment["standoff_xy"]
                            ),
                            face_direction_xy=committed_face,
                            step_m=INAV_APPROACH_STEP_M,
                            arrival_tolerance_m=(
                                INAV_STRUCTURED_VERIFY_ARRIVAL_TOLERANCE_M
                            ),
                        )
                    )
                    if committed_waypoint is None:
                        release_reason = str(
                            committed_meta.get("release_reason")
                            or "commitment_unavailable"
                        )
                    else:
                        approach_result = committed_waypoint
                        approach_target_xy = (
                            float(committed_face[0]),
                            float(committed_face[1]),
                        )
                        target_standoff_commitment["actions"] = int(
                            target_standoff_commitment["actions"]
                        ) + 1
                        target_standoff_commitment_actions += 1
                        nav_meta = {
                            **committed_meta,
                            "cluster_uid": navigation_cluster_uid,
                            "commitment_action_index": int(
                                target_standoff_commitment["actions"]
                            ),
                            "commitment_action_budget": (
                                INAV_TARGET_STANDOFF_COMMITMENT_MAX_ACTIONS
                            ),
                            "target_drift_m": round(
                                float(target_drift_m), 4
                            ),
                        }
                        target_standoff_commitment_ledger.append({
                            "step": int(step),
                            "event": "follow",
                            **nav_meta,
                            "waypoint_xy": [
                                float(approach_result[0]),
                                float(approach_result[1]),
                            ],
                            "uses_evaluator_target": False,
                        })
                        commitment_used = True
                if release_reason is not None:
                    target_standoff_commitment_releases += 1
                    target_standoff_commitment_ledger.append({
                        "step": int(step),
                        "event": "release",
                        "cluster_uid": committed_uid,
                        "reason": release_reason,
                        "target_drift_m": round(float(target_drift_m), 4),
                        "uses_evaluator_target": False,
                    })
                    target_standoff_commitment = None

            if not commitment_used:
                approach_result, nav_meta = approach_waypoint(
                    wm,
                    current_xy=current_xy,
                    target_xy=approach_target_xy,
                    observer_history=observer_history,
                    desired_standoff_m=INAV_STANDOFF_M,
                    step_m=(
                        INAV_APPROACH_STEP_M
                        if navigation_cluster_kind == "target"
                        else INAV_BELIEF_STEP_M
                        if navigation_cluster_kind == "belief"
                        else INAV_TENTATIVE_PROBE_STEP_M
                        if navigation_cluster_kind == "tentative"
                        else INAV_CARRIER_STEP_M
                        if navigation_cluster_kind == "carrier"
                        else INAV_SUPPORT_INSPECTION_STEP_M
                    ),
                    prefer_novel_view=(
                        structured_last_mile or bounded_commitment_active
                        or (INAV_BELIEF_REPERCEPTION and belief_view_control)
                        or (
                            navigation_cluster_kind == "carrier"
                            and bool(observer_history)
                        )
                    ),
                    min_view_angle_rad=(
                        INAV_CARRIER_MIN_VIEW_ANGLE_RAD
                        if navigation_cluster_kind == "carrier"
                        else INAV_STRUCTURED_VERIFY_MIN_ANGLE_RAD
                    ),
                    arrival_tolerance_m=(
                        INAV_STRUCTURED_VERIFY_ARRIVAL_TOLERANCE_M
                    ),
                    allow_reachable_region_fallback=(
                        _use_reachable_target_region(
                            navigation_cluster_kind
                        )
                    ),
                    reachable_region_max_standoff_m=(
                        INAV_REACHABLE_TARGET_REGION_MAX_M
                    ),
                    visibility_aware=(
                        INAV_BELIEF_REPERCEPTION and belief_view_control
                    ),
                    target_extent_radius_m=target_extent_radius_m,
                )
                if (
                    INAV_TARGET_STANDOFF_COMMITMENT
                    and navigation_cluster_kind == "target"
                    and approach_result is not None
                    and nav_meta.get("standoff_xy")
                    and target_standoff_commitment_actions < (
                        INAV_TARGET_STANDOFF_COMMITMENT_EPISODE_ACTION_BUDGET
                    )
                    and math.hypot(
                        current_xy[0] - approach_target_xy[0],
                        current_xy[1] - approach_target_xy[1],
                    ) > INAV_VERIFY_DISTANCE_M
                ):
                    target_standoff_commitment = {
                        "cluster_uid": navigation_cluster_uid,
                        "standoff_xy": [
                            float(value) for value in nav_meta["standoff_xy"]
                        ],
                        "face_direction_xy": [
                            float(approach_target_xy[0]),
                            float(approach_target_xy[1]),
                        ],
                        "actions": 1,
                        "created_step": int(step),
                    }
                    target_standoff_commitment_initializations += 1
                    target_standoff_commitment_actions += 1
                    nav_meta["target_standoff_commitment_initialized"] = True
                    nav_meta["commitment_action_index"] = 1
                    nav_meta["commitment_action_budget"] = (
                        INAV_TARGET_STANDOFF_COMMITMENT_MAX_ACTIONS
                    )
                    target_standoff_commitment_ledger.append({
                        "step": int(step),
                        "event": "initialize",
                        "cluster_uid": navigation_cluster_uid,
                        "standoff_xy": list(
                            target_standoff_commitment["standoff_xy"]
                        ),
                        "face_direction_xy": list(
                            target_standoff_commitment["face_direction_xy"]
                        ),
                        "waypoint_xy": [
                            float(approach_result[0]),
                            float(approach_result[1]),
                        ],
                        "uses_evaluator_target": False,
                    })
            nav_meta["candidate_kind"] = navigation_cluster_kind
            nav_meta["structured_last_mile"] = structured_last_mile
            nav_meta["bounded_commitment"] = bounded_commitment_active
            if tentative_cluster is not None:
                semantic_verification = (
                    (tentative_cluster.get("observations") or [{}])[-1].get(
                        "semantic_verification"
                    ) or {}
                )
                nav_meta["room_context_match"] = True
                nav_meta["tentative_semantic_rank"] = int(
                    semantic_verification.get("target_rank", 10_000)
                )
            if belief_cluster is not None:
                belief_candidate_age = (
                    int(step) - int(belief_cluster.get("step", -10_000))
                )
                belief_candidate_distance = math.hypot(
                    current_xy[0] - approach_target_xy[0],
                    current_xy[1] - approach_target_xy[1],
                )
                nav_meta.update({
                    "belief_posterior": belief_cluster.get(
                        "belief_posterior"
                    ),
                    "belief_entropy": belief_cluster.get("belief_entropy"),
                    "belief_viewpoints": belief_cluster.get(
                        "belief_viewpoints"
                    ),
                    "belief_semantic_support_viewpoints": (
                        belief_cluster.get(
                            "belief_semantic_support_viewpoints"
                        )
                    ),
                    "belief_candidate_age_steps": int(belief_candidate_age),
                    "belief_candidate_distance_m": round(
                        float(belief_candidate_distance), 6
                    ),
                })
            if (
                INAV_TARGET_LOCK_CONTROLLER
                and approach_result is None
                and navigation_cluster_kind == "target"
                and approach_cluster is not None
                and not bounded_commitment_active
                and approach_target_xy is not None
            ):
                cluster_uid = str(
                    approach_cluster.get("cluster_uid")
                    or "target:{:.3f}:{:.3f}".format(*approach_target_xy)
                )
                cluster_action_count = int(
                    target_lock_actions_by_cluster.get(cluster_uid, 0)
                )
                target_lock_cluster_distance = math.hypot(
                    current_xy[0] - approach_target_xy[0],
                    current_xy[1] - approach_target_xy[1],
                )
                target_lock_activation_ready = target_lock_activation_allowed(
                    current_xy=current_xy,
                    target_xy=approach_target_xy,
                    activation_radius_m=(
                        INAV_TARGET_LOCK_ACTIVATION_RADIUS_M
                    ),
                )
                target_lock_budget_available = bool(
                    target_lock_actions < INAV_TARGET_LOCK_ACTION_BUDGET
                    and cluster_action_count
                        < INAV_TARGET_LOCK_ACTION_BUDGET
                )
                if not target_lock_activation_ready:
                    target_lock_outside_activation_radius += 1
                elif target_lock_budget_available:
                    ordinary_approach_failure = dict(nav_meta)
                    current_frame_detected = bool(
                        current_dino_evidence is not None
                        and current_dino_evidence.get("cluster")
                            is approach_cluster
                    )
                    approach_result, target_lock_meta = target_lock_waypoint(
                        wm,
                        current_xy=current_xy,
                        target_xy=approach_target_xy,
                        current_frame_detected=current_frame_detected,
                        observer_history=observer_history,
                        max_step_m=INAV_TARGET_LOCK_MAX_STEP_M,
                        terminal_radius_m=(
                            INAV_TARGET_LOCK_TERMINAL_RADIUS_M
                        ),
                        min_progress_m=INAV_TARGET_LOCK_MIN_PROGRESS_M,
                        min_translation_m=(
                            INAV_TARGET_LOCK_MIN_TRANSLATION_M
                        ),
                        max_distance_increase_m=(
                            INAV_TARGET_LOCK_MAX_DISTANCE_INCREASE_M
                        ),
                    )
                    target_lock_actions += 1
                    target_lock_actions_by_cluster[cluster_uid] = (
                        cluster_action_count + 1
                    )
                    target_lock_mode = str(target_lock_meta.get("mode"))
                    if target_lock_mode == "target_lock_monotonic":
                        target_lock_monotonic_moves += 1
                    elif target_lock_mode == "target_lock_tangential":
                        target_lock_tangential_moves += 1
                    else:
                        target_lock_reobservations += 1
                    nav_meta = {
                        **target_lock_meta,
                        "cluster_uid": cluster_uid,
                        "target_lock_action_index": int(
                            target_lock_actions
                        ),
                        "target_lock_cluster_action_index": int(
                            cluster_action_count + 1
                        ),
                        "target_lock_action_budget": (
                            INAV_TARGET_LOCK_ACTION_BUDGET
                        ),
                        "target_lock_activation_radius_m": (
                            INAV_TARGET_LOCK_ACTIVATION_RADIUS_M
                        ),
                        "ordinary_approach_failure": (
                            ordinary_approach_failure
                        ),
                    }
                    target_lock_action_ledger.append({
                        "step": int(step),
                        "cluster_uid": cluster_uid,
                        "mode": target_lock_mode,
                        "current_xy": [
                            float(current_xy[0]), float(current_xy[1])
                        ],
                        "waypoint_xy": [
                            float(approach_result[0]),
                            float(approach_result[1]),
                        ],
                        "face_direction_xy": [
                            float(approach_target_xy[0]),
                            float(approach_target_xy[1]),
                        ],
                        "current_frame_detected": current_frame_detected,
                        "translation_allowed": bool(
                            target_lock_meta.get(
                                "translation_allowed", False
                            )
                        ),
                        "waypoint_displacement_m": target_lock_meta.get(
                            "waypoint_displacement_m"
                        ),
                        "target_distance_before_m": target_lock_meta.get(
                            "target_distance_before_m"
                        ),
                        "target_distance_after_m": target_lock_meta.get(
                            "target_distance_after_m"
                        ),
                        "ordinary_fallback_reason": (
                            ordinary_approach_failure.get(
                                "fallback_reason"
                            )
                        ),
                        "activation_radius_m": (
                            INAV_TARGET_LOCK_ACTIVATION_RADIUS_M
                        ),
                        "uses_evaluator_target": False,
                    })
            if support_candidate is not None:
                nav_meta.update({
                    "room_context_match": True,
                    "support_label": support_candidate.get("label"),
                    "support_score": round(float(
                        support_candidate.get("score", 0.0)
                    ), 6),
                    "support_priority": int(
                        support_candidate.get("support_priority", -1)
                    ),
                    "support_target_category": target_guess,
                    "support_current_room": support_candidate.get(
                        "current_room"
                    ),
                    "support_projected_room": support_candidate.get(
                        "projected_room"
                    ),
                })
                support_waypoint_displacement = float(nav_meta.get(
                    "waypoint_displacement_m", 0.0
                ))
                if navigation_cluster_kind == "carrier":
                    carrier_move_ready = bool(
                        approach_result is not None
                        and math.isfinite(support_waypoint_displacement)
                        and 0.1 <= support_waypoint_displacement
                            <= INAV_CARRIER_STEP_M + 0.15
                    )
                    nav_meta.update({
                        "carrier_move_ready": carrier_move_ready,
                        "carrier_action_index": int(
                            support_candidate.get("actions_taken", 0)
                        ) + 1,
                        "carrier_action_budget": INAV_CARRIER_ACTION_BUDGET,
                        "carrier_session_index": carrier_sessions_started,
                    })
                    if not carrier_move_ready:
                        approach_result = None
                        approach_target_xy = None
                else:
                    support_move_ready = bool(
                        approach_result is not None
                        and math.isfinite(support_waypoint_displacement)
                        and 0.1 <= support_waypoint_displacement <= 0.75
                    )
                    nav_meta["support_move_ready"] = support_move_ready
                    if not support_move_ready:
                        approach_result = None
                        approach_target_xy = None
            if bounded_commitment_active:
                bounded_move_ready = bool(
                    approach_result is not None
                    and nav_meta.get("novel_view_used", False)
                    and float(nav_meta.get(
                        "waypoint_view_angle_novelty_rad", 0.0
                    )) >= INAV_STRUCTURED_VERIFY_MIN_ANGLE_RAD
                    and float(nav_meta.get(
                        "waypoint_displacement_m", 0.0
                    )) >= 0.1
                )
                if not bounded_move_ready:
                    approach_result = None
                    approach_target_xy = None
                    bounded_commitment_active = False
                nav_meta["quality_before_last_failure"] = round(float(
                    approach_cluster.get(
                        "quality_before_last_failure", 0.0
                    )
                ), 6)
                nav_meta["room_context_match"] = bounded_move_ready
            if tentative_cluster is not None and approach_result is None:
                # No target-directed action was executable.  The next branch
                # will run the ordinary frontier policy, so do not label that
                # unrelated MOVE as a tentative probe or attach its UID.
                navigation_cluster = None
                navigation_cluster_kind = None
                tentative_cluster = None
                approach_target_xy = None
            if belief_cluster is not None and approach_result is None:
                navigation_cluster = None
                navigation_cluster_kind = None
                belief_cluster = None
                approach_target_xy = None
            elif (
                belief_cluster is not None
                and INAV_BELIEF_COVERAGE_PRESERVING
            ):
                displacement_m = float(
                    nav_meta.get("waypoint_displacement_m", float("inf"))
                )
                opportunistic_ready = bool(
                    nav_meta.get("arrived_standoff", False)
                    and nav_meta.get("novel_view_used", False)
                    and math.isfinite(displacement_m)
                    and 0.1 <= displacement_m
                        <= INAV_BELIEF_OPPORTUNISTIC_MAX_MOVE_M
                )
                nav_meta["belief_opportunistic_ready"] = opportunistic_ready
                if not opportunistic_ready:
                    belief_opportunistic_rejections.append({
                        "step": int(step),
                        "proposal_cluster_uid": belief_cluster.get(
                            "cluster_uid"
                        ),
                        "candidate_age_steps": nav_meta.get(
                            "belief_candidate_age_steps"
                        ),
                        "candidate_distance_m": nav_meta.get(
                            "belief_candidate_distance_m"
                        ),
                        "waypoint_displacement_m": (
                            round(displacement_m, 6)
                            if math.isfinite(displacement_m) else None
                        ),
                        "arrived_standoff": bool(
                            nav_meta.get("arrived_standoff", False)
                        ),
                        "novel_view_used": bool(
                            nav_meta.get("novel_view_used", False)
                        ),
                    })
                    approach_result = None
                    navigation_cluster = None
                    navigation_cluster_kind = None
                    belief_cluster = None
                    approach_target_xy = None
            if support_candidate is not None and approach_result is None:
                # A non-executable support approach must not relabel the
                # ordinary frontier fallback as an attributed inspection.
                if (
                    navigation_cluster_kind == "carrier"
                    and active_carrier_candidate is support_candidate
                ):
                    carrier_failures += 1
                    carrier_rejected_xy.append([
                        float(support_candidate["xy"][0]),
                        float(support_candidate["xy"][1]),
                    ])
                    carrier_resolutions.append({
                        "carrier_candidate_uid": support_candidate.get(
                            "cluster_uid"
                        ),
                        "support_label": support_candidate.get("label"),
                        "support_xy": support_candidate.get("xy"),
                        "started_step": support_candidate.get("started_step"),
                        "resolved_step": int(step),
                        "actions_taken": int(
                            support_candidate.get("actions_taken", 0)
                        ),
                        "outcome": "unreachable",
                        "accepted_target_cluster_uid": None,
                        "session_observations": list(
                            support_candidate.get("session_observations", [])
                        ),
                    })
                    active_carrier_candidate = None
                navigation_cluster = None
                navigation_cluster_kind = None
                support_candidate = None
                approach_target_xy = None
        local_frontier_result, local_frontier_meta = None, {}
        if approach_result is None and INAV_HYBRID_FRONTIER:
            approach_target_xy = None
            numeric_id = int("".join(ch for ch in sel_id if ch.isdigit()) or "0")
            frontier_candidates = env.sample_frontier_waypoints(
                wm,
                K=INAV_FRONTIER_WAYPOINTS,
                seed=INAV_FRONTIER_SEED + numeric_id * 1009 + step,
                force_angular_spread_rad=2.0 * math.pi,
            )
            if (
                INAV_TARGET_VISUAL_MEMORY_FRONTIER
                and target_visual_memory_ready
            ):
                target_visual_frontier_calls += 1
                target_visual_memory_ledger[-1]["frontier_called"] = True
                local_frontier_result, local_frontier_meta = (
                    select_coverage_constrained_expected_improvement_frontier(
                        frontier_candidates,
                        wm,
                        history=agent_path,
                        current_xy=current_xy,
                        semantic_belief_at=(
                            vmap.probabilistic_visual_belief_at
                        ),
                    )
                )
                target_visual_memory_ledger[-1].update({
                    "frontier_selected": bool(
                        local_frontier_result is not None
                    ),
                    "semantic_rerank_applied": bool(
                        local_frontier_meta.get(
                            "semantic_rerank_applied", False
                        )
                    ),
                    "semantic_mean": local_frontier_meta.get(
                        "semantic_mean"
                    ),
                    "semantic_variance": local_frontier_meta.get(
                        "semantic_variance"
                    ),
                    "expected_improvement": local_frontier_meta.get(
                        "expected_improvement"
                    ),
                    "frontier_score": local_frontier_meta.get(
                        "frontier_score"
                    ),
                    "best_geometry_score": local_frontier_meta.get(
                        "best_geometry_score"
                    ),
                })
                if local_frontier_result is not None:
                    target_visual_frontier_moves += 1
                if local_frontier_meta.get("semantic_rerank_applied", False):
                    target_visual_rerank_changes += 1
            else:
                local_frontier_result, local_frontier_meta = (
                    select_frontier_waypoint(
                        frontier_candidates,
                        wm,
                        history=agent_path,
                        current_xy=current_xy,
                    )
                )
            if local_frontier_result is None:
                hybrid_frontier_fallbacks += 1

        use_global_frontier = should_use_global_frontier(
            enabled=INAV_GLOBAL_FRONTIER,
            navigation_cluster=navigation_cluster,
            local_frontier_meta=local_frontier_meta,
            score_threshold=INAV_GLOBAL_FRONTIER_SCORE_THRESHOLD,
        )
        if approach_result is None and use_global_frontier:
            approach_target_xy = None
            global_frontier_triggers += 1
            global_waypoint, global_meta = global_geometry_frontier_waypoint(
                enabled=True,
                navigation_cluster=None,
                value_map=vmap,
                current_xy=current_xy,
                agent_path=agent_path,
            )
            if global_waypoint is not None:
                approach_result, nav_meta = global_waypoint, global_meta
                global_frontier_moves += 1
            else:
                global_frontier_fallbacks += 1
        elif (approach_result is None and INAV_GLOBAL_FRONTIER
              and navigation_cluster is None
              and local_frontier_result is not None):
            global_frontier_high_novelty_skips += 1

        if approach_result is None and local_frontier_result is not None:
            approach_result, nav_meta = (
                local_frontier_result, local_frontier_meta
            )
            hybrid_frontier_moves += 1

        if approach_result is not None:
            nx, ny = approach_result
            if INAV_GEOMETRY_SAFE_FRONTIER:
                geometry_safe_waypoint_checks += 1
                proposed_xy = (float(nx), float(ny))
                if not policy_waypoint_is_reachable(
                    wm, current_xy, proposed_xy
                ):
                    geometry_safe_waypoint_rejections += 1
                    rejected_meta = dict(nav_meta)
                    rejected_kind = navigation_cluster_kind
                    recovery, recovery_meta = (
                        global_geometry_frontier_waypoint(
                            enabled=True,
                            navigation_cluster=None,
                            value_map=vmap,
                            current_xy=current_xy,
                            agent_path=agent_path,
                        )
                    )
                    recovery_valid = bool(
                        recovery is not None
                        and policy_waypoint_is_reachable(
                            wm, current_xy, recovery
                        )
                    )
                    if recovery_valid:
                        nx, ny = recovery
                        geometry_safe_global_recoveries += 1
                        recovery_mode = "global_frontier_recovery"
                    else:
                        nx, ny = current_xy
                        geometry_safe_hold_position_failures += 1
                        recovery_mode = "hold_position"
                    approach_result = (float(nx), float(ny))
                    approach_target_xy = None
                    navigation_cluster = None
                    navigation_cluster_kind = None
                    nav_meta = {
                        **dict(recovery_meta),
                        "mode": "geometry_safe_frontier",
                        "geometry_safe_recovery_mode": recovery_mode,
                        "rejected_waypoint": [
                            float(proposed_xy[0]), float(proposed_xy[1])
                        ],
                        "rejected_candidate_kind": rejected_kind,
                        "rejected_nav_meta": rejected_meta,
                    }
                    geometry_safe_ledger.append({
                        "step": int(step),
                        "rejected_waypoint": [
                            float(proposed_xy[0]), float(proposed_xy[1])
                        ],
                        "rejected_candidate_kind": rejected_kind,
                        "recovery_mode": recovery_mode,
                        "recovery_waypoint": [float(nx), float(ny)],
                    })
            if navigation_cluster_kind == "target" and approach_target_xy:
                verification_ready = bool(
                    not INAV_STRUCTURED_VERIFICATION
                    or (
                        nav_meta.get("structured_last_mile", False)
                        and nav_meta.get("arrived_standoff", False)
                        and nav_meta.get("novel_view_used", False)
                        and float(nav_meta.get("view_angle_novelty_rad", 0.0))
                            >= INAV_STRUCTURED_VERIFY_MIN_ANGLE_RAD
                    )
                )
                if verification_ready:
                    verification_candidate = approach_cluster
                    verification_candidate_from_bounded_commitment = bool(
                        bounded_commitment_active
                    )
                    if bounded_commitment_active:
                        approach_cluster["bounded_commitment_attempts"] = (
                            int(approach_cluster.get(
                                "bounded_commitment_attempts", 0
                            )) + 1
                        )
                        approach_cluster["last_bounded_commitment_step"] = int(
                            step
                        )
                        bounded_commitment_attempts += 1
                    if bounded_commitment_active or INAV_STRUCTURED_VERIFICATION:
                        mark_verification_attempt(
                            approach_cluster,
                            step=step,
                            observer_xy=(float(nx), float(ny)),
                            target_xy=approach_target_xy,
                            min_angle_rad=(
                                INAV_STRUCTURED_VERIFY_MIN_ANGLE_RAD
                            ),
                        )
                        evidence_verification_attempts += 1
            elif (
                navigation_cluster_kind == "pose_graph"
                and approach_target_xy
            ):
                pending_pose_graph_candidate = pose_graph_cluster
                pending_pose_graph_action_step = int(step)
                mark_verification_attempt(pose_graph_cluster, step=step)
                pose_graph_actions += 1
                pose_graph_action_ledger.append({
                    "proposal_cluster_uid": pose_graph_cluster.get(
                        "cluster_uid"
                    ),
                    "action_step": int(step),
                    "waypoint_xy": [float(nx), float(ny)],
                    "face_direction_xy": [
                        float(approach_target_xy[0]),
                        float(approach_target_xy[1]),
                    ],
                    "source_anchor_xy": nav_meta.get("source_anchor_xy"),
                    "source_ray_yaw": nav_meta.get("source_ray_yaw"),
                    "view_baseline_m": nav_meta.get("view_baseline_m"),
                    "predicted_crossing_angle_rad": nav_meta.get(
                        "predicted_crossing_angle_rad"
                    ),
                    "waypoint_displacement_m": nav_meta.get(
                        "waypoint_displacement_m"
                    ),
                    "uncertain_world_xy_used_as_goal": False,
                    "expected_observation_step": int(step + 1),
                })
            elif navigation_cluster_kind == "belief" and approach_target_xy:
                pending_belief_candidate = belief_cluster
                pending_belief_action_step = int(step)
                mark_verification_attempt(
                    belief_cluster,
                    step=step,
                    observer_xy=(float(nx), float(ny)),
                    target_xy=approach_target_xy,
                    min_angle_rad=INAV_STRUCTURED_VERIFY_MIN_ANGLE_RAD,
                )
                belief_reperception_actions += 1
            elif navigation_cluster_kind == "tentative" and approach_target_xy:
                tentative_verification_candidate = tentative_cluster
                tentative_verification_action_step = int(step)
                mark_verification_attempt(tentative_cluster, step=step)
                tentative_probes += 1
            elif navigation_cluster_kind == "support" and approach_target_xy:
                pending_support_candidate = support_candidate
                pending_support_action_step = int(step)
                support_inspections += 1
            elif navigation_cluster_kind == "carrier" and approach_target_xy:
                if active_carrier_candidate is not support_candidate:
                    raise RuntimeError(
                        "carrier MOVE is not linked to the active session"
                    )
                support_candidate["actions_taken"] = int(
                    support_candidate.get("actions_taken", 0)
                ) + 1
                support_candidate["approach_actions"] = int(
                    support_candidate.get("approach_actions", 0)
                ) + 1
                carrier_actions += 1
                pending_carrier_action = {
                    "carrier_candidate_uid": support_candidate.get(
                        "cluster_uid"
                    ),
                    "action_step": int(step),
                    "arrived_standoff": bool(
                        nav_meta.get("arrived_standoff", False)
                    ),
                    "view_angle_novelty_rad": nav_meta.get(
                        "view_angle_novelty_rad"
                    ),
                }
        else:
            (nx, ny), nav_meta = vmap.next_waypoint(
                current_xy,
                target_memory, agent_path,
                direction_hint=direction_hint,
            )
            approach_target_xy = None
            if nav_meta.get("fallback_reason") == "argmax = current cell":
                # Already at the value-map peak but target is unverified:
                # take a small forward step to break a discretization tie.
                nx = pose["position"][0] + 0.5 * math.cos(pose["yaw"])
                ny = pose["position"][1] + 0.5 * math.sin(pose["yaw"])
        executed_policy_candidate_kind = (
            navigation_cluster_kind
            if approach_target_xy is not None else None
        )
        executed_navigation_cluster_uid = (
            navigation_cluster.get("cluster_uid")
            if executed_policy_candidate_kind is not None
            and navigation_cluster is not None else None
        )
        env.teleport_to(
            (nx, ny),
            face_direction_xy=approach_target_xy,
        )
        pose = env.get_pose()
        agent_path.append((pose["position"][0], pose["position"][1]))
        room = wm.room_at(pose["position"][0], pose["position"][1])
        if room and room not in rooms_visited:
            rooms_visited.append(room)
        traj.append({"step": step, "action": "MOVE",
                     "waypoint": [nx, ny],
                     "position": pose["position"], "yaw": pose["yaw"],
                     "room": room,
                     "see_target_verdict": verdict,
                     "engine_stop_meta": stop_meta,
                     "nav_meta": nav_meta,
                     "policy_candidate_kind": (
                         executed_policy_candidate_kind
                     ),
                     "selected_navigation_cluster_uid": (
                         executed_navigation_cluster_uid
                     ),
                     "vlm_admitted": vlm_admitted,
                     "current_dino_evidence": (
                         {key: value for key, value in current_dino_evidence.items()
                          if key != "cluster"}
                         if current_dino_evidence is not None else None
                     ),
                     "current_tentative_evidence": (
                         {
                             key: value
                             for key, value in current_tentative_evidence.items()
                             if key != "cluster"
                         }
                         if current_tentative_evidence is not None else None
                     ),
                     "observation_from_budgeted_scan": scan_frame_meta,
                     "evidence_stop_meta": evidence_stop_meta,
                     "dino": dino_dets,
                     "usage": vlm_usage,
                     **({"retried": True} if retried else {})})

    # ---- Episode end ----
    # No post-budget recovery action is allowed. ``step_cap`` means the last
    # valid pose is simply the last pose reached inside the action budget.
    if final_rgb is None:
        final_rgb = env.render_rgb()
    final_frame_path = epi_dir / "final.png"
    env.save_frame(final_rgb, final_frame_path)
    # Evaluation-only ground truth.  This is read strictly after the policy
    # has terminated and never enters target memory, planning, STOP, or any
    # other agent branch.
    simulator_instance_visibility = None
    if INAV_INSTANCE_VISIBILITY_DIAGNOSTIC and hasattr(
        env, "target_instance_visibility"
    ):
        simulator_instance_visibility = env.target_instance_visibility(
            str((episode_meta_for_record or {}).get("target_object_id") or "")
        )

    usage_total = {
        "n_calls": len(usage_calls),
        "input_tokens_total": sum(u.get("input_tokens", 0) for u in usage_calls),
        "output_tokens_total": sum(u.get("output_tokens", 0) for u in usage_calls),
        "latency_s_total": round(sum(u.get("latency_s", 0.0) for u in usage_calls), 4),
    }
    if open_weight_provenance:
        plan_model = (
            f"{open_weight_provenance['model_id']}@"
            f"{open_weight_provenance['model_revision']}"
        )
        model_provider = "local_open_weight"
        model_name = (
            f"{plan_model} + IDEA-Research/grounding-dino-tiny"
        )
    elif local_perception_only:
        plan_model = "deterministic_explicit_target"
        model_provider = "local"
        model_name = "IDEA-Research/grounding-dino-tiny"
    else:
        plan_model = MODEL_CATALOG[model_key]["model"]
        model_provider = MODEL_CATALOG[model_key]["provider"]
        model_name = MODEL_CATALOG[model_key]["model"]

    record = {
        "selection_id": sel_id,
        "tier": tier_name,
        "navigation_agent": "IntentionNav",
        "model": record_model,
        "style": style,
        "scene_id": item["scene_id"],
        # Ground truth must remain immutable; the inferred category is stored
        # only under prediction/plan and is what IM evaluates.
        "target_category": item["target_category"],
        "intent": intent,
        "photo": item.get("photo"),
        "final_frame": str(final_frame_path.relative_to(REPO)),
        "prediction": {"target": final_target},
        "plan": {**plan,
                 "plan_model": plan_model,
                 "plan_usage": plan_usage},
        "trajectory": traj,
        "rooms_visited": rooms_visited,
        "seen_objects": sorted(seen_objects),
        "target_memory": target_memory,
        "tentative_memory": tentative_memory,
        "stop_reason": stop_reason,
        "step_cap": step_cap,
        "episode_meta": episode_meta_for_record,
        "evaluation_diagnostics": {
            "simulator_instance_visibility": simulator_instance_visibility,
        },
        "evaluation_protocol": {
            "version": (
                "intentionnav_curated_v3_2026_09_02"
                if open_weight_provenance else "strict_stop_2026_08"
            ),
            "goal_input": "explicit_target" if objectnav else "implicit_intent",
            "metric_scope": "navigation_only" if objectnav else "full_chain",
            "stop_is_terminal": True,
            "post_budget_actions": False,
            "post_budget_policy_observations": False,
            "scene_candidates": scene_candidates,
            "sensor_protocol": _sensor_protocol(
                auto_reorient=AUTO_REORIENT,
                pano_init=VM_PANO_INIT,
                mid_pano_scan=VM_MID_PANO_SCAN,
                budgeted_view_scan=INAV_BUDGETED_VIEW_SCAN,
            ),
            "automatic_start_reorientation": AUTO_REORIENT,
            "episode_start_pano": VM_PANO_INIT,
            "mid_episode_pano": VM_MID_PANO_SCAN,
        },
        "model_meta": {
            "provider": model_provider,
            "model": model_name,
            "temperature": 0.0,
            "api_calls": 0 if local_perception_only else len(usage_calls),
            "execution_seed": execution_seed,
        },
        "ablation_flags": {
            "fallback_retry": FALLBACK_RETRY_ENABLED,
            "engine_picker": True,   # always-on for this tier
            "objectnav": objectnav,
            "scene_candidates": scene_candidates,
            "local_perception_only": local_perception_only,
            "robustness_mode": normalize_robustness_mode(robustness_mode),
            "vm_dino_threshold": VM_DINO_THRESHOLD,
            "vm_no_admit": VM_NO_ADMIT,
            "vm_no_vlm_admit": VM_NO_VLM_ADMIT,
            "vm_stop_consecutive": VM_STOP_CONSECUTIVE,
            # v2 additions
            "vm_strong_stop_radius_m": VM_STRONG_STOP_RADIUS_M,
            "vm_strong_stop_score_min": VM_STRONG_STOP_SCORE_MIN,
            "vm_pano_init": VM_PANO_INIT,
            "pano_init_admitted": pano_init_admitted,
            "vm_mid_pano_scan": VM_MID_PANO_SCAN,
            "inav_mid_pano_all_rooms": INAV_MID_PANO_ALL_ROOMS,
            "mid_pano_scans": mid_pano_scans,
            "mid_pano_admitted": mid_pano_admitted,
            "stop_path": stop_path,  # which gate fired (strong/standard/None)
            "vm_dino_admit_rooms_only": VM_DINO_ADMIT_ROOMS_ONLY,
            "dino_admit_room_rejected": dino_admit_room_rejected,
            "matched_likely_rooms": vmap.matched_rooms_diagnostic,
            # Fix D additions
            "vm_wall_pierce_reject": VM_WALL_PIERCE_REJECT,
            "wall_pierce_rejected": wall_pierce_rejected,
            "vm_strong_stop_require_confirm": VM_STRONG_STOP_REQUIRE_CONFIRM,
            "vm_strong_stop_yaw_spread_rad": VM_STRONG_STOP_YAW_SPREAD_RAD,
            "strong_stop_demoted": strong_stop_demoted,
            "vm_standard_stop_require_dino": VM_STANDARD_STOP_REQUIRE_DINO,
            "standard_stop_demoted": standard_stop_demoted,
            # X1 detection-driven STOP
            "vm_dino_stop_enabled": VM_DINO_STOP_ENABLED,
            "vm_dino_stop_score_min": VM_DINO_STOP_SCORE_MIN,
            "vm_dino_stop_radius_m": VM_DINO_STOP_RADIUS_M,
            "dino_stop_fired": dino_stop_fired,
            # IntentionNav evidence-aware target approach + verification.
            "inav_evidence_nav": INAV_EVIDENCE_NAV,
            "inav_target_only_dino": INAV_TARGET_ONLY_DINO,
            "inav_target_detector_queries": target_queries,
            "inav_approach_quality": INAV_APPROACH_QUALITY,
            "inav_stop_dino_score": INAV_STOP_DINO_SCORE,
            "inav_stop_depth_m": INAV_STOP_DEPTH_M,
            "inav_stop_cluster_distance_m": INAV_STOP_CLUSTER_DISTANCE_M,
            "inav_adaptive_stop": INAV_ADAPTIVE_STOP,
            "inav_verified_stop_dino_score": INAV_VERIFIED_STOP_DINO_SCORE,
            "inav_verified_stop_cluster_distance_m": (
                INAV_VERIFIED_STOP_CLUSTER_DISTANCE_M
            ),
            "inav_large_bbox_max_depth_m": INAV_LARGE_BBOX_MAX_DEPTH_M,
            "inav_large_bbox_max_area_fraction": (
                INAV_LARGE_BBOX_MAX_AREA_FRACTION
            ),
            "adaptive_stop_relaxations": adaptive_stop_relaxations,
            "inav_approach_step_m": INAV_APPROACH_STEP_M,
            "inav_reachable_target_region": (
                INAV_REACHABLE_TARGET_REGION
            ),
            "inav_reachable_target_region_max_m": (
                INAV_REACHABLE_TARGET_REGION_MAX_M
            ),
            "inav_bounded_commitment": INAV_BOUNDED_COMMITMENT,
            "bounded_commitment_attempts": bounded_commitment_attempts,
            "bounded_commitment_redetections": (
                bounded_commitment_redetections
            ),
            "bounded_commitment_misses": bounded_commitment_misses,
            "inav_structured_verification": INAV_STRUCTURED_VERIFICATION,
            "inav_structured_verify_radius_m": (
                INAV_STRUCTURED_VERIFY_RADIUS_M
            ),
            "inav_structured_verify_min_angle_rad": (
                INAV_STRUCTURED_VERIFY_MIN_ANGLE_RAD
            ),
            "inav_structured_verify_arrival_tolerance_m": (
                INAV_STRUCTURED_VERIFY_ARRIVAL_TOLERANCE_M
            ),
            "terminal_frame_source": terminal_frame_source,
            "inav_global_frontier": INAV_GLOBAL_FRONTIER,
            "inav_global_frontier_score_threshold": (
                INAV_GLOBAL_FRONTIER_SCORE_THRESHOLD
            ),
            "global_frontier_moves": global_frontier_moves,
            "global_frontier_fallbacks": global_frontier_fallbacks,
            "global_frontier_triggers": global_frontier_triggers,
            "global_frontier_high_novelty_skips": (
                global_frontier_high_novelty_skips
            ),
            "inav_geometry_safe_frontier": INAV_GEOMETRY_SAFE_FRONTIER,
            "geometry_safe_waypoint_checks": geometry_safe_waypoint_checks,
            "geometry_safe_waypoint_rejections": (
                geometry_safe_waypoint_rejections
            ),
            "geometry_safe_global_recoveries": (
                geometry_safe_global_recoveries
            ),
            "geometry_safe_hold_position_failures": (
                geometry_safe_hold_position_failures
            ),
            "geometry_safe_ledger": geometry_safe_ledger,
            "inav_target_lock_controller": INAV_TARGET_LOCK_CONTROLLER,
            "inav_target_lock_action_budget": (
                INAV_TARGET_LOCK_ACTION_BUDGET
            ),
            "inav_target_lock_max_step_m": INAV_TARGET_LOCK_MAX_STEP_M,
            "inav_target_lock_terminal_radius_m": (
                INAV_TARGET_LOCK_TERMINAL_RADIUS_M
            ),
            "inav_target_lock_activation_radius_m": (
                INAV_TARGET_LOCK_ACTIVATION_RADIUS_M
            ),
            "inav_target_lock_min_progress_m": (
                INAV_TARGET_LOCK_MIN_PROGRESS_M
            ),
            "inav_target_lock_min_translation_m": (
                INAV_TARGET_LOCK_MIN_TRANSLATION_M
            ),
            "inav_target_lock_max_distance_increase_m": (
                INAV_TARGET_LOCK_MAX_DISTANCE_INCREASE_M
            ),
            "target_lock_actions": target_lock_actions,
            "target_lock_monotonic_moves": target_lock_monotonic_moves,
            "target_lock_tangential_moves": target_lock_tangential_moves,
            "target_lock_reobservations": target_lock_reobservations,
            "target_lock_outside_activation_radius": (
                target_lock_outside_activation_radius
            ),
            "target_lock_actions_by_cluster": (
                target_lock_actions_by_cluster
            ),
            "target_lock_action_ledger": target_lock_action_ledger,
            "inav_target_standoff_commitment": (
                INAV_TARGET_STANDOFF_COMMITMENT
            ),
            "inav_target_standoff_commitment_max_actions": (
                INAV_TARGET_STANDOFF_COMMITMENT_MAX_ACTIONS
            ),
            "inav_target_standoff_commitment_episode_action_budget": (
                INAV_TARGET_STANDOFF_COMMITMENT_EPISODE_ACTION_BUDGET
            ),
            "inav_target_standoff_commitment_max_drift_m": (
                INAV_TARGET_STANDOFF_COMMITMENT_MAX_DRIFT_M
            ),
            "target_standoff_commitment_initializations": (
                target_standoff_commitment_initializations
            ),
            "target_standoff_commitment_actions": (
                target_standoff_commitment_actions
            ),
            "target_standoff_commitment_releases": (
                target_standoff_commitment_releases
            ),
            "target_standoff_commitment_ledger": (
                target_standoff_commitment_ledger
            ),
            "inav_target_bbox_recenter": INAV_TARGET_BBOX_RECENTER,
            "inav_target_bbox_recenter_action_budget": (
                INAV_TARGET_BBOX_RECENTER_ACTION_BUDGET
            ),
            "inav_target_bbox_recenter_score_min": (
                INAV_TARGET_BBOX_RECENTER_SCORE_MIN
            ),
            "inav_target_bbox_recenter_edge_fraction_min": (
                INAV_TARGET_BBOX_RECENTER_EDGE_FRACTION_MIN
            ),
            "inav_target_bbox_recenter_max_rotation_deg": (
                INAV_TARGET_BBOX_RECENTER_MAX_ROTATION_DEG
            ),
            "inav_target_bbox_recenter_memory_radius_m": (
                INAV_TARGET_BBOX_RECENTER_MEMORY_RADIUS_M
            ),
            "target_bbox_recenter_actions": target_bbox_recenter_actions,
            "target_bbox_recenter_hits": target_bbox_recenter_hits,
            "target_bbox_recenter_misses": target_bbox_recenter_misses,
            "target_bbox_recenter_stop_vetoes": (
                target_bbox_recenter_stop_vetoes
            ),
            "target_bbox_recenter_pending_at_termination": bool(
                pending_target_bbox_recenter_event is not None
            ),
            "target_bbox_recenter_ledger": target_bbox_recenter_ledger,
            "inav_persistent_single_encoder_veto": (
                INAV_PERSISTENT_SINGLE_ENCODER_VETO
            ),
            "inav_persistent_single_encoder_veto_observations": (
                INAV_PERSISTENT_SINGLE_ENCODER_VETO_OBSERVATIONS
            ),
            "persistent_single_encoder_stop_vetoes": (
                persistent_single_encoder_stop_vetoes
            ),
            "persistent_single_encoder_stop_veto_ledger": (
                persistent_single_encoder_stop_veto_ledger
            ),
            "inav_terminal_evidence_quorum": (
                INAV_TERMINAL_EVIDENCE_QUORUM
            ),
            "inav_terminal_quorum_persistent_observations": (
                INAV_TERMINAL_QUORUM_PERSISTENT_OBSERVATIONS
            ),
            "inav_terminal_quorum_weak_rank_max": (
                INAV_TERMINAL_QUORUM_WEAK_RANK_MAX
            ),
            "inav_terminal_quorum_contextual_min_votes": (
                INAV_TERMINAL_QUORUM_CONTEXTUAL_MIN_VOTES
            ),
            "terminal_evidence_quorum_vetoes": (
                terminal_evidence_quorum_vetoes
            ),
            "terminal_evidence_quorum_ledger": (
                terminal_evidence_quorum_ledger
            ),
            "legacy_default_fan_calls": int(getattr(
                env, "_legacy_default_fan_calls", 0
            )),
            "geometry_safe_default_fan_calls": int(getattr(
                env, "_geometry_safe_default_fan_calls", 0
            )),
            "geometry_safe_default_fan_candidates": int(getattr(
                env, "_geometry_safe_default_fan_candidates", 0
            )),
            "inav_target_visual_memory_frontier": (
                INAV_TARGET_VISUAL_MEMORY_FRONTIER
            ),
            "target_visual_memory_observations": (
                target_visual_memory_observations
            ),
            "target_visual_memory_informative": (
                target_visual_memory_informative
            ),
            "target_visual_frontier_calls": target_visual_frontier_calls,
            "target_visual_frontier_moves": target_visual_frontier_moves,
            "target_visual_frontier_fallbacks": (
                target_visual_frontier_fallbacks
            ),
            "target_visual_rerank_changes": target_visual_rerank_changes,
            "target_visual_memory_ledger": target_visual_memory_ledger,
            "inav_hybrid_frontier": INAV_HYBRID_FRONTIER,
            "inav_frontier_waypoints": INAV_FRONTIER_WAYPOINTS,
            "inav_frontier_seed": INAV_FRONTIER_SEED,
            "inav_explored_radius_m": INAV_EXPLORED_RADIUS_M,
            "hybrid_frontier_moves": hybrid_frontier_moves,
            "hybrid_frontier_fallbacks": hybrid_frontier_fallbacks,
            "inav_tentative_memory": INAV_TENTATIVE_MEMORY,
            "inav_tentative_probe_budget": INAV_TENTATIVE_PROBE_BUDGET,
            "inav_tentative_probe_step_m": INAV_TENTATIVE_PROBE_STEP_M,
            "inav_tentative_max_rank": INAV_TENTATIVE_MAX_RANK,
            "inav_pose_evidence_graph": INAV_POSE_EVIDENCE_GRAPH,
            "inav_pose_graph_action_budget": (
                INAV_POSE_GRAPH_ACTION_BUDGET
            ),
            "inav_pose_graph_lateral_m": INAV_POSE_GRAPH_LATERAL_M,
            "inav_pose_graph_min_baseline_m": (
                INAV_POSE_GRAPH_MIN_BASELINE_M
            ),
            "inav_pose_graph_max_move_m": INAV_POSE_GRAPH_MAX_MOVE_M,
            "inav_pose_graph_min_crossing_rad": (
                INAV_POSE_GRAPH_MIN_CROSSING_RAD
            ),
            "inav_pose_graph_max_crossing_rad": (
                INAV_POSE_GRAPH_MAX_CROSSING_RAD
            ),
            "inav_pose_graph_max_range_m": INAV_POSE_GRAPH_MAX_RANGE_M,
            "inav_pose_graph_accepted_residual_m": (
                INAV_POSE_GRAPH_ACCEPTED_RESIDUAL_M
            ),
            "pose_graph_actions": pose_graph_actions,
            "pose_graph_promotions": pose_graph_promotions,
            "pose_graph_misses": pose_graph_misses,
            "pose_graph_unreachable_candidates": (
                pose_graph_unreachable_candidates
            ),
            "pose_graph_resolutions": pose_graph_resolutions,
            "pose_graph_action_ledger": pose_graph_action_ledger,
            "pose_graph_pending_at_termination": bool(
                pending_pose_graph_candidate is not None
            ),
            "inav_belief_reperception": INAV_BELIEF_REPERCEPTION,
            "inav_belief_confirm_threshold": BELIEF_CONFIRM_THRESHOLD,
            "inav_belief_stop_threshold": BELIEF_STOP_THRESHOLD,
            "inav_belief_max_age": INAV_BELIEF_MAX_AGE,
            "inav_belief_max_attempts": INAV_BELIEF_MAX_ATTEMPTS,
            "inav_belief_step_m": INAV_BELIEF_STEP_M,
            "inav_belief_coverage_preserving": (
                INAV_BELIEF_COVERAGE_PRESERVING
            ),
            "inav_belief_opportunistic_radius_m": (
                INAV_BELIEF_OPPORTUNISTIC_RADIUS_M
            ),
            "inav_belief_opportunistic_max_age": (
                INAV_BELIEF_OPPORTUNISTIC_MAX_AGE
            ),
            "inav_belief_opportunistic_episode_budget": (
                INAV_BELIEF_OPPORTUNISTIC_EPISODE_BUDGET
            ),
            "inav_belief_opportunistic_max_move_m": (
                INAV_BELIEF_OPPORTUNISTIC_MAX_MOVE_M
            ),
            "inav_camera_hfov_deg": INAV_CAMERA_HFOV_DEG,
            "belief_reperception_actions": belief_reperception_actions,
            "belief_reperception_hits": belief_reperception_hits,
            "belief_reperception_misses": belief_reperception_misses,
            "belief_promotions": belief_promotions,
            "belief_reperception_resolutions": (
                belief_reperception_resolutions
            ),
            "belief_opportunistic_rejections": (
                belief_opportunistic_rejections
            ),
            "belief_pending_at_termination": bool(
                pending_belief_candidate is not None
            ),
            "inav_passive_multi_view": INAV_PASSIVE_MULTI_VIEW,
            "inav_passive_min_viewpoints": INAV_PASSIVE_MIN_VIEWPOINTS,
            "inav_passive_min_semantic_supports": (
                INAV_PASSIVE_MIN_SEMANTIC_SUPPORTS
            ),
            "inav_passive_max_dispersion_m": (
                INAV_PASSIVE_MAX_DISPERSION_M
            ),
            "tentative_probes": tentative_probes,
            "tentative_probe_promotions": tentative_probe_promotions,
            "tentative_probe_failures": tentative_probe_failures,
            "tentative_probe_resolutions": tentative_probe_resolutions,
            "inav_support_inspection": INAV_SUPPORT_INSPECTION,
            "inav_support_inspection_budget": (
                INAV_SUPPORT_INSPECTION_BUDGET
            ),
            "inav_support_inspection_confidence": (
                INAV_SUPPORT_INSPECTION_CONFIDENCE
            ),
            "inav_support_inspection_step_m": (
                INAV_SUPPORT_INSPECTION_STEP_M
            ),
            "inav_support_labels": list(
                support_labels_for_object(target_guess)
            ),
            "support_inspection_detector_calls": (
                support_inspection_detector_calls
            ),
            "support_inspection_candidates": support_inspection_candidates,
            "support_inspections": support_inspections,
            "support_inspection_target_hits": (
                support_inspection_target_hits
            ),
            "support_inspection_failures": support_inspection_failures,
            "support_inspection_errors": support_inspection_errors,
            "support_inspection_resolutions": (
                support_inspection_resolutions
            ),
            "inav_visual_target_descriptors": (
                INAV_VISUAL_TARGET_DESCRIPTORS
            ),
            "inav_carrier_active_verify": INAV_CARRIER_ACTIVE_VERIFY,
            "inav_carrier_session_budget": INAV_CARRIER_SESSION_BUDGET,
            "inav_carrier_action_budget": INAV_CARRIER_ACTION_BUDGET,
            "inav_carrier_approach_budget": (
                INAV_CARRIER_APPROACH_BUDGET
            ),
            "inav_carrier_rotation_budget": (
                INAV_CARRIER_ROTATION_BUDGET
            ),
            "inav_carrier_rotation_deg": INAV_CARRIER_ROTATION_DEG,
            "inav_carrier_step_m": INAV_CARRIER_STEP_M,
            "inav_carrier_blacklist_radius_m": (
                INAV_CARRIER_BLACKLIST_RADIUS_M
            ),
            "inav_carrier_min_view_angle_rad": (
                INAV_CARRIER_MIN_VIEW_ANGLE_RAD
            ),
            "carrier_detector_calls": carrier_detector_calls,
            "carrier_candidates": carrier_candidates,
            "carrier_sessions_started": carrier_sessions_started,
            "carrier_actions": carrier_actions,
            "carrier_target_hits": carrier_target_hits,
            "carrier_failures": carrier_failures,
            "carrier_rejected_xy": carrier_rejected_xy,
            "carrier_resolutions": carrier_resolutions,
            "inav_execution_seed": INAV_EXECUTION_SEED,
            "inav_render_antialiasing_mode": (
                INAV_RENDER_ANTIALIASING_MODE
            ),
            "inav_render_ticks": INAV_RENDER_TICKS,
            "inav_clip_verify": INAV_CLIP_VERIFY,
            "inav_dual_detector_fusion": INAV_DUAL_DETECTOR_FUSION,
            "inav_dual_detector_confidence": (
                INAV_DUAL_DETECTOR_CONFIDENCE
            ),
            "inav_dual_detector_overlap": INAV_DUAL_DETECTOR_OVERLAP,
            "inav_dual_detector_stop_support": (
                INAV_DUAL_DETECTOR_STOP_SUPPORT
            ),
            "dual_detector_calls": dual_detector_calls,
            "dual_detector_consensus_hits": dual_detector_consensus_hits,
            "dual_detector_unknowns": dual_detector_unknowns,
            "dual_detector_sibling_vetoes": dual_detector_sibling_vetoes,
            "inav_dual_detector_model": (
                _load_yolo_world_verifier().configured_model()
                if INAV_DUAL_DETECTOR_FUSION else None
            ),
            "inav_clip_model": clip_verifier.MODEL_ID,
            "inav_clip_revision": clip_verifier.MODEL_REVISION,
            "inav_clip_background_calibration": (
                clip_verifier.BACKGROUND_CALIBRATION
            ),
            "inav_instance_visibility_diagnostic": (
                INAV_INSTANCE_VISIBILITY_DIAGNOSTIC
            ),
            "inav_dino_model": dino_detector.configured_model_id(),
            "inav_dino_revision": dino_detector.configured_model_revision(),
            "inav_coco_specialist": INAV_COCO_SPECIALIST,
            "inav_coco_stop_gate": INAV_COCO_STOP_GATE,
            "inav_contextual_stop_consensus": (
                INAV_CONTEXTUAL_STOP_CONSENSUS
            ),
            "inav_contextual_stop_strong_margin": (
                INAV_CONTEXTUAL_STOP_STRONG_MARGIN
            ),
            "inav_contextual_stop_min_viewpoints": (
                INAV_CONTEXTUAL_STOP_MIN_VIEWPOINTS
            ),
            "inav_contextual_stop_min_area_fraction": (
                INAV_CONTEXTUAL_STOP_MIN_AREA_FRACTION
            ),
            "inav_contextual_stop_min_votes": (
                INAV_CONTEXTUAL_STOP_MIN_VOTES
            ),
            "inav_budgeted_view_scan": INAV_BUDGETED_VIEW_SCAN,
            "inav_budgeted_scan_rotation_deg": (
                INAV_BUDGETED_SCAN_ROTATION_DEG
            ),
            "inav_budgeted_scan_rotations_per_site": (
                INAV_BUDGETED_SCAN_ROTATIONS_PER_SITE
            ),
            "inav_budgeted_scan_max_sites": (
                INAV_BUDGETED_SCAN_MAX_SITES
            ),
            "inav_budgeted_scan_max_sites_per_room": (
                INAV_BUDGETED_SCAN_MAX_SITES_PER_ROOM
            ),
            "inav_budgeted_scan_min_anchor_distance_m": (
                INAV_BUDGETED_SCAN_MIN_ANCHOR_DISTANCE_M
            ),
            "inav_budgeted_scan_min_step": INAV_BUDGETED_SCAN_MIN_STEP,
            "inav_budgeted_scan_min_budget_fraction": (
                INAV_BUDGETED_SCAN_MIN_BUDGET_FRACTION
            ),
            "inav_budgeted_scan_likely_room_only": (
                INAV_BUDGETED_SCAN_LIKELY_ROOM_ONLY
            ),
            "inav_budgeted_scan_yolo_world": (
                INAV_BUDGETED_SCAN_YOLO_WORLD
            ),
            "inav_budgeted_scan_yolo_confidence": (
                INAV_BUDGETED_SCAN_YOLO_CONFIDENCE
            ),
            "inav_budgeted_scan_approach_quality": (
                INAV_BUDGETED_SCAN_APPROACH_QUALITY
            ),
            "inav_budgeted_scan_yolo_model": (
                _load_yolo_world_verifier().configured_model()
                if (
                    INAV_BUDGETED_SCAN_YOLO_WORLD
                    or INAV_DUAL_DETECTOR_FUSION
                    or INAV_SUPPORT_INSPECTION
                ) else None
            ),
            "budgeted_scan_sites": budgeted_scan_sites,
            "budgeted_scan_actions": budgeted_scan_actions,
            "budgeted_scan_observation_frames": (
                budgeted_scan_observation_frames
            ),
            "budgeted_scan_evidence_hits": budgeted_scan_evidence_hits,
            "budgeted_scan_tentative_hits": budgeted_scan_tentative_hits,
            "budgeted_scan_policy_evidence_hits": (
                budgeted_scan_policy_evidence_hits
            ),
            "budgeted_scan_terminal_stops": budgeted_scan_terminal_stops,
            "yolo_world_scan_calls": perception_counters.get(
                "yolo_world_scan_calls", 0
            ),
            "yolo_world_scan_hits": perception_counters.get(
                "yolo_world_scan_hits", 0
            ),
            "yolo_world_scan_errors": perception_counters.get(
                "yolo_world_scan_errors", 0
            ),
            "yolo_world_scan_last_error": perception_counters.get(
                "yolo_world_scan_last_error"
            ),
            "budgeted_scan_anchors": {
                room: [[float(x), float(y)] for x, y in anchors]
                for room, anchors in budgeted_scan_anchors.items()
            },
            "budgeted_scan_site_ledger": budgeted_scan_site_ledger,
            "budgeted_scan_events": budgeted_scan_events,
            "inav_coco_specialist_model": coco_detector.configured_model(),
            "coco_specialist_calls": perception_counters.get(
                "coco_specialist_calls", 0
            ),
            "coco_specialist_hits": perception_counters.get(
                "coco_specialist_hits", 0
            ),
            "coco_specialist_fallbacks": perception_counters.get(
                "coco_specialist_fallbacks", 0
            ),
            "coco_specialist_errors": perception_counters.get(
                "coco_specialist_errors", 0
            ),
            "coco_specialist_last_error": perception_counters.get(
                "coco_specialist_last_error"
            ),
            "coco_stop_gate_checks": perception_counters.get(
                "coco_stop_gate_checks", 0
            ),
            "coco_stop_gate_positives": perception_counters.get(
                "coco_stop_gate_positives", 0
            ),
            "coco_stop_gate_vetoes": perception_counters.get(
                "coco_stop_gate_vetoes", 0
            ),
            "contextual_stop_checks": perception_counters.get(
                "contextual_stop_checks", 0
            ),
            "contextual_stop_positives": perception_counters.get(
                "contextual_stop_positives", 0
            ),
            "contextual_stop_vetoes": perception_counters.get(
                "contextual_stop_vetoes", 0
            ),
            "inav_mask_refinement": INAV_MASK_REFINEMENT,
            "inav_mask_model": "MobileSAM/vit_t",
            "inav_mask_model_commit": mask_refiner.MODEL_COMMIT,
            "inav_mask_checkpoint": mask_refiner.configured_checkpoint(),
            "inav_mask_checkpoint_sha256": mask_refiner.CHECKPOINT_SHA256,
            "mask_refinement_calls": mask_refinement_calls,
            "mask_refinement_successes": mask_refinement_successes,
            "inav_clip_admit_max_rank": INAV_CLIP_ADMIT_MAX_RANK,
            "inav_clip_stop_max_rank": INAV_CLIP_STOP_MAX_RANK,
            "inav_stop_weak_semantic_min_viewpoints": (
                INAV_STOP_WEAK_SEMANTIC_MIN_VIEWPOINTS
            ),
            "inav_stop_max_weak_semantic_failures": (
                INAV_STOP_MAX_WEAK_SEMANTIC_FAILURES
            ),
            "inav_persistent_stop_confirmation": (
                INAV_PERSISTENT_STOP_CONFIRMATION
            ),
            "inav_strong_semantic_relaxation": (
                INAV_STRONG_SEMANTIC_RELAXATION
            ),
            "inav_clip_strong_margin": INAV_CLIP_STRONG_MARGIN,
            "evidence_stop_fired": evidence_stop_fired,
            "evidence_verification_attempts": evidence_verification_attempts,
            "evidence_verification_successes": evidence_verification_successes,
            "evidence_verification_failures": evidence_verification_failures,
            "evidence_clusters_suppressed": evidence_clusters_suppressed,
            "post_budget_recovery": False,
        },
        "usage_total": usage_total,
        "timestamp": now_iso(),
    }
    if open_weight_provenance:
        record["open_weight_intent_source"] = open_weight_provenance
    if robustness_meta:
        record["robustness"] = robustness_meta
    save_atomic(record, out_path)
    return record


# ---------------- main / CLI / queue mode ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=MODEL_CHOICES)
    ap.add_argument("--style", default="formal",
                    choices=["formal", "natural", "casual", "emotional", "all"])
    ap.add_argument("--step-cap", type=int, default=30)
    ap.add_argument("--objectnav", action="store_true")
    ap.add_argument(
        "--local-perception-only",
        action="store_true",
        help="Disable per-step hosted-VLM calls and use the local GroundingDINO "
             "frontier/evidence/approach policy.",
    )
    ap.add_argument(
        "--intent-predictions-dir",
        type=Path,
        default=None,
        help="Frozen open-weight intent-inference output directory. This mode "
             "loads only ground-truth-free policy_records and automatically "
             "enables --local-perception-only.",
    )
    ap.add_argument("--scene-candidates", action="store_true")
    ap.add_argument("--robustness", default=os.environ.get("ROBUSTNESS_MODE", "none"),
                    choices=ROBUSTNESS_ARG_CHOICES,
                    help="Optional robustness stress test. target_absent "
                         "runs an ObjectNav negative episode whose requested "
                         "target category is absent from the scene.")
    ap.add_argument("--robustness-seed", type=int,
                    default=int(os.environ.get("ROBUSTNESS_SEED", "0")),
                    help="Deterministic seed for robustness perturbations.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--queue-dir", type=str, default=None)
    ap.add_argument("--worker-id", type=str, default=None)
    args = ap.parse_args()
    args.robustness = normalize_robustness_mode(args.robustness)

    if args.intent_predictions_dir is not None:
        args.intent_predictions_dir = args.intent_predictions_dir.resolve()
        # A cached local plan must never silently re-enable hosted calls.
        args.local_perception_only = True

    if args.local_perception_only and not args.model:
        # Internal key is never called or reported in local mode; keeping one
        # lets the shared queue/record code retain a single interface.
        args.model = next(iter(MODEL_CATALOG))

    if not args.queue_dir and not args.model:
        ap.error("--model is required unless --queue-dir is set")
    if args.local_perception_only and not (
        args.objectnav or args.intent_predictions_dir is not None
    ):
        ap.error(
            "--local-perception-only requires --objectnav or "
            "--intent-predictions-dir"
        )
    if args.intent_predictions_dir is not None and (
        args.objectnav or args.scene_candidates or args.robustness != "none"
    ):
        ap.error(
            "--intent-predictions-dir cannot be combined with --objectnav, "
            "--scene-candidates, or robustness modes"
        )
    if args.objectnav and args.scene_candidates:
        ap.error("--objectnav and --scene-candidates are mutually exclusive")
    if args.robustness == "target_absent" and (args.scene_candidates or args.objectnav):
        ap.error("--robustness target_absent is an explicit ObjectNav negative task; "
                 "do not combine it with --objectnav or --scene-candidates")

    # Strip CLI args before Isaac Sim import (Kit reads sys.argv).
    import sys as _sys
    _sys.argv = _sys.argv[:1]
    from simulator.iss_env import IsaacSimEnv

    items = load_items()
    items_by_sel = {it["selection_id"]: it for it in items}
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        items = [it for it in items if it["selection_id"] in wanted]
    if args.limit:
        items = items[:args.limit]
    episodes = load_episodes()
    styles = STYLES if args.style == "all" else (args.style,)

    env = IsaacSimEnv(headless=args.headless)
    wm_cache: dict[str, WalkableMap] = {}
    loaded_scenes: set[str] = set()

    def _queue_run():
        if not args.worker_id:
            print("[engine] --queue-dir requires --worker-id", file=sys.stderr); return
        qdir = Path(args.queue_dir)
        pending = qdir / "pending"; inflight = qdir / "inflight"; done = qdir / "done"
        for d in (pending, inflight, done): d.mkdir(parents=True, exist_ok=True)
        wid = args.worker_id; n_done_here = 0
        while True:
            claimed = None
            try: names = sorted(p.name for p in pending.iterdir())
            except FileNotFoundError: break
            if not names: break
            for name in names:
                src = pending / name; dst = inflight / f"{name}.{wid}"
                try:
                    src.rename(dst); claimed = (name, dst); break
                except (OSError, FileNotFoundError): continue
            if claimed is None: break
            name, dst = claimed
            # Item name formats supported (oldest → newest):
            #   "SEL__style"            → 1 ep, model from --model
            #   "SEL__style__model"     → 1 ep, explicit model
            #   "SEL"                   → 4 styles bundled, model from --model
            #   "SEL__model"            → 4 styles bundled, explicit model
            # The 4-style bundle keeps the SimulationApp's loaded scene
            # warm across all 4 styles for one SEL — saves the cold-start
            # USD load (≈30 min on a40_vln contention). Old per-style
            # format still works for backward compat.
            parts = name.split("__")
            ep_styles = None  # None → bundle all styles; list → just these
            if len(parts) == 1:
                sel_id = parts[0]; item_model = args.model
            elif len(parts) == 2:
                # Disambiguate "SEL__style" vs "SEL__model" by checking model catalog.
                if parts[1] in MODEL_CATALOG:
                    sel_id, item_model = parts
                else:
                    sel_id, style = parts; item_model = args.model
                    ep_styles = [style]
            elif len(parts) == 3:
                sel_id, style, item_model = parts
                if item_model not in MODEL_CATALOG:
                    print(f"[engine/queue] unknown model {item_model!r} in {name!r}", file=sys.stderr)
                    (done / name).touch(); dst.unlink(missing_ok=True); continue
                ep_styles = [style]
            else:
                print(f"[engine/queue] bad item name {name!r}", file=sys.stderr)
                dst.unlink(missing_ok=True); continue
            if not item_model:
                print(f"[engine/queue] no model for {name!r}", file=sys.stderr)
                dst.unlink(missing_ok=True); continue
            if ep_styles is None:
                ep_styles = list(STYLES)
            item = items_by_sel.get(sel_id)
            if item is None or sel_id not in episodes:
                print(f"[engine/queue] {sel_id}: missing item or episode", file=sys.stderr)
                (done / name).touch(); dst.unlink(missing_ok=True); continue
            ep = episodes[sel_id]; scene_id = ep["scene_id"]
            if (args.max_scenes is not None and scene_id not in loaded_scenes
                    and len(loaded_scenes) >= args.max_scenes):
                try: dst.rename(pending / name)
                except Exception: pass
                print(f"[engine/queue] {wid}: hit --max-scenes, exiting "
                      f"(did {n_done_here} items)")
                break
            if scene_id not in wm_cache:
                wm_cache[scene_id] = WalkableMap.load(scene_id)
            wm = wm_cache[scene_id]
            if wm is None:
                print(f"[engine/queue] {sel_id}: walkable_map missing for {scene_id}",
                      file=sys.stderr)
                (done / name).touch(); dst.unlink(missing_ok=True); continue
            usd_path = USD_ROOT / scene_id / "start_result_navigation.usd"
            # load_scene caches by scene_id — first call is the cold-start;
            # subsequent calls for the same scene_id are no-ops, so all
            # styles share one USD load.
            env.load_scene(scene_id, str(usd_path))
            loaded_scenes.add(scene_id)
            n_styles_done = 0
            tier_name = ("open_weight_intentionnav"
                         if args.intent_predictions_dir is not None else
                         "explicit_objectnav" if args.objectnav else
                         "scene_candidate" if args.scene_candidates else
                         "vlm_engine")
            if args.intent_predictions_dir is not None:
                record_model = open_weight_output_model_name(
                    args.intent_predictions_dir
                )
            elif args.local_perception_only:
                record_model = "intentionnav_local"
            else:
                record_model = robustness_output_model_name(
                    item_model, args.robustness
                )
            path_model = record_model if tier_name == "vlm_engine" else f"{tier_name}_{record_model}"
            item_failed = False
            attempted_out_paths = []
            for style in ep_styles:
                out_path = episode_path(item["scene_id"], tier_name, path_model, style, sel_id)
                attempted_out_paths.append(out_path)
                if out_path.exists() and not args.force:
                    n_styles_done += 1; continue
                try:
                    run_episode_engine(
                        env, wm, item, ep, style,
                        model_key=item_model, step_cap=args.step_cap,
                        out_path=out_path, objectnav=args.objectnav,
                        scene_candidates=args.scene_candidates,
                        robustness_mode=args.robustness,
                        robustness_seed=args.robustness_seed,
                        local_perception_only=args.local_perception_only,
                        intent_predictions_dir=args.intent_predictions_dir,
                    )
                    n_styles_done += 1
                except Exception as e:
                    import traceback
                    item_failed = True
                    print(f"[engine/queue] {sel_id}/{style}/{item_model}: EXC {e}")
                    traceback.print_exc()
            if item_failed:
                import shutil
                for p in attempted_out_paths:
                    # Drop any partial successes for this bundled item. The
                    # retry round reruns the whole SEL/model bundle, keeping
                    # aggregation from seeing mixed clean/failed records.
                    shutil.rmtree(p.parent, ignore_errors=True)
                # Keep the claim in inflight. run_distributed.sh reclaims
                # inflight items in the next retry round; no valid record.json
                # is written for the failed style/model.
                print(f"[engine/queue] {name}: leaving inflight for retry "
                      f"(completed_styles={n_styles_done}/{len(ep_styles)})",
                      file=sys.stderr)
            else:
                n_done_here += n_styles_done
                (done / name).touch(); dst.unlink(missing_ok=True)
            if n_done_here % 5 < n_styles_done:  # crossed a multiple of 5
                print(f"[engine/queue] {wid}: {n_done_here} eps done")
        print(f"[engine/queue] {wid}: drained, {n_done_here} eps processed")

    try:
        if args.queue_dir:
            _queue_run(); return
        for idx, item in enumerate(items):
            sel_id = item["selection_id"]
            if sel_id not in episodes: continue
            ep = episodes[sel_id]; scene_id = ep["scene_id"]
            if (args.max_scenes is not None and scene_id not in loaded_scenes
                    and len(loaded_scenes) >= args.max_scenes):
                print(f"[engine] recycle: hit --max-scenes")
                break
            if scene_id not in wm_cache:
                wm_cache[scene_id] = WalkableMap.load(scene_id)
            wm = wm_cache[scene_id]
            if wm is None:
                print(f"[engine] {sel_id}: walkable_map missing for {scene_id}",
                      file=sys.stderr)
                continue
            usd_path = USD_ROOT / scene_id / "start_result_navigation.usd"
            env.load_scene(scene_id, str(usd_path))
            loaded_scenes.add(scene_id)
            tier_name = ("open_weight_intentionnav"
                         if args.intent_predictions_dir is not None else
                         "explicit_objectnav" if args.objectnav else
                         "scene_candidate" if args.scene_candidates else
                         "vlm_engine")
            if args.intent_predictions_dir is not None:
                record_model = open_weight_output_model_name(
                    args.intent_predictions_dir
                )
            elif args.local_perception_only:
                record_model = "intentionnav_local"
            else:
                record_model = robustness_output_model_name(
                    args.model, args.robustness
                )
            path_model = record_model if tier_name == "vlm_engine" else f"{tier_name}_{record_model}"
            for style in styles:
                out_path = episode_path(item["scene_id"], tier_name,
                                          path_model, style, sel_id)
                if out_path.exists() and not args.force: continue
                try:
                    run_episode_engine(env, wm, item, ep, style,
                                        model_key=args.model,
                                        step_cap=args.step_cap,
                                        out_path=out_path,
                                        objectnav=args.objectnav,
                                        scene_candidates=args.scene_candidates,
                                        robustness_mode=args.robustness,
                                        robustness_seed=args.robustness_seed,
                                        local_perception_only=args.local_perception_only,
                                        intent_predictions_dir=args.intent_predictions_dir)
                except Exception as e:
                    import traceback
                    print(f"[engine] {sel_id}/{style}: EXC {e}")
                    traceback.print_exc()
            if (idx + 1) % 5 == 0:
                print(f"[engine] {idx+1}/{len(items)} processed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
