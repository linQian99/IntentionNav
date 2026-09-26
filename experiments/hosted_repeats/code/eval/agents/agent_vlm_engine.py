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
import io
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent / "simulator"))
sys.path.insert(0, str(THIS_DIR.parent))

from clients import call_vlm, MODEL_CATALOG  # noqa: E402
from common import (
    EVAL_DIR, REPO, USD_ROOT, DATASET_ROOT, STYLES, load_prompt, load_items,
    tolerant_json_parse, intent_for_style, episode_path, save_atomic, now_iso,
    scene_manifest_path,
)  # noqa: E402
from walkable_map import WalkableMap  # noqa: E402
import dino_detector  # noqa: E402
from value_map import ValueMap, STOP_RADIUS_M  # noqa: E402
# Reuse episode-start plan from agent_vlm so plan format is identical.
from agent_vlm import make_episode_plan, pil_to_jpeg_bytes  # noqa: E402


EPISODES_FILE = EVAL_DIR / "splits/episodes.jsonl"
MODEL_CHOICES = list(MODEL_CATALOG.keys())

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
VM_DINO_ADMIT_ROOMS_ONLY = os.environ.get("VM_DINO_ADMIT_ROOMS_ONLY", "1") == "1"
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
# Q2: step_cap fallback cluster recovery. When the agent runs out of step
# budget without an explicit STOP, the existing argmax-of-value-map
# fallback teleports to the highest-value visited cell — but the value
# map combines target_memory + room_prior + frontier + cone, so the
# argmax often lands at an exploration frontier rather than at the best
# detected cluster. Many V19 step_cap eps had moderate target_memory
# clusters (score 0.3-0.6) within a few meters of the visited path,
# which the value-map argmax overlooked. This recovery prefers the
# strongest cluster within VM_RECOVERY_DIST_M of any visited cell when
# its score ≥ VM_RECOVERY_SCORE_MIN, falling through to the original
# argmax otherwise.
VM_STEP_CAP_CLUSTER_RECOVERY = os.environ.get(
    "VM_STEP_CAP_CLUSTER_RECOVERY", "1") == "1"
VM_RECOVERY_SCORE_MIN = float(os.environ.get("VM_RECOVERY_SCORE_MIN", "0.3"))
VM_RECOVERY_DIST_M = float(os.environ.get("VM_RECOVERY_DIST_M", "3.0"))
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
                    intent: str, *, sel_id: str, style: str, step: int):
    """Single VLM call: see target? Returns dict with see_target/confidence/
    pixel_x_norm/pixel_y_norm/observed_objects/reason; usage; err."""
    user_text = (
        f"Intent: {intent}\n"
        f"Target object: {target_name}\n"
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
                                  pose: dict, hfov_rad: float = math.pi / 2
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
    # Fix E: robust depth — sample a small window around the click pixel
    # and take median of FINITE positive values. Single-pixel depth is
    # noisy at object edges (the bed centerpixel may sit on a depth
    # discontinuity between bed surface and floor → returns floor depth).
    # Median over a 7x7 window mostly votes for the dominant surface.
    HALF = int(os.environ.get("VM_BACKPROJECT_DEPTH_HALF", "3"))
    y0, y1 = max(0, py - HALF), min(H_d, py + HALF + 1)
    x0, x1 = max(0, px - HALF), min(W_d, px + HALF + 1)
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
    initial_pos = env.get_pose()["position"]
    phrases = [target_guess]
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
                dets = dino_detector.detect(rgb_p, phrases, threshold=dino_threshold)
            except Exception as e:
                print(f"[{source_tag}/dino] {sel_id}/{style} yaw+{offset_deg}: {e}",
                      file=sys.stderr)
                dets = []
            if depth_p is None:
                continue
            H_d, W_d = depth_p.shape[:2]
            # Filter to strict-label match (same gate as per-step DINO admit)
            tg = target_guess.lower().strip().replace('_', ' ')
            non_target_cands = {
                c.lower().strip().replace('_', ' ')
                for c in candidates if c.lower().strip() != target_guess.lower().strip()
            }
            for label, score, bbox in dets:
                lab = str(label).lower().strip().replace('_', ' ')
                if lab in non_target_cands:
                    continue
                if not (tg in lab or lab in tg):
                    continue
                cx_px = (bbox[0] + bbox[2]) / 2.0
                cy_px = bbox[1] * (1 - VM_DINO_BBOX_Y_FRAC) + bbox[3] * VM_DINO_BBOX_Y_FRAC
                px_norm = cx_px / W_d if W_d > 0 else 0.5
                py_norm = cy_px / H_d if H_d > 0 else 0.5
                xy = _backproject_pixel_to_world(px_norm, py_norm, depth_p, pose_p)
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
                _admit_cluster(target_memory, xy,
                                float(min(score, 0.9)),
                                label, step_for_admit, source_tag,
                                admit_yaw=pose_p["yaw"])
                admitted += 1
        env.look_at_yaw(initial_yaw)
    except Exception as e:
        print(f"[{source_tag}] {sel_id}/{style} EXC {e}", file=sys.stderr)
    return admitted


def _admit_cluster(target_memory: list[dict], xy: tuple[float, float],
                   score: float, label: str, step: int, source: str,
                   admit_yaw: float = 0.0,
                   cluster_radius_m: float = 1.0):
    """Merge new admit into existing cluster within radius, else append.

    XY estimation uses score-weighted EMA across observations (not score-max
    overwrite), so multi-frame admits of the same object converge to a
    stable centroid. Single-frame back-projection noise is ~0.3-0.6m;
    averaging 3-5 admits cuts that noise by 1/sqrt(N). Strong_stop fires
    on cluster XY directly, so XY stability translates to SR (cluster off
    by 0.5m → agent stops 0.5m past target → SR=0 if real-target distance
    crosses 2m boundary)."""
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
                       scene_candidates=False):
    """Engine-driven episode loop. See module docstring."""
    sel_id = item["selection_id"]
    epi_dir = out_path.parent

    # ---- Plan ----
    if objectnav:
        tier_name = "explicit_objectnav"
        tgt_cat = (episode_meta or {}).get("target_category") or item.get("target_category", "")
        tgt_room = (episode_meta or {}).get("target_room") or item.get("target_room", "")
        if not tgt_cat:
            raise ValueError(f"objectnav mode but no target_category for {sel_id}")
        intent = f"Navigate to a {tgt_cat}."
        plan = {"target_guess": tgt_cat, "candidate_objects": [tgt_cat],
                "likely_rooms": [tgt_room] if tgt_room else [],
                "strategy": f"Walk toward and STOP next to a {tgt_cat}.",
                "action_plan": []}
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
    target_guess = (plan.get("target_guess") or "").strip().lower()
    if STRICT_VLM_FAILURE and plan.get("error"):
        raise RuntimeError(f"VLM_PLAN_FAILED {sel_id}/{style}/{model_key}: "
                           f"{plan.get('error')}")
    candidates = [str(c).strip().lower()
                   for c in (plan.get("candidate_objects") or [])
                   if str(c).strip()]
    if target_guess and target_guess not in candidates:
        candidates = [target_guess] + candidates

    # ---- Episode init ----
    env.place_agent(episode_meta["start_position"],
                    episode_meta["start_rotation_quat_wxyz"])
    if hasattr(wm, "reset_explored"):
        wm.reset_explored()
    AUTO_REORIENT = os.environ.get("AUTO_REORIENT", "1") == "1"
    AUTO_REORIENT_THRESHOLD_M = float(os.environ.get("AUTO_REORIENT_THRESHOLD_M", "1.0"))
    reorient_log = None
    if AUTO_REORIENT:
        reorient_log = env.find_open_view(walkable_map=wm,
                                            threshold_m=AUTO_REORIENT_THRESHOLD_M)

    pose = env.get_pose()
    rooms_visited: list[str] = []
    r0 = wm.room_at(pose["position"][0], pose["position"][1])
    if r0:
        rooms_visited.append(r0)
    traj = [{"step": 0, "position": pose["position"], "yaw": pose["yaw"],
             "room": r0, "reorient": reorient_log}]

    # State
    target_memory: list[dict] = []
    agent_path: list[tuple[float, float]] = [(pose["position"][0], pose["position"][1])]
    seen_objects: set[str] = set()
    target: str = target_guess
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
    # Last direction_hint passed to value_map.build (used by step_cap
    # fallback to recompute the final value map consistently).
    last_direction_hint = None
    # Diagnostic: which STOP path fired this episode.
    stop_path = None
    pano_init_admitted = 0
    dino_admit_room_rejected = 0  # DINO admits dropped by VM_DINO_ADMIT_ROOMS_ONLY
    wall_pierce_rejected = 0  # admits dropped by VM_WALL_PIERCE_REJECT (D1)
    strong_stop_demoted = 0  # strong_stops suppressed by D2 confirmation gate
    standard_stop_demoted = 0  # standard_stops suppressed by Q2.1 DINO-confirm gate
    dino_stop_fired = 0  # X1: detection-driven STOP triggered (DINO + depth → distance gate)
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
        counters = {"dino_admit_room_rejected": dino_admit_room_rejected,
                    "wall_pierce_rejected": wall_pierce_rejected}
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
        if hasattr(wm, "mark_explored"):
            wm.mark_explored(pose["position"][0], pose["position"][1], radius_m=0.5)

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
                if any(m == cur_room_norm or m in cur_room_norm or cur_room_norm in m for m in matched):
                    rooms_pano_scanned.add(cur_room_full)
                    counters_mid = {
                        "dino_admit_room_rejected": dino_admit_room_rejected,
                        "wall_pierce_rejected": wall_pierce_rejected,
                    }
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
        verdict, vlm_usage, vlm_err, retried = call_see_target(
            model_key, rgb_bytes, target_guess, intent,
            sel_id=sel_id, style=style, step=step,
        )
        if vlm_usage:
            usage_calls.append({**vlm_usage, "step": step, "kind": "see_target"})
        if STRICT_VLM_FAILURE and verdict is None:
            raise RuntimeError(f"VLM_SEE_TARGET_FAILED {sel_id}/{style}/{model_key} "
                               f"step={step}: {vlm_err}")
        # Trace
        try:
            tag = f"step_{step:02d}"
            (epi_dir / f"{tag}_rgb.jpg").parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(epi_dir / f"{tag}_rgb.jpg",
                                       "JPEG", quality=70, optimize=True)
            (epi_dir / f"{tag}_response.txt").write_text(
                json.dumps(verdict, indent=2) if verdict else f"[ERROR] {vlm_err}",
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
                                                   admit_yaw=pose["yaw"])
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
                                                   admit_yaw=pose["yaw"])

        # ---- DINO admit (always runs, supplements VLM) ----
        # Query ALL plan candidates (not just target_guess) — the plan's
        # candidate_objects list catches synonyms / related objects that
        # the LLM thinks could satisfy the intent. Rare target words
        # (e.g. "menorah") often fail open-vocab detection alone, but
        # broader queries ("candle holder", "altar piece") sometimes catch
        # the same object. Threshold raised vs DINO default 0.30 to
        # suppress Kujiale false positives.
        dino_dets = []
        dino_stop_xy = None      # X1: best DINO target XY for detection-driven STOP
        dino_stop_score = 0.0    # X1: matched DINO score for diagnostic
        try:
            phrases = []
            if target_guess: phrases.append(target_guess)
            for c in candidates:
                if c and c not in phrases: phrases.append(c)
            phrases = phrases[:5]  # cap at 5 to keep DINO call cheap
            if phrases:
                dets = dino_detector.detect(rgb, phrases,
                                              threshold=VM_DINO_THRESHOLD)
                for label, score, bbox in dets:
                    dino_dets.append({"label": label, "score": float(score),
                                       "bbox": bbox})
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
                    if target_guess and VM_DINO_LABEL_STRICT:
                        tg = target_guess.lower().strip().replace('_', ' ')
                        non_target_cands = {
                            c.lower().strip().replace('_', ' ')
                            for c in candidates
                            if c.lower().strip() != target_guess.lower().strip()
                        }
                        strict_dets = []
                        for d in dets:
                            lab = str(d[0]).lower().strip().replace('_', ' ')
                            if lab in non_target_cands:
                                continue  # DINO chose a sibling candidate
                            if tg in lab or lab in tg:
                                strict_dets.append(d)
                        if strict_dets:
                            best = max(strict_dets, key=lambda d: d[1])
                    elif target_guess:
                        best = dino_detector.find_best_match(dets, target_guess)
                        if best is None:
                            best = max(dets, key=lambda d: d[1])
                    if best is not None:
                        label, score, bbox = best
                        cx_px = (bbox[0] + bbox[2]) / 2.0
                        cy_px = bbox[1] * (1 - VM_DINO_BBOX_Y_FRAC) + bbox[3] * VM_DINO_BBOX_Y_FRAC
                        if depth is not None:
                            H_d, W_d = depth.shape[:2]
                            px_norm = cx_px / W_d if W_d > 0 else 0.5
                            py_norm = cy_px / H_d if H_d > 0 else 0.5
                            xy = _backproject_pixel_to_world(px_norm, py_norm,
                                                               depth, pose)
                            if xy is not None:
                                if VM_DINO_ADMIT_ROOMS_ONLY \
                                        and not vmap.is_in_likely_room(xy[0], xy[1]):
                                    dino_admit_room_rejected += 1
                                elif VM_WALL_PIERCE_REJECT \
                                        and _is_admit_xy_on_wall(wm, xy):
                                    wall_pierce_rejected += 1
                                else:
                                    _admit_cluster(target_memory, xy,
                                                    float(min(score, 0.9)),
                                                    label, step, "dino",
                                                    admit_yaw=pose["yaw"])
                                    # X1: capture this DINO admit as a STOP
                                    # candidate if it's the highest-score one
                                    # this step, regardless of room/wall gates
                                    # (those still gate the cluster admit
                                    # above; STOP gate also re-checks
                                    # distance + walkability below).
                                    if float(score) > dino_stop_score:
                                        dino_stop_xy = (float(xy[0]), float(xy[1]))
                                        dino_stop_score = float(score)
        except Exception as e:
            print(f"[dino] {sel_id}/{style} step {step}: {e}", file=sys.stderr)

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
                tg = target_guess.lower().strip().replace('_', ' ')
                non_target_cands = {
                    c.lower().strip().replace('_', ' ')
                    for c in candidates
                    if c.lower().strip() != target_guess.lower().strip()
                } if VM_DINO_LABEL_STRICT else set()
                for d in dino_dets:
                    lab = str(d.get("label", "")).lower().strip().replace('_', ' ')
                    if lab in non_target_cands:
                        continue
                    if tg in lab or lab in tg:
                        dino_saw_target = True; break
            if not dino_saw_target:
                standard_stop = False
                standard_stop_demoted += 1

        if dino_stop or strong_stop or standard_stop:
            if dino_stop:
                stop_path = "dino_stop"
                dino_stop_fired += 1
            elif strong_stop:
                stop_path = "strong_stop"
            else:
                stop_path = "standard_stop"
            stop_reason = stop_path
            # X1: face the DINO-anchored target world XY before final approach.
            # Without this, agent stops with whatever yaw the nav module left
            # it at — often off-axis from the actual target → GSR=0.
            if dino_stop and dino_stop_anchor_xy is not None:
                try:
                    tx, ty = dino_stop_anchor_xy
                    target_yaw = math.atan2(ty - pose["position"][1],
                                             tx - pose["position"][0])
                    env.look_at_yaw(target_yaw)
                    pose = env.get_pose()
                except Exception:
                    pass
            # Engine-aided final approach
            creep_d = 0.0
            try:
                creep_d = env.creep_forward(wm, max_creep_m=1.0)
            except Exception:
                pass
            if creep_d > 0:
                pose = env.get_pose()
                agent_path.append((pose["position"][0], pose["position"][1]))
                final_rgb = env.render_rgb()
            else:
                pose = env.get_pose()
            final_target = target_guess
            traj.append({"step": step, "action": "STOP",
                         "target": final_target,
                         "creep_distance_m": creep_d,
                         "see_target_verdict": verdict,
                         "engine_stop_meta": stop_meta,
                         "stop_path": stop_path,
                         "dino": dino_dets,
                         "position": pose["position"], "yaw": pose["yaw"]})
            break

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
        last_direction_hint = direction_hint
        (nx, ny), nav_meta = vmap.next_waypoint(
            (pose["position"][0], pose["position"][1]),
            target_memory, agent_path,
            direction_hint=direction_hint,
        )
        if nav_meta.get("fallback_reason") == "argmax = current cell":
            # Already at the value-map peak but VLM didn't see target →
            # take a small random walk to break out (avoid infinite stuck).
            nx = pose["position"][0] + 0.5 * math.cos(pose["yaw"])
            ny = pose["position"][1] + 0.5 * math.sin(pose["yaw"])
        env.teleport_to((nx, ny))
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
                     "vlm_admitted": vlm_admitted,
                     "dino": dino_dets,
                     "usage": vlm_usage,
                     **({"retried": True} if retried else {})})

    # ---- Episode end ----
    cluster_recovery_fired = False
    # Q2: cluster recovery — when step budget runs out, prefer the
    # strongest target_memory cluster within reach over the value-map
    # argmax, since the value map mixes in room_prior/frontier/cone and
    # often picks an exploration cell instead of a real detection.
    if stop_reason == "step_cap" and VM_STEP_CAP_CLUSTER_RECOVERY \
            and target_memory and agent_path:
        try:
            best_cluster = None
            best_visited = None
            best_pair_score = -1.0
            for c in target_memory:
                cs = float(c.get("score", 0))
                if cs < VM_RECOVERY_SCORE_MIN:
                    continue
                cx, cy = c["xy"][0], c["xy"][1]
                # Closest visited cell to this cluster
                d_min = float("inf")
                pv = None
                for (px, py) in agent_path:
                    d = math.hypot(px - cx, py - cy)
                    if d < d_min:
                        d_min = d
                        pv = (px, py)
                if d_min > VM_RECOVERY_DIST_M or pv is None:
                    continue
                # Rank by score, tiebreak by closeness
                pair_score = cs - 0.05 * d_min
                if pair_score > best_pair_score:
                    best_pair_score = pair_score
                    best_cluster = c
                    best_visited = pv
            if best_cluster is not None:
                env.teleport_to(best_visited,
                                face_direction_xy=best_cluster["xy"])
                try:
                    creep_d = env.creep_forward(wm, max_creep_m=1.0)
                except Exception:
                    creep_d = 0.0
                pose = env.get_pose()
                agent_path.append((pose["position"][0], pose["position"][1]))
                final_target = target_guess
                stop_reason = "step_cap_cluster_recovery"
                cluster_recovery_fired = True
                traj.append({"step": step_cap + 1, "action": "FALLBACK_CLUSTER_RECOVERY",
                              "target": final_target,
                              "cluster_xy": [round(best_cluster["xy"][0], 3),
                                              round(best_cluster["xy"][1], 3)],
                              "cluster_score": round(float(best_cluster["score"]), 3),
                              "visited_xy": [round(best_visited[0], 3),
                                              round(best_visited[1], 3)],
                              "creep_distance_m": creep_d,
                              "position": pose["position"], "yaw": pose["yaw"]})
                final_rgb = env.render_rgb()
        except Exception as e:
            print(f"[cluster_recovery] {sel_id}/{style}: {e}", file=sys.stderr)

    # v2 step_cap fallback: agent ran out of budget without explicit STOP.
    # Recompute final value map (target_memory + room_prior + frontier +
    # cone - visited), mask to ONLY visited cells, teleport to argmax.
    # Reached only if cluster recovery above didn't fire (no qualifying
    # cluster within reach). Combined value map can still find useful
    # final-pose hints from room_prior + cone when target_memory is empty.
    if stop_reason == "step_cap":
        try:
            final_v = vmap.build(target_memory, agent_path,
                                  direction_hint=last_direction_hint)
            visited_mask = np.zeros_like(final_v, dtype=bool)
            for (px, py) in agent_path:
                yi, xi = wm.world_to_cell(px, py)
                if 0 <= yi < final_v.shape[0] and 0 <= xi < final_v.shape[1]:
                    visited_mask[yi, xi] = True
            final_v[~visited_mask] = -1e9
            flat_idx = int(np.argmax(final_v))
            best_v = float(final_v.flatten()[flat_idx])
            if best_v > -1e8:
                best_yi = flat_idx // final_v.shape[1]
                best_xi = flat_idx % final_v.shape[1]
                best_xy = wm.cell_to_world(best_yi, best_xi)
                env.teleport_to(best_xy)
                try:
                    creep_d = env.creep_forward(wm, max_creep_m=1.0)
                except Exception:
                    creep_d = 0.0
                pose = env.get_pose()
                agent_path.append((pose["position"][0], pose["position"][1]))
                final_target = target_guess
                stop_reason = "step_cap_argmax_visited"
                traj.append({"step": step_cap + 1, "action": "FALLBACK_ARGMAX",
                              "target": final_target,
                              "best_xy": [round(best_xy[0], 3), round(best_xy[1], 3)],
                              "best_value": round(best_v, 3),
                              "creep_distance_m": creep_d,
                              "position": pose["position"], "yaw": pose["yaw"]})
                final_rgb = env.render_rgb()
        except Exception as e:
            print(f"[fallback_argmax] {sel_id}/{style}: {e}", file=sys.stderr)

    if final_rgb is None:
        final_rgb = env.render_rgb()
    final_frame_path = epi_dir / "final.png"
    env.save_frame(final_rgb, final_frame_path)

    usage_total = {
        "n_calls": len(usage_calls),
        "input_tokens_total": sum(u.get("input_tokens", 0) for u in usage_calls),
        "output_tokens_total": sum(u.get("output_tokens", 0) for u in usage_calls),
        "latency_s_total": round(sum(u.get("latency_s", 0.0) for u in usage_calls), 4),
    }
    record = {
        "selection_id": sel_id,
        "tier": tier_name,
        "model": model_key,
        "style": style,
        "scene_id": item["scene_id"],
        "target_category": item["target_category"],
        "intent": intent,
        "photo": item.get("photo"),
        "final_frame": str(final_frame_path.relative_to(REPO)),
        "prediction": {"target": final_target},
        "plan": {**plan,
                 "plan_model": MODEL_CATALOG[model_key]["model"],
                 "plan_usage": plan_usage},
        "trajectory": traj,
        "rooms_visited": rooms_visited,
        "seen_objects": sorted(seen_objects),
        "target_memory": target_memory,
        "stop_reason": stop_reason,
        "step_cap": step_cap,
        "episode_meta": episode_meta,
        "model_meta": {
            "provider": MODEL_CATALOG[model_key]["provider"],
            "model": MODEL_CATALOG[model_key]["model"],
            "temperature": 0.0,
        },
        "ablation_flags": {
            "fallback_retry": FALLBACK_RETRY_ENABLED,
            "engine_picker": True,   # always-on for this tier
            "objectnav": objectnav,
            "scene_candidates": scene_candidates,
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
            # Q2 step_cap cluster recovery
            "vm_step_cap_cluster_recovery": VM_STEP_CAP_CLUSTER_RECOVERY,
            "vm_recovery_score_min": VM_RECOVERY_SCORE_MIN,
            "vm_recovery_dist_m": VM_RECOVERY_DIST_M,
            "cluster_recovery_fired": cluster_recovery_fired,
        },
        "usage_total": usage_total,
        "timestamp": now_iso(),
    }
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
    ap.add_argument("--scene-candidates", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--only", type=str, default=None)
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--queue-dir", type=str, default=None)
    ap.add_argument("--worker-id", type=str, default=None)
    args = ap.parse_args()

    if not args.queue_dir and not args.model:
        ap.error("--model is required unless --queue-dir is set")
    if args.objectnav and args.scene_candidates:
        ap.error("--objectnav and --scene-candidates are mutually exclusive")

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
            tier_name = ("explicit_objectnav" if args.objectnav else
                         "scene_candidate" if args.scene_candidates else
                         "vlm_engine")
            path_model = item_model if tier_name == "vlm_engine" else f"{tier_name}_{item_model}"
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
            tier_name = ("explicit_objectnav" if args.objectnav else
                         "scene_candidate" if args.scene_candidates else
                         "vlm_engine")
            path_model = args.model if tier_name == "vlm_engine" else f"{tier_name}_{args.model}"
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
                                        scene_candidates=args.scene_candidates)
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
