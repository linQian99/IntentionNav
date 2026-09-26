"""Aggregate per-episode judged outputs into headline metrics.

ACTIVE tiers (vlm / vlm_engine / fbe / random / oracle / blind): produces
IM / SR / OSR / SPL / G + per-style and per-intent_mode breakdowns.
See `report/make_tables.py` for the 5-table paper layout.

Scans `eval_out/<tier>/<model>/<style>/*.json` and produces:
  - results/<tag>/metrics.json      — all aggregate numbers
  - results/<tag>/per_item.csv      — per-(tier,model,style,item) scores
  - prints a summary table to stdout

Usage:
  python aggregate/compute_metrics.py --out results/current
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent / "agents"))
from common import (
    DATASET_ROOT,
    DATASET_JSONL,
    EPISODES_FILE,
    EVAL_DIR,
    EPISODES_OUT,
    STYLES,
    load_items,
)
from visibility import DEFAULT_SCENE_SUMMARY, trajectory_visibility


INSTANCE_VISIBILITY_PROTOCOL = "renderer_semantic_instance_pixels_v1"


def validated_instance_visibility(rec: dict) -> tuple[dict | None, str | None]:
    """Validate an evaluation-only renderer diagnostic before scoring it."""
    diagnostic = (
        (rec.get("evaluation_diagnostics") or {}).get(
            "simulator_instance_visibility"
        )
    )
    if not isinstance(diagnostic, dict) or not diagnostic:
        return None, None
    expected_id = str(
        (rec.get("episode_meta") or {}).get("target_object_id") or ""
    ).replace("\\", "/").strip("/")
    observed_id = str(diagnostic.get("target_object_id") or "").replace(
        "\\", "/"
    ).strip("/")
    if diagnostic.get("protocol") != INSTANCE_VISIBILITY_PROTOCOL:
        return None, "protocol_mismatch"
    if diagnostic.get("policy_access") is not False:
        return None, "policy_access_not_false"
    if not expected_id:
        return None, "missing_expected_target_id"
    if observed_id != expected_id:
        return None, "target_id_mismatch"
    if diagnostic.get("target_prim_resolved") is not True:
        return None, str(diagnostic.get("reason") or "target_prim_unresolved")
    if diagnostic.get("target_semantic_tag_resolved") is not True:
        return None, str(
            diagnostic.get("reason") or "instance_semantic_tag_missing"
        )
    if diagnostic.get("available") is not True:
        return None, str(diagnostic.get("reason") or "diagnostic_unavailable")
    try:
        visible_pixels = int(diagnostic.get("visible_pixels", -1))
    except (TypeError, ValueError):
        return None, "invalid_visible_pixels"
    if visible_pixels < 0:
        return None, "invalid_visible_pixels"
    return {**diagnostic, "visible_pixels": visible_pixels}, None


def instance_visibility_fields(
    rec: dict,
) -> tuple[dict | None, dict[str, object]]:
    """Return exact-instance visibility plus explicit availability fields."""
    diagnostic, error = validated_instance_visibility(rec)
    fields: dict[str, object] = {
        "InstanceG_available": int(diagnostic is not None),
    }
    if diagnostic is None:
        fields["InstanceG_unavailable_reason"] = (
            error or "diagnostic_missing"
        )
    return diagnostic, fields


def _configured_protocol_path(env_name: str, default: Path) -> Path:
    path = Path(os.environ.get(env_name, str(default)))
    return path if path.is_absolute() else EVAL_DIR.parent / path


DEFAULT_CATEGORY_GOAL_SETS = _configured_protocol_path(
    "INTENTIONNAV_CATEGORY_GOAL_SETS",
    EVAL_DIR.parent / "results/dataset_quality_v2/category_goal_sets.jsonl",
)
DEFAULT_GOAL_REGION_EPISODES = _configured_protocol_path(
    "INTENTIONNAV_GOAL_REGION_EPISODES",
    EVAL_DIR.parent / "results/dataset_quality_v2/episodes_goal_regions_v2.jsonl",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest_hash(paths: list[Path], base: Path) -> tuple[str, int]:
    """Hash relative paths and bytes so source sets are content-addressed."""
    digest = hashlib.sha256()
    count = 0
    for path in sorted(paths, key=lambda value: str(value)):
        if not path.is_file():
            raise FileNotFoundError(f"required protocol source missing: {path}")
        relative = str(path.relative_to(base))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
        count += 1
    return digest.hexdigest(), count


def validate_split_alignment(
    dataset_ids: set[str],
    episodes_path: Path,
) -> None:
    episode_ids = {
        json.loads(line)["selection_id"]
        for line in episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    if episode_ids != dataset_ids:
        missing_episodes = sorted(dataset_ids - episode_ids)
        extra_episodes = sorted(episode_ids - dataset_ids)
        raise ValueError(
            "dataset/episode split mismatch; set both "
            "INTENTIONNAV_DATASET_JSONL and INTENTIONNAV_EPISODES_JSONL "
            "to a matched pair. "
            f"missing episodes={missing_episodes[:5]} (n={len(missing_episodes)}), "
            f"extra episodes={extra_episodes[:5]} (n={len(extra_episodes)})"
        )


# ---- Category normalization / synonym match ----
def _load_synonyms() -> dict[str, set[str]]:
    """Load category synonym dict for intent-match scoring."""
    try:
        import yaml
    except ImportError:
        return {}
    path = EVAL_DIR / "vocab/category_synonyms.yaml"
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = {}
    for cat, aliases in raw.items():
        norm_cat = _norm(cat)
        s = {norm_cat}
        for a in (aliases or []):
            s.add(_norm(a))
        out[norm_cat] = s
    return out


def _norm(s: str) -> str:
    """Normalize string for IM matching: lower + snake + strip articles."""
    s = (s or "").lower().strip()
    s = re.sub(r"[\s_\-]+", "_", s)
    s = re.sub(r"^(a_|an_|the_)", "", s)
    return s


def intent_match(pred: str, target: str, synonyms: dict[str, set[str]]) -> bool:
    """IM: did LLM's plan target_guess match the dataset target_category?

    Measures intent comprehension separately from navigation success.
    Uses synonym dict for semantic equivalence (bed↔bedframe, basin↔sink).
    """
    p, t = _norm(pred), _norm(target)
    if not p or not t:
        return False
    if p == t:
        return True
    if t in synonyms and p in synonyms[t]:
        return True
    if p in synonyms and t in synonyms[p]:
        return True
    return False


# ---- Episode loading ----
def iter_episode_files(tier_dir: Path):
    for p in sorted(tier_dir.rglob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            yield p, json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] skip unreadable {p}: {e}", file=sys.stderr)


def protocol_trajectory(rec: dict) -> tuple[list[dict], int]:
    """Return actions that are valid under the fixed navigation protocol.

    The protocol is terminal and budgeted: ``STOP`` ends the episode, actions
    after ``STOP`` are ignored, and an action whose step exceeds ``step_cap``
    is invalid. Historical post-budget recovery actions are also excluded.

    Returns ``(valid_trajectory, n_invalid_actions)``.  Keeping this filtering
    in the aggregator lets old logs be recomputed under the corrected protocol
    without silently crediting post-budget teleports.
    """
    raw = rec.get("trajectory") or []
    try:
        step_cap = int(rec.get("step_cap"))
    except (TypeError, ValueError):
        step_cap = None

    valid: list[dict] = []
    invalid = 0
    terminal = False
    for event_index, event in enumerate(raw):
        action = str(event.get("action", "")).strip().upper()
        try:
            step = int(event.get("step", 0))
        except (TypeError, ValueError):
            step = 0

        initial_pose = event_index == 0 and step == 0 and action in {
            "", "START",
        }
        valid_action = action in {
            "MOVE", "ROTATE_SCAN", "ROTATE_TARGET_RECENTER", "STOP",
            "FALLBACK_WAYPOINT",
        }
        if terminal or action in {
            "FALLBACK_ARGMAX",
            "FALLBACK_CLUSTER_RECOVERY",
        }:
            invalid += 1
            continue
        if not initial_pose and not valid_action:
            # An actionless/unknown event after step 0 is not a policy action.
            # Excluding it prevents hidden teleports or free observations from
            # entering OSR, path length, or terminal-distance calculations.
            invalid += 1
            continue
        if step_cap is not None and step > step_cap:
            invalid += 1
            continue

        valid.append(event)
        if action == "STOP":
            terminal = True
    return valid, invalid


def nav_metrics(rec: dict, radius_m: float = 2.0) -> dict:
    """Compute strict navigation metrics from an active-tier record.

    ``SR`` requires an explicit, in-budget ``STOP`` at the goal. ``EndSR`` is
    the weaker endpoint-only diagnostic retained for comparison with legacy
    results. ``OSR`` records whether any valid trajectory pose entered the
    goal radius. Returns ``{}`` if geometry or trajectory data are missing.

    Note: σ_T (LLM-judge target match) was removed from SR. Visual
    confirmation of "agent stopped facing target" is now handled by G
    (geometric occlusion-aware projection of target bbox onto final-frame
    camera) — deterministic, no LLM call, doesn't truncate.
    """
    import math
    traj, invalid_actions = protocol_trajectory(rec)
    meta = rec.get("episode_meta") or {}
    target_xy = meta.get("target_position")
    if not traj:
        return {}
    if target_xy is None or len(target_xy) < 2:
        nav_metrics._dropped = getattr(nav_metrics, "_dropped", 0) + 1
        return {}
    target = (float(target_xy[0]), float(target_xy[1]))

    def d(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    positions = []
    position_events = []
    for t in traj:
        p = t.get("position")
        if p and len(p) >= 2:
            position = (float(p[0]), float(p[1]))
            positions.append(position)
            position_events.append((t, position))
    if not positions:
        return {}

    d_final = d(positions[-1], target)
    d_min = min(d(p, target) for p in positions)
    path_len = sum(d(positions[i], positions[i - 1]) for i in range(1, len(positions)))
    geo_len = float(meta.get("geodesic_to_target") or 0.0)

    stop_event = next(
        (event for event in reversed(traj)
         if str(event.get("action", "")).strip().upper() == "STOP"),
        None,
    )
    stopped = int(bool(stop_event and stop_event.get("position")))
    end_sr_hit = int(d_final <= radius_m)
    SR_hit = int(stopped and end_sr_hit)
    OSR_hit = int(d_min <= radius_m)
    terminal_false_stop = int(stopped and not end_sr_hit)
    reached_but_failed = int(OSR_hit and not SR_hit)
    first_entry_event = next(
        (event for event, position in position_events
         if d(position, target) <= radius_m),
        None,
    )
    geodesic_definition = str(meta.get("geodesic_definition") or "legacy")
    center_spl_compatible = geodesic_definition in {
        "",
        "legacy",
        "target_center",
        "snapped_target_center",
    }

    numeric_steps = []
    for event in traj:
        try:
            numeric_steps.append(int(event.get("step", 0)))
        except (TypeError, ValueError):
            pass

    result = {
        "d_final": round(d_final, 3),
        "d_min": round(d_min, 3),
        "path_length": round(path_len, 3),
        "geodesic_length": round(geo_len, 3),
        "stopped": stopped,
        "stop_step": int(stop_event.get("step", 0)) if stop_event else None,
        "SR_hit": SR_hit,
        "EndSR_hit": end_sr_hit,
        "OSR_hit": OSR_hit,
        # Perfect first-entry stopping on this exact trajectory.
        "FirstEntryOracleSR_hit": OSR_hit,
        "terminal_false_stop": terminal_false_stop,
        "reached_but_failed": reached_but_failed,
        "first_entry_step": (
            int(first_entry_event.get("step", 0))
            if first_entry_event is not None else None
        ),
        "steps": max(numeric_steps, default=0),
        "invalid_actions": invalid_actions,
        "budget_violation": int(invalid_actions > 0),
        # GSR_hit = SR_hit AND G_hit, filled in aggregate() (needs visibility)
    }
    if center_spl_compatible:
        spl = (SR_hit * geo_len / max(path_len, geo_len)) if geo_len > 0 else 0.0
        end_spl = (
            end_sr_hit * geo_len / max(path_len, geo_len)
            if geo_len > 0 else 0.0
        )
        result["SPL"] = round(spl, 4)
        result["EndSPL"] = round(end_spl, 4)
    return result


def _distance_to_xy_bbox(position: tuple[float, float], bbox: dict) -> float:
    bbox_min = bbox.get("min") or []
    bbox_max = bbox.get("max") or []
    if len(bbox_min) < 2 or len(bbox_max) < 2:
        return float("inf")
    dx = max(float(bbox_min[0]) - position[0], 0.0,
             position[0] - float(bbox_max[0]))
    dy = max(float(bbox_min[1]) - position[1], 0.0,
             position[1] - float(bbox_max[1]))
    return math.hypot(dx, dy)


def goal_region_metrics(
    rec: dict,
    instances: list[dict],
    radius_m: float,
    prefix: str,
    geodesic_m: float | None = None,
) -> dict:
    """Score a trajectory against the union of object-surface goal regions.

    Reports SR/OSR, terminal false-stop, surface distances, and—when a
    shortest-path distance to the same goal-region union is available—SPL.
    The entire family is withheld if any declared union member lacks geometry.
    """
    valid_instances = [
        instance for instance in instances
        if len((instance.get("bbox") or {}).get("min") or []) >= 2
        and len((instance.get("bbox") or {}).get("max") or []) >= 2
    ]
    # A union goal must not silently shrink when any declared instance lacks
    # geometry. Withhold the whole metric instead of scoring a favorable
    # subset. This also fails closed for a missing fixed target instance.
    if not instances or len(valid_instances) != len(instances):
        return {}
    traj, _ = protocol_trajectory(rec)
    position_events = []
    for event in traj:
        position = event.get("position") or []
        if len(position) >= 2:
            position_events.append((event, (float(position[0]), float(position[1]))))
    if not position_events:
        return {}

    def distance(position: tuple[float, float]) -> float:
        return min(
            _distance_to_xy_bbox(position, instance["bbox"])
            for instance in valid_instances
        )

    stop_event = next(
        (
            event for event, _ in reversed(position_events)
            if str(event.get("action", "")).strip().upper() == "STOP"
        ),
        None,
    )
    final_distance = distance(position_events[-1][1])
    min_distance = min(distance(position) for _, position in position_events)
    start_distance = distance(position_events[0][1])
    sr_hit = int(stop_event is not None and final_distance <= radius_m)
    path_length = sum(
        math.dist(position_events[index - 1][1], position_events[index][1])
        for index in range(1, len(position_events))
    )
    result = {
        f"{prefix}_SR_hit": sr_hit,
        f"{prefix}_OSR_hit": int(min_distance <= radius_m),
        f"{prefix}_terminal_false_stop": int(
            stop_event is not None and final_distance > radius_m
        ),
        f"{prefix}_d_final": round(final_distance, 3),
        f"{prefix}_d_min": round(min_distance, 3),
        f"{prefix}_d_start": round(start_distance, 3),
        f"{prefix}_start_inside": int(start_distance <= radius_m),
        f"{prefix}_n_instances": len(valid_instances),
    }
    if geodesic_m is not None and geodesic_m > 0:
        result[f"{prefix}_geodesic_length"] = round(float(geodesic_m), 4)
        result[f"{prefix}_SPL"] = round(
            sr_hit * float(geodesic_m) / max(path_length, float(geodesic_m)),
            4,
        )
    return result


def goal_region_geometry_reason(instances: list[dict]) -> str | None:
    """Explain why a fixed/union surface goal is unavailable."""
    if not instances:
        return "no_goal_instances"
    for instance in instances:
        bbox = instance.get("bbox") or {}
        if (
            len(bbox.get("min") or []) < 2
            or len(bbox.get("max") or []) < 2
        ):
            return "incomplete_instance_bbox"
    return None


def add_fixed_surface_visibility_metrics(
    row: dict,
    *,
    analytical_visible: int,
    instance_visible: int | None = None,
) -> None:
    """Compose visibility only with the matching fixed-instance surface SR.

    The available renderer diagnostic identifies one fixed target instance.
    It cannot score an any-instance category union, so this helper
    deliberately never creates ``AnyInstanceSurface_*GSR`` fields.
    """
    if "FixedSurface_SR_hit" not in row:
        return
    row["FixedSurface_GSR_hit"] = int(
        row["FixedSurface_SR_hit"] and int(analytical_visible)
    )
    if instance_visible is not None:
        row["FixedSurface_InstanceGSR_hit"] = int(
            row["FixedSurface_SR_hit"] and int(instance_visible)
        )


def availability_summary(rows: list[dict], prefix: str) -> dict:
    """Summarize an optional metric family without hiding its denominator."""
    requested = [
        row for row in rows if row.get(f"{prefix}_requested") == 1
    ]
    available = [
        row for row in requested if row.get(f"{prefix}_available") == 1
    ]
    reasons: dict[str, int] = defaultdict(int)
    for row in requested:
        if row.get(f"{prefix}_available") == 1:
            continue
        reason = str(
            row.get(f"{prefix}_unavailable_reason") or "unknown"
        )
        reasons[reason] += 1
    return {
        f"{prefix}_requested_n": len(requested),
        f"{prefix}_available_n": len(available),
        f"{prefix}_unavailable_reasons": dict(reasons),
    }


@lru_cache(maxsize=4)
def load_category_goal_sets(path: str) -> dict[tuple[str, str], dict]:
    source = Path(path)
    if not source.exists():
        return {}
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        (row["scene_id"], row["target_category"]): row
        for row in rows
    }


@lru_cache(maxsize=4)
def load_goal_region_episodes(path: str) -> dict[str, dict]:
    source = Path(path)
    if not source.exists():
        return {}
    return {
        row["selection_id"]: row
        for row in (
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def target_absent_metrics(rec: dict) -> dict:
    """Metrics for endpoint-object-absent ObjectNav episodes.

    There is no valid target_position, so success means the agent does not
    issue an explicit STOP on a non-existent object before the step budget ends.
    """
    traj, invalid_actions = protocol_trajectory(rec)
    stop_steps = [
        t for t in traj
        if str(t.get("action", "")).upper() == "STOP"
    ]
    false_stop = int(bool(stop_steps))
    return {
        "target_absent": 1,
        "false_stop": false_stop,
        "ABSENT_success": int(not false_stop),
        "steps": max(0, len(traj) - 1),
        "stop_reason": rec.get("stop_reason", ""),
        "invalid_actions": invalid_actions,
        "budget_violation": int(invalid_actions > 0),
    }


def collect_records(split_ids: set[str]):
    """Yield one (tier, model, style, record) per judged episode.

    New layout: results/eval_out/<scene>/<model>/<tier>_<style>_<sel>.json
    We just walk all .json files and read tier/model/style/selection_id
    from inside the record (which is authoritative).
    """
    if not EPISODES_OUT.exists():
        return
    seen_cells: dict[tuple[str, ...], Path] = {}
    for p in sorted(EPISODES_OUT.rglob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] skip unreadable {p}: {e}", file=sys.stderr)
            continue
        if rec.get("selection_id") not in split_ids:
            continue
        # judge no longer required (SR uses geometric d + G visibility, no σ_T).
        tier = rec.get("tier", "")
        model = rec.get("model", "")
        style = rec.get("style", "")
        if style not in STYLES:
            continue
        cell = result_cell_key(rec)
        previous = seen_cells.get(cell)
        if previous is not None:
            raise RuntimeError(
                "duplicate evaluation cell detected; aggregation refuses to "
                f"change denominators silently: {cell}\n"
                f"  first: {previous}\n"
                f"  duplicate: {p}"
            )
        seen_cells[cell] = p
        yield tier, model, style, rec


def result_cell_key(rec: dict) -> tuple[str, ...]:
    """Stable identity of one benchmark result cell."""
    robustness_mode = str((rec.get("robustness") or {}).get("mode") or "none")
    return (
        str(rec.get("tier") or ""),
        str(rec.get("model") or ""),
        str(rec.get("style") or ""),
        str(rec.get("selection_id") or ""),
        robustness_mode,
        goal_input_type(rec),
    )


def goal_input_type(rec: dict) -> str:
    """Return implicit/explicit goal condition, including legacy records."""
    legacy_objectnav = bool(
        (rec.get("ablation_flags") or {}).get("objectnav")
    )
    return str(
        (rec.get("evaluation_protocol") or {}).get("goal_input")
        or rec.get("goal_input")
        or ("explicit_target" if legacy_objectnav else "")
        or "implicit_intent"
    )


# ---- Bootstrap helpers ----
def bootstrap_ci(values, n_boot=1000, alpha=0.05, seed=42):
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    # Clamp index to valid range [0, n_boot-1]. With small n_boot or extreme
    # alpha, int(alpha/2 * n_boot) could be 0 (using min instead of true 2.5%)
    # which under-reports CI width — and at n_boot - 1 we don't want to read
    # past the array end.
    lo_idx = max(0, min(n_boot - 1, int(alpha / 2 * n_boot)))
    hi_idx = max(0, min(n_boot - 1, int((1 - alpha / 2) * n_boot)))
    return (means[lo_idx], means[hi_idx])


def mean(x):
    return sum(x) / len(x) if x else float("nan")




# ---- Per-intent_mode stratification (paper Table 3) ----
def stratify_by_intent_mode(rows: list[dict], items_lookup: dict) -> dict:
    """Join SR_hit + _intent_mode.mode from dataset jsonl.
    Output per (tier, model, mode) → {n, SR, SR_ci95, SPL, T}.

    Logs how many rows were dropped due to missing `_intent_mode` so a
    half-labeled dataset doesn't silently bias the per-mode stratification
    (e.g., if labeling is in progress and only some SELs have the field)."""
    by_tmm = defaultdict(list)
    dropped = 0
    for r in rows:
        if r.get("robustness_mode", "none") != "none":
            continue
        sel = r["selection_id"]
        item = items_lookup.get(sel) or {}
        mode = (item.get("_intent_mode") or {}).get("mode")
        if not mode:
            dropped += 1
            continue
        by_tmm[(r["tier"], r["model"], mode)].append(r)
    if dropped > 0:
        print(f"[metrics] per_intent_mode: dropped {dropped}/{len(rows)} rows "
              f"with no _intent_mode label (label dataset items to include them)")
    out = {}
    for (tier, model, mode), rs in sorted(by_tmm.items()):
        im_vals = [r["IM"] for r in rs if "IM" in r]
        sr_vals = [r["SR_hit"] for r in rs if "SR_hit" in r]
        spl_vals = [r["SPL"] for r in rs if "SPL" in r]
        entry = {
            "n": len(rs),
            "IM": round(mean(im_vals), 4) if im_vals else None,
        }
        if sr_vals:
            entry["SR"] = round(mean(sr_vals), 4)
            sr_ci = bootstrap_ci(sr_vals)
            entry["SR_ci95"] = [round(sr_ci[0], 4), round(sr_ci[1], 4)]
        if spl_vals:
            entry["SPL"] = round(mean(spl_vals), 4)
        out[f"{tier}/{model}/{mode}"] = entry
    return out


# ---- Main aggregation ----
def aggregate(
    out_dir: Path,
    radius_m: float = 1.0,
    require_occluders: bool = True,
):
    all_items = load_items()
    split_ids = {it["selection_id"] for it in all_items}
    validate_split_alignment(split_ids, EPISODES_FILE)
    items_lookup = {it["selection_id"]: it for it in all_items}
    nav_metrics._dropped = 0  # reset counter for this aggregate pass
    print(f"[metrics] items={len(split_ids)} radius_m={radius_m}")

    # IM (Intent Match) — synonym-aware string match between agent's
    # predicted target and dataset's target_category. Measures pure intent
    # comprehension, independent of nav execution.
    synonyms = _load_synonyms()
    category_goal_sets_path = Path(os.environ.get(
        "INTENTIONNAV_CATEGORY_GOAL_SETS",
        str(DEFAULT_CATEGORY_GOAL_SETS),
    ))
    if not category_goal_sets_path.is_absolute():
        category_goal_sets_path = EVAL_DIR.parent / category_goal_sets_path
    category_goal_sets = load_category_goal_sets(str(category_goal_sets_path.resolve()))
    goal_region_episodes_path = Path(os.environ.get(
        "INTENTIONNAV_GOAL_REGION_EPISODES",
        str(DEFAULT_GOAL_REGION_EPISODES),
    ))
    if not goal_region_episodes_path.is_absolute():
        goal_region_episodes_path = EVAL_DIR.parent / goal_region_episodes_path
    goal_region_episodes = load_goal_region_episodes(
        str(goal_region_episodes_path.resolve())
    )
    scene_summary_root = Path(os.environ.get(
        "INTENTIONNAV_SCENE_SUMMARY", str(DEFAULT_SCENE_SUMMARY)
    )).resolve()
    object_dict_paths = [
        scene_summary_root / scene_id / "object_dict.json"
        for scene_id in sorted({item["scene_id"] for item in all_items})
    ]
    object_dict_manifest_sha256, n_object_dicts = file_manifest_hash(
        object_dict_paths, scene_summary_root
    )
    record_paths = [
        path for path in EPISODES_OUT.rglob("*.json")
        if not path.name.startswith("_")
    ] if EPISODES_OUT.exists() else []
    record_manifest_sha256, n_record_files = file_manifest_hash(
        record_paths, EPISODES_OUT
    )

    # Gather per-(tier,model,item,style) rows
    rows = []  # list of dicts
    target_label_mismatches = 0
    for tier, model, style, rec in collect_records(split_ids):
        prediction = (rec.get("prediction") or {}).get("target", "")
        robustness = rec.get("robustness") or {}
        robustness_mode = str(robustness.get("mode") or "none")
        record_target_cat = rec.get("target_category", "")
        canonical_target_cat = (
            items_lookup.get(rec["selection_id"], {}).get("target_category", "")
        )
        # The fixed dataset is authoritative. This also repairs historical
        # logs from a writer regression that accidentally copied target_guess
        # into target_category. Target-absent stress tests intentionally use a
        # different requested category, so retain their record value.
        target_cat = (
            canonical_target_cat
            if robustness_mode == "none" and canonical_target_cat
            else record_target_cat
        )
        target_label_mismatch = int(
            robustness_mode == "none"
            and bool(record_target_cat)
            and _norm(record_target_cat) != _norm(target_cat)
        )
        target_label_mismatches += target_label_mismatch
        goal_input = goal_input_type(rec)
        row = {
            "tier": tier, "model": model, "style": style,
            "selection_id": rec["selection_id"],
            "scene_id": rec.get("scene_id", ""),
            "room_type": rec.get("room_type", "") or _derive_room_type(rec),
            "target_category": target_cat,
            "record_target_category": record_target_cat,
            "target_label_mismatch": target_label_mismatch,
            "prediction": prediction,
            "robustness_mode": robustness_mode,
            "goal_input": goal_input,
        }
        if robustness_mode == "target_absent":
            row.update(target_absent_metrics(rec))
        else:
            # IM is meaningful only when the category must be inferred from
            # an implicit instruction. Explicit-target baselines are nav-only
            # and must not receive a misleading perfect IM score.
            if goal_input == "implicit_intent":
                row["IM"] = int(intent_match(prediction, target_cat, synonyms))
            # Extra active-tier metrics if this record has a trajectory.
            nav = nav_metrics(rec, radius_m=radius_m)
            row.update(nav)  # adds SR_hit (geometric), OSR_hit, SPL
            if nav:
                row["FixedSurface_requested"] = 1
                if goal_input == "explicit_target":
                    row["AnyInstanceSurface_requested"] = 1
                goal_set = category_goal_sets.get((row["scene_id"], target_cat))
                if goal_set:
                    goal_region_episode = goal_region_episodes.get(
                        rec["selection_id"], {}
                    )
                    goal_regions = (
                        (rec.get("episode_meta") or {}).get("goal_regions_v2")
                        or goal_region_episode.get("goal_regions_v2")
                        or {}
                    )
                    fixed_region = goal_regions.get("fixed_instance") or {}
                    any_region = goal_regions.get("any_exact_category_instance") or {}
                    fixed_geodesic = (
                        fixed_region.get("geodesic_to_goal_region")
                        if fixed_region.get("radius_m") == radius_m else None
                    )
                    any_geodesic = (
                        any_region.get("geodesic_to_goal_region")
                        if any_region.get("radius_m") == radius_m else None
                    )
                    target_object_id = str(
                        (rec.get("episode_meta") or {}).get("target_object_id") or ""
                    )
                    exact_instances = goal_set.get("exact_category_instances") or []
                    fixed_instances = [
                        instance for instance in exact_instances
                        if instance.get("object_id") == target_object_id
                    ]
                    fixed_metrics = goal_region_metrics(
                        rec,
                        fixed_instances,
                        radius_m,
                        prefix="FixedSurface",
                        geodesic_m=fixed_geodesic,
                    )
                    row["FixedSurface_available"] = int(bool(fixed_metrics))
                    if fixed_metrics:
                        row.update(fixed_metrics)
                    else:
                        row["FixedSurface_unavailable_reason"] = (
                            goal_region_geometry_reason(fixed_instances)
                            or "surface_metric_unavailable"
                        )
                    if goal_input == "explicit_target":
                        any_metrics = goal_region_metrics(
                            rec, exact_instances, radius_m,
                            prefix="AnyInstanceSurface",
                            geodesic_m=any_geodesic,
                        )
                        row["AnyInstanceSurface_available"] = int(
                            bool(any_metrics)
                        )
                        if any_metrics:
                            row.update(any_metrics)
                        else:
                            row["AnyInstanceSurface_unavailable_reason"] = (
                                goal_region_geometry_reason(exact_instances)
                                or "surface_metric_unavailable"
                            )
                else:
                    row["FixedSurface_available"] = 0
                    row["FixedSurface_unavailable_reason"] = (
                        "category_goal_set_missing"
                    )
                    if goal_input == "explicit_target":
                        row["AnyInstanceSurface_available"] = 0
                        row["AnyInstanceSurface_unavailable_reason"] = (
                            "category_goal_set_missing"
                        )
                valid_traj, _ = protocol_trajectory(rec)
                vis_rec = {**rec, "trajectory": valid_traj}
                vis = trajectory_visibility(
                    vis_rec,
                    require_occluders=require_occluders,
                    dataset_root=DATASET_ROOT,
                    dataset_jsonl=DATASET_JSONL,
                )
                g_hit = int(vis.get("G_seen", 0)) if vis else 0
                # GSR = geometric reach AND target visible in final frame
                row["GSR_hit"] = int(row.get("SR_hit", 0) and g_hit)
                add_fixed_surface_visibility_metrics(
                    row, analytical_visible=g_hit
                )
                sim_vis, instance_fields = instance_visibility_fields(rec)
                row.update(instance_fields)
                if sim_vis is not None:
                    instance_g_hit = int(
                        int(sim_vis.get("visible_pixels", 0) or 0) > 0
                    )
                    row["InstanceG_hit"] = instance_g_hit
                    row["InstanceG_visible_pixels"] = int(
                        sim_vis.get("visible_pixels", 0) or 0
                    )
                    row["InstanceGSR_hit"] = int(
                        row.get("SR_hit", 0) and instance_g_hit
                    )
                    add_fixed_surface_visibility_metrics(
                        row,
                        analytical_visible=g_hit,
                        instance_visible=instance_g_hit,
                    )
        rows.append(row)
    print(f"[metrics] collected {len(rows)} judged records")
    if target_label_mismatches:
        print(
            f"[metrics] repaired {target_label_mismatches} record target label(s) "
            "from the canonical fixed split"
        )
    if not rows:
        print("[metrics] no records — aborting")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Per-item CSV ----
    # Build fieldnames from union of all row keys so active rows (with nav metrics)
    # don't break on csv.DictWriter.
    fieldnames = list({k for r in rows for k in r.keys()})
    # Stable ordering: put common columns first, nav columns last
    preferred = ["tier", "model", "style", "selection_id", "scene_id",
                 "room_type", "target_category", "prediction",
                 "record_target_category", "target_label_mismatch",
                 "robustness_mode", "goal_input", "IM", "stopped",
                 "SR_hit", "EndSR_hit", "GSR_hit", "InstanceGSR_hit",
                 "InstanceG_hit", "InstanceG_visible_pixels",
                 "InstanceG_available", "InstanceG_unavailable_reason",
                 "OSR_hit", "FirstEntryOracleSR_hit", "first_entry_step",
                 "terminal_false_stop", "reached_but_failed",
                 "FixedSurface_SR_hit", "FixedSurface_OSR_hit",
                 "FixedSurface_GSR_hit", "FixedSurface_InstanceGSR_hit",
                 "FixedSurface_terminal_false_stop",
                 "FixedSurface_requested", "FixedSurface_available",
                 "FixedSurface_unavailable_reason",
                 "FixedSurface_d_final", "FixedSurface_d_min",
                 "FixedSurface_d_start", "FixedSurface_start_inside",
                 "FixedSurface_geodesic_length", "FixedSurface_SPL",
                 "AnyInstanceSurface_SR_hit", "AnyInstanceSurface_OSR_hit",
                 "AnyInstanceSurface_terminal_false_stop",
                 "AnyInstanceSurface_requested",
                 "AnyInstanceSurface_available",
                 "AnyInstanceSurface_unavailable_reason",
                 "AnyInstanceSurface_d_final", "AnyInstanceSurface_d_min",
                 "AnyInstanceSurface_d_start", "AnyInstanceSurface_start_inside",
                 "AnyInstanceSurface_geodesic_length", "AnyInstanceSurface_SPL",
                 "SPL", "d_final", "d_min",
                 "path_length", "geodesic_length", "ABSENT_success",
                 "false_stop", "target_absent", "stop_reason", "stop_step",
                 "steps", "invalid_actions", "budget_violation"]
    fieldnames = [k for k in preferred if k in fieldnames] + \
                 [k for k in fieldnames if k not in preferred]
    csv_path = out_dir / "per_item.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[metrics] wrote {csv_path}")
    if nav_metrics._dropped > 0:
        print(f"[metrics] WARN: {nav_metrics._dropped} record(s) dropped from "
              f"nav metrics due to malformed/missing target_position in episode_meta")

    # ---- Headline: per (tier, model, style) means ----
    by_tms = defaultdict(list)
    for r in rows:
        by_tms[(r["tier"], r["model"], r["style"])].append(r)

    headline = {}
    for (tier, model, style), rs in sorted(by_tms.items()):
        im_vals = [r["IM"] for r in rs if "IM" in r]
        entry = {
            "n": len(rs),
            "IM": round(mean(im_vals), 4) if im_vals else None,
        }
        # Active-tier nav metrics: SR (geometric reach), GSR (reach + visible),
        # OSR, SPL — paper table columns.
        sr_vals = [r["SR_hit"] for r in rs if "SR_hit" in r]
        if sr_vals:
            entry["SR"] = round(mean(sr_vals), 4)
            sr_ci = bootstrap_ci(sr_vals)
            entry["SR_ci95"] = [round(sr_ci[0], 4), round(sr_ci[1], 4)]
            gsr_vals = [r["GSR_hit"] for r in rs if "GSR_hit" in r]
            if gsr_vals:
                entry["GSR"] = round(mean(gsr_vals), 4)
            instance_gsr_vals = [
                r["InstanceGSR_hit"] for r in rs if "InstanceGSR_hit" in r
            ]
            instance_requested = [
                r for r in rs if "InstanceG_available" in r
            ]
            if instance_requested:
                entry["InstanceG_requested_n"] = len(instance_requested)
                entry["InstanceG_available_n"] = sum(
                    r["InstanceG_available"] for r in instance_requested
                )
                reason_counts: dict[str, int] = defaultdict(int)
                for result in instance_requested:
                    reason = result.get("InstanceG_unavailable_reason")
                    if reason:
                        reason_counts[str(reason)] += 1
                entry["InstanceG_unavailable_reasons"] = dict(reason_counts)
            if instance_gsr_vals:
                entry["InstanceGSR"] = round(mean(instance_gsr_vals), 4)
                entry["InstanceGSR_n"] = len(instance_gsr_vals)
                entry["InstanceG_visible_rate"] = round(mean([
                    r["InstanceG_hit"] for r in rs if "InstanceG_hit" in r
                ]), 4)
            endsr_vals = [r["EndSR_hit"] for r in rs if "EndSR_hit" in r]
            if endsr_vals:
                entry["EndSR"] = round(mean(endsr_vals), 4)
            entry["OSR"] = round(mean([r["OSR_hit"] for r in rs if "OSR_hit" in r]), 4)
            oracle_stop_vals = [
                r["FirstEntryOracleSR_hit"]
                for r in rs if "FirstEntryOracleSR_hit" in r
            ]
            if oracle_stop_vals:
                entry["FirstEntryOracleSR"] = round(mean(oracle_stop_vals), 4)
                entry["reach_stop_gap"] = round(
                    entry["FirstEntryOracleSR"] - entry["SR"], 4
                )
            stopped_vals = [r["stopped"] for r in rs if "stopped" in r]
            if stopped_vals:
                entry["stop_rate"] = round(mean(stopped_vals), 4)
                false_stop_vals = [
                    r["terminal_false_stop"]
                    for r in rs if "terminal_false_stop" in r
                ]
                entry["terminal_false_stop_rate"] = round(
                    mean(false_stop_vals), 4
                ) if false_stop_vals else None
                n_stops = sum(stopped_vals)
                entry["stop_precision"] = (
                    round(sum(sr_vals) / n_stops, 4) if n_stops else None
                )
            spl_values = [r["SPL"] for r in rs if "SPL" in r]
            if spl_values:
                entry["SPL"] = round(mean(spl_values), 4)
                entry["SPL_n"] = len(spl_values)
            for field in (
                "FixedSurface_SR_hit",
                "FixedSurface_OSR_hit",
                "FixedSurface_GSR_hit",
                "FixedSurface_InstanceGSR_hit",
                "AnyInstanceSurface_SR_hit",
                "AnyInstanceSurface_OSR_hit",
            ):
                values = [r[field] for r in rs if field in r]
                if values:
                    entry[field.removesuffix("_hit")] = round(mean(values), 4)
                    entry[f"{field.removesuffix('_hit')}_n"] = len(values)
            for prefix in ("FixedSurface", "AnyInstanceSurface"):
                entry.update(availability_summary(rs, prefix))
            for prefix in ("FixedSurface", "AnyInstanceSurface"):
                field = f"{prefix}_terminal_false_stop"
                values = [r[field] for r in rs if field in r]
                if values:
                    entry[f"{field}_rate"] = round(mean(values), 4)
            for field in (
                "FixedSurface_start_inside",
                "AnyInstanceSurface_start_inside",
            ):
                values = [r[field] for r in rs if field in r]
                if values:
                    entry[f"{field}_rate"] = round(mean(values), 4)
            for field in ("FixedSurface_SPL", "AnyInstanceSurface_SPL"):
                values = [r[field] for r in rs if field in r]
                if values:
                    entry[field] = round(mean(values), 4)
                    entry[f"{field}_n"] = len(values)
            # Diagnostic only (not in paper tables); kept for appendix.
            entry["TL_mean_m"] = round(mean([r["path_length"] for r in rs if "path_length" in r]), 2)
            entry["steps_mean"] = round(mean([r["steps"] for r in rs if "steps" in r]), 2)
        absent_vals = [r["ABSENT_success"] for r in rs if "ABSENT_success" in r]
        if absent_vals:
            entry["ABSENT_success"] = round(mean(absent_vals), 4)
            entry["false_stop_rate"] = round(
                mean([r["false_stop"] for r in rs if "false_stop" in r]), 4
            )
            entry["steps_mean"] = round(mean([r["steps"] for r in rs if "steps" in r]), 2)
        headline[f"{tier}/{model}/{style}"] = entry

    # ---- Style sensitivity per (tier, model): SR + SPL only ----
    style_results = {}
    nav_rows = [r for r in rows if r.get("robustness_mode", "none") == "none"]
    for (tier, model), group_key in _group_keys(nav_rows, ("tier", "model")):
        rs = [r for r in nav_rows if (r["tier"], r["model"]) == group_key]
        per_item_SR = defaultdict(dict)
        per_item_SPL = defaultdict(dict)
        for r in rs:
            if "SR_hit" in r:
                per_item_SR[r["selection_id"]][r["style"]] = r["SR_hit"]
            if "SPL" in r:
                per_item_SPL[r["selection_id"]][r["style"]] = r["SPL"]

        # CSR: items where SR_hit == 1 across all 4 styles (cross-style robustness)
        csr_items = [i for i, s in per_item_SR.items() if len(s) == len(STYLES)]
        csr = None
        per_style_SR = None
        delta_SR_style = None
        per_style_SPL = None
        delta_SPL_style = None
        if csr_items:
            csr = round(sum(1 for i in csr_items if all(v == 1 for v in per_item_SR[i].values()))
                        / len(csr_items), 4)
            per_style_SR_lists = {st: [per_item_SR[i][st] for i in per_item_SR if st in per_item_SR[i]]
                                  for st in STYLES}
            per_style_SR = {st: round(mean(v), 4) if v else None
                            for st, v in per_style_SR_lists.items()}
            valid_sr = [per_style_SR[st] for st in STYLES if per_style_SR[st] is not None]
            if valid_sr:
                delta_SR_style = round(max(valid_sr) - min(valid_sr), 4)

            per_style_SPL_lists = {st: [per_item_SPL[i][st] for i in per_item_SPL if st in per_item_SPL[i]]
                                   for st in STYLES}
            per_style_SPL = {st: round(mean(v), 4) if v else None
                             for st, v in per_style_SPL_lists.items()}
            valid_spl = [per_style_SPL[st] for st in STYLES if per_style_SPL[st] is not None]
            if valid_spl:
                delta_SPL_style = round(max(valid_spl) - min(valid_spl), 4)

        style_results[f"{tier}/{model}"] = {
            "per_style_SR": per_style_SR,
            "per_style_SPL": per_style_SPL,
            "delta_SR_style": delta_SR_style,
            "delta_SPL_style": delta_SPL_style,
            "CSR_n_items": len(csr_items),
            "CSR": csr,  # cross-style navigation consistency
        }

    # ---- Per-intent_mode stratification (Table 3) ----
    per_intent_mode = stratify_by_intent_mode(rows, items_lookup)

    # ---- Write metrics.json ----
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps({
        "protocol_version": "strict_stop_2026_08",
        "success_definition": "explicit in-budget STOP within radius_m",
        "visibility_protocol": {
            "definition": (
                "synthetic analytical proxy: final-pose target-AABB projection "
                "with scene-AABB occlusion; not direct rendered-image evidence"
            ),
            "dataset_root": str(DATASET_ROOT.resolve()),
            "dataset_jsonl": str(DATASET_JSONL.resolve()),
            "require_occluders": require_occluders,
            "scene_summary_root": str(Path(os.environ.get(
                "INTENTIONNAV_SCENE_SUMMARY", str(DEFAULT_SCENE_SUMMARY)
            )).resolve()),
            "status": (
                "occlusion_aware"
                if require_occluders
                else "legacy_no_occluder_fallback_allowed"
            ),
            "renderer_instance_diagnostic": (
                "when present and valid: exact raw renderer instance-ID pixels "
                "from the post-policy final pose; unavailable to the agent"
            ),
            "renderer_instance_protocol": INSTANCE_VISIBILITY_PROTOCOL,
            "InstanceGSR_formula": (
                "strict center-based SR_hit AND exact target visible_pixels > 0; "
                "mean is conditional on validated available diagnostics"
            ),
            "FixedSurface_InstanceGSR_formula": (
                "fixed-instance XY-AABB surface SR_hit AND exact target "
                "visible_pixels > 0; mean is conditional on validated "
                "available diagnostics"
            ),
        },
        "goal_region_diagnostics": {
            "source": str(category_goal_sets_path.resolve()),
            "available": bool(category_goal_sets),
            "geodesic_source": str(goal_region_episodes_path.resolve()),
            "geodesics_available": bool(goal_region_episodes),
            "fixed_surface": "distance to fixed target XY AABB surface",
            "any_instance_surface": (
                "explicit-target only; distance to union of exact-category "
                "instance XY AABB surfaces"
            ),
            "spl": (
                "reported only when the requested radius matches a stored "
                "goal-region geodesic and the start is outside that region"
            ),
        },
        "source_manifest": {
            "dataset_jsonl": str(DATASET_JSONL.resolve()),
            "dataset_sha256": sha256_file(DATASET_JSONL.resolve()),
            "episodes_jsonl": str(EPISODES_FILE.resolve()),
            "episodes_sha256": sha256_file(EPISODES_FILE.resolve()),
            "evaluation_records_root": str(EPISODES_OUT.resolve()),
            "evaluation_record_files": n_record_files,
            "evaluation_records_manifest_sha256": record_manifest_sha256,
            "scene_summary_root": str(scene_summary_root),
            "scene_object_dict_files": n_object_dicts,
            "scene_object_dict_manifest_sha256": object_dict_manifest_sha256,
            "category_goal_sets_sha256": (
                sha256_file(category_goal_sets_path.resolve())
                if category_goal_sets_path.is_file() else None
            ),
            "goal_region_episodes_sha256": (
                sha256_file(goal_region_episodes_path.resolve())
                if goal_region_episodes_path.is_file() else None
            ),
        },
        "n_records": len(rows),
        "radius_m": radius_m,
        "headline_per_tier_model_style": headline,
        "style_sensitivity": style_results,
        "per_intent_mode": per_intent_mode,
    }, indent=2))
    print(f"[metrics] wrote {metrics_path}")

    # ---- Stdout summary ----
    print("\n===== Headline per (tier/model/style) =====")
    print(f"{'group':<50} {'n':>4} {'SR':>6} {'EndSR':>6} {'OSR':>6} "
          f"{'SPL':>6} {'ABS':>6} {'FS':>6}")
    for key, v in sorted(headline.items()):
        sr = f"{v.get('SR','-'):>6}" if "SR" in v else f"{'-':>6}"
        endsr = f"{v.get('EndSR','-'):>6}" if "EndSR" in v else f"{'-':>6}"
        osr = f"{v.get('OSR','-'):>6}" if "OSR" in v else f"{'-':>6}"
        spl = f"{v.get('SPL','-'):>6}" if "SPL" in v else f"{'-':>6}"
        abs_s = f"{v.get('ABSENT_success','-'):>6}" if "ABSENT_success" in v else f"{'-':>6}"
        fs = f"{v.get('false_stop_rate','-'):>6}" if "false_stop_rate" in v else f"{'-':>6}"
        print(f"{key:<50} {v['n']:>4} {sr} {endsr} {osr} {spl} {abs_s} {fs}")

    print("\n===== Style sensitivity per (tier/model) =====")
    for key, v in sorted(style_results.items()):
        ps_sr = v.get("per_style_SR") or {}
        ps_str = " ".join(f"{st[0]}={ps_sr[st]:.3f}" if ps_sr.get(st) is not None
                           else f"{st[0]}=?" for st in STYLES)
        print(f"{key:<35} ΔSR={v.get('delta_SR_style')}  CSR={v.get('CSR')} "
              f"(n={v.get('CSR_n_items', 0)})  {ps_str}")


def _group_keys(rows, fields):
    seen = set()
    for r in rows:
        key = tuple(r[f] for f in fields)
        if key in seen:
            continue
        seen.add(key)
        yield key, key


def _derive_room_type(rec: dict) -> str:
    meta = rec.get("episode_meta") or {}
    rt = (rec.get("room_type") or meta.get("start_room_type") or "").strip()
    if rt:
        return rt.lower()
    room = (rec.get("room") or meta.get("start_room") or "").strip()
    stripped = re.sub(r"_\d+$", "", room)
    return stripped.lower() if stripped else "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--radius-m", type=float, default=None,
                    help="If set, run a single aggregate at this radius. "
                         "Otherwise run primary (1.0m) + sensitivity (3.0m).")
    ap.add_argument(
        "--allow-missing-occluders",
        action="store_true",
        help="Allow legacy projection-only GSR when scene object summaries are "
             "missing. The output is explicitly marked as non-occlusion-aware.",
    )
    args = ap.parse_args()
    require_occluders = not args.allow_missing_occluders
    if args.radius_m is not None:
        aggregate(
            args.out,
            radius_m=args.radius_m,
            require_occluders=require_occluders,
        )
    else:
        # Habitat-ObjectNav-aligned reporting:
        #   primary 2.0m  — matches the "viewpoint-radius ≈ object-center
        #                   distance" of Habitat 0.1m-to-VIEW_POINTS
        #                   (viewpoints are pre-sampled 1-2m from object).
        #   strict  1.0m  — Anderson 2018 / Batra 2020 to-object-center.
        #   loose   3.0m  — sensitivity / context.
        aggregate(
            args.out / "primary_r2m",
            radius_m=2.0,
            require_occluders=require_occluders,
        )
        print()
        aggregate(
            args.out / "strict_r1m",
            radius_m=1.0,
            require_occluders=require_occluders,
        )
        print()
        aggregate(
            args.out / "sensitivity_r3m",
            radius_m=3.0,
            require_occluders=require_occluders,
        )


if __name__ == "__main__":
    main()
