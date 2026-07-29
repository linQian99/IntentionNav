"""T_agent: VLM-as-policy active ObjectNav agent.

Per episode:
  0. Episode-start planning call (text-only):
     intent + scene rooms → {target_guess, likely_rooms, strategy}
  Per step:
    1. Render current RGB from Isaac Sim
    2. Sample K waypoint candidates from frontier
    3. Filter out candidates within 0.5m of any cell already walked
       (visited-waypoint suppression)
    4. Overlay A/B/C/D markers on RGB
    5. Call VLM with marker image + intent + plan + last actions + rooms seen
    6. Parse action {"action": "A/B/C/D/STOP", "target"}
    7. If STOP, terminate; else teleport to chosen waypoint

Trajectory + final frame + plan + final declared target are saved for judging.

Usage (requires goodnav env + setup_conda_env.sh):
  python agents/agent_vlm.py --model gpt5_4 --style formal --step-cap 50
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent / "simulator"))
sys.path.insert(0, str(THIS_DIR.parent))    # so `import simulator.iss_env` works

from clients import call_vlm, MODEL_CATALOG  # noqa: E402
from common import (
    EVAL_DIR, REPO, USD_ROOT, DATASET_ROOT, STYLES, load_prompt, load_items,
    tolerant_json_parse, intent_for_style, episode_path, save_atomic, now_iso,
)  # noqa: E402
from walkable_map import WalkableMap  # noqa: E402
import dino_detector  # noqa: E402
import time as _time  # noqa: E402  used by call_vlm_step retry

EPISODES_FILE = EVAL_DIR / "splits/episodes.jsonl"
LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]
SUPPRESS_RADIUS_M = 0.3  # candidates within this distance of any walked cell are dropped
                          # (kept tight so end-game 0.5m-step waypoints near the target
                          # aren't suppressed — was 0.5m, collided with new min_step=0.5m)
HISTORY_KEEP = 5         # last N actions shown to VLM
MODEL_CHOICES = list(MODEL_CATALOG.keys())  # gpt5_4 / gemini_3_1_flash / qwen3_6_plus

# Step-level retry on VLM exhaustion. clients.call_vlm already does 5 retries
# with up-to-30s backoff; if it still returns (None, err, None), the step
# would historically be lost as `fallback_skip`. STATUS §8.3 flagged
# recovering 1–3 lost steps per episode. We add ONE additional retry at the
# step level after a longer pause, since transient overload bursts often
# clear on a 30–60s window. Disable with FALLBACK_RETRY=0.
FALLBACK_RETRY_ENABLED = os.environ.get("FALLBACK_RETRY", "1") == "1"
FALLBACK_RETRY_SLEEP_S = float(os.environ.get("FALLBACK_RETRY_SLEEP_S", "10"))

# Track A: angular-spread sampler. When ON, sample_frontier_waypoints is
# called with force_angular_spread_rad=2π/num_waypoints, forcing K candidates
# into K distinct quadrants instead of letting them cluster in one direction.
# Tests whether the VLM picker fails because of clustered candidates (this
# fixes it) vs because the picker primitive is fundamentally wrong (this
# won't help). §10 follow-up; details in plan file + STATUS §10.5.
ANGULAR_SPREAD_ENABLED = os.environ.get("ANGULAR_SPREAD", "0") == "1"


def call_vlm_step(model_key, system, prompt, img_bytes, *, kind: str,
                  step: int, sel_id: str, style: str,
                  temperature: float = 0.0, json_mode: bool = True):
    """Wrap clients.call_vlm with one extra step-level retry on hard failure.
    Returns (resp, err, usage, retried_bool). `retried_bool` lets the caller
    log retry events for post-hoc analysis."""
    def _one_call():
        return call_vlm(model_key, system, prompt, img_bytes,
                         temperature=temperature, json_mode=json_mode)
    resp, err, usage = _one_call()
    if resp is not None or not FALLBACK_RETRY_ENABLED:
        return resp, err, usage, False
    print(f"[vlm/retry] {sel_id}/{style} step {step} kind={kind} "
          f"first attempt exhausted ({err!r}); sleeping {FALLBACK_RETRY_SLEEP_S}s "
          "then retrying once", file=sys.stderr)
    _time.sleep(FALLBACK_RETRY_SLEEP_S)
    resp2, err2, usage2 = _one_call()
    if resp2 is not None:
        print(f"[vlm/retry] {sel_id}/{style} step {step} kind={kind} "
              "retry succeeded", file=sys.stderr)
        return resp2, None, usage2, True
    return None, err2 or err, None, True

def load_episodes() -> dict[str, dict]:
    out = {}
    with EPISODES_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            out[r["selection_id"]] = r
    return out


def pil_to_jpeg_bytes(img: Image.Image, max_dim: int = 768, quality: int = 95) -> bytes:
    w, h = img.size
    if max(w, h) > max_dim:
        s = max_dim / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


# ---------------- Episode-start planner ----------------
PLAN_SYSTEM = (
    "You are a navigation planner for an exploratory agent. Given an "
    "indirect human intent, the navigator will EXPLORE first and decide "
    "WHICH specific object best fits the intent ONLY AFTER OBSERVING "
    "the scene. Do not lock onto a single target prematurely.\n\n"
    "Output JSON only:\n"
    "  candidate_objects — 4-7 plausible objects that could satisfy this "
    "intent, ranked roughly best-first. For each, the navigator will "
    "evaluate fit during exploration. Be reasonably broad: include the "
    "obvious + nearby alternatives.\n"
    "  target_guess     — your single best initial guess; the navigator "
    "treats this as TENTATIVE and may switch to a different candidate "
    "after observing the scene. Pick the one most likely on prior alone.\n"
    "  likely_rooms     — 1-3 generic room types where these candidates "
    "tend to live (e.g. bathroom, kitchen, bedroom). NOT scene-specific.\n"
    "  strategy         — ≤25 words imperative: 'explore X room first, "
    "look for any of the candidates, prefer the one closest to user's "
    "described use-case'.\n"
    "  action_plan      — 2-4 ordered high-level phases ('leave current "
    "room → enter likely room → scan for candidates → approach the best "
    "fit and confirm'). Frame as observable phases, NOT internal reasoning."
)


# In-process cache: same SEL × same model → same plan across all 4 styles.
# Empirically observed (2026-04-28 test 4): 4 styles of SEL_171 all produced
# identical plan (target_guess=sink, likely_rooms=[bathroom_4]) — temp=0
# VLM at semantic-identical intents converges. Caching saves 75% of plan
# calls per SEL. Cache hits return usage=None so token_stats doesn't
# double-count; the real per-SEL cost is logged once on the cache miss.
_plan_cache: dict[tuple[str, str], dict] = {}


def make_episode_plan(intent: str, model_key: str,
                      sel_id: str | None = None,
                      style: str | None = None
                      ) -> tuple[dict, dict | None]:
    """Text-only plan call (intent → plan dict). Returns
    (plan_dict, usage_or_None). plan_dict always has the keys
    (target_guess / candidate_objects / likely_rooms / strategy /
    action_plan), empty if call/parse failed. Cache key is
    (sel_id, style, model_key) — each style gets its OWN plan so
    per-style ablation measures intent phrasing through the WHOLE
    pipeline (planner + agent), not just the agent loop.

    History: 2026-05-01 briefly tried (a) RGB+freemap image and
    (b) RGB+caption text variants. Both lowered SR (0.167 →
    0.089/0.064) on the 12-SEL × 4-style smoke. Reverted to text-
    only; the per-style cache fix is kept."""
    if sel_id is not None:
        cache_key = (sel_id, style or "_", model_key)
        cached = _plan_cache.get(cache_key)
        if cached is not None:
            # Mark as cached so downstream knows usage was charged elsewhere
            return {**cached, "_cached": True}, None

    user_text = (
        f"User intent: \"{intent}\"\n\n"
        'Output JSON exactly: '
        '{"target_guess":"<single object word/phrase>",'
        '"candidate_objects":["<3-5 related objects that include target_guess>"],'
        '"likely_rooms":["<generic room types e.g. bathroom, kitchen>"],'
        '"strategy":"<10-25 words imperative>",'
        '"action_plan":["step 1 ≤15 words","step 2 ≤15 words","..."]}'
    )
    out, err, usage = call_vlm(model_key, PLAN_SYSTEM, user_text,
                               image_bytes=None, json_mode=True, max_retries=3)
    plan = {"target_guess": "", "candidate_objects": [], "likely_rooms": [],
            "strategy": "", "action_plan": []}
    if out:
        parsed, _perr = tolerant_json_parse(out)
        if isinstance(parsed, dict):
            plan["target_guess"] = str(parsed.get("target_guess", "") or "").strip()
            co = parsed.get("candidate_objects") or []
            plan["candidate_objects"] = [str(x).strip() for x in co if str(x).strip()][:6]
            lr = parsed.get("likely_rooms") or []
            plan["likely_rooms"] = [str(x).strip() for x in lr if str(x).strip()][:3]
            plan["strategy"] = str(parsed.get("strategy", "") or "").strip()[:300]
            ap = parsed.get("action_plan") or []
            if isinstance(ap, list):
                plan["action_plan"] = [str(x).strip() for x in ap if str(x).strip()][:6]
    if err:
        plan["error"] = err

    if sel_id is not None and not err:
        _plan_cache[(sel_id, style or "_", model_key)] = dict(plan)

    return plan, usage


def make_replan(intent: str, scene_id: str, model_key: str,
                prev_plan: dict
                ) -> tuple[dict, dict | None]:
    """Re-plan when the agent is stuck (no progress for N steps). Bypasses
    cache (always real call). Provides previous plan that didn't pan out so
    the planner can suggest a different target_guess / room set / strategy."""
    prev_target = prev_plan.get('target_guess', '?')
    prev_likely = ', '.join(prev_plan.get('likely_rooms', []) or [])
    user_text = (
        f"User intent: \"{intent}\"\n"
        f"Previous plan suggested target='{prev_target}' in rooms=[{prev_likely}].\n"
        f"That plan failed — propose a DIFFERENT plan: pick a new target_guess "
        f"(if first guess was wrong), or new likely_rooms, and a fresh "
        f"action_plan that gets the agent there.\n"
        'Output JSON exactly: '
        '{"target_guess":"<single object word>",'
        '"candidate_objects":["<3-5 related objects>"],'
        '"likely_rooms":["<generic room types>"],'
        '"strategy":"<10-25 words imperative>",'
        '"action_plan":["step 1","step 2","..."]}'
    )
    out, err, usage = call_vlm(model_key, PLAN_SYSTEM, user_text,
                               image_bytes=None, json_mode=True, max_retries=3)
    plan = {"target_guess": "", "candidate_objects": [], "likely_rooms": [],
            "strategy": "", "action_plan": []}
    if out:
        parsed, _perr = tolerant_json_parse(out)
        if isinstance(parsed, dict):
            plan["target_guess"] = str(parsed.get("target_guess", "") or "").strip()
            co = parsed.get("candidate_objects") or []
            plan["candidate_objects"] = [str(x).strip() for x in co if str(x).strip()][:6]
            lr = parsed.get("likely_rooms") or []
            plan["likely_rooms"] = [str(x).strip() for x in lr if str(x).strip()][:3]
            plan["strategy"] = str(parsed.get("strategy", "") or "").strip()[:300]
            ap = parsed.get("action_plan") or []
            if isinstance(ap, list):
                plan["action_plan"] = [str(x).strip() for x in ap if str(x).strip()][:6]
    if err:
        plan["error"] = err
    return plan, usage


# ---------------- Prompt construction (cache-aware) ----------------
# We split per-step prompt into two pieces:
#   - SYSTEM role: agent_system.txt rules + episode-fixed context (intent, plan).
#     CONSTANT across all 30 steps within one episode → API prefix cache hits
#     after step 1 (~50% input cost discount on cached portion, ~80% latency
#     reduction on cached prefix processing for OpenAI/Anthropic; Gemini Flash
#     2.0+ implicit cache for prefixes ≥1K tokens).
#   - USER role: step-varying state (step idx, waypoints, last actions,
#     rooms visited, trajectory tail) + image. Fresh each step.
#
# Old code put agent_system.txt + intent + plan into the user role each step,
# which still trips implicit prefix cache but is less reliable across
# vendors. Splitting by role makes caching explicit and works on every
# OpenAI-compatible endpoint.
def build_episode_system(intent: str, plan: dict | None) -> str:
    """System role payload: cached across all 30 steps of an episode."""
    sys_text = load_prompt("agent_system")
    ctx_lines = [f"User intent: {intent}"]
    if plan:
        tg = plan.get("target_guess") or ""
        co = plan.get("candidate_objects") or []
        lr = plan.get("likely_rooms") or []
        st = plan.get("strategy") or ""
        ap = plan.get("action_plan") or []
        if tg or co or lr or st or ap:
            parts = []
            if tg: parts.append(f"target={tg}")
            if co: parts.append(f"look-for={','.join(co)}")
            if lr: parts.append(f"rooms={','.join(lr)}")
            if ap: parts.append("→".join(ap))
            elif st: parts.append(st)
            ctx_lines.append("Plan: " + "; ".join(parts))
    return sys_text + "\n\n" + "\n".join(ctx_lines)


def make_step_position_caption(wm, agent_xy: tuple[float, float],
                               agent_yaw: float | None) -> str:
    """Compact per-step text describing agent's pose on the freemap +
    exploration progress. Pure geometry: no room labels (room_region.json
    is unreliable). Goes into the per-step user prompt so the VLM has a
    persistent sense of 'where am I, where am I facing, how much have I
    seen so far' without us shipping the freemap as an image."""
    parts = []
    if agent_yaw is not None:
        deg = (math.degrees(agent_yaw) + 360) % 360
        labels = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        idx = int((deg + 22.5) // 45) % 8
        parts.append(f"facing {labels[idx]} (yaw {deg:.0f}°)")
    if wm is not None and len(wm.x_coords) > 1 and len(wm.y_coords) > 1:
        x_min, x_max = float(wm.x_coords.min()), float(wm.x_coords.max())
        y_min, y_max = float(wm.y_coords.min()), float(wm.y_coords.max())
        x_extent = abs(x_max - x_min)
        y_extent = abs(y_max - y_min)
        nx = (agent_xy[0] - x_min) / x_extent if x_extent > 0 else 0.5
        ny = (agent_xy[1] - y_min) / y_extent if y_extent > 0 else 0.5
        ew = "west" if nx < 0.33 else ("east" if nx > 0.67 else "center")
        ns = "south" if ny < 0.33 else ("north" if ny > 0.67 else "center")
        if ew == ns:
            zone = ew
        elif ew == "center":
            zone = ns
        elif ns == "center":
            zone = ew
        else:
            zone = f"{ns}{ew}"
        parts.append(f"in {zone} zone of the scene")
        if hasattr(wm, "explored_count"):
            walkable_total = int((wm.grid > 0).sum())
            if walkable_total > 0:
                explored_total = int(((wm.grid > 0) & (wm.explored_count > 0)).sum())
                pct = explored_total / walkable_total * 100
                parts.append(f"{pct:.0f}% of scene explored so far")
    return "; ".join(parts) + "." if parts else ""


def build_step_prompt(step: int, step_cap: int,
                      n_waypoints: int,
                      last_actions: list[str] | None = None,
                      waypoint_bearings: list[float] | None = None,
                      waypoint_distances: list[float] | None = None,
                      step_history: list[dict] | None = None,
                      seen_objects: set[str] | None = None,
                      room_mismatch_hint: str | None = None,
                      dino_dets: list[dict] | None = None,
                      memory_hint: str | None = None,
                      focus_history: list[str] | None = None,
                      position_caption: str | None = None) -> str:
    """User role payload: only step-varying state. NO agent_system content
    here (that's in the system role for prefix caching)."""
    if waypoint_bearings and n_waypoints:
        wp_lines = []
        for i in range(n_waypoints):
            br = _bearing_label(waypoint_bearings[i]) if i < len(waypoint_bearings) else "?"
            if waypoint_distances and i < len(waypoint_distances):
                wp_lines.append(f"  {LABELS[i]} — {br}, {waypoint_distances[i]:.1f}m")
            else:
                wp_lines.append(f"  {LABELS[i]} — {br}")
        wp_block = "Waypoints:\n" + "\n".join(wp_lines) + "\n"
    else:
        wp_block = (f"Waypoints: "
                    f"{', '.join(LABELS[:n_waypoints]) if n_waypoints else '(none)'}\n")
    state = (
        f"Step {step}/{step_cap} ({step_cap - step} remaining)\n"
        f"{wp_block}"
    )
    if position_caption:
        state += f"You are: {position_caption}\n"
    if last_actions:
        state += f"Last actions: {' → '.join(last_actions[-HISTORY_KEEP:])}\n"
    if step_history:
        tail = step_history[-5:] if len(step_history) > 5 else step_history
        compact = ", ".join(f"{h['step']}({h.get('via','?')})"
                            for h in tail)
        state += f"Trajectory (BACK rewinds): {compact}\n"
    if seen_objects:
        # Sorted + capped so prompt stays bounded across long episodes.
        s = sorted(seen_objects)[:24]
        state += f"Seen so far: {', '.join(s)}\n"
    if room_mismatch_hint:
        state += f"Room hint: {room_mismatch_hint}\n"
    if dino_dets:
        # External detector evidence. VLM reads this together with the RGB
        # to decide STOP — depth ≤ ~1m + matches target = STOP candidate.
        lines = []
        for d in dino_dets[:5]:
            depth_s = f"{d['depth_m']:.2f}m" if d.get('depth_m') is not None else "?"
            lines.append(f"  '{d['label']}' at {d.get('grid','?')}, conf={d['score']:.2f}, depth={depth_s}")
        state += "Detector found:\n" + "\n".join(lines) + "\n"
    if memory_hint:
        state += f"Target memory: {memory_hint}\n"
    if focus_history:
        # Show last few candidate-focus picks so VLM sees its own
        # evolving belief and can refine or commit accordingly.
        recent = list(focus_history)[-6:]
        state += f"Focus track: {' → '.join(recent)}\n"
    return state


def _parse_marker(d) -> dict | None:
    """Validate a marker dict {x: float, y: float, [label: str]}. Returns
    None if missing / malformed / out of [0, 1] range. x = horizontal
    pixel ratio (0=left, 1=right), y = vertical (0=top, 1=bottom)."""
    if not isinstance(d, dict):
        return None
    try:
        x = float(d.get("x"))
        y = float(d.get("y"))
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    out = {"x": x, "y": y}
    lab = d.get("label")
    if isinstance(lab, str) and lab.strip():
        out["label"] = lab.strip().lower()
    return out


def parse_action(resp_text: str, n_waypoints: int
                 ) -> tuple[str | None, str, int | None, list[str], str | None, str, str, str, dict | None, dict | None]:
    """Returns (action_token, target, back_to_step, observed, look_dir,
    current_room_type, candidate_focus, commitment, target_marker,
    explore_marker).

    target_marker: VLM's mark of where it sees the target object in the
    current frame. Dict {x, y, [label]} with x/y in [0, 1] normalized
    pixel coords, or None. Used by the engine to back-project the marker
    pixel + depth into world coords and admit a target_memory cell —
    more accurate than guessing '0.8m ahead'.

    explore_marker: VLM's mark of which direction it wants to explore
    next. Dict {x, y} in normalized pixel coords, or None. Logged for
    analysis; future iterations may use it to bias frontier sampling.

    commitment: 'low' / 'medium' / 'high'. Only STOP at 'high'.
    """
    EMPTY = (None, "", None, [], None, "", "", "", None, None)
    parsed, err = tolerant_json_parse(resp_text)
    if parsed is None:
        return EMPTY
    raw = str(parsed.get("action", "") or "").strip().upper()
    target = str(parsed.get("target", "") or "").strip()
    back_to = parsed.get("back_to_step")
    try:
        back_to = int(back_to) if back_to is not None else None
    except (ValueError, TypeError):
        back_to = None
    obs = parsed.get("observed") or []
    if isinstance(obs, list):
        observed = [str(x).strip().lower() for x in obs if str(x).strip()][:8]
    else:
        observed = []
    look_dir = str(parsed.get("look_dir", "") or "").strip().lower() or None
    crt = str(parsed.get("current_room_type", "") or "").strip().lower()
    cfocus = str(parsed.get("candidate_focus", "") or "").strip().lower()
    commit = str(parsed.get("commitment", "") or "").strip().lower()
    if commit not in ("low", "medium", "high"):
        commit = ""
    target_marker = _parse_marker(parsed.get("target_marker"))
    explore_marker = _parse_marker(parsed.get("explore_marker"))
    if raw.startswith("STOP"):
        return "STOP", target, None, observed, None, crt, cfocus, commit, target_marker, explore_marker
    if raw.startswith("LOOK"):
        if not look_dir:
            import re
            m = re.search(r"LOOK[_\s]*(LEFT|RIGHT|BACK)", raw)
            if m:
                look_dir = m.group(1).lower()
        return "LOOK", target, None, observed, look_dir, crt, cfocus, commit, target_marker, explore_marker
    if raw.startswith("BACK"):
        if back_to is None:
            import re
            m = re.search(r"BACK[_\s]*(\d+)", raw)
            if m:
                back_to = int(m.group(1))
        return "BACK", target, back_to, observed, None, crt, cfocus, commit, target_marker, explore_marker
    if raw in LABELS[:n_waypoints]:
        return raw, target, None, observed, None, crt, cfocus, commit, target_marker, explore_marker
    for lbl in LABELS[:n_waypoints]:
        if lbl in raw:
            return lbl, target, None, observed, None, crt, cfocus, commit, target_marker, explore_marker
    return None, target, None, observed, None, crt, cfocus, commit, target_marker, explore_marker


# ---------------- Visited-waypoint suppression ----------------
MIN_WAYPOINTS_KEEP = 2   # filter only kicks in if we'd still have ≥ this many

def filter_visited_waypoints(waypoints: list[tuple],
                             agent_path: list[tuple[float, float]],
                             radius: float = SUPPRESS_RADIUS_M,
                             min_keep: int = MIN_WAYPOINTS_KEEP,
                             ) -> list[tuple]:
    """Drop any waypoint within `radius` (m, XY) of any cell in agent_path.
    Waypoint may be (x, y) or (x, y, angle) — first two coords define
    suppression check; tail is preserved.

    Late-game safeguard: if filtering leaves the agent with FEWER than
    `min_keep` choices (observed: agent in 1 room for 6+ steps had only 1
    waypoint surviving → forced into single-option corridor), return the
    full unfiltered list. Better an A/B/C with 1 revisit candidate than
    a single A that the agent has to take."""
    if not agent_path:
        return waypoints
    kept = []
    for w in waypoints:
        wx, wy = w[0], w[1]
        too_close = any(
            (wx - px) ** 2 + (wy - py) ** 2 < radius ** 2
            for (px, py) in agent_path
        )
        if not too_close:
            kept.append(w)
    return kept if len(kept) >= min_keep else waypoints


def _bearing_label(angle_rad: float) -> str:
    """Render a signed angle (radians, agent-relative, +ccw left, -ccw right)
    as a short bearing string. Drops the precise degree (saves tokens) —
    the VLM doesn't reason about exact degrees, just direction sectors."""
    deg = math.degrees(angle_rad)
    if abs(deg) < 22.5:                      return "front"
    if deg >= 22.5 and deg < 67.5:           return "front-left"
    if deg >= 67.5 and deg <= 112.5:         return "left"
    if deg > 112.5:                          return "rear-left"
    if deg <= -22.5 and deg > -67.5:         return "front-right"
    if deg <= -67.5 and deg >= -112.5:       return "right"
    if deg < -112.5:                         return "rear-right"
    return "front"


def run_episode(env, wm: WalkableMap, item: dict, episode_meta: dict,
                style: str, model_key: str, step_cap: int, num_waypoints: int,
                out_path: Path, objectnav: bool = False) -> dict:
    """The episode directory is `out_path.parent`; we write per-step trace
    files (overlay JPG + prompt + response) directly into it as the run
    progresses, so users can `tail -f` and watch incremental progress.
    record.json is atomic-saved at the very end."""
    from waypoint_overlay import overlay_waypoints

    epi_dir = out_path.parent
    epi_dir.mkdir(parents=True, exist_ok=True)

    if objectnav:
        # ObjectNav diagnostic mode: bypass intent decoding entirely. Feed
        # the ground-truth target_category as the navigation goal, so the
        # step loop runs identically but on a perfect plan. Used to isolate
        # "VLM intent decoding is hurting" vs "navigation pipeline is hurting".
        tgt_cat = (episode_meta.get("target_category") or "").strip()
        tgt_room = (episode_meta.get("target_room") or "").strip()
        if not tgt_cat:
            raise ValueError(
                f"objectnav mode but episode_meta has no target_category "
                f"for {item['selection_id']}")
        intent = f"Navigate to a {tgt_cat}."
        plan = {
            "target_guess": tgt_cat,
            "candidate_objects": [tgt_cat],
            "likely_rooms": [tgt_room] if tgt_room else [],
            "strategy": f"Walk toward and STOP next to a {tgt_cat}.",
            "action_plan": [],
        }
        plan_usage = None  # no API call billed
    else:
        intent = intent_for_style(item, style)
        if not intent:
            # Dataset entry missing the `<style>_en` field — skip episode rather
            # than send empty user prompt to the VLM (which would silently waste
            # tokens and produce a meaningless trajectory).
            raise ValueError(
                f"empty intent for {item['selection_id']}/{style} "
                f"(missing '{style}_en' in dataset jsonl?)")
        # ---- Episode-start planning (text-only: intent → plan dict) ----
        plan, plan_usage = make_episode_plan(
            intent, model_key,
            sel_id=item["selection_id"], style=style)

    # Cache-aware: build the SYSTEM role ONCE per episode. agent_system
    # rules + intent + plan are all constant across the 30 steps, so the
    # API can prefix-cache them after step 1 (~50% input cost discount,
    # ~80% latency reduction on cached prefix processing).
    episode_system = build_episode_system(intent, plan)
    plan_call = {**(plan_usage or {}), "step": 0, "kind": "plan"}

    env.place_agent(episode_meta["start_position"],
                    episode_meta["start_rotation_quat_wxyz"])
    # WalkableMap is cached across episodes within a worker; wipe the
    # explored-count grid at episode start so frontier scoring sees a
    # clean slate.
    if hasattr(wm, "reset_explored"):
        wm.reset_explored()

    # ----- Auto-reorient if start view is wall-pinned -----
    # Pure freemap ray-cast (16 directions, ~ms): if current yaw can clear
    # < threshold_m forward without hitting a wall, rotate to the direction
    # with the longest unobstructed ray. No Isaac Sim render. Matches the
    # offline dataset-level fix algorithm exactly so behavior is consistent.
    # The dataset-level fix in episodes.jsonl already handles 30%; this
    # runtime check is a safety net for any spawn that wasn't caught.
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

    # Memory accumulators
    action_history: deque[str] = deque(maxlen=HISTORY_KEEP)
    seen_objects: set[str] = set()  # accumulated VLM-claimed objects across steps
    # Persistent target memory: every step we strict-match DINO detection
    # of plan.target_guess (NOT candidates), back-project bbox center to
    # world (x, y) using camera yaw + depth + 90° HFOV, and accumulate.
    # Solves "agent walked past target — turned head — target out of view"
    # by remembering where we saw target regardless of current view.
    target_memory: list[dict] = []  # [{xy, score, step, label}, ...]
    # Running record of which candidate the VLM is currently focused on
    # — exploration → refinement → commit. Maxlen 8 = last ~8 steps of
    # focus, surfaced back to VLM next step as "your last focus track".
    focus_history: deque[str] = deque(maxlen=8)
    # Last VLM-reported room type (for room-mismatch hint to NEXT step's prompt).
    # Engine compares against plan.likely_rooms — purely VLM-driven, no oracle.
    last_room_type: str = ""
    last_room_match: bool = True
    agent_path: list[tuple[float, float]] = [(pose["position"][0], pose["position"][1])]
    # Per-step pose log for back-tracking. Stores (step, x, y, yaw, room)
    # so the VLM can issue {"action":"BACK","back_to_step":N} to rewind
    # to a previous viewpoint without walking back step-by-step.
    step_history: list[dict] = [{"step": 0, "x": pose["position"][0],
                                  "y": pose["position"][1], "yaw": pose["yaw"],
                                  "room": r0, "via": "start"}]

    stop_reason = "step_limit"
    target = ""
    final_rgb = None
    usage_calls: list[dict] = []
    # CRITICAL: only count plan tokens on cache MISS (plan_usage is None on
    # cache hit). Counting cache-hit plan_call would double-count tokens
    # across the 4 styles of the same SEL — e.g., 4 records would each
    # claim ~150 plan tokens when only 150 were actually billed once.
    if plan_usage:
        usage_calls.append(plan_call)

    # Stuck detection for plan refresh:
    #   Trigger replan when agent has been in the SAME room for >REPLAN_STREAK
    #   steps without entering any new room. Disabled by default after test 4
    #   showed it FLIPPED CORRECT PLANS to wrong ones (e.g., SEL_003 had
    #   correct sink/kitchen plan, replan changed it to laundry_room/bedroom).
    #   The replan VLM doesn't have enough info to give a *better* plan —
    #   it just gives a *different* one. Re-enable via env REPLAN_ENABLE=1
    #   for ablation studies.
    REPLAN_ENABLE = os.environ.get("REPLAN_ENABLE", "0") == "1"
    REPLAN_STREAK = 8
    MAX_REPLANS = 2 if REPLAN_ENABLE else 0
    same_room_streak = 0
    prev_room_observed = r0
    replan_count = 0
    replan_log: list[dict] = []


    for step in range(1, step_cap + 1):
        # Mark the agent's current pose as explored before this step's
        # frontier sample. Builds up a coarse "visited" grid that
        # sample_frontier_waypoints uses to bias toward unexplored.
        if hasattr(wm, "mark_explored"):
            wm.mark_explored(pose["position"][0], pose["position"][1],
                             radius_m=0.5)

        rgb = env.render_rgb()
        # Render depth right after RGB so the replicator step is shared.
        # depth_grid: 9 cells of min depth, fed to VLM in user prompt
        # so it has actual distance numbers when picking target_loc.
        # center_depth kept for trajectory logging.
        depth = env.render_depth()

        # ----- GroundingDINO detection — fed to VLM as auxiliary signal.
        # Engine does NOT override VLM's STOP. VLM sees DINO outputs in
        # the user prompt and decides itself. This avoids the
        # false-positive trap of engine-side force-STOP (D2.5/D3/D4 all
        # fired at wrong locations when DINO matched a candidate that
        # wasn't the actual target).
        dino_dets = []  # list of dicts {label, score, bbox, center_xy, depth_m, grid}
        tg_for_dino = (plan.get("target_guess") or "").strip()
        cand_for_dino = [c.strip() for c in (plan.get("candidate_objects") or []) if c]
        dino_terms = []
        if tg_for_dino:
            dino_terms.append(tg_for_dino)
        for c in cand_for_dino[:5]:
            if c.lower() not in [t.lower() for t in dino_terms]:
                dino_terms.append(c)
        if dino_terms:
            raw_dets = dino_detector.detect(rgb, dino_terms,
                                            threshold=0.30, text_threshold=0.20)
            for label, score, bbox in raw_dets[:5]:  # top-5 max
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                bbox_depth = None
                if depth is not None:
                    h_d, w_d = depth.shape[:2]
                    if 0 <= cy < h_d and 0 <= cx < w_d:
                        v = float(depth[cy, cx])
                        import math as _math
                        if _math.isfinite(v) and v > 0.05:
                            bbox_depth = v
                # Map bbox center to 3x3 grid label for the VLM prompt
                if depth is not None:
                    h_d, w_d = depth.shape[:2]
                    col = "left" if cx < w_d/3 else ("right" if cx > 2*w_d/3 else "center")
                    row = "top" if cy < h_d/3 else ("btm" if cy > 2*h_d/3 else "ctr")
                else:
                    col, row = "?", "?"
                grid = f"{row}-{col}"
                dino_dets.append({
                    "label": label, "score": score, "bbox": bbox,
                    "center_xy": [cx, cy], "depth_m": bbox_depth,
                    "grid": grid,
                })

        # ----- Update target memory (strict admit only) -----
        # When DINO sees the planner's target_guess (substring match) at
        # score >= 0.40 with a valid depth, back-project to world xy and
        # store. Cells within CLUSTER_RADIUS=1m merge so repeated hits of
        # the same physical location don't blow up the list.
        if tg_for_dino and dino_dets:
            tg_lo = tg_for_dino.lower()
            CLUSTER_RADIUS = 1.0
            for d in dino_dets:
                if d.get("depth_m") is None:
                    continue
                if not (tg_lo and tg_lo in d["label"].lower() and d["score"] >= 0.40):
                    continue
                px, py = d["center_xy"]
                W_img, H_img = rgb.shape[1], rgb.shape[0]
                u_norm = (px / W_img) * 2.0 - 1.0  # [-1, 1], +1 = right edge
                # Camera HFOV=90° (focal=10, h_aperture=20). half = 45°.
                # Pixel-right (u>0) maps to world right of camera forward,
                # which is yaw - delta in standard CCW math convention.
                bearing_offset = -u_norm * (math.pi / 4.0)
                world_yaw = pose["yaw"] + bearing_offset
                d_m = d["depth_m"]
                tx = pose["position"][0] + d_m * math.cos(world_yaw)
                ty = pose["position"][1] + d_m * math.sin(world_yaw)
                merged = False
                for mem in target_memory:
                    if math.hypot(mem["xy"][0]-tx, mem["xy"][1]-ty) < CLUSTER_RADIUS:
                        if d["score"] > mem["score"]:
                            mem["xy"] = [tx, ty]
                            mem["score"] = d["score"]
                            mem["label"] = d["label"]
                        mem["step"] = step
                        mem.setdefault("sources", []).append("dino")
                        merged = True
                        break
                if not merged:
                    target_memory.append({
                        "xy": [tx, ty],
                        "score": d["score"],
                        "step": step,
                        "label": d["label"],
                        "sources": ["dino"],
                    })

        # Track last frame the agent actually saw — used as final_rgb if the
        # episode hits step_cap (otherwise judge sees a post-decision render
        # that doesn't reflect the agent's last observation).
        final_rgb = rgb
        _spread = (2 * math.pi / num_waypoints) if ANGULAR_SPREAD_ENABLED else None
        candidates = env.sample_frontier_waypoints(
            wm, K=num_waypoints, seed=step,
            force_angular_spread_rad=_spread,
        )
        waypoints = filter_visited_waypoints(candidates, agent_path)

        # Build room-mismatch hint from PREVIOUS step's reported room type
        # (the one VLM gave on its last observation). Compare against
        # plan.likely_rooms; if mismatch, hint to head toward exits.
        likely = [r.lower() for r in (plan.get("likely_rooms") or [])]
        mismatch_hint = None
        if last_room_type and last_room_type not in ("", "unknown") and likely:
            if not any(lr in last_room_type or last_room_type in lr for lr in likely):
                mismatch_hint = (
                    f"You reported being in '{last_room_type}' but target is "
                    f"likely in {'/'.join(likely)}. Pick a waypoint heading "
                    f"toward a doorway/opening to leave this room."
                )

        # ----- Target memory hint -----
        # If DINO has detected the planner's target_guess earlier, surface
        # the location to the VLM as a soft hint. The VLM still decides
        # whether to STOP — the engine never forces a STOP based on memory.
        memory_hint = None
        if target_memory:
            cx_p, cy_p = pose["position"][0], pose["position"][1]
            nearest = min(target_memory,
                          key=lambda m: math.hypot(m["xy"][0]-cx_p, m["xy"][1]-cy_p))
            nx, ny = nearest["xy"]
            mem_d = math.hypot(nx-cx_p, ny-cy_p)
            abs_brg = math.atan2(ny-cy_p, nx-cx_p)
            rel_brg = abs_brg - pose["yaw"]
            # normalize to [-π, π]
            while rel_brg > math.pi:  rel_brg -= 2*math.pi
            while rel_brg < -math.pi: rel_brg += 2*math.pi
            deg = math.degrees(rel_brg)
            if abs(deg) < 30:    dir_word = "directly ahead"
            elif abs(deg) < 60:  dir_word = "front-left" if deg > 0 else "front-right"
            elif abs(deg) < 120: dir_word = "left" if deg > 0 else "right"
            elif abs(deg) < 150: dir_word = "rear-left" if deg > 0 else "rear-right"
            else:                dir_word = "behind"
            label = nearest["label"]
            ds = nearest["step"]
            # When close to a previously-detected target cell, surface a
            # strong STOP nudge — the VLM otherwise tends to keep walking
            # past the target. 1.5 m matches the 2 m success radius with
            # a small margin so an early STOP from this hint still scores.
            strong_radius = float(os.environ.get("MEMORY_HINT_STRONG_RADIUS", "1.5"))
            if mem_d < strong_radius:
                memory_hint = (
                    f"You detected '{label}' here at step {ds} ({mem_d:.2f}m "
                    f"from your current pose). STRONGLY consider STOP — "
                    f"this is where the target was confirmed."
                )
            else:
                memory_hint = (
                    f"You detected '{label}' previously (step {ds}); "
                    f"the location is now {mem_d:.1f}m {dir_word} of you. "
                    f"Approach if you still need to verify."
                )

        # Per-step location/orientation caption for the VLM (no map image,
        # just text). Lets the VLM track 'where am I, where am I facing,
        # how much have I covered'.
        position_caption = make_step_position_caption(
            wm,
            (pose["position"][0], pose["position"][1]),
            pose["yaw"],
        )

        if not waypoints:
            # No walkable options → forced stop via separate VLM query with no markers
            prompt = build_step_prompt(
                step, step_cap, 0,
                last_actions=list(action_history),
                seen_objects=seen_objects,
                room_mismatch_hint=mismatch_hint,
                dino_dets=dino_dets,
                memory_hint=memory_hint,
                focus_history=list(focus_history),
                position_caption=position_caption,
            )
            img_bytes = pil_to_jpeg_bytes(Image.fromarray(rgb))
            resp, err, usage, retried = call_vlm_step(
                model_key, episode_system, prompt, img_bytes,
                kind="no_waypoints", step=step,
                sel_id=item["selection_id"], style=style,
            )
            if usage:
                usage_calls.append({**usage, "step": step, "kind": "no_waypoints"})
            if resp:
                act, t, _, obs, _, _crt, _cf, _cm, _tm, _em = parse_action(resp, 0)
                target = t or target
                if obs:
                    seen_objects.update(obs)
                traj.append({"step": step, "action": "no_waypoints_stop",
                             "target": target, "observed": obs, "usage": usage,
                             **({"retried": True} if retried else {})})
            else:
                # VLM failed at the forced-stop call. Don't drop the step
                # silently — log it so post-hoc analysis sees the gap.
                traj.append({"step": step, "action": "no_waypoints_vlm_fail",
                             "error": err,
                             **({"retried": True} if retried else {})})
            stop_reason = "no_waypoints"
            final_rgb = rgb
            break

        cam_pos = np.array(pose["position"])
        cam_look = np.array([np.cos(pose["yaw"]), np.sin(pose["yaw"]), 0.0])
        # Strip bearing for the overlay renderer (it only needs xy).
        wp_xy = [(w[0], w[1]) for w in waypoints]
        wp_bearings = [w[2] if len(w) > 2 else 0.0 for w in waypoints]
        # Engine-computed Euclidean distance from current pose to each
        # candidate. Not an oracle (a real robot has depth); gives VLM
        # an absolute scale so it can judge "if I take this waypoint,
        # I'll be ~Xm from current view".
        wp_distances = [math.hypot(w[0] - cam_pos[0], w[1] - cam_pos[1])
                         for w in waypoints]
        overlay = overlay_waypoints(rgb, wp_xy, cam_pos, cam_look)
        img_bytes = pil_to_jpeg_bytes(overlay)

        prompt = build_step_prompt(
            step, step_cap, len(waypoints),
            last_actions=list(action_history),
            waypoint_bearings=wp_bearings,
            waypoint_distances=wp_distances,
            step_history=step_history,
            seen_objects=seen_objects,
            room_mismatch_hint=mismatch_hint,
            dino_dets=dino_dets,
            memory_hint=memory_hint,
            focus_history=list(focus_history),
            position_caption=position_caption,
        )

        resp, err, usage, retried = call_vlm_step(
            model_key, episode_system, prompt, img_bytes,
            kind="policy", step=step,
            sel_id=item["selection_id"], style=style,
        )
        if usage:
            usage_calls.append({**usage, "step": step, "kind": "policy"})

        # Per-step trace dropped directly into the episode dir
        # (visible during run for `tail -f` style monitoring)
        tag = f"step_{step:02d}"
        try:
            overlay.save(epi_dir / f"{tag}_overlay.jpg", "JPEG", quality=70, optimize=True)
            (epi_dir / f"{tag}_prompt.txt").write_text(prompt, encoding="utf-8")
            (epi_dir / f"{tag}_response.txt").write_text(
                (resp or f"[ERROR] {err}"), encoding="utf-8")
        except Exception as save_err:
            print(f"[trace warn] {item['selection_id']}/{style} step {step}: {save_err}",
                  file=sys.stderr)

        if resp is None:
            # VLM still down after step-level retry. Don't sit in place — pick
            # the nearest valid waypoint so the trajectory keeps progressing
            # (otherwise the agent loses N steps of budget on a remote outage,
            # and SR/SPL nosedive). Mark the step so post-hoc analysis can
            # attribute lost decisions to outage rather than bad policy.
            fb_idx = int(np.argmin(wp_distances)) if wp_distances else 0
            fb_xy = (waypoints[fb_idx][0], waypoints[fb_idx][1])
            env.teleport_to(fb_xy)
            pose = env.get_pose()
            agent_path.append((pose["position"][0], pose["position"][1]))
            traj.append({"step": step, "error": err, "action": "fallback_waypoint",
                         "waypoint_idx": fb_idx,
                         "position": pose["position"], "yaw": pose["yaw"],
                         **({"retried": True} if retried else {})})
            action_history.append(f"FALLBACK_{LABELS[fb_idx]}")
            continue

        action, t, back_to, obs, look_dir, current_room_type, candidate_focus, commitment, target_marker, explore_marker = parse_action(resp, len(waypoints))
        target = t or target
        if obs:
            seen_objects.update(obs)
        if current_room_type:
            last_room_type = current_room_type
        # Track focus evolution; "?" placeholder when VLM didn't commit.
        focus_history.append(candidate_focus or "?")

        # ----- VLM-side target memory admit (red marker) -----
        # When VLM stamps a red star at the target's pixel position in the
        # current frame, look up depth at that pixel and back-project to
        # world xy via camera HFOV=90°. Cluster-merges with any DINO admit
        # at the same physical location.
        # Fallback: if marker has no valid depth (looking at sky / out of
        # range), AND target_guess appears in `observed`, push a coarse
        # 0.8m-ahead cell as in the previous heuristic.
        admit_xy = None; admit_label = None; admit_score = 0.5
        if target_marker is not None and depth is not None:
            H_d, W_d = depth.shape[:2]
            px = int(round(target_marker["x"] * W_d))
            py = int(round(target_marker["y"] * H_d))
            if 0 <= py < H_d and 0 <= px < W_d:
                d_v = float(depth[py, px])
                if math.isfinite(d_v) and d_v > 0.05:
                    u_norm = (px / W_d) * 2.0 - 1.0  # [-1, 1]
                    bearing_offset = -u_norm * (math.pi / 4.0)  # 90° HFOV
                    world_yaw = pose["yaw"] + bearing_offset
                    admit_xy = (
                        pose["position"][0] + d_v * math.cos(world_yaw),
                        pose["position"][1] + d_v * math.sin(world_yaw),
                    )
                    admit_label = target_marker.get("label") or (
                        next((o for o in obs if isinstance(o, str)), "target")
                    )
                    admit_score = 0.6  # marker-backed: more confident than no-bbox VLM
        if admit_xy is None and tg_for_dino and obs:
            tg_lo_obs = tg_for_dino.lower()
            matched = next((o for o in obs
                             if isinstance(o, str) and tg_lo_obs in o.lower()),
                           None)
            if matched is not None:
                d_m = 0.8
                admit_xy = (
                    pose["position"][0] + d_m * math.cos(pose["yaw"]),
                    pose["position"][1] + d_m * math.sin(pose["yaw"]),
                )
                admit_label = matched
                admit_score = 0.5
        if admit_xy is not None:
            tx, ty = admit_xy
            CLUSTER_RADIUS = 1.0
            merged = False
            for mem in target_memory:
                if math.hypot(mem["xy"][0]-tx, mem["xy"][1]-ty) < CLUSTER_RADIUS:
                    mem["step"] = step
                    if admit_score > mem.get("score", 0):
                        mem["xy"] = [tx, ty]
                        mem["score"] = admit_score
                        mem["label"] = admit_label
                    mem.setdefault("sources", []).append("vlm")
                    merged = True
                    break
            if not merged:
                target_memory.append({
                    "xy": [tx, ty],
                    "score": admit_score,
                    "step": step,
                    "label": admit_label,
                    "sources": ["vlm"],
                })

        # ----- Explore marker: just log for now (drives nothing yet) -----
        # Future: bias next sample_frontier_waypoints toward this yaw.
        last_explore_marker = explore_marker

        # No engine-side Force-STOP. DINO results are fed to VLM as auxiliary
        # signal in the user prompt; VLM decides STOP itself.

        if action == "STOP":
            stop_reason = "declared_stop"
            # Engine-aided final approach: creep forward up to 0.5m along
            # the camera direction. Closes the gap between discrete-waypoint
            # sampler (min_step=0.5m) and canonical 1m success radius.
            # Only creeps if agent actually claimed a target — empty-target
            # STOPs ("I don't see anything") are honest fails and creep
            # would just move the agent randomly.
            if target:
                creep_d = env.creep_forward(wm, max_creep_m=1.0)
            else:
                creep_d = 0.0
            if creep_d > 0:
                pose = env.get_pose()
                agent_path.append((pose["position"][0], pose["position"][1]))
                final_rgb = env.render_rgb()
            else:
                pose = env.get_pose()
            # IMPORTANT: STOP entry MUST carry a `position` field so
            # compute_metrics.nav_metrics() picks the post-creep pos as
            # d_final. Without this, metric falls back to second-to-last
            # entry (the pre-STOP teleport pos) and creep is invisible.
            traj.append({"step": step, "action": "STOP",
                          "target": target, "usage": usage,
                          "creep_distance_m": creep_d,
                          "dino": dino_dets,
                          "observed": obs,
                          "target_marker": target_marker,
                          "explore_marker": explore_marker,
                          "position": pose["position"],
                          "yaw": pose["yaw"]})
            action_history.append("STOP")
            break

        if action == "LOOK":
            # In-place rotation. No teleport, no walkable check.
            # Used when agent suspects target nearby but current view
            # doesn't show it (orientation problem).
            ok = env.look(look_dir)
            pose = env.get_pose()
            traj.append({"step": step, "action": f"LOOK_{look_dir or '?'}",
                          "look_ok": ok,
                          "position": pose["position"], "yaw": pose["yaw"],
                          "observed": obs, "usage": usage})
            action_history.append(f"LOOK_{look_dir}" if ok else "LOOK_invalid")
            # Don't update step_history here — LOOK doesn't add a new
            # navigable viewpoint, and BACK should not rewind to a LOOK.
            continue

        if action == "BACK":
            # Validate target step number; clamp to history range.
            if back_to is None or not (0 <= back_to < len(step_history)):
                # Malformed BACK — treat as fallback waypoint
                action = LABELS[0]
                traj.append({"step": step,
                             "action": f"BACK_invalid→{action}",
                             "raw": (resp or "")[:200], "usage": usage})
            else:
                tgt = step_history[back_to]
                # Reconstruct face-direction from saved yaw so the camera
                # restores the same view the agent had at that step.
                face_xy = (tgt["x"] + math.cos(tgt["yaw"]),
                           tgt["y"] + math.sin(tgt["yaw"]))
                env.teleport_to((tgt["x"], tgt["y"]), face_direction_xy=face_xy)
                pose = env.get_pose()
                agent_path.append((pose["position"][0], pose["position"][1]))
                action_history.append(f"BACK_{back_to}")
                step_history.append({"step": step,
                                     "x": pose["position"][0],
                                     "y": pose["position"][1],
                                     "yaw": pose["yaw"],
                                     "room": tgt.get("room"),
                                     "via": f"BACK→{back_to}"})
                traj.append({"step": step, "action": f"BACK_{back_to}",
                             "position": pose["position"], "yaw": pose["yaw"],
                             "room": tgt.get("room"), "usage": usage})
                continue   # ← skip the normal A/B/C/D teleport below

        if action is None:
            action = LABELS[0]
            traj.append({"step": step, "action": f"malformed→{action}",
                          "raw": resp[:200], "usage": usage})

        idx = LABELS.index(action)
        # waypoints[idx] is now (x, y, angle); teleport only takes xy
        wx, wy = waypoints[idx][0], waypoints[idx][1]
        pre_xy = (pose["position"][0], pose["position"][1])
        env.teleport_to((wx, wy))
        pose = env.get_pose()
        agent_path.append((pose["position"][0], pose["position"][1]))
        action_history.append(action)
        room = wm.room_at(pose["position"][0], pose["position"][1])
        if room and room not in rooms_visited:
            rooms_visited.append(room)
        step_history.append({"step": step, "x": pose["position"][0],
                              "y": pose["position"][1], "yaw": pose["yaw"],
                              "room": room, "via": action})
        traj.append({"step": step, "action": action, "waypoint": [wx, wy],
                      "position": pose["position"], "yaw": pose["yaw"],
                      "room": room, "target_guess": target,
                      "observed": obs, "vlm_room_type": current_room_type,
                      "candidate_focus": candidate_focus,
                      "commitment": commitment,
                      "target_marker": target_marker,
                      "explore_marker": explore_marker,
                      "dino": dino_dets, "usage": usage})

        # ----- Stuck detection & plan refresh (legacy room-streak path) -----
        # Track consecutive same-room steps; if agent has been spinning in
        # one room for >=REPLAN_STREAK steps without entering a new one,
        # call replanner with previous plan as context.
        if room == prev_room_observed:
            same_room_streak += 1
        else:
            same_room_streak = 0
            prev_room_observed = room

        if same_room_streak >= REPLAN_STREAK and replan_count < MAX_REPLANS:
            new_plan, replan_usage = make_replan(
                intent, item["scene_id"], model_key,
                prev_plan=plan,
            )
            # Only adopt if replan actually returned a target_guess (not empty)
            if new_plan.get("target_guess"):
                old_summary = {"target": plan.get("target_guess"),
                               "rooms": plan.get("likely_rooms")}
                plan = {**plan, **new_plan, "_replan_at_step": step}
                replan_log.append({
                    "step": step, "old": old_summary,
                    "new": {"target": new_plan["target_guess"],
                            "rooms": new_plan["likely_rooms"]},
                })
                replan_count += 1
                same_room_streak = 0  # reset to give the new plan time
                if replan_usage:
                    usage_calls.append({**replan_usage, "step": step, "kind": "replan"})

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
        "selection_id": item["selection_id"],
        "tier": "vlm",
        "model": model_key,
        "style": style,
        "scene_id": item["scene_id"],
        "target_category": item["target_category"],
        "intent": intent,
        "photo": item.get("photo"),
        "final_frame": str(final_frame_path.relative_to(REPO)),
        "prediction": {"target": target},
        "plan": {**plan,
                 "plan_model": MODEL_CATALOG[model_key]["model"],
                 "plan_usage": plan_usage},
        "replans": replan_log,        # log of plan refreshes triggered by stuck
        "trajectory": traj,
        "rooms_visited": rooms_visited,
        "seen_objects": sorted(seen_objects),
        "target_memory": target_memory,
        "stop_reason": stop_reason,
        "step_cap": step_cap,
        "num_waypoints": num_waypoints,
        "episode_meta": episode_meta,
        "model_meta": {
            "provider": MODEL_CATALOG[model_key]["provider"],
            "model": MODEL_CATALOG[model_key]["model"],
            "temperature": 0.0,
        },
        "ablation_flags": {
            "fallback_retry": FALLBACK_RETRY_ENABLED,
            "angular_spread": ANGULAR_SPREAD_ENABLED,
        },
        "usage_total": usage_total,
        "timestamp": now_iso(),
    }
    save_atomic(record, out_path)
    return record


def main():
    ap = argparse.ArgumentParser()
    # `--model` is required UNLESS --queue-dir is set. In queue mode the
    # model can be encoded in the item filename as <SEL>__<STYLE>__<MODEL>,
    # letting one worker dispatch across multiple VLM tiers (no re-launch
    # of Isaac Sim — only the API target changes per item).
    ap.add_argument("--model", default=None, choices=MODEL_CHOICES,
                    help="Default model for non-queue mode, OR fallback "
                         "model when queue items lack a __<MODEL> suffix.")
    ap.add_argument("--style", default="formal",
                    choices=["formal", "natural", "casual", "emotional", "all"])
    ap.add_argument("--step-cap", type=int, default=30)
    ap.add_argument("--num-waypoints", type=int, default=4)
    ap.add_argument("--objectnav", action="store_true",
                    help="Diagnostic mode: bypass intent decoding, feed "
                         "ground-truth target_category as the goal. Same "
                         "step loop / sampler / VLM, just perfect plan. "
                         "Use to isolate intent-vs-nav bottleneck.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated selection_ids to run (skip everything else)")
    ap.add_argument("--max-scenes", type=int, default=None,
                    help="Exit cleanly after N distinct USD scenes have been "
                         "loaded into this Isaac Sim instance. Workaround for "
                         "Kit's USD-reload memory leak (~150-300 MB/scene "
                         "leaks; OOMs on long runs after ~50-80 scenes). "
                         "The driver script restarts the worker to continue.")
    # Queue-pull mode: workers stay persistent and atomically claim
    # (SEL, STYLE) units from a shared on-disk queue. Solves the
    # static-partitioning problem (idle GPUs after a worker drains its
    # batch). When --queue-dir is set, --only/--style/--limit are ignored.
    ap.add_argument("--queue-dir", type=str, default=None,
                    help="Pull (SEL, STYLE) work units from this dir's "
                         "pending/ subdir atomically (os.rename to inflight/<wid>). "
                         "Worker loops until pending+inflight empty.")
    ap.add_argument("--worker-id", type=str, default=None,
                    help="Required with --queue-dir. Used as suffix on the "
                         "atomic claim filename (inflight/<sel>__<style>.<wid>).")
    args = ap.parse_args()

    # Sanity: legacy non-queue mode requires --model.
    if not args.queue_dir and not args.model:
        ap.error("--model is required unless --queue-dir is set "
                 "(in which case items can encode model as <SEL>__<STYLE>__<MODEL>)")

    # Strip our CLI args from sys.argv BEFORE importing SimulationApp:
    # Isaac Sim's Kit framework reads sys.argv on init and tries to interpret
    # unknown args as Kit options. Some Kit/driver combinations hang or abort
    # when they see our --model / --only arguments. Argparse already consumed
    # the values into `args`, so clearing sys.argv is safe.
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

    # ---------------- Queue-pull mode ----------------
    # Worker stays persistent and atomically claims (SEL, STYLE) markers from
    # queue_dir/pending/<SEL>__<STYLE>. The os.rename() to inflight/<...>.<wid>
    # is the linearization point — only one worker can succeed; others see
    # the file gone and try the next. When pending is empty, worker exits.
    # Scenes are loaded on demand and cached; --max-scenes still respected.
    def _queue_run():
        from pathlib import Path
        if not args.worker_id:
            print("[vlm] --queue-dir requires --worker-id", file=sys.stderr); return
        qdir = Path(args.queue_dir)
        pending = qdir / "pending"
        inflight = qdir / "inflight"
        done = qdir / "done"
        for d in (pending, inflight, done):
            d.mkdir(parents=True, exist_ok=True)
        wid = args.worker_id
        n_done_here = 0
        while True:
            claimed = None
            try:
                names = sorted(p.name for p in pending.iterdir())
            except FileNotFoundError:
                break
            if not names:
                break
            for name in names:
                src = pending / name
                dst = inflight / f"{name}.{wid}"
                try:
                    src.rename(dst)
                    claimed = (name, dst)
                    break
                except (OSError, FileNotFoundError):
                    continue   # race lost — try next
            if claimed is None:
                break          # pending dir non-empty but all racing
            name, dst = claimed
            # Item name formats supported:
            #   <SEL>__<STYLE>                    (single-model mode; uses args.model)
            #   <SEL>__<STYLE>__<MODEL>           (multi-model mode; per-item dispatch)
            parts = name.split("__")
            if len(parts) == 2:
                sel_id, style = parts
                item_model = args.model
            elif len(parts) == 3:
                sel_id, style, item_model = parts
                if item_model not in MODEL_CATALOG:
                    print(f"[vlm/queue] unknown model {item_model!r} in {name!r}, skip",
                          file=sys.stderr)
                    (done / name).touch(); dst.unlink(missing_ok=True); continue
            else:
                print(f"[vlm/queue] bad item name {name!r}, skipping", file=sys.stderr)
                dst.unlink(missing_ok=True)
                continue
            if not item_model:
                print(f"[vlm/queue] no model resolved for {name!r}; pass --model "
                      f"or put __<MODEL> suffix in queue items", file=sys.stderr)
                dst.unlink(missing_ok=True)
                continue
            item = items_by_sel.get(sel_id)
            if item is None or sel_id not in episodes:
                print(f"[vlm/queue] {sel_id}: missing item or episode, skipping",
                      file=sys.stderr)
                (done / name).touch()
                dst.unlink(missing_ok=True)
                continue
            ep = episodes[sel_id]
            scene_id = ep["scene_id"]
            if (args.max_scenes is not None
                    and scene_id not in loaded_scenes
                    and len(loaded_scenes) >= args.max_scenes):
                # This worker is full; release the claim back to pending so
                # another worker can pick it up.
                try: dst.rename(pending / name)
                except Exception: pass
                print(f"[vlm/queue] {wid}: hit --max-scenes={args.max_scenes}, "
                      f"released {name} back to pending and exiting "
                      f"(did {n_done_here} items)")
                break
            if scene_id not in wm_cache:
                wm_cache[scene_id] = WalkableMap.load(scene_id)
            wm = wm_cache[scene_id]
            if wm is None:
                print(f"[vlm/queue] {sel_id}: walkable_map missing for {scene_id} — skip",
                      file=sys.stderr)
                (done / name).touch()
                dst.unlink(missing_ok=True)
                continue
            usd_path = USD_ROOT / scene_id / "start_result_navigation.usd"
            env.load_scene(scene_id, str(usd_path))
            loaded_scenes.add(scene_id)
            out_path = episode_path(item["scene_id"], "vlm", item_model, style, sel_id)
            if out_path.exists() and not args.force:
                (done / name).touch()
                dst.unlink(missing_ok=True)
                continue
            try:
                run_episode(
                    env, wm, item, ep, style,
                    model_key=item_model,
                    step_cap=args.step_cap,
                    num_waypoints=args.num_waypoints,
                    out_path=out_path,
                    objectnav=args.objectnav,
                )
                (done / name).touch()
                dst.unlink(missing_ok=True)
                n_done_here += 1
                if n_done_here % 5 == 0:
                    print(f"[vlm/queue] {wid}: {n_done_here} items done")
            except Exception as e:
                import traceback
                print(f"[vlm/queue] {sel_id}/{style}/{item_model}: EXC {e}")
                traceback.print_exc()
                # Leave inflight marker as-is so post-run scan flags the leak
                # (don't move to done; failed item stays visible for triage).

        print(f"[vlm/queue] {wid}: queue drained, {n_done_here} items processed")

    try:
        if args.queue_dir:
            _queue_run()
            return
        for idx, item in enumerate(items):
            sel_id = item["selection_id"]
            if sel_id not in episodes:
                continue
            ep = episodes[sel_id]
            scene_id = ep["scene_id"]

            # Memory-recycle gate: if this is a NEW scene and we've already
            # loaded `--max-scenes` distinct ones, exit cleanly so the driver
            # can restart us with a fresh Kit. Already-loaded scenes can keep
            # running (no extra USD-load leak).
            if (args.max_scenes is not None
                    and scene_id not in loaded_scenes
                    and len(loaded_scenes) >= args.max_scenes):
                print(f"[vlm] recycle: hit --max-scenes={args.max_scenes}, "
                      f"exiting cleanly so driver can restart "
                      f"(processed {idx}/{len(items)} items)")
                break

            if scene_id not in wm_cache:
                wm_cache[scene_id] = WalkableMap.load(scene_id)
            wm = wm_cache[scene_id]
            if wm is None:
                # Walkable map missing — skip with explicit warning so silent
                # path mis-config (e.g. INTENTIONNAV_METAROOT pointing one level
                # too high) doesn't manifest as zero records. Was the
                # distributed-runner bug on remote: METAROOT missing
                # /metadata_train suffix.
                print(f"[vlm] {sel_id}: walkable_map missing for {scene_id} "
                      f"(check INTENTIONNAV_METAROOT) — skipping", file=sys.stderr)
                continue

            usd_path = USD_ROOT / scene_id / "start_result_navigation.usd"
            env.load_scene(scene_id, str(usd_path))
            loaded_scenes.add(scene_id)

            for style in styles:
                out_path = episode_path(item["scene_id"], "vlm", args.model, style, sel_id)
                if out_path.exists() and not args.force:
                    continue
                try:
                    run_episode(
                        env, wm, item, ep, style,
                        model_key=args.model,
                        step_cap=args.step_cap,
                        num_waypoints=args.num_waypoints,
                        out_path=out_path,
                        objectnav=args.objectnav,
                    )
                except Exception as e:
                    import traceback
                    print(f"[vlm] {sel_id}/{style}: EXC {e}")
                    traceback.print_exc()

            if (idx + 1) % 5 == 0:
                print(f"[vlm] {idx+1}/{len(items)} processed "
                      f"({len(loaded_scenes)} scenes loaded)")
    finally:
        env.close()


if __name__ == "__main__":
    main()
