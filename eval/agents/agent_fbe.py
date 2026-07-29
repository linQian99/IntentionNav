"""T_FBE: frontier-based exploration baseline (simplified EXPRESS-Bench port).

Instead of building a full TSDF like EXPRESS-Bench/src/tsdf.py, we use the
walkable freemap directly:
  - Maintain a "visited" mask on the grid
  - Each step, score candidate waypoints by proximity to the nearest
    unvisited walkable cell (the "frontier")
  - Pick the highest-scoring waypoint (break ties by directional spread)
  - After step_cap, produce a final target prediction via a VLM looking
    at the final frame.

Matches the spirit of EXPRESS-Bench's FBE (no semantic signal during nav,
just "go somewhere new") while reusing our walkable_map infrastructure.

Usage (requires goodnav env + setup_conda_env.sh):
  python agents/agent_fbe.py --style formal --step-cap 50
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent / "simulator"))
sys.path.insert(0, str(THIS_DIR.parent))    # so `import simulator.iss_env` works

from clients import call_vlm, MODEL_CATALOG  # noqa: E402
from common import (
    EVAL_DIR, REPO, USD_ROOT, STYLES, render_prompt, load_items, tolerant_json_parse,
    intent_for_style, episode_path, save_atomic, now_iso,
)  # noqa: E402
from walkable_map import WalkableMap  # noqa: E402

EPISODES_FILE = EVAL_DIR / "splits/episodes.jsonl"
FINAL_ANSWER_MODEL = "qwen3_6_plus"  # VLM that answers from the final frame


def load_episodes() -> dict[str, dict]:
    out = {}
    with EPISODES_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            out[r["selection_id"]] = r
    return out


def _jpeg_bytes_from_rgb(rgb, max_dim: int = 768, quality: int = 90) -> bytes:
    from PIL import Image
    import io as _io
    img = Image.fromarray(rgb)
    w, h = img.size
    if max(w, h) > max_dim:
        s = max_dim / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = _io.BytesIO()
    img.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def vlm_final_answer(intent: str, image_bytes: bytes, model_key: str = FINAL_ANSWER_MODEL
                     ) -> tuple[str, str, dict | None]:
    """VLM sees final-frame image + intent, outputs target. Returns
    (target, raw_response, usage)."""
    system = "You are a helpful embodied-QA assistant. You output only valid JSON. No prose."
    user = render_prompt("oracle_system", intent=intent)
    resp, err, usage = call_vlm(model_key=model_key, system=system, user_text=user,
                                 image_bytes=image_bytes, temperature=0.0,
                                 json_mode=True)
    if resp is None:
        return "", "", None
    parsed, perr = tolerant_json_parse(resp)
    target = str((parsed or {}).get("target", "") or "").strip()
    return target, resp, usage


def mark_visited(visited: np.ndarray, wm: WalkableMap, x: float, y: float,
                 radius_cells: int = 3):
    yi, xi = wm.world_to_cell(x, y)
    h, w = visited.shape
    for dyi in range(-radius_cells, radius_cells + 1):
        for dxi in range(-radius_cells, radius_cells + 1):
            if dyi * dyi + dxi * dxi > radius_cells * radius_cells:
                continue
            yy, xx = yi + dyi, xi + dxi
            if 0 <= yy < h and 0 <= xx < w:
                visited[yy, xx] = True


def frontier_score(waypoint_xy: tuple[float, float], visited: np.ndarray,
                   wm: WalkableMap, max_radius_cells: int = 40) -> float:
    """Score a waypoint by inverse distance to the nearest unvisited walkable
    cell. Higher score = closer to frontier of exploration."""
    h, w = wm.grid.shape
    yi, xi = wm.world_to_cell(*waypoint_xy)
    # r=0: the waypoint cell itself. Without this, a fresh frontier cell at
    # the candidate location returns score from the next ring (1.0 vs ∞).
    if 0 <= yi < h and 0 <= xi < w:
        if wm.grid[yi, xi] and not visited[yi, xi]:
            return 1.0  # waypoint IS the frontier cell — best possible score
    # BFS-like ring expansion to find nearest unvisited walkable cell.
    for r in range(1, max_radius_cells):
        for dyi in range(-r, r + 1):
            for dxi in range(-r, r + 1):
                if max(abs(dyi), abs(dxi)) != r:
                    continue  # ring boundary only
                yy, xx = yi + dyi, xi + dxi
                if 0 <= yy < h and 0 <= xx < w:
                    if wm.grid[yy, xx] and not visited[yy, xx]:
                        return 1.0 / r  # closer → higher score
    return 0.0


def run_episode(env, wm: WalkableMap, item: dict, episode_meta: dict,
                style: str, step_cap: int, num_waypoints: int,
                out_path: Path) -> dict:
    intent = intent_for_style(item, style)

    env.place_agent(episode_meta["start_position"],
                    episode_meta["start_rotation_quat_wxyz"])

    visited = np.zeros_like(wm.grid, dtype=bool)
    pose = env.get_pose()
    mark_visited(visited, wm, pose["position"][0], pose["position"][1])
    rooms_visited = []
    r0 = wm.room_at(pose["position"][0], pose["position"][1])
    if r0:
        rooms_visited.append(r0)
    traj = [{"step": 0, "position": pose["position"], "yaw": pose["yaw"], "room": r0}]

    for step in range(1, step_cap + 1):
        rgb = env.render_rgb()  # rendered but not used (no VLM in the loop)
        waypoints = env.sample_frontier_waypoints(wm, K=num_waypoints, seed=step)
        if not waypoints:
            traj.append({"step": step, "action": "no_waypoints"})
            break
        # waypoints are now (x, y, angle); frontier_score takes xy
        scores = [frontier_score((wp[0], wp[1]), visited, wm) for wp in waypoints]
        idx = int(np.argmax(scores))
        wx, wy = waypoints[idx][0], waypoints[idx][1]
        env.teleport_to((wx, wy))
        pose = env.get_pose()
        mark_visited(visited, wm, wx, wy)
        room = wm.room_at(wx, wy)
        if room and room not in rooms_visited:
            rooms_visited.append(room)
        traj.append({"step": step, "action": f"fbe_best_{idx}",
                      "position": pose["position"], "yaw": pose["yaw"],
                      "waypoint": [wx, wy], "frontier_scores": scores,
                      "room": room})

    epi_dir = out_path.parent
    epi_dir.mkdir(parents=True, exist_ok=True)
    final_rgb = env.render_rgb()
    final_frame_path = epi_dir / "final.png"
    env.save_frame(final_rgb, final_frame_path)

    final_bytes = _jpeg_bytes_from_rgb(final_rgb)
    target, raw, usage = vlm_final_answer(intent, final_bytes)
    usage_total = {
        "n_calls": 1 if usage else 0,
        "input_tokens_total": (usage or {}).get("input_tokens", 0),
        "output_tokens_total": (usage or {}).get("output_tokens", 0),
        "latency_s_total": round((usage or {}).get("latency_s", 0.0), 4),
    }

    record = {
        "selection_id": item["selection_id"],
        "tier": "fbe",
        "model": f"frontier_walk+{FINAL_ANSWER_MODEL}_vlm",
        "style": style,
        "scene_id": item["scene_id"],
        "target_category": item["target_category"],
        "intent": intent,
        "photo": item.get("photo"),
        "final_frame": str(final_frame_path.relative_to(REPO)),
        "prediction": {"target": target},
        "trajectory": traj,
        "rooms_visited": rooms_visited,
        "step_cap": step_cap,
        "episode_meta": episode_meta,
        "model_meta": {
            "provider": f"isaac_sim+{MODEL_CATALOG[FINAL_ANSWER_MODEL]['provider']}",
            "model": f"fbe_frontier+{MODEL_CATALOG[FINAL_ANSWER_MODEL]['model']}",
            "final_answer_image_used": True,
            "temperature": 0.0,
        },
        "raw_response": raw,
        "usage_total": usage_total,
        "timestamp": now_iso(),
    }
    save_atomic(record, out_path)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="formal",
                    choices=["formal", "natural", "casual", "emotional", "all"])
    ap.add_argument("--step-cap", type=int, default=30)
    ap.add_argument("--num-waypoints", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated selection_ids to run (skip everything else)")
    ap.add_argument("--max-scenes", type=int, default=None,
                    help="Exit cleanly after N distinct USD scenes loaded "
                         "(driver-side recycle to dodge USD-reload memory leak).")
    args = ap.parse_args()

    # Strip CLI args before SimulationApp init (see agent_vlm.py).
    import sys as _sys
    _sys.argv = _sys.argv[:1]

    from simulator.iss_env import IsaacSimEnv

    items = load_items()
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

    try:
        for idx, item in enumerate(items):
            sel_id = item["selection_id"]
            if sel_id not in episodes:
                continue
            ep = episodes[sel_id]
            scene_id = ep["scene_id"]

            if (args.max_scenes is not None
                    and scene_id not in loaded_scenes
                    and len(loaded_scenes) >= args.max_scenes):
                print(f"[fbe] recycle: hit --max-scenes={args.max_scenes}, "
                      f"exiting (processed {idx}/{len(items)})")
                break

            if scene_id not in wm_cache:
                wm_cache[scene_id] = WalkableMap.load(scene_id)
            wm = wm_cache[scene_id]
            if wm is None:
                continue

            usd_path = USD_ROOT / scene_id / "start_result_navigation.usd"
            env.load_scene(scene_id, str(usd_path))
            loaded_scenes.add(scene_id)

            for style in styles:
                out_path = episode_path(item["scene_id"], "fbe", "frontier_walk", style, sel_id)
                if out_path.exists() and not args.force:
                    continue
                try:
                    run_episode(env, wm, item, ep, style,
                                step_cap=args.step_cap,
                                num_waypoints=args.num_waypoints,
                                out_path=out_path)
                except Exception as e:
                    import traceback
                    print(f"[fbe] {sel_id}/{style}: EXC {e}")
                    traceback.print_exc()

            if (idx + 1) % 10 == 0:
                print(f"[fbe] {idx+1}/{len(items)} processed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
