"""T_RE: random exploration baseline.

Agent is spawned in scene, walks uniformly-random waypoints each step, and
stops at step_cap. Final target prediction: ONE VLM call looking at the final
frame + intent → target. Mirrors EXPRESS-Bench's "see last frame, answer"
pattern and makes the tier a measurement of "stupid-walk + VLM-final-pick".

Reports navigation metrics (SR / OSR / SPL) via trajectory, plus the same
prediction.target field as oracle/blind so the judge can score uniformly.

Usage (requires goodnav env + setup_conda_env.sh):
  python agents/agent_random.py --style formal --step-cap 50
"""

from __future__ import annotations

import argparse
import json
import random
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


def load_episodes() -> dict[str, dict]:
    out = {}
    with EPISODES_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            out[r["selection_id"]] = r
    return out


def vlm_final_answer(intent: str, image_bytes: bytes, model_key: str = FINAL_ANSWER_MODEL
                     ) -> tuple[str, str, dict | None]:
    """VLM sees final-frame image + intent, outputs target.
    Returns (target, raw_response, usage)."""
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


def run_episode(env, wm: WalkableMap, item: dict, episode_meta: dict,
                style: str, step_cap: int, num_waypoints: int,
                out_path: Path, seed: int) -> dict:
    from simulator.iss_env import IsaacSimEnv  # noqa: F401 (import check)

    intent = intent_for_style(item, style)
    rng = random.Random(seed)

    env.place_agent(episode_meta["start_position"],
                    episode_meta["start_rotation_quat_wxyz"])

    traj = []
    pose = env.get_pose()
    traj.append({"step": 0, "position": pose["position"], "yaw": pose["yaw"]})

    stopped = False
    for step in range(1, step_cap + 1):
        rgb = env.render_rgb()
        waypoints = env.sample_frontier_waypoints(
            wm, K=num_waypoints, seed=rng.randint(0, 2**31 - 1),
        )
        if not waypoints:
            break
        idx = rng.randrange(len(waypoints))
        # waypoints are (x, y, angle); only xy needed to teleport
        wx, wy = waypoints[idx][0], waypoints[idx][1]
        env.teleport_to((wx, wy))
        pose = env.get_pose()
        traj.append({
            "step": step,
            "position": pose["position"],
            "yaw": pose["yaw"],
            "action": f"waypoint_{idx}",
            "n_waypoints": len(waypoints),
        })

    epi_dir = out_path.parent
    epi_dir.mkdir(parents=True, exist_ok=True)
    final_rgb = env.render_rgb()
    final_frame_path = epi_dir / "final.png"
    env.save_frame(final_rgb, final_frame_path)

    # Final answer: VLM sees final frame + intent → target. Mirrors
    # EXPRESS-Bench's "look at last frame, answer" pattern.
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
        "tier": "random",
        "model": f"random_walk+{FINAL_ANSWER_MODEL}_vlm",
        "style": style,
        "scene_id": item["scene_id"],
        "target_category": item["target_category"],
        "intent": intent,
        "photo": item.get("photo"),
        "final_frame": str(final_frame_path.relative_to(REPO)),
        "prediction": {"target": target},
        "trajectory": traj,
        "step_cap": step_cap,
        "episode_meta": episode_meta,
        "model_meta": {
            "provider": f"isaac_sim+{MODEL_CATALOG[FINAL_ANSWER_MODEL]['provider']}",
            "model": f"random_walk+{MODEL_CATALOG[FINAL_ANSWER_MODEL]['model']}",
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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--headless", action="store_true", default=True)
    ap.add_argument("--only", type=str, default=None,
                    help="Comma-separated selection_ids to run (skip everything else)")
    ap.add_argument("--max-scenes", type=int, default=None,
                    help="Exit cleanly after N distinct USD scenes loaded "
                         "(driver-side recycle to dodge USD-reload memory leak).")
    args = ap.parse_args()

    # Strip CLI args from sys.argv before SimulationApp init (Kit otherwise
    # tries to interpret unknown args; some hardware/driver combos silently
    # break on this). See agent_vlm.py for the same fix.
    import sys as _sys
    _sys.argv = _sys.argv[:1]

    from simulator.iss_env import IsaacSimEnv  # lazy, requires goodnav env

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
                print(f"[random] {sel_id}: no episode spec — skipping")
                continue
            ep = episodes[sel_id]
            scene_id = ep["scene_id"]

            if (args.max_scenes is not None
                    and scene_id not in loaded_scenes
                    and len(loaded_scenes) >= args.max_scenes):
                print(f"[random] recycle: hit --max-scenes={args.max_scenes}, "
                      f"exiting (processed {idx}/{len(items)})")
                break

            # Load walkable map (cached)
            if scene_id not in wm_cache:
                wm_cache[scene_id] = WalkableMap.load(scene_id)
            wm = wm_cache[scene_id]
            if wm is None:
                print(f"[random] {sel_id}: walkable_map missing — skipping")
                continue

            # Load scene USD (cached in env by scene_id)
            usd_path = USD_ROOT / scene_id / "start_result_navigation.usd"
            env.load_scene(scene_id, str(usd_path))
            loaded_scenes.add(scene_id)

            for style in styles:
                out_path = episode_path(item["scene_id"], "random", "random_walk", style, sel_id)
                if out_path.exists() and not args.force:
                    continue
                # Per-SEL deterministic seed: stable across re-runs even when
                # `idx` changes due to chunk re-slicing (recycle, --only).
                # Hash (args.seed, sel_id) so the same SEL always gets the
                # same trajectory regardless of how the master script
                # partitioned work.
                import hashlib
                _h = hashlib.md5(f"{args.seed}_{sel_id}_{style}".encode()).digest()
                _episode_seed = int.from_bytes(_h[:4], "big") % (2 ** 31 - 1)
                try:
                    rec = run_episode(
                        env, wm, item, ep, style,
                        step_cap=args.step_cap,
                        num_waypoints=args.num_waypoints,
                        out_path=out_path,
                        seed=_episode_seed,
                    )
                except Exception as e:
                    import traceback
                    print(f"[random] {sel_id}/{style}: EXC {e}")
                    traceback.print_exc()

            if (idx + 1) % 10 == 0:
                print(f"[random] {idx+1}/{len(items)} processed")
    finally:
        env.close()


if __name__ == "__main__":
    main()
