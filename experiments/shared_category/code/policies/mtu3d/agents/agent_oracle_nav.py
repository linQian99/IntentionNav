"""Privileged shortest-path navigation oracle for IntentionNav.

This baseline uses the fixed target position and the precomputed walkable map.
It does not initialize Isaac Sim and does not call any model API.  Its purpose
is to calibrate reachability, action budget, and terminal stopping under the
same record schema as practical agents; it is not a deployable method.

Example:
  python agents/agent_oracle_nav.py --style formal
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR.parent / "simulator"))

from common import (  # noqa: E402
    EPISODES_FILE,
    STYLES,
    episode_path,
    intent_for_style,
    load_items,
    now_iso,
    save_atomic,
)
from walkable_map import WalkableMap  # noqa: E402


PROTOCOL_VERSION = "strict_stop_2026_08"


def load_episodes() -> dict[str, dict]:
    episodes = {}
    with EPISODES_FILE.open(encoding="utf-8") as f:
        for line in f:
            episode = json.loads(line)
            episodes[episode["selection_id"]] = episode
    return episodes


def yaw_from_quat_wxyz(quat: list[float]) -> float:
    """Extract planar yaw from a wxyz quaternion."""
    w, x, y, z = (float(v) for v in quat)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def compress_path(
    path: list[tuple[float, float]],
    wm: WalkableMap,
    max_step_m: float,
) -> list[tuple[float, float]]:
    """Compress a dense grid path into collision-free high-level moves."""
    if len(path) <= 1:
        return path

    compressed = [path[0]]
    anchor_idx = 0
    while anchor_idx < len(path) - 1:
        best_idx = anchor_idx + 1
        for idx in range(anchor_idx + 1, len(path)):
            distance = math.dist(path[anchor_idx], path[idx])
            if distance > max_step_m + 1e-9:
                break
            if wm.line_of_sight(
                *path[anchor_idx], *path[idx], tolerance=0.0
            ):
                best_idx = idx
        compressed.append(path[best_idx])
        anchor_idx = best_idx
    return compressed


def run_episode(
    wm: WalkableMap,
    item: dict,
    episode: dict,
    style: str,
    step_cap: int,
    max_step_m: float,
    stop_radius_m: float,
    out_path: Path,
) -> dict:
    start_xy = tuple(float(v) for v in episode["start_position"][:2])
    target_xy = tuple(float(v) for v in episode["target_position"][:2])
    dense_path = wm.shortest_path_2d(start_xy, target_xy)

    start_yaw = yaw_from_quat_wxyz(episode["start_rotation_quat_wxyz"])
    trajectory = [{
        "step": 0,
        "action": "START",
        "position": [start_xy[0], start_xy[1], 1.5],
        "yaw": start_yaw,
        "room": wm.room_at(*start_xy),
    }]
    stop_reason = "no_path"
    path_points = 0

    if dense_path:
        compressed = compress_path(dense_path, wm, max_step_m=max_step_m)
        path_points = len(dense_path)
        # STOP consumes one action, so reserve the final budget slot.
        move_points = compressed[1:step_cap]
        for step, point in enumerate(move_points, start=1):
            next_point = (
                compressed[step + 1]
                if step + 1 < len(compressed)
                else target_xy
            )
            yaw = math.atan2(next_point[1] - point[1], next_point[0] - point[0])
            trajectory.append({
                "step": step,
                "action": "MOVE",
                "position": [point[0], point[1], 1.5],
                "yaw": yaw,
                "room": wm.room_at(*point),
            })

        reached_path_end = len(move_points) == max(0, len(compressed) - 1)
        final_position = trajectory[-1]["position"]
        final_distance = math.hypot(
            final_position[0] - target_xy[0],
            final_position[1] - target_xy[1],
        )
        if reached_path_end and final_distance <= stop_radius_m:
            stop_step = len(move_points) + 1
            target_yaw = math.atan2(
                target_xy[1] - final_position[1],
                target_xy[0] - final_position[0],
            )
            trajectory.append({
                "step": stop_step,
                "action": "STOP",
                "position": final_position,
                "yaw": target_yaw,
                "room": wm.room_at(final_position[0], final_position[1]),
            })
            stop_reason = "oracle_goal_reached"
        elif not reached_path_end:
            stop_reason = "step_cap"
        else:
            stop_reason = "no_navigable_pose_within_stop_radius"

    record = {
        "selection_id": item["selection_id"],
        "tier": "oracle_nav",
        "model": "shortest_path_oracle",
        "style": style,
        "scene_id": item["scene_id"],
        "target_category": item["target_category"],
        "intent": intent_for_style(item, style),
        "photo": item.get("photo"),
        "prediction": {"target": item["target_category"]},
        "trajectory": trajectory,
        "stop_reason": stop_reason,
        "step_cap": step_cap,
        "episode_meta": episode,
        "evaluation_protocol": {
            "version": PROTOCOL_VERSION,
            "goal_input": "explicit_target",
            "metric_scope": "navigation_only",
            "stop_is_terminal": True,
            "post_budget_actions": False,
            "max_step_m": max_step_m,
            "stop_radius_m": stop_radius_m,
            "privileged_target_position": True,
        },
        "run_config": {
            "command": [sys.executable, *sys.argv],
            "step_cap": step_cap,
            "max_step_m": max_step_m,
            "stop_radius_m": stop_radius_m,
        },
        "planner_meta": {
            "planner": "astar_8_connected",
            "dense_path_points": path_points,
        },
        "model_meta": {
            "provider": "none",
            "model": "privileged_shortest_path_oracle",
            "api_calls": 0,
        },
        "timestamp": now_iso(),
    }
    save_atomic(record, out_path)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--style",
        default="formal",
        choices=[*STYLES, "all"],
    )
    parser.add_argument("--step-cap", type=int, default=30)
    parser.add_argument("--max-step-m", type=float, default=1.7)
    parser.add_argument("--stop-radius-m", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    items = load_items()
    if args.only:
        wanted = {value.strip() for value in args.only.split(",") if value.strip()}
        items = [item for item in items if item["selection_id"] in wanted]
    if args.limit is not None:
        items = items[:args.limit]

    episodes = load_episodes()
    styles = STYLES if args.style == "all" else (args.style,)
    wm_cache: dict[str, WalkableMap | None] = {}
    completed = skipped = failed = 0

    for item in items:
        selection_id = item["selection_id"]
        episode = episodes.get(selection_id)
        if episode is None:
            failed += 1
            print(f"[oracle_nav] missing episode: {selection_id}", file=sys.stderr)
            continue

        scene_id = episode["scene_id"]
        if scene_id not in wm_cache:
            wm_cache[scene_id] = WalkableMap.load(scene_id)
        wm = wm_cache[scene_id]
        if wm is None:
            failed += 1
            print(f"[oracle_nav] missing walkable map: {scene_id}", file=sys.stderr)
            continue

        for style in styles:
            out_path = episode_path(
                scene_id,
                "oracle_nav",
                "shortest_path_oracle",
                style,
                selection_id,
            )
            if out_path.exists() and not args.force:
                skipped += 1
                continue
            run_episode(
                wm,
                item,
                episode,
                style,
                step_cap=args.step_cap,
                max_step_m=args.max_step_m,
                stop_radius_m=args.stop_radius_m,
                out_path=out_path,
            )
            completed += 1

    print(
        f"[oracle_nav] completed={completed} skipped={skipped} failed={failed}"
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
