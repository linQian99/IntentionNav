"""Generate deterministic start poses for 500 IntentionNav episodes.

For each item in selected_500_intents.jsonl:
  1. Load target 3D position from capture_manifest.json (match by target_representative).
  2. Load walkable freemap for scene.
  3. Sample walkable points with geodesic distance to target ∈ [5m, 10m].
  4. Prefer points in a different room than the target's room.
  5. Assert frontier_count ≥ 3 (walkable cells within 1m) — else re-sample.
  6. Pick the first valid candidate (seeded per selection_id for determinism).

Writes `splits/episodes.jsonl` with one record per item:
  {
    "selection_id": "SEL_001",
    "scene_id": "kujiale_0262",
    "target_category": "air_conditioner",
    "target_object_id": "air_conditioner_0003/Meshes",
    "target_position": [x, y, z],
    "target_room": "living room_0",
    "start_position": [x, y, camera_height],
    "start_rotation_quat_wxyz": [w, x, y, z],
    "start_room": "bedroom_1",
    "start_room_type": "bedroom",
    "geodesic_to_target": 7.42,
    "euclidean_to_target": 6.18,
    "different_room": true,
    "frontier_count_at_start": 5
  }

Usage:
  python splits/make_episodes.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
EVAL_DIR = THIS_DIR.parent
REPO = EVAL_DIR.parent
DATASET_ROOT = Path(os.environ.get(
    "INTENTIONNAV_DATASET_ROOT",
    str(REPO / "data/benchmark"),
))
if not DATASET_ROOT.is_absolute():
    DATASET_ROOT = REPO / DATASET_ROOT
DATASET_JSONL = DATASET_ROOT / "selected_500_intents.jsonl"

sys.path.insert(0, str(EVAL_DIR / "simulator"))
from walkable_map import WalkableMap  # noqa: E402

CAMERA_HEIGHT = 1.5


def load_manifest(scene_id: str) -> list[dict]:
    path = DATASET_ROOT / scene_id / "manifest.json"
    m = json.loads(path.read_text())
    if isinstance(m, dict) and "entries" in m:
        m = m["entries"]
    if isinstance(m, list):
        return m
    return list(m.values())


def find_target(entries: list[dict], item: dict) -> tuple[dict | None, dict | None]:
    """Match (sequence_id, target_representative) → manifest entry + specific
    target_details row. Returns (entry, td) or (None, None) on miss."""
    seq = item.get("sequence_id")
    rep = item.get("target_representative")
    for e in entries:
        if e.get("sequence_id") != seq:
            continue
        for td in (e.get("target_details") or []):
            if td.get("object_id") == rep:
                return e, td
        # fallback: if only one td, accept it
        tds = e.get("target_details") or []
        if len(tds) == 1:
            return e, tds[0]
    return None, None


def frontier_count(wm: WalkableMap, x: float, y: float, radius_m: float = 1.0) -> int:
    """Number of walkable cells within radius_m of (x, y), grid-sampled."""
    if len(wm.x_coords) < 2 or len(wm.y_coords) < 2:
        return 0
    dx = abs(float(wm.x_coords[1] - wm.x_coords[0]))
    dy = abs(float(wm.y_coords[1] - wm.y_coords[0]))
    cells_r = int(math.ceil(radius_m / min(dx, dy)))
    yi0, xi0 = wm.world_to_cell(x, y)
    h, w = wm.grid.shape
    count = 0
    for dyi in range(-cells_r, cells_r + 1):
        for dxi in range(-cells_r, cells_r + 1):
            yi, xi = yi0 + dyi, xi0 + dxi
            if 0 <= yi < h and 0 <= xi < w and wm.grid[yi, xi]:
                count += 1
    return count


def yaw_to_quat_wxyz(yaw_rad: float) -> list[float]:
    """Quaternion for rotation about +Z axis (yaw only). World up = +Z."""
    return [math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2)]


def sample_start_pose(
    wm: WalkableMap,
    target_xy: tuple[float, float],
    target_room: str | None,
    seed: int,
    min_dist: float = 5.0,
    max_dist: float = 10.0,
    min_frontier: int = 3,
    max_trials: int = 400,
) -> dict | None:
    """Deterministically sample a start pose satisfying constraints. Returns
    None if no valid sample found within max_trials."""
    ys, xs = wm.grid.nonzero()
    if len(ys) == 0:
        return None
    rng = random.Random(seed)

    # Pre-shuffle a pool of walkable indices
    indices = list(range(len(ys)))
    rng.shuffle(indices)
    indices = indices[:max_trials * 4]  # bounded pool

    best_same_room = None  # fallback if we can't find different-room candidate
    for trial, idx in enumerate(indices):
        if trial >= max_trials:
            break
        yi, xi = int(ys[idx]), int(xs[idx])
        wx, wy = wm.cell_to_world(yi, xi)

        # Euclidean early reject
        eucl = math.hypot(wx - target_xy[0], wy - target_xy[1])
        if eucl < min_dist or eucl > max_dist * 2.0:
            continue

        # Geodesic range check
        geo = wm.geodesic_distance_2d((wx, wy), target_xy)
        if geo is None or geo < min_dist or geo > max_dist:
            continue

        # Frontier count (avoid broom-closet starts)
        fc = frontier_count(wm, wx, wy, radius_m=1.0)
        if fc < min_frontier:
            continue

        room = wm.room_at(wx, wy)
        different_room = (room is not None and target_room is not None
                           and room != target_room)

        yaw = rng.uniform(-math.pi, math.pi)
        candidate = {
            "start_position": [round(wx, 4), round(wy, 4), CAMERA_HEIGHT],
            "start_rotation_quat_wxyz": [round(v, 6) for v in yaw_to_quat_wxyz(yaw)],
            "start_room": room,
            "start_room_type": WalkableMap.normalize_room_type(room) if room else None,
            "geodesic_to_target": round(geo, 3),
            "euclidean_to_target": round(eucl, 3),
            "different_room": different_room,
            "frontier_count_at_start": fc,
        }

        if different_room:
            return candidate
        if best_same_room is None:
            best_same_room = candidate

    return best_same_room


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=THIS_DIR / "episodes.jsonl")
    ap.add_argument("--limit", type=int, default=None,
                    help="process only first N items (for debugging)")
    args = ap.parse_args()

    items = [json.loads(l) for l in DATASET_JSONL.open()]
    if args.limit:
        items = items[:args.limit]
    print(f"[episodes] processing {len(items)} items")

    # Cache maps + manifests per scene
    wm_cache: dict[str, WalkableMap] = {}
    manifest_cache: dict[str, list[dict]] = {}

    records = []
    fails = {"manifest_miss": 0, "target_miss": 0, "wm_miss": 0, "pose_miss": 0}
    same_room_fallback = 0

    for i, item in enumerate(items):
        scene = item["scene_id"]
        sel_id = item["selection_id"]

        if scene not in manifest_cache:
            try:
                manifest_cache[scene] = load_manifest(scene)
            except Exception as e:
                manifest_cache[scene] = []
                print(f"[warn] {sel_id} manifest read failed: {e}")
        entry, td = find_target(manifest_cache[scene], item)
        if entry is None or td is None:
            fails["target_miss"] += 1
            print(f"[miss] {sel_id}: target_details not found in manifest")
            continue
        pos = td.get("position")
        if not pos or len(pos) < 2:
            fails["target_miss"] += 1
            continue

        if scene not in wm_cache:
            wm = WalkableMap.load(scene)
            if wm is None:
                wm_cache[scene] = None
                fails["wm_miss"] += 1
                continue
            wm_cache[scene] = wm
        wm = wm_cache[scene]
        if wm is None:
            fails["wm_miss"] += 1
            continue

        target_xy = (float(pos[0]), float(pos[1]))
        target_room = td.get("room")

        # Snap target_xy to nearest walkable cell — many targets sit on
        # tabletops/counters that aren't directly walkable, which would make
        # geodesic_distance_2d return None for ALL start candidates. The
        # snapped point is what the agent actually navigates to.
        if not wm.is_walkable(*target_xy):
            for r in (1.5, 3.0, 5.0):
                snap = wm.nearby_walkable(*target_xy, radius_m=r)
                if snap:
                    target_xy = (float(snap[0]), float(snap[1]))
                    break

        # Per-item deterministic seed = base_seed ^ selection_id numeric suffix
        try:
            sid_num = int(sel_id.rsplit("_", 1)[-1])
        except ValueError:
            sid_num = hash(sel_id) & 0xFFFF
        item_seed = args.seed * 1000 + sid_num

        pose = sample_start_pose(wm, target_xy, target_room, seed=item_seed)
        if pose is None:
            # Retry once with a wider distance window for tight scenes that
            # can't satisfy the default [5, 10] m geodesic constraint.
            pose = sample_start_pose(wm, target_xy, target_room, seed=item_seed,
                                     min_dist=3.0, max_dist=12.0)
            if pose is not None:
                pose["wide_window_fallback"] = True
        if pose is None:
            fails["pose_miss"] += 1
            print(f"[miss] {sel_id}: no valid start pose (tried [5,10] and [3,12])")
            continue
        if not pose["different_room"]:
            same_room_fallback += 1

        rec = {
            "selection_id": sel_id,
            "scene_id": scene,
            "target_category": item["target_category"],
            "target_object_id": item.get("target_representative"),
            "target_position": [round(float(pos[0]), 4),
                                round(float(pos[1]), 4),
                                round(float(pos[2] if len(pos) > 2 else 0.0), 4)],
            "target_room": target_room,
            **pose,
        }
        records.append(rec)

        if (i + 1) % 25 == 0 or i + 1 == len(items):
            print(f"[episodes] {i+1}/{len(items)} ok={len(records)} "
                  f"same_room_fallback={same_room_fallback} fails={fails}")

    with args.out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Summary
    geodist = [r["geodesic_to_target"] for r in records]
    diff_room = sum(1 for r in records if r["different_room"])
    print(f"\n[episodes] DONE")
    print(f"  total ok:               {len(records)}/{len(items)}")
    print(f"  different_room:         {diff_room}/{len(records)}")
    print(f"  same_room_fallback:     {same_room_fallback}")
    print(f"  geodesic min/mean/max:  {min(geodist):.2f} / "
          f"{sum(geodist)/len(geodist):.2f} / {max(geodist):.2f}" if geodist else "")
    print(f"  fails: {fails}")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
