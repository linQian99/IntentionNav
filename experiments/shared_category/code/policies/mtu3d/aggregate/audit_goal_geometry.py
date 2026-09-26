"""Audit centre-point versus target-instance-surface navigation metrics.

The released protocol measures planar distance to one object centre.  That is
fragile for large furniture because its centre may lie inside non-navigable
geometry.  This script leaves the official metric unchanged and reports the
parallel distance to the closest point on the target instance's XY bounding
box, using the frozen scene summary.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from functools import lru_cache
from pathlib import Path

from compute_metrics import protocol_trajectory


def distance_to_xy_aabb(
    point: tuple[float, float] | list[float],
    min_points: list[float],
    max_points: list[float],
) -> float:
    """Euclidean XY distance from a point to an axis-aligned object box."""
    x, y = float(point[0]), float(point[1])
    dx = max(float(min_points[0]) - x, 0.0, x - float(max_points[0]))
    dy = max(float(min_points[1]) - y, 0.0, y - float(max_points[1]))
    return math.hypot(dx, dy)


def normalize_room(room: object) -> str:
    """Normalize simulator/object-summary room keys without dropping index."""
    return re.sub(r"[^a-z0-9]+", "_", str(room or "").lower()).strip("_")


def audit_records(
    records_dir: Path,
    scene_summary_root: Path,
    radius_m: float,
) -> list[dict]:
    @lru_cache(maxsize=None)
    def scene_objects(scene_id: str) -> dict:
        path = scene_summary_root / scene_id / "object_dict.json"
        return json.loads(path.read_text(encoding="utf-8"))

    rows: list[dict] = []
    for path in sorted(records_dir.rglob("record.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        trajectory, _ = protocol_trajectory(record)
        positions = []
        position_rooms = []
        current_room = ""
        for event in trajectory:
            current_room = event.get("room") or current_room
            if event.get("position") and len(event["position"]) >= 2:
                positions.append(event["position"])
                position_rooms.append(current_room)
        meta = record.get("episode_meta") or {}
        target_position = meta.get("target_position")
        target_object_id = str(meta.get("target_object_id") or "")
        scene_id = str(record.get("scene_id") or meta.get("scene_id") or "")
        target_object = scene_objects(scene_id).get(target_object_id)
        if not positions or not target_position or target_object is None:
            continue

        centre_distances = [
            math.hypot(
                float(position[0]) - float(target_position[0]),
                float(position[1]) - float(target_position[1]),
            )
            for position in positions
        ]
        surface_distances = [
            distance_to_xy_aabb(
                position,
                target_object["min_points"],
                target_object["max_points"],
            )
            for position in positions
        ]
        target_room = normalize_room(
            target_object.get("room") or meta.get("target_room")
        )
        same_room = [
            bool(target_room and normalize_room(room) == target_room)
            for room in position_rooms
        ]
        surface_room_hits = [
            distance <= radius_m and room_hit
            for distance, room_hit in zip(surface_distances, same_room)
        ]
        stopped = int(any(
            str(event.get("action", "")).strip().upper() == "STOP"
            for event in trajectory
        ))
        centre_sr = int(stopped and centre_distances[-1] <= radius_m)
        surface_sr = int(stopped and surface_distances[-1] <= radius_m)
        rows.append({
            "selection_id": record.get("selection_id", ""),
            "scene_id": scene_id,
            "target_category": record.get("target_category", ""),
            "target_object_id": target_object_id,
            "stopped": stopped,
            "centre_final_m": round(centre_distances[-1], 4),
            "surface_final_m": round(surface_distances[-1], 4),
            "centre_min_m": round(min(centre_distances), 4),
            "surface_min_m": round(min(surface_distances), 4),
            "centre_SR": centre_sr,
            "surface_SR": surface_sr,
            "centre_OSR": int(min(centre_distances) <= radius_m),
            "surface_OSR": int(min(surface_distances) <= radius_m),
            "surface_same_room_SR": int(stopped and surface_room_hits[-1]),
            "surface_same_room_OSR": int(any(surface_room_hits)),
            "target_room": target_room,
            "final_room": normalize_room(position_rooms[-1]),
            "centre_false_surface_true": int(not centre_sr and surface_sr),
            "xy_extent_m": round(math.hypot(
                float(target_object["max_points"][0])
                - float(target_object["min_points"][0]),
                float(target_object["max_points"][1])
                - float(target_object["min_points"][1]),
            ), 4),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--scene-summary-root", type=Path, required=True)
    parser.add_argument("--radius-m", type=float, default=2.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = audit_records(
        args.records_dir, args.scene_summary_root, args.radius_m
    )
    if not rows:
        raise SystemExit("no auditable record.json files found")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    summary = {
        "n_records": n,
        "radius_m": args.radius_m,
        "centre_SR": sum(row["centre_SR"] for row in rows) / n,
        "surface_SR": sum(row["surface_SR"] for row in rows) / n,
        "centre_OSR": sum(row["centre_OSR"] for row in rows) / n,
        "surface_OSR": sum(row["surface_OSR"] for row in rows) / n,
        "surface_same_room_SR": sum(
            row["surface_same_room_SR"] for row in rows
        ) / n,
        "surface_same_room_OSR": sum(
            row["surface_same_room_OSR"] for row in rows
        ) / n,
        "n_centre_false_surface_true": sum(
            row["centre_false_surface_true"] for row in rows
        ),
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
