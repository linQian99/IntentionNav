"""Audit whether an episode's category names one unique scene instance.

The benchmark's primary metric uses one fixed ``target_object_id``. A plain
ObjectNav category cannot distinguish that object from another instance with
the same category. This audit reports such episodes before an explicit-target
calibration is interpreted as an oracle for target inference.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def load_episodes(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_episode(episode: dict, scene_summary_root: Path) -> dict:
    room_path = scene_summary_root / episode["scene_id"] / "room_dict.json"
    rooms = json.loads(room_path.read_text(encoding="utf-8"))
    object_ids = sorted({object_id for values in rooms.values() for object_id in values})
    category = str(episode["target_category"])
    pattern = re.compile(rf"^{re.escape(category)}_[0-9]+/Meshes$")
    matching_ids = [object_id for object_id in object_ids if pattern.match(object_id)]
    target_id = str(episode["target_object_id"])
    return {
        "selection_id": episode["selection_id"],
        "scene_id": episode["scene_id"],
        "target_category": category,
        "target_object_id": target_id,
        "category_instance_count": len(matching_ids),
        "category_unique": int(len(matching_ids) == 1),
        "target_id_present": int(target_id in matching_ids),
        "matching_object_ids": "|".join(matching_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes",
        type=Path,
        default=REPO / "eval/splits/episodes.jsonl",
    )
    parser.add_argument("--scene-summary-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        audit_episode(episode, args.scene_summary_root)
        for episode in load_episodes(args.episodes)
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    histogram = Counter(row["category_instance_count"] for row in rows)
    ambiguous = sum(not row["category_unique"] for row in rows)
    missing_target = sum(not row["target_id_present"] for row in rows)
    summary = {
        "n_episodes": len(rows),
        "n_category_ambiguous": ambiguous,
        "category_ambiguous_fraction": ambiguous / max(1, len(rows)),
        "n_target_id_missing": missing_target,
        "instance_count_histogram": dict(sorted(histogram.items())),
    }
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
