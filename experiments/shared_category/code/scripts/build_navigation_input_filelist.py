#!/usr/bin/env python3
"""Resolve every file consumed by a frozen navigation cohort.

The resulting newline-delimited absolute paths are fed to ``sha256sum`` before
and after each run.  USD crates can hide nested asset references, so selected
scene packages are intentionally covered in full rather than approximated from
the ASCII root layer.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SELECTION_RE = re.compile(r"^SEL_[0-9]+$")


def jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--episodes-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--usd-root", type=Path, required=True)
    parser.add_argument("--metaroot", type=Path, required=True)
    parser.add_argument("--scene-summary-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection_ids = [
        line.strip()
        for line in args.selection_file.read_text(encoding="utf-8").splitlines()
        if SELECTION_RE.fullmatch(line.strip())
    ]
    if not selection_ids or len(selection_ids) != len(set(selection_ids)):
        raise ValueError("selection must contain unique canonical IDs")
    episodes = {row["selection_id"]: row for row in jsonl(args.episodes_jsonl)}
    missing = sorted(set(selection_ids) - set(episodes))
    if missing:
        raise ValueError(f"selection IDs missing from episodes: {missing[:5]}")
    selected_scenes = sorted({
        str(episodes[selection_id]["scene_id"])
        for selection_id in selection_ids
    })
    dataset_scenes = sorted({
        str(row["scene_id"]) for row in jsonl(args.dataset_jsonl)
    })

    paths: set[Path] = set()
    for scene_id in selected_scenes:
        scene_package = args.usd_root / scene_id
        if not scene_package.is_dir():
            raise FileNotFoundError(f"missing USD scene package: {scene_package}")
        paths.update(path for path in scene_package.rglob("*") if path.is_file())
        paths.add(args.metaroot / scene_id / "freemap.npy")
        paths.add(args.metaroot / scene_id / "room_region.json")
        paths.add(args.dataset_root / scene_id / "manifest.json")

    # compute_metrics validates/provenances object dictionaries for the full
    # fixed dataset, not only the selected record subset.
    for scene_id in dataset_scenes:
        paths.add(args.scene_summary_root / scene_id / "object_dict.json")

    missing_paths = sorted(str(path) for path in paths if not path.is_file())
    if missing_paths:
        raise FileNotFoundError(
            "resolved input files are missing:\n" + "\n".join(missing_paths[:20])
        )
    resolved = sorted(str(path.resolve()) for path in paths)
    if any("\n" in path for path in resolved):
        raise ValueError("input filenames containing newlines are unsupported")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text("\n".join(resolved) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "selection_count": len(selection_ids),
        "selected_scene_count": len(selected_scenes),
        "dataset_scene_count": len(dataset_scenes),
        "input_file_count": len(resolved),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
