"""Locate archived experiment documents and verify their original bytes.

Use --resolve OLD_PATH for moved/deduplicated files; add --snapshot to read
the saved pre-cleanup version of a mutable current document.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "experiments/records.json"


def load() -> dict:
    catalog = json.loads(CATALOG.read_text())
    if catalog.get("layout_version") != 1:
        raise ValueError("Unsupported record catalog version")
    seen = set()
    for row in catalog["records"]:
        if row["original_path"] in seen:
            raise ValueError(f"Duplicate original path: {row['original_path']}")
        seen.add(row["original_path"])
        for key in ("original_path", "path"):
            path = Path(row[key])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Expected repository-relative path: {path}")
        if row["action"] not in {"move", "deduplicate", "snapshot"}:
            raise ValueError(f"Unknown action: {row['action']}")
    return catalog


def verify(catalog: dict) -> dict:
    if os.environ.get("INAV_LEGACY_EVAL_VIEW"):
        raise RuntimeError("Run --check on the host, outside experiments/run")
    hashes = {}
    for row in catalog["records"]:
        path = ROOT / row["path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected physical document: {path}")
        if path not in hashes:
            hashes[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes[path] != row["sha256"]:
            raise ValueError(f"Archived document changed: {path}")
        if row["action"] != "snapshot" and (ROOT / row["original_path"]).exists():
            raise ValueError(f"Old document path reappeared: {row['original_path']}")
    return dict(passed=True, records=len(catalog["records"]),
                distinct_documents=len(hashes), original_bytes_preserved=True)


def resolve(catalog: dict, name: str, *, snapshot: bool = False) -> Path:
    path = Path(os.path.abspath(name))
    for row in catalog["records"]:
        if path == ROOT / row["original_path"]:
            if row["action"] == "snapshot" and not snapshot:
                return path
            target = ROOT / row["path"]
            if not target.is_file():
                raise FileNotFoundError(target)
            return target
    if path.exists():
        return path
    raise FileNotFoundError(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--resolve")
    parser.add_argument("--snapshot", action="store_true")
    args = parser.parse_args()
    if args.snapshot and not args.resolve:
        parser.error("--snapshot requires --resolve")
    catalog = load()
    if args.check:
        print(json.dumps(verify(catalog)))
    elif args.resolve:
        print(resolve(catalog, args.resolve, snapshot=args.snapshot))
    else:
        for row in catalog["records"]:
            print(f"{row['action']:12} {row['original_path']} -> {row['path']}")


if __name__ == "__main__":
    main()
