#!/usr/bin/env python3
"""Verify the checked-in experiment code against its release source manifests."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / "experiments"
    total = 0
    for study in ("shared_category", "hosted_repeats"):
        directory = root / study
        manifest = json.loads((directory / "SOURCE_MANIFEST.json").read_text())
        seen = set()
        for entry in manifest["files"]:
            relative = entry["path"]
            path = (directory / relative).resolve()
            if not path.is_relative_to(directory.resolve()) or relative in seen:
                raise ValueError(f"Invalid or duplicate manifest path: {relative}")
            seen.add(relative)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != entry["sha256"]:
                raise ValueError(f"Source checksum mismatch: {study}/{relative}")
        expected = {str(path.relative_to(directory)) for path in (directory / "code").rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts}
        expected.add("recompute.py")
        if seen != expected:
            raise ValueError(f"Source inventory mismatch: {study}")
        print(f"{study}: verified {len(seen)} source files")
        total += len(seen)
    print(f"Verified {total} files; no simulator or model calls were made.")


if __name__ == "__main__":
    main()
