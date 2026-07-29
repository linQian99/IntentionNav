#!/usr/bin/env python3
"""Download fixed IntentionNav release components from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path


REPO_ID = "Anonymous260726/IntentionNav"

METADATA_PATTERNS = [
    "README.md",
    "croissant.json",
    "selected_500_intents.jsonl",
    "episodes.jsonl",
    "kujiale_*/*.json",
    "reference_results/MANIFEST.json",
    "reference_results/reproducibility/*",
    "reference_results/evaluation_code/*",
]
RENDER_PATTERNS = ["kujiale_*/photos/*.png"]
LOG_PATTERNS = [
    "reference_results/logs/*",
]
ARTIFACT_PATTERNS = [
    "reference_results/episode_artifacts/*",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the fixed IntentionNav benchmark and optional evaluation payloads."
    )
    parser.add_argument(
        "component",
        choices=("benchmark", "logs", "artifacts", "all"),
        help=(
            "benchmark: fixed data and metadata; logs: benchmark plus the 6,000 "
            "record archive; artifacts: benchmark plus multi-GB visual artifacts; "
            "all: complete dataset repository"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/benchmark",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required; run: pip install huggingface_hub"
        ) from exc

    patterns: list[str] | None
    if args.component == "benchmark":
        patterns = METADATA_PATTERNS + RENDER_PATTERNS
    elif args.component == "logs":
        patterns = METADATA_PATTERNS + LOG_PATTERNS
    elif args.component == "artifacts":
        patterns = METADATA_PATTERNS + ARTIFACT_PATTERNS
    else:
        patterns = None

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.component} from {REPO_ID} to {output}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=output,
        allow_patterns=patterns,
    )
    print(f"Ready: {output}")


if __name__ == "__main__":
    main()
