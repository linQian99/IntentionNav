#!/usr/bin/env bash
# Validate annotations and build the final IntentionNav benchmark JSON.
#
# Usage:
#   bash scripts/build_benchmark.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE_SUMMARY="${SCENE_SUMMARY:-/mnt/ssd0/Lin4T/datasets/vlntube/SceneSummary/kujiale_scene_summary}"
SPLITS="${SPLITS:-$REPO_ROOT/splits/scene_splits.json}"
CAPTURE_DIR="${CAPTURE_DIR:-$REPO_ROOT/work_dirs}"
INTENTS_DIR="${INTENTS_DIR:-$REPO_ROOT/work_dir_phase2}"
BENCHMARK_DIR="${BENCHMARK_DIR:-$REPO_ROOT/benchmark}"
OUTPUT="${OUTPUT:-$BENCHMARK_DIR/intentionnav_benchmark.json}"
ISSUES_OUTPUT="${ISSUES_OUTPUT:-$BENCHMARK_DIR/issues.jsonl}"
PYTHON="${PYTHON:-python3}"

cd "$REPO_ROOT"
mkdir -p "$BENCHMARK_DIR"

"$PYTHON" -m intentionnav.data_collection.validate_and_build \
    --intents-dir "$INTENTS_DIR" \
    --scene-summary "$SCENE_SUMMARY" \
    --splits-file "$SPLITS" \
    --capture-dir "$CAPTURE_DIR" \
    --output "$OUTPUT" \
    --issues-output "$ISSUES_OUTPUT"

echo ""
echo "Benchmark: $OUTPUT"
echo "Issues:    $ISSUES_OUTPUT"
