#!/usr/bin/env bash
# Recompute the released metrics from the fixed 6,000-record archive.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="${INTENTIONNAV_DATASET_ROOT:-$REPO_ROOT/data/benchmark}"
SCENE_SUMMARY="${INTENTIONNAV_SCENE_SUMMARY:-$REPO_ROOT/data/SceneSummary/kujiale_scene_summary}"
ARCHIVE="${REFERENCE_LOG_ARCHIVE:-$DATASET_ROOT/reference_results/logs/reference_logs_6000.tar.zst}"
RECORD_ROOT="${REFERENCE_RECORD_ROOT:-$REPO_ROOT/results/final_6000_merged_20260507}"
METRICS_OUT="${METRICS_OUT:-$REPO_ROOT/results/recomputed_metrics}"
PY="${PY:-python3}"

if [[ ! -f "$DATASET_ROOT/selected_500_intents.jsonl" ]]; then
    echo "fixed benchmark not found under $DATASET_ROOT" >&2
    echo "run: python scripts/download_release.py logs" >&2
    exit 2
fi
if [[ ! -f "$ARCHIVE" ]]; then
    echo "reference log archive not found: $ARCHIVE" >&2
    echo "run: python scripts/download_release.py logs" >&2
    exit 2
fi
if [[ ! -f "$SCENE_SUMMARY/kujiale_0005/object_dict.json" ]]; then
    echo "SceneSummary not found under $SCENE_SUMMARY" >&2
    echo "download Eyz/SceneSummary and set INTENTIONNAV_SCENE_SUMMARY" >&2
    exit 2
fi

if [[ ! -d "$RECORD_ROOT" ]]; then
    echo "Extracting released records..."
    tar --zstd -xf "$ARCHIVE" -C "$REPO_ROOT"
fi

export INTENTIONNAV_DATASET_ROOT="$DATASET_ROOT"
export INTENTIONNAV_SCENE_SUMMARY="$SCENE_SUMMARY"
export EVAL_OUT_DIR="$RECORD_ROOT"

"$PY" "$REPO_ROOT/eval/aggregate/compute_metrics.py" --out "$METRICS_OUT"
echo "Metrics written to $METRICS_OUT"
