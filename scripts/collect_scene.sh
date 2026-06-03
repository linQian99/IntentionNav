#!/usr/bin/env bash
# Run IntentionNav data collection for one scene.
#
# Usage:
#   bash scripts/collect_scene.sh kujiale_0005
#
# Override paths with env vars when needed:
#   SCENE_SUMMARY=/path/to/kujiale_scene_summary
#   USD_ROOT=/path/to/VLNVerse_scene
#   METAROOT=/path/to/metadata_train
#   WORK_DIR=/path/to/output

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE="${1:-}"

if [ -z "$SCENE" ]; then
    echo "Usage: bash scripts/collect_scene.sh <scene_id>" >&2
    exit 2
fi

SCENE_SUMMARY="${SCENE_SUMMARY:-$REPO_ROOT/data/SceneSummary/kujiale_scene_summary}"
USD_ROOT="${USD_ROOT:-$REPO_ROOT/data/VLNVerse_scene}"
METAROOT="${METAROOT:-$REPO_ROOT/data/SceneMeta/metadata_train}"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/work_dirs}"
PYTHON="${PYTHON:-python3}"
NUM_VIEWS="${NUM_VIEWS:-1}"
RESOLUTION="${RESOLUTION:-1024}"

cd "$REPO_ROOT"
mkdir -p "$WORK_DIR"

echo "=== Phase 0: surface targets ($SCENE) ==="
"$PYTHON" -m intentionnav.data_collection.surface_finder \
    --scene-summary "$SCENE_SUMMARY" \
    --output-dir "$WORK_DIR" \
    --scene "$SCENE"

echo ""
echo "=== Phase 1: Isaac Sim capture ($SCENE) ==="
"$PYTHON" -m intentionnav.data_collection.capture_surfaces \
    --usd-root "$USD_ROOT" \
    --metaroot "$METAROOT" \
    --surface-targets-dir "$WORK_DIR" \
    --scene-summary "$SCENE_SUMMARY" \
    --output-dir "$WORK_DIR" \
    --scene "$SCENE" \
    --num-views "$NUM_VIEWS" \
    --resolution "$RESOLUTION"

echo ""
echo "Done: $WORK_DIR/$SCENE/capture_manifest.json"
