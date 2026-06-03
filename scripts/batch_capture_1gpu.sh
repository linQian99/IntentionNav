#!/usr/bin/env bash
# Single-GPU IntentionNav capture loop.
#
# Usage:
#   bash scripts/batch_capture_1gpu.sh
#   bash scripts/batch_capture_1gpu.sh 0 20
#   GPU_ID=1 BATCH_SIZE=30 bash scripts/batch_capture_1gpu.sh

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE_SUMMARY="${SCENE_SUMMARY:-$REPO_ROOT/data/SceneSummary/kujiale_scene_summary}"
USD_ROOT="${USD_ROOT:-$REPO_ROOT/data/VLNVerse_scene}"
METAROOT="${METAROOT:-$REPO_ROOT/data/SceneMeta/metadata_train}"
SPLITS="${SPLITS:-$REPO_ROOT/splits/scene_splits.json}"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/work_dirs}"
LOG_DIR="${LOG_DIR:-$WORK_DIR/batch_logs}"
ISAACSIM_ROOT="${ISAACSIM_ROOT:-$REPO_ROOT/external/isaacsim}"
GOODNAV_BIN="${GOODNAV_BIN:-}"
PYTHON="${PYTHON:-python3}"

GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-20}"
ROUND_TIMEOUT="${ROUND_TIMEOUT:-1800}"
NUM_VIEWS="${NUM_VIEWS:-1}"
RESOLUTION="${RESOLUTION:-1024}"
START_IDX="${1:-0}"
END_IDX="${2:-999999}"

EXIT_ALL_DONE=10
EXIT_BATCH_DONE=11
CHILD_PID=""

cleanup() {
    echo ""
    echo "=== Caught signal, killing children ==="
    [ -n "$CHILD_PID" ] && kill -9 -- "-$CHILD_PID" 2>/dev/null
    pkill -9 -P $$ 2>/dev/null
    pkill -9 -f "capture_surfaces.py" 2>/dev/null
    exit 130
}
trap cleanup INT TERM

cd "$REPO_ROOT"
mkdir -p "$LOG_DIR"

mapfile -t ALL_SCENES < <("$PYTHON" - <<PY
import json
import os

splits_path = "$SPLITS"
metaroot = "$METAROOT"
with open(splits_path, "r", encoding="utf-8") as f:
    splits = json.load(f)
for scene_id in sorted(splits["trainval"]):
    if os.path.exists(os.path.join(metaroot, scene_id, "freemap.npy")):
        print(scene_id)
PY
)

TOTAL="${#ALL_SCENES[@]}"
SCENES=("${ALL_SCENES[@]:$START_IDX:$((END_IDX - START_IDX))}")
N="${#SCENES[@]}"

echo "Total trainval scenes: $TOTAL"
echo "Slice [$START_IDX, $END_IDX): $N scenes"
echo "GPU: $GPU_ID, batch size: $BATCH_SIZE"
echo "Output: $WORK_DIR"
echo ""

echo "=== Phase 0: surface_targets.json ==="
for sid in "${SCENES[@]}"; do
    if [ ! -f "$WORK_DIR/$sid/surface_targets.json" ]; then
        "$PYTHON" -m intentionnav.data_collection.surface_finder \
            --scene-summary "$SCENE_SUMMARY" \
            --output-dir "$WORK_DIR" \
            --scene "$sid" >/dev/null 2>&1
    fi
done

SCENE_LIST="$LOG_DIR/gpu${GPU_ID}_scenes.txt"
: > "$SCENE_LIST"
for sid in "${SCENES[@]}"; do
    echo "$sid" >> "$SCENE_LIST"
done

MASTER_LOG="$LOG_DIR/gpu${GPU_ID}_master.log"
: > "$MASTER_LOG"

echo "Scene list: $SCENE_LIST"
echo "Tail: tail -f $MASTER_LOG"
echo ""

round=0
while true; do
    round=$((round + 1))
    batch_log="$LOG_DIR/gpu${GPU_ID}_round${round}.log"
    echo "=== Round $round ===" | tee -a "$MASTER_LOG"

    export REPO_ROOT SCENE_SUMMARY USD_ROOT METAROOT WORK_DIR SCENE_LIST
    export ISAACSIM_ROOT GOODNAV_BIN PYTHON GPU_ID BATCH_SIZE ROUND_TIMEOUT
    export NUM_VIEWS RESOLUTION

    setsid bash -c '
        set -u
        [ -d "$GOODNAV_BIN" ] && export PATH="$GOODNAV_BIN:$PATH"
        export CUDA_VISIBLE_DEVICES="$GPU_ID"
        [ -f "$ISAACSIM_ROOT/setup_conda_env.sh" ] && source "$ISAACSIM_ROOT/setup_conda_env.sh" 2>/dev/null
        cd "$REPO_ROOT"
        exec timeout "$ROUND_TIMEOUT" "$PYTHON" -m intentionnav.data_collection.capture_surfaces \
            --usd-root "$USD_ROOT" \
            --metaroot "$METAROOT" \
            --surface-targets-dir "$WORK_DIR" \
            --scene-summary "$SCENE_SUMMARY" \
            --output-dir "$WORK_DIR" \
            --scene-list "$SCENE_LIST" \
            --batch-size "$BATCH_SIZE" \
            --num-views "$NUM_VIEWS" \
            --resolution "$RESOLUTION"
    ' > "$batch_log" 2>&1 &

    CHILD_PID=$!
    wait "$CHILD_PID"
    ec=$?
    CHILD_PID=""
    echo "Round $round exit code: $ec" | tee -a "$MASTER_LOG"

    if [ "$ec" -eq "$EXIT_ALL_DONE" ]; then
        echo "All done" | tee -a "$MASTER_LOG"
        break
    fi
    if [ "$ec" -eq "$EXIT_BATCH_DONE" ]; then
        echo "Batch done, restarting" | tee -a "$MASTER_LOG"
        sleep 3
        continue
    fi

    echo "Unexpected exit ($ec); checking pending" | tee -a "$MASTER_LOG"
    sleep 10
    pending=0
    while IFS= read -r sid; do
        [ -n "$sid" ] && [ ! -f "$WORK_DIR/$sid/capture_manifest.json" ] && pending=$((pending + 1))
    done < "$SCENE_LIST"
    if [ "$pending" -eq 0 ]; then
        echo "No pending, exiting" | tee -a "$MASTER_LOG"
        break
    fi
    if [ "$round" -ge 30 ]; then
        echo "Too many rounds, giving up with $pending pending" | tee -a "$MASTER_LOG"
        break
    fi
done

echo ""
echo "=== Final Summary ==="
ok=$(find "$WORK_DIR" -name capture_manifest.json | wc -l)
photos=$(find "$WORK_DIR" -name "*.png" -path "*/surface_photos/*" | wc -l)
echo "Scenes captured: $ok"
echo "Total photos: $photos"
