#!/usr/bin/env bash
# Multi-GPU IntentionNav capture loop. Set NUM_GPUS for non-2GPU machines.
#
# Usage:
#   bash scripts/batch_capture_2gpu.sh
#   bash scripts/batch_capture_2gpu.sh 0 20
#   NUM_GPUS=4 BATCH_SIZE=30 bash scripts/batch_capture_2gpu.sh

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE_SUMMARY="${SCENE_SUMMARY:-/mnt/ssd0/Lin4T/datasets/vlntube/SceneSummary/kujiale_scene_summary}"
USD_ROOT="${USD_ROOT:-/mnt/ssd0/Lin4T/datasets/vlntube/TataServices}"
METAROOT="${METAROOT:-/mnt/ssd0/Lin4T/datasets/vlntube/TaTaMeta/metadata_train}"
SPLITS="${SPLITS:-$REPO_ROOT/splits/scene_splits.json}"
WORK_DIR="${WORK_DIR:-$REPO_ROOT/work_dirs}"
LOG_DIR="${LOG_DIR:-$WORK_DIR/batch_logs}"
ISAACSIM_ROOT="${ISAACSIM_ROOT:-/mnt/ssd0/Lin4T/Project/ISAACSIM_ROOT}"
GOODNAV_BIN="${GOODNAV_BIN:-/mnt/ssd0/Lin4T/envs/goodnav/bin}"
PYTHON="${PYTHON:-python3}"

NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-20}"
ROUND_TIMEOUT="${ROUND_TIMEOUT:-1800}"
NUM_VIEWS="${NUM_VIEWS:-1}"
RESOLUTION="${RESOLUTION:-1024}"
START_IDX="${1:-0}"
END_IDX="${2:-999999}"

EXIT_ALL_DONE=10
EXIT_BATCH_DONE=11
CHILD_PIDS=()

cleanup() {
    echo ""
    echo "=== Caught signal, killing children ==="
    for pid in "${CHILD_PIDS[@]}"; do
        [ -n "$pid" ] && kill -9 -- "-$pid" 2>/dev/null
    done
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
echo "GPUs: $NUM_GPUS, batch size: $BATCH_SIZE"
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

for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    : > "$LOG_DIR/gpu${gpu_id}_scenes.txt"
done
for idx in "${!SCENES[@]}"; do
    gpu_id=$((idx % NUM_GPUS))
    echo "${SCENES[$idx]}" >> "$LOG_DIR/gpu${gpu_id}_scenes.txt"
done
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    n=$(wc -l < "$LOG_DIR/gpu${gpu_id}_scenes.txt")
    echo "GPU $gpu_id: $n scenes"
done

run_gpu() {
    local gpu_id="$1"
    local scene_list="$LOG_DIR/gpu${gpu_id}_scenes.txt"
    local master_log="$LOG_DIR/gpu${gpu_id}_master.log"
    : > "$master_log"

    echo "[GPU $gpu_id] Starting loop" | tee -a "$master_log"
    local round=0
    while true; do
        round=$((round + 1))
        local batch_log="$LOG_DIR/gpu${gpu_id}_round${round}.log"
        echo "[GPU $gpu_id] === Round $round ===" | tee -a "$master_log"

        export REPO_ROOT SCENE_SUMMARY USD_ROOT METAROOT WORK_DIR
        export ISAACSIM_ROOT GOODNAV_BIN PYTHON BATCH_SIZE ROUND_TIMEOUT
        export NUM_VIEWS RESOLUTION
        SCENE_LIST="$scene_list" GPU_ID="$gpu_id" setsid bash -c '
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

        local child_pid=$!
        CHILD_PIDS[$gpu_id]="$child_pid"
        wait "$child_pid"
        local ec=$?
        CHILD_PIDS[$gpu_id]=""
        echo "[GPU $gpu_id] Round $round exit code: $ec" | tee -a "$master_log"

        if [ "$ec" -eq "$EXIT_ALL_DONE" ]; then
            echo "[GPU $gpu_id] All done" | tee -a "$master_log"
            break
        fi
        if [ "$ec" -eq "$EXIT_BATCH_DONE" ]; then
            echo "[GPU $gpu_id] Batch done, restarting" | tee -a "$master_log"
            sleep 3
            continue
        fi

        echo "[GPU $gpu_id] Unexpected exit ($ec); checking pending" | tee -a "$master_log"
        sleep 10
        local pending=0
        while IFS= read -r sid; do
            [ -n "$sid" ] && [ ! -f "$WORK_DIR/$sid/capture_manifest.json" ] && pending=$((pending + 1))
        done < "$scene_list"
        if [ "$pending" -eq 0 ]; then
            echo "[GPU $gpu_id] No pending, exiting" | tee -a "$master_log"
            break
        fi
        if [ "$round" -ge 30 ]; then
            echo "[GPU $gpu_id] Too many rounds, giving up with $pending pending" | tee -a "$master_log"
            break
        fi
    done

    echo "[GPU $gpu_id] DONE" | tee -a "$master_log"
}

echo ""
echo "=== Launching $NUM_GPUS GPU workers ==="
PIDS=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
    run_gpu "$gpu_id" &
    PIDS+=($!)
    sleep 2
done

echo "Worker PIDs: ${PIDS[*]}"
echo "Tail: tail -f $LOG_DIR/gpu0_master.log"
wait "${PIDS[@]}"

echo ""
echo "=== Final Summary ==="
ok=$(find "$WORK_DIR" -name capture_manifest.json | wc -l)
photos=$(find "$WORK_DIR" -name "*.png" -path "*/surface_photos/*" | wc -l)
echo "Scenes captured: $ok"
echo "Total photos: $photos"
