#!/usr/bin/env bash
# Batch Gemini annotation for captured IntentionNav scenes.
#
# Usage:
#   bash scripts/batch_annotate.sh
#   bash scripts/batch_annotate.sh 0 20
#   MODEL=models/gemini-2.5-flash WORKERS=4 bash scripts/batch_annotate.sh

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCENE_SUMMARY="${SCENE_SUMMARY:-$REPO_ROOT/data/SceneSummary/kujiale_scene_summary}"
SPLITS="${SPLITS:-$REPO_ROOT/splits/scene_splits.json}"
CAPTURE_DIR="${CAPTURE_DIR:-$REPO_ROOT/work_dirs}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/work_dir_phase2}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
PYTHON="${PYTHON:-python3}"
MODEL="${MODEL:-models/gemini-3-flash-preview}"
WORKERS="${WORKERS:-8}"

START_IDX="${1:-0}"
END_IDX="${2:-999999}"

CHILD_PID=""
TAIL_PID=""
cleanup() {
    echo ""
    echo "=== Caught signal, killing children ==="
    [ -n "$TAIL_PID" ] && kill "$TAIL_PID" 2>/dev/null
    [ -n "$CHILD_PID" ] && kill -9 -- "-$CHILD_PID" 2>/dev/null
    pkill -9 -P $$ 2>/dev/null
    pkill -9 -f "generate_intents" 2>/dev/null
    exit 130
}
trap cleanup INT TERM

cd "$REPO_ROOT"
mkdir -p "$LOG_DIR"

MASTER_LOG="$LOG_DIR/annotate_master.log"
PROGRESS_LOG="$LOG_DIR/progress.log"
SCENE_LIST="$LOG_DIR/annotate_scenes.txt"
: > "$MASTER_LOG"
: > "$PROGRESS_LOG"

log() { echo "$@" | tee -a "$MASTER_LOG"; }

log "================= IntentionNav Phase 2: Batch Annotate ================="
log "Model:    $MODEL"
log "Workers:  $WORKERS"
log "Capture:  $CAPTURE_DIR"
log "Output:   $OUTPUT_DIR"
log ""

if [ -z "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ]; then
    log "[ERROR] neither GEMINI_API_KEY nor GOOGLE_API_KEY is set"
    exit 1
fi

export PYTHONPATH=""
export PYTHONNOUSERSITE=1

log "=== Scene slicing ==="
mapfile -t ALL_SCENES < <("$PYTHON" - <<PY
import json
import os

splits_path = "$SPLITS"
capture_dir = "$CAPTURE_DIR"
with open(splits_path, "r", encoding="utf-8") as f:
    splits = json.load(f)
for scene_id in sorted(splits["trainval"]):
    if os.path.exists(os.path.join(capture_dir, scene_id, "capture_manifest.json")):
        print(scene_id)
PY
)

TOTAL="${#ALL_SCENES[@]}"
SCENES=("${ALL_SCENES[@]:$START_IDX:$((END_IDX - START_IDX))}")
log "Trainval with captures: $TOTAL"
log "Slice [$START_IDX, $END_IDX): ${#SCENES[@]} scenes"

log ""
log "=== Pre-flight status ==="
"$PYTHON" -m intentionnav.data_collection.check_status \
    --capture-dir "$CAPTURE_DIR" \
    --output-dir "$OUTPUT_DIR" 2>&1 | tee -a "$MASTER_LOG" || true

ALLOWED_LIST="$LOG_DIR/allowed_scenes.tmp"
: > "$ALLOWED_LIST"
for scene_id in "${SCENES[@]}"; do
    echo "$scene_id" >> "$ALLOWED_LIST"
done

mapfile -t INCOMPLETE_SCENES < <("$PYTHON" - <<PY
from intentionnav.data_collection.check_status import check_scenes

allowed = set(open("$ALLOWED_LIST", encoding="utf-8").read().split())
for row in check_scenes("$CAPTURE_DIR", "$OUTPUT_DIR"):
    if row["status"] != "complete" and row["scene_id"] in allowed:
        print(row["scene_id"])
PY
)
rm -f "$ALLOWED_LIST"

SCENES=("${INCOMPLETE_SCENES[@]}")
N="${#SCENES[@]}"
log ""
log "Scenes to process: $N"

if [ "$N" -eq 0 ]; then
    log "Nothing to do."
    exit 0
fi

: > "$SCENE_LIST"
for scene_id in "${SCENES[@]}"; do
    echo "$scene_id" >> "$SCENE_LIST"
done

START_TS="$(date +%s)"
setsid bash -c '
    set -u
    cd "$1"
    scene_list="$2"
    total="$3"
    idx=0
    while IFS= read -r scene_id; do
        [ -z "$scene_id" ] && continue
        idx=$((idx + 1))
        echo ""
        echo "====== [$idx/$total] $scene_id ======"
        scene_start=$(date +%s)
        "$4" -m intentionnav.data_collection.generate_intents \
            --surface-targets-dir "$5" \
            --scene-summary "$6" \
            --photos-dir "$5" \
            --output-dir "$7" \
            --scene "$scene_id" \
            --model "$8" \
            --workers "$9" || echo "[WARN] $scene_id failed, continuing"
        scene_end=$(date +%s)
        n_ann=0
        if [ -f "$7/$scene_id/intent_annotations.json" ]; then
            n_ann=$("$4" -c "import json; print(len(json.load(open('$7/$scene_id/intent_annotations.json'))))" 2>/dev/null || echo 0)
        fi
        printf '[%(%F %T)T] [%d/%d] %-18s %3d entries  (%ds)\n' -1 "$idx" "$total" "$scene_id" "$n_ann" "$((scene_end - scene_start))" >> "${10}"
    done < "$scene_list"
' _ "$REPO_ROOT" "$SCENE_LIST" "$N" "$PYTHON" "$CAPTURE_DIR" "$SCENE_SUMMARY" "$OUTPUT_DIR" "$MODEL" "$WORKERS" "$PROGRESS_LOG" >> "$MASTER_LOG" 2>&1 &
CHILD_PID=$!

log ""
log "PID: $CHILD_PID"
log "Progress file: $PROGRESS_LOG"
log "Tail: tail -f $MASTER_LOG"

tail -f "$MASTER_LOG" &
TAIL_PID=$!
wait "$CHILD_PID"
ec=$?
[ -n "$TAIL_PID" ] && kill "$TAIL_PID" 2>/dev/null
CHILD_PID=""
TAIL_PID=""

END_TS="$(date +%s)"
ELAPSED="$((END_TS - START_TS))"

log ""
log "================= Final Summary ================="
scenes_done=$(find "$OUTPUT_DIR" -maxdepth 2 -name intent_annotations.json | wc -l)
total_entries=$(find "$OUTPUT_DIR" -maxdepth 2 -name intent_annotations.json \
    -exec "$PYTHON" -c "import json,sys; print(sum(len(json.load(open(f))) for f in sys.argv[1:]))" {} + 2>/dev/null || echo 0)
log "Scenes annotated: $scenes_done"
log "Total entries:    $total_entries"
log "Elapsed:          ${ELAPSED}s ($((ELAPSED / 60))m $((ELAPSED % 60))s)"
log "Exit code:        $ec"
exit "$ec"
