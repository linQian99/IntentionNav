#!/usr/bin/env bash
# Active ObjectNav evaluation driver. Runs the reference engine, random, and
# frontier tiers in Isaac Sim, then aggregates their records.
#
# Usage:
#   bash scripts/run_active_eval.sh                       # --test 3 (default), reference engine, all 3 VLMs
#   bash scripts/run_active_eval.sh --test 20             # 20 items × 4 styles = 80 episodes
#   bash scripts/run_active_eval.sh --full                # all 500 items
#   bash scripts/run_active_eval.sh --tier all            # reference engine + random + fbe
#   bash scripts/run_active_eval.sh --model gpt5_4        # only 1 VLM
#   bash scripts/run_active_eval.sh --model qwen3_6_plus  # only 1 VLM
#
# Flags:
#   --test N        run on first N items (default 3)
#   --full          run on all 500 items (overrides --test)
#   --tier T        T ∈ {vlm_engine, random, fbe, all}  (default vlm_engine)
#   --model M       M ∈ {gpt5_4, gemini_3_1_flash, qwen3_6_plus, all}  (default all; only affects reference engine)
#   --step-cap N    max steps per episode (default 30; was 50, dropped after
#                   --test 4 showed only 25% of episodes used >30 steps and
#                   most of those hit step_limit anyway)
#   --procs-per-gpu N  parallel Isaac Sim processes per GPU (default 1).
#                  Each proc uses ~4GB VRAM. API I/O is bottleneck; more procs = more concurrent
#                  VLM calls. Lower to 1-2 if VRAM contention.
#   --skip-aggregate skip the aggregate step (just run agents)
#   --force         re-run already-saved episodes
#   --help          show this
#
# GPU behavior:
#   - If CUDA_VISIBLE_DEVICES is set, that pin is honored.
#   - Otherwise nvidia-smi enumerates all GPUs.
#   - For each (tier, model) combo, the SEL list is split round-robin across all
#     GPUs and N parallel Isaac Sim processes run concurrently. So 2 GPUs ≈ 2×
#     speedup whether you have 1 model or 3.
#   - For --model all (3 models), models run sequentially per slot (Isaac Sim
#     singleton per process); within each slot, SELs split across GPUs.
#   - Per-process logs go to results/active_logs/.

set -e

# ----------- Cleanup on exit / interrupt -----------
# Each Isaac Sim worker uses a unique OMNI_USER_PATH=/tmp/ov_user_$$_<label>_<idx>
# to isolate Carb/Omni cache + lock files (multi-instance launches deadlock
# when they share ~/.local/share/ov). Sweep them on exit so we don't leave
# tmp dirs behind across runs.
cleanup_ov_dirs() {
    rm -rf "/tmp/ov_user_${$}_"* 2>/dev/null || true
    rm -rf "/tmp/eval_queue_${$}_"* 2>/dev/null || true
}
trap cleanup_ov_dirs EXIT INT TERM

# ----------- Defaults -----------
TEST=3
FULL=0
TIER="vlm_engine"
MODEL="all"
STEP_CAP=30
SKIP_AGG=0
FORCE=0
# One SimulationApp per GPU is the conservative default; increase only after
# checking the memory and driver limits of the target machine.
PROCS_PER_GPU=1

# ----------- Parse args -----------
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)         TEST="$2"; shift 2 ;;
        --full)         FULL=1; shift ;;
        --tier)         TIER="$2"; shift 2 ;;
        --model)        MODEL="$2"; shift 2 ;;
        --step-cap)     STEP_CAP="$2"; shift 2 ;;
        --procs-per-gpu) PROCS_PER_GPU="$2"; shift 2 ;;
        --skip-aggregate) SKIP_AGG=1; shift ;;
        --force)        FORCE=1; shift ;;
        --help|-h)      sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)              echo "unknown arg: $1"; exit 1 ;;
    esac
done

# Derive the repository root from this script location.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# ----------- Env setup -----------
# Activate goodnav conda env
if [[ -z "$CONDA_DEFAULT_ENV" || "$CONDA_DEFAULT_ENV" != "goodnav" ]]; then
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
        source "/opt/conda/etc/profile.d/conda.sh"
    fi
    conda activate goodnav 2>/dev/null || true
fi

# Source Isaac Sim when a standalone installation is provided. Pip-based
# installations may leave ISAACSIM_ROOT unset.
if [[ -n "${ISAACSIM_ROOT:-}" && -f "$ISAACSIM_ROOT/setup_conda_env.sh" ]]; then
    source "$ISAACSIM_ROOT/setup_conda_env.sh"
fi

# Use the active environment unless explicitly overridden.
PY="${PY:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "python executable not found: $PY" >&2
    exit 2
fi

DATASET_ROOT="${INTENTIONNAV_DATASET_ROOT:-$REPO/data/benchmark}"
if [[ ! -f "$DATASET_ROOT/selected_500_intents.jsonl" ]]; then
    echo "benchmark data not found under $DATASET_ROOT" >&2
    echo "run: python scripts/download_release.py benchmark" >&2
    exit 2
fi
if [[ -z "${INTENTIONNAV_API_KEY:-}${PP_API_KEY:-}" ]]; then
    echo "INTENTIONNAV_API_KEY is required for hosted-VLM evaluation" >&2
    exit 2
fi
if [[ -z "${INTENTIONNAV_API_BASE_URL:-}${PP_API_BASE_URL:-}" ]]; then
    echo "INTENTIONNAV_API_BASE_URL is required for hosted-VLM evaluation" >&2
    exit 2
fi

# ----------- Build SEL list (always concrete; split across GPUs later) -----------
PILOT_SELS_DEFAULT="SEL_003,SEL_099,SEL_171,SEL_195,SEL_004,SEL_101,SEL_102,SEL_186,SEL_005,SEL_021,SEL_036,SEL_098,SEL_006,SEL_140,SEL_177,SEL_178,SEL_007,SEL_180,SEL_182,SEL_184"

if [[ -n "$EVAL_SELS_OVERRIDE" ]]; then
    # External driver (e.g. run_distributed.sh) supplies the exact SEL list.
    # Skips --full / --test / pilot defaults.
    SELS="$EVAL_SELS_OVERRIDE"
    n_override=$(echo "$SELS" | tr ',' '\n' | wc -l)
    SCOPE_DESC="EVAL_SELS_OVERRIDE ($n_override SELs)"
elif [[ "$FULL" == "1" ]]; then
    SELS=$("$PY" -c \
        'import json,sys; print(",".join(json.loads(line)["selection_id"] for line in open(sys.argv[1]) if line.strip()))' \
        "$DATASET_ROOT/selected_500_intents.jsonl")
    SCOPE_DESC="full 500 items"
else
    if [[ "$TEST" -le 20 ]]; then
        SELS=$(echo "$PILOT_SELS_DEFAULT" | cut -d',' -f1-"$TEST")
    else
        SELS=$("$PY" -c \
            'import json,sys; items=[json.loads(line) for line in open(sys.argv[1]) if line.strip()]; print(",".join(item["selection_id"] for item in items[:int(sys.argv[2])]))' \
            "$DATASET_ROOT/selected_500_intents.jsonl" "$TEST")
    fi
    SCOPE_DESC="--test $TEST items"
fi

# Split a CSV SEL list into NUM_GPUS round-robin chunks, return chunk i (0-indexed)
sel_chunk() {
    local sels="$1"; local nchunks="$2"; local idx="$3"
    echo "$sels" | tr ',' '\n' | awk -v n="$nchunks" -v i="$idx" 'NR%n == i%n' | paste -sd,
}

# ----------- Resolve tier list -----------
case $TIER in
    vlm_engine|random|fbe) TIERS=("$TIER") ;;
    vlm)                    TIERS=(vlm_engine) ;;  # backward-compatible alias
    all)                    TIERS=(vlm_engine random fbe) ;;
    *) echo "unknown tier: $TIER"; exit 1 ;;
esac

# ----------- Resolve model list (only matters for the reference engine) -----------
case $MODEL in
    all) MODELS=(gpt5_4 gemini_3_1_flash qwen3_6_plus) ;;
    *)   MODELS=("$MODEL") ;;
esac

# ----------- Force flag -----------
if [[ "$FORCE" == "1" ]]; then
    FORCE_FLAG="--force"
else
    FORCE_FLAG=""
fi

# ----------- GPU detection (auto-max unless user pinned CUDA_VISIBLE_DEVICES) -----------
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    AVAIL_GPUS="$CUDA_VISIBLE_DEVICES"
    GPU_SOURCE="user CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null 2>&1; then
    N=$(nvidia-smi -L 2>/dev/null | wc -l)
    if (( N > 0 )); then
        AVAIL_GPUS=$(seq 0 $((N-1)) | paste -sd,)
        GPU_SOURCE="nvidia-smi auto-detect"
    else
        AVAIL_GPUS="0"
        GPU_SOURCE="fallback (nvidia-smi gave 0)"
    fi
else
    AVAIL_GPUS="0"
    GPU_SOURCE="no nvidia-smi"
fi
GPU_ARR=(${AVAIL_GPUS//,/ })
NUM_GPUS=${#GPU_ARR[@]}

# ----------- Per-GPU effective slot count (Xorg-aware) -----------
# RTX shader cache deadlock means each GPU has a hard concurrent-Kit cap.
# A display-bound GPU often supports fewer concurrent SimulationApp workers:
#   - GPU with Xorg attached  → 1 concurrent SimulationApp
#   - GPU headless            → 2 concurrent
# Auto-detect Xorg per-GPU and adjust GPU_SLOTS[gpu]. Workers beyond the
# cap would deadlock at TLAS, so we never launch them.
declare -A GPU_SLOTS
declare -A GPU_HAS_XORG
for g in "${GPU_ARR[@]}"; do
    GPU_HAS_XORG[$g]=0
    GPU_SLOTS[$g]=$PROCS_PER_GPU
done
if command -v nvidia-smi >/dev/null 2>&1 && [[ "${IGNORE_XORG_CAP:-0}" != "1" ]]; then
    # Xorg shows up on EVERY GPU it can render to, including a tiny ~4 MiB
    # fallback context on otherwise-headless GPUs. We only want to flag a
    # GPU as "display-bound" when Xorg is using REAL memory there
    # (>10 MiB threshold separates the fallback from the active server).
    # Override: set IGNORE_XORG_CAP=1 to skip this auto-throttle and trust
    # the user-supplied --procs-per-gpu — risky on Xorg-bound GPUs (historical
    # TLAS init deadlock) but useful for benchmarking max throughput.
    for g in "${GPU_ARR[@]}"; do
        # Extract just the numeric MiB count of any /usr/lib/xorg/Xorg line
        # for this GPU. `grep -oP 'Xorg\s+\K\d+(?=MiB)'` would need PCRE;
        # use awk to stay portable.
        xorg_mem=$(nvidia-smi -i "$g" 2>/dev/null \
            | awk '/Xorg/ { for (i=1; i<=NF; i++) if ($i ~ /MiB$/) { gsub(/MiB/, "", $i); print $i; exit } }')
        if [[ -n "$xorg_mem" && "$xorg_mem" -gt 10 ]]; then
            GPU_HAS_XORG[$g]=1
            if (( GPU_SLOTS[$g] > 1 )); then GPU_SLOTS[$g]=1; fi
        fi
    done
fi
# Build SLOT_GPU_MAP via interleaved round-robin (alternate GPUs while quotas
# remain). For caps {GPU0:1, GPU1:2}: SLOT_GPU_MAP = [0, 1, 1].
SLOT_GPU_MAP=()
declare -A USED; for g in "${GPU_ARR[@]}"; do USED[$g]=0; done
TOTAL_QUOTA=0
for g in "${GPU_ARR[@]}"; do TOTAL_QUOTA=$(( TOTAL_QUOTA + GPU_SLOTS[$g] )); done
i=0
while (( ${#SLOT_GPU_MAP[@]} < TOTAL_QUOTA )); do
    g="${GPU_ARR[$((i % NUM_GPUS))]}"
    if (( USED[$g] < GPU_SLOTS[$g] )); then
        SLOT_GPU_MAP+=("$g")
        USED[$g]=$(( USED[$g] + 1 ))
    fi
    i=$(( i + 1 ))
    if (( i > 1000 )); then break; fi   # safety
done
TOTAL_PROCS=${#SLOT_GPU_MAP[@]}

# ----------- Banner -----------
echo "=================================================================="
echo "  Active ObjectNav Eval"
echo "  scope:      $SCOPE_DESC × 4 styles"
echo "  tiers:      ${TIERS[*]}"
echo "  vlm models: ${MODELS[*]}"
echo "  step-cap:   $STEP_CAP"
echo "  GPUs:       $AVAIL_GPUS  ($GPU_SOURCE)"
# Print per-GPU slot allocation (Xorg-aware).
SLOT_DESC=""
for g in "${GPU_ARR[@]}"; do
    tag=""
    if (( GPU_HAS_XORG[$g] )); then tag=" (Xorg→cap 1)"; fi
    SLOT_DESC+=" GPU${g}=${GPU_SLOTS[$g]}${tag},"
done
SLOT_DESC=${SLOT_DESC#  }; SLOT_DESC=${SLOT_DESC%,}
echo "  procs/GPU:  requested=$PROCS_PER_GPU → effective slots:$SLOT_DESC"
echo "  parallel:   $TOTAL_PROCS workers (Isaac Sim shader-cache cap)"
echo "  python:     $PY"

# ----------- Output dir (timestamped if not overridden) -----------
# EVAL_OUT_DIR is read by eval/agents/common.py:EPISODES_OUT and by all
# downstream scripts that import it. Set per-run to keep experiments
# from contaminating each other.
if [[ -z "$EVAL_OUT_DIR" ]]; then
    EVAL_OUT_DIR="results/eval_out_$(date +%Y%m%d_%H%M%S)"
fi
export EVAL_OUT_DIR
echo "  output:     $EVAL_OUT_DIR"
echo "=================================================================="

ts() { date +%T; }
mkdir -p results/active_logs "$EVAL_OUT_DIR"

# ----------- Run agents -----------
# Memory-recycle defaults. Isaac Sim leaks ~150-300 MB of GPU memory each
# time it loads/unloads a USD; on a long run a worker eventually OOMs
# (~50-80 unique scenes for our Kujiale apartments). The agent honors
# `--max-scenes N` and exits cleanly at that boundary; the driver below
# relaunches the same chunk and the agent's "skip if record.json exists"
# logic resumes seamlessly. Tunable via env vars.
MAX_SCENES_PER_WORKER="${MAX_SCENES_PER_WORKER:-30}"
MAX_RECYCLE_ATTEMPTS="${MAX_RECYCLE_ATTEMPTS:-20}"

# Count how many record.json files exist for a SEL under <tier>_<model>.
# Returns 4 (one per style) when fully done. Used by the recycler to
# decide if a chunk still has pending work.
records_for_sel() {
    local sel="$1" dir_name="$2"
    find "$EVAL_OUT_DIR" -path "*/${dir_name}/${sel}/*/record.json" 2>/dev/null | wc -l
}

# Filter chunk to SELs that don't yet have all 4 styles' records.
pending_sels() {
    local chunk="$1" dir_name="$2"
    local out=""
    for sel in $(echo "$chunk" | tr ',' ' '); do
        local n; n=$(records_for_sel "$sel" "$dir_name")
        if (( n < 4 )); then out+="${sel},"; fi
    done
    echo "${out%,}"
}

# Resolve the on-disk dir name from (tier, model_key) — matches
# common.episode_dir's naming. The reference engine writes under the bare
# model name; legacy vlm/oracle/blind tiers use a tier prefix.
dir_name_for() {
    local tier="$1" model="$2"
    case "$tier" in
        vlm|oracle|blind) echo "${tier}_${model}" ;;
        vlm_engine)       echo "$model" ;;
        random)           echo "random_walk" ;;
        fbe)              echo "frontier_walk" ;;
        *)                echo "$model" ;;
    esac
}

# ---------------- Dynamic work-queue ----------------
# All workers share a single pending queue, atomically claim batches of up
# to MAX_SCENES_PER_WORKER SELs at a time, run them, then loop. A "fast"
# worker naturally claims more SELs than a slow one — no idle workers
# waiting for a hard SEL on another GPU to finish.
#
# Layout (per batch, per (tier×model)):
#   /tmp/eval_queue_$$_<label>/
#       pending/<sel>      ← waiting to be claimed (atomic mv → inflight/)
#       inflight/<sel>.<wid> ← claimed by worker <wid>, run in progress
#       done/<sel>          ← all 4 styles' record.json present

# Initialize the queue from current SELS list, accounting for already-done
# SELs from prior runs (their record.jsons exist → straight to done).
queue_init() {
    local queue_dir="$1" dir_name="$2"
    rm -rf "$queue_dir"
    mkdir -p "$queue_dir/pending" "$queue_dir/inflight" "$queue_dir/done"
    for sel in $(echo "$SELS" | tr ',' ' '); do
        local n; n=$(records_for_sel "$sel" "$dir_name")
        if (( n >= 4 )); then
            touch "$queue_dir/done/$sel"
        else
            touch "$queue_dir/pending/$sel"
        fi
    done
}

# Atomically claim up to N pending SELs for this worker. Uses `mv` (atomic
# on POSIX same-FS): the SEL marker moves from pending/ to inflight/. If
# two workers race on the same file, exactly one mv succeeds; the other
# silently moves to the next.
#
# `n_workers` is passed so we can cap the per-claim batch at FAIR-SHARE.
# Without this, with small total_pending (e.g. --test 4 → 4 SELs) and
# default MAX_SCENES_PER_WORKER=30, the first worker greedily grabs all
# pending SELs before the second worker spins up — second worker exits
# idle. Fix: limit each claim to ceil(remaining / n_workers).
queue_claim() {
    local queue_dir="$1" n_max="$2" wid="$3" n_workers="${4:-1}"
    # Compute fair share; ceiling division so 5/2=3 (not 2 → strands 1 SEL).
    local n_remaining; n_remaining=$(ls "$queue_dir/pending" 2>/dev/null | wc -l)
    if (( n_remaining == 0 )); then echo ""; return; fi
    local fair=$(( (n_remaining + n_workers - 1) / n_workers ))
    if (( fair < 1 )); then fair=1; fi
    local actual_max=$n_max
    if (( fair < actual_max )); then actual_max=$fair; fi

    local batch="" n=0
    for path in "$queue_dir/pending"/*; do
        [ -f "$path" ] || continue
        if (( n >= actual_max )); then break; fi
        local sel; sel=$(basename "$path")
        if mv "$path" "$queue_dir/inflight/${sel}.${wid}" 2>/dev/null; then
            batch+="${sel},"
            n=$(( n + 1 ))
        fi
    done
    echo "${batch%,}"
}

# After agent returns, route each claimed SEL to done/ if all 4 records
# now exist, else back to pending/ (so another worker can retry — handles
# partial completion, OOM, agent crash, --max-scenes mid-batch).
queue_finalize() {
    local queue_dir="$1" batch="$2" wid="$3" dir_name="$4"
    for sel in $(echo "$batch" | tr ',' ' '); do
        local marker="$queue_dir/inflight/${sel}.${wid}"
        local n; n=$(records_for_sel "$sel" "$dir_name")
        if (( n >= 4 )); then
            mv "$marker" "$queue_dir/done/$sel" 2>/dev/null || true
        else
            mv "$marker" "$queue_dir/pending/$sel" 2>/dev/null || true
        fi
    done
}

# Worker loop: claim batch → run agent → finalize → repeat until queue empty.
# Each iteration's batch is up to MAX_SCENES_PER_WORKER SELs (memory recycle).
recycle_worker() {
    local queue_dir="$1" wid="$2" gpu="$3" ov="$4" log="$5"
    local agent_script="$6" model_arg="$7" dir_name="$8" n_workers="${9:-1}"
    for ((a=1; a<=MAX_RECYCLE_ATTEMPTS; a++)); do
        local batch; batch=$(queue_claim "$queue_dir" "$MAX_SCENES_PER_WORKER" "$wid" "$n_workers")
        if [[ -z "$batch" ]]; then
            echo "[$(ts)]   ✓ worker $wid (GPU $gpu) queue drained after $((a-1)) batch(es)" >> "$log"
            return 0
        fi
        local n_in; n_in=$(echo "$batch" | tr ',' '\n' | wc -l)
        echo "[$(ts)]   [batch $a] worker $wid GPU $gpu claimed $n_in SELs: $batch" >> "$log"
        CUDA_VISIBLE_DEVICES="$gpu" \
        OMNI_USER_PATH="$ov" \
        __GL_SHADER_DISK_CACHE_PATH="$ov/glcache" \
        CUDA_CACHE_PATH="$ov/cudacache" \
            $PY "$agent_script" \
                $model_arg \
                --style all \
                --step-cap "$STEP_CAP" \
                --only "$batch" \
                --max-scenes "$MAX_SCENES_PER_WORKER" \
                $FORCE_FLAG \
                >> "$log" 2>&1
        queue_finalize "$queue_dir" "$batch" "$wid" "$dir_name"
    done
    echo "[$(ts)]   ✗ worker $wid (GPU $gpu) hit max-attempts" >> "$log"
    return 1
}

run_split_across_gpus() {
    local agent_script="$1"   # path to eval/agents/<agent>.py
    local model_arg="$2"      # "" for random/fbe; "--model X" for the engine
    local label="$3"          # human label for logs
    local tier="$4"           # vlm_engine/random/fbe (for dir-name resolution)
    local model_key="$5"      # gpt5_4 / random_walk / etc.
    local pids=()

    local n_sels; n_sels=$(echo "$SELS" | tr ',' '\n' | sed '/^$/d' | wc -l)

    local actual_procs=$TOTAL_PROCS
    if (( n_sels < actual_procs )); then actual_procs=$n_sels; fi

    local STAGGER_PER_GPU_S="${STAGGER_PER_GPU_S:-8}"
    declare -A LAST_LAUNCH
    local dir_name; dir_name=$(dir_name_for "$tier" "$model_key")

    # Build single shared queue for this batch. All workers pull from it
    # — fast workers grab more SELs, slow ones grab fewer, no waiting.
    local QUEUE_DIR="/tmp/eval_queue_${$}_${label}"
    queue_init "$QUEUE_DIR" "$dir_name"
    local n_pending; n_pending=$(ls "$QUEUE_DIR/pending" 2>/dev/null | wc -l)
    local n_done_already; n_done_already=$(ls "$QUEUE_DIR/done" 2>/dev/null | wc -l)
    echo "[$(ts)]   queue: $n_pending pending, $n_done_already already-done"

    for ((p=0; p<actual_procs; p++)); do
        local GPU="${SLOT_GPU_MAP[$p]}"
        local WID="p${p}_gpu${GPU}"

        # Per-GPU stagger (initial launch only).
        local now last elapsed need
        now=$(date +%s)
        last="${LAST_LAUNCH[$GPU]:-0}"
        if (( last > 0 )); then
            elapsed=$(( now - last ))
            need=$(( STAGGER_PER_GPU_S - elapsed ))
            if (( need > 0 )); then sleep "$need"; fi
        fi

        local LOG="results/active_logs/${label}_${WID}.log"
        local OV_DIR="/tmp/ov_user_${$}_${label}_${p}"
        echo "[$(ts)]   worker $WID launched (batch ≤$MAX_SCENES_PER_WORKER, ov=$OV_DIR)  →  $LOG"
        (
            recycle_worker "$QUEUE_DIR" "$WID" "$GPU" "$OV_DIR" "$LOG" \
                "$agent_script" "$model_arg" "$dir_name" "$actual_procs"
        ) &
        pids+=($!)
        LAST_LAUNCH[$GPU]=$(date +%s)
    done
    # Tolerate individual worker failures: don't kill the batch if one
    # proc crashes. Track exit codes; report any non-zero so user can
    # check the corresponding log.
    local rc fails=0
    for p in "${pids[@]}"; do
        if ! wait "$p"; then
            rc=$?
            echo "[$(ts)]   ⚠ worker pid=$p exited rc=$rc (check its log)"
            fails=$(( fails + 1 ))
        fi
    done
    if (( fails > 0 )); then
        echo "[$(ts)]   batch finished with $fails/${#pids[@]} worker failures"
    fi
    # Final queue stats + cleanup
    local n_done_final; n_done_final=$(ls "$QUEUE_DIR/done" 2>/dev/null | wc -l)
    local n_left; n_left=$(ls "$QUEUE_DIR/pending" 2>/dev/null | wc -l)
    local n_stuck; n_stuck=$(ls "$QUEUE_DIR/inflight" 2>/dev/null | wc -l)
    echo "[$(ts)]   queue final: $n_done_final done, $n_left pending, $n_stuck inflight (stuck)"
    rm -rf "$QUEUE_DIR" 2>/dev/null
}

for T in "${TIERS[@]}"; do
    case $T in
        vlm_engine)
            for M in "${MODELS[@]}"; do
                echo "[$(ts)] === reference engine × $M  (split across $NUM_GPUS GPU) ==="
                run_split_across_gpus eval/agents/agent_vlm_engine.py "--model $M" "vlm_engine_${M}" "vlm_engine" "$M"
            done
            echo "[$(ts)] reference engine tier done"
            ;;
        random)
            echo "[$(ts)] === random walk + final-frame VLM  (split across $NUM_GPUS GPU) ==="
            run_split_across_gpus eval/agents/agent_random.py "" "random" "random" "random_walk"
            ;;
        fbe)
            echo "[$(ts)] === fbe (frontier exploration) + final-frame VLM  (split across $NUM_GPUS GPU) ==="
            run_split_across_gpus eval/agents/agent_fbe.py "" "fbe" "fbe" "frontier_walk"
            ;;
    esac
done

# ----------- Aggregate -----------
if [[ "$SKIP_AGG" != "1" ]]; then
    echo "[$(ts)] === aggregate ==="
    OUT_TAG="active"
    if [[ "$FULL" != "1" ]]; then OUT_TAG="active_test${TEST}"; fi
    $PY eval/aggregate/compute_metrics.py --out "results/$OUT_TAG"
    echo
    echo "=================================================================="
    echo "  Done. Metrics in results/$OUT_TAG/"
    echo "=================================================================="
fi
