#!/usr/bin/env bash
set -euo pipefail

# Derived input-only launcher; no automatic performance acceptance.
repo_dir=.
check_only=0
check_resources=0
if [[ "$#" -eq 1 && "$1" == "--check" ]]; then
    check_only=1
elif [[ "$#" -eq 1 && "$1" == "--check-resources" ]]; then
    check_resources=1
elif [[ "$#" -ne 0 ]]; then
    printf 'Only --check or --check-resources is supported\n' >&2; exit 2
fi
shared_query_inputs="${INAV_SHARED_QUERY_INPUTS:?Set frozen shared policy directory}"
reviewed_data_root=$(realpath "${INAV_REVIEWED_DATA_ROOT:?Set the frozen reviewed dataset directory}")
[[ -f "$reviewed_data_root/output_sha256.json" ]] || { printf 'Missing reviewed dataset closure\n' >&2; exit 2; }
shared_input_mode="${INAV_SHARED_INPUT_MODE:?Set explicit or implicit_category}"
shared_query_inputs=$(realpath "$shared_query_inputs")
case "$shared_input_mode" in explicit|implicit_category) ;; *) exit 2 ;; esac
[[ "${INAV_PHYSICAL_GPU:-0}" == 0 && "${INAV_RESUME_RUN:-0}" == 0 ]] || exit 2
shared_query_args=(--shared-query-inputs "$shared_query_inputs" --shared-input-mode "$shared_input_mode")
shared_input_tool="$repo_dir/scripts/shared_navigation_run_inputs.py"
python_bin=/path/to/workspace/envs/goodnav/bin/python
yolo_world_checkpoint=/path/to/workspace/models/YOLOWorld/yolov8s-worldv2.pt
launcher_path=$(realpath "${BASH_SOURCE[0]}")
agent_source_dir="${INAV_AGENT_SOURCE_DIR:-$repo_dir/eval/agents}"
input_resolver="${INAV_INPUT_RESOLVER:-$repo_dir/scripts/build_navigation_input_filelist.py}"
budgeted_view_arm="${INAV_BUDGETED_VIEW_SCAN:-0}"
late_room_profile="${INAV_LATE_ROOM_RESCUE_PROFILE:-0}"
reachable_target_region_arm="${INAV_REACHABLE_TARGET_REGION:-0}"
bounded_commitment_arm="${INAV_BOUNDED_COMMITMENT:-0}"
tentative_probe_arm="${INAV_TENTATIVE_MEMORY:-0}"
pose_evidence_graph_arm="${INAV_POSE_EVIDENCE_GRAPH:-0}"
target_lock_controller_arm="${INAV_TARGET_LOCK_CONTROLLER:-0}"
target_standoff_commitment_arm="${INAV_TARGET_STANDOFF_COMMITMENT:-0}"
target_bbox_recenter_arm="${INAV_TARGET_BBOX_RECENTER:-0}"
persistent_single_encoder_veto_arm="${INAV_PERSISTENT_SINGLE_ENCODER_VETO:-0}"
terminal_evidence_quorum_arm="${INAV_TERMINAL_EVIDENCE_QUORUM:-0}"
support_inspection_arm="${INAV_SUPPORT_INSPECTION:-0}"
carrier_active_verify_arm="${INAV_CARRIER_ACTIVE_VERIFY:-0}"
visual_target_descriptors_arm="${INAV_VISUAL_TARGET_DESCRIPTORS:-0}"
belief_reperception_arm="${INAV_BELIEF_REPERCEPTION:-0}"
belief_coverage_preserving_arm="${INAV_BELIEF_COVERAGE_PRESERVING:-0}"
dual_detector_fusion_arm="${INAV_DUAL_DETECTOR_FUSION:-0}"
dual_detector_stop_support_arm="${INAV_DUAL_DETECTOR_STOP_SUPPORT:-0}"
target_visual_memory_arm="${INAV_TARGET_VISUAL_MEMORY_FRONTIER:-0}"
geometry_safe_frontier_arm="${INAV_GEOMETRY_SAFE_FRONTIER:-0}"
no_admit_arm="${VM_NO_ADMIT:-0}"
physical_gpu="${INAV_PHYSICAL_GPU:-0}"
resume_run="${INAV_RESUME_RUN:-0}"
execution_seed="${INAV_EXECUTION_SEED:-20260825}"
render_antialiasing_mode="${INAV_RENDER_ANTIALIASING_MODE:-3}"
render_ticks="${INAV_RENDER_TICKS:-${ISS_RENDER_TICKS:-8}}"
agent_source_dir=$(realpath "$agent_source_dir")
case "$agent_source_dir" in
    "$repo_dir"/eval/agents*|\
    "$repo_dir"/eval_*_frozen_*/agents|\
    "$repo_dir"/results/navigation_active_perception_20260825/*/source_snapshot/eval/agents) ;;
    *) printf 'Agent source must be a frozen directory under eval/: %s\n' \
           "$agent_source_dir" >&2; exit 24 ;;
esac
agent_eval_root=$(dirname "$agent_source_dir")
simulator_source_dir="$agent_eval_root/simulator"
aggregate_source_dir="$agent_eval_root/aggregate"
vocab_source_file="$agent_eval_root/vocab/category_synonyms.yaml"
if [[ ! -f "$simulator_source_dir/iss_env.py" ]]; then
    simulator_source_dir="$repo_dir/eval/simulator"
fi
if [[ ! -f "$vocab_source_file" ]]; then
    vocab_source_file="$repo_dir/eval/vocab/category_synonyms.yaml"
fi
if [[ ! -f "$aggregate_source_dir/compute_metrics.py" ]]; then
    aggregate_source_dir="$repo_dir/eval/aggregate"
fi
if [[ ! -f "$agent_source_dir/agent_intentionnav.py" ]]; then
    printf 'Missing agent entry point: %s\n' \
        "$agent_source_dir/agent_intentionnav.py" >&2
    exit 25
fi
: "${INAV_RUN_NAME:?Set a unique immutable INAV_RUN_NAME}"
run_name="$INAV_RUN_NAME"
run_root="$repo_dir/results/navigation_active_perception_20260825/$run_name"
if [[ -e "$run_root" ]]; then
    if [[ "$resume_run" != "1" ]]; then
        printf 'Refusing to overwrite existing run: %s\n' "$run_root" >&2
        exit 20
    fi
    if [[ ! -d "$run_root/episodes" || -f "$run_root/.RUN_SUCCESS" ]]; then
        printf 'Refusing to resume invalid or completed run: %s\n' "$run_root" >&2
        exit 20
    fi
elif [[ "$resume_run" == "1" ]]; then
    printf 'Refusing resume because run directory is missing: %s\n' "$run_root" >&2
    exit 20
fi
# Diagnostic cohort: 11 reached-without-memory failures, two unconfirmed-memory
# failures, two prior false-stop cases, and three prior successful controls.
selection_file="${INAV_SELECTION_FILE:-}"
if [[ -n "$selection_file" ]]; then
    selection_file=$(realpath "$selection_file")
    case "$selection_file" in
        "$repo_dir"/*) ;;
        *) printf 'Selection file must be inside the repository: %s\n' \
               "$selection_file" >&2; exit 22 ;;
    esac
    selection_count=$(rg -c '^SEL_[0-9]+$' "$selection_file")
    unique_selection_count=$(rg '^SEL_[0-9]+$' "$selection_file" \
        | sort -u | wc -l)
    if [[ "$selection_count" -eq 0 \
          || "$selection_count" -ne "$unique_selection_count" ]]; then
        printf 'Selection file has zero or duplicate canonical IDs: %s\n' \
            "$selection_file" >&2
        exit 23
    fi
    probe_sels=$(rg '^SEL_[0-9]+$' "$selection_file" | paste -sd,)
    expected_records="${INAV_EXPECTED_RECORDS:-$selection_count}"
else
    probe_sels="${INAV_ONLY_SELECTIONS:-SEL_104,SEL_193,SEL_207,SEL_215,SEL_231,SEL_237,SEL_262,SEL_279,SEL_338,SEL_364,SEL_507,SEL_246,SEL_429,SEL_217,SEL_295,SEL_008,SEL_078,SEL_081}"
    expected_records="${INAV_EXPECTED_RECORDS:-18}"
fi

# Make every non-arm policy setting independent of the parent shell.  The
# launcher captures its six routing inputs above, then clears all experiment
# knobs before exporting the frozen protocol below.
while IFS= read -r environment_name; do
    case "$environment_name" in
        INAV_*|VM_*|AUTO_REORIENT*|ANGULAR_SPREAD|FALLBACK_*|MEMORY_HINT_*|REPLAN_ENABLE|ROBUSTNESS_*|STRICT_VLM_FAILURE|PP_API_*|ISS_RENDER_TICKS|HF_HOME|HF_HUB_CACHE|HUGGINGFACE_HUB_CACHE|TRANSFORMERS_CACHE|XDG_CACHE_HOME)
            unset "$environment_name" ;;
    esac
done < <(compgen -e)
case "$budgeted_view_arm" in
    0|1) ;;
    *) printf 'INAV_BUDGETED_VIEW_SCAN must be 0 or 1\n' >&2; exit 27 ;;
esac
case "$late_room_profile" in
    0|1) ;;
    *) printf 'INAV_LATE_ROOM_RESCUE_PROFILE must be 0 or 1\n' >&2; exit 27 ;;
esac
case "$reachable_target_region_arm" in
    0|1) ;;
    *) printf 'INAV_REACHABLE_TARGET_REGION must be 0 or 1\n' >&2; exit 27 ;;
esac
case "$bounded_commitment_arm" in
    0|1) ;;
    *) printf 'INAV_BOUNDED_COMMITMENT must be 0 or 1\n' >&2; exit 27 ;;
esac
case "$tentative_probe_arm" in
    0|1) ;;
    *) printf 'INAV_TENTATIVE_MEMORY must be 0 or 1\n' >&2; exit 27 ;;
esac
case "$pose_evidence_graph_arm" in
    0|1) ;;
    *) printf 'INAV_POSE_EVIDENCE_GRAPH must be 0 or 1\n' >&2; exit 27 ;;
esac
case "$target_lock_controller_arm" in
    0|1) ;;
    *) printf 'INAV_TARGET_LOCK_CONTROLLER must be 0 or 1\n' >&2; exit 27 ;;
esac
case "$target_standoff_commitment_arm" in
    0|1) ;;
    *) printf 'INAV_TARGET_STANDOFF_COMMITMENT must be 0 or 1\n' >&2; exit 27 ;;
esac
case "$target_bbox_recenter_arm" in
    0|1) ;;
    *) printf 'INAV_TARGET_BBOX_RECENTER must be 0 or 1\n' >&2; exit 27 ;;
esac
case "$persistent_single_encoder_veto_arm" in
    0|1) ;;
    *) printf 'INAV_PERSISTENT_SINGLE_ENCODER_VETO must be 0 or 1\n' >&2; exit 27 ;;
esac
case "$terminal_evidence_quorum_arm" in
    0|1) ;;
    *) printf 'INAV_TERMINAL_EVIDENCE_QUORUM must be 0 or 1\n' >&2; exit 27 ;;
esac
case "$support_inspection_arm" in
    0|1) ;;
    *) printf 'INAV_SUPPORT_INSPECTION must be 0 or 1\n' >&2; exit 27 ;;
esac
case "$carrier_active_verify_arm" in
    0|1) ;;
    *) printf 'INAV_CARRIER_ACTIVE_VERIFY must be 0 or 1\n' >&2; exit 27 ;;
esac
case "$visual_target_descriptors_arm" in
    0|1) ;;
    *) printf 'INAV_VISUAL_TARGET_DESCRIPTORS must be 0 or 1\n' >&2; exit 27 ;;
esac
case "$belief_reperception_arm" in
    0|1) ;;
    *) printf 'INAV_BELIEF_REPERCEPTION must be 0 or 1\n' >&2; exit 27 ;;
esac
case "$belief_coverage_preserving_arm" in
    0|1) ;;
    *) printf 'INAV_BELIEF_COVERAGE_PRESERVING must be 0 or 1\n' >&2; exit 27 ;;
esac
case "$dual_detector_fusion_arm" in
    0|1) ;;
    *) printf 'INAV_DUAL_DETECTOR_FUSION must be 0 or 1\n' >&2; exit 27 ;;
esac
case "$dual_detector_stop_support_arm" in
    0|1) ;;
    *) printf 'INAV_DUAL_DETECTOR_STOP_SUPPORT must be 0 or 1\n' >&2; exit 27 ;;
esac
case "$target_visual_memory_arm" in
    0|1) ;;
    *) printf 'INAV_TARGET_VISUAL_MEMORY_FRONTIER must be 0 or 1\n' >&2; exit 27 ;;
esac
case "$geometry_safe_frontier_arm" in
    0|1) ;;
    *) printf 'INAV_GEOMETRY_SAFE_FRONTIER must be 0 or 1\n' >&2; exit 27 ;;
esac
case "$no_admit_arm" in
    0|1) ;;
    *) printf 'VM_NO_ADMIT must be 0 or 1\n' >&2; exit 27 ;;
esac
if [[ "$dual_detector_stop_support_arm" == "1" \
      && "$dual_detector_fusion_arm" != "1" ]]; then
    printf 'Dual-detector STOP support requires detector fusion\n' >&2
    exit 27
fi
if [[ "$belief_coverage_preserving_arm" == "1" \
      && "$belief_reperception_arm" != "1" ]]; then
    printf 'Coverage-preserving belief requires belief re-perception\n' >&2
    exit 27
fi
case "$physical_gpu" in
    0|1) ;;
    *) printf 'INAV_PHYSICAL_GPU must be 0 or 1\n' >&2; exit 27 ;;
esac
case "$resume_run" in
    0|1) ;;
    *) printf 'INAV_RESUME_RUN must be 0 or 1\n' >&2; exit 27 ;;
esac
if [[ ! "$execution_seed" =~ ^[0-9]+$ ]]; then
    printf 'INAV_EXECUTION_SEED must be a non-negative integer\n' >&2
    exit 27
fi
case "$render_antialiasing_mode" in
    0|1|2|3|4) ;;
    *) printf 'INAV_RENDER_ANTIALIASING_MODE must be 0..4\n' >&2; exit 27 ;;
esac
if [[ ! "$render_ticks" =~ ^[0-9]+$ ]] \
      || [[ "$render_ticks" -lt 1 || "$render_ticks" -gt 64 ]]; then
    printf 'INAV_RENDER_TICKS must be an integer in [1, 64]\n' >&2
    exit 27
fi
unset PP_API_KEY OPENAI_API_KEY GEMINI_API_KEY GOOGLE_API_KEY \
    ANTHROPIC_API_KEY QWEN_API_KEY DASHSCOPE_API_KEY || true

export ISAACSIM_ROOT=/path/to/isaacsim
export INTENTIONNAV_PYTHON="$python_bin"
export INTENTIONNAV_USD_ROOT=/path/to/workspace/datasets/vlntube/TataServices
export INTENTIONNAV_METAROOT=/path/to/workspace/datasets/vlntube/TaTaMeta/metadata_train
export INTENTIONNAV_SCENE_SUMMARY=/path/to/workspace/datasets/vlntube/SceneSummary/kujiale_scene_summary
export INTENTIONNAV_DATASET_ROOT="$reviewed_data_root"
export INTENTIONNAV_DATASET_JSONL="$INTENTIONNAV_DATASET_ROOT/selected_500_intents.jsonl"
export INTENTIONNAV_EPISODES_JSONL="$reviewed_data_root/episodes_explicit_category.jsonl"
export INTENTIONNAV_CATEGORY_GOAL_SETS="$reviewed_data_root/category_goal_sets.jsonl"
export INTENTIONNAV_GOAL_REGION_EPISODES="$INTENTIONNAV_EPISODES_JSONL"

# One physical RTX 3090 only.  Physical GPU1 must retain its native index for
# Omniverse/Vulkan: CUDA_VISIBLE_DEVICES=1 remaps PyTorch to cuda:0 while
# GPU Foundation still enumerates physical devices, which makes activeGpu 0
# invalid and can segfault Isaac Sim during startup.  GPU0 keeps the historical
# isolated mapping; GPU1 uses explicit cuda:1 devices without remapping.
export INAV_PHYSICAL_GPU="$physical_gpu"
if [[ "$physical_gpu" == "0" ]]; then
    export CUDA_VISIBLE_DEVICES=0
    runtime_cuda_device=cuda:0
    isaac_gpu_index=0
else
    unset CUDA_VISIBLE_DEVICES
    runtime_cuda_device=cuda:1
    isaac_gpu_index=1
fi
export DINO_DEVICE="$runtime_cuda_device"
export DINO_MODEL_ID=IDEA-Research/grounding-dino-tiny
export DINO_MODEL_REVISION=a2bb814dd30d776dcf7e30523b00659f4f141c71
export INAV_COCO_DEVICE="$runtime_cuda_device"
export INAV_COCO_CONFIDENCE=0.25
export INAV_COCO_IMAGE_SIZE=640
export SIGLIP_DEVICE="$runtime_cuda_device"
export ISAACSIM_ACTIVE_GPU="$isaac_gpu_index"
export ISAACSIM_PHYSICS_GPU="$isaac_gpu_index"
export ISAACSIM_MULTI_GPU=0
export ISS_RENDER_TICKS="$render_ticks"
export INAV_RENDER_ANTIALIASING_MODE="$render_antialiasing_mode"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/anonymous/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

export INAV_EVIDENCE_NAV=1
export INAV_TARGET_ONLY_DINO=1
export INAV_CLIP_VERIFY=1
export INAV_CLIP_REVISION=1a25a446712ba5ee05982a381eed697ef9b435cf
export INAV_CLIP_BACKGROUND_CALIBRATION=0
export INAV_DUAL_DETECTOR_FUSION="$dual_detector_fusion_arm"
export INAV_DUAL_DETECTOR_CONFIDENCE=0.05
export INAV_DUAL_DETECTOR_OVERLAP=0.5
export INAV_DUAL_DETECTOR_STOP_SUPPORT="$dual_detector_stop_support_arm"
export INAV_INSTANCE_VISIBILITY_DIAGNOSTIC=1
export INAV_SIGLIP_ENSEMBLE=1
export INAV_SIGLIP_MODEL=google/siglip2-base-patch16-384
export INAV_SIGLIP_REVISION=f775b65a79762255128c981547af89addcfe0f88
export INAV_SIGLIP_MIN_MARGIN=0.01
export INAV_COCO_SPECIALIST=0
export INAV_COCO_STOP_GATE=0
export INAV_CONTEXTUAL_STOP_CONSENSUS=0
export INAV_CONTEXTUAL_STOP_STRONG_MARGIN=0.01
export INAV_CONTEXTUAL_STOP_MIN_VIEWPOINTS=3
export INAV_CONTEXTUAL_STOP_MIN_AREA_FRACTION=0.01
export INAV_CONTEXTUAL_STOP_MIN_VOTES=2
export INAV_BUDGETED_VIEW_SCAN="$budgeted_view_arm"
export INAV_BUDGETED_SCAN_ROTATION_DEG=120.0
export INAV_BUDGETED_SCAN_ROTATIONS_PER_SITE=2
export INAV_BUDGETED_SCAN_MIN_ANCHOR_DISTANCE_M=2.5
export INAV_BUDGETED_SCAN_MIN_STEP=2
if [[ "$late_room_profile" == "1" \
      || "$support_inspection_arm" == "1" \
      || "$carrier_active_verify_arm" == "1" \
      || "$dual_detector_fusion_arm" == "1" ]]; then
    export INAV_YOLO_WORLD_DEVICE="$runtime_cuda_device"
    export INAV_YOLO_WORLD_CHECKPOINT="$yolo_world_checkpoint"
fi
if [[ "$late_room_profile" == "1" ]]; then
    export INAV_BUDGETED_SCAN_MAX_SITES=1
    export INAV_BUDGETED_SCAN_MAX_SITES_PER_ROOM=1
    export INAV_BUDGETED_SCAN_MIN_BUDGET_FRACTION=0.4
    export INAV_BUDGETED_SCAN_LIKELY_ROOM_ONLY=1
    export INAV_BUDGETED_SCAN_YOLO_WORLD=1
    export INAV_BUDGETED_SCAN_YOLO_CONFIDENCE=0.25
    export INAV_BUDGETED_SCAN_APPROACH_QUALITY=0.30
else
    export INAV_BUDGETED_SCAN_MAX_SITES=3
    export INAV_BUDGETED_SCAN_MAX_SITES_PER_ROOM=2
    export INAV_BUDGETED_SCAN_MIN_BUDGET_FRACTION=0.0
    export INAV_BUDGETED_SCAN_LIKELY_ROOM_ONLY=0
    export INAV_BUDGETED_SCAN_YOLO_WORLD=0
    export INAV_BUDGETED_SCAN_YOLO_CONFIDENCE=0.25
    export INAV_BUDGETED_SCAN_APPROACH_QUALITY=0.30
fi
export INAV_COCO_CHECKPOINT=/path/to/workspace/models/YOLOWorld/yolo11m.pt
export INAV_HYBRID_FRONTIER=1
export INAV_GLOBAL_FRONTIER=0
export INAV_GLOBAL_FRONTIER_SCORE_THRESHOLD=0.5
export INAV_TARGET_VISUAL_MEMORY_FRONTIER="$target_visual_memory_arm"
export INAV_GEOMETRY_SAFE_FRONTIER="$geometry_safe_frontier_arm"
export VM_NO_ADMIT="$no_admit_arm"
export INAV_MID_PANO_ALL_ROOMS=0
export VM_PANO_INIT=0
export VM_MID_PANO_SCAN=0
export AUTO_REORIENT=0
export INAV_FRONTIER_SEED=42
export INAV_ADAPTIVE_STOP=1
export INAV_APPROACH_QUALITY=0.35
export INAV_REACHABLE_TARGET_REGION="$reachable_target_region_arm"
export INAV_REACHABLE_TARGET_REGION_MAX_M=2.0
export INAV_BOUNDED_COMMITMENT="$bounded_commitment_arm"
export INAV_STRUCTURED_VERIFICATION=0
export INAV_STRUCTURED_VERIFY_RADIUS_M=2.4
export INAV_STRUCTURED_VERIFY_MIN_ANGLE_DEG=30.0
export INAV_STRUCTURED_VERIFY_ARRIVAL_TOLERANCE_M=0.25
export INAV_TENTATIVE_MEMORY="$tentative_probe_arm"
export INAV_TENTATIVE_PROBE_BUDGET=1
export INAV_TENTATIVE_PROBE_STEP_M=0.6
export INAV_TENTATIVE_MAX_RANK=5
export INAV_POSE_EVIDENCE_GRAPH="$pose_evidence_graph_arm"
export INAV_POSE_GRAPH_ACTION_BUDGET=2
export INAV_POSE_GRAPH_LATERAL_M=0.8
export INAV_POSE_GRAPH_MIN_BASELINE_M=0.4
export INAV_POSE_GRAPH_MAX_MOVE_M=1.0
export INAV_POSE_GRAPH_MIN_CROSSING_DEG=8.0
export INAV_POSE_GRAPH_MAX_CROSSING_DEG=90.0
export INAV_POSE_GRAPH_MAX_RANGE_M=12.0
export INAV_POSE_GRAPH_ACCEPTED_RESIDUAL_M=1.5
export INAV_TARGET_LOCK_CONTROLLER="$target_lock_controller_arm"
export INAV_TARGET_LOCK_ACTION_BUDGET=3
export INAV_TARGET_LOCK_MAX_STEP_M=0.45
export INAV_TARGET_LOCK_TERMINAL_RADIUS_M=1.5
export INAV_VERIFY_DISTANCE_M=1.8
export INAV_TARGET_LOCK_ACTIVATION_RADIUS_M=1.8
export INAV_TARGET_LOCK_MIN_PROGRESS_M=0.05
export INAV_TARGET_LOCK_MIN_TRANSLATION_M=0.25
export INAV_TARGET_LOCK_MAX_DISTANCE_INCREASE_M=0.05
export INAV_TARGET_STANDOFF_COMMITMENT="$target_standoff_commitment_arm"
export INAV_TARGET_STANDOFF_COMMITMENT_MAX_ACTIONS=6
export INAV_TARGET_STANDOFF_COMMITMENT_EPISODE_ACTION_BUDGET=12
export INAV_TARGET_STANDOFF_COMMITMENT_MAX_DRIFT_M=0.75
export INAV_TARGET_BBOX_RECENTER="$target_bbox_recenter_arm"
export INAV_TARGET_BBOX_RECENTER_ACTION_BUDGET=2
export INAV_TARGET_BBOX_RECENTER_SCORE_MIN=0.30
export INAV_TARGET_BBOX_RECENTER_EDGE_FRACTION_MIN=0.15
export INAV_TARGET_BBOX_RECENTER_MAX_ROTATION_DEG=45.0
export INAV_TARGET_BBOX_RECENTER_MEMORY_RADIUS_M=2.5
export INAV_PERSISTENT_SINGLE_ENCODER_VETO="$persistent_single_encoder_veto_arm"
export INAV_PERSISTENT_SINGLE_ENCODER_VETO_OBSERVATIONS=4
export INAV_TERMINAL_EVIDENCE_QUORUM="$terminal_evidence_quorum_arm"
export INAV_TERMINAL_QUORUM_PERSISTENT_OBSERVATIONS=4
export INAV_TERMINAL_QUORUM_WEAK_RANK_MAX=5
export INAV_TERMINAL_QUORUM_CONTEXTUAL_MIN_VOTES=2
export INAV_SUPPORT_INSPECTION="$support_inspection_arm"
export INAV_SUPPORT_INSPECTION_BUDGET=1
export INAV_SUPPORT_INSPECTION_CONFIDENCE=0.25
export INAV_SUPPORT_INSPECTION_STEP_M=0.6
export INAV_CARRIER_ACTIVE_VERIFY="$carrier_active_verify_arm"
export INAV_CARRIER_SESSION_BUDGET=2
export INAV_CARRIER_ACTION_BUDGET=8
export INAV_CARRIER_APPROACH_BUDGET=5
export INAV_CARRIER_ROTATION_BUDGET=3
export INAV_CARRIER_ROTATION_DEG=90.0
export INAV_CARRIER_STEP_M=1.2
export INAV_CARRIER_BLACKLIST_RADIUS_M=1.0
export INAV_CARRIER_MIN_VIEW_ANGLE_DEG=30.0
export INAV_VISUAL_TARGET_DESCRIPTORS="$visual_target_descriptors_arm"
export INAV_BELIEF_REPERCEPTION="$belief_reperception_arm"
export INAV_BELIEF_MAX_AGE=10
export INAV_BELIEF_MAX_ATTEMPTS=3
export INAV_BELIEF_STEP_M=1.2
export INAV_BELIEF_COVERAGE_PRESERVING="$belief_coverage_preserving_arm"
export INAV_BELIEF_OPPORTUNISTIC_RADIUS_M=2.5
export INAV_BELIEF_OPPORTUNISTIC_MAX_AGE=1
export INAV_BELIEF_OPPORTUNISTIC_EPISODE_BUDGET=2
export INAV_BELIEF_OPPORTUNISTIC_MAX_MOVE_M=1.1
export INAV_CAMERA_HFOV_DEG=90.0
export INAV_PASSIVE_MULTI_VIEW=0
export INAV_PASSIVE_MIN_VIEWPOINTS=3
export INAV_PASSIVE_MIN_SEMANTIC_SUPPORTS=2
export INAV_PASSIVE_MAX_DISPERSION_M=0.5
export INAV_EXECUTION_SEED="$execution_seed"
export INAV_MASK_REFINEMENT=0
export INAV_MOBILE_SAM_CHECKPOINT=/path/to/workspace/models/MobileSAM/mobile_sam.pt

[[ -n "$selection_file" ]] || { printf 'Shared launcher requires a selection file\n' >&2; exit 2; }
if [[ "$check_only" == 1 ]]; then
    CUDA_VISIBLE_DEVICES='' exec "$python_bin" "$agent_source_dir/agent_intentionnav.py" \
        --objectnav --local-perception-only --style formal --step-cap 30 \
        --only "$probe_sels" "${shared_query_args[@]}" --check-navigation-inputs
fi
# Check the actual selected card; other cards may run unrelated jobs.
# --check-resources exercises this same runtime branch without creating output.
"$python_bin" - <<'GPU'
import json
import os
import subprocess
gpu = os.environ['INAV_PHYSICAL_GPU']
value = subprocess.check_output(['nvidia-smi', '--id=' + gpu,
    '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
if int(value.strip()) >= 500:
    raise SystemExit('Selected GPU is occupied; no process was interrupted')
print(json.dumps(dict(resource_check_passed=True, physical_gpu=int(gpu),
    memory_mib=int(value.strip()), gpu_initialized=False, output_created=False)))
GPU
if [[ "$check_resources" == 1 ]]; then
    exit 0
fi
mkdir -p "$run_root/episodes" "$run_root/logs" "$run_root/omni" \
    "$run_root/source_snapshot/eval/agents" \
    "$run_root/source_snapshot/eval/aggregate" \
    "$run_root/source_snapshot/eval/simulator" \
    "$run_root/source_snapshot/eval/vocab" \
    "$run_root/source_snapshot/scripts"
"$python_bin" "$shared_input_tool" snapshot --inputs "$shared_query_inputs" \
    --items "$INTENTIONNAV_DATASET_JSONL" --selections "$selection_file" \
    --mode "$shared_input_mode" --archive "$run_root/shared_input_snapshot" \
    > "$run_root/logs/shared_input_snapshot.log"
cp "$shared_input_tool" "$run_root/source_snapshot/scripts/"
cp "$agent_source_dir/agent_vlm_engine.py" \
    "$agent_source_dir/navigation_query.py" \
    "$agent_source_dir/agent_intentionnav.py" \
    "$agent_source_dir/agent_vlm.py" \
    "$agent_source_dir/clients.py" \
    "$agent_source_dir/common.py" \
    "$agent_source_dir/dino_detector.py" \
    "$agent_source_dir/evidence_nav.py" \
    "$agent_source_dir/pose_evidence_graph.py" \
    "$agent_source_dir/target_belief.py" \
    "$agent_source_dir/clip_verifier.py" \
    "$agent_source_dir/siglip_verifier.py" \
    "$agent_source_dir/ensemble_verifier.py" \
    "$agent_source_dir/coco_detector.py" \
    "$agent_source_dir/object_vocabulary.py" \
    "$agent_source_dir/mask_refiner.py" \
    "$agent_source_dir/object_room_priors.py" \
    "$agent_source_dir/object_support_priors.py" \
    "$agent_source_dir/visual_target_descriptors.py" \
    "$agent_source_dir/yolo_world_verifier.py" \
    "$agent_source_dir/value_map.py" \
    "$run_root/source_snapshot/eval/agents/"
cp "$aggregate_source_dir/compute_metrics.py" \
    "$aggregate_source_dir/visibility.py" \
    "$run_root/source_snapshot/eval/aggregate/"
cp "$simulator_source_dir/iss_env.py" \
    "$simulator_source_dir/walkable_map.py" \
    "$run_root/source_snapshot/eval/simulator/"
cp "$repo_dir/eval/agents/common.py" \
    "$run_root/source_snapshot/eval/agents/metrics_common.py"
cp "$vocab_source_file" \
    "$run_root/source_snapshot/eval/vocab/"
cp "$input_resolver" \
    "$run_root/source_snapshot/scripts/build_navigation_input_filelist.py"
cp "$launcher_path" \
    "$run_root/source_snapshot/launch_probe18.sh"
cp "$(dirname "$launcher_path")/hash_navigation_inputs.py" \
    "$run_root/source_snapshot/hash_navigation_inputs.py"

# Preserve all frozen modules, including the acquired-observation helper.
"$python_bin" - "$agent_eval_root" "$run_root/source_snapshot/eval" <<'ARCHIVE_SOURCE'
from pathlib import Path
import shutil
import sys
source, destination = map(Path, sys.argv[1:])
for path in source.rglob('*'):
    if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != path.read_bytes():
                raise ValueError(f'Conflicting existing source snapshot: {target}')
            continue
        shutil.copyfile(path, target)
ARCHIVE_SOURCE

input_filelist="$run_root/source_snapshot/resolved_input_files.txt"
"$python_bin" "$input_resolver" \
    --selection-file "$selection_file" \
    --episodes-jsonl "$INTENTIONNAV_EPISODES_JSONL" \
    --dataset-jsonl "$INTENTIONNAV_DATASET_JSONL" \
    --dataset-root "$INTENTIONNAV_DATASET_ROOT" \
    --usd-root "$INTENTIONNAV_USD_ROOT" \
    --metaroot "$INTENTIONNAV_METAROOT" \
    --scene-summary-root "$INTENTIONNAV_SCENE_SUMMARY" \
    --output "$input_filelist" \
    > "$run_root/logs/input_resolution.log"

dino_snapshot="$HF_HUB_CACHE/models--IDEA-Research--grounding-dino-tiny/snapshots/$DINO_MODEL_REVISION"
clip_snapshot="$HF_HUB_CACHE/models--laion--CLIP-ViT-B-32-laion2B-s34B-b79K/snapshots/$INAV_CLIP_REVISION"
siglip_snapshot="$HF_HUB_CACHE/models--google--siglip2-base-patch16-384/snapshots/$INAV_SIGLIP_REVISION"
model_filelist="$run_root/source_snapshot/resolved_model_files.txt"
: > "$model_filelist"
for model_snapshot in "$dino_snapshot" "$clip_snapshot" "$siglip_snapshot"; do
    if [[ ! -d "$model_snapshot" ]]; then
        printf 'Missing frozen model snapshot: %s\n' "$model_snapshot" >&2
        exit 28
    fi
    broken_link=$(find "$model_snapshot" -maxdepth 1 -type l \
        ! -exec test -e '{}' \; -print -quit)
    if [[ -n "$broken_link" ]]; then
        printf 'Broken model snapshot symlink: %s\n' "$broken_link" >&2
        exit 29
    fi
    find -L "$model_snapshot" -maxdepth 1 -type f -print \
        >> "$model_filelist"
done
sort -u -o "$model_filelist" "$model_filelist"
if [[ $(wc -l < "$model_filelist") -lt 2 ]]; then
    printf 'Resolved model snapshot closure is unexpectedly empty\n' >&2
    exit 30
fi
sha256sum "$agent_source_dir/agent_vlm_engine.py" \
    "$agent_source_dir/navigation_query.py" \
    "$agent_source_dir/agent_intentionnav.py" \
    "$agent_source_dir/agent_vlm.py" \
    "$agent_source_dir/clients.py" \
    "$agent_source_dir/common.py" \
    "$agent_source_dir/dino_detector.py" \
    "$agent_source_dir/evidence_nav.py" \
    "$agent_source_dir/pose_evidence_graph.py" \
    "$agent_source_dir/target_belief.py" \
    "$agent_source_dir/clip_verifier.py" \
    "$agent_source_dir/siglip_verifier.py" \
    "$agent_source_dir/ensemble_verifier.py" \
    "$agent_source_dir/coco_detector.py" \
    "$agent_source_dir/object_vocabulary.py" \
    "$agent_source_dir/mask_refiner.py" \
    "$agent_source_dir/object_room_priors.py" \
    "$agent_source_dir/object_support_priors.py" \
    "$agent_source_dir/visual_target_descriptors.py" \
    "$agent_source_dir/yolo_world_verifier.py" \
    "$agent_source_dir/value_map.py" \
    "$aggregate_source_dir/compute_metrics.py" \
    "$aggregate_source_dir/visibility.py" \
    "$simulator_source_dir/iss_env.py" \
    "$simulator_source_dir/walkable_map.py" \
    "$repo_dir/eval/agents/common.py" \
    "$vocab_source_file" \
    "$input_resolver" \
    "$launcher_path" \
    "$yolo_world_checkpoint" \
    "$INAV_COCO_CHECKPOINT" \
    "$INAV_MOBILE_SAM_CHECKPOINT" \
    "$INTENTIONNAV_DATASET_JSONL" "$INTENTIONNAV_EPISODES_JSONL" \
    "$INTENTIONNAV_CATEGORY_GOAL_SETS" > "$run_root/source_manifest.sha256"
# Stream the same digests, then discard only these clean input cache pages.
# A real CUDA allocation/kernel check fails before Isaac if the driver cannot
# initialize after the large input scan. Navigation policy stays frozen.
sha256sum "$shared_input_tool" "$(dirname "$launcher_path")/hash_navigation_inputs.py" \
    >> "$run_root/source_manifest.sha256"
"$python_bin" "$(dirname "$launcher_path")/hash_navigation_inputs.py" \
    --file-list "$model_filelist" --file-list "$input_filelist" \
    --telemetry "$run_root/logs/input_hashing.json" --cuda-witness \
    >> "$run_root/source_manifest.sha256"
# All frozen modules participate in the launcher's before/after closure.
find "$agent_eval_root" -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 \
    | sort -z | xargs -0 sha256sum >> "$run_root/source_manifest.sha256"
if [[ -n "$selection_file" ]]; then
    cp "$selection_file" "$run_root/source_snapshot/"
    sha256sum "$selection_file" >> "$run_root/source_manifest.sha256"
fi
{
if [[ -v CUDA_VISIBLE_DEVICES ]]; then
    declare -p CUDA_VISIBLE_DEVICES
else
    printf 'CUDA_VISIBLE_DEVICES_UNSET=1\n'
fi
declare -p DINO_DEVICE DINO_MODEL_ID \
    DINO_MODEL_REVISION ISAACSIM_ACTIVE_GPU \
    ISAACSIM_PHYSICS_GPU ISAACSIM_MULTI_GPU ISS_RENDER_TICKS \
    INAV_CLIP_REVISION \
    INAV_CLIP_BACKGROUND_CALIBRATION INAV_INSTANCE_VISIBILITY_DIAGNOSTIC \
    INAV_DUAL_DETECTOR_FUSION INAV_DUAL_DETECTOR_CONFIDENCE \
    INAV_DUAL_DETECTOR_OVERLAP INAV_DUAL_DETECTOR_STOP_SUPPORT \
    INAV_COCO_SPECIALIST INAV_COCO_STOP_GATE INAV_COCO_CHECKPOINT \
    INAV_COCO_DEVICE INAV_COCO_CONFIDENCE INAV_COCO_IMAGE_SIZE \
    INAV_CONTEXTUAL_STOP_CONSENSUS \
    INAV_CONTEXTUAL_STOP_STRONG_MARGIN \
    INAV_CONTEXTUAL_STOP_MIN_VIEWPOINTS \
    INAV_CONTEXTUAL_STOP_MIN_AREA_FRACTION \
    INAV_CONTEXTUAL_STOP_MIN_VOTES \
    INAV_BUDGETED_VIEW_SCAN INAV_BUDGETED_SCAN_ROTATION_DEG \
    INAV_BUDGETED_SCAN_ROTATIONS_PER_SITE \
    INAV_BUDGETED_SCAN_MAX_SITES \
    INAV_BUDGETED_SCAN_MAX_SITES_PER_ROOM \
    INAV_BUDGETED_SCAN_MIN_ANCHOR_DISTANCE_M \
    INAV_BUDGETED_SCAN_MIN_STEP \
    INAV_BUDGETED_SCAN_MIN_BUDGET_FRACTION \
    INAV_BUDGETED_SCAN_LIKELY_ROOM_ONLY \
    INAV_BUDGETED_SCAN_YOLO_WORLD \
    INAV_BUDGETED_SCAN_YOLO_CONFIDENCE \
    INAV_BUDGETED_SCAN_APPROACH_QUALITY \
    INAV_REACHABLE_TARGET_REGION \
    INAV_REACHABLE_TARGET_REGION_MAX_M \
    INAV_BOUNDED_COMMITMENT \
    INAV_SUPPORT_INSPECTION INAV_SUPPORT_INSPECTION_BUDGET \
    INAV_SUPPORT_INSPECTION_CONFIDENCE INAV_SUPPORT_INSPECTION_STEP_M \
    INAV_CARRIER_ACTIVE_VERIFY INAV_CARRIER_SESSION_BUDGET \
    INAV_CARRIER_ACTION_BUDGET INAV_CARRIER_APPROACH_BUDGET \
    INAV_CARRIER_ROTATION_BUDGET INAV_CARRIER_ROTATION_DEG \
    INAV_CARRIER_STEP_M \
    INAV_CARRIER_BLACKLIST_RADIUS_M INAV_CARRIER_MIN_VIEW_ANGLE_DEG \
    INAV_VISUAL_TARGET_DESCRIPTORS \
    INAV_BELIEF_REPERCEPTION INAV_BELIEF_MAX_AGE \
    INAV_BELIEF_MAX_ATTEMPTS INAV_BELIEF_STEP_M \
    INAV_BELIEF_COVERAGE_PRESERVING \
    INAV_BELIEF_OPPORTUNISTIC_RADIUS_M \
    INAV_BELIEF_OPPORTUNISTIC_MAX_AGE \
    INAV_BELIEF_OPPORTUNISTIC_EPISODE_BUDGET \
    INAV_BELIEF_OPPORTUNISTIC_MAX_MOVE_M \
    INAV_CAMERA_HFOV_DEG INAV_SIGLIP_ENSEMBLE INAV_SIGLIP_REVISION \
    VM_PANO_INIT VM_MID_PANO_SCAN AUTO_REORIENT \
    INAV_GLOBAL_FRONTIER INAV_GLOBAL_FRONTIER_SCORE_THRESHOLD \
    INAV_TARGET_VISUAL_MEMORY_FRONTIER \
    INAV_GEOMETRY_SAFE_FRONTIER VM_NO_ADMIT \
    INAV_STRUCTURED_VERIFICATION INAV_TENTATIVE_MEMORY \
    INAV_POSE_EVIDENCE_GRAPH INAV_POSE_GRAPH_ACTION_BUDGET \
    INAV_POSE_GRAPH_LATERAL_M INAV_POSE_GRAPH_MIN_BASELINE_M \
    INAV_POSE_GRAPH_MAX_MOVE_M INAV_POSE_GRAPH_MIN_CROSSING_DEG \
    INAV_POSE_GRAPH_MAX_CROSSING_DEG \
    INAV_POSE_GRAPH_MAX_RANGE_M INAV_POSE_GRAPH_ACCEPTED_RESIDUAL_M \
    INAV_TARGET_LOCK_CONTROLLER INAV_TARGET_LOCK_ACTION_BUDGET \
    INAV_TARGET_LOCK_MAX_STEP_M INAV_TARGET_LOCK_TERMINAL_RADIUS_M \
    INAV_VERIFY_DISTANCE_M INAV_TARGET_LOCK_ACTIVATION_RADIUS_M \
    INAV_TARGET_LOCK_MIN_PROGRESS_M \
    INAV_TARGET_LOCK_MIN_TRANSLATION_M \
    INAV_TARGET_LOCK_MAX_DISTANCE_INCREASE_M \
    INAV_TARGET_STANDOFF_COMMITMENT \
    INAV_TARGET_STANDOFF_COMMITMENT_MAX_ACTIONS \
    INAV_TARGET_STANDOFF_COMMITMENT_EPISODE_ACTION_BUDGET \
    INAV_TARGET_STANDOFF_COMMITMENT_MAX_DRIFT_M \
    INAV_TARGET_BBOX_RECENTER \
    INAV_TARGET_BBOX_RECENTER_ACTION_BUDGET \
    INAV_TARGET_BBOX_RECENTER_SCORE_MIN \
    INAV_TARGET_BBOX_RECENTER_EDGE_FRACTION_MIN \
    INAV_TARGET_BBOX_RECENTER_MAX_ROTATION_DEG \
    INAV_TARGET_BBOX_RECENTER_MEMORY_RADIUS_M \
    INAV_PERSISTENT_SINGLE_ENCODER_VETO \
    INAV_PERSISTENT_SINGLE_ENCODER_VETO_OBSERVATIONS \
    INAV_TERMINAL_EVIDENCE_QUORUM \
    INAV_TERMINAL_QUORUM_PERSISTENT_OBSERVATIONS \
    INAV_TERMINAL_QUORUM_WEAK_RANK_MAX \
    INAV_TERMINAL_QUORUM_CONTEXTUAL_MIN_VOTES \
    INAV_PASSIVE_MULTI_VIEW INAV_MASK_REFINEMENT
} > "$run_root/configuration.env"
if [[ "$late_room_profile" == "1" \
      || "$support_inspection_arm" == "1" \
      || "$carrier_active_verify_arm" == "1" \
      || "$dual_detector_fusion_arm" == "1" ]]; then
    declare -p INAV_YOLO_WORLD_DEVICE INAV_YOLO_WORLD_CHECKPOINT \
        >> "$run_root/configuration.env"
fi
printf 'INAV_ONLY_SELECTIONS=%q\nINAV_EXPECTED_RECORDS=%q\n' \
    "$probe_sels" "$expected_records" >> "$run_root/configuration.env"
printf 'INAV_SHARED_QUERY_INPUTS=%q\nINAV_SHARED_INPUT_MODE=%q\n' \
    "$shared_query_inputs" "$shared_input_mode" >> "$run_root/configuration.env"
printf 'INAV_SELECTION_FILE=%q\n' "$selection_file" \
    >> "$run_root/configuration.env"
printf 'INAV_AGENT_SOURCE_DIR=%q\n' "$agent_source_dir" \
    >> "$run_root/configuration.env"
printf 'INAV_INPUT_RESOLVER=%q\nHOSTED_API_KEYS=unset\nCACHE_ALIASES=unset\n' "$input_resolver" \
    >> "$run_root/configuration.env"
printf 'INAV_LATE_ROOM_RESCUE_PROFILE=%q\n' "$late_room_profile" \
    >> "$run_root/configuration.env"
printf 'INAV_RESUME_RUN=%q\n' "$resume_run" \
    >> "$run_root/configuration.env"
while IFS= read -r environment_name; do
    case "$environment_name" in
        INAV_*|VM_*|AUTO_REORIENT|CUDA_VISIBLE_DEVICES|DINO_*|SIGLIP_*|ISAACSIM_*|ISS_RENDER_TICKS|HF_HOME|HF_HUB_CACHE|HF_HUB_OFFLINE|TRANSFORMERS_OFFLINE)
            declare -p "$environment_name" ;;
    esac
done < <(compgen -e | sort) >> "$run_root/configuration.env"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_root/started_at.txt"

set +u
source "$ISAACSIM_ROOT/setup_conda_env.sh"
set -u
cd "$repo_dir"

set +e
EVAL_OUT_DIR="$run_root/episodes" OMNI_USER_PATH="$run_root/omni" \
    "$python_bin" "$agent_source_dir/agent_intentionnav.py" \
        --objectnav --local-perception-only --style formal --step-cap 30 \
        --only "$probe_sels" "${shared_query_args[@]}" > "$run_root/logs/run.log" 2>&1
simulator_exit=$?
set -e
printf '%s\n' "$simulator_exit" > "$run_root/simulator_exit_code.txt"
[[ "$simulator_exit" -eq 0 ]] || exit "$simulator_exit"

record_count=$(find "$run_root/episodes" -name record.json -type f | wc -l)
if [[ "$record_count" -ne "$expected_records" ]]; then
    printf 'Expected %s records, found %s\n' \
        "$expected_records" "$record_count" >&2
    exit 21
fi

# Legacy aggregator does not recognize shared goal_input values. The queue
# independently scores this completed run with the declared common goal set.
"$python_bin" "$shared_input_tool" audit --archive "$run_root/shared_input_snapshot" \
    --run "$run_root" --system r055 --source "$agent_eval_root" \
    --output "$run_root/shared_input_audit.json" > "$run_root/logs/shared_input_audit.log"
find "$run_root/episodes" "$run_root/shared_input_snapshot" -type f -print0 \
    | sort -z | xargs -0 sha256sum > "$run_root/output_manifest.sha256"

while IFS= read -r manifest_line; do
    manifest_path="${manifest_line#*  }"
    sha256sum "$manifest_path"
done < "$run_root/source_manifest.sha256" \
    > "$run_root/source_manifest_after.sha256"
if ! cmp -s "$run_root/source_manifest.sha256" \
        "$run_root/source_manifest_after.sha256"; then
    printf 'Source/model/data hashes changed during execution\n' >&2
    exit 26
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$run_root/completed_at.txt"
touch "$run_root/.RUN_SUCCESS"
