# IntentionNav

This repository contains the IntentionNav data collection code for Kujiale-style
indoor scenes in Isaac Sim.

The current release includes the data collection and benchmark construction pipeline:

1. `surface_finder`: selects target objects and support surfaces from scene graphs.
2. `capture_surfaces`: renders target-centric RGB photos and top-down overview maps.
3. `generate_intents`: annotates captured targets with implicit human intents.
4. `validate_and_build`: validates annotations and builds the benchmark JSON.
5. `regen_overview`: regenerates overview images from existing capture manifests.

Evaluation, model agents, paper drafts, generated data, logs, and archived baselines are intentionally not included here.

## Data Download

The scene collection follows the VLNTube scene assets:
[william13077/VLNTube](https://github.com/william13077/VLNTube).

Before running the collection pipeline, download the required scene data:

| Dataset | Contents | Link |
|---|---|---|
| Envs | USD scene files | [Eyz/VLNVerse_scene](https://huggingface.co/datasets/Eyz/VLNVerse_scene) |
| Room Meta | Scene metadata, collision maps, room layouts | [Eyz/SceneMeta](https://huggingface.co/datasets/Eyz/SceneMeta) |
| Scene Graph | Object relationships | [Eyz/SceneSummary](https://huggingface.co/datasets/Eyz/SceneSummary) |

## Requirements

- Isaac Sim 4.5.0
- Python environment with Isaac Sim available, plus `numpy`, `Pillow`, `opencv-python`,
  `matplotlib`, and `pyquaternion`
- `google-generativeai` for the Gemini-based annotation stage
- External scene assets:
  - `VLNVerse_scene`: USD scenes
  - `SceneMeta/metadata_train`: `freemap.npy` and `room_region.json`
  - `SceneSummary/kujiale_scene_summary`: per-scene `object_dict.json`

By default, scripts look for data under `data/` in this repository:

```text
data/
  VLNVerse_scene/
  SceneMeta/metadata_train/
  SceneSummary/kujiale_scene_summary/
```

Override paths with environment variables when running elsewhere:

```bash
export SCENE_SUMMARY=/path/to/kujiale_scene_summary
export USD_ROOT=/path/to/VLNVerse_scene
export METAROOT=/path/to/metadata_train
export ISAACSIM_ROOT=/path/to/ISAACSIM_ROOT
export WORK_DIR=/path/to/output/work_dirs
```

## Single Scene

Run one scene end to end:

```bash
conda activate goodnav
source "$ISAACSIM_ROOT/setup_conda_env.sh"
bash scripts/collect_scene.sh kujiale_0005
```

Outputs are written under:

```text
$WORK_DIR/kujiale_0005/
  surface_targets.json
  capture_manifest.json
  surface_photos/
  scene_overview.png
  scene_rendered.png
```

## Batch Capture

Single GPU:

```bash
bash scripts/batch_capture_1gpu.sh
bash scripts/batch_capture_1gpu.sh 0 20
GPU_ID=1 BATCH_SIZE=30 bash scripts/batch_capture_1gpu.sh
```

Two or more GPUs:

```bash
bash scripts/batch_capture_2gpu.sh
NUM_GPUS=4 BATCH_SIZE=30 bash scripts/batch_capture_2gpu.sh
```

Useful knobs:

```bash
NUM_VIEWS=2          # photos per target
RESOLUTION=1024     # square RGB resolution
ROUND_TIMEOUT=1800  # seconds per Isaac Sim worker round
WORK_DIR=work_dirs  # output root
```

## Intent Annotation

Phase 2 uses Gemini to turn each captured target photo into four implicit-intent
queries with formal, natural, casual, and emotional variants.

Set an API key first:

```bash
export GOOGLE_API_KEY=...
# or
export GEMINI_API_KEY=...
```

Then annotate all captured trainval scenes:

```bash
bash scripts/batch_annotate.sh
```

Annotate a slice:

```bash
bash scripts/batch_annotate.sh 0 20
```

Useful knobs:

```bash
MODEL=models/gemini-3-flash-preview
WORKERS=8
CAPTURE_DIR=work_dirs
OUTPUT_DIR=work_dir_phase2
```

Check annotation progress:

```bash
python3 -m intentionnav.data_collection.check_status \
  --capture-dir "$CAPTURE_DIR" \
  --output-dir "$OUTPUT_DIR"
```

## Build Benchmark

Phase 3 validates the annotations and builds the final benchmark file:

```bash
bash scripts/build_benchmark.sh
```

Default outputs:

```text
benchmark/intentionnav_benchmark.json
benchmark/issues.jsonl
```

The benchmark JSON contains metadata, aggregate statistics, and an `episodes`
array. Each episode includes:

- `episode_id`
- `scene_id`
- `room`
- `target_category`
- `target_objects`
- `photo`
- `intent_variants`
- `difficulty`
- `split`

Run the full local pipeline in sequence:

```bash
bash scripts/run_benchmark_pipeline.sh
```

To reuse existing captures or annotations:

```bash
SKIP_CAPTURE=1 bash scripts/run_benchmark_pipeline.sh
SKIP_CAPTURE=1 SKIP_ANNOTATE=1 bash scripts/run_benchmark_pipeline.sh
```

## Regenerate Overviews

No Isaac Sim required:

```bash
python3 -m intentionnav.data_collection.regen_overview \
  --metaroot "$METAROOT" \
  --work-dir "$WORK_DIR"
```

For one scene:

```bash
python3 -m intentionnav.data_collection.regen_overview \
  --metaroot "$METAROOT" \
  --work-dir "$WORK_DIR" \
  --scene kujiale_0005
```
