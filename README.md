# IntentionNav

This repository contains the IntentionNav data collection code for Kujiale-style
indoor scenes in Isaac Sim.

The current release includes the collection stage only:

1. `surface_finder`: selects target objects and support surfaces from scene graphs.
2. `capture_surfaces`: renders target-centric RGB photos and top-down overview maps.
3. `regen_overview`: regenerates overview images from existing capture manifests.

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
- External scene assets:
  - `VLNVerse_scene`: USD scenes
  - `SceneMeta/metadata_train`: `freemap.npy` and `room_region.json`
  - `SceneSummary/kujiale_scene_summary`: per-scene `object_dict.json`

The scripts default to the local lab paths used during development. Override them
with environment variables when running elsewhere:

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
