# IntentionNav

IntentionNav is a controlled diagnostic benchmark for object navigation from
implicit human instructions. This repository contains both the benchmark
construction pipeline and the released reference evaluation agent.

The fixed benchmark and evaluation payload are hosted separately so that the
code repository remains lightweight:

- **Dataset and evaluation package:**
  https://huggingface.co/datasets/Anonymous260726/IntentionNav
- **Third-party scene assets:**
  https://huggingface.co/datasets/Eyz/VLNVerse_scene

The Hugging Face release contains the unchanged 500-item/2,000-instruction
benchmark, 500 fixed navigation episode specifications, per-scene annotations
and manifests, benchmark renders, the 6,000 reference records, retained episode
artifacts, exact evaluation-code archives, and simulator/environment metadata.
Large USD scenes remain under their upstream licenses and are downloaded from
VLNVerse.

## Repository Contents

```text
intentionnav/data_collection/   target discovery, rendering, intent generation,
                                validation, and candidate benchmark building
eval/                           released agents, simulator wrapper, prompts,
                                metrics, reporting, fixed episode specifications
scripts/                        collection, download, evaluation, and metric entrypoints
splits/                         scene split definitions
```

The portable `eval/` tree corresponds to the frozen evaluation implementation.
The exact unmodified source archives used for the initial runs and completion
runs remain available under `reference_results/evaluation_code/` in the dataset
release. Hosted model APIs may change over time, so future API responses are not
expected to be bit-identical; the released records support exact metric
recomputation independently of future model calls.

## Requirements

Data collection and active evaluation use Isaac Sim 4.5.0. Lightweight data
validation and metric aggregation can run without Isaac Sim.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-collection.txt
```

For the active agent, install the additional packages in
`requirements-eval.txt` inside an Isaac-Sim-compatible environment. The frozen
Conda specification and package inventory are also provided in the dataset
release under `reference_results/reproducibility/`.

## Download the Fixed Benchmark

Install `huggingface_hub`, then download the benchmark to the default path used
by the evaluation code:

```bash
pip install huggingface_hub
python scripts/download_release.py benchmark
```

Download the lightweight 6,000-record archive as well when recomputing the
reported metrics:

```bash
python scripts/download_release.py logs
```

The retained episode artifacts are several gigabytes and are therefore an
explicit opt-in:

```bash
python scripts/download_release.py artifacts
```

All modes default to `data/benchmark/`; use `--output` to choose another path.

## Third-Party Scene Data

The collection follows the VLNTube/VLNVerse scene organization. Download the
three upstream resources before collecting or rerunning active navigation:

| Dataset | Contents | Link |
|---|---|---|
| Envs | USD scene files | [Eyz/VLNVerse_scene](https://huggingface.co/datasets/Eyz/VLNVerse_scene) |
| Room Meta | Collision maps and room layouts | [Eyz/SceneMeta](https://huggingface.co/datasets/Eyz/SceneMeta) |
| Scene Graph | Per-scene object relationships | [Eyz/SceneSummary](https://huggingface.co/datasets/Eyz/SceneSummary) |

The default directory layout is:

```text
data/
  benchmark/
  VLNVerse_scene/
  SceneMeta/metadata_train/
  SceneSummary/kujiale_scene_summary/
```

Paths can be overridden without editing code:

```bash
export INTENTIONNAV_DATASET_ROOT=/path/to/benchmark
export INTENTIONNAV_USD_ROOT=/path/to/VLNVerse_scene
export INTENTIONNAV_METAROOT=/path/to/metadata_train
export INTENTIONNAV_SCENE_SUMMARY=/path/to/kujiale_scene_summary
export ISAACSIM_ROOT=/path/to/ISAACSIM_ROOT
```

The exact 176 scene IDs and selected-file manifest used by the benchmark are in
the dataset release's reproducibility directory.

## Recompute the Reference Metrics

After downloading `logs` and the upstream `SceneSummary`, run:

```bash
bash scripts/recompute_reference_metrics.sh
```

This extracts the released record archive, evaluates the fixed records at 1 m,
2 m, and 3 m, and writes the derived tables under `results/recomputed_metrics/`.
No hosted-model calls or new navigation episodes are made.

## Run the Reference Agent

The released active agent uses RGB-D observations and pose, maintains a
walkable/semantic value map, applies open-vocabulary detection, selects
navigation waypoints, and uses a fixed stopping gate. All three hosted backends
share the same downstream perception, mapping, planning, and stopping code.

Configure an OpenAI-compatible endpoint that exposes the requested model IDs:

```bash
export INTENTIONNAV_API_BASE_URL=https://your-endpoint.example/v1
export INTENTIONNAV_API_KEY=...
```

Then run a small smoke evaluation or the fixed 500 episodes:

```bash
bash scripts/run_active_eval.sh --test 3 --model gemini_3_1_flash
bash scripts/run_active_eval.sh --full --model all
```

Use `--tier random`, `--tier fbe`, or `--tier all` for the additional released
reference policies. See `bash scripts/run_active_eval.sh --help` and
`eval/README.md` for the output layout, explicit-target diagnostic, and
configuration details.

## Data Collection Pipeline

The construction code has five stages:

1. `surface_finder` selects target objects and support surfaces from scene graphs.
2. `capture_surfaces` renders target-centric RGB images and overview maps.
3. `generate_intents` produces four controlled instruction styles.
4. `validate_and_build` filters invalid candidates and builds a candidate JSON.
5. `regen_overview` regenerates overview images from capture manifests.

### Single scene

```bash
conda activate goodnav
source "$ISAACSIM_ROOT/setup_conda_env.sh"
bash scripts/collect_scene.sh kujiale_0005
```

Outputs are written under `$WORK_DIR/kujiale_0005/` and include
`surface_targets.json`, `capture_manifest.json`, target images, and overview
renders.

### Batch capture

```bash
bash scripts/batch_capture_1gpu.sh
bash scripts/batch_capture_1gpu.sh 0 20
NUM_GPUS=4 BATCH_SIZE=30 bash scripts/batch_capture_2gpu.sh
```

Useful environment variables are `NUM_VIEWS`, `RESOLUTION`, `ROUND_TIMEOUT`,
`WORK_DIR`, and `GPU_ID`.

### Intent annotation

Set a Gemini API key and annotate all captured train/validation scenes:

```bash
export GOOGLE_API_KEY=...
bash scripts/batch_annotate.sh
```

The script is resume-safe. A slice can be selected with
`bash scripts/batch_annotate.sh 0 20`; `MODEL`, `WORKERS`, `CAPTURE_DIR`, and
`OUTPUT_DIR` are configurable.

Check progress with:

```bash
python3 -m intentionnav.data_collection.check_status \
  --capture-dir "$CAPTURE_DIR" \
  --output-dir "$OUTPUT_DIR"
```

### Build candidate annotations

```bash
bash scripts/build_benchmark.sh
```

This produces a validated candidate benchmark JSON and an issue report. The
published benchmark underwent a fixed model-assisted refinement and human audit;
that judgment is not treated as a deterministic regeneration step. Fair method
comparison should use the released fixed records rather than regenerate or
recurate a new evaluation set.

Run all automatic construction stages in sequence with:

```bash
bash scripts/run_benchmark_pipeline.sh
SKIP_CAPTURE=1 bash scripts/run_benchmark_pipeline.sh
SKIP_CAPTURE=1 SKIP_ANNOTATE=1 bash scripts/run_benchmark_pipeline.sh
```

## Regenerate Overviews

This utility does not require Isaac Sim:

```bash
python3 -m intentionnav.data_collection.regen_overview \
  --metaroot "$INTENTIONNAV_METAROOT" \
  --work-dir "$WORK_DIR"
```
