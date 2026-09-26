# Reference Evaluation Code

This directory contains the active-navigation agent and deterministic metric
implementation released with IntentionNav.

The shared-category reference/MTU3D comparison and hosted repetition study have
their own executed source snapshots in [`../experiments/`](../experiments/README.md).
Use the snapshot corresponding to the result being reproduced.

## Layout

- `agents/`: hosted-VLM, random, frontier, oracle, and text-only agents.
- `simulator/`: Isaac Sim wrapper, waypoint visualization, and walkable maps.
- `aggregate/`: IM/OSR/SR/GSR/SPL and uncertainty aggregation.
- `judge/`: prompts and the optional legacy judge scaffold.
- `report/`: table and figure generation.
- `splits/episodes.jsonl`: frozen target/start specifications for 500 items.
- `vocab/category_synonyms.yaml`: deterministic IM normalization vocabulary.

The exact unmodified evaluation archives used for the reported runs are stored
in the Hugging Face dataset under `reference_results/evaluation_code/`. The copy
in this repository keeps the same agent and metric logic while replacing local
path defaults with environment variables.

## Required Paths

```bash
export INTENTIONNAV_DATASET_ROOT=/path/to/fixed/benchmark
export INTENTIONNAV_USD_ROOT=/path/to/VLNVerse_scene
export INTENTIONNAV_METAROOT=/path/to/SceneMeta/metadata_train
export INTENTIONNAV_SCENE_SUMMARY=/path/to/SceneSummary/kujiale_scene_summary
export ISAACSIM_ROOT=/path/to/ISAACSIM_ROOT
```

The dataset path must contain `selected_500_intents.jsonl` and the per-scene
directories. The checked-in `splits/episodes.jsonl` is the same fixed episode
specification also distributed at the dataset root.

## Hosted Models

The client uses an OpenAI-compatible `/v1/chat/completions` endpoint because the
reported backends were accessed through one gateway:

```bash
export INTENTIONNAV_API_BASE_URL=https://your-endpoint.example/v1
export INTENTIONNAV_API_KEY=...
```

`PP_API_BASE_URL`, `PP_API_KEY`, and `PP_API_PROXY` remain accepted only for
compatibility with the frozen source archives. The model catalog is in
`agents/clients.py` and `configs/models.yaml`.

## Commands

```bash
# Small active-agent run
bash scripts/run_active_eval.sh --test 3 --model gemini_3_1_flash

# Full fixed set for all three hosted backends
bash scripts/run_active_eval.sh --full --model all

# Explicit-target diagnostic used to isolate category inference
EVAL_OUT_DIR=results/explicit_target \
python eval/agents/agent_vlm_engine.py \
  --model gemini_3_1_flash --style formal --objectnav --limit 500 --headless

# Aggregate an existing record directory
INTENTIONNAV_DATASET_ROOT=/path/to/benchmark \
INTENTIONNAV_SCENE_SUMMARY=/path/to/kujiale_scene_summary \
EVAL_OUT_DIR=/path/to/records \
python eval/aggregate/compute_metrics.py --out results/recomputed_metrics
```

The active agent writes one `record.json` and its retained visual artifacts per
`(model, style, selection_id)` cell. API outputs can vary in future runs; use the
released 6,000 records when verifying the paper's reported scores.
