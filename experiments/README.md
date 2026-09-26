# Experiment Source and Reproduction

These directories contain the frozen implementations executed for the two
additional studies. Every copied source file was checked against its original
run snapshot and the corresponding public Hugging Face archive. Workstation
paths and user identifiers are anonymized; policy and scoring logic are preserved.

## Source map

| Study | Source | Contents |
|---|---|---|
| Shared-category comparison | [Reference](shared_category/code/policies/r055/) | Local detection, spatial memory, search, collision-checked movement, and completion |
| Shared-category comparison | [MTU3D](shared_category/code/policies/mtu3d/) | Learned grounding/frontier policy, coordinate adapter, inference worker, and action interface |
| Shared category prediction | [Frontend](shared_category/code/policies/mtu3d/agents/open_weight_intent.py) | Qwen3.5-4B inference and prompts |
| Comparison execution | [Scripts](shared_category/code/scripts/) and [launchers](shared_category/code/launchers/) | Shared inputs, job preparation/execution, motion/input audits, scoring, and paired analysis |
| Hosted repetitions | [Agent](hosted_repeats/code/eval/) and [scripts](hosted_repeats/code/scripts/) | Hosted agent, fresh initial planning per episode, call receipts, regression tests, and disagreement analysis |

The shared-category study uses the `reviewed-v3` tasks: 500 tasks in each of
four system/input conditions, plus two repetitions of 40 development tasks per
condition. The hosted study repeats four expressions of 40 original tasks three
times. Each study retains its own task manifest and scoring definitions.

## Verify the code

From the repository root:

```bash
python3 scripts/verify_experiment_sources.py
```

Each `SOURCE_MANIFEST.json` lists original source hashes and distributed hashes.
The checked-in copies match the code in the public release byte for byte.
`configuration.json` records the study settings and source identities.

## Recompute the published results

Install `huggingface_hub`, `numpy`, and `PyYAML`, then download and extract the
desired release. The following commands run from the repository root:

```bash
python3 scripts/download_release.py system-comparison
tar -xzf data/benchmark/system_comparison/20260924/shared_category_20260924.tar.gz \
  -C data/benchmark/system_comparison/20260924
python3 experiments/shared_category/recompute.py \
  data/benchmark/system_comparison/20260924/shared_category_20260924

python3 scripts/download_release.py hosted-repeats
tar -xzf data/benchmark/hosted_repeats/20260925/hosted_repeats_20260925.tar.gz \
  -C data/benchmark/hosted_repeats/20260925
python3 experiments/hosted_repeats/recompute.py \
  data/benchmark/hosted_repeats/20260925/hosted_repeats_20260925
```

Each scorer verifies the extracted archive's checksums before recomputing its
results. The comparison scorer checks all 2,320 trajectory labels and the 2,000
full-set GSR labels. The repeat scorer checks all 480 trajectories and planner
receipts, then reconstructs disagreement rates and uncertainty. These operations
require neither a GPU nor simulator/model calls.

## Run new navigation episodes

Use Isaac Sim 4.5 and the scene, room-map, and object-summary resources described
in the root README. The comparison uses 768 × 768 RGB-D views, a 90-degree field
of view, eight render ticks, and a 30-action budget including rotations and STOP.
MTU3D runs its inference worker in a separate environment with its upstream
compiled dependencies and pretrained checkpoints; their revisions are recorded
in `shared_category/configuration.json` and the release provenance.

The archived launchers record the exact execution settings, including local
path bindings and run-input hash checks. Configure those bindings for your
machine before rerunning them. In particular:

- The comparison queue's `invocation()` in `code/scripts/run_iclr_core_comparison.py`
  specifies the environment and arguments for each system/input condition.
- R055's launcher uses the shared predicted/correct queries and the frozen
  `code/policies/r055/` tree.
- MTU3D's `agent_mtu3d.py` needs the local upstream MTU3D checkout, checkpoint
  directory, and worker Python executable configured in `Worker`.
- The hosted repetition runner needs its `ROOT`, `SOURCE`, and `OUT` paths,
  fresh-run manifest, scene assets, and hosted API credentials configured.
  Keep the per-episode cache reset and planner receipts enabled.

Keep original archives and their hashes intact. Use a separate output directory
and record relocated paths/source hashes for new executions. Hosted responses
may change across service versions; released trajectories reproduce the paper's
numerical results independently of future calls.
