"""Build the compatibility-preserving R057 intent handoff.

The frozen v1 answer wins when it already resolves to one public canonical
category.  The structured v2 answer is used only for v1 out-of-vocabulary or
ambiguous labels.  A small model retry repairs the rare case where both
branches still emit an invalid label.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))

from common import save_atomic  # noqa: E402
from object_vocabulary import BENCHMARK_VOCABULARY  # noqa: E402
from open_weight_intent import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    aggregate,
    intent_match,
    load_jsonl,
    load_model,
    load_synonyms,
    record_manifest_hash,
    selected_styles,
    sha256_file,
    tolerant_json_parse,
)


PROTOCOL_VERSION = "open_weight_intent_compat_fusion_v3"


def normalized(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text).split())


def load_records(root: Path) -> dict[tuple[str, str], dict]:
    records = {}
    for path in sorted((root / "records").glob("*/*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        key = row["selection_id"], row["style"]
        if key in records:
            raise ValueError(f"duplicate source record: {key}")
        records[key] = row
    return records


def alias_owners(synonyms: dict[str, set[str]]) -> dict[str, set[str]]:
    canonical = {normalized(value) for value in BENCHMARK_VOCABULARY}
    owners = defaultdict(set)
    for category, aliases in synonyms.items():
        category_norm = normalized(category)
        if category_norm not in canonical:
            continue
        for alias in aliases:
            owners[normalized(alias)].add(category_norm)
    return owners


def correction_prompt(processor, row: dict) -> str:
    categories = ", ".join(BENCHMARK_VOCABULARY)
    messages = [
        {
            "role": "system",
            "content": (
                "Repair an invalid indoor-navigation object prediction. "
                "The previous guesses are forbidden because they are not in "
                "the canonical list. Select the closest valid object that "
                "satisfies the intent. Return only the category label and no "
                "explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f'Intent: "{row["intent"]}"\n'
                f'Forbidden previous guesses: {row["invalid_guesses"]}\n'
                f"Canonical categories: [{categories}]\n"
                "Return exactly one category label from the list."
            ),
        },
    ]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )


@torch.inference_mode()
def repair_unresolved(args: argparse.Namespace, unresolved: list[dict]) -> None:
    if not unresolved:
        return
    processor, model, device = load_model(args)
    canonical = {normalized(value): value for value in BENCHMARK_VOCABULARY}
    tokenizer = getattr(processor, "tokenizer", processor)
    candidate_tokens = [
        tokenizer.encode(value, add_special_tokens=False)
        for value in BENCHMARK_VOCABULARY
    ]
    for batch_start in range(0, len(unresolved), args.batch_size):
        batch = unresolved[batch_start:batch_start + args.batch_size]
        prompts = [correction_prompt(processor, row) for row in batch]
        inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)
        input_width = int(inputs["input_ids"].shape[1])

        def allowed_tokens(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
            prefix = input_ids[input_width:].tolist()
            matches = [
                candidate for candidate in candidate_tokens
                if candidate[:len(prefix)] == prefix
            ]
            allowed = {
                candidate[len(prefix)]
                for candidate in matches
                if len(candidate) > len(prefix)
            }
            if any(len(candidate) == len(prefix) for candidate in matches):
                allowed.add(tokenizer.eos_token_id)
            return sorted(allowed) or [tokenizer.eos_token_id]

        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            prefix_allowed_tokens_fn=allowed_tokens,
        )
        decoded = processor.batch_decode(
            generated[:, input_width:], skip_special_tokens=True
        )
        for row, response in zip(batch, decoded):
            target = canonical.get(normalized(response))
            row["correction_response"] = response
            row["correction_target"] = target or ""
            row["correction_error"] = (
                None if target else "constrained correction target is noncanonical"
            )
            row["correction_input_tokens_padded"] = input_width
            row["correction_output_tokens"] = int(generated.shape[1] - input_width)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--structured-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--synonyms", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--styles", default="all")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    baseline_dir = args.baseline_dir.resolve()
    structured_dir = args.structured_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    items = load_jsonl(args.dataset.resolve())
    styles = selected_styles(args.styles)
    expected = {(item["selection_id"], style) for item in items for style in styles}
    sources = {
        "baseline": load_records(baseline_dir),
        "structured": load_records(structured_dir),
    }
    if any(set(records) != expected for records in sources.values()):
        raise ValueError("source record populations differ from requested dataset/styles")
    synonyms = load_synonyms(args.synonyms.resolve())
    owners = alias_owners(synonyms)
    canonical = {normalized(value): value for value in BENCHMARK_VOCABULARY}

    decisions = {}
    unresolved = []
    for key in sorted(expected):
        v1 = sources["baseline"][key]
        v2 = sources["structured"][key]
        v1_target = normalized((v1.get("prediction") or {}).get("target"))
        v2_target = normalized((v2.get("prediction") or {}).get("target"))
        if v1_target in canonical:
            target, route = canonical[v1_target], "v1_exact"
        elif len(owners.get(v1_target, set())) == 1:
            target = canonical[next(iter(owners[v1_target]))]
            route = "v1_unique_alias"
        elif v2_target in canonical:
            target, route = canonical[v2_target], "structured_v2"
        else:
            target, route = "", "correction_retry"
        decision = {
            "key": key,
            "intent": v1["intent"],
            "invalid_guesses": [
                (v1.get("prediction") or {}).get("target"),
                (v2.get("raw_response") or "").strip(),
            ],
            "target": target,
            "route": route,
            "correction_response": None,
            "correction_target": "",
            "correction_error": None,
            "correction_input_tokens_padded": 0,
            "correction_output_tokens": 0,
        }
        decisions[key] = decision
        if not target:
            unresolved.append(decision)

    repair_unresolved(args, unresolved)
    route_counts = defaultdict(int)
    correction_failures = 0
    for key in sorted(expected):
        v1 = sources["baseline"][key]
        v2 = sources["structured"][key]
        decision = decisions[key]
        target = decision["target"] or decision["correction_target"]
        error = decision["correction_error"] if not target else None
        if not target:
            correction_failures += 1
        route_counts[decision["route"]] += 1
        prediction = {
            "target": target,
            "room_constraints": copy.deepcopy(
                (v2.get("prediction") or {}).get("room_constraints") or []
            ),
            "support_constraints": copy.deepcopy(
                (v2.get("prediction") or {}).get("support_constraints") or []
            ),
        }
        record = copy.deepcopy(v2)
        record.update({
            "protocol_version": PROTOCOL_VERSION,
            "prediction": prediction,
            "parse_error": error,
            "IM_hit": int(intent_match(target, v1["target_category"], synonyms)),
            "raw_response": {
                "v1": v1.get("raw_response"),
                "structured_v2": v2.get("raw_response"),
                "correction_retry": decision["correction_response"],
            },
            "fusion": {
                "route": decision["route"],
                "baseline_record_sha256": sha256_file(
                    baseline_dir / "records" / key[1] / f"{key[0]}.json"
                ),
                "structured_record_sha256": sha256_file(
                    structured_dir / "records" / key[1] / f"{key[0]}.json"
                ),
                "correction_input_tokens_padded": decision[
                    "correction_input_tokens_padded"
                ],
                "correction_output_tokens": decision["correction_output_tokens"],
            },
        })
        record["plan"]["target_guess"] = target
        record["model"]["max_new_tokens"] = args.max_new_tokens
        record["usage"]["configured_batch_size"] = args.batch_size
        save_atomic(
            record,
            output_dir / "records" / key[1] / f"{key[0]}.json",
        )

    metrics = aggregate(
        output_dir,
        items,
        styles,
        model_id=args.model_id,
        revision=args.revision,
        synonyms=synonyms,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
        configured_batch_size=args.batch_size,
        protocol_version=PROTOCOL_VERSION,
    )
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset.resolve()),
        "model_id": args.model_id,
        "model_revision": args.revision,
        "styles": styles,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
        "source_protocols": {
            "baseline": json.loads((baseline_dir / "summary.json").read_text())[
                "protocol_version"
            ],
            "structured": json.loads((structured_dir / "summary.json").read_text())[
                "protocol_version"
            ],
        },
        "source_record_manifests": {
            name: record_manifest_hash(
                sorted((root / "records").glob("*/*.json")), root / "records"
            )
            for name, root in (
                ("baseline", baseline_dir), ("structured", structured_dir)
            )
        },
        "fusion_route_counts": dict(sorted(route_counts.items())),
        "correction_failures": correction_failures,
        **metrics,
    }
    save_atomic(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if not correction_failures else 3)


if __name__ == "__main__":
    main()
