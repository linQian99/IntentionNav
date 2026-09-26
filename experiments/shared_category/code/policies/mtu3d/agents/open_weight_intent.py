"""Run open-weight intent inference on the frozen curated-v3 language set.

This is stage one of the open-weight baseline.  It produces one reusable,
content-addressed plan for each item/style pair without launching Isaac Sim or
calling a hosted API.  A later navigation stage can consume the frozen
``target_guess`` while keeping the navigation protocol unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForMultimodalLM, AutoProcessor


THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
from common import intent_for_style, now_iso, save_atomic, tolerant_json_parse  # noqa: E402
from object_vocabulary import BENCHMARK_VOCABULARY  # noqa: E402


MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PROTOCOL_V1 = "open_weight_intent_inference_v1"
PROTOCOL_V2 = "open_weight_intent_inference_v2"
PROTOCOL_VERSION = PROTOCOL_V1
STYLES = ("formal", "natural", "casual", "emotional")
SYSTEM_PROMPT = (
    "You infer the physical object that best satisfies an indirect human "
    "intent for an indoor navigation agent. Use ordinary object category "
    "names. Do not describe your reasoning. Return exactly one JSON object "
    "with one field, target_guess, containing one object category."
)
STRUCTURED_SYSTEM_PROMPT = (
    "You convert an indirect indoor-navigation intent into a grounded search "
    "specification. Choose target_guess exactly from the supplied canonical "
    "object categories. Preserve explicit location context instead of "
    "replacing it with generic commonsense. A context constraint is allowed "
    "only when its evidence is copied verbatim from the user intent. Do not "
    "output reasoning or any keys outside the requested JSON schema."
)
ROOM_TYPES = (
    "bathroom", "bedroom", "dining room", "kitchen", "living room",
    "study room", "balcony", "hallway",
)
SUPPORT_TYPES = (
    "basin", "bed", "cabinet", "desk", "dining table", "night stand",
    "shelf", "sofa", "table",
)
ROOM_EVIDENCE_CUES = {
    "bathroom": ("bathroom", "bath", "bathtub", "shower", "toilet"),
    "bedroom": ("bedroom", "bed", "bedside", "night stand", "nightstand"),
    "dining room": ("dining room", "dining table", "dinner table"),
    "kitchen": ("kitchen", "oven", "stove", "fridge", "kitchen counter"),
    "living room": ("living room", "sofa", "couch", "television", "tv"),
    "study room": ("study room", "study", "office", "desk", "computer"),
    "balcony": ("balcony",),
    "hallway": ("hallway", "corridor"),
}
SUPPORT_EVIDENCE_CUES = {
    "basin": ("basin", "sink"),
    "bed": ("bed", "bedside"),
    "cabinet": ("cabinet", "cupboard"),
    "desk": ("desk",),
    "dining table": ("dining table", "dinner table"),
    "night stand": ("night stand", "nightstand", "bedside table"),
    "shelf": ("shelf", "bookshelf"),
    "sofa": ("sofa", "couch"),
    "table": ("table",),
}


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_manifest_hash(paths: list[Path], base: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(base)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def export_policy_inputs(output_dir: Path, rows: list[dict]) -> dict:
    """Write navigation inputs that contain no benchmark ground-truth fields."""
    policy_root = output_dir / "policy_records"
    paths = []
    for row in rows:
        path = policy_root / row["style"] / f"{row['selection_id']}.json"
        policy_record = {
            "protocol_version": row["protocol_version"],
            "selection_id": row["selection_id"],
            "scene_id": row["scene_id"],
            "style": row["style"],
            "intent_sha256": row["intent_sha256"],
            "model": {
                "id": row["model"]["id"],
                "revision": row["model"]["revision"],
            },
            "prediction": dict(row.get("prediction") or {}),
            "source_record_sha256": sha256_file(
                output_dir / "records" / row["style"]
                / f"{row['selection_id']}.json"
            ),
        }
        save_atomic(policy_record, path)
        paths.append(path)
    return {
        "root": str(policy_root.resolve()),
        "files": len(paths),
        "sha256": record_manifest_hash(paths, policy_root),
        "ground_truth_fields_excluded": [
            "target_category",
            "IM_hit",
            "intent_mode",
            "raw_response",
        ],
    }


def normalize_label(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def load_synonyms(path: Path) -> dict[str, set[str]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    synonyms = {}
    for category, aliases in raw.items():
        key = normalize_label(category)
        synonyms[key] = {key, *(normalize_label(value) for value in aliases or [])}
    return synonyms


def intent_match(prediction: str, target: str, synonyms: dict[str, set[str]]) -> bool:
    prediction_norm = normalize_label(prediction)
    target_norm = normalize_label(target)
    if not prediction_norm or not target_norm:
        return False
    if prediction_norm == target_norm:
        return True
    return prediction_norm in synonyms.get(target_norm, set())


def prompt_parts(intent: str, protocol_version: str) -> tuple[str, str]:
    """Return the frozen system/user prompt pair for an inference protocol."""
    if protocol_version == PROTOCOL_V1:
        return SYSTEM_PROMPT, (
            f'User intent: "{intent}"\n\n'
            "Return exactly: "
            '{"target_guess":"<single object category>"}'
        )
    if protocol_version != PROTOCOL_V2:
        raise ValueError(f"unsupported protocol version: {protocol_version}")
    categories = ", ".join(BENCHMARK_VOCABULARY)
    rooms = ", ".join(ROOM_TYPES)
    supports = ", ".join(SUPPORT_TYPES)
    return STRUCTURED_SYSTEM_PROMPT, (
        f'User intent: "{intent}"\n\n'
        f"canonical_object_categories: [{categories}]\n"
        f"allowed_room_labels: [{rooms}]\n"
        f"allowed_support_labels: [{supports}]\n\n"
        "Return exactly one JSON object with this schema: "
        '{"target_guess":"<one canonical object category>",'
        '"room_constraints":[{"label":"<allowed room>",'
        '"evidence":"<exact quote from intent>"}],'
        '"support_constraints":[{"label":"<allowed support>",'
        '"evidence":"<exact quote from intent>"}]}. '
        "Use [] when no explicit constraint is stated."
    )


def user_prompt(intent: str) -> str:
    """Backward-compatible v1 user prompt used by frozen result checks."""
    return prompt_parts(intent, PROTOCOL_V1)[1]


def _normalized_words(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", " ").replace("_", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return " ".join(text.split())


def _context_constraints(
    values: object,
    *,
    intent: str,
    allowed_labels: tuple[str, ...],
    evidence_cues: dict[str, tuple[str, ...]],
) -> list[dict[str, str]]:
    """Keep only bounded constraints whose evidence occurs in the intent."""
    if not isinstance(values, list):
        return []
    normalized_intent = _normalized_words(intent)
    allowed = {_normalized_words(label): label for label in allowed_labels}
    constraints = []
    for value in values[:3]:
        if not isinstance(value, dict):
            continue
        label_key = _normalized_words(value.get("label"))
        evidence = str(value.get("evidence") or "").strip()
        evidence_key = _normalized_words(evidence)
        label = allowed.get(label_key)
        cues = evidence_cues.get(label or "", ())
        if (
            label is None
            or not evidence_key
            or evidence_key not in normalized_intent
            or not any(_normalized_words(cue) in evidence_key for cue in cues)
        ):
            continue
        constraint = {
            "label": label,
            "evidence": evidence,
        }
        if constraint not in constraints:
            constraints.append(constraint)
    return constraints


def parse_plan(
    text: str,
    *,
    protocol_version: str = PROTOCOL_V1,
    intent: str = "",
) -> tuple[dict, str | None]:
    parsed, error = tolerant_json_parse(text)
    if not isinstance(parsed, dict):
        plan = {
            "target_guess": "",
            "candidate_objects": [],
            "likely_rooms": [],
            "strategy": "",
            "action_plan": [],
        }
        if protocol_version == PROTOCOL_V2:
            plan.update({
                "room_constraints": [],
                "support_constraints": [],
            })
        return plan, error or "response is not a JSON object"
    candidates = parsed.get("candidate_objects") or []
    rooms = parsed.get("likely_rooms") or []
    phases = parsed.get("action_plan") or []
    target_guess = str(parsed.get("target_guess") or "").strip()
    if protocol_version == PROTOCOL_V2:
        canonical = {
            _normalized_words(label): label for label in BENCHMARK_VOCABULARY
        }
        target_guess = canonical.get(_normalized_words(target_guess), "")
    plan = {
        "target_guess": target_guess,
        "candidate_objects": [
            str(value).strip() for value in candidates if str(value).strip()
        ][:6],
        "likely_rooms": [
            str(value).strip() for value in rooms if str(value).strip()
        ][:3],
        "strategy": str(parsed.get("strategy") or "").strip()[:300],
        "action_plan": [
            str(value).strip() for value in phases if str(value).strip()
        ][:4],
    }
    if protocol_version == PROTOCOL_V2:
        plan["room_constraints"] = _context_constraints(
            parsed.get("room_constraints"),
            intent=intent,
            allowed_labels=ROOM_TYPES,
            evidence_cues=ROOM_EVIDENCE_CUES,
        )
        plan["support_constraints"] = _context_constraints(
            parsed.get("support_constraints"),
            intent=intent,
            allowed_labels=SUPPORT_TYPES,
            evidence_cues=SUPPORT_EVIDENCE_CUES,
        )
    return plan, None if plan["target_guess"] else "target_guess is empty"


def selected_styles(value: str) -> tuple[str, ...]:
    if value == "all":
        return STYLES
    styles = tuple(part.strip() for part in value.split(",") if part.strip())
    invalid = sorted(set(styles) - set(STYLES))
    if invalid:
        raise ValueError(f"unsupported styles: {invalid}")
    return styles


def build_jobs(
    items: list[dict],
    styles: tuple[str, ...],
    output_dir: Path,
    force: bool,
) -> list[tuple[dict, str, str, Path]]:
    jobs = []
    for item in sorted(items, key=lambda row: row["selection_id"]):
        for style in styles:
            intent = intent_for_style(item, style)
            if not intent:
                raise ValueError(f"{item['selection_id']}/{style}: empty intent")
            path = output_dir / "records" / style / f"{item['selection_id']}.json"
            if path.is_file() and not force:
                continue
            jobs.append((item, style, intent, path))
    return jobs


def load_model(args: argparse.Namespace):
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("the selected GPU does not support bfloat16")
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=args.local_files_only,
    )
    model.to(device)
    model.eval()
    return processor, model, device


def render_chat(
    processor,
    intent: str,
    protocol_version: str = PROTOCOL_V1,
) -> str:
    system, user = prompt_parts(intent, protocol_version)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )


def bootstrap_cluster_ci(
    rows: list[dict],
    cluster_key,
    *,
    repetitions: int,
    seed: int,
) -> list[float] | None:
    groups = defaultdict(list)
    for row in rows:
        groups[cluster_key(row)].append(int(row["IM_hit"]))
    keys = sorted(groups, key=str)
    if not keys:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(repetitions):
        sample = [rng.choice(keys) for _ in keys]
        numerator = sum(sum(groups[key]) for key in sample)
        denominator = sum(len(groups[key]) for key in sample)
        values.append(numerator / denominator)
    values.sort()
    lower = values[int(0.025 * repetitions)]
    upper = values[min(repetitions - 1, int(0.975 * repetitions))]
    return [round(lower, 6), round(upper, 6)]


def aggregate(
    output_dir: Path,
    items: list[dict],
    styles: tuple[str, ...],
    *,
    model_id: str,
    revision: str,
    synonyms: dict[str, set[str]],
    bootstrap_repetitions: int,
    bootstrap_seed: int,
    configured_batch_size: int,
    protocol_version: str = PROTOCOL_V1,
) -> dict:
    item_by_id = {row["selection_id"]: row for row in items}
    rows = []
    record_paths = []
    for style in styles:
        for path in sorted((output_dir / "records" / style).glob("*.json")):
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("selection_id") in item_by_id:
                rows.append(row)
                record_paths.append(path)
    expected_keys = {
        (item["selection_id"], style) for item in items for style in styles
    }
    actual_keys = {(row["selection_id"], row["style"]) for row in rows}
    if actual_keys != expected_keys or len(rows) != len(expected_keys):
        raise ValueError(
            "open-weight intent records are incomplete or duplicated: "
            f"expected={len(expected_keys)}, actual={len(rows)}, "
            f"missing={sorted(expected_keys - actual_keys)[:5]}, "
            f"extra={sorted(actual_keys - expected_keys)[:5]}"
        )
    for row in rows:
        item = item_by_id[row["selection_id"]]
        expected_intent = intent_for_style(item, row["style"])
        expected_hit = int(intent_match(
            (row.get("prediction") or {}).get("target", ""),
            item["target_category"],
            synonyms,
        ))
        if row.get("protocol_version") != protocol_version:
            raise ValueError(f"{row['selection_id']}/{row['style']}: protocol mismatch")
        model = row.get("model") or {}
        if model.get("id") != model_id or model.get("revision") != revision:
            raise ValueError(f"{row['selection_id']}/{row['style']}: model mismatch")
        if row.get("target_category") != item["target_category"]:
            raise ValueError(f"{row['selection_id']}/{row['style']}: target mismatch")
        if row.get("intent_sha256") != sha256_bytes(expected_intent.encode("utf-8")):
            raise ValueError(f"{row['selection_id']}/{row['style']}: intent hash mismatch")
        if int(row.get("IM_hit", -1)) != expected_hit:
            raise ValueError(f"{row['selection_id']}/{row['style']}: IM mismatch")
        if int((row.get("usage") or {}).get("configured_batch_size", 0)) != (
            configured_batch_size
        ):
            raise ValueError(
                f"{row['selection_id']}/{row['style']}: configured batch-size "
                "mismatch"
            )
    by_style = defaultdict(list)
    by_mode = defaultdict(list)
    for row in rows:
        by_style[row["style"]].append(int(row["IM_hit"]))
        by_mode[row.get("intent_mode") or "UNKNOWN"].append(int(row["IM_hit"]))
    item_groups = defaultdict(list)
    for row in rows:
        item_groups[row["selection_id"]].append(int(row["IM_hit"]))
    physical_by_id = {
        item["selection_id"]: (
            item["scene_id"], item["target_representative"]
        )
        for item in items
    }
    scene_by_id = {
        item["selection_id"]: item["scene_id"] for item in items
    }
    style_intervals = {}
    for style_index, style in enumerate(styles):
        style_rows = [row for row in rows if row["style"] == style]
        style_intervals[style] = bootstrap_cluster_ci(
            style_rows,
            lambda row: row["selection_id"],
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed + 100 + style_index,
        )
    return {
        "n_records": len(rows),
        "IM": round(sum(row["IM_hit"] for row in rows) / len(rows), 6)
        if rows else None,
        "parse_failures": sum(bool(row.get("parse_error")) for row in rows),
        "per_style_IM": {
            key: round(sum(values) / len(values), 6)
            for key, values in sorted(by_style.items())
        },
        "per_intent_mode_IM": {
            key: round(sum(values) / len(values), 6)
            for key, values in sorted(by_mode.items())
        },
        "cross_style_all_correct": round(
            sum(all(values) for values in item_groups.values()) / len(item_groups),
            6,
        ),
        "bootstrap": {
            "repetitions": bootstrap_repetitions,
            "seed": bootstrap_seed,
            "overall_95_ci": {
                "selection_id_cluster": bootstrap_cluster_ci(
                    rows,
                    lambda row: row["selection_id"],
                    repetitions=bootstrap_repetitions,
                    seed=bootstrap_seed,
                ),
                "physical_goal_cluster": bootstrap_cluster_ci(
                    rows,
                    lambda row: physical_by_id[row["selection_id"]],
                    repetitions=bootstrap_repetitions,
                    seed=bootstrap_seed + 1,
                ),
                "scene_cluster": bootstrap_cluster_ci(
                    rows,
                    lambda row: scene_by_id[row["selection_id"]],
                    repetitions=bootstrap_repetitions,
                    seed=bootstrap_seed + 2,
                ),
            },
            "per_style_selection_id_cluster_95_ci": style_intervals,
        },
        "records_manifest": {
            "root": str((output_dir / "records").resolve()),
            "files": len(record_paths),
            "sha256": record_manifest_hash(record_paths, output_dir / "records"),
        },
        "policy_inputs_manifest": export_policy_inputs(output_dir, rows),
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    protocol_version = getattr(args, "protocol_version", PROTOCOL_V1)
    dataset_path = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    items = load_jsonl(dataset_path)
    if len(items) != len({row["selection_id"] for row in items}):
        raise ValueError("dataset contains duplicate selection IDs")
    if args.limit is not None:
        items = items[:args.limit]
    styles = selected_styles(args.styles)
    jobs = build_jobs(items, styles, output_dir, args.force)
    if args.count_only:
        return {
            "items": len(items),
            "styles": styles,
            "pending": len(jobs),
            "output_dir": str(output_dir),
        }

    synonyms = load_synonyms(args.synonyms.resolve())
    if args.aggregate_only:
        previous_summary = {}
        summary_path = output_dir / "summary.json"
        if summary_path.is_file():
            previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        preserve_runtime = (
            previous_summary.get("protocol_version") == protocol_version
            and previous_summary.get("dataset_sha256") == sha256_file(dataset_path)
            and previous_summary.get("model_id") == args.model_id
            and previous_summary.get("model_revision") == args.revision
        )
        summary = {
            "protocol_version": protocol_version,
            "dataset": str(dataset_path),
            "dataset_sha256": sha256_file(dataset_path),
            "model_id": args.model_id,
            "model_revision": args.revision,
            "styles": styles,
            "elapsed_s": previous_summary.get("elapsed_s") if preserve_runtime else None,
            "peak_cuda_memory_bytes": (
                previous_summary.get("peak_cuda_memory_bytes")
                if preserve_runtime else None
            ),
            **aggregate(
                output_dir,
                items,
                styles,
                model_id=args.model_id,
                revision=args.revision,
                synonyms=synonyms,
                bootstrap_repetitions=args.bootstrap_repetitions,
                bootstrap_seed=args.bootstrap_seed,
                configured_batch_size=args.batch_size,
                protocol_version=protocol_version,
            ),
        }
        save_atomic(summary, output_dir / "summary.json")
        return summary
    processor, model, device = load_model(args)
    started = time.perf_counter()
    for batch_start in range(0, len(jobs), args.batch_size):
        batch = jobs[batch_start:batch_start + args.batch_size]
        prompts = [
            render_chat(processor, intent, protocol_version)
            for _, _, intent, _ in batch
        ]
        inputs = processor(
            text=prompts,
            padding=True,
            return_tensors="pt",
        ).to(device)
        input_width = int(inputs["input_ids"].shape[1])
        batch_started = time.perf_counter()
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        latency = time.perf_counter() - batch_started
        decoded = processor.batch_decode(
            generated[:, input_width:], skip_special_tokens=True
        )
        for (item, style, intent, path), response in zip(batch, decoded):
            plan, parse_error = parse_plan(
                response,
                protocol_version=protocol_version,
                intent=intent,
            )
            prediction = {"target": plan["target_guess"]}
            if protocol_version == PROTOCOL_V2:
                prediction.update({
                    "room_constraints": plan["room_constraints"],
                    "support_constraints": plan["support_constraints"],
                })
            target = item["target_category"]
            record = {
                "protocol_version": protocol_version,
                "selection_id": item["selection_id"],
                "scene_id": item["scene_id"],
                "style": style,
                "intent": intent,
                "intent_sha256": sha256_bytes(intent.encode("utf-8")),
                "intent_mode": (item.get("_intent_mode") or {}).get("mode"),
                "target_category": target,
                "model": {
                    "id": args.model_id,
                    "revision": args.revision,
                    "dtype": "bfloat16",
                    "device": str(device),
                    "thinking": False,
                    "do_sample": False,
                    "max_new_tokens": args.max_new_tokens,
                },
                "prompt_sha256": sha256_bytes(
                    "\n".join(prompt_parts(intent, protocol_version)).encode(
                        "utf-8"
                    )
                ),
                "raw_response": response,
                "plan": plan,
                "prediction": prediction,
                "parse_error": parse_error,
                "IM_hit": int(intent_match(plan["target_guess"], target, synonyms)),
                "usage": {
                    "input_tokens_padded": input_width,
                    "output_tokens": int(generated.shape[1] - input_width),
                    "batch_size": len(batch),
                    "configured_batch_size": args.batch_size,
                    "batch_latency_s": round(latency, 6),
                },
                "created_at": now_iso(),
            }
            save_atomic(record, path)
        completed = min(batch_start + len(batch), len(jobs))
        print(f"[open-weight-intent] {completed}/{len(jobs)}", flush=True)

    summary = {
        "protocol_version": protocol_version,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "model_id": args.model_id,
        "model_revision": args.revision,
        "styles": styles,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else None
        ),
        **aggregate(
            output_dir,
            items,
            styles,
            model_id=args.model_id,
            revision=args.revision,
            synonyms=synonyms,
            bootstrap_repetitions=args.bootstrap_repetitions,
            bootstrap_seed=args.bootstrap_seed,
            configured_batch_size=args.batch_size,
            protocol_version=protocol_version,
        ),
    }
    save_atomic(summary, output_dir / "summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO / "results/dataset_curated_v3/selected_500_intents.jsonl",
    )
    parser.add_argument(
        "--synonyms",
        type=Path,
        default=REPO / "eval/vocab/category_synonyms.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            REPO / "results/open_weight_intent_curated_v3/qwen3_5_4b"
        ),
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--styles", default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260902)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--protocol-version",
        choices=(PROTOCOL_V1, PROTOCOL_V2),
        default=PROTOCOL_V1,
    )
    args = parser.parse_args()
    if (
        args.batch_size <= 0
        or args.max_new_tokens <= 0
        or args.bootstrap_repetitions <= 0
    ):
        parser.error(
            "batch-size, max-new-tokens, and bootstrap-repetitions must be positive"
        )
    return args


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
