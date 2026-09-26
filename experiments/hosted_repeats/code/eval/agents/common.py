"""Shared helpers for agent scripts: item loader, atomic write, prompt loader,
tolerant JSON parse, output-path conventions.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

# All paths relative to project root (derived via __file__ so the tree can
# relocate without code edits). Override external dataset roots via env vars:
#   INTENTEQA_USD_ROOT, INTENTEQA_METAROOT.
_THIS = Path(__file__).resolve()
REPO = _THIS.parents[2]  # eval/agents/common.py → IntentEQA/
EVAL_DIR = REPO / "eval"
DATASET_ROOT = REPO / "results/dataset"      # canonical per-scene dataset
DATASET_JSONL = DATASET_ROOT / "selected_500_intents.jsonl"
PROMPTS_DIR = EVAL_DIR / "judge/prompts"
# Override via EVAL_OUT_DIR env var. Caller is expected to set this when
# running successive experiments — the default writes into one shared dir,
# which mixes baselines across runs.
EPISODES_OUT = Path(os.environ.get("EVAL_OUT_DIR",
                                    str(REPO / "results/eval_out")))
if not EPISODES_OUT.is_absolute():
    EPISODES_OUT = REPO / EPISODES_OUT

# External (not part of the shipped dataset — USD scenes, VLNTube freemaps)
USD_ROOT = Path(os.environ.get(
    "INTENTEQA_USD_ROOT",
    "/path/to/workspace/datasets/vlntube/TataServices",
))
METAROOT = Path(os.environ.get(
    "INTENTEQA_METAROOT",
    "/path/to/workspace/datasets/vlntube/TaTaMeta/metadata_train",
))


def scene_photo_path(scene_id: str, photo_rel: str) -> Path:
    """Resolve a photo reference from the 500-item jsonl to the dataset/ layout."""
    return DATASET_ROOT / scene_id / "photos" / Path(photo_rel).name


def scene_manifest_path(scene_id: str) -> Path:
    return DATASET_ROOT / scene_id / "manifest.json"


def scene_intents_path(scene_id: str) -> Path:
    return DATASET_ROOT / scene_id / "intents.json"


STYLE_KEYS = ("formal_en", "natural_en", "casual_en", "emotional_en")
STYLES = ("formal", "natural", "casual", "emotional")


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


def render_prompt(name: str, **vars) -> str:
    """Render a prompt by replacing <<KEY>> markers with values.

    Avoids str.format's brace-collision problems with literal `{...}` in prompts
    (e.g., rubric notation `{1,2,3,4,5}` or JSON schema examples).
    """
    text = load_prompt(name)
    for k, v in vars.items():
        text = text.replace(f"<<{k.upper()}>>", str(v))
    return text


def load_items() -> list[dict]:
    """Load the canonical 500-item set."""
    return [json.loads(l) for l in DATASET_JSONL.open(encoding="utf-8")]


def tolerant_json_parse(text: str):
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.lstrip("`")
        for kw in ("json", "JSON"):
            if cleaned.startswith(kw):
                cleaned = cleaned[len(kw):]
                break
        cleaned = cleaned.strip()
    start = cleaned.find("{")
    if start < 0:
        return None, "no opening brace"
    try:
        result, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        return result, None
    except json.JSONDecodeError as e:
        return None, str(e)


def intent_for_style(item: dict, style: str) -> str:
    """Return the English intent string for the requested style."""
    key = f"{style}_en"
    return str(item.get(key, "") or "").strip()


def episode_dir(scene_id: str, tier: str, model: str, style: str,
                selection_id: str) -> Path:
    """Per-episode directory holding all artifacts for ONE (tier, model, style,
    sel) episode. Layout:

        results/eval_out/<scene>/<tier>_<model>/<sel>/<style>/
            record.json              (main record, atomic-saved at end)
            final.png                (last RGB frame)
            step_01_overlay.jpg      (waypoint overlay shown to VLM)  [vlm tier]
            step_01_prompt.txt
            step_01_response.txt
            step_02_*
            ...

    Each SEL groups its 4 styles together so a reviewer can compare
    formal/natural/casual/emotional side-by-side per intent. Tier prefix on the
    model dir avoids oracle/vlm/blind collisions when they all use the same
    VLM model (e.g. gpt5_4)."""
    model_dir = f"{tier}_{model}" if tier in ("oracle", "blind", "vlm") else model
    return EPISODES_OUT / scene_id / model_dir / selection_id / style


def episode_path(scene_id: str, tier: str, model: str, style: str,
                 selection_id: str) -> Path:
    """Path to the canonical record JSON inside the episode directory."""
    return episode_dir(scene_id, tier, model, style, selection_id) / "record.json"


def save_atomic(record: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def now_iso() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
