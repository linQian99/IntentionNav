"""Shared helpers for agent scripts: item loader, atomic write, prompt loader,
tolerant JSON parse, output-path conventions.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
from pathlib import Path

# All paths relative to project root (derived via __file__ so the tree can
# relocate without code edits). Override external dataset roots via env vars:
#   INTENTIONNAV_USD_ROOT, INTENTIONNAV_METAROOT.
_THIS = Path(__file__).resolve()
REPO = _THIS.parents[2]  # project root
EVAL_DIR = REPO / "eval"
DEFAULT_DATASET_ROOT = REPO / "results/dataset"  # frozen per-scene dataset


def _configured_path(env_name: str, default: Path) -> Path:
    path = Path(os.environ.get(env_name, str(default)))
    return path if path.is_absolute() else REPO / path


# Keep the frozen release as the default while allowing quality-filtered or
# future versioned splits without editing agent code.
DATASET_ROOT = _configured_path(
    "INTENTIONNAV_DATASET_ROOT",
    DEFAULT_DATASET_ROOT,
)
DATASET_JSONL = _configured_path(
    "INTENTIONNAV_DATASET_JSONL",
    DATASET_ROOT / "selected_500_intents.jsonl",
)
EPISODES_FILE = _configured_path(
    "INTENTIONNAV_EPISODES_JSONL",
    EVAL_DIR / "splits/episodes.jsonl",
)
PROMPTS_DIR = EVAL_DIR / "judge/prompts"
# Override via EVAL_OUT_DIR env var. Caller is expected to set this when
# running successive experiments — the default writes into one shared dir,
# which mixes baselines across runs.
EPISODES_OUT = _configured_path(
    "EVAL_OUT_DIR",
    REPO / "results/eval_out",
)

# External (not part of the shipped dataset — USD scenes, VLNTube freemaps).
USD_ROOT = Path(os.environ.get(
    "INTENTIONNAV_USD_ROOT",
    str(REPO / "data/VLNVerse_scene"),
))
METAROOT = Path(os.environ.get(
    "INTENTIONNAV_METAROOT",
    str(REPO / "data/SceneMeta"),
))
SCENE_SUMMARY_ROOT = Path(os.environ.get(
    "INTENTIONNAV_SCENE_SUMMARY",
    str(REPO / "data/SceneSummary/kujiale_scene_summary"),
))


def scene_photo_path(scene_id: str, photo_rel: str) -> Path:
    """Resolve a photo reference from the 500-item jsonl to the dataset/ layout."""
    referenced = DATASET_ROOT / scene_id / photo_rel
    if referenced.is_file():
        return referenced
    # Frozen releases flatten ``surface_photos/...`` into ``photos/`` during
    # packaging.  Retain that layout as a compatibility fallback.
    return DATASET_ROOT / scene_id / "photos" / Path(photo_rel).name


def scene_manifest_path(scene_id: str) -> Path:
    return DATASET_ROOT / scene_id / "manifest.json"


def scene_intents_path(scene_id: str) -> Path:
    return DATASET_ROOT / scene_id / "intents.json"


STYLE_KEYS = ("formal_en", "natural_en", "casual_en", "emotional_en")
STYLES = ("formal", "natural", "casual", "emotional")

ROBUSTNESS_MODES = ("none", "target_absent")
ROBUSTNESS_ARG_CHOICES = (
    "none",
    "off",
    "false",
    "0",
    "target_absent",
    "endpoint_absent",
    "target_missing",
    "goal_missing",
)

_ROBUSTNESS_ALIASES = {
    "": "none",
    "none": "none",
    "off": "none",
    "false": "none",
    "0": "none",
    "target_absent": "target_absent",
    "endpoint_absent": "target_absent",
    "target_missing": "target_absent",
    "goal_missing": "target_absent",
}

_TARGET_ABSENT_FALLBACK_POOL = (
    "air_conditioner",
    "air_purifier",
    "bathtub",
    "bed",
    "bench",
    "book",
    "bookshelf",
    "bowl",
    "builtin_oven",
    "cabinet",
    "ceiling_light",
    "coffee_maker",
    "curtain",
    "desk",
    "dining_table",
    "dish_washer",
    "electric_cooker",
    "floor_lamp",
    "flower",
    "fridge",
    "kettle",
    "knife",
    "menorah",
    "microwave",
    "mirror",
    "piano",
    "plate",
    "range_hood",
    "shoe_rack",
    "sofa",
    "spoon",
    "table",
    "table_lamp",
    "television",
    "toilet",
    "washing_machine",
    "water_heater",
    "wine_set",
)


def normalize_robustness_mode(mode: str | None) -> str:
    key = str(mode or "none").strip().lower().replace("-", "_")
    if key not in _ROBUSTNESS_ALIASES:
        valid = ", ".join(ROBUSTNESS_ARG_CHOICES)
        raise ValueError(f"unknown robustness mode {mode!r}; valid: {valid}")
    return _ROBUSTNESS_ALIASES[key]


def robustness_output_model_name(model_key: str, mode: str | None) -> str:
    mode = normalize_robustness_mode(mode)
    if mode == "none":
        return model_key
    return f"{model_key}_robust_{mode}"


def normalize_category(label: object) -> str:
    text = str(label or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = re.sub(r"_\d+(?:/.*)?$", "", text)
    return text.split("/", 1)[0]


def _scene_object_dict_path(scene_id: str) -> Path:
    return SCENE_SUMMARY_ROOT / scene_id / "object_dict.json"


def load_scene_categories(scene_id: str) -> set[str]:
    path = _scene_object_dict_path(scene_id)
    if not path.exists():
        raise FileNotFoundError(
            f"object_dict.json not found for {scene_id}: {path}. "
            "Set INTENTIONNAV_SCENE_SUMMARY to the kujiale_scene_summary root "
            "before using --robustness target_absent."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    cats: set[str] = set()
    for object_id, data in raw.items():
        cats.add(normalize_category(object_id))
        if isinstance(data, dict):
            cats.add(normalize_category(data.get("category")))
            cats.add(normalize_category(data.get("instance_id")))
    return {c for c in cats if c}


def _target_absent_pool() -> list[str]:
    cats = set(_TARGET_ABSENT_FALLBACK_POOL)
    if DATASET_JSONL.exists():
        try:
            for line in DATASET_JSONL.open(encoding="utf-8"):
                if not line.strip():
                    continue
                item = json.loads(line)
                cat = normalize_category(item.get("target_category"))
                if cat:
                    cats.add(cat)
        except Exception:
            pass
    return sorted(cats)


def make_target_absent_robustness(
    mode: str | None,
    *,
    item: dict,
    episode_meta: dict,
    seed: int = 0,
    style: str = "",
) -> dict | None:
    """Return metadata for endpoint-object-absent ObjectNav episodes.

    The requested target category is chosen from known household targets but
    verified absent from the current scene's object_dict. The scene/USD itself
    is unchanged.
    """
    mode = normalize_robustness_mode(mode)
    if mode == "none":
        return None
    if mode != "target_absent":
        raise ValueError(f"unsupported robustness mode: {mode}")

    scene_id = str(item.get("scene_id") or episode_meta.get("scene_id") or "")
    scene_categories = load_scene_categories(scene_id)
    original_target = normalize_category(item.get("target_category"))
    pool = [
        cat for cat in _target_absent_pool()
        if cat and cat != original_target and cat not in scene_categories
    ]
    if not pool:
        raise ValueError(f"no absent target category available for {scene_id}")
    salt = f"{item.get('selection_id', '')}|{style}|{seed}|{mode}"
    idx = int(hashlib.sha1(salt.encode("utf-8")).hexdigest()[:8], 16) % len(pool)
    absent_target = pool[idx]
    note = (
        f"The requested target object category is '{absent_target}'. "
        "It may be absent from the scene. Search normally, but do not STOP "
        "unless that exact object is clearly visible."
    )
    robust_episode_meta = dict(episode_meta)
    robust_episode_meta.update(
        {
            "source_target_category": item.get("target_category"),
            "source_target_object_id": episode_meta.get("target_object_id"),
            "source_target_position": episode_meta.get("target_position"),
            "source_target_room": episode_meta.get("target_room"),
            "target_category": absent_target,
            "target_object_id": None,
            "target_position": None,
            "target_room": None,
            "geodesic_to_target": None,
            "euclidean_to_target": None,
        }
    )
    return {
        "mode": mode,
        "seed": int(seed),
        "type": "endpoint_object_absent",
        "absent_target_category": absent_target,
        "source_target_category": item.get("target_category"),
        "scene_summary_root": str(SCENE_SUMMARY_ROOT),
        "scene_category_count": len(scene_categories),
        "prompt_note": note,
        "episode_meta": robust_episode_meta,
    }


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
