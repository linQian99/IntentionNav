"""Validate intent annotations and build the IntentionNav benchmark file."""

import argparse
import json
import os
import re
from collections import defaultdict


CATEGORY_SYNONYMS = {
    "book": ["book", "书", "书本", "书籍", "读", "翻", "阅读", "read", "flip through"],
    "plate": ["plate", "盘子", "盘", "碟子", "碟", "盛", "装菜", "端菜"],
    "wine_set": [
        "wine", "酒", "酒具", "酒杯", "wine glass", "wine set", "倒酒", "品酒",
        "喝", "饮", "drink", "pour", "sip", "beverage",
    ],
    "cosmetic": ["cosmetic", "化妆品", "化妆", "makeup", "补妆", "上妆"],
    "ornament": ["ornament", "装饰品", "摆件", "装饰"],
    "picture_frame": ["frame", "相框", "picture frame", "画框"],
    "spoon": ["spoon", "勺子", "勺", "汤匙", "舀"],
    "fork": ["fork", "叉子", "叉"],
    "vase": ["vase", "花瓶", "插花"],
    "cup": [
        "cup", "杯子", "杯", "mug", "喝", "饮", "倒", "泡", "drink", "pour",
        "brew", "sip", "beverage", "茶", "tea", "咖啡", "coffee", "饮料",
    ],
    "flower": ["flower", "花", "鲜花", "浇花", "water the flower"],
    "bathroom_product": [
        "shampoo", "soap", "洗发水", "肥皂", "沐浴", "洗澡", "洗手",
        "shower", "wash hands",
    ],
    "kettle": ["kettle", "水壶", "壶", "烧水", "烫水", "boil water"],
    "tray": ["tray", "托盘"],
    "table_lamp": [
        "lamp", "灯", "台灯", "开灯", "关灯", "调灯", "turn on light",
        "turn off light", "dim the light", "lighting",
    ],
    "daily_equipment": ["equipment", "设备"],
    "clock": ["clock", "钟", "时钟", "看时间", "check the time"],
    "pillow": ["pillow", "枕头", "靠垫"],
    "electric_appliance": ["appliance", "电器"],
    "cookware": ["pot", "pan", "锅", "炒菜", "cook", "煮"],
    "kitchenware": ["kitchenware", "厨具", "餐具"],
    "office_supply": ["pen", "笔", "办公用品", "文具", "写字", "签字", "write"],
}


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_base_category(object_id):
    match = re.match(r"([a-zA-Z_]+?)(_\d+)(/.*)?$", object_id)
    if match:
        return match.group(1)
    name = object_id.split("/")[0]
    match = re.match(r"([a-zA-Z_]+)", name)
    return match.group(1) if match else name


def check_intent_leaks_object(variants, target_categories):
    """Return (leaked, reason) if an intent directly names the target."""
    for cat in target_categories:
        synonyms = CATEGORY_SYNONYMS.get(cat, [cat])
        for key, text in variants.items():
            if not text:
                continue
            lowered = text.lower()
            for synonym in synonyms:
                if key.endswith("_en"):
                    if synonym.lower() in lowered:
                        return True, f"{key} contains '{synonym}'"
                elif synonym in text:
                    return True, f"{key} contains '{synonym}'"
    return False, None


def extract_variants(intent):
    if isinstance(intent.get("intent_variants"), dict):
        return intent["intent_variants"]
    return {
        "natural_zh": intent.get("intent_zh", ""),
        "natural_en": intent.get("intent_en", ""),
    }


def load_manifest(capture_dir, scene_id):
    manifest = load_json(os.path.join(capture_dir, scene_id, "capture_manifest.json"), [])
    return {
        (entry.get("surface_id"), entry.get("target_category")): entry
        for entry in manifest
    }


def build_target_details(annotation, manifest_entry, object_dict):
    details = []
    if manifest_entry:
        details = list(manifest_entry.get("target_details") or [])
    if not details:
        for object_id in annotation.get("all_targets_in_category", []):
            obj = object_dict.get(object_id, {})
            details.append({
                "object_id": object_id,
                "category": get_base_category(object_id),
                "room": obj.get("room", annotation.get("room", "unknown")),
                "position": obj.get("position"),
                "bbox": obj.get("bbox"),
            })
    return details


def validate_scene(scene_id, args):
    ann_path = os.path.join(args.intents_dir, scene_id, "intent_annotations.json")
    annotations = load_json(ann_path, [])
    if not annotations:
        return [], []
    if not isinstance(annotations, list):
        return [], [f"{scene_id}: intent_annotations.json must be a list"]

    object_dict = load_json(
        os.path.join(args.scene_summary, scene_id, "object_dict.json"), {}
    )
    manifest_by_key = load_manifest(args.capture_dir, scene_id) if args.capture_dir else {}

    episodes = []
    issues = []

    for annotation_index, annotation in enumerate(annotations):
        surface_id = annotation.get("surface_id")
        target_category = annotation.get("target_category")
        key = (surface_id, target_category)
        manifest_entry = manifest_by_key.get(key)

        target_ids = annotation.get("all_targets_in_category") or []
        if not target_ids:
            issues.append(f"{scene_id} {surface_id}->{target_category}: no target ids")
            continue

        missing = [object_id for object_id in target_ids if object_id not in object_dict]
        if missing:
            issues.append(
                f"{scene_id} {surface_id}->{target_category}: missing objects {missing}"
            )
            continue

        photo_rel = annotation.get("photo")
        if args.capture_dir and photo_rel:
            photo_abs = os.path.join(args.capture_dir, scene_id, photo_rel)
            if not os.path.exists(photo_abs):
                issues.append(
                    f"{scene_id} {surface_id}->{target_category}: missing photo {photo_rel}"
                )
                continue

        target_details = build_target_details(annotation, manifest_entry, object_dict)
        target_categories = sorted({
            detail.get("category") or get_base_category(detail["object_id"])
            for detail in target_details
            if detail.get("object_id")
        })

        intents = annotation.get("intents") or []
        for intent_idx, intent in enumerate(intents):
            variants = extract_variants(intent)
            leaked, reason = check_intent_leaks_object(variants, target_categories)
            if leaked:
                issues.append(
                    f"{scene_id} {surface_id}->{target_category} intent {intent_idx}: "
                    f"leaks target ({reason})"
                )
                continue

            episodes.append({
                "episode_id": None,
                "scene_id": scene_id,
                "sequence_id": annotation.get("sequence_id"),
                "surface_id": surface_id,
                "surface_category": annotation.get("surface_category"),
                "room": annotation.get("room", "unknown"),
                "target_category": target_category,
                "target_representative": annotation.get("target_representative"),
                "all_targets_in_category": target_ids,
                "target_objects": target_details,
                "photo": os.path.join(scene_id, photo_rel) if photo_rel else None,
                "specific_description": annotation.get("specific_description", {}),
                "intent_variants": variants,
                "difficulty": intent.get("difficulty", "medium"),
                "reasoning": intent.get("reasoning", ""),
                "source": {
                    "annotation_file": os.path.join(scene_id, "intent_annotations.json"),
                    "annotation_index": annotation_index,
                    "intent_index": intent_idx,
                },
            })

    return episodes, issues


def assign_splits(episodes, splits_data):
    trainval = set(splits_data.get("trainval", []))
    val_unseen = set(splits_data.get("val_unseen", []))
    test = set(splits_data.get("test", []))

    import random

    random.seed(42)
    trainval_list = sorted(trainval)
    random.shuffle(trainval_list)
    split_idx = int(len(trainval_list) * 0.8)
    train = set(trainval_list[:split_idx])
    val_seen = set(trainval_list[split_idx:])

    for episode in episodes:
        scene_id = episode["scene_id"]
        if scene_id in train:
            episode["split"] = "train"
        elif scene_id in val_seen:
            episode["split"] = "val_seen"
        elif scene_id in val_unseen:
            episode["split"] = "val_unseen"
        elif scene_id in test:
            episode["split"] = "test"
        else:
            episode["split"] = "unknown"
    return episodes


def add_episode_ids(episodes):
    episodes = sorted(
        episodes,
        key=lambda e: (
            e["scene_id"],
            e.get("sequence_id") or 0,
            e.get("target_category") or "",
            e["source"]["intent_index"],
        ),
    )
    for idx, episode in enumerate(episodes, start=1):
        episode["episode_id"] = f"INTNAV_{idx:06d}"
    return episodes


def summarize(episodes, issues):
    split_counts = defaultdict(int)
    difficulty_counts = defaultdict(int)
    category_counts = defaultdict(int)
    for episode in episodes:
        split_counts[episode.get("split", "unknown")] += 1
        difficulty_counts[episode.get("difficulty", "medium")] += 1
        category_counts[episode.get("target_category", "unknown")] += 1
    return {
        "total_episodes": len(episodes),
        "by_split": dict(split_counts),
        "by_difficulty": dict(difficulty_counts),
        "by_target_category": dict(
            sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))
        ),
        "total_issues_filtered": len(issues),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: validate annotations and build IntentionNav benchmark"
    )
    parser.add_argument("--intents-dir", required=True,
                        help="Root with kujiale_*/intent_annotations.json")
    parser.add_argument("--scene-summary", required=True,
                        help="Path to kujiale_scene_summary/")
    parser.add_argument("--splits-file", required=True,
                        help="Path to scene_splits.json")
    parser.add_argument("--output", required=True,
                        help="Output path for intentionnav_benchmark.json")
    parser.add_argument("--capture-dir", default=None,
                        help="Optional Phase 1 root for photo existence checks")
    parser.add_argument("--scene", default=None, help="Single scene ID for testing")
    parser.add_argument("--issues-output", default=None,
                        help="Optional JSONL path for filtered issue details")
    args = parser.parse_args()

    splits_data = load_json(args.splits_file, {})
    if args.scene:
        scene_ids = [args.scene]
    else:
        scene_ids = sorted(
            name for name in os.listdir(args.intents_dir)
            if os.path.isdir(os.path.join(args.intents_dir, name))
        )

    all_episodes = []
    all_issues = []
    for scene_id in scene_ids:
        episodes, issues = validate_scene(scene_id, args)
        if episodes:
            print(f"[OK] {scene_id}: {len(episodes)} benchmark episodes")
        elif issues:
            print(f"[WARN] {scene_id}: 0 episodes, {len(issues)} issue(s)")
        all_episodes.extend(episodes)
        all_issues.extend(issues)

    all_episodes = assign_splits(all_episodes, splits_data)
    all_episodes = add_episode_ids(all_episodes)
    stats = summarize(all_episodes, all_issues)

    dataset = {
        "name": "IntentionNav",
        "version": "1.0",
        "description": "Implicit-intention object navigation benchmark built from VLNTube scenes.",
        "statistics": stats,
        "episodes": all_episodes,
    }
    save_json(dataset, args.output)

    if args.issues_output:
        os.makedirs(os.path.dirname(args.issues_output) or ".", exist_ok=True)
        with open(args.issues_output, "w", encoding="utf-8") as f:
            for issue in all_issues:
                f.write(json.dumps({"issue": issue}, ensure_ascii=False) + "\n")

    print()
    print("=" * 60)
    print("IntentionNav benchmark built")
    print(f"  Episodes: {stats['total_episodes']}")
    print(f"  By split: {stats['by_split']}")
    print(f"  By difficulty: {stats['by_difficulty']}")
    print(f"  Issues filtered: {stats['total_issues_filtered']}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
