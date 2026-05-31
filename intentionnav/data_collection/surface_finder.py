"""Phase 0: Find support surfaces with small manipulable objects on them.

Parses object_dict.json per scene, identifies surfaces that have small objects
sitting on them (via "on" relationship), and outputs surface_targets.json.

In the scene graph, when object A's nearby_objects has B with rel="on",
it means B is physically on top of A (A supports B).
"""

import argparse
import json
import os
import re
from collections import defaultdict


# --- Strategy: open by default — only reject pure structural / generic carriers ---

# Structural elements (never targets — building shell, openings)
STRUCTURAL_CATEGORIES = {
    "wall", "floor", "ceiling", "celling",  # dataset has both spellings
    "door", "window", "doorsill", "entrance",
    "unknown", "background", "skirting", "wall_decoration",
    "pillar", "stair",
}

# Generic carriers / decorative elements that aren't themselves intent targets.
# (cup-on-tablecloth: target = cup, not tablecloth; tray same logic)
GENERIC_CARRIER_CATEGORIES = {
    "tablecloth", "tray", "tv_stand",
}

# Fuzzy umbrella categories. Language like "pick up the ornament" or
# "find the daily_equipment" is too ambiguous for an intent task — the
# referent would be any of dozens of household items. Drop so downstream
# Gemini prompts don't see them as candidate targets.
FUZZY_UMBRELLA_CATEGORIES = {
    "ornament", "daily_equipment", "bathroom_product", "office_supply",
    "kitchenware", "cookware", "other_cooker", "hardware_decoration",
    "decorative_box", "electric_appliance", "other_light",
    "other_furniture", "tooling",
}

# Standalone functional items that ARE valid intent targets themselves.
# capture_surfaces aims the camera at the item's centroid (no "on" relation needed).
# Pulled from the comprehensive Kujiale category list — anything you might
# meaningfully want to interact with in the home.
STANDALONE_TARGET_CATEGORIES = {
    # Beds / seating
    "bed", "sofa", "couch", "chair", "stool", "bench",
    # Storage / wardrobes (you might want to "put away clothes", "find a book")
    "cabinet", "wardrobe", "closet", "cupboard", "bookshelf",
    "storage", "shelf", "night_stand",
    # Tables (gathering / working / dining intents)
    "table", "dining_table", "tea_table", "desk",
    # Bathroom fixtures
    "basin", "bathtub", "shower", "closestool", "toilet",
    # Kitchen appliances
    "refrigerator", "fridge", "freezer", "oven", "builtin_oven",
    "range_hood", "microwave", "dish_washer", "washing_machine",
    "kettle", "bread_machine", "coffee_maker", "induction_cooker",
    "electric_cooker", "water_cooler",
    # Entertainment / electronics
    "television", "screen", "sound", "piano", "fireplace", "computer",
    # Climate / utility
    "air_conditioner", "water_heater", "curtain", "air_purifier",
    # Lighting
    "floor_lamp", "wall_light",
    # Other functional
    "menorah",
}

# Wall-mounted / high-placed targets for "above" relation
WALL_MOUNTED_CATEGORIES = {
    "mirror", "picture_frame", "painting", "chandelier", "clock",
}

# Surfaces where items sit ON TOP (used to find "small object on surface")
SUPPORT_SURFACE_CATEGORIES = {
    "table", "cabinet", "shelf", "dining_table", "desk", "stool",
    "bed", "sofa", "storage", "tea_table",
    "tablecloth", "tray",
    "chair",
    "tv_stand",
}

# Reference surfaces for "above" relationships
ABOVE_REFERENCE_CATEGORIES = {
    "bed", "sofa", "dining_table", "table", "cabinet", "basin",
    "desk", "tea_table", "piano", "fireplace", "tv_stand",
}

# Combined blacklist: things that are NEVER a target object.
# (structural + generic carriers, but NOT standalone-target items even if they
# can also be support surfaces — those are still valid as their own targets.)
CATEGORY_BLACKLIST = (
    STRUCTURAL_CATEGORIES | GENERIC_CARRIER_CATEGORIES | FUZZY_UMBRELLA_CATEGORIES
)

# Maximum bbox dimension for a "target object" (in meters).
# Anything bigger than this is treated as furniture, not a target.
MAX_TARGET_DIMENSION = 1.0  # ~1m max side length

# Accepted spatial relationships — only VISIBLE items:
#   "on" = object sits on top of surface (visible)
#   "above" = object mounted/hanging above reference (visible, e.g. mirror above basin)
# NOTE: "in" excluded because items inside closed containers aren't visible in photos
VALID_RELATIONS = {"on", "above"}


def get_base_category(object_id: str) -> str:
    """Extract base category from object_id like 'cabinet_0001/Meshes' -> 'cabinet'."""
    match = re.match(r"([a-zA-Z_]+?)(_\d+)(/.*)?$", object_id)
    if match:
        return match.group(1)
    # Fallback: strip everything after first digit sequence
    name = object_id.split("/")[0]
    match2 = re.match(r"([a-zA-Z_]+)", name)
    return match2.group(1) if match2 else name


def find_surfaces_with_objects(object_dict: dict) -> list:
    """Find all surfaces/containers with small objects (on or in them).

    Returns list of entries, each with surface info and target objects with
    their relation ("on" or "in"). Objects_on_surface remains the key name
    for backward compatibility but includes both "on" and "in" targets.
    """
    surfaces = defaultdict(lambda: {
        "surface_id": None,
        "surface_category": None,
        "room": None,
        "position": None,
        "bbox": None,
        "objects_on_surface": [],
    })

    for obj_id, obj_data in object_dict.items():
        obj_cat = get_base_category(obj_id)

        # Skip purely structural elements (walls/floors etc.) as surfaces too
        if obj_cat in STRUCTURAL_CATEGORIES:
            continue

        # Accept this object as a candidate if it can be a reference for any relation
        # NOTE: bed/sofa/cabinet are in LARGE_FURNITURE_CATEGORIES (target blacklist)
        # but they ARE valid surfaces. So we use SUPPORT_SURFACE check, not blacklist.
        is_support = obj_cat in SUPPORT_SURFACE_CATEGORIES
        is_above_ref = obj_cat in ABOVE_REFERENCE_CATEGORIES
        if not (is_support or is_above_ref):
            continue

        nearby = obj_data.get("nearby_objects", {})
        for neighbor_id, rel_info in nearby.items():
            rel = rel_info[0] if isinstance(rel_info, list) else rel_info
            if rel not in VALID_RELATIONS:
                continue
            # Relation consistency
            if rel == "on" and not is_support:
                continue
            if rel == "above" and not is_above_ref:
                continue

            neighbor_cat = get_base_category(neighbor_id)
            # Blacklist: structural + large furniture excluded as targets
            if neighbor_cat in CATEGORY_BLACKLIST:
                continue
            # For above: must be a wall-mounted decoration (specific list)
            if rel == "above" and neighbor_cat not in WALL_MOUNTED_CATEGORIES:
                continue

            # Get neighbor data + size filter (skip objects too large to be "targets")
            neighbor_data = object_dict.get(neighbor_id, {})
            n_min = neighbor_data.get("min_points")
            n_max = neighbor_data.get("max_points")
            if n_min and n_max:
                size = max(n_max[0] - n_min[0], n_max[1] - n_min[1], n_max[2] - n_min[2])
                if size > MAX_TARGET_DIMENSION:
                    continue

            key = obj_id
            entry = surfaces[key]
            entry["surface_id"] = obj_id
            entry["surface_category"] = obj_cat
            entry["room"] = obj_data.get("room", "unknown")
            entry["position"] = obj_data.get("position")
            entry["bbox"] = {
                "min": obj_data.get("min_points"),
                "max": obj_data.get("max_points"),
            }
            entry["objects_on_surface"].append({
                "object_id": neighbor_id,
                "category": neighbor_cat,
                "relation": rel,  # "on" or "in"
                "position": neighbor_data.get("position"),
                "bbox": {
                    "min": neighbor_data.get("min_points"),
                    "max": neighbor_data.get("max_points"),
                },
            })

    result = [v for v in surfaces.values() if len(v["objects_on_surface"]) > 0]

    # Pattern 2: standalone large items that ARE intent targets themselves
    # (TV, fridge, oven, ...). Add as a "self-surface" entry so capture aims
    # the camera directly at the item.
    for obj_id, obj_data in object_dict.items():
        cat = get_base_category(obj_id)
        if cat not in STANDALONE_TARGET_CATEGORIES:
            continue
        pos = obj_data.get("position")
        if not pos:
            continue
        result.append({
            "surface_id": obj_id,
            "surface_category": cat,
            "room": obj_data.get("room", "unknown"),
            "position": pos,
            "bbox": {
                "min": obj_data.get("min_points"),
                "max": obj_data.get("max_points"),
            },
            # The standalone item is "on itself" — capture aims at it directly.
            "objects_on_surface": [{
                "object_id": obj_id,
                "category": cat,
                "relation": "self",
                "position": pos,
                "bbox": {
                    "min": obj_data.get("min_points"),
                    "max": obj_data.get("max_points"),
                },
            }],
        })

    # Per-scene category dedup: each target_category is annotated only ONCE
    # (first occurrence wins, in insertion order). Avoids 12× curtain entries
    # in a scene with one curtain per window. Small-on-surface entries come
    # before standalone entries (so a vase-on-cabinet is preferred over the
    # cabinet itself if both exist in the scene).
    seen_cats = set()
    deduped = []
    for s in result:
        kept_objs = []
        for o in s["objects_on_surface"]:
            if o["category"] in seen_cats:
                continue
            seen_cats.add(o["category"])
            kept_objs.append(o)
        if kept_objs:
            s_copy = dict(s)
            s_copy["objects_on_surface"] = kept_objs
            deduped.append(s_copy)
    return deduped


def process_scene(scene_summary_dir: str, scene_id: str, output_dir: str) -> dict:
    """Process a single scene and write surface_targets.json."""
    obj_path = os.path.join(scene_summary_dir, scene_id, "object_dict.json")
    if not os.path.exists(obj_path):
        print(f"[SKIP] {scene_id}: object_dict.json not found")
        return None

    with open(obj_path, "r", encoding="utf-8") as f:
        object_dict = json.load(f)

    surfaces = find_surfaces_with_objects(object_dict)

    if not surfaces:
        print(f"[SKIP] {scene_id}: no surfaces with small objects found")
        return None

    # Write output
    scene_out = os.path.join(output_dir, scene_id)
    os.makedirs(scene_out, exist_ok=True)
    out_path = os.path.join(scene_out, "surface_targets.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(surfaces, f, ensure_ascii=False, indent=2)

    # Stats
    total_objects = sum(len(s["objects_on_surface"]) for s in surfaces)
    rooms = set(s["room"] for s in surfaces)
    categories = defaultdict(int)
    for s in surfaces:
        for o in s["objects_on_surface"]:
            categories[o["category"]] += 1

    stats = {
        "scene_id": scene_id,
        "num_surfaces": len(surfaces),
        "num_objects_on_surfaces": total_objects,
        "rooms": list(rooms),
        "object_categories": dict(categories),
    }

    print(f"[OK] {scene_id}: {len(surfaces)} surfaces, {total_objects} small objects, rooms: {rooms}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Find support surfaces with small objects")
    parser.add_argument("--scene-summary", required=True, help="Path to kujiale_scene_summary/")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--scene", default=None, help="Single scene ID (for testing)")
    parser.add_argument("--splits-file", default=None, help="Path to scene_splits.json (filter to trainval)")
    args = parser.parse_args()

    # Get scene list
    if args.scene:
        scene_ids = [args.scene]
    elif args.splits_file:
        with open(args.splits_file, "r") as f:
            splits = json.load(f)
        scene_ids = sorted(splits.get("trainval", []) + splits.get("val_unseen", []) + splits.get("test", []))
    else:
        scene_ids = sorted([
            d for d in os.listdir(args.scene_summary)
            if os.path.isdir(os.path.join(args.scene_summary, d))
        ])

    os.makedirs(args.output_dir, exist_ok=True)

    all_stats = []
    for sid in scene_ids:
        stats = process_scene(args.scene_summary, sid, args.output_dir)
        if stats:
            all_stats.append(stats)

    # Summary
    print(f"\n{'='*60}")
    print(f"Total: {len(all_stats)}/{len(scene_ids)} scenes have surfaces with small objects")
    total_surfaces = sum(s["num_surfaces"] for s in all_stats)
    total_objects = sum(s["num_objects_on_surfaces"] for s in all_stats)
    print(f"Total surfaces: {total_surfaces}, total small objects: {total_objects}")

    # Category distribution
    cat_totals = defaultdict(int)
    for s in all_stats:
        for cat, cnt in s["object_categories"].items():
            cat_totals[cat] += cnt
    print(f"\nObject category distribution:")
    for cat, cnt in sorted(cat_totals.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}")

    # Save summary
    summary_path = os.path.join(args.output_dir, "surface_finder_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_scenes_processed": len(scene_ids),
            "scenes_with_surfaces": len(all_stats),
            "total_surfaces": total_surfaces,
            "total_objects_on_surfaces": total_objects,
            "category_distribution": dict(cat_totals),
            "per_scene": all_stats,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
