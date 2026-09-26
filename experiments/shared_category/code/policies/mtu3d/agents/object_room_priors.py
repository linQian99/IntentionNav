"""Small, episode-independent room priors for explicit ObjectNav.

These are common-sense search hints, not labels learned from or looked up in
the benchmark episode.  Keeping them in one auditable table avoids the prior
reference-agent bug where ``episode_meta.target_room`` leaked the answer.
"""
from __future__ import annotations


ROOM_GROUPS = {
    "bathroom": {
        "basin", "bathtub", "closestool", "cosmetic", "mirror", "toilet",
        "reed_diffuser", "towel_rack", "water_heater",
    },
    "kitchen": {
        "bowl", "bread_machine", "builtin_oven", "cabinet", "chopstick",
        "coffee_maker", "cup", "dish_washer", "electric_cooker", "fork",
        "fridge", "fruit", "kettle", "knife", "microwave", "plate",
        "range_hood", "spoon", "tea_set", "water_cooler", "wine_set",
    },
    "bedroom": {
        "bed", "night_stand", "pillow", "remote_control", "storage", "table_lamp",
        "throw_pillow",
    },
    "living room": {
        "air_conditioner", "air_purifier", "bench", "ceiling_light",
        "chandelier", "clock", "couch", "curtain", "floor_lamp", "flower",
        "menorah", "painting", "piano", "picture_frame", "sculpture",
        "sofa", "sound", "television", "toy", "vase", "wall_light",
    },
    "study room": {"book", "computer_monitor", "desk", "shelf"},
    "dining room": {"dining_table"},
}


MULTI_ROOM = {
    "basin": ["bathroom", "kitchen"],
    "bowl": ["kitchen", "dining room", "living room"],
    "cabinet": ["kitchen", "living room", "bedroom", "bathroom"],
    "chair": ["living room", "dining room", "study room", "bedroom"],
    "cup": ["kitchen", "dining room", "living room", "study room"],
    "mirror": ["bathroom", "bedroom", "living room"],
    "plate": ["kitchen", "dining room", "living room"],
    "reed_diffuser": ["bathroom", "bedroom", "living room"],
    "remote_control": ["bedroom", "living room"],
    "room_divider": ["living room", "dining room", "bathroom"],
    "screen": ["living room", "study room", "bedroom"],
    "computer_monitor": ["study room", "bedroom", "living room"],
    "stool": ["living room", "kitchen", "dining room"],
    "table": ["living room", "dining room", "study room", "bedroom"],
    "towel_rack": ["bathroom", "bedroom"],
    "tea_set": ["living room", "dining room", "kitchen"],
    "washing_machine": ["bathroom", "balcony", "kitchen"],
    "water_heater": ["bathroom", "kitchen"],
    "wine_set": ["living room", "dining room", "kitchen"],
}


def likely_rooms_for_object(category: str) -> list[str]:
    """Return deterministic common-sense rooms for a category."""
    normalized = str(category or "").strip().lower().replace(" ", "_")
    if normalized in MULTI_ROOM:
        return list(MULTI_ROOM[normalized])
    matches = [room for room, categories in ROOM_GROUPS.items()
               if normalized in categories]
    # Unknown/general decor should not be forced into a hallucinated room.
    return matches
