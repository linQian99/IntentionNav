"""Canonical fixed object vocabulary for the curated IntentionNav release.

Keep this list synchronized with ``selected_500_intents.jsonl``.  Perception
models must compete over the benchmark's actual categories rather than legacy
aliases from earlier dataset drafts.
"""
from __future__ import annotations


BENCHMARK_VOCABULARY = (
    "air conditioner", "air purifier", "basin", "bathtub", "bed", "bench",
    "book", "bowl", "bread machine", "builtin oven", "cabinet",
    "ceiling light", "chair", "chopstick", "clock", "coffee maker",
    "computer monitor", "cosmetic", "cup", "curtain", "desk",
    "dining table", "dish washer", "electric cooker", "floor lamp",
    "flower", "fork", "fridge", "fruit", "kettle", "knife", "menorah",
    "microwave", "mirror", "night stand", "painting", "piano",
    "picture frame", "pillow", "plate", "range hood", "reed diffuser",
    "remote control", "room divider", "sculpture", "shelf", "sofa",
    "sound", "spoon", "stool", "storage", "table", "table lamp",
    "tea set", "television", "toilet", "towel rack", "toy", "vase",
    "wall light", "washing machine", "water cooler", "water heater",
    "wine set",
)

