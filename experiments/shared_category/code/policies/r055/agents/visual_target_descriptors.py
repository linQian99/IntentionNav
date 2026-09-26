"""Auditable visual descriptions for opaque benchmark category names.

The simulator asset taxonomy occasionally names a grouped object by its use
(``tea set``) while open-vocabulary detectors see its rendered parts (a cup and
saucer).  These fixed, category-level descriptions are proposal queries only:
they do not bypass the independent crop verifier, spatial fusion, or STOP
gates.
"""
from __future__ import annotations


VISUAL_TARGET_DESCRIPTORS: dict[str, tuple[str, ...]] = {
    "bread machine": ("bread maker", "countertop bread maker"),
    "builtin oven": ("built in oven", "wall oven"),
    "cosmetic": ("makeup container", "cosmetics bottle"),
    "dish washer": ("dishwasher", "built in dishwasher"),
    "electric cooker": ("rice cooker", "electric cooking pot"),
    "menorah": ("seven branch candle holder", "branched candelabrum"),
    "range hood": ("kitchen extractor hood", "stove hood"),
    "reed diffuser": ("fragrance diffuser with reeds", "reed diffuser bottle"),
    "room divider": ("folding room screen", "partition screen"),
    "storage": ("storage unit", "storage organizer"),
    "tea set": ("cup and saucer", "teacup set", "tea service"),
    "water cooler": ("water dispenser", "bottled water dispenser"),
    "wine set": ("wine glass set", "wine glasses and bottle", "wine service"),
}


def visual_detector_queries(
    target: str,
    canonical_queries: list[str] | tuple[str, ...],
    *,
    max_queries: int = 5,
) -> list[str]:
    """Return deterministic detector proposals without broad component nouns.

    Descriptions such as ``cup and saucer`` are intentionally preferred over
    the broad noun ``cup``.  The latter is another benchmark category and
    would turn fixed-instance search into search for any related object.
    """
    normalized = " ".join(
        str(target or "").strip().lower().replace("_", " ").split()
    )
    queries: list[str] = []
    for query in (
        normalized,
        *VISUAL_TARGET_DESCRIPTORS.get(normalized, ()),
        *canonical_queries,
    ):
        value = " ".join(str(query or "").strip().lower().split())
        if value and value not in queries:
            queries.append(value)
        if len(queries) >= max(1, int(max_queries)):
            break
    return queries
