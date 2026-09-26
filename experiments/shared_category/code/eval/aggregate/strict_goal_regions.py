"""Collision-safe eight-neighbor grid reference for navigation SPL.

This is a grid-center reference, not a continuous-space shortest-path oracle.
The actual start is connected to its cell center without snapping or cutting
corners; success/start-inside still use the continuous XY AABB surface region.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from mtu3d_strict_grid import strict_segment

PROTOCOL = "strict_grid8_with_start_connector_v1"


def region_record(wm, start_xy, instances: list[dict], radius_m: float = 2.0) -> dict:
    if not math.isfinite(radius_m) or radius_m <= 0 or not instances:
        raise ValueError("Require positive radius and nonempty goal geometry")
    start = tuple(float(v) for v in start_xy)
    if len(start) != 2 or not all(math.isfinite(v) for v in start):
        raise ValueError("Invalid actual start")
    if not strict_segment(wm, start, start):
        raise ValueError("Actual start is not strictly walkable; refusing to snap")
    mask = np.zeros(wm.grid.shape, dtype=bool)
    inside = False
    for instance in instances:
        bbox = instance["bbox"]
        lo, hi = np.asarray(bbox["min"][:2], dtype=float), np.asarray(bbox["max"][:2], dtype=float)
        if lo.shape != (2,) or hi.shape != (2,) or not np.isfinite([lo, hi]).all() or np.any(lo > hi):
            raise ValueError("Invalid goal bounding box")
        delta = np.maximum(np.maximum(lo - start, 0.), np.asarray(start) - hi)
        inside |= float(np.linalg.norm(delta)) <= radius_m
        dx = np.maximum(np.maximum(lo[0] - wm.x_coords[None, :], 0.), wm.x_coords[None, :] - hi[0])
        dy = np.maximum(np.maximum(lo[1] - wm.y_coords[:, None], 0.), wm.y_coords[:, None] - hi[1])
        mask |= dx * dx + dy * dy <= radius_m * radius_m
    mask &= wm.grid.astype(bool)
    distance, endpoint = (0., start) if inside else geodesic_to_goal_mask(wm, start, mask)
    return dict(distance_mode="xy_aabb_surface", radius_m=radius_m,
                n_instances=len(instances), n_navigable_goal_cells=int(mask.sum()),
                start_inside=bool(inside), geodesic_to_goal_region=distance,
                nearest_goal_position=list(endpoint) if endpoint is not None else None,
                reference_protocol=PROTOCOL)


def geodesic_to_goal_mask(wm, start, mask):
    """Dijkstra with strict diagonal edges and a checked actual-start connector."""
    if mask.shape != wm.grid.shape or np.any(mask & ~wm.grid.astype(bool)):
        raise ValueError("Invalid goal mask")
    cell = wm.world_to_cell(*start)
    center = wm.cell_to_world(*cell)
    if not strict_segment(wm, start, center):
        raise ValueError("Actual start cannot connect to its cell center")
    dx, dy = abs(float(wm.x_coords[1] - wm.x_coords[0])), abs(float(wm.y_coords[1] - wm.y_coords[0]))
    initial = math.dist(start, center)
    distances, queue = {cell: initial}, [(initial, cell)]
    height, width = wm.grid.shape
    neighbors = [(dr, dc, math.hypot(dr * dy, dc * dx))
                 for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    while queue:
        cost, cell = heapq.heappop(queue)
        if cost != distances[cell]:
            continue
        if mask[cell]:
            return cost, wm.cell_to_world(*cell)
        for dr, dc, step in neighbors:
            row, col = cell[0] + dr, cell[1] + dc
            if not (0 <= row < height and 0 <= col < width and wm.grid[row, col]):
                continue
            if dr and dc and (not wm.grid[cell[0] + dr, cell[1]] or not wm.grid[cell[0], cell[1] + dc]):
                continue
            proposed = cost + step
            if proposed < distances.get((row, col), math.inf):
                distances[row, col] = proposed
                heapq.heappush(queue, (proposed, (row, col)))
    return None, None
