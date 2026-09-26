"""Full-segment grid collision checks and A* without diagonal corner cutting."""
from __future__ import annotations
import heapq
import math

import numpy as np


def segment_cells(wm, start, goal) -> list[tuple[int, int]]:
    """Return every closed grid cell touched by a segment, including endpoints."""
    xs, ys = np.asarray(wm.x_coords), np.asarray(wm.y_coords)
    if len(xs) < 2 or len(ys) < 2:
        raise ValueError('Require a regular navigation grid')
    dx, dy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
    if (not dx or not dy or not np.allclose(np.diff(xs), dx, atol=1e-8, rtol=1e-6)
            or not np.allclose(np.diff(ys), dy, atol=1e-8, rtol=1e-6)):
        raise ValueError('Unsupported nonuniform navigation grid')
    a = np.array([(start[0] - xs[0]) / dx, (start[1] - ys[0]) / dy], dtype=float)
    b = np.array([(goal[0] - xs[0]) / dx, (goal[1] - ys[0]) / dy], dtype=float)
    if not np.isfinite([a, b]).all():
        raise ValueError('Nonfinite navigation segment')
    if not wm.is_walkable(*start) or not wm.is_walkable(*goal):
        return []
    times = {0., 1.}
    for first, last in zip(a, b):
        if abs(last - first) <= 1e-12:
            continue
        lo, hi = sorted((first, last))
        for index in range(math.ceil(lo - .5), math.floor(hi - .5) + 1):
            value = (index + .5 - first) / (last - first)
            if 0 <= value <= 1:
                times.add(float(value))
    ordered = sorted(times)
    sample_times = ordered + [(x + y) / 2 for x, y in zip(ordered, ordered[1:])]
    cells = set()
    for t in sample_times:
        x, y = a + t * (b - a)
        def touching(value):
            below = math.floor(value)
            return [below, below + 1] if abs(value - below - .5) <= 1e-9 else [math.floor(value + .5)]
        for col in touching(x):
            for row in touching(y):
                cells.add((row, col))
    return sorted(cells)


def strict_segment(wm, start, goal) -> bool:
    cells = segment_cells(wm, start, goal)
    return bool(cells and all(0 <= row < wm.grid.shape[0] and 0 <= col < wm.grid.shape[1]
                            and wm.grid[row, col] for row, col in cells))


def strict_path(wm, start, goal, snap_radius_m=2.0):
    """A* on the same permitted map, checking both sides of every diagonal."""
    if not wm.is_walkable(*start):
        return None
    target = tuple(goal)
    if not wm.is_walkable(*target):
        target = wm.nearby_walkable(*target, radius_m=snap_radius_m)
        if target is None:
            return None
    first, last = wm.world_to_cell(*start), wm.world_to_cell(*target)
    if not strict_segment(wm, start, wm.cell_to_world(*first)):
        return None
    dx = abs(float(wm.x_coords[1] - wm.x_coords[0]))
    dy = abs(float(wm.y_coords[1] - wm.y_coords[0]))
    def walkable(cell):
        row, col = cell
        return 0 <= row < wm.grid.shape[0] and 0 <= col < wm.grid.shape[1] and bool(wm.grid[row, col])
    def heuristic(cell):
        return math.hypot((cell[0] - last[0]) * dy, (cell[1] - last[1]) * dx)
    heap, costs, parents = [(heuristic(first), 0., first)], {first: 0.}, {}
    while heap:
        _, cost, cell = heapq.heappop(heap)
        if cost > costs[cell]:
            continue
        if cell == last:
            cells = [cell]
            while cell in parents:
                cell = parents[cell]
                cells.append(cell)
            return [wm.cell_to_world(*c) for c in reversed(cells)]
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == dc == 0:
                    continue
                nxt = (cell[0] + dr, cell[1] + dc)
                if not walkable(nxt):
                    continue
                if dr and dc and (not walkable((cell[0] + dr, cell[1])) or not walkable((cell[0], cell[1] + dc))):
                    continue
                candidate = cost + math.hypot(dr * dy, dc * dx)
                if candidate < costs.get(nxt, float('inf')):
                    costs[nxt], parents[nxt] = candidate, cell
                    heapq.heappush(heap, (candidate + heuristic(nxt), candidate, nxt))
    return None


def strict_path_action(wm, current, target, maximum=1.7):
    if (1e-5 < math.dist(current, target) <= maximum
            and strict_segment(wm, current, target)):
        return tuple(target)
    path = strict_path(wm, current, target)
    if not path:
        return None
    selected = None
    for point in path:
        distance = math.dist(current, point)
        if distance > maximum + 1e-8 or not strict_segment(wm, current, point):
            break
        if distance > 1e-5:
            selected = tuple(point)
    return selected
