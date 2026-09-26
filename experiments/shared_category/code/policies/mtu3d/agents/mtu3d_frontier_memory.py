"""Adapt the official geometric frontier interface to the Isaac map axes."""
from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

from mtu3d_frontier_grid import detect_frontier_waypoints, reveal_fog_of_war


class FrontierMemory:
    """Track acquired view cones separately from visited navigation positions.

    Uses the same permitted, nonsemantic walkability map as path execution.
    A caller must invoke observe only after acquiring the actual forward view.
    Visibility is the upstream top-down occlusion proxy, not object visibility.
    """

    def __init__(self, walkable_map):
        self.map = walkable_map
        raw_dx = float(walkable_map.x_coords[1] - walkable_map.x_coords[0])
        raw_dy = float(walkable_map.y_coords[1] - walkable_map.y_coords[0])
        self.resolution = min(abs(raw_dx), abs(raw_dy))
        if not math.isfinite(self.resolution) or self.resolution <= 0:
            raise ValueError('Map cells must have positive, finite metric size')
        if not np.allclose(np.diff(walkable_map.x_coords), raw_dx, rtol=1e-3, atol=1e-6) or not np.allclose(np.diff(walkable_map.y_coords), raw_dy, rtol=1e-3, atol=1e-6):
            raise ValueError('Map coordinates must have uniform spacing')
        # Scene freemaps have slightly different x/y pitches. The official
        # circular visibility operation needs metric square pixels; resample
        # only its geometry raster, using nearest source cells. Path execution
        # and final proposal walkability stay on the original map.
        self.dx = math.copysign(self.resolution, raw_dx)
        self.dy = math.copysign(self.resolution, raw_dy)
        nx = int(math.ceil(np.ptp(walkable_map.x_coords) / self.resolution)) + 1
        ny = int(math.ceil(np.ptp(walkable_map.y_coords) / self.resolution)) + 1
        self.x_coords = walkable_map.x_coords[0] + self.dx * np.arange(nx)
        self.y_coords = walkable_map.y_coords[0] + self.dy * np.arange(ny)
        cols = np.floor((self.x_coords - walkable_map.x_coords[0]) / raw_dx + .5).astype(int)
        rows = np.floor((self.y_coords - walkable_map.y_coords[0]) / raw_dy + .5).astype(int)
        valid_cols = (cols >= 0) & (cols < len(walkable_map.x_coords))
        valid_rows = (rows >= 0) & (rows < len(walkable_map.y_coords))
        self.free = np.asarray(walkable_map.grid, dtype=np.uint8)[
            np.ix_(np.clip(rows, 0, len(walkable_map.y_coords)-1),
                   np.clip(cols, 0, len(walkable_map.x_coords)-1))].copy()
        self.free[~valid_rows, :] = 0
        self.free[:, ~valid_cols] = 0
        self.seen = np.zeros_like(self.free)
        _, self.components = cv2.connectedComponents(self.free, connectivity=8)
        self.visited: set[tuple[float, float]] = set()
        self.observations = 0

    def observe(self, position: Sequence[float], look_dir: Sequence[float]) -> dict:
        if not self.map.is_walkable(*position[:2]):
            raise ValueError('Cannot reveal a view from an unwalkable pose')
        if not np.isfinite(position).all() or not np.isfinite(look_dir).all() or math.hypot(*look_dir[:2]) < 1e-8:
            raise ValueError('Expected a finite horizontal viewing direction')
        cell = np.asarray(self.world_to_cell(position))
        # OpenCV columns/rows follow array order. In these scenes columns run
        # toward negative world x; using world yaw directly mirrors visibility.
        pixel_angle = math.atan2(look_dir[1] / self.dy, look_dir[0] / self.dx)
        upstream_angle = math.pi / 2 - pixel_angle
        before = int(np.count_nonzero(self.seen))
        self.seen = reveal_fog_of_war(self.free, self.seen, cell, upstream_angle,
                                     fov=90, max_line_len=max(1, int(3 / self.resolution)))
        self.seen[self.free == 0] = 0
        self.observations += 1
        return {'acquired_views': self.observations, 'seen_cells': int(np.count_nonzero(self.seen)),
                'new_seen_cells': int(np.count_nonzero(self.seen)) - before}

    def proposals(self, position: Sequence[float]) -> list[list[float]]:
        cell = self.world_to_cell(position)
        component = self.components[cell]
        if component == 0:
            return []
        # Preserve the released code's area-threshold formula (9/meters-per-
        # pixel), including its units convention. Do not describe it as 9 m².
        points = detect_frontier_waypoints(self.free, self.seen.copy(),
                                          area_thresh=int(9 / self.resolution))
        output = []
        keys = set()
        for col, row in points:
            yi, xi = int(round(row)), int(round(col))
            if not (0 <= yi < self.free.shape[0] and 0 <= xi < self.free.shape[1]):
                continue
            if not self.free[yi, xi] or self.components[yi, xi] != component:
                continue
            point = (float(self.x_coords[xi]), float(self.y_coords[yi]))
            if not self.map.is_walkable(*point):
                continue
            key = self._key(point)
            if key in self.visited or key in keys:
                continue
            # A midpoint on the current cell is not an executable exploration
            # target. This geometric case is explicit, never an implicit STOP.
            if math.dist(position[:2], point) <= self.arrival_radius:
                continue
            if self.map.shortest_path_2d(tuple(position[:2]), point) is None:
                continue
            output.append([float(point[0]), float(point[1]), 0.])
            keys.add(key)
        return output

    @property
    def arrival_radius(self) -> float:
        return max(.05, self.resolution * math.sqrt(2))

    def world_to_cell(self, position: Sequence[float]) -> tuple[int, int]:
        row = int(np.argmin(np.abs(self.y_coords - position[1])))
        col = int(np.argmin(np.abs(self.x_coords - position[0])))
        return row, col

    @staticmethod
    def _key(position: Sequence[float]) -> tuple[float, float]:
        return tuple(float(x) for x in np.round(position[:2], 1))

    def visit(self, position: Sequence[float]) -> None:
        # The original runner excludes a frontier immediately after selection.
        self.visited.add(self._key(position))
