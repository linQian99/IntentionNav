"""Walkable-map utilities for IntentionNav evaluation.

Wraps the VLNTube freemap/room_region data in a clean interface:
- WalkableMap.load(scene_id) — load freemap.npy + room_region.json
- .is_walkable(x, y) — 2D walkability query
- .line_of_sight(x0, y0, x1, y1) — 2D LOS check on freemap
- .sample_walkable_points(N, seed) — random walkable (x, y) points
- .nearby_walkable(x, y, radius_m) — find nearest walkable cell within radius
- .room_at(x, y) — room ID at (x, y) or None
- .geodesic_distance_2d((x0,y0), (x1,y1)) — A* distance on freemap grid

Freemap format (from VLNTube):
  freemap[0, 1:]  = x coordinates (columns)
  freemap[1:, 0]  = y coordinates (rows)
  freemap[1:, 1:] = grid, 1 = walkable, 0 = obstacle
  (x_coords is typically DESCENDING; y_coords is ASCENDING)

Room region JSON:
  {"<room_name>_<instance>": [[row, col], [row, col], ...]}  (polygon vertices in grid coords)
"""

from __future__ import annotations

import heapq
import json
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import os

# External VLNTube dataset (freemap.npy + room_region.json per scene).
# Override via INTENTIONNAV_METAROOT.
REPO = Path(__file__).resolve().parents[2]
METAROOT = Path(os.environ.get(
    "INTENTIONNAV_METAROOT",
    str(REPO / "data/SceneMeta/metadata_train"),
))


@dataclass
class WalkableMap:
    scene_id: str
    x_coords: np.ndarray  # 1D, descending in this dataset
    y_coords: np.ndarray  # 1D, ascending
    grid: np.ndarray      # 2D (len(y), len(x)), values in {0, 1}
    rooms: dict[str, list[list[int]]] = field(default_factory=dict)

    # ---- Load ----
    @classmethod
    def load(cls, scene_id: str, metaroot: Path = METAROOT) -> "WalkableMap | None":
        fm_path = metaroot / scene_id / "freemap.npy"
        rr_path = metaroot / scene_id / "room_region.json"
        if not fm_path.exists():
            return None
        m = np.load(fm_path)
        obj = cls(
            scene_id=scene_id,
            x_coords=m[0, 1:].astype(np.float64),
            y_coords=m[1:, 0].astype(np.float64),
            grid=(m[1:, 1:] == 1).astype(np.uint8),
            rooms={},
        )
        if rr_path.exists():
            obj.rooms = json.loads(rr_path.read_text(encoding="utf-8"))
        # explored_count: agent-exploration accumulator. Same shape as grid.
        # Reset per-episode by run_episode (this WalkableMap may be cached
        # across episodes in the same worker for the same scene_id).
        obj.explored_count = np.zeros_like(obj.grid, dtype=np.uint16)
        return obj

    # ---- Exploration tracking (NOT freemap-derived; updated by agent) ----
    def reset_explored(self) -> None:
        """Wipe the explored-count grid. Call at episode start."""
        if hasattr(self, "explored_count"):
            self.explored_count.fill(0)

    def mark_explored(self, x: float, y: float, radius_m: float = 0.5) -> None:
        """Increment explored_count for every walkable cell within radius_m
        of (x, y). Call once per step from the agent loop with the agent's
        current pose. uint16 so each cell can saturate at 65535 — way more
        than any 30-step episode."""
        if not hasattr(self, "explored_count"):
            return
        if len(self.x_coords) < 2 or len(self.y_coords) < 2:
            return
        yi0, xi0 = self.world_to_cell(x, y)
        h, w = self.grid.shape
        dx = float(abs(self.x_coords[1] - self.x_coords[0]))
        dy = float(abs(self.y_coords[1] - self.y_coords[0]))
        cells_r = max(1, int(math.ceil(radius_m / min(dx, dy))))
        y0, y1 = max(0, yi0 - cells_r), min(h, yi0 + cells_r + 1)
        x0, x1 = max(0, xi0 - cells_r), min(w, xi0 + cells_r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist2 = (dy * (yy - yi0)) ** 2 + (dx * (xx - xi0)) ** 2
        mask = (dist2 <= radius_m ** 2) & (self.grid[y0:y1, x0:x1] > 0)
        # uint16 saturating add
        self.explored_count[y0:y1, x0:x1] = np.minimum(
            np.iinfo(np.uint16).max,
            self.explored_count[y0:y1, x0:x1].astype(np.uint32) + mask.astype(np.uint32),
        ).astype(np.uint16)

    def explored_density_at(self, x: float, y: float,
                            radius_m: float = 1.0) -> float:
        """Fraction of walkable cells within radius_m of (x,y) that have
        been visited (explored_count > 0). Returns in [0, 1]; 0 = pure
        unexplored frontier, 1 = fully revisited area. Fast vectorized."""
        if not hasattr(self, "explored_count"):
            return 0.0
        if len(self.x_coords) < 2 or len(self.y_coords) < 2:
            return 0.0
        yi0, xi0 = self.world_to_cell(x, y)
        h, w = self.grid.shape
        dx = float(abs(self.x_coords[1] - self.x_coords[0]))
        dy = float(abs(self.y_coords[1] - self.y_coords[0]))
        cells_r = max(1, int(math.ceil(radius_m / min(dx, dy))))
        y0, y1 = max(0, yi0 - cells_r), min(h, yi0 + cells_r + 1)
        x0, x1 = max(0, xi0 - cells_r), min(w, xi0 + cells_r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist2 = (dy * (yy - yi0)) ** 2 + (dx * (xx - xi0)) ** 2
        in_radius = dist2 <= radius_m ** 2
        walkable = self.grid[y0:y1, x0:x1] > 0
        explored = self.explored_count[y0:y1, x0:x1] > 0
        n_walkable = int((in_radius & walkable).sum())
        if n_walkable == 0:
            return 0.0
        n_explored = int((in_radius & walkable & explored).sum())
        return n_explored / n_walkable

    # ---- Coordinate conversion ----
    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """Return (row, col) = (yi, xi). Row-major grid indexing."""
        xi = int(np.argmin(np.abs(self.x_coords - x)))
        yi = int(np.argmin(np.abs(self.y_coords - y)))
        return yi, xi

    def cell_to_world(self, yi: int, xi: int) -> tuple[float, float]:
        return float(self.x_coords[xi]), float(self.y_coords[yi])

    # ---- Basic queries ----
    def is_walkable(self, x: float, y: float) -> bool:
        # CRITICAL: bounds check before world_to_cell. The latter uses
        # np.argmin on |coord - target| which always returns a valid index
        # (clamps to the nearest edge cell) — without this guard, a waypoint
        # at x=+43m on a [-4, +4]m freemap maps to the rightmost edge cell
        # and may pass walkability if that cell happens to be walkable.
        # Bug observed in SEL_195: agent wandered 43m outside apartment.
        if len(self.x_coords) >= 2 and len(self.y_coords) >= 2:
            dx_half = abs(self.x_coords[1] - self.x_coords[0]) / 2
            dy_half = abs(self.y_coords[1] - self.y_coords[0]) / 2
            if (x < self.x_coords.min() - dx_half
                    or x > self.x_coords.max() + dx_half
                    or y < self.y_coords.min() - dy_half
                    or y > self.y_coords.max() + dy_half):
                return False
        yi, xi = self.world_to_cell(x, y)
        if not (0 <= yi < self.grid.shape[0] and 0 <= xi < self.grid.shape[1]):
            return False
        return bool(self.grid[yi, xi])

    def nearby_walkable(self, x: float, y: float, radius_m: float = 1.0) -> tuple[float, float] | None:
        """Nearest walkable cell within radius_m meters of (x, y). None if none."""
        # Grid cell size approx:
        if len(self.x_coords) < 2 or len(self.y_coords) < 2:
            return None
        dx = float(abs(self.x_coords[1] - self.x_coords[0]))
        dy = float(abs(self.y_coords[1] - self.y_coords[0]))
        cells_r = int(math.ceil(radius_m / min(dx, dy)))
        yi0, xi0 = self.world_to_cell(x, y)
        h, w = self.grid.shape
        best = None
        best_d = float("inf")
        for dyi in range(-cells_r, cells_r + 1):
            for dxi in range(-cells_r, cells_r + 1):
                yi, xi = yi0 + dyi, xi0 + dxi
                if not (0 <= yi < h and 0 <= xi < w):
                    continue
                if not self.grid[yi, xi]:
                    continue
                wx, wy = self.cell_to_world(yi, xi)
                d = math.hypot(wx - x, wy - y)
                if d <= radius_m and d < best_d:
                    best = (wx, wy)
                    best_d = d
        return best

    def line_of_sight(self, x0: float, y0: float, x1: float, y1: float,
                      n_samples: int = 30, tolerance: float = 0.15) -> bool:
        """2D LOS: sample points along the line on freemap. Lifted from
        capture_surfaces.has_line_of_sight.

        Bound-aware: a sample point outside the freemap counts as BLOCKED
        (otherwise world_to_cell clamps to the edge and a path leaving the
        apartment looks "in line of sight" — see SEL_195 wandering bug).
        """
        ts = np.linspace(0.15, 0.85, n_samples)
        blocked = checked = 0
        h, w = self.grid.shape
        if len(self.x_coords) >= 2 and len(self.y_coords) >= 2:
            dx_half = abs(self.x_coords[1] - self.x_coords[0]) / 2
            dy_half = abs(self.y_coords[1] - self.y_coords[0]) / 2
            x_min, x_max = self.x_coords.min() - dx_half, self.x_coords.max() + dx_half
            y_min, y_max = self.y_coords.min() - dy_half, self.y_coords.max() + dy_half
        else:
            x_min = y_min = -float("inf"); x_max = y_max = float("inf")
        for t in ts:
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            checked += 1
            if x < x_min or x > x_max or y < y_min or y > y_max:
                blocked += 1
                continue
            yi, xi = self.world_to_cell(x, y)
            if 0 <= yi < h and 0 <= xi < w:
                if not self.grid[yi, xi]:
                    blocked += 1
            else:
                blocked += 1
        if checked == 0:
            return True
        return (blocked / checked) <= tolerance

    def max_clear_distance(self, x0: float, y0: float, yaw: float,
                           probe_distances: list[float] | None = None
                           ) -> float:
        """Walk a ray from (x0, y0) along `yaw` (radians) on the freemap.
        Return the largest probe distance whose endpoint is walkable AND
        has line-of-sight back to the origin. 0 if even the first probe
        fails (blocked nearby)."""
        if probe_distances is None:
            probe_distances = [0.3, 0.6, 1.0, 1.5, 2.0, 3.0]
        d_clear = 0.0
        for d in probe_distances:
            cx = x0 + d * math.cos(yaw)
            cy = y0 + d * math.sin(yaw)
            if not self.is_walkable(cx, cy):
                break
            if not self.line_of_sight(x0, y0, cx, cy):
                break
            d_clear = d
        return d_clear

    def pick_open_yaw(self, x0: float, y0: float,
                      current_yaw: float,
                      n_directions: int = 16,
                      probe_distances: list[float] | None = None
                      ) -> tuple[float, float, list[tuple[float, float]]]:
        """Sample n_directions evenly around (x0, y0), pick the yaw with
        the longest unobstructed ray on the freemap (ties go to the
        direction closest to current_yaw to avoid needless rotation).
        Returns (best_yaw_rad, best_clear_dist_m, all_samples).
        all_samples = [(yaw_rad, clear_dist_m), ...] — useful for logging."""
        samples = []
        for k in range(n_directions):
            theta = current_yaw + k * (2 * math.pi / n_directions)
            d = self.max_clear_distance(x0, y0, theta, probe_distances)
            samples.append((theta, d))
        # Pick max d; tie-break by minimum |delta| to current_yaw.
        def _dist_to_cur(theta):
            d = (theta - current_yaw) % (2 * math.pi)
            return min(d, 2 * math.pi - d)
        best = max(samples, key=lambda s: (s[1], -_dist_to_cur(s[0])))
        return best[0], best[1], samples

    # ---- Walkable-point sampling ----
    def sample_walkable_points(self, n: int, seed: int = 42) -> list[tuple[float, float]]:
        ys, xs = np.nonzero(self.grid)
        if len(ys) == 0:
            return []
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
        return [self.cell_to_world(int(ys[i]), int(xs[i])) for i in idx]

    # ---- Room queries ----
    def room_at(self, x: float, y: float) -> str | None:
        """Return the room key whose polygon contains (x, y), else None."""
        if not self.rooms:
            return None
        yi, xi = self.world_to_cell(x, y)
        # Room polygons are in grid coords: [[row, col], ...]. Ray-cast.
        for room_key, poly in self.rooms.items():
            if _point_in_polygon((yi, xi), poly):
                return room_key
        return None

    @staticmethod
    def normalize_room_type(room_key: str) -> str:
        """Strip trailing _<idx> from a room key, lowercase."""
        return re.sub(r"_\d+$", "", room_key).lower()

    # ---- Geodesic distance (A* on 8-connected grid) ----
    def geodesic_distance_2d(self, p0: tuple[float, float], p1: tuple[float, float]) -> float | None:
        """A* shortest path on the walkable grid. Returns meters, or None if unreachable."""
        start = self.world_to_cell(*p0)
        goal = self.world_to_cell(*p1)
        h, w = self.grid.shape
        if not (0 <= start[0] < h and 0 <= start[1] < w):
            return None
        if not self.grid[start]:
            # snap to nearest walkable
            near = self.nearby_walkable(*p0, radius_m=1.5)
            if not near:
                return None
            start = self.world_to_cell(*near)
        if not self.grid[goal]:
            near = self.nearby_walkable(*p1, radius_m=1.5)
            if not near:
                return None
            goal = self.world_to_cell(*near)

        dx_m = float(abs(self.x_coords[1] - self.x_coords[0]))
        dy_m = float(abs(self.y_coords[1] - self.y_coords[0]))
        step_m = (dx_m + dy_m) / 2.0

        def heuristic(a, b):
            return math.hypot((a[0] - b[0]) * dy_m, (a[1] - b[1]) * dx_m)

        open_heap = [(heuristic(start, goal), 0.0, start)]
        g_score = {start: 0.0}
        closed = set()
        neighbors = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                     (-1, -1, 1.41421356), (-1, 1, 1.41421356),
                     (1, -1, 1.41421356), (1, 1, 1.41421356)]
        while open_heap:
            _, g, cur = heapq.heappop(open_heap)
            if cur == goal:
                return g * step_m
            if cur in closed:
                continue
            closed.add(cur)
            for dy, dx, cost in neighbors:
                nb = (cur[0] + dy, cur[1] + dx)
                if not (0 <= nb[0] < h and 0 <= nb[1] < w):
                    continue
                if not self.grid[nb]:
                    continue
                new_g = g + cost
                if new_g < g_score.get(nb, float("inf")):
                    g_score[nb] = new_g
                    heapq.heappush(open_heap, (new_g + heuristic(nb, goal), new_g, nb))
        return None


def _point_in_polygon(pt, poly) -> bool:
    """Ray-casting point-in-polygon. Works on 2D integer or float coords.

    poly: list of [row, col] vertices.
    pt: (row, col)
    """
    x, y = pt[1], pt[0]
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][1], poly[i][0]
        xj, yj = poly[j][1], poly[j][0]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


# ---- Simple CLI smoke test ----
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_id")
    args = ap.parse_args()
    wm = WalkableMap.load(args.scene_id)
    if wm is None:
        print("freemap missing")
        raise SystemExit(1)
    print(f"scene={args.scene_id} grid={wm.grid.shape} "
          f"walkable_pct={wm.grid.mean():.3f} rooms={len(wm.rooms)}")
    samples = wm.sample_walkable_points(5)
    for x, y in samples:
        room = wm.room_at(x, y)
        near = wm.nearby_walkable(x, y, 0.5)
        print(f"  ({x:+.2f},{y:+.2f}) walkable={wm.is_walkable(x,y)} "
              f"room={room} near={near}")
