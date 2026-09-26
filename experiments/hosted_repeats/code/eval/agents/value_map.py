"""Per-step value map + A* picker for the engine-driven nav agent.

Replaces the broken VLM-as-letter-picker with a geometry-aware planner:
each step we score every walkable cell from four sources, take the
argmax-cell, and run A* on the walkable grid to produce the next waypoint.

Sources (per `_sources` table below):
  target_memory  : Gaussian blob at each detected target cluster
                   (value = cluster.score, sigma = TARGET_SIGMA_M)
  room_prior     : constant boost in cells whose room matches plan.likely_rooms
  frontier       : boost in cells with low explored density
  visited_penalty: Gaussian negative blob at each cell in agent_path

Tuning lives at module-level constants, all overridable via env vars so
ablations can be done without touching code.

Used by `agent_vlm_engine.py`; does not depend on Isaac Sim.
"""
from __future__ import annotations

import heapq
import math
import os
from collections import deque

import numpy as np


# ---- Tuning (env-var-overridable) ----
TARGET_SIGMA_M = float(os.environ.get("VM_TARGET_SIGMA_M", "1.5"))
TARGET_WEIGHT = float(os.environ.get("VM_TARGET_WEIGHT", "1.0"))
ROOM_WEIGHT = float(os.environ.get("VM_ROOM_WEIGHT", "0.3"))
FRONTIER_WEIGHT = float(os.environ.get("VM_FRONTIER_WEIGHT", "0.4"))
VISITED_SIGMA_M = float(os.environ.get("VM_VISITED_SIGMA_M", "0.6"))
VISITED_WEIGHT = float(os.environ.get("VM_VISITED_WEIGHT", "0.5"))
# Step distance the agent attempts per A* call (meters). Picks the first
# cell on the planned path at >= this distance from current pose.
STEP_M = float(os.environ.get("VM_STEP_M", "1.2"))
# Min agent-to-target-memory distance to consider STOP (engine-side).
# When any cluster.score >= STOP_SCORE_MIN is within STOP_RADIUS_M of
# current pose, the agent STOPs (paired with VLM "see target" check).
STOP_RADIUS_M = float(os.environ.get("VM_STOP_RADIUS_M", "1.5"))
STOP_SCORE_MIN = float(os.environ.get("VM_STOP_SCORE_MIN", "0.5"))
# Frontier definition: a cell is "frontier" if its 3x3 neighborhood has
# both walkable AND unexplored cells (boundary of explored region).
# explored_density_at fraction below which a cell is considered unvisited.
FRONTIER_DENSITY_THRESHOLD = float(
    os.environ.get("VM_FRONTIER_DENSITY_THRESHOLD", "0.4")
)
# Source E: VLFM-style relevance cone painted along agent yaw. Triggered
# by the see_target VLM call's `relevance_score` (0-10) — when the VLM
# thinks current view is relevant to finding the target, paint a forward
# Gaussian cone weighted by relevance/10. Cheap cognitive directional
# signal that doesn't require a separate VLM call.
CONE_WEIGHT = float(os.environ.get("VM_CONE_WEIGHT", "0.3"))
# Fix C2: dynamic room_prior boost. A room is "unvisited" when fewer
# than ROOM_VISITED_FRACTION of its cells have explored_count > 0.
# Cells in an unvisited likely room get room_prior multiplied by
# ROOM_PRIOR_UNVISITED_BOOST (default 4.0 — produces effective value
# 0.3 × 1.0 × 4.0 = 1.2, exceeding frontier max ~0.4 even with peak
# density). This pulls the agent OUT of starting room and TOWARD
# unvisited likely rooms — independent of whether they're adjacent
# to explored area (frontier-based fix doesn't trigger until agent
# is at the doorway, which never happens if frontier in starting
# room dominates).
ROOM_PRIOR_UNVISITED_BOOST = float(
    os.environ.get("VM_ROOM_PRIOR_UNVISITED_BOOST", "4.0")
)
ROOM_VISITED_FRACTION = float(
    os.environ.get("VM_ROOM_VISITED_FRACTION", "0.05")
)
CONE_SIGMA_RAD = float(os.environ.get("VM_CONE_SIGMA_RAD", "0.6"))   # ~35°
CONE_RANGE_M = float(os.environ.get("VM_CONE_RANGE_M", "4.0"))
# X3.1 + X3.2 (VLFM-style cone improvements). When enabled the cone is
# (a) depth-masked: cells along a ray whose distance > depth at the
# corresponding pixel are excluded (target can't be behind a wall the
# camera sees), and (b) confidence-weighted-fused with persistent state
# across steps using cos²(θ/(fov/2)·π/2) as the per-cell confidence.
# Pixels closer to the optical axis weight more, and old confident
# observations are not overwritten by new low-confidence ones at FOV
# edges. Matches VLFM paper Eqs. for value/confidence updates.
CONE_VLFM_FUSION = os.environ.get("VM_CONE_VLFM_FUSION", "1") == "1"
CONE_HFOV_RAD = float(os.environ.get("VM_CONE_HFOV_RAD",
                                       str(math.pi / 2)))   # 90° default
# Tolerance band for `is_in_likely_room` admit gate (NOT the value-map
# layer). DINO bbox back-projection has depth + bearing noise of order
# ~0.3-0.6m; doorway / window / wall cells in walkable_map often fall
# OUTSIDE the geometric room polygon even when the object is physically
# in the room. Diagnosis on Easy-24 Fix B run: rejected=14-15 admits/ep
# for bed cases when likely_rooms IS correct → gate too strict killing
# good admits. Dilate the admit mask by this radius (default 1.0m).
ROOM_ADMIT_DILATE_M = float(os.environ.get("VM_ROOM_ADMIT_DILATE_M", "0.0"))


def _gaussian_disk(grid_shape, center_yx, sigma_cells, peak):
    """Return a 2D Gaussian disk of `peak` magnitude at `center_yx` on a
    grid of `grid_shape`, with `sigma_cells` standard deviation (in cell
    units). Truncated at 3σ. Returns dense array same shape as grid."""
    h, w = grid_shape
    cy, cx = center_yx
    rad = int(math.ceil(3 * sigma_cells))
    y0 = max(0, cy - rad)
    y1 = min(h, cy + rad + 1)
    x0 = max(0, cx - rad)
    x1 = min(w, cx + rad + 1)
    if y1 <= y0 or x1 <= x0:
        return None
    yy, xx = np.ogrid[y0:y1, x0:x1]
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    blob = peak * np.exp(-d2 / (2.0 * sigma_cells * sigma_cells))
    return (slice(y0, y1), slice(x0, x1)), blob


class ValueMap:
    """Per-step value map driven by 4 sources. A* picks next waypoint.

    Constructor takes the walkable_map (with frontier + room metadata) and
    the episode-start plan (for likely_rooms prior). One-time prep computes
    the room-prior layer; per-step `update()` fuses target_memory + frontier
    + visited.
    """

    def __init__(self, walkable_map, plan: dict):
        self.wm = walkable_map
        # Cell size in meters (assume isotropic).
        self.cell_m = float(abs(walkable_map.x_coords[1]
                                 - walkable_map.x_coords[0]))
        self.h, self.w = walkable_map.grid.shape
        self.walkable = walkable_map.grid.astype(np.bool_)
        # X3.2 VLFM cone fusion state (persistent across steps within
        # this episode). Each step's new cone observation is merged
        # confidence-weighted with these.
        self._cone_value = np.zeros((self.h, self.w), dtype=np.float32)
        self._cone_conf = np.zeros((self.h, self.w), dtype=np.float32)

        # ---- Pre-compute room labels per walkable cell (one pass) ----
        # Used for room_prior + the "no plan room matches scene" fallback.
        cell_room: dict[tuple[int, int], str] = {}
        scene_rooms_norm: set[str] = set()
        # Per-instance room cell list (e.g. "bedroom_1": [(yi,xi), ...]).
        # Used by room-aware frontier (Fix C2) to compute per-room visited
        # fraction at runtime.
        self._room_instance_cells: dict[str, list[tuple[int, int]]] = {}
        if hasattr(walkable_map, "room_at"):
            for yi in range(self.h):
                for xi in range(self.w):
                    if not self.walkable[yi, xi]:
                        continue
                    wx, wy = walkable_map.cell_to_world(yi, xi)
                    room = walkable_map.room_at(wx, wy)
                    if room:
                        rt = walkable_map.normalize_room_type(room).lower()
                        cell_room[(yi, xi)] = rt
                        scene_rooms_norm.add(rt)
                        self._room_instance_cells.setdefault(
                            str(room), []).append((yi, xi))

        # ---- Pre-compute room_prior (constant across steps) ----
        # FIX D: LLM may hallucinate room names not present in the scene.
        # Soft-match each likely_room to the scene's actual rooms; if no
        # plan room matches any scene room (rare but happens), give EVERY
        # non-trivial room a small uniform boost (encourages leaving the
        # current room rather than locking onto a wrong likely_room).
        likely_rooms = [str(r).strip().lower()
                         for r in (plan.get("likely_rooms") or [])]
        self.room_prior = np.zeros(walkable_map.grid.shape, dtype=np.float32)
        matched_likely_rooms = {sr for sr in scene_rooms_norm
                                 for lr in likely_rooms
                                 if lr in sr or sr in lr}
        if not matched_likely_rooms and scene_rooms_norm:
            # Plan-vs-scene mismatch fallback: prior on ALL scene rooms
            # uniformly — engine picks the closest unexplored room.
            matched_likely_rooms = scene_rooms_norm
        # Cell-level "is in any matched likely room" mask, used by Fix C2
        # frontier weighting (per-cell lookup is faster than per-room iter).
        # Also keep a list of likely room *instances* (e.g. ['bedroom_1',
        # 'living room_0']) for the per-room visited-fraction computation.
        self.likely_room_cell_mask = np.zeros(walkable_map.grid.shape, dtype=np.bool_)
        self._likely_room_instances: list[str] = []
        for (yi, xi), rt in cell_room.items():
            if rt in matched_likely_rooms:
                self.room_prior[yi, xi] = 1.0
                self.likely_room_cell_mask[yi, xi] = True
        for room_inst, cells in self._room_instance_cells.items():
            wx, wy = walkable_map.cell_to_world(*cells[0])
            rt_norm = walkable_map.normalize_room_type(
                walkable_map.room_at(wx, wy) or ""
            ).lower()
            if rt_norm in matched_likely_rooms:
                self._likely_room_instances.append(room_inst)
        # Cache last build for diagnostics.
        self.last_value = None
        self.matched_rooms_diagnostic = sorted(matched_likely_rooms)
        # If the fallback (no plan room matched) fires, room_prior is on
        # for every walkable cell in any room — `is_in_likely_room` then
        # acts as a no-op pass-through gate (correct: when LLM is wrong,
        # we shouldn't reject DINO admits anywhere).
        self._room_prior_is_fallback = (matched_likely_rooms == scene_rooms_norm)
        # ---- Pre-compute dilated admit mask (separate from room_prior) ----
        # room_prior stays sharp (used as value-map signal — must not bleed
        # into corridors). admit_mask is dilated by ROOM_ADMIT_DILATE_M so
        # back-projection noise + door/wall cells get a tolerance band.
        self.room_admit_mask = self.room_prior > 0
        if not self._room_prior_is_fallback and self.room_admit_mask.any() \
                and ROOM_ADMIT_DILATE_M > 0:
            try:
                from scipy.ndimage import binary_dilation
                iters = max(1, int(round(ROOM_ADMIT_DILATE_M / max(self.cell_m, 1e-3))))
                self.room_admit_mask = binary_dilation(self.room_admit_mask,
                                                          iterations=iters)
            except Exception:
                pass

    def is_in_likely_room(self, wx: float, wy: float) -> bool:
        """True iff world (wx, wy) maps to a cell within
        ROOM_ADMIT_DILATE_M of any cell whose room is in plan.likely_rooms
        (or fallback when no plan room matched). Used to gate DINO admits.
        Tolerance band catches doorway / window / wall cells where back-
        projection lands but the object is physically in the room."""
        try:
            yi, xi = self.wm.world_to_cell(wx, wy)
        except Exception:
            return True  # fail open: don't block on coord errors
        if yi < 0 or yi >= self.h or xi < 0 or xi >= self.w:
            return True
        return bool(self.room_admit_mask[yi, xi])

    # ---- Per-step update ----
    def build(self, target_memory: list[dict],
              agent_path: list[tuple[float, float]],
              direction_hint: dict | None = None) -> np.ndarray:
        """Return value map (HxW float32). Caller's responsibility to mask
        non-walkable for argmax (but we already do that here).

        `direction_hint` (optional, VLFM-style cone): dict with keys
            agent_xy: (x, y) current agent world position
            yaw: float radians, current agent forward direction
            relevance_score: int 0..10 — VLM's "is target in this direction?"
            explore_direction: str ∈ {forward, left, right, behind, no_clue}
        Cone is painted only when relevance_score >= 5 and explore_direction
        is not "no_clue".
        """
        v = np.zeros((self.h, self.w), dtype=np.float32)

        # Source A: target_memory clusters (Gaussian blob at each).
        # FIX C: snap each cluster XY to the nearest WALKABLE cell — DINO
        # bbox center on a tabletop/sculpture backprojects to a cell that's
        # not on the floor (z>0.7), and walkable_map.world_to_cell + grid
        # boolean returns False for that cell, so the blob lands "in a
        # wall" and never adds value to a cell A* can reach.
        sigma_cells = TARGET_SIGMA_M / self.cell_m
        for cluster in target_memory:
            xy = cluster.get("xy")
            score = float(cluster.get("score", 0.5))
            if not xy:
                continue
            yi, xi = self.wm.world_to_cell(xy[0], xy[1])
            if not (0 <= yi < self.h and 0 <= xi < self.w
                    and self.walkable[yi, xi]):
                # Snap to nearest walkable
                near = self.wm.nearby_walkable(xy[0], xy[1], radius_m=2.0)
                if near is None:
                    continue
                yi, xi = self.wm.world_to_cell(*near)
            placed = _gaussian_disk(v.shape, (yi, xi), sigma_cells,
                                     TARGET_WEIGHT * score)
            if placed is not None:
                slc, blob = placed
                v[slc] += blob

        # Source B: room_prior + Fix C2 dynamic per-room boost.
        # For each likely-room INSTANCE, check the visited fraction
        # (cells with explored_count > 0). If below ROOM_VISITED_FRACTION,
        # boost room_prior in those cells by ROOM_PRIOR_UNVISITED_BOOST.
        # This is what makes the agent leave the starting room and head
        # for an unvisited likely room without waiting for frontier-edge
        # adjacency.
        room_layer = self.room_prior.copy()
        ec_for_visit = getattr(self.wm, "explored_count", None)
        if ec_for_visit is not None and isinstance(ec_for_visit, np.ndarray) \
                and ec_for_visit.shape == (self.h, self.w) \
                and self._likely_room_instances:
            for room_inst in self._likely_room_instances:
                cells = self._room_instance_cells.get(room_inst, [])
                if not cells:
                    continue
                n_total = len(cells)
                n_visited = sum(1 for (yi, xi) in cells
                                  if ec_for_visit[yi, xi] > 0)
                frac = n_visited / max(n_total, 1)
                if frac < ROOM_VISITED_FRACTION:
                    for (yi, xi) in cells:
                        room_layer[yi, xi] *= ROOM_PRIOR_UNVISITED_BOOST
        v += ROOM_WEIGHT * room_layer

        # Source E: VLFM-style relevance cone (cheap cognitive direction)
        if direction_hint is not None:
            self._paint_cone(v, direction_hint)

        # Source C: frontier = BOUNDARY of explored region (InstructNav's
        # outer-border definition; not "any unexplored cell"). A cell is
        # frontier iff it's walkable, NOT yet visited (explored_count==0),
        # AND has at least one visited walkable neighbor (so it sits on
        # the edge of the explored area, not deep in the unknown). At
        # episode start everything is unvisited → no frontier yet → seed
        # with cells in a 1-meter ring around the agent's start (handled
        # outside this method via the visited_penalty term doing nothing
        # at step 0; agent's first step is then driven by room_prior +
        # target_memory).
        # FIX A: the attribute is `explored_count` (no leading underscore).
        ec = getattr(self.wm, "explored_count", None)
        if ec is not None and isinstance(ec, np.ndarray) \
                and ec.shape == (self.h, self.w):
            visited = (ec > 0) & self.walkable
            unvisited_walkable = (ec == 0) & self.walkable
            # FIX B: 8-neighborhood "has-visited-neighbor" via cheap shifts.
            has_visited_nbr = np.zeros_like(visited)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    sy0, sy1 = max(0, dy), self.h + min(0, dy)
                    sx0, sx1 = max(0, dx), self.w + min(0, dx)
                    ty0, ty1 = max(0, -dy), self.h + min(0, -dy)
                    tx0, tx1 = max(0, -dx), self.w + min(0, -dx)
                    has_visited_nbr[ty0:ty1, tx0:tx1] |= visited[sy0:sy1, sx0:sx1]
            frontier = unvisited_walkable & has_visited_nbr
            if frontier.any():
                v += FRONTIER_WEIGHT * frontier.astype(np.float32)
            else:
                # Step 0: nothing visited yet → seed exploration uniformly
                # over walkable so room_prior + target_memory dominate the
                # picking but argmax doesn't collapse to the visited cell.
                v += (FRONTIER_WEIGHT * 0.25
                      * self.walkable.astype(np.float32))
        else:
            v += FRONTIER_WEIGHT * self.walkable.astype(np.float32)

        # Source D: visited penalty (Gaussian negative)
        sigma_v_cells = VISITED_SIGMA_M / self.cell_m
        for (px, py) in agent_path:
            yi, xi = self.wm.world_to_cell(px, py)
            if not (0 <= yi < self.h and 0 <= xi < self.w):
                continue
            placed = _gaussian_disk(v.shape, (yi, xi), sigma_v_cells,
                                     VISITED_WEIGHT)
            if placed is not None:
                slc, blob = placed
                v[slc] -= blob

        # Mask non-walkable
        v[~self.walkable] = -1e9

        self.last_value = v
        return v

    def _reachable_mask(self, start_yx: tuple[int, int]) -> np.ndarray:
        """BFS over walkable grid from `start_yx`. Returns bool array True
        for every cell connected to start. FIX E: argmax over only the
        connected component prevents picking a cell that A* can't reach
        (which previously triggered the 0.3m straight-walk fallback —
        useless when argmax is across an obstacle)."""
        h, w = self.h, self.w
        seen = np.zeros((h, w), dtype=bool)
        if not (0 <= start_yx[0] < h and 0 <= start_yx[1] < w
                and self.walkable[start_yx]):
            return seen
        q = deque([start_yx])
        seen[start_yx] = True
        while q:
            yi, xi = q.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = yi + dy, xi + dx
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue
                    if seen[ny, nx] or not self.walkable[ny, nx]:
                        continue
                    seen[ny, nx] = True
                    q.append((ny, nx))
        return seen

    # ---- A* picker ----
    def next_waypoint(self, current_pose_xy: tuple[float, float],
                      target_memory: list[dict],
                      agent_path: list[tuple[float, float]],
                      step_m: float | None = None,
                      direction_hint: dict | None = None,
                      ) -> tuple[tuple[float, float], dict]:
        """Pick next waypoint via A* on walkable grid toward argmax(value).

        Returns ((wx, wy), meta) where meta has diagnostic fields:
          target_cell: (yi, xi) of value-map argmax
          target_xy: world coords of target cell
          path_len_cells: number of cells in A* path (None if no path)
          chosen_path_idx: index along path that became the waypoint
          fallback_reason: str if we couldn't A*, else None
        """
        step_m = step_m or STEP_M
        v = self.build(target_memory, agent_path,
                        direction_hint=direction_hint)
        # FIX E: mask non-reachable cells (BFS-connected component from pose)
        cur_yi, cur_xi = self.wm.world_to_cell(*current_pose_xy)
        if not (0 <= cur_yi < self.h and 0 <= cur_xi < self.w
                and self.walkable[cur_yi, cur_xi]):
            near = self.wm.nearby_walkable(*current_pose_xy, radius_m=1.5)
            if near is not None:
                cur_yi, cur_xi = self.wm.world_to_cell(*near)
        reachable = self._reachable_mask((cur_yi, cur_xi))
        v[~reachable] = -1e9
        # argmax over reachable walkable cells
        flat_idx = int(np.argmax(v))
        target_cell = (flat_idx // self.w, flat_idx % self.w)
        target_xy = self.wm.cell_to_world(*target_cell)
        meta = {"target_cell": list(target_cell),
                "target_xy": [round(target_xy[0], 3), round(target_xy[1], 3)],
                "target_value": float(v[target_cell]),
                "path_len_cells": None,
                "chosen_path_idx": None,
                "fallback_reason": None}

        # Snap current pose to nearest walkable
        cur_yi, cur_xi = self.wm.world_to_cell(*current_pose_xy)
        if not (0 <= cur_yi < self.h and 0 <= cur_xi < self.w
                and self.walkable[cur_yi, cur_xi]):
            near = self.wm.nearby_walkable(*current_pose_xy, radius_m=1.5)
            if near is None:
                meta["fallback_reason"] = "current pose unwalkable, no snap"
                return current_pose_xy, meta
            cur_yi, cur_xi = self.wm.world_to_cell(*near)

        # Same cell → already at target, return current pose
        if (cur_yi, cur_xi) == target_cell:
            meta["fallback_reason"] = "argmax = current cell"
            return current_pose_xy, meta

        # A*
        path = self._a_star((cur_yi, cur_xi), target_cell)
        if not path:
            meta["fallback_reason"] = "no A* path"
            # Fallback: walk straight 0.3m toward argmax cell
            tx, ty = target_xy
            dx, dy = tx - current_pose_xy[0], ty - current_pose_xy[1]
            dist = math.hypot(dx, dy)
            if dist < 1e-3:
                return current_pose_xy, meta
            r = min(0.3, dist)
            wx = current_pose_xy[0] + r * dx / dist
            wy = current_pose_xy[1] + r * dy / dist
            return (wx, wy), meta

        meta["path_len_cells"] = len(path)
        # Walk the path until accumulated euclidean ≥ step_m, return that cell.
        accum = 0.0
        prev_yx = path[0]
        chosen_idx = 0
        for i, (yi, xi) in enumerate(path[1:], 1):
            wx_prev, wy_prev = self.wm.cell_to_world(*prev_yx)
            wx_cur, wy_cur = self.wm.cell_to_world(yi, xi)
            accum += math.hypot(wx_cur - wx_prev, wy_cur - wy_prev)
            prev_yx = (yi, xi)
            chosen_idx = i
            if accum >= step_m:
                break
        meta["chosen_path_idx"] = chosen_idx
        wx, wy = self.wm.cell_to_world(*path[chosen_idx])
        return (wx, wy), meta

    def _paint_cone(self, v: np.ndarray, direction_hint: dict) -> None:
        """Add VLFM-style forward cone to value map IN-PLACE. Painted only
        when relevance_score >= 5 and explore_direction is not 'no_clue'.
        Cone weight = (relevance_score / 10) * CONE_WEIGHT.

        When CONE_VLFM_FUSION=1 (default), also:
          (X3.1) depth-mask: cells along a ray whose distance > depth
                 image at the corresponding pixel are excluded.
          (X3.2) confidence-weighted fusion with persistent state across
                 steps. Per-cell confidence = cos²(θ/(fov/2)·π/2),
                 highest at optical axis. New value/conf merged via
                 VLFM Eqs.: v_new = (c·v_curr + c_prev·v_prev)/(c+c_prev)
                 c_new² = (c² + c_prev²)/(c + c_prev).
        """
        rel = int(direction_hint.get("relevance_score", 0) or 0)
        explore = str(direction_hint.get("explore_direction", "no_clue") or "no_clue")
        if rel < 5 or explore == "no_clue":
            # Still paint persistent fused cone if any (decays naturally
            # because new observations would lower confidence over time).
            if CONE_VLFM_FUSION:
                v += self._cone_value
            return
        agent_xy = direction_hint.get("agent_xy")
        yaw = direction_hint.get("yaw")
        if agent_xy is None or yaw is None:
            if CONE_VLFM_FUSION:
                v += self._cone_value
            return
        # Map explore_direction to absolute cone yaw (relative to agent forward)
        offset = {"forward": 0.0, "left": math.pi / 2,
                  "right": -math.pi / 2, "behind": math.pi}.get(explore)
        if offset is None:
            if CONE_VLFM_FUSION:
                v += self._cone_value
            return
        cone_yaw = float(yaw) + float(offset)
        peak = (rel / 10.0) * CONE_WEIGHT

        # Bbox of cells within CONE_RANGE_M of agent (cheap iteration window)
        cells_r = int(math.ceil(CONE_RANGE_M / self.cell_m))
        cy_idx, cx_idx = self.wm.world_to_cell(agent_xy[0], agent_xy[1])
        y0 = max(0, cy_idx - cells_r)
        y1 = min(self.h, cy_idx + cells_r + 1)
        x0 = max(0, cx_idx - cells_r)
        x1 = min(self.w, cx_idx + cells_r + 1)
        if y1 <= y0 or x1 <= x0:
            if CONE_VLFM_FUSION:
                v += self._cone_value
            return
        # Build per-cell world coordinates by iterating cell_to_world.
        # (Avoids assumptions about grid origin / sign conventions.)
        wx_grid = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        wy_grid = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        for ly, gy in enumerate(range(y0, y1)):
            for lx, gx in enumerate(range(x0, x1)):
                wx_grid[ly, lx], wy_grid[ly, lx] = self.wm.cell_to_world(gy, gx)
        dx_arr = wx_grid - agent_xy[0]
        dy_arr = wy_grid - agent_xy[1]
        dist = np.hypot(dx_arr, dy_arr)
        bearing = np.arctan2(dy_arr, dx_arr)
        angdiff = np.arctan2(np.sin(bearing - cone_yaw),
                              np.cos(bearing - cone_yaw))
        # Distance falloff (linear) × angular Gaussian
        dist_w = np.clip(1.0 - dist / CONE_RANGE_M, 0.0, 1.0)
        ang_w = np.exp(-(angdiff ** 2) / (2.0 * CONE_SIGMA_RAD ** 2))

        if not CONE_VLFM_FUSION:
            # Legacy path: just add cone to v additively.
            cone = peak * dist_w * ang_w
            v[y0:y1, x0:x1] += cone.astype(np.float32)
            return

        # ---- X3.1 depth mask ----
        # Mask cells whose along-ray distance exceeds depth image at the
        # corresponding camera pixel column. depth=H×W meters, HFoV=π/2.
        depth = direction_hint.get("depth_image")
        hfov = float(direction_hint.get("hfov_rad", CONE_HFOV_RAD))
        in_fov = np.abs(angdiff) <= (hfov / 2.0 + 1e-3)
        if depth is not None:
            try:
                Hd, Wd = depth.shape[:2]
                # u_norm in [0,1] left→right; bearing within ±hfov/2.
                u_norm = 0.5 - angdiff / hfov   # angdiff +ve = left in std frame; sign chosen so left → smaller u
                u_idx = np.clip((u_norm * (Wd - 1)).astype(np.int32), 0, Wd - 1)
                # Sample depth at horizon row (middle of image).
                row = Hd // 2
                d_at_pix = depth[row, u_idx]
                # Cells beyond depth at that ray are occluded. Allow 0.3m slack.
                visible_mask = (dist <= d_at_pix + 0.3) & in_fov
            except Exception:
                visible_mask = in_fov
        else:
            visible_mask = in_fov

        # ---- X3.2 confidence ----
        # cos²(θ/(fov/2)·π/2): 1 on optical axis, 0 at FOV edge
        with np.errstate(divide="ignore", invalid="ignore"):
            theta_norm = np.clip(np.abs(angdiff) / (hfov / 2.0), 0.0, 1.0)
        c_curr = np.cos(theta_norm * (math.pi / 2.0)) ** 2
        c_curr = c_curr * dist_w * visible_mask.astype(np.float32)
        v_curr = peak * c_curr   # value scaled by confidence (so cell value is the cone peak weighted by reliability)

        # Persistent state slice
        v_prev = self._cone_value[y0:y1, x0:x1]
        c_prev = self._cone_conf[y0:y1, x0:x1]

        # VLFM fusion (avoid div-by-zero)
        denom = c_curr + c_prev
        safe = denom > 1e-6
        v_fused = np.where(safe,
                            (c_curr * v_curr + c_prev * v_prev) / np.maximum(denom, 1e-6),
                            v_prev)
        c_fused2 = np.where(safe,
                             (c_curr ** 2 + c_prev ** 2) / np.maximum(denom, 1e-6),
                             c_prev ** 2)
        c_fused = np.sqrt(np.clip(c_fused2, 0.0, None))

        self._cone_value[y0:y1, x0:x1] = v_fused.astype(np.float32)
        self._cone_conf[y0:y1, x0:x1] = c_fused.astype(np.float32)
        # Add persistent cone state to value map.
        v += self._cone_value

    def _a_star(self, start: tuple[int, int], goal: tuple[int, int]
                ) -> list[tuple[int, int]] | None:
        """8-connected A* on walkable grid. Returns list of (yi, xi) cells
        from start to goal inclusive, or None if unreachable. Reuses the
        WalkableMap.geodesic_distance_2d structure but returns the path."""
        h, w = self.h, self.w

        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        open_heap = [(heuristic(start, goal), 0.0, start)]
        g_score = {start: 0.0}
        came_from: dict = {}
        closed = set()
        neighbors = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                     (-1, -1, 1.41421356), (-1, 1, 1.41421356),
                     (1, -1, 1.41421356), (1, 1, 1.41421356)]
        while open_heap:
            _, g, cur = heapq.heappop(open_heap)
            if cur == goal:
                # Reconstruct path
                path = [cur]
                while cur in came_from:
                    cur = came_from[cur]
                    path.append(cur)
                path.reverse()
                return path
            if cur in closed:
                continue
            closed.add(cur)
            for dy, dx, cost in neighbors:
                nb = (cur[0] + dy, cur[1] + dx)
                if not (0 <= nb[0] < h and 0 <= nb[1] < w):
                    continue
                if not self.walkable[nb]:
                    continue
                new_g = g + cost
                if new_g < g_score.get(nb, float("inf")):
                    g_score[nb] = new_g
                    came_from[nb] = cur
                    heapq.heappush(open_heap, (new_g + heuristic(nb, goal),
                                                new_g, nb))
        return None

    # ---- STOP check (engine side) ----
    def should_stop(self, current_pose_xy: tuple[float, float],
                    target_memory: list[dict]) -> tuple[bool, dict]:
        """Returns (stop_bool, meta). True iff at least one target_memory
        cluster with score >= STOP_SCORE_MIN is within STOP_RADIUS_M of
        current pose. Caller should ALSO check VLM 'see target' verdict
        before committing — engine STOP alone risks early termination on
        bad memory. Pair: (engine_should_stop AND vlm_sees_target) → STOP."""
        meta = {"closest_dist": None, "closest_score": None,
                 "reason": None}
        if not target_memory:
            meta["reason"] = "no memory"
            return False, meta
        best = None
        for c in target_memory:
            xy = c.get("xy")
            sc = float(c.get("score", 0.5))
            if not xy or sc < STOP_SCORE_MIN:
                continue
            d = math.hypot(xy[0] - current_pose_xy[0],
                           xy[1] - current_pose_xy[1])
            if best is None or d < best[0]:
                best = (d, sc, c)
        if best is None:
            meta["reason"] = "no cluster above STOP_SCORE_MIN"
            return False, meta
        meta["closest_dist"] = round(best[0], 3)
        meta["closest_score"] = round(best[1], 3)
        meta["closest_cluster"] = best[2]
        if best[0] <= STOP_RADIUS_M:
            meta["reason"] = "in radius + above min score"
            return True, meta
        meta["reason"] = f"closest cluster {best[0]:.2f}m > {STOP_RADIUS_M}m"
        return False, meta
