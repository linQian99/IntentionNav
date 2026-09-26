"""Isaac Sim environment wrapper for IntentEQA active-EQA evaluation.

Wraps a single Isaac Sim SimulationApp instance and provides a batched API
for running many episodes back-to-back. Reuses the camera/scene patterns
from top-level demo.py's `FloatingCameraController`.

IMPORTANT — runtime requirements (not pip-installable):
  conda activate goodnav
  source $ISAACSIM_ROOT/setup_conda_env.sh

This module imports isaacsim / pxr / carb / omni at class init time, so
importing it from outside that env will raise. Agents should only `import`
this module inside a function called from the Isaac-Sim-side entry point.

Usage sketch:
  env = IsaacSimEnv(headless=True)
  for episode in episodes:
      env.load_scene(episode["scene_id"], usd_path)
      env.place_agent(episode["start_position"], episode["start_rotation_quat_wxyz"])
      for step in range(step_cap):
          rgb = env.render_rgb()
          waypoints = env.sample_frontier_waypoints(walkable_map=wm, K=8)
          image_with_markers = overlay_waypoints(rgb, waypoints)
          action = agent.act(image_with_markers, intent)
          if action == "STOP": break
          env.teleport_to(waypoints[action_idx])
      env.save_final_frame(out_path)
  env.close()

Camera parity with capture_surfaces.py:
  focal_length = 10, horizontal_aperture = 20, vertical_aperture = 20,
  height = 1.5m, horizontal yaw only (no pitch).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# NOTE: isaacsim / pxr imports deferred to __init__ — see module docstring.


CAMERA_HEIGHT = 1.5
FOCAL_LENGTH = 10.0
HORIZONTAL_APERTURE = 20.0
VERTICAL_APERTURE = 20.0
RENDER_RESOLUTION = (768, 768)   # 768 stays in low-res VLM tier (Gemini Flash 258 tokens),
                                  # but pixel detail helps small-object recognition (plate /
                                  # menorah / air_purifier) — 512 was visibly losing detail
                                  # on Kujiale renders.


def yaw_from_quat_wxyz(quat_wxyz) -> float:
    w, x, y, z = quat_wxyz
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_look_dir(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)


class IsaacSimEnv:
    """Singleton-style Isaac Sim wrapper. Create once per process."""

    def __init__(self, headless: bool = True, render_scale: int = 1):
        # Defer Isaac Sim imports until instance creation (module still importable
        # outside goodnav env for static analysis / scaffolding).
        from isaacsim import SimulationApp
        # Render quality boost (v8): default Isaac config produced very soft,
        # painterly RGBs (visible edge mush, illegible book spines) because
        # the default RTX RealTime path uses TAA + low SPP + heavy denoiser
        # — fine for moving viewport but bad for single-shot navigation
        # frames. Switch AA from TAA → DLAA (spatial only, no temporal
        # smearing) and bump pathtracing-style denoiser quality. Doesn't
        # change perf class — DLAA is GPU-cheap; the bigger cost is the
        # subdiv level bump set via carb below.
        self._app = SimulationApp({
            "headless": headless,
            "width": RENDER_RESOLUTION[0] * render_scale,
            "height": RENDER_RESOLUTION[1] * render_scale,
            "anti_aliasing": 3,            # 0=off 1=TAA 2=FXAA 3=DLSS+DLAA (default 1)
            "renderer": "RayTracedLighting",
        })

        import carb  # noqa: F401 (imported for side effects)
        from isaacsim.core.api import World  # noqa: F401
        import omni.replicator.core as rep  # noqa: F401

        # Tune Carb settings for sharper output:
        # - DLAA op (3) is the sharpest spatial AA; suppresses TAA's blur on still frames
        # - subdiv level 2 (default 0/1) yields smoother surfaces on Kujiale USDs
        # - DLSS quality 2 = "Quality" preset (vs Performance), no upscale artifacts at 1:1
        # If a setting is unknown to this Kit version, set() is a no-op.
        #
        # Disable async rendering throttling — required when ≥3 Isaac Sim
        # processes share one GPU. Async render gives back partial/empty
        # frames at random when the renderer hasn't caught up by deadline.
        # Confirmed in Isaac Sim 5.1 troubleshooting docs and GitHub
        # issue #507 ("Replicator annotators giving empty images on
        # random frames when using multiple GPUs"). With this off, render
        # is synchronous and frames are guaranteed to be the requested
        # camera output. ~5-15% slower per render but 100% reliable.
        try:
            settings = carb.settings.get_settings()
            settings.set("/rtx/post/aa/op", 3)
            settings.set("/rtx/raytracing/subdivisionLevel", 2)
            settings.set("/rtx/post/dlss/execMode", 2)
            # Crank a few more knobs that hurt perceived sharpness:
            settings.set("/rtx/sceneDb/ambientLightIntensity", 1.0)
            settings.set("/rtx/raytracing/cached/enabled", True)
            # Force synchronous rendering — kills random black-frame issue
            # when multiple Isaac Sim processes share a GPU.
            settings.set("/exts/isaacsim.core.throttling/enable_async", False)
            settings.set("/app/asyncRendering", False)
            settings.set("/app/asyncRenderingLowLatency", False)
        except Exception as e:
            print(f"[iss_env] carb settings tweak skipped: {e}")

        self._current_scene_id: Optional[str] = None
        self._world = None
        self._camera_prim_path = "/World/IntentEQACamera"
        self._render_product = None
        self._annotator_rgb = None
        self._annotator_depth = None

        self._current_position = np.array([0.0, 0.0, CAMERA_HEIGHT])
        self._look_dir = np.array([1.0, 0.0, 0.0])

        self._init_world()

    # ---------------- World / camera init ----------------
    def _init_world(self):
        from isaacsim.core.api import World
        from pxr import Sdf
        self._world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0 / 200.0,
            rendering_dt=8.0 / 200.0,
        )
        self._world.scene.add_default_ground_plane(
            z_position=0, name="default_ground_plane",
            prim_path="/World/defaultGroundPlane",
        )
        stage = self._app.context.get_stage()
        dome = stage.DefinePrim("/World/DomeLight", "DomeLight")
        dome.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(450.0)
        self._create_camera()

    def _create_camera(self):
        from pxr import UsdGeom
        stage = self._app.context.get_stage()
        cam = UsdGeom.Camera.Define(stage, self._camera_prim_path)
        cam.CreateFocalLengthAttr(FOCAL_LENGTH)
        cam.CreateClippingRangeAttr().Set((0.01, 1000.0))
        cam.CreateHorizontalApertureAttr(HORIZONTAL_APERTURE)
        cam.CreateVerticalApertureAttr(VERTICAL_APERTURE)

        # Build replicator render product bound to this camera
        import omni.replicator.core as rep
        self._render_product = rep.create.render_product(
            self._camera_prim_path, resolution=RENDER_RESOLUTION,
        )
        self._annotator_rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        self._annotator_rgb.attach(self._render_product)
        # Depth (distance to image plane in meters) — used for proximity
        # grounding at STOP decision. Per-pixel depth from this camera.
        self._annotator_depth = rep.AnnotatorRegistry.get_annotator(
            "distance_to_image_plane"
        )
        self._annotator_depth.attach(self._render_product)

    # ---------------- Scene management ----------------
    def load_scene(self, scene_id: str, usd_path: str):
        """Load a new scene. If scene is already loaded, do nothing."""
        if scene_id == self._current_scene_id:
            return
        self._clear_current_scene()
        from isaacsim.core.utils.prims import define_prim
        prim = define_prim("/World/Ground", "Xform")
        prim.GetReferences().AddReference(usd_path, "/Root")
        self._current_scene_id = scene_id
        self._world.reset()
        # Warm up the renderer (2-3 steps clears initial NaN frames)
        for _ in range(3):
            self._world.step(render=True)

    def _clear_current_scene(self):
        if self._current_scene_id is None:
            return
        from pxr import Sdf
        stage = self._app.context.get_stage()
        prim = stage.GetPrimAtPath("/World/Ground")
        if prim.IsValid():
            stage.RemovePrim("/World/Ground")
        self._current_scene_id = None

    # ---------------- Agent pose ----------------
    def place_agent(self, position_xyz, rotation_quat_wxyz):
        """Teleport the camera to (position, rotation). Rotation is yaw-only."""
        self._current_position = np.array(position_xyz, dtype=np.float64)
        if abs(self._current_position[2]) < 1e-6:
            self._current_position[2] = CAMERA_HEIGHT
        yaw = yaw_from_quat_wxyz(rotation_quat_wxyz)
        self._look_dir = yaw_to_look_dir(yaw)
        self._update_camera_transform()

    def teleport_to(self, target_xy, face_direction_xy=None):
        """Move camera to new (x, y) at CAMERA_HEIGHT. Face towards
        face_direction_xy if given, else toward the target.
        """
        new_pos = np.array([target_xy[0], target_xy[1], CAMERA_HEIGHT])
        if face_direction_xy is not None:
            dx = face_direction_xy[0] - new_pos[0]
            dy = face_direction_xy[1] - new_pos[1]
        else:
            dx = target_xy[0] - self._current_position[0]
            dy = target_xy[1] - self._current_position[1]
        if abs(dx) + abs(dy) > 1e-6:
            yaw = math.atan2(dy, dx)
            self._look_dir = yaw_to_look_dir(yaw)
        self._current_position = new_pos
        self._update_camera_transform()

    def _update_camera_transform(self):
        """Apply position + look direction to the camera prim.

        Build cam→world rotation correctly:
          rot_mat columns = [right, cam_up, -fwd]
            (Isaac camera looks down its OWN -Z axis, so cam_z = -fwd in world)
        Then decompose to XYZ-Euler (same op type as the legacy code, so we
        avoid the OrientOp+Quatd path that triggered an Isaac silent-hang)
        and apply via AddRotateXYZOp without any manual axis-swap hack.

        Old code used non-standard column order [right, fwd, up2] then added
        +90°/-90° to euler X/Z to "convert" — but euler decomposition is
        non-unique near gimbal configs, so that hack worked at some yaws and
        broke at others (delta of pose["yaw"] vs actual camera direction
        was 60-160° depending on yaw, verified empirically across SR=1 and
        SR=0 cases). The right column order eliminates the need for any
        post-decomposition fixup.
        """
        from pxr import UsdGeom, Gf
        from scipy.spatial.transform import Rotation as R
        stage = self._app.context.get_stage()
        cam = stage.GetPrimAtPath(self._camera_prim_path)
        xformable = UsdGeom.Xformable(cam)
        xformable.ClearXformOpOrder()
        t = xformable.AddTranslateOp()
        t.Set(Gf.Vec3d(*self._current_position.tolist()))

        world_up = np.array([0.0, 0.0, 1.0])
        fwd = self._look_dir / max(np.linalg.norm(self._look_dir), 1e-9)
        right = np.cross(fwd, world_up)
        right /= max(np.linalg.norm(right), 1e-9)
        cam_up = np.cross(right, fwd)
        # Standard cam→world: cam_x→right, cam_y→up, cam_z→-fwd
        rot_mat = np.column_stack([right, cam_up, -fwd])
        euler = R.from_matrix(rot_mat).as_euler("xyz", degrees=True)
        rotate_op = xformable.AddRotateXYZOp()
        rotate_op.Set(Gf.Vec3f(float(euler[0]), float(euler[1]), float(euler[2])))

    def find_open_view(self, walkable_map=None, threshold_m: float = 1.0,
                       n_directions: int = 16) -> dict:
        """Pick an unblocked yaw via freemap ray-cast (no Isaac render).

        Old version rendered depth in 4 cardinal directions to detect a
        wall-pinned spawn — 5 expensive renders per episode-start. New
        version uses pure 2D geometry: from (x, y) at agent's spawn, ray-
        cast on freemap in n_directions evenly-spaced yaws. Pick the one
        with the longest unobstructed ray. ~ms vs seconds.

        Limitations vs depth-render approach:
        - Tall furniture (wardrobe/bed) is invisible to freemap (it sits
          on a walkable cell at floor level). If spawn yaw faces such a
          piece, the freemap call won't reorient — but the dataset-level
          fix in `scripts/fix_dataset_start_orientations_geom.py` already
          uses the same freemap geometry, so any fixable case has been
          fixed offline.
        - We accept the ~residual cases that need visual depth checks;
          they are the long tail. Matching the dataset-level method
          keeps the fixup deterministic + reproducible.

        Returns {scanned: [(delta_deg, clear_dist_m)],
                 picked_delta_deg, init_dist, best_dist, reason}.
        """
        if walkable_map is None:
            return {"scanned": [], "picked_delta_deg": 0, "init_dist": None,
                    "reason": "no_walkable_map_supplied"}
        cur_x = float(self._current_position[0])
        cur_y = float(self._current_position[1])
        cur_yaw = math.atan2(float(self._look_dir[1]),
                             float(self._look_dir[0]))
        # Distance the current yaw can clear forward.
        init_dist = walkable_map.max_clear_distance(cur_x, cur_y, cur_yaw)
        if init_dist >= threshold_m:
            return {"scanned": [(0, init_dist)], "picked_delta_deg": 0,
                    "init_dist": init_dist, "reason": "no_reorient_needed"}
        # Scan n_directions, pick longest clear ray.
        best_yaw, best_dist, samples = walkable_map.pick_open_yaw(
            cur_x, cur_y, current_yaw=cur_yaw, n_directions=n_directions,
        )
        delta_deg = math.degrees(best_yaw - cur_yaw)
        # Normalize to (-180, 180]
        delta_deg = ((delta_deg + 180) % 360) - 180
        # Convert samples to (delta_deg, dist) for logging.
        scanned = []
        for theta, d in samples:
            dd = ((math.degrees(theta - cur_yaw) + 180) % 360) - 180
            scanned.append((round(dd, 1), round(d, 2)))
        if best_dist <= init_dist:
            # No direction is better than current — keep current yaw.
            return {"scanned": scanned, "picked_delta_deg": 0,
                    "init_dist": init_dist, "best_dist": best_dist,
                    "reason": "no_better_direction"}
        # Rotate to best
        self._look_dir = np.array(
            [math.cos(best_yaw), math.sin(best_yaw), 0.0]
        )
        self._update_camera_transform()
        return {"scanned": scanned, "picked_delta_deg": round(delta_deg, 1),
                "init_dist": init_dist, "best_dist": best_dist,
                "reason": "freemap_reoriented"}

    def look_at_yaw(self, target_yaw: float) -> None:
        """Rotate camera in place to absolute world yaw. No teleport, no
        walkable check (rotating in place is always safe). Used by the
        engine agent's episode-start panoramic DINO scan."""
        cos_y, sin_y = math.cos(target_yaw), math.sin(target_yaw)
        self._look_dir = np.array([cos_y, sin_y, 0.0])
        self._update_camera_transform()

    def look(self, direction: str) -> bool:
        """Rotate camera in place by 90° (left/right/back). No teleport,
        no walkable check (rotating in place is always safe).

        Used when the agent is at a position close to target but facing
        the wrong way — gives it a chance to find the object visually
        without burning a waypoint move. Cost: one VLM call but no
        spatial progress.

        Returns True on success, False if direction unknown.
        """
        d = (direction or "").strip().lower()
        if d == "left":   delta = math.pi / 2
        elif d == "right": delta = -math.pi / 2
        elif d == "back":  delta = math.pi
        else: return False
        cur_yaw = math.atan2(float(self._look_dir[1]), float(self._look_dir[0]))
        new_yaw = cur_yaw + delta
        cos_y, sin_y = math.cos(new_yaw), math.sin(new_yaw)
        self._look_dir = np.array([cos_y, sin_y, 0.0])
        self._update_camera_transform()
        return True

    def creep_forward(self, walkable_map, max_creep_m: float = 1.0
                      ) -> float:
        """Take a small step along the current camera look direction.

        Used at STOP time to close the gap between the discrete-waypoint
        sampler (min_step=0.5m) and the canonical 1m ObjectNav success
        radius. Without this, ~16% of episodes land at 1.0–2.0 m from the
        target — agent identified it correctly, just couldn't reach 1m.

        Tries `max_creep_m` (default 1.0m), then progressively smaller
        distances (0.7, 0.5, 0.3, 0.2, 0.1) if blocked by walkable / LoS
        check. Returns the actual distance moved (0.0 if fully blocked).

        Not an oracle — uses only camera yaw + freemap. Engine-aided
        rather than oracle-aided: relaxes the discrete-waypoint
        quantization at termination only, similar in spirit to ESC /
        VLFM subgoal-style final approach.
        """
        cx = float(self._current_position[0])
        cy = float(self._current_position[1])
        yaw = math.atan2(float(self._look_dir[1]), float(self._look_dir[0]))
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        for d in (max_creep_m, 0.7, 0.5, 0.3, 0.2, 0.1):
            nx = cx + d * cos_y
            ny = cy + d * sin_y
            if (walkable_map.is_walkable(nx, ny)
                    and walkable_map.line_of_sight(cx, cy, nx, ny)):
                # Preserve yaw by passing a face_direction along current look
                self.teleport_to((nx, ny),
                                 face_direction_xy=(nx + cos_y, ny + sin_y))
                return d
        return 0.0

    # ---------------- Rendering ----------------
    # Accumulate this many sub-frames before grabbing the final RGB. Without
    # multiple steps, DLAA + RTX RealTime denoiser gets a single-frame raw
    # path-traced output → "salt-pepper" noise on textured surfaces (rugs,
    # pillows) that fools DINO into seeing nonexistent objects. 8 ticks let
    # the temporal denoiser converge while still being cheap (~80ms total).
    # Override via env var ISS_RENDER_TICKS for ablation.
    _RENDER_TICKS = int(os.environ.get("ISS_RENDER_TICKS", "8"))

    def render_rgb(self) -> np.ndarray:
        """Step sim several ticks + return final RGB (H, W, 3) uint8.
        Multiple steps allow the RTX temporal denoiser to clean path-tracing
        sample noise — single-step rendering produced visible salt-pepper
        artifacts that broke downstream DINO grounding."""
        import omni.replicator.core as rep
        for _ in range(self._RENDER_TICKS):
            self._world.step(render=True)
        rep.orchestrator.step(delta_time=0.0, pause_timeline=False)
        data = self._annotator_rgb.get_data()
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        # data is (H, W, 4) RGBA or (H, W, 3) RGB
        img = np.asarray(data)
        if img.shape[-1] == 4:
            img = img[..., :3]
        return img.astype(np.uint8)

    def render_depth(self) -> np.ndarray | None:
        """Return per-pixel depth (H, W) float32 in meters, or None if
        the depth annotator hasn't produced data yet (first call after
        scene swap). Caller should run after render_rgb to ensure the
        replicator has stepped."""
        if self._annotator_depth is None:
            return None
        try:
            data = self._annotator_depth.get_data()
        except Exception:
            return None
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        d = np.asarray(data)
        if d.ndim < 2:
            return None
        return d.astype(np.float32)

    def central_min_depth(self, depth: np.ndarray | None,
                           crop_frac: float = 0.5) -> float | None:
        """Min finite depth (meters) inside the central `crop_frac` of
        the frame. Used as proximity proxy: 'how close is the nearest
        thing in front of me'. Returns None on bad data.

        crop_frac=0.5 → central 50% of width AND height.
        Excludes inf/nan values which Replicator emits for sky / no-hit.
        """
        if depth is None:
            return None
        h, w = depth.shape[:2]
        if h < 4 or w < 4:
            return None
        cy, cx = h // 2, w // 2
        rh, rw = int(h * crop_frac / 2), int(w * crop_frac / 2)
        crop = depth[cy - rh:cy + rh, cx - rw:cx + rw]
        finite = crop[np.isfinite(crop) & (crop > 0.05)]  # exclude near-0 noise
        if finite.size == 0:
            return None
        return float(np.min(finite))

    # 3×3 grid cell layout (row, col) — keys match agent_system.txt prompt
    # so VLM strings like "top-left", "center", "btm-right" map directly.
    _GRID_RC = {
        "top-left":     (0, 0), "top":     (0, 1), "top-right":     (0, 2),
        "center-left":  (1, 0), "center":  (1, 1), "center-right":  (1, 2),
        "btm-left":     (2, 0), "btm":     (2, 1), "btm-right":     (2, 2),
        # Common synonyms / typos
        "bottom-left": (2, 0), "bottom": (2, 1), "bottom-right": (2, 2),
        "left":        (1, 0), "right":  (1, 2),
    }

    def depth_grid_summary(self, depth: np.ndarray | None,
                            crop_frac: float = 0.25) -> dict[str, float | None]:
        """Compute min finite depth in each of the 9 grid cells. Used
        to feed the VLM a depth-per-cell map so it has actual distance
        numbers when picking target_loc / deciding STOP.

        Returns dict {cell_label: depth_m or None}. None means no
        finite depth in that cell (sky / no-hit) — the VLM should
        treat as far/unknown."""
        out: dict[str, float | None] = {}
        # Use canonical 9 keys (no synonyms — keeps prompt compact)
        for label in ("top-left","top","top-right",
                       "center-left","center","center-right",
                       "btm-left","btm","btm-right"):
            out[label] = self.depth_at_grid(depth, label, crop_frac=crop_frac)
        return out

    def depth_at_grid(self, depth: np.ndarray | None,
                       grid_label: str | None,
                       crop_frac: float = 0.25) -> float | None:
        """Min finite depth in one cell of a 3x3 grid over the depth
        image. `grid_label` is the VLM's pointer (e.g. 'center',
        'btm-left'). crop_frac=0.25 → cell width/height = 1/3 of frame
        × 0.75 padding (avoid bleeding into adjacent cells).

        Returns None on missing depth or unrecognized label. Use this
        to ground Force-STOP at the location VLM claims target is in,
        not at the geometric image center.
        """
        if depth is None or not grid_label:
            return None
        rc = self._GRID_RC.get(grid_label.strip().lower())
        if rc is None:
            return None
        r, c = rc
        h, w = depth.shape[:2]
        if h < 6 or w < 6:
            return None
        # Cell center
        cy = int(h * (r + 0.5) / 3)
        cx = int(w * (c + 0.5) / 3)
        # Cell half-extent (each cell is h/3 × w/3, take crop_frac of full)
        rh = int(h * crop_frac / 2)
        rw = int(w * crop_frac / 2)
        y0, y1 = max(0, cy - rh), min(h, cy + rh)
        x0, x1 = max(0, cx - rw), min(w, cx + rw)
        crop = depth[y0:y1, x0:x1]
        finite = crop[np.isfinite(crop) & (crop > 0.05)]
        if finite.size == 0:
            return None
        return float(np.min(finite))

    # ---------------- Frontier sampling (VLMnav-aligned action proposer) ----
    def sample_frontier_waypoints(self, walkable_map, K: int = 8,
                                   min_step_m: float = 0.5,
                                   max_step_m: float = 1.7,
                                   clip_frac: float = 0.66,
                                   num_theta: int = 30,
                                   front_half_cone_rad: float = None,
                                   min_separation_rad: float = None,
                                   explore_bias: float = 4.0,
                                   explored_density_threshold: float = 0.4,
                                   explored_radius_m: float = 0.5,
                                   force_angular_spread_rad: float | None = None,
                                   seed: int = None) -> list[tuple[float, float, float]]:
        """VLMnav-style action proposer (`_action_proposer`) ported to our
        freemap. Returns a list of (x, y, angle_offset_rad) triples sorted
        by angle. K is a hard cap (default 8 = LABELS A-H).

        Algorithm (VLMnav `agent.py:_action_proposer`):
        1. Cast num_theta evenly-spaced rays in the front cone [-α, +α]
           where α = FOV*1.5/2, find max walkable distance on freemap.
        2. Filter rays with r < min_step_m.
        3. Compute clip_frac × r as the action's effective distance, capped
           at max_step_m. Also compute is_unexplored = (density < threshold).
        4. Seed-and-walk with explore bias:
           - seed = longest UNEXPLORED action
           - walk left + right from seed adding actions when the angle gap
             to the last picked is > min_separation_rad * 0.9
           - then add EXPLORED actions when gap > min_separation_rad * explore_bias
        5. If no unexplored seed: pure spacing fallback (longest + walk).
        6. If still empty: default fan = 4 actions at ±α*0.7, ±α*0.7/4.
        7. Sort by angle, return up to K.

        Defaults mirror VLMnav ObjectNav config (`config/ObjectNav.yaml`).
        explore_bias=0 disables the unexplored preference (pure spacing).
        """
        import random
        _rng = random.Random(seed if seed is not None else 0)
        FOV = math.pi / 2
        if front_half_cone_rad is None:
            front_half_cone_rad = FOV * 1.5 / 2          # ±67.5°
        if min_separation_rad is None:
            min_separation_rad = FOV / 360.0             # ≈ 0.25°: effectively off

        cur_x = float(self._current_position[0])
        cur_y = float(self._current_position[1])
        cur_yaw = math.atan2(float(self._look_dir[1]), float(self._look_dir[0]))

        # Helper: per-ray max distance walkable on freemap.
        if hasattr(walkable_map, "max_clear_distance"):
            def ray_dist(theta):
                return walkable_map.max_clear_distance(
                    cur_x, cur_y, cur_yaw + theta,
                    probe_distances=[0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
                                     2.0, 2.25, 2.5, 3.0]
                )
        else:
            def ray_dist(theta):
                # Fallback: walk a fine ray manually.
                d = 0.0
                for step_d in [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
                               2.0, 2.25, 2.5, 3.0]:
                    cx = cur_x + step_d * math.cos(cur_yaw + theta)
                    cy = cur_y + step_d * math.sin(cur_yaw + theta)
                    if not walkable_map.is_walkable(cx, cy):
                        break
                    if not walkable_map.line_of_sight(cur_x, cur_y, cx, cy):
                        break
                    d = step_d
                return d

        # ----- Step 1+2: evenly-spaced rays, filter by min_action_dist ------
        def _try_ray(theta_off: float) -> list | None:
            r = ray_dist(theta_off)
            if r is None or r < min_step_m:
                return None
            mag = min(clip_frac * r, max_step_m)
            if mag < min_step_m:
                return None
            cx = cur_x + mag * math.cos(cur_yaw + theta_off)
            cy = cur_y + mag * math.sin(cur_yaw + theta_off)
            if hasattr(walkable_map, "explored_density_at"):
                density = walkable_map.explored_density_at(
                    cx, cy, radius_m=explored_radius_m)
                is_unexplored = density < explored_density_threshold
            else:
                is_unexplored = True
            return [mag, theta_off, is_unexplored, cx, cy]

        thetas = np.linspace(-front_half_cone_rad, front_half_cone_rad, num_theta)
        arrowData: list[list] = []
        for theta in thetas:
            e = _try_ray(float(theta))
            if e is not None:
                arrowData.append(e)

        # Stage-2 cascade: when the front cone is sparse (wall-pinned, dead-end,
        # wrong-room corner), augment with rear-arc rays so the agent has a
        # turn-around option. Restores the cascade behavior of the pre-VLMnav
        # sampler (commit 9583aa7); diagnosis: smoke_vlmnav OSR@1.5-3m fell to
        # 10.4% (vs 22-25% across other variants) because the agent could only
        # walk further into wrong rooms with no behind-the-shoulder fallback.
        # Trigger: < K candidates from the front cone. The rear rays are added
        # to the same arrowData and pass through the same explore-bias seed-
        # walk, so they only get picked when they're materially better than
        # whatever front offers (longest-unexplored seed selection).
        # Trigger rear-arc augmentation when (a) front yielded < K candidates
        # OR (b) force_angular_spread is on (need full-360° pool for slice-
        # pick to populate non-front quadrants).
        if len(arrowData) < K or force_angular_spread_rad is not None:
            rear_count = max(8, num_theta - len(arrowData))
            rear_thetas = np.linspace(front_half_cone_rad,
                                       2 * math.pi - front_half_cone_rad,
                                       rear_count, endpoint=False)[1:]
            for theta in rear_thetas:
                t = float(theta if theta <= math.pi else theta - 2 * math.pi)
                e = _try_ray(t)
                if e is not None:
                    arrowData.append(e)

        if not arrowData:
            return self._default_fan(cur_x, cur_y, cur_yaw,
                                     front_half_cone_rad, max_step_m)

        # Sort by theta (leftmost first).
        arrowData.sort(key=lambda x: x[1])

        # ----- Track A: angular-spread slice-pick (overrides seed-walk) -----
        # When `force_angular_spread_rad` is set, divide [-π, π] into K equal
        # slices centered on agent forward (slice 0 = [-π/K, +π/K], etc.)
        # and pick the longest-unexplored ray inside each slice. Falls back
        # to longest-explored if a slice has no unexplored candidate. Drops
        # empty slices, fills any remaining slots from the largest θ-gap
        # between picked candidates. Skips the seed-walk and down-sample
        # below entirely. §10 follow-up: tests whether the broken VLM
        # picker comes from clustered K candidates rather than a
        # fundamentally bad primitive.
        if force_angular_spread_rad is not None:
            slice_w = 2 * math.pi / K
            # K slice centers at 0, slice_w, 2*slice_w, ... wrapped to (-π, π].
            slice_centers = []
            for k in range(K):
                c = k * slice_w
                if c > math.pi:
                    c -= 2 * math.pi
                slice_centers.append(c)
            # For each slice, find the candidate whose θ is closest to the
            # slice center. Tiebreak (and quality filter) by mag — among the
            # top-3 closest, prefer longest. Picking by closeness rather than
            # by longest avoids the edge-collision problem where two adjacent
            # slices each pick a candidate near their shared boundary.
            picked: list[list] = []
            picked_ids = set()
            for center in slice_centers:
                # Score = abs angular distance to center (smaller = better).
                def angdist(a, c=center):
                    d = abs(a[1] - c)
                    return min(d, 2 * math.pi - d)
                in_slice = [a for a in arrowData
                            if angdist(a) <= slice_w / 2 + 1e-6
                            and id(a) not in picked_ids]
                if not in_slice:
                    continue
                # Pick the candidate **closest to slice center**. Picking by
                # longest mag puts us at slice edges, where two adjacent
                # slice picks can collide right across the boundary. Prefer
                # unexplored: if any unexplored exists in slice, restrict
                # the closeness search to those.
                unexplored = [a for a in in_slice if a[2]]
                pool = unexplored if unexplored else in_slice
                pick = min(pool, key=lambda x: angdist(x))
                picked.append(pick)
                picked_ids.add(id(pick))
            # Backfill empty slices from largest θ-gap (longest ray inside the gap).
            # Less common with closeness-based picking but still possible when
            # a slice has zero walkable candidates.
            while len(picked) < K and picked:
                picked.sort(key=lambda x: x[1])
                thetas = [p[1] for p in picked]
                gaps = []
                for i in range(len(thetas)):
                    lo = thetas[i]
                    hi = (thetas[(i + 1) % len(thetas)]
                          + (2 * math.pi if i + 1 == len(thetas) else 0))
                    gaps.append((hi - lo, lo, hi))
                _, lo, hi = max(gaps)
                best = None
                for a in arrowData:
                    if id(a) in picked_ids:
                        continue
                    th = a[1]
                    in_gap = (lo < th < hi) or (lo < th + 2 * math.pi < hi)
                    if in_gap and (best is None or a[0] > best[0]):
                        best = a
                if best is None:
                    break
                picked.append(best)
                picked_ids.add(id(best))
            if picked:
                picked.sort(key=lambda x: x[1])
                return [(a[3], a[4], a[1]) for a in picked]
            # all slices empty → drop into default fan below

        # ----- Step 4: explore-biased seed-and-walk -------------------------
        out: list[list] = []
        picked_thetas: set = set()
        if explore_bias > 0:
            unexplored = [a for a in arrowData if a[2]]
            if unexplored:
                # Seed = longest unexplored.
                seed_action = max(unexplored, key=lambda x: x[0])
                seed_theta = seed_action[1]
                seed_idx = unexplored.index(seed_action)
                out.append(seed_action)
                picked_thetas.add(seed_theta)

                # Walk RIGHT (higher theta) from seed.
                last_theta = seed_theta
                for i in range(seed_idx + 1, len(unexplored)):
                    if unexplored[i][1] - last_theta > min_separation_rad * 0.9:
                        out.append(unexplored[i])
                        picked_thetas.add(unexplored[i][1])
                        last_theta = unexplored[i][1]

                # Walk LEFT (lower theta) from seed.
                last_theta = seed_theta
                for i in range(seed_idx - 1, -1, -1):
                    if last_theta - unexplored[i][1] > min_separation_rad * 0.9:
                        out.append(unexplored[i])
                        picked_thetas.add(unexplored[i][1])
                        last_theta = unexplored[i][1]

                # Add EXPLORED actions with wider spacing (explore_bias × min_sep).
                for a in arrowData:
                    if a[1] in picked_thetas:
                        continue
                    if not picked_thetas:
                        gap_ok = True
                    else:
                        gap_ok = min(abs(a[1] - t) for t in picked_thetas) > \
                                 min_separation_rad * explore_bias
                    if gap_ok:
                        out.append(a)
                        picked_thetas.add(a[1])

        # ----- Step 5: pure-spacing fallback if seed-walk produced nothing ---
        if not out:
            seed_action = max(arrowData, key=lambda x: x[0])
            seed_theta = seed_action[1]
            seed_idx = arrowData.index(seed_action)
            out.append(seed_action)
            picked_thetas.add(seed_theta)
            last_theta = seed_theta
            for i in range(seed_idx + 1, len(arrowData)):
                if arrowData[i][1] - last_theta > min_separation_rad:
                    out.append(arrowData[i])
                    picked_thetas.add(arrowData[i][1])
                    last_theta = arrowData[i][1]
            last_theta = seed_theta
            for i in range(seed_idx - 1, -1, -1):
                if last_theta - arrowData[i][1] > min_separation_rad:
                    out.append(arrowData[i])
                    picked_thetas.add(arrowData[i][1])
                    last_theta = arrowData[i][1]

        # ----- Step 6: still empty → default fan ----------------------------
        if not out:
            return self._default_fan(cur_x, cur_y, cur_yaw,
                                     front_half_cone_rad, max_step_m)

        # ----- Sort by theta, cap at K, build output ------------------------
        out.sort(key=lambda x: x[1])
        if len(out) > K:
            # Down-sample evenly across theta-sorted list to keep visual spread.
            idxs = [i * (len(out) - 1) // (K - 1) for i in range(K)]
            out = [out[i] for i in idxs]
        return [(a[3], a[4], a[1]) for a in out]

    def _default_fan(self, cur_x: float, cur_y: float, cur_yaw: float,
                     half_cone_rad: float, mag: float
                     ) -> list[tuple[float, float, float]]:
        """4 evenly-fanned default actions when no real frontier candidate.
        Mirrors VLMnav `_get_default_arrows`. Uses the front cone ±70%."""
        angle = half_cone_rad * 0.7
        thetas = [-angle, -angle / 4, angle / 4, angle]
        out = []
        for theta in thetas:
            cx = cur_x + mag * math.cos(cur_yaw + theta)
            cy = cur_y + mag * math.sin(cur_yaw + theta)
            out.append((cx, cy, float(theta)))
        return out

    # ---------------- Observation export ----------------
    def save_frame(self, rgb: np.ndarray, path: Path):
        from PIL import Image
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(str(path), format="PNG", optimize=True)

    def get_pose(self) -> dict:
        return {
            "position": self._current_position.tolist(),
            "look_dir": self._look_dir.tolist(),
            "yaw": math.atan2(self._look_dir[1], self._look_dir[0]),
        }

    # ---------------- Lifecycle ----------------
    def close(self):
        if self._app is not None:
            self._app.close()
            self._app = None
