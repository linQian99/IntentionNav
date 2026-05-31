"""Phase 1: Capture support surface photos using Isaac Sim.

Uses goodnav-style camera (height=1.5m, focal_length=10, aperture=20, 1024x1024)
to photograph each identified support surface from human viewpoints.
Also generates overhead maps with camera/surface positions for review.

Must run in goodnav conda env with Isaac Sim sourced:
  conda activate goodnav
  source $ISAACSIM_ROOT/setup_conda_env.sh
"""

# --- Parse CLI before Isaac Sim init (SimulationApp consumes sys.argv) ---
import argparse
import sys

parser = argparse.ArgumentParser(description="Phase 1: Capture surface photos with Isaac Sim")
parser.add_argument("--usd-root", required=True, help="Root dir with USD scenes (TataServices/)")
parser.add_argument("--metaroot", required=True, help="Root dir with metadata (freemap.npy, room_region.json)")
parser.add_argument("--surface-targets-dir", required=True, help="Dir with surface_targets.json per scene")
parser.add_argument("--scene-summary", default=None, help="kujiale_scene_summary dir; enables scene-graph enrichment of the manifest")
parser.add_argument("--output-dir", required=True, help="Output dir for photos")
parser.add_argument("--scene", default=None, help="Single scene ID (ignores --scene-list)")
parser.add_argument("--scene-list", default=None, help="Text file with one scene_id per line (for batch)")
parser.add_argument("--resolution", type=int, default=1024, help="Surface photo resolution (default 1024)")
parser.add_argument("--num-views", type=int, default=1, help="Number of viewpoints per surface")
parser.add_argument("--camera-distance", type=float, default=1.5, help="Camera distance from surface (meters)")
parser.add_argument("--camera-height", type=float, default=1.5, help="Camera height (meters)")
parser.add_argument("--batch-size", type=int, default=20, help="Scenes per process before restart (default 20)")
parser.add_argument("--force", action="store_true", help="Re-capture scenes even if manifest exists")
parser.add_argument("--save-miss-photos", action="store_true",
    help="Save rejected viewpoint photos to miss_photos/ for debugging. "
         "Default off: misses are only summarised in progress.log / miss_debug.json.")
args, unknown_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown_args

# Exit codes used by the batch shell wrappers.
EXIT_ALL_DONE = 10      # No more pending scenes
EXIT_BATCH_DONE = 11    # Batch limit reached, shell should restart
EXIT_ERROR = 1

# --- Isaac Sim imports ---
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera
from omni.isaac.core.prims import XFormPrim
import numpy as np
from PIL import Image
import os
import json
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from intentionnav.data_collection.geometry import DEFAULT_CAMERA_FORWARD, rot3_from_o_to_ab
import copy as _copy


O = DEFAULT_CAMERA_FORWARD

# ---- Camera parameters matching goodnav demo.py ----
CAMERA_HEIGHT = args.camera_height
FOCAL_LENGTH = 10.0
APERTURE = 20.0  # horizontal and vertical
CLIPPING_NEAR = 0.01
CLIPPING_FAR = 1000.0
IMAGE_SIZE = args.resolution
NUM_VIEWS = args.num_views
CAMERA_DISTANCE = args.camera_distance



def _project_bbox_to_2d_rect(camera_pos, aim_pos, bbox_min, bbox_max, image_size):
    """Project 3D bbox to 2D image rectangle (x_min, y_min, x_max, y_max).
    Returns None if bbox is behind camera."""
    cam = np.array(camera_pos, dtype=np.float64)
    aim = np.array(aim_pos, dtype=np.float64)
    direction = np.array([aim[0]-cam[0], aim[1]-cam[1], 0.0])
    dn = np.linalg.norm(direction)
    if dn < 1e-6: return None
    fwd = direction / dn
    right = np.cross(np.array([0,0,1.0]), fwd)
    rn = np.linalg.norm(right)
    if rn < 1e-6: return None
    right = right / rn
    up = np.cross(fwd, right)
    R_inv = np.column_stack((fwd, right, up)).T

    bmin = np.array(bbox_min, dtype=np.float64)
    bmax = np.array(bbox_max, dtype=np.float64)
    corners = np.array([
        [bmin[0],bmin[1],bmin[2]], [bmin[0],bmin[1],bmax[2]],
        [bmin[0],bmax[1],bmin[2]], [bmin[0],bmax[1],bmax[2]],
        [bmax[0],bmin[1],bmin[2]], [bmax[0],bmin[1],bmax[2]],
        [bmax[0],bmax[1],bmin[2]], [bmax[0],bmax[1],bmax[2]],
    ])
    local = (R_inv @ (corners - cam).T).T
    in_front = local[:, 0] > 0.01
    if not np.any(in_front): return None

    half = image_size / 2.0
    pxs, pys = [], []
    for i in range(8):
        if not in_front[i]: continue
        px = local[i,1] / local[i,0] * half + half
        py = -local[i,2] / local[i,0] * half + half
        pxs.append(px); pys.append(py)
    if not pxs: return None
    return (min(pxs), min(pys), max(pxs), max(pys))


def compute_occlusion_ratio(camera_pos, aim_pos, target_bbox_min, target_bbox_max,
                             scene_objects, image_size=1024):
    """Compute what fraction of the target's 2D projection is blocked by
    other objects that sit between camera and target.

    scene_objects: list of dicts with 'position', 'min_points', 'max_points'.
    Returns occlusion ratio in [0, 1]: 0 = fully visible, 1 = fully blocked.
    """
    cam = np.array(camera_pos[:2], dtype=np.float64)
    tgt_center = np.array([
        (target_bbox_min[0]+target_bbox_max[0])/2,
        (target_bbox_min[1]+target_bbox_max[1])/2,
    ])
    dist_to_target = float(np.linalg.norm(tgt_center - cam))

    # Project target to 2D rect
    t_rect = _project_bbox_to_2d_rect(
        camera_pos, aim_pos, target_bbox_min, target_bbox_max, image_size
    )
    if t_rect is None: return 1.0
    # Clip to image bounds
    tx0 = max(0, t_rect[0]); ty0 = max(0, t_rect[1])
    tx1 = min(image_size, t_rect[2]); ty1 = min(image_size, t_rect[3])
    t_area = max(0, tx1-tx0) * max(0, ty1-ty0)
    if t_area < 1: return 1.0

    # For each scene object between camera and target, compute overlap
    total_occluded = 0.0
    for obj in scene_objects:
        pos = obj.get("position")
        bmin = obj.get("min_points")
        bmax = obj.get("max_points")
        if not pos or not bmin or not bmax: continue
        if len(bmin) < 3 or len(bmax) < 3: continue
        # Skip if object is farther than target (behind target)
        obj_center = np.array(pos[:2], dtype=np.float64)
        dist_to_obj = float(np.linalg.norm(obj_center - cam))
        if dist_to_obj >= dist_to_target * 0.95: continue
        # Skip if object is behind camera
        to_obj = obj_center - cam
        to_tgt = tgt_center - cam
        if float(np.dot(to_obj, to_tgt)) < 0: continue
        # Project object to 2D rect
        o_rect = _project_bbox_to_2d_rect(
            camera_pos, aim_pos, bmin, bmax, image_size
        )
        if o_rect is None: continue
        # Rectangle intersection with target
        ix0 = max(tx0, o_rect[0]); iy0 = max(ty0, o_rect[1])
        ix1 = min(tx1, o_rect[2]); iy1 = min(ty1, o_rect[3])
        overlap = max(0, ix1-ix0) * max(0, iy1-iy0)
        total_occluded += overlap

    return min(1.0, total_occluded / t_area)



def get_camera_orientation_horizontal(camera_pos, target_pos):
    """Compute camera orientation with pure horizontal gaze (goodnav demo style).

    Agent camera only rotates around Z-axis (yaw), never pitches up/down.
    Matches demo.py FloatingCameraController which applies only z-euler rotations.
    """
    from scipy.spatial.transform import Rotation as R

    # Horizontal direction only (project to XY plane)
    direction = np.array([
        target_pos[0] - camera_pos[0],
        target_pos[1] - camera_pos[1],
        0.0,
    ], dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([1, 0, 0], dtype=np.float64)
    else:
        direction = direction / norm

    world_up = np.array([0, 0, 1], dtype=np.float64)
    forward = direction
    right = np.cross(world_up, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        right = np.array([1, 0, 0], dtype=np.float64)
    else:
        right = right / right_norm
    up = np.cross(forward, right)

    rot_mat = np.column_stack((forward, right, up))
    quat = R.from_matrix(rot_mat).as_quat()  # xyzw
    # Isaac Sim wants wxyz
    return np.array([quat[3], quat[0], quat[1], quat[2]])


def build_fat_map(freemap, dilate_iters=0):
    """Build occupancy map for walkability check.

    With dilate_iters=0 we use the raw freemap (no safety margin).
    Camera may end up close to walls but viewpoint coverage is maximized.
    """
    if freemap is None:
        return None, None
    raw = _copy.copy(freemap[1:, 1:])
    raw[raw == 2] = 0
    if dilate_iters > 0:
        raw = 1 - cv2.dilate(
            (1 - raw).astype(np.uint8),
            np.ones([3, 3], dtype=np.uint8),
            iterations=dilate_iters,
        )
    return raw, freemap


# Viewpoint sampling: face-arc with wider/multi-radius fallback
def _has_walkable_within(target_pos, freemap, radius_m=1.0):
    """Pre-filter: target must have a walkable cell within radius_m AND that
    walkable cell must NOT be separated from target by a wall (value=2).

    Rejects wall-adjacent targets where walkable exists on the other side of
    the wall (agent can't cross walls).
    """
    if freemap is None:
        return True
    x_coords = freemap[0, 1:]
    y_coords = freemap[1:, 0]
    grid = freemap[1:, 1:]
    cy = np.argmin(np.abs(y_coords - target_pos[1]))
    cx = np.argmin(np.abs(x_coords - target_pos[0]))
    cell_size = float(abs(x_coords[1] - x_coords[0]))
    r = max(1, int(radius_m / cell_size))
    H, W = grid.shape
    r0, r1 = max(0, cy - r), min(H, cy + r + 1)
    c0, c1 = max(0, cx - r), min(W, cx + r + 1)

    walkables = np.argwhere(grid[r0:r1, c0:c1] == 1)
    for (dr, dc) in walkables:
        wy = r0 + dr
        wx = c0 + dc
        dist_cells = max(abs(cy - wy), abs(cx - wx))
        n_samples = max(10, int(dist_cells * 2))
        wall_hit = False
        for t in np.linspace(0.0, 1.0, n_samples):
            sy = wy + t * (cy - wy)
            sx = wx + t * (cx - wx)
            si, sj = int(round(sy)), int(round(sx))
            if 0 <= si < H and 0 <= sj < W and grid[si, sj] == 2:
                # Skip wall cells adjacent to target (rounding artifact)
                if abs(si - cy) <= 1 and abs(sj - cx) <= 1:
                    continue
                wall_hit = True
                break
        if not wall_hit:
            return True
    return False


def compute_viewpoints(target_pos, height, freemap=None, radii=None, walk_radius=None):
    """Generate candidate viewpoints around target via angular sampling.

    Returns (viewpoints, gate_reason) where gate_reason is None on success,
    or one of: "no_freemap" | "no_walkable_nearby" | "no_los_viewpoint".
    Viewpoints are unordered — caller ranks by occlusion score.
    """
    if freemap is None:
        return [], "no_freemap"
    wr = walk_radius if walk_radius is not None else 1.0
    if not _has_walkable_within(target_pos, freemap, radius_m=wr):
        return [], "no_walkable_nearby"

    fat_map, occ_map = build_fat_map(freemap)

    kwargs = {}
    if radii is not None:
        kwargs["radii"] = radii
    vps = _angular_viewpoints(
        target_pos, occ_map, fat_map, height, **kwargs
    )
    if not vps:
        return [], "no_los_viewpoint"
    return vps, None


def _angular_viewpoints(target_pos, occ_map, fat_map, height,
                          num_angles=24, radii=(1.5, 1.2, 1.0, 0.8)):
    """Generate candidate viewpoints around target.

    For each of num_angles equal sectors, try radii near→far and take the
    first walkable+LOS cell. Returns unordered list — caller ranks by
    occlusion ratio.
    """
    x_coords = occ_map[0, 1:]
    y_coords = occ_map[1:, 0]
    grid = occ_map[1:, 1:]
    H, W = grid.shape
    cell = float(abs(x_coords[1] - x_coords[0]))

    cy = int(np.argmin(np.abs(y_coords - target_pos[1])))
    cx = int(np.argmin(np.abs(x_coords - target_pos[0])))

    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

    def los_clear(wy, wx):
        dist_cells = max(abs(cy - wy), abs(cx - wx))
        n = max(12, int(dist_cells * 2))
        for t in np.linspace(0.0, 1.0, n):
            sy = wy + t * (cy - wy)
            sx = wx + t * (cx - wx)
            si, sj = int(round(sy)), int(round(sx))
            if not (0 <= si < H and 0 <= sj < W):
                continue
            if grid[si, sj] != 2:
                continue
            if abs(si - cy) <= 1 and abs(sj - cx) <= 1:
                continue
            return False
        return True

    result = []
    radii_iter = sorted(radii)
    for angle in angles:
        for r in radii_iter:
            r_cells = r / cell
            wy = int(round(cy + r_cells * np.sin(angle)))
            wx = int(round(cx + r_cells * np.cos(angle)))
            if not (0 <= wy < H and 0 <= wx < W):
                continue
            if fat_map[wy, wx] != 1:
                continue
            if not los_clear(wy, wx):
                continue
            world_x = float(x_coords[wx])
            world_y = float(y_coords[wy])
            result.append(np.array([world_x, world_y, height]))
            break

    return result


def load_freemap(metaroot, scene_id):
    """Load freemap and metadata for walkability checks."""
    freemap_path = os.path.join(metaroot, scene_id, "freemap.npy")
    region_path = os.path.join(metaroot, scene_id, "room_region.json")

    if not os.path.exists(freemap_path):
        return None, None

    freemap = np.load(freemap_path)

    meta = None
    if os.path.exists(region_path):
        with open(region_path, "r") as f:
            meta = json.load(f)

    return freemap, meta


def has_line_of_sight(cam_x, cam_y, target_x, target_y, freemap, n_samples=30,
                      tolerance=0.15):
    """2D line-of-sight check on freemap.

    Tolerance controls how much of the line can cross non-walkable cells.
    0.15 = up to 15% (only edge cells near surface). Above this = camera sees
    walls/furniture in front instead of the actual surface.
    """
    if freemap is None:
        return True

    x_coords = freemap[0, 1:]
    y_coords = freemap[1:, 0]
    grid = freemap[1:, 1:]

    if len(x_coords) == 0 or len(y_coords) == 0:
        return True

    # Sample along line, skip both ends (surface/camera edges hit obstacle grid cells)
    ts = np.linspace(0.15, 0.75, n_samples)
    blocked = 0
    checked = 0
    for t in ts:
        x = cam_x + t * (target_x - cam_x)
        y = cam_y + t * (target_y - cam_y)
        xi = np.argmin(np.abs(x_coords - x))
        yi = np.argmin(np.abs(y_coords - y))
        if 0 <= yi < grid.shape[0] and 0 <= xi < grid.shape[1]:
            checked += 1
            if grid[yi, xi] != 1:
                blocked += 1

    if checked == 0:
        return True
    return (blocked / checked) <= tolerance


def generate_scene_overview_map(freemap, surfaces_with_viewpoints, save_path,
                                 scene_id, missed_targets=None):
    """Generate overview map: numbered surfaces + cameras + side legend.

    Captured targets shown as coloured numbered circles with camera arrows.
    Missed targets shown as red X markers with category label so you can
    see exactly where on the map failed items sit.
    """
    if freemap is None:
        return

    x_coords = freemap[0, 1:]
    y_coords = freemap[1:, 0]
    grid = freemap[1:, 1:]

    n = len(surfaces_with_viewpoints)
    n_missed = len(missed_targets) if missed_targets else 0

    # Two-panel: map on left, legend on right
    fig, (ax, lax) = plt.subplots(1, 2, figsize=(16, 12),
                                    gridspec_kw={"width_ratios": [3, 1]})
    ax.imshow(grid, cmap="gray_r", origin="lower",
              extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])

    import matplotlib.cm as cm
    colors = cm.get_cmap("tab20", max(n, 1))

    legend_lines = []

    for idx, s in enumerate(surfaces_with_viewpoints):
        num = idx + 1
        color = colors(idx % 20)
        sid = s["surface_id"]
        sp = s["surface_pos"]
        obj_cats = s.get("objects_cats", [])
        vps = s["viewpoints"]

        ax.plot(sp[0], sp[1], "o", markersize=22,
                color=color, markeredgecolor="black", markeredgewidth=1.5,
                zorder=3)
        ax.text(sp[0], sp[1], str(num), fontsize=11, fontweight="bold",
                ha="center", va="center", color="white", zorder=4)

        for vi, vp in enumerate(vps):
            ax.plot(vp[0], vp[1], "o", markersize=5, color=color,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=2)
            ddx = sp[0] - vp[0]
            ddy = sp[1] - vp[1]
            norm = np.sqrt(ddx**2 + ddy**2)
            if norm > 0:
                ax.arrow(vp[0], vp[1], ddx * 0.6, ddy * 0.6,
                         head_width=0.1, head_length=0.08,
                         fc=color, ec=color, alpha=0.7, linewidth=1.2, zorder=2)

        surface_short = sid.split("/")[0]
        objs_str = ",".join(obj_cats[:3]) if obj_cats else "-"
        legend_lines.append((num, color, f"{surface_short}", f"[{objs_str}]"))

    # --- Missed targets: red X markers ---
    miss_legend = []
    if missed_targets:
        for mt in missed_targets:
            pos = mt.get("position")
            if not pos or len(pos) < 2:
                continue
            ax.plot(pos[0], pos[1], "x", markersize=14, color="red",
                    markeredgewidth=2.5, zorder=5)
            label = f"{mt.get('category', '?')}({mt.get('reason', '?')[:8]})"
            ax.text(pos[0] + 0.15, pos[1] + 0.15, label,
                    fontsize=7, color="red", zorder=5)
            miss_legend.append(mt)

    title = f"{scene_id} ({n} captured"
    if n_missed:
        title += f", {n_missed} missed"
    title += ")"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")

    # Right panel: numbered legend
    total_legend = n + n_missed
    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.axis("off")
    line_height = 1.0 / (total_legend + 3)
    lax.text(0.5, 1.0 - line_height * 0.4,
             f"{n} captured, {n_missed} missed",
             fontsize=12, fontweight="bold", ha="center",
             transform=lax.transAxes)
    for i, (num, color, surf, objs) in enumerate(legend_lines):
        y = 1.0 - line_height * (i + 1.5)
        lax.scatter([0.1], [y], s=400, c=[color],
                    edgecolors="black", linewidths=1.0,
                    transform=lax.transAxes, zorder=2)
        lax.text(0.1, y, str(num), fontsize=10, fontweight="bold",
                 ha="center", va="center", color="white",
                 transform=lax.transAxes, zorder=3)
        lax.text(0.22, y, f"{surf}", fontsize=9, fontweight="bold",
                 va="center", ha="left", transform=lax.transAxes)
        lax.text(0.22, y - line_height * 0.35, objs, fontsize=8,
                 va="center", ha="left", color="#444",
                 transform=lax.transAxes)
    # Missed entries in legend (red)
    for j, mt in enumerate(miss_legend):
        y = 1.0 - line_height * (n + j + 1.5)
        lax.scatter([0.1], [y], s=200, marker="x", c=["red"],
                    linewidths=2.0, transform=lax.transAxes, zorder=2)
        lax.text(0.22, y, mt.get("category", "?"), fontsize=9,
                 fontweight="bold", va="center", ha="left", color="red",
                 transform=lax.transAxes)
        lax.text(0.22, y - line_height * 0.35,
                 mt.get("reason", "")[:20], fontsize=8,
                 va="center", ha="left", color="#c44",
                 transform=lax.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def _detect_building_bbox_in_render(td_arr):
    """Bounding box of building pixels (non-blue-floor) in rotated topdown.
    Returns (row_min, row_max, col_min, col_max) or None if detection fails.
    """
    try:
        from scipy import ndimage
    except ImportError:
        return None
    if td_arr.shape[2] < 3:
        return None
    r, g, b = td_arr[:, :, 0], td_arr[:, :, 1], td_arr[:, :, 2]
    is_building = ~((b > r + 0.1) & (b > 0.65))
    mask = ndimage.binary_opening(is_building, iterations=3)
    labeled, nf = ndimage.label(mask)
    if nf == 0:
        return None
    sizes = ndimage.sum(mask, labeled, range(1, nf + 1))
    largest = int(np.argmax(sizes)) + 1
    clean = labeled == largest
    rows, cols = np.where(clean)
    if len(rows) == 0:
        return None
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def generate_annotated_topdown_map(freemap, surfaces_with_viewpoints, save_path,
                                      scene_id, topdown_path):
    """Annotated topdown: same overlays as scene_overview, but uses the raw 3D
    top-down render as background instead of the freemap.

    Pixel→world alignment: capture_topdown_render places the camera at
    (cx+0.1, cy, altitude) looking at (cx, cy, 0) with 90° FOV. The resulting
    image, rotated 90° counter-clockwise, aligns with world XY under
    matplotlib origin='upper'. Half-extent at altitude A is A meters.
    """
    if freemap is None or not os.path.exists(topdown_path):
        return
    x_coords = freemap[0, 1:]
    y_coords = freemap[1:, 0]
    n = len(surfaces_with_viewpoints)
    if n == 0:
        return

    td_img = plt.imread(topdown_path)
    cx = float((x_coords[0] + x_coords[-1]) / 2)
    cy = float((y_coords[0] + y_coords[-1]) / 2)
    size_x = float(abs(x_coords[-1] - x_coords[0]))
    size_y = float(abs(y_coords[-1] - y_coords[0]))
    scene_size = max(size_x, size_y)
    altitude = max(8.0, scene_size * 0.7)
    half = altitude  # world half-extent covered by raw render

    # After rot90(k=1, CCW) + origin='upper', pixel (0,0) ↔ (cx-half, cy+half).
    # Verified via freemap wall overlay test — this orientation is correct.
    td_arr = np.rot90(td_img, k=1)
    H, W = td_arr.shape[:2]

    # Crop to detected building bbox so outermost walls touch axis edges.
    # Falls back to freemap bounds if color detection fails.
    bbox = _detect_building_bbox_in_render(td_arr)
    if bbox is not None:
        r0, r1, c0, c1 = bbox
        r1 += 1
        c1 += 1
    else:
        x_min_w, x_max_w = float(min(x_coords[0], x_coords[-1])), float(max(x_coords[0], x_coords[-1]))
        y_min_w, y_max_w = float(min(y_coords[0], y_coords[-1])), float(max(y_coords[0], y_coords[-1]))
        r0 = max(0, int(round(((cy + half) - y_max_w) / (2 * half) * H)))
        r1 = min(H, int(round(((cy + half) - y_min_w) / (2 * half) * H)))
        c0 = max(0, int(round((x_min_w - (cx - half)) / (2 * half) * W)))
        c1 = min(W, int(round((x_max_w - (cx - half)) / (2 * half) * W)))
    td_crop = td_arr[r0:r1, c0:c1]
    ext_left = (cx - half) + (c0 / W) * 2 * half
    ext_right = (cx - half) + (c1 / W) * 2 * half
    ext_top = (cy + half) - (r0 / H) * 2 * half
    ext_bottom = (cy + half) - (r1 / H) * 2 * half

    fig, (ax, lax) = plt.subplots(1, 2, figsize=(16, 12),
                                    gridspec_kw={"width_ratios": [3, 1]})
    ax.imshow(td_crop,
              extent=[ext_left, ext_right, ext_bottom, ext_top],
              origin="upper", interpolation="bilinear")
    ax.set_xlim(ext_right, ext_left)  # invert X to match freemap convention
    ax.set_ylim(ext_bottom, ext_top)

    import matplotlib.cm as cm
    colors = cm.get_cmap("tab20", max(n, 1))

    legend_lines = []
    for idx, s in enumerate(surfaces_with_viewpoints):
        num = idx + 1
        color = colors(idx % 20)
        sid = s["surface_id"]
        sp = s["surface_pos"]
        obj_cats = s.get("objects_cats", [])
        vps = s["viewpoints"]

        ax.plot(sp[0], sp[1], "o", markersize=22,
                color=color, markeredgecolor="black", markeredgewidth=1.5,
                zorder=3)
        ax.text(sp[0], sp[1], str(num), fontsize=11, fontweight="bold",
                ha="center", va="center", color="white", zorder=4)

        for vp in vps:
            ax.plot(vp[0], vp[1], "o", markersize=5, color=color,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=2)
            dx = sp[0] - vp[0]
            dy = sp[1] - vp[1]
            norm = (dx * dx + dy * dy) ** 0.5
            if norm > 0:
                ax.arrow(vp[0], vp[1], dx * 0.6, dy * 0.6,
                         head_width=0.1, head_length=0.08,
                         fc=color, ec=color, alpha=0.7, linewidth=1.2, zorder=2)

        surface_short = sid.split("/")[0]
        objs_str = ",".join(obj_cats[:3]) if obj_cats else "-"
        legend_lines.append((num, color, surface_short, f"[{objs_str}]"))

    ax.set_title(f"{scene_id} ({n} surfaces)", fontsize=14)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")

    lax.set_xlim(0, 1); lax.set_ylim(0, 1); lax.axis("off")
    line_height = 1.0 / (n + 2)
    lax.text(0.5, 1.0 - line_height * 0.4, f"{n} surfaces",
             fontsize=12, fontweight="bold", ha="center", transform=lax.transAxes)
    for i, (num, color, surf, objs) in enumerate(legend_lines):
        y = 1.0 - line_height * (i + 1.5)
        lax.scatter([0.1], [y], s=400, c=[color],
                    edgecolors="black", linewidths=1.0,
                    transform=lax.transAxes, zorder=2)
        lax.text(0.1, y, str(num), fontsize=10, fontweight="bold",
                 ha="center", va="center", color="white",
                 transform=lax.transAxes, zorder=3)
        lax.text(0.22, y, f"{surf}", fontsize=9, fontweight="bold",
                 va="center", ha="left", transform=lax.transAxes)
        lax.text(0.22, y - line_height * 0.35, objs, fontsize=8,
                 va="center", ha="left", color="#444", transform=lax.transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def toggle_ceilings_visibility(visible: bool):
    """Hide (or show) all ceiling meshes in the current stage.

    Walks the stage under /World/SceneAsset, finds prims whose name starts with
    'ceiling' or 'celling' (Kujiale typo variant), and toggles visibility.
    Returns list of prim paths modified.
    """
    from pxr import UsdGeom
    stage = simulation_app.context.get_stage()
    modified = []
    root = stage.GetPrimAtPath("/World/SceneAsset")
    if not root or not root.IsValid():
        return modified

    for prim in stage.TraverseAll():
        if not prim.IsValid():
            continue
        name = prim.GetName().lower()
        # Ceilings and ceiling lights / chandeliers
        if name.startswith("ceiling") or name.startswith("celling") or name.startswith("chandelier"):
            # Only toggle top-level ceiling prim, not nested children
            parent = prim.GetParent()
            if parent and parent.GetName().lower() in ("ceiling", "celling"):
                continue
            imageable = UsdGeom.Imageable(prim)
            if imageable:
                if visible:
                    imageable.MakeVisible()
                else:
                    imageable.MakeInvisible()
                modified.append(str(prim.GetPath()))
    return modified


TOPDOWN_RESOLUTION = 1536  # Hi-res render for scene_topdown.png (stretched to
                            # overview figures, so 512 would look blurry).


def capture_topdown_render(camera, world, freemap, save_path, altitude=None):
    """Capture a top-down render using two-step yaw-then-pitch rotation.

    Temporarily hides ceiling meshes so the room interiors are visible from above,
    and bumps the camera resolution to TOPDOWN_RESOLUTION so the rendered image
    stays sharp when displayed at overview figure size.
    """
    if freemap is None:
        return False

    try:
        from pyquaternion import Quaternion

        x_coords = freemap[0, 1:]
        y_coords = freemap[1:, 0]
        cx = float((x_coords[0] + x_coords[-1]) / 2)
        cy = float((y_coords[0] + y_coords[-1]) / 2)
        size_x = float(x_coords[-1] - x_coords[0])
        size_y = float(y_coords[-1] - y_coords[0])
        scene_size = max(size_x, size_y)

        if altitude is None:
            altitude = max(8.0, scene_size * 0.7)

        # Place camera slightly offset from center so yaw is well-defined (avoids gimbal)
        cam_pos = np.array([cx + 0.1, cy, altitude])
        target = np.array([cx, cy, 0.0])

        # Step 1: horizontal yaw to point camera in XY direction toward target
        horizontal_target = np.array([target[0], target[1], cam_pos[2]])
        horizontal_rot = rot3_from_o_to_ab(O, cam_pos, horizontal_target)
        horizontal_quat = Quaternion(matrix=horizontal_rot)

        # Step 2: pitch down to look at ground
        vec = target - cam_pos
        h_dist = float(np.linalg.norm([vec[0], vec[1]]))
        v_dist = float(vec[2])
        pitch_rad = -np.arctan2(v_dist, h_dist)  # negative because looking down

        pitch_axis = horizontal_rot @ np.array([0, 1, 0])
        pitch_quat = Quaternion(axis=pitch_axis, angle=pitch_rad)

        final = pitch_quat * horizontal_quat
        orientation_wxyz = np.array([final.w, final.x, final.y, final.z])

        print(f"  topdown: altitude={altitude:.1f}m pitch={np.degrees(pitch_rad):.1f}° res={TOPDOWN_RESOLUTION}")

        # Hide ceilings for top-down visibility
        hidden = toggle_ceilings_visibility(visible=False)
        print(f"  hid {len(hidden)} ceiling prims")

        # Temporarily raise camera resolution for sharp topdown render
        try:
            camera.set_resolution((TOPDOWN_RESOLUTION, TOPDOWN_RESOLUTION))
        except Exception as e:
            print(f"  [WARN] couldn't raise resolution for topdown: {e}")

        try:
            camera.set_world_pose(position=cam_pos, orientation=orientation_wxyz)
            for _ in range(40):
                world.step(render=True)

            rgba = camera.get_rgba()
            if rgba is not None and rgba.shape[0] > 0:
                if "torch" in str(type(rgba)):
                    rgba = rgba.cpu().numpy()
                image = Image.fromarray(np.clip(rgba, 0, 255).astype(np.uint8), "RGBA")
                image.save(save_path)
                result = True
            else:
                result = False
        finally:
            # Restore original camera resolution + ceilings
            try:
                camera.set_resolution((IMAGE_SIZE, IMAGE_SIZE))
                for _ in range(3):
                    world.step(render=True)
            except Exception:
                pass
            toggle_ceilings_visibility(visible=True)
        return result
    except Exception as e:
        print(f"[WARN] Top-down render failed: {e}")
        import traceback
        traceback.print_exc()
    return False


def capture_photo(camera, world, viewpoint_pos, target_pos, save_path):
    """Capture a photo from viewpoint_pos aimed at target_pos.
    Returns success (bool)."""
    orientation_wxyz = get_camera_orientation_horizontal(viewpoint_pos, target_pos)
    camera.set_world_pose(position=viewpoint_pos, orientation=orientation_wxyz)

    # More iterations = better RTX denoising / DLSS convergence → sharper image
    for _ in range(40):
        world.step(render=True)

    rgba_data = camera.get_rgba()
    if rgba_data is None or rgba_data.shape[0] == 0:
        return False

    if "torch" in str(type(rgba_data)):
        rgba_data = rgba_data.cpu().numpy()
    image_data = np.clip(rgba_data, 0, 255).astype(np.uint8)
    Image.fromarray(image_data, "RGBA").save(save_path)

    return True


# Prim path constants (reused across all scenes in a single Isaac Sim session)
ASSET_PRIM_PATH = "/World/SceneAsset"
CAMERA_PRIM_PATH = "/World/IntentionNavCamera"


_STRUCTURAL_CATS = {
    "wall", "floor", "ceiling", "door", "doorsill",
    "window", "curtain", "entrance", "ceiling_light",
    "unknown",
}


def _extract_cat(object_id):
    head = object_id.split("/")[0]
    parts = head.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return head


def enrich_manifest_inplace(manifest, scene_summary_dir, scene_id):
    """Attach target_details (pos/bbox/room/nearby_objects per target_id) and
    room_objects (non-structural categories in the same room) to each entry,
    so the manifest is self-contained and downstream Gemini prompts don't
    need another pass over the scene graph."""
    if not scene_summary_dir:
        return
    obj_path = os.path.join(scene_summary_dir, scene_id, "object_dict.json")
    room_path = os.path.join(scene_summary_dir, scene_id, "room_dict.json")
    if not (os.path.exists(obj_path) and os.path.exists(room_path)):
        return
    with open(obj_path, "r", encoding="utf-8") as f:
        object_dict = json.load(f)
    with open(room_path, "r", encoding="utf-8") as f:
        room_dict = json.load(f)

    for entry in manifest:
        target_ids = entry.get("all_targets_in_category", [])
        details = []
        for oid in target_ids:
            od = object_dict.get(oid)
            if not od:
                details.append({"object_id": oid, "missing": True})
                continue
            nearby = {}
            for nid, rel in (od.get("nearby_objects") or {}).items():
                if isinstance(rel, list) and len(rel) >= 2:
                    relation, dist = rel[0], rel[1]
                else:
                    relation, dist = str(rel), None
                nearby[nid] = {
                    "category": _extract_cat(nid),
                    "relation": relation,
                    "distance": dist,
                }
            details.append({
                "object_id": oid,
                "category": _extract_cat(oid),
                "room": od.get("room"),
                "position": od.get("position"),
                "bbox": {"min": od.get("min_points"), "max": od.get("max_points")},
                "nearby_objects": nearby,
            })
        entry["target_details"] = details

        room = entry.get("room")
        exclude = set(target_ids)
        room_objs = []
        seen_cats = set()
        for oid in room_dict.get(room, []):
            if oid in exclude:
                continue
            c = _extract_cat(oid)
            if c in _STRUCTURAL_CATS or c in seen_cats:
                continue
            seen_cats.add(c)
            room_objs.append(c)
        entry["room_objects"] = room_objs


def clear_scene_asset():
    """Remove previously loaded USD asset prim, preparing for next scene."""
    from pxr import Sdf
    stage = simulation_app.context.get_stage()
    prim = stage.GetPrimAtPath(ASSET_PRIM_PATH)
    if prim and prim.IsValid():
        stage.RemovePrim(Sdf.Path(ASSET_PRIM_PATH))




def process_scene(scene_id, world, camera):
    """Process a single scene: load USD, capture photos, clear USD. Reuses world/camera."""
    surface_file = os.path.join(args.surface_targets_dir, scene_id, "surface_targets.json")
    if not os.path.exists(surface_file):
        print(f"[SKIP] {scene_id}: no surface_targets.json")
        return None

    with open(surface_file, "r") as f:
        surfaces = json.load(f)

    usd_path = os.path.join(args.usd_root, scene_id, "start_result_navigation.usd")
    if not os.path.exists(usd_path):
        print(f"[SKIP] {scene_id}: USD file not found at {usd_path}")
        return None

    # Output dirs
    photos_dir = os.path.join(args.output_dir, scene_id, "surface_photos")
    overhead_dir = os.path.join(args.output_dir, scene_id, "overhead_maps")
    manifest_path = os.path.join(args.output_dir, scene_id, "capture_manifest.json")

    # Checkpoint: skip if manifest exists
    if os.path.exists(manifest_path) and not args.force:
        try:
            with open(manifest_path, "r") as f:
                existing = json.load(f)
            if existing and len(existing) > 0:
                print(f"[SKIP] {scene_id}: already done ({len(existing)} surfaces)")
                return {"scene_id": scene_id, "skipped": True,
                        "num_surfaces": len(existing),
                        "total_photos": sum(len(s.get("photos", [])) for s in existing),
                        "rejected_photos": 0}
        except Exception:
            pass

    freemap, meta = load_freemap(args.metaroot, scene_id)
    # Load all scene objects for occlusion check (pure geometry, no Isaac Sim)
    scene_objects = []
    if args.scene_summary:
        obj_path = os.path.join(args.scene_summary, scene_id, "object_dict.json")
        if os.path.exists(obj_path):
            with open(obj_path, "r", encoding="utf-8") as f:
                od = json.load(f)
            scene_objects = [
                {"position": v.get("position"),
                 "min_points": v.get("min_points"),
                 "max_points": v.get("max_points")}
                for v in od.values()
                if v.get("position") and v.get("min_points") and v.get("max_points")
            ]
    # Clean stale photos from previous runs (numbering may have changed)
    if os.path.isdir(photos_dir):
        import shutil
        shutil.rmtree(photos_dir)
    os.makedirs(photos_dir, exist_ok=True)
    # overhead_maps per-surface deprecated — scene_overview.png shows all on one map

    # --- Load this scene's USD into the persistent stage ---
    clear_scene_asset()  # Remove previous scene's asset, if any
    asset_prim = add_reference_to_stage(usd_path=usd_path, prim_path=ASSET_PRIM_PATH)
    if not asset_prim or not asset_prim.IsValid():
        print(f"[ERROR] {scene_id}: Failed to load USD")
        return None

    try:
        XFormPrim(prim_path=ASSET_PRIM_PATH, name="scene_asset").set_local_scale(
            np.array([1, 1, 1])
        )
    except Exception:
        pass



    # Reset world + warm up frames to let USD load
    try:
        world.reset()
        for _ in range(10):
            world.step(render=True)
    except Exception as e:
        print(f"[ERROR] {scene_id}: world.reset/step failed: {e}")
        return None

    # --- Capture photos for each surface ---
    manifest = []
    overview_surfaces = []  # Collect for scene overview map
    total_photos = 0
    total_rejected = 0
    miss_debug_entries = []  # per-miss diagnostic records for miss_debug.json

    # Top-down Isaac Sim render (ceilings hidden temporarily)
    topdown_path = os.path.join(args.output_dir, scene_id, "scene_topdown.png")
    if not os.path.exists(topdown_path):
        capture_topdown_render(camera, world, freemap, topdown_path)

    # Sequential surface ID (1-indexed) — same as on scene_overview map
    surface_id_counter = 0

    for surface in surfaces:
        sid = surface["surface_id"]
        surface_pos = np.array(surface["position"])
        safe_sid = sid.replace("/", "_")
        surface_cat = surface["surface_category"]
        surface_idx = sid.split("/")[0].replace(surface_cat, "").lstrip("_")

        # Group objects by category. One photo per (surface, category), with
        # camera aimed at THAT category's object centroid (not surface center,
        # not all-objects centroid). Surface is just the support structure;
        # we annotate the objects, so the specific category's items must
        # fill the frame.
        cat_to_objs = {}
        obj_cats = []
        for o in surface["objects_on_surface"]:
            c = o["category"]
            if c not in cat_to_objs:
                cat_to_objs[c] = []
                obj_cats.append(c)
            cat_to_objs[c].append(o)

        surface_viewpoints = []  # for scene overview (per-category cams)

        for cat in obj_cats:
            cat_objs = cat_to_objs[cat]
            # Target = centroid of objects in THIS category (XY only, z forced
            # to camera height so gaze stays horizontal — goodnav-style).
            cat_positions = [o["position"] for o in cat_objs if "position" in o]
            if cat_positions:
                cat_centroid = np.mean(np.array(cat_positions)[:, :2], axis=0)
                target = np.array([cat_centroid[0], cat_centroid[1], CAMERA_HEIGHT])
            else:
                target = np.array([surface_pos[0], surface_pos[1], CAMERA_HEIGHT])

            # No occlusion threshold — always render the LEAST occluded
            # viewpoint. If it fails the void check, try the next one.

            # Compute the target's combined bbox for projection check.
            cat_bbox_min = cat_bbox_max = None
            for o in cat_objs:
                bb = (o or {}).get("bbox") or {}
                bmin, bmax = bb.get("min"), bb.get("max")
                if bmin and bmax and len(bmin) >= 3:
                    if cat_bbox_min is None:
                        cat_bbox_min = list(bmin)
                        cat_bbox_max = list(bmax)
                    else:
                        for i in range(3):
                            cat_bbox_min[i] = min(cat_bbox_min[i], bmin[i])
                            cat_bbox_max[i] = max(cat_bbox_max[i], bmax[i])

            # Per-object orbit range: geometric min_r (don't fall out of FOV
            # below) + size-based preferred max_r, capped so orbits stay in
            # the same room. Returns 4 radii in the good middle zone —
            # avoids the too-far (wall crossing) / too-close (inside item)
            # extremes that linspace(min_r, max_r) picks.
            camera_h = CAMERA_HEIGHT

            target_z_vals = []
            size_vals = []
            for o in cat_objs:
                bb = (o or {}).get("bbox") or {}
                b_min, b_max = bb.get("min"), bb.get("max")
                if b_min and b_max and len(b_min) >= 3 and len(b_max) >= 3:
                    dx = abs(b_max[0] - b_min[0])
                    dy = abs(b_max[1] - b_min[1])
                    dz = abs(b_max[2] - b_min[2])
                    target_z_vals.append((b_min[2] + b_max[2]) / 2)
                    size_vals.append(max(dx, dy, dz))
            if size_vals:
                tgt_z = float(np.mean(target_z_vals))
                tgt_size = float(max(size_vals))
                # Floor: stay outside the "below FOV" cone. 1.2x adds margin
                # so object isn't at the very edge of the frame.
                # Use the most extreme Z of the bbox (not center) so the
                # ENTIRE object stays in the 45° half-FOV, not just its middle.
                bbox_z_min = min(b[2] for b in [cat_bbox_min] if b) if cat_bbox_min else tgt_z
                bbox_z_max = max(b[2] for b in [cat_bbox_max] if b) if cat_bbox_max else tgt_z
                worst_dz = max(abs(camera_h - bbox_z_min), abs(camera_h - bbox_z_max))
                r_floor = max(0.5, worst_dz * 1.2)
                # Pick a tier based on size; ALL radii are in the empirically
                # good middle zone (2-4m for big, 1.2-2.5m medium, 0.8-1.5m
                # small), then clamped up by r_floor so low objects don't
                # drop out of FOV.
                if tgt_size > 1.5:
                    base = (4.0, 3.2, 2.5, 2.0)
                    walk = 4.0
                elif tgt_size > 0.5:
                    base = (2.5, 2.0, 1.5, 1.2)
                    walk = 2.5
                else:
                    base = (1.5, 1.2, 1.0, 0.8)
                    walk = 1.5
                # If r_floor > base min, shift the whole tier up so we keep
                # 4 distinct radii instead of collapsing onto r_floor.
                shift = max(0.0, r_floor - min(base))
                vp_radii = tuple(r + shift for r in base)
                vp_walk_radius = max(walk, max(vp_radii))
            else:
                vp_radii, vp_walk_radius = None, None  # missing bbox → default
            # Orbit around the TARGET position (vase/cup centroid), not the
            # surface center (cabinet). For large surfaces the target can be
            # 1m+ from surface center — orbiting the surface puts the camera
            # at an angle where the target is off-screen or behind obstacles.
            # Pass the TARGET's bbox (not surface bbox) so wall-direction
            # and CW/CCW sweep are computed for the actual object.
            orbit_center = target[:2].tolist()
            viewpoints, gate_reason = compute_viewpoints(
                orbit_center, CAMERA_HEIGHT,
                freemap=freemap,
                radii=vp_radii, walk_radius=vp_walk_radius,
            )

            # Debug flow log (only with --save-miss-photos)
            if args.save_miss_photos:
                import time as _t
                prog_dir = os.path.join(args.output_dir, "batch_logs")
                os.makedirs(prog_dir, exist_ok=True)
                dbg_path = os.path.join(prog_dir, "debug_flow.log")
                with open(dbg_path, "a", encoding="utf-8") as df:
                    df.write(f"\n--- {scene_id} / {sid} / {cat} ---\n")
                    df.write(f"  target: ({target[0]:.2f},{target[1]:.2f},{target[2]:.2f})\n")
                    df.write(f"  orbit_center: ({orbit_center[0]:.2f},{orbit_center[1]:.2f})\n")
                    df.write(f"  bbox: min={cat_bbox_min} max={cat_bbox_max}\n")
                    if size_vals:
                        df.write(f"  size={tgt_size:.2f} z={tgt_z:.2f} r_floor={r_floor:.2f}\n")
                        df.write(f"  radii={vp_radii} walk_r={vp_walk_radius:.2f}\n")
                    df.write(f"  gate: {gate_reason or 'passed'}, vps={len(viewpoints)}\n")
                    if gate_reason:
                        df.write(f"  → SKIPPED: {gate_reason}\n")
                    df.flush()

            accepted = None
            vp_stats = []
            debug_photos = []

            # Pre-compute visibility score for ALL viewpoints (pure geometry).
            # Score = target_projected_pixels × (1 - occlusion_ratio)
            # Higher = better (big target + not blocked).
            # In open environments all occ≈0, so projected size breaks ties
            # (closer viewpoints with larger target projection win).
            scored_vps = []
            for vp_idx, vp in enumerate(viewpoints):
                occ_ratio = 0.0
                proj_area = 0.0
                if cat_bbox_min and cat_bbox_max:
                    t_rect = _project_bbox_to_2d_rect(
                        vp, target, cat_bbox_min, cat_bbox_max, IMAGE_SIZE
                    )
                    if t_rect:
                        tx0 = max(0, t_rect[0]); ty0 = max(0, t_rect[1])
                        tx1 = min(IMAGE_SIZE, t_rect[2]); ty1 = min(IMAGE_SIZE, t_rect[3])
                        proj_area = max(0, tx1-tx0) * max(0, ty1-ty0)
                    if scene_objects:
                        occ_ratio = compute_occlusion_ratio(
                            vp, target, cat_bbox_min, cat_bbox_max,
                            scene_objects, image_size=IMAGE_SIZE,
                        )
                effective_vis = proj_area * (1.0 - occ_ratio)
                scored_vps.append((-effective_vis, occ_ratio, vp_idx, vp))
            scored_vps.sort(key=lambda x: x[0])  # negative → highest first

            if args.save_miss_photos and scored_vps:
                with open(dbg_path, "a") as df:
                    best = scored_vps[0]
                    df.write(f"  best: vis={-best[0]:.0f}px occ={best[1]:.2f} "
                             f"n_vps={len(scored_vps)}\n")

            for neg_vis, occ_ratio, vp_idx, vp in scored_vps:
                # --- Render (least occluded first) ---
                tmp_name = f"_pending_{safe_sid}_{cat}_vp{vp_idx}.png"
                photo_path = os.path.join(photos_dir, tmp_name)

                success = capture_photo(
                    camera, world, vp, target, photo_path,
                )
                total_photos += 1

                if not success or not os.path.exists(photo_path):
                    vp_stats.append({"idx": vp_idx, "status": "capture_failed"})
                    continue

                # Reject if large portion of image is pure black (camera
                # looking outside the scene boundary into void).
                try:
                    img_arr = np.array(Image.open(photo_path).convert("RGB"))
                    black_ratio = float((img_arr.max(axis=2) < 10).mean())
                    if black_ratio > 0.3:
                        total_rejected += 1
                        vp_stats.append({"idx": vp_idx, "status": "void",
                                         "black": round(black_ratio, 2)})
                        debug_photos.append((photo_path, vp_idx, "void"))
                        if args.save_miss_photos:
                            with open(dbg_path, "a") as df:
                                df.write(f"  vp{vp_idx}: VOID cam=({vp[0]:.2f},{vp[1]:.2f}) "
                                         f"black={black_ratio:.0%}\n")
                        continue
                except Exception:
                    pass

                # All checks passed → accept
                dist_r = round(float(np.linalg.norm(np.array(target) - np.array(vp))), 2)
                vp_stats.append({
                    "idx": vp_idx,
                    "status": "accepted",
                    "occ": round(occ_ratio, 2),
                    "r": dist_r,
                })
                if args.save_miss_photos:
                    with open(dbg_path, "a") as df:
                        df.write(f"  vp{vp_idx}: ACCEPTED cam=({vp[0]:.2f},{vp[1]:.2f}) "
                                 f"occ={occ_ratio:.2f} r={dist_r}m\n")
                accepted = (photo_path, vp)
                break

            if accepted:
                for p, _i, _st in debug_photos:
                    try: os.remove(p)
                    except OSError: pass
            else:
                if len(viewpoints) == 0:
                    reason = gate_reason or "no_viewpoints"
                else:
                    low_proj = [s for s in vp_stats if s.get("status") == "low_projection"]
                    if len(low_proj) == len(vp_stats):
                        reason = "low_projection_everywhere"
                    else:
                        reason = "capture_failed_everywhere"
                best_occ = min((s.get("occ", 1.0) for s in vp_stats if s.get("occ") is not None), default=1.0)
                print(f"  [SKIP] {sid} / {cat}: {reason} "
                      f"(vps={len(viewpoints)}, best_occ={best_occ:.2f})")
                saved_photos = []
                if args.save_miss_photos and debug_photos:
                    miss_dir = os.path.join(
                        args.output_dir, scene_id, "miss_photos",
                        f"{safe_sid}__{cat}",
                    )
                    os.makedirs(miss_dir, exist_ok=True)
                    for p, idx, status in debug_photos:
                        if os.path.exists(p):
                            dst = os.path.join(miss_dir, f"vp{idx:02d}_{status}.png")
                            try:
                                os.replace(p, dst)
                                saved_photos.append(os.path.relpath(dst, args.output_dir))
                            except OSError: pass
                else:
                    for p, _i, _st in debug_photos:
                        try: os.remove(p)
                        except OSError: pass
                miss_debug_entries.append({
                    "surface_id": sid,
                    "category": cat,
                    "relation": cat_objs[0].get("relation") if cat_objs else None,
                    "target_z": round(tgt_z, 2) if size_vals else None,
                    "target_size": round(tgt_size, 2) if size_vals else None,
                    "num_viewpoints": len(viewpoints),
                    "reason": reason,
                    "best_occ": round(best_occ, 3),
                    "vp_stats": vp_stats,
                    "miss_photos": saved_photos,
                })
                continue

            tmp_path, vp = accepted
            # Sequence_id is per (surface, category) — bump now since we captured
            surface_id_counter += 1
            sequence_id = surface_id_counter
            final_name = f"{sequence_id:02d}_{surface_cat}_{surface_idx}__{cat}.png"
            final_path = os.path.join(photos_dir, final_name)
            os.rename(tmp_path, final_path)

            photo_entry = {
                "path": f"surface_photos/{final_name}",
                "camera_position": vp.tolist(),
                "target_position": target.tolist(),
                "quality_ok": True,
            }

            representative = cat_objs[0]
            manifest.append({
                "sequence_id": sequence_id,
                "surface_id": sid,
                "room": surface["room"],
                "surface_category": surface_cat,
                "target_category": cat,
                "target_representative": representative["object_id"],
                "all_targets_in_category": [o["object_id"] for o in cat_objs],
                "photos": [photo_entry],
            })
            # One overview entry per captured category (matches photo seq_id)
            overview_surfaces.append({
                "surface_id": sid,
                "surface_pos": target[:2].tolist(),
                "objects_cats": [cat],
                "viewpoints": [vp.tolist()],
            })
            surface_viewpoints.append(vp.tolist())

        if surface_viewpoints:
            print(f"  {sid}: {len(surface_viewpoints)}/{len(obj_cats)} categories captured")

    # Build miss list with positions for the overview map (debug only).
    missed_for_map = None
    if args.save_miss_photos and miss_debug_entries:
        missed_for_map = []
        for me in miss_debug_entries:
            for s in surfaces:
                for o in s["objects_on_surface"]:
                    if s["surface_id"] == me.get("surface_id") and o["category"] == me.get("category"):
                        pos = o.get("position") or s.get("position")
                        if pos:
                            missed_for_map.append({
                                "position": pos,
                                "category": me.get("category"),
                                "reason": me.get("reason", ""),
                            })
                        break

    # Generate scene-level overviews: freemap-bg and rendered-bg (with same overlays)
    if overview_surfaces:
        scene_overview_path = os.path.join(args.output_dir, scene_id, "scene_overview.png")
        generate_scene_overview_map(freemap, overview_surfaces, scene_overview_path,
                                     scene_id, missed_targets=missed_for_map)
        if os.path.exists(topdown_path):
            scene_rendered_path = os.path.join(args.output_dir, scene_id, "scene_rendered.png")
            generate_annotated_topdown_map(
                freemap, overview_surfaces, scene_rendered_path, scene_id, topdown_path,
            )
            # Raw topdown is just an intermediate used to build scene_rendered;
            # the annotated version is what we keep.
            try:
                os.remove(topdown_path)
            except OSError:
                pass

    # Enrich manifest with scene-graph context (nearby_objects, room_objects)
    enrich_manifest_inplace(manifest, args.scene_summary, scene_id)

    # Save manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # Save per-miss diagnostic log only when debugging.
    if args.save_miss_photos and miss_debug_entries:
        miss_log_path = os.path.join(args.output_dir, scene_id, "miss_debug.json")
        with open(miss_log_path, "w", encoding="utf-8") as f:
            json.dump(miss_debug_entries, f, ensure_ascii=False, indent=2)

    # Don't clear world (we reuse across scenes); just remove this scene's USD
    # (will happen at start of next process_scene call)

    attempted_targets = sum(len(s["objects_on_surface"]) for s in surfaces)
    captured_targets = len(manifest)
    missed_targets = attempted_targets - captured_targets
    capture_rate = (captured_targets / attempted_targets) if attempted_targets else 0.0

    # Enumerate (surface, category) pairs that never yielded a photo so
    # downstream readers see which items were lost without grepping for
    # [SKIP] lines across the log.
    captured_pairs = {(e["surface_id"], e["target_category"]) for e in manifest}
    missed_pairs = []
    for s in surfaces:
        for o in s["objects_on_surface"]:
            pair = (s["surface_id"], o["category"])
            if pair not in captured_pairs and pair not in [
                (ms[0], ms[1]) for ms in missed_pairs
            ]:
                missed_pairs.append((s["surface_id"], o["category"], o.get("relation")))

    print(f"[OK] {scene_id}: {captured_targets}/{attempted_targets} targets captured "
          f"({missed_targets} missed, {capture_rate:.0%}), "
          f"{total_photos} photos taken, {total_rejected} rejected")
    if missed_pairs:
        print(f"  [MISSED in {scene_id}] ({len(missed_pairs)}):")
        for sid_miss, cat_miss, rel_miss in missed_pairs:
            print(f"    - {sid_miss} / {cat_miss} (rel={rel_miss})")

    # Dedicated progress log: one append per scene, no Isaac Sim warning
    # noise. Path: <output_dir>/batch_logs/progress.log — tail -f this to
    # see real-time scene completion without grep.
    import time as _t
    prog_dir = os.path.join(args.output_dir, "batch_logs")
    os.makedirs(prog_dir, exist_ok=True)
    prog_path = os.path.join(prog_dir, "progress.log")
    # Build a reason lookup so the progress log shows WHY each miss failed.
    reason_by_pair = {
        (e["surface_id"], e["category"]): (
            e["reason"], e.get("max_visible_px", 0),
            e.get("num_viewpoints", 0),
        )
        for e in miss_debug_entries
    }
    with open(prog_path, "a", encoding="utf-8") as pf:
        ts = _t.strftime("%Y-%m-%d %H:%M:%S")
        pf.write(
            f"{ts} [OK] {scene_id}: {captured_targets}/{attempted_targets} "
            f"({capture_rate:.0%}), {total_photos} photos, {total_rejected} rejected\n"
        )
        if missed_pairs:
            pf.write(f"  MISSED ({len(missed_pairs)}):\n")
            for s, c, r in missed_pairs:
                info = reason_by_pair.get((s, c))
                if info:
                    reason, max_vis, n_vps = info
                    pf.write(f"    - {s}/{c}[{r}] → {reason} "
                             f"(vps={n_vps}, max_vis={max_vis}px)\n")
                else:
                    pf.write(f"    - {s}/{c}[{r}] → (no diag)\n")
        pf.flush()

    return {
        "scene_id": scene_id,
        "num_surfaces": len(manifest),
        "attempted_targets": attempted_targets,
        "captured_targets": captured_targets,
        "missed_targets": missed_targets,
        "missed_pairs": missed_pairs,
        "total_photos": total_photos,
        "rejected_photos": total_rejected,
    }


def get_scene_list():
    """Determine which scenes to process (priority: --scene > --scene-list > all)."""
    if args.scene:
        return [args.scene]
    if args.scene_list and os.path.exists(args.scene_list):
        with open(args.scene_list, "r") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return sorted([
        d for d in os.listdir(args.surface_targets_dir)
        if os.path.isdir(os.path.join(args.surface_targets_dir, d))
        and os.path.exists(os.path.join(args.surface_targets_dir, d, "surface_targets.json"))
    ])


def filter_pending(scene_ids):
    """Remove scenes that already have a capture_manifest.json."""
    pending = []
    for sid in scene_ids:
        manifest = os.path.join(args.output_dir, sid, "capture_manifest.json")
        if args.force or not os.path.exists(manifest):
            pending.append(sid)
    return pending


def init_sim_context():
    """Create World + Camera + DomeLight ONCE, reused across all scenes this process."""
    from pxr import Sdf

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane(
        z_position=0,
        name="default_ground_plane",
        prim_path="/World/defaultGroundPlane",
    )

    # Dome light (goodnav demo.py style) — persistent across scenes
    stage = simulation_app.context.get_stage()
    dome_light = stage.DefinePrim("/World/DomeLight", "DomeLight")
    dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(450.0)

    # Camera — persistent; only pose changes per capture
    camera = Camera(
        prim_path=CAMERA_PRIM_PATH,
        frequency=30,
        resolution=(IMAGE_SIZE, IMAGE_SIZE),
    )
    camera.initialize()
    camera.set_clipping_range(near_distance=CLIPPING_NEAR, far_distance=CLIPPING_FAR)
    camera.set_focal_length(FOCAL_LENGTH)
    camera.set_horizontal_aperture(APERTURE)
    camera.set_vertical_aperture(APERTURE)

    return world, camera


def main():
    all_scenes = get_scene_list()
    pending = filter_pending(all_scenes)

    print(f"Total scenes: {len(all_scenes)}, pending: {len(pending)}")
    if not pending:
        print("[ALL DONE] no pending scenes")
        simulation_app.close()
        sys.exit(EXIT_ALL_DONE)

    # Process up to batch_size scenes in this process, then exit for shell to restart
    batch = pending[: args.batch_size]
    print(f"Processing batch of {len(batch)} scenes (batch_size={args.batch_size})")

    world, camera = init_sim_context()
    print(f"[INIT DONE] World + Camera ready, beginning scene loop")

    all_stats = []
    failures = []
    total_all = len(all_scenes)
    already_done = total_all - len(pending)
    for i, sid in enumerate(batch):
        print(f"\n{'='*30} [{i+1}/{len(batch)}] {sid} {'='*30}")
        try:
            stats = process_scene(sid, world, camera)
            if stats:
                all_stats.append(stats)
            else:
                failures.append(sid)
        except Exception as e:
            print(f"[ERROR] {sid}: unexpected exception: {e}")
            import traceback
            traceback.print_exc()
            failures.append(sid)
        # Cumulative progress: count manifests on disk (includes prior sessions
        # and the scene just processed). `i+1` is this session's index.
        done_now = sum(
            1 for s in all_scenes
            if os.path.exists(os.path.join(args.output_dir, s, "capture_manifest.json"))
        )
        print(f"[PROGRESS] {done_now}/{total_all} scenes complete "
              f"(this session: {i+1}/{len(batch)}, "
              f"pre-existing: {already_done})")

    # Summary
    print(f"\n{'='*60}")
    print(f"Batch done: {len(all_stats)} OK, {len(failures)} failed/skipped")
    fresh = [s for s in all_stats if not s.get("skipped")]
    if fresh:
        total_p = sum(s.get("total_photos", 0) for s in fresh)
        total_att = sum(s.get("attempted_targets", 0) for s in fresh)
        total_cap = sum(s.get("captured_targets", 0) for s in fresh)
        total_miss = sum(s.get("missed_targets", 0) for s in fresh)
        rate = (total_cap / total_att) if total_att else 0.0
        print(f"This batch: {total_cap}/{total_att} targets captured "
              f"({total_miss} missed, {rate:.0%}), {total_p} raw photos")
    if failures:
        print(f"Failed scenes: {failures[:5]}{'...' if len(failures) > 5 else ''}")

    # List every scene with a manifest on disk (cumulative across sessions)
    done_list = sorted(
        s for s in all_scenes
        if os.path.exists(os.path.join(args.output_dir, s, "capture_manifest.json"))
    )
    print(f"\n[DONE SCENES] {len(done_list)}/{len(all_scenes)}:")
    for chunk_start in range(0, len(done_list), 8):
        print("  " + " ".join(done_list[chunk_start:chunk_start + 8]))

    # Determine exit code for shell wrapper
    remaining = filter_pending(all_scenes)
    simulation_app.close()
    if not remaining:
        print("[ALL DONE]")
        sys.exit(EXIT_ALL_DONE)
    else:
        print(f"[BATCH DONE] {len(remaining)} scenes still pending, shell should restart")
        sys.exit(EXIT_BATCH_DONE)


if __name__ == "__main__":
    main()
