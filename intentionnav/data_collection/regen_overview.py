"""Regenerate scene_overview.png (freemap bg) and scene_topdown.png
(annotated 3D render bg) using existing capture_manifest.json + raw topdown.
No Isaac Sim needed.

Usage:
    python -m intentionnav.data_collection.regen_overview \\
        --metaroot /path/to/metadata_train \\
        --work-dir /path/to/work_dir [--scene kujiale_0005]
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _build_overview_surfaces(manifest):
    """Collapse per-category manifest entries into per-surface records.

    Each entry has: surface_id, target_category, photos[0].camera_position,
    target_representative (to get surface_pos via... wait, surface has no
    position field in manifest). We use the first entry's target_position
    as surface center approximation, or the mean of target_positions of
    all same-surface entries.
    """
    by_surface = {}
    for e in manifest:
        sid = e["surface_id"]
        rec = by_surface.setdefault(sid, {
            "surface_id": sid,
            "targets": [],
            "viewpoints": [],
            "objects_cats": [],
        })
        rec["targets"].append(e["photos"][0]["target_position"])
        rec["viewpoints"].append(e["photos"][0]["camera_position"])
        if e["target_category"] not in rec["objects_cats"]:
            rec["objects_cats"].append(e["target_category"])

    out = []
    for sid, rec in by_surface.items():
        targets = np.array(rec["targets"])
        # surface position = centroid of targets (approximation)
        sp = targets.mean(axis=0).tolist()
        out.append({
            "surface_id": sid,
            "surface_pos": sp,
            "objects_cats": rec["objects_cats"],
            "viewpoints": rec["viewpoints"],
        })
    return out


def _draw_overlays(ax, lax, surfaces, scene_id):
    """Shared marker + legend drawing. ax is the map axes, lax the legend."""
    import matplotlib.cm as cm
    n = len(surfaces)
    colors = cm.get_cmap("tab20", max(n, 1))

    legend_lines = []
    for idx, s in enumerate(surfaces):
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

    lax.set_xlim(0, 1)
    lax.set_ylim(0, 1)
    lax.axis("off")
    line_height = 1.0 / (n + 2)
    lax.text(0.5, 1.0 - line_height * 0.4, f"{n} surfaces",
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


def generate_overview(freemap, surfaces, save_path, scene_id):
    """Freemap-background overview (unchanged style)."""
    if not surfaces:
        return
    x_coords = freemap[0, 1:]
    y_coords = freemap[1:, 0]
    grid = freemap[1:, 1:]

    fig, (ax, lax) = plt.subplots(1, 2, figsize=(16, 12),
                                    gridspec_kw={"width_ratios": [3, 1]})
    ax.imshow(grid, cmap="gray_r", origin="lower",
              extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]])
    _draw_overlays(ax, lax, surfaces, scene_id)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def _detect_building_bbox_in_render(td_arr):
    """Find bounding box of building (non-blue-floor) pixels in a rotated topdown.

    The Isaac Sim render has a blue-grey tiled infinite floor as background.
    Building walls + room interiors are non-blue. We threshold on color,
    morphologically clean up, keep the largest connected component, and
    return its bounding box in pixel coords.

    Returns (row_min, row_max, col_min, col_max) or None if detection fails.
    """
    try:
        from scipy import ndimage
    except ImportError:
        return None
    if td_arr.shape[2] < 3:
        return None
    r, g, b = td_arr[:, :, 0], td_arr[:, :, 1], td_arr[:, :, 2]
    # Background blue tile: B > R significantly, B high, G mid-high
    is_building = ~((b > r + 0.1) & (b > 0.65))
    # Opening to remove tile grid noise, then pick largest component
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


def generate_annotated_topdown(freemap, surfaces, save_path, scene_id, topdown_path):
    """Same overlays as overview, but 3D top-down render as background.

    Alignment: capture_topdown_render places camera at (cx+0.1, cy, altitude)
    looking at (cx, cy, 0) with 90° FOV. At altitude A the image covers
    roughly ±A meters in both X and Y around the scene center. Empirically
    (see tools/orientation test), applying `np.rot90(k=1)` to the rendered
    image and using matplotlib origin='upper' makes world XY align with
    image pixels for the yaw/pitch used by capture_topdown_render.
    """
    if not surfaces or not os.path.exists(topdown_path):
        return
    x_coords = freemap[0, 1:]
    y_coords = freemap[1:, 0]

    td_img = plt.imread(topdown_path)
    cx = float((x_coords[0] + x_coords[-1]) / 2)
    cy = float((y_coords[0] + y_coords[-1]) / 2)
    size_x = float(abs(x_coords[-1] - x_coords[0]))
    size_y = float(abs(y_coords[-1] - y_coords[0]))
    scene_size = max(size_x, size_y)
    altitude = max(8.0, scene_size * 0.7)
    half = altitude  # world half-extent covered by the raw rendered image

    # Empirically verified (freemap wall overlay test): rot90(k=1, CCW) +
    # origin='upper' aligns Isaac Sim topdown render to freemap world XY.
    # After rotation: pixel (row=0, col=0) ↔ (cx-half, cy+half).
    td_arr = np.rot90(td_img, k=1)
    H, W = td_arr.shape[:2]

    # Crop to the ACTUAL building bbox (detected from render pixels) so the
    # outermost walls align with the coordinate axes — avoids showing large
    # empty floor areas around the building.
    bbox = _detect_building_bbox_in_render(td_arr)
    if bbox is None:
        # Fallback: crop to freemap bounds
        x_min_w, x_max_w = float(min(x_coords[0], x_coords[-1])), float(max(x_coords[0], x_coords[-1]))
        y_min_w, y_max_w = float(min(y_coords[0], y_coords[-1])), float(max(y_coords[0], y_coords[-1]))
        r0 = max(0, int(round(((cy + half) - y_max_w) / (2 * half) * H)))
        r1 = min(H, int(round(((cy + half) - y_min_w) / (2 * half) * H)))
        c0 = max(0, int(round((x_min_w - (cx - half)) / (2 * half) * W)))
        c1 = min(W, int(round((x_max_w - (cx - half)) / (2 * half) * W)))
    else:
        r0, r1, c0, c1 = bbox
        r1 += 1  # inclusive → slice bound
        c1 += 1

    td_crop = td_arr[r0:r1, c0:c1]
    ext_left = (cx - half) + (c0 / W) * 2 * half
    ext_right = (cx - half) + (c1 / W) * 2 * half
    ext_top = (cy + half) - (r0 / H) * 2 * half
    ext_bottom = (cy + half) - (r1 / H) * 2 * half

    fig, (ax, lax) = plt.subplots(1, 2, figsize=(16, 12),
                                    gridspec_kw={"width_ratios": [3, 1]})
    ax.imshow(
        td_crop,
        extent=[ext_left, ext_right, ext_bottom, ext_top],
        origin="upper",
        interpolation="bilinear",
    )
    # Match freemap X-inversion convention so markers' XY system aligns
    ax.set_xlim(ext_right, ext_left)
    ax.set_ylim(ext_bottom, ext_top)

    _draw_overlays(ax, lax, surfaces, scene_id)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def process(scene_id, args):
    work = os.path.join(args.work_dir, scene_id)
    manifest_path = os.path.join(work, "capture_manifest.json")
    topdown_path = os.path.join(work, "scene_topdown.png")   # raw 3D render
    rendered_path = os.path.join(work, "scene_rendered.png")  # annotated
    overview_path = os.path.join(work, "scene_overview.png")  # freemap bg
    freemap_path = os.path.join(args.metaroot, scene_id, "freemap.npy")

    if not os.path.exists(manifest_path):
        return "no_manifest"
    if not os.path.exists(freemap_path):
        return "no_freemap"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not manifest:
        return "empty_manifest"

    freemap = np.load(freemap_path)
    surfaces = _build_overview_surfaces(manifest)

    # scene_overview.png: freemap background with overlays (unchanged style)
    generate_overview(freemap, surfaces, overview_path, scene_id)

    # scene_rendered.png: 3D render with the same overlays
    if os.path.exists(topdown_path):
        generate_annotated_topdown(
            freemap, surfaces, rendered_path, scene_id, topdown_path,
        )
        return f"ok ({len(surfaces)} surfaces + rendered)"
    return f"ok ({len(surfaces)} surfaces, no topdown render)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metaroot", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--scene", default=None)
    args = parser.parse_args()

    if args.scene:
        scenes = [args.scene]
    else:
        scenes = sorted([
            d for d in os.listdir(args.work_dir)
            if os.path.isdir(os.path.join(args.work_dir, d))
            and os.path.exists(os.path.join(args.work_dir, d, "capture_manifest.json"))
        ])

    print(f"Regenerating overview for {len(scenes)} scenes...")
    for sid in scenes:
        status = process(sid, args)
        print(f"  {sid}: {status}")


if __name__ == "__main__":
    main()
