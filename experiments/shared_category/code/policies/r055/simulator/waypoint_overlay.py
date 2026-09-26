"""Draw A/B/C/D waypoint markers on the agent's current RGB observation.

The resulting image is what a VLM-as-policy sees each step. Markers are
large, high-contrast, and labeled so the VLM reliably picks one letter.

Each waypoint is a 3D world position; we project it to 2D using the camera
extrinsics (position + look direction) and intrinsics (focal + aperture)
matching capture_surfaces.py / iss_env.py.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]
MARKER_COLORS = [
    (230, 50, 50, 220),    # A - red
    (50, 150, 230, 220),   # B - blue
    (50, 200, 80, 220),    # C - green
    (230, 180, 40, 220),   # D - amber
    (180, 80, 220, 220),   # E - purple
    (240, 120, 50, 220),   # F - orange
    (60, 200, 200, 220),   # G - cyan
    (200, 60, 150, 220),   # H - magenta
]


def _project(world_xyz: np.ndarray, cam_pos: np.ndarray, cam_look: np.ndarray,
             image_w: int, image_h: int,
             focal: float = 10.0, aperture: float = 20.0
             ) -> tuple[tuple[int, int], bool]:
    """Pinhole projection for up-right camera + off-screen edge placement.

    Returns (pixel_uv, in_view). When the world point is in the camera's
    forward FOV, pixel_uv is the projected location and in_view=True.
    When the point is OUTSIDE the FOV (rear, far left, far right), we
    fall back to a deterministic edge-of-frame anchor:
      bearing in [+45°, +135°]  → right edge (mid-height)
      bearing in [-135°, -45°]  → left edge (mid-height)
      |bearing| > 135°          → bottom edge, side hinted by sign
    The VLM sees a marker at that edge so it knows the waypoint exists
    and roughly where, even when it's not directly visible.
    """
    fwd = cam_look / max(np.linalg.norm(cam_look), 1e-9)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up)
    right /= max(np.linalg.norm(right), 1e-9)
    cam_up = np.cross(right, fwd)

    rel = world_xyz - cam_pos
    x_cam = float(np.dot(rel, right))
    y_cam = float(np.dot(rel, cam_up))
    z_cam = float(np.dot(rel, fwd))

    half_fov = math.atan2(aperture / 2.0, focal)
    f_px = (image_w / 2.0) / math.tan(half_fov)

    # Bearing in xy plane relative to camera forward (signed). Positive
    # = right of camera, negative = left.
    bearing = math.atan2(x_cam, max(z_cam, 1e-6) if z_cam > 0 else z_cam)
    # When z_cam <= 0 the bearing computed via atan2(x, z) gives a
    # value with magnitude > 90°, which is what we want.
    bearing = math.atan2(x_cam, z_cam)

    in_fov = (z_cam > 0.05 and abs(bearing) < half_fov)
    if in_fov:
        u = image_w / 2.0 + x_cam * f_px / z_cam
        v = image_h / 2.0 - y_cam * f_px / z_cam
        return (int(round(u)), int(round(v))), True

    # Off-screen: place at edge based on bearing.
    margin = 50
    deg = math.degrees(bearing)
    if deg >= 135 or deg <= -135:
        # Behind. Place at bottom, biased by sign of x_cam.
        u = image_w * (0.7 if x_cam > 0 else 0.3)
        v = image_h - margin
    elif deg > 0:
        # Off to the right.
        u = image_w - margin
        # Vertical: closer to half_fov boundary → higher; bigger angle → lower
        # Map bearing in (half_fov, π) to v in (image_h/2, image_h - margin)
        t = (deg - math.degrees(half_fov)) / (180 - math.degrees(half_fov))
        t = max(0.0, min(1.0, t))
        v = image_h * (0.5 + 0.4 * t)
    else:
        # Off to the left.
        u = margin
        t = (-deg - math.degrees(half_fov)) / (180 - math.degrees(half_fov))
        t = max(0.0, min(1.0, t))
        v = image_h * (0.5 + 0.4 * t)

    return (int(round(u)), int(round(v))), False


def overlay_waypoints(rgb: np.ndarray, waypoints_xy: list[tuple[float, float]],
                      cam_pos_xyz: np.ndarray, cam_look: np.ndarray,
                      marker_z: float = 0.1) -> Image.Image:
    """Return a PIL Image with numbered markers drawn on the RGB observation.

    Waypoints outside the camera FOV are drawn as edge arrows instead of
    hidden (so the VLM knows they exist and roughly where they are).
    """
    img = Image.fromarray(rgb.astype(np.uint8)).convert("RGBA")
    draw = ImageDraw.Draw(img, mode="RGBA")
    W, H = img.size

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 48)
    except Exception:
        font = ImageFont.load_default()

    for i, (wx, wy) in enumerate(waypoints_xy):
        if i >= len(LABELS):
            break
        label = LABELS[i]
        color = MARKER_COLORS[i]

        world_xyz = np.array([wx, wy, marker_z])
        (u, v), in_view = _project(world_xyz, np.asarray(cam_pos_xyz),
                                    np.asarray(cam_look), W, H)
        u_clamped = max(40, min(W - 40, u))
        v_clamped = max(40, min(H - 40, v))

        r = 40
        if in_view:
            # Filled circle for in-view waypoints
            draw.ellipse([u_clamped - r, v_clamped - r,
                          u_clamped + r, v_clamped + r],
                         fill=color, outline=(0, 0, 0, 255), width=3)
        else:
            # Off-screen: dashed/striped circle to differentiate.
            # Visually distinct so VLM treats text bearing as authoritative.
            draw.ellipse([u_clamped - r, v_clamped - r,
                          u_clamped + r, v_clamped + r],
                         fill=None, outline=color, width=8)
            # Cross-hatch indicating "outside FOV"
            draw.line([u_clamped - r * 0.6, v_clamped - r * 0.6,
                       u_clamped + r * 0.6, v_clamped + r * 0.6],
                      fill=color, width=4)
            draw.line([u_clamped - r * 0.6, v_clamped + r * 0.6,
                       u_clamped + r * 0.6, v_clamped - r * 0.6],
                      fill=color, width=4)
        # Text centered
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        text_color = (255, 255, 255, 255) if in_view else color
        draw.text((u_clamped - tw / 2, v_clamped - th / 2 - 4),
                  label, fill=text_color, font=font)

    return img.convert("RGB")


def save_overlay(rgb: np.ndarray, waypoints_xy: list[tuple[float, float]],
                 cam_pos_xyz: np.ndarray, cam_look: np.ndarray,
                 out_path: Path):
    img = overlay_waypoints(rgb, waypoints_xy, cam_pos_xyz, cam_look)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path), format="PNG", optimize=True)
