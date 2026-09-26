"""Reach an observed side of a learned object while preserving its view volume."""
from __future__ import annotations

import math

import numpy as np
from mtu3d_strict_grid import strict_segment, strict_path


def view_volume(points_world, view: dict) -> dict:
    """Compress all accepted surface points into exact fixed-yaw frustum planes.

For each side plane n, all points P fit a camera at C iff n.C >= max(n.P).
This contains the actual sampled learned mask, not the unobserved whole object.
The orientation and intrinsics are from the actual consumed view; no GT enters.
"""
    points = np.asarray(points_world, dtype=float)
    forward = np.asarray(view['look_dir'], dtype=float)
    forward /= np.linalg.norm(forward)
    if abs(forward[2]) > 1e-8 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Expected horizontal navigation camera and Nx3 surface')
    if not len(points) or not np.isfinite(points).all():
        raise ValueError('Empty/nonfinite learned surface')
    right = np.cross(forward, [0., 0., 1.])
    up = np.array([0., 0., 1.])
    intr = view['intrinsics']
    normals = np.array([
        right - (intr['width'] - 1 - intr['cx']) / intr['fx'] * forward,
        -right - intr['cx'] / intr['fx'] * forward,
        up - intr['cy'] / intr['fy'] * forward,
        -up - (intr['height'] - 1 - intr['cy']) / intr['fy'] * forward,
    ])
    projections = points @ normals.T
    limits = projections.max(axis=0)
    return {'plane_normals': normals.tolist(), 'plane_lower_bounds': limits.tolist(),
        'plane_witness_points_xyz': points[projections.argmax(axis=0)].tolist(),
        'forward': forward.tolist(), 'minimum_surface_forward': float((points @ forward).min()),
        'source_yaw': math.atan2(forward[1], forward[0]), 'surface_points': len(points),
        'source': 'all_actual_accepted_mask_points_fixed_source_orientation'}


def surface_fits_camera(volume: dict, camera_position) -> bool:
    """Check every sampled surface point via its exact four support bounds."""
    position = np.asarray(camera_position, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError('Invalid camera position')
    normals = np.asarray(volume['plane_normals'], dtype=float)
    lower = np.asarray(volume['plane_lower_bounds'], dtype=float)
    if normals.shape != (4, 3) or lower.shape != (4,) or not np.isfinite(normals).all() or not np.isfinite(lower).all():
        raise ValueError('Invalid learned viewing volume')
    # Float32 source points can drift ~1e-7m; no depth/occlusion tolerance is used.
    return bool(np.all(normals @ position >= lower - 1e-6)
        and float(np.dot(position, volume['forward'])) < volume['minimum_surface_forward'])


def plan_supported_view(walkable_map, current_position, predicted_center, supports: list[dict]) -> dict | None:
    """Choose reachable grid endpoints along already observed approach rays.

Source-to-endpoint ground LOS keeps the approach on the observed free side.
Frustum bounds retain the observed mask's framing at its recorded orientation.
Neither proves fresh semantic visibility; later policy observations remain paid.
"""
    current = np.asarray(current_position, dtype=float)
    center = np.asarray(predicted_center, dtype=float)
    if current.shape != (3,) or center.shape != (3,) or not np.isfinite([current, center]).all():
        raise ValueError('Invalid predicted center or camera pose')
    resolution = min(abs(float(walkable_map.x_coords[1] - walkable_map.x_coords[0])),
                     abs(float(walkable_map.y_coords[1] - walkable_map.y_coords[0])))
    if resolution <= 0:
        raise ValueError('Invalid map resolution')
    candidates = []
    for support in supports:
        source = np.asarray(support['position'], dtype=float)
        surface = np.asarray(support['surface_median_xyz'], dtype=float)
        volume = support['view_volume']
        if (abs(source[2] - current[2]) > 1e-8
                or not walkable_map.is_walkable(*source[:2])):
            continue
        direction = surface[:2] - source[:2]
        length = float(np.linalg.norm(direction))
        if length <= 1e-8:
            continue
        direction /= length
        seen = set()
        # Half a grid cell covers each cell crossed by the actual viewing ray.
        for distance in np.linspace(0., length, max(2, math.ceil(2 * length / resolution) + 1)):
            ray_point = source[:2] + distance * direction
            if (not walkable_map.is_walkable(*ray_point)
                    or not strict_segment(walkable_map, source[:2], ray_point)):
                break
            cell = walkable_map.world_to_cell(*ray_point)
            if cell in seen:
                continue
            seen.add(cell)
            endpoint = walkable_map.cell_to_world(*cell)
            camera = [*endpoint, current[2]]
            if (not walkable_map.is_walkable(*endpoint)
                    or not strict_segment(walkable_map, source[:2], endpoint)
                    or not surface_fits_camera(volume, camera)):
                continue
            candidates.append((math.dist(endpoint, center[:2]), math.dist(current[:2], endpoint),
                -support['sampled_mask_points'], support['view_id'], {
                    'navigation_endpoint_xy': list(endpoint), 'target_yaw': volume['source_yaw'],
                    'predicted_center_xyz': center.tolist(), 'source_view': support,
                    'source_to_endpoint_m': math.dist(source[:2], endpoint),
                    'arrival_tolerance_m': 1e-5,
                    'source': 'accepted_observation_side_and_surface_frustum'}))
    for candidate in sorted(candidates, key=lambda row: row[:4]):
        plan = candidate[-1]
        endpoint = plan['navigation_endpoint_xy']
        # Query A* only for geometrically valid candidates, nearest first.
        path = strict_path(walkable_map, tuple(current[:2]), tuple(endpoint))
        if path and math.dist(path[-1], endpoint) <= 1e-8:
            plan['route_length_m'] = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
            return plan
    return None


def terminal_view_action(current_position, yaw: float, plan: dict | None) -> dict:
    """Return MOVE, a paid <=90deg rotation, or STOP at the planned view pose."""
    if plan is None:
        return {'action': 'NO_SUPPORTED_ROUTE'}
    if math.dist(current_position[:2], plan['navigation_endpoint_xy']) > plan['arrival_tolerance_m']:
        return {'action': 'MOVE', 'target_xy': plan['navigation_endpoint_xy']}
    delta = math.atan2(math.sin(plan['target_yaw'] - yaw), math.cos(plan['target_yaw'] - yaw))
    if abs(delta) > 1e-6:
        turn = max(-math.pi / 2, min(math.pi / 2, delta))
        return {'action': 'ROTATE_TARGET', 'target_yaw': yaw + turn}
    if not surface_fits_camera(plan['source_view']['view_volume'], current_position):
        raise ValueError('Arrived camera violates its planned surface framing')
    return {'action': 'STOP'}
