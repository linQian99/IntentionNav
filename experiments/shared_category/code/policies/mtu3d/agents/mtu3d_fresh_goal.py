"""Close the learned-object approach loop using actual current RGB-D evidence.

Predicted boxes and accepted masks remain fallible model outputs. This module
never receives evaluation identities, visibility labels or ground-truth boxes.
"""
from __future__ import annotations

import math
import numpy as np

from mtu3d_observation_support import describe_view
from mtu3d_strict_grid import strict_path, strict_segment
from mtu3d_surface_sample import sample_range


def selected_box(decision: dict) -> dict:
    trace = decision['stage2_trace']
    return next(box for box in trace['top_objects']
                if box['memory_index'] == trace['selected_memory_index'])


def geometry(position, yaw, box, intrinsics) -> dict:
    """Check the estimated2m surface region and center in the actual camera.

    Full box containment is deliberately not required: partial observations of
    large objects are valid navigation evidence. Visibility is checked anew.
    """
    position = np.asarray(position, dtype=float)
    center, size = np.asarray(box['position']), np.asarray(box['box_size'])
    if (position.shape != (3,) or center.shape != (3,) or size.shape != (3,)
            or not np.isfinite([position, center, size]).all() or np.any(size <= 0)
            or not math.isfinite(yaw)):
        raise ValueError('Invalid model box or camera')
    intr = intrinsics
    if (not all(math.isfinite(intr[k]) for k in ['width', 'height', 'fx', 'fy', 'cx', 'cy'])
            or intr['width'] <= 1 or intr['height'] <= 1 or min(intr['fx'], intr['fy']) <= 0):
        raise ValueError('Invalid camera calibration')
    delta = center - position
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.])
    axial = float(delta @ forward)
    distance = float(np.linalg.norm(np.maximum(np.abs(delta[:2]) - size[:2] / 2, 0.)))
    pixels = None
    if axial > 0:
        pixels = [float(intr['cx'] + intr['fx'] * (delta @ np.cross(forward, [0., 0., 1.])) / axial),
                  float(intr['cy'] - intr['fy'] * delta[2] / axial)]
    in_view = pixels is not None and 0 <= pixels[0] <= intr['width'] - 1 and 0 <= pixels[1] <= intr['height'] - 1
    return {'predicted_surface_distance_m': distance, 'center_pixel_xy': pixels,
            'center_in_camera': bool(in_view), 'inside_estimated_goal_region': distance <= 2.0,
            'feasible': bool(in_view and distance <= 2.0), 'radius_m': 2.0}


def consistent_support(support, box) -> bool:
    """Reject support whose actual representative is outside the selected learned box.

    This is a geometric correspondence check, not a semantic or confidence
    score. The tiny tolerance covers only float32 coordinate roundoff.
    """
    if 'surface_sample' not in support:
        return False
    median = np.asarray(support['surface_sample']['world_xyz'], dtype=float)
    center, size = np.asarray(box['position']), np.asarray(box['box_size'])
    return bool(median.shape == (3,) and np.isfinite(median).all()
                and support['sampled_mask_points'] > 0
                and np.all(np.abs(median - center) <= size / 2 + 1e-6))


def fresh_stop_evidence(decision, view, rgb, depth) -> dict:
    """Require this call's selected object to have accepted current-frame support.

    A recent history view, old memory membership or a merely reached waypoint
    cannot authorize STOP. The complete current RGB-D/pose identity must match.
    """
    evidence = {'passed': False, 'ground_truth_used': False,
                'source': 'current_learned_object_actual_sample_and_estimated_goal_region'}
    if not decision.get('is_object_decision') or decision.get('decision_source') != 'mtu3d_learned':
        return {**evidence, 'reason': 'no_current_learned_object_decision'}
    box = selected_box(decision)
    yaw = math.atan2(view['look_dir'][1], view['look_dir'][0])
    geo = geometry(view['position'], yaw, box, view['intrinsics'])
    index = decision['observations_supplied'] - 1
    actual = describe_view({**view, 'rgb': rgb, 'depth': depth}, decision['decision_index'], index)
    matches = [s for s in decision['stage2_trace']['selected_observation_support']
               if all(s.get(k) == v for k, v in actual.items()) and consistent_support(s, box)]
    if index not in decision['valid_view_indices']:
        matches = []
    measured = [sample_range(view['position'], s) for s in matches]
    eligible = [s for s, distance in zip(matches, measured) if distance <= 2.0]
    return {**evidence, 'passed': bool(geo['feasible'] and eligible), 'geometry': geo,
            'current_surface_sample_distances_m': measured,
            'within_range_support_view_ids': [s['view_id'] for s in eligible],
            'current_view_id': actual['view_id'], 'decision_index': decision['decision_index'],
            'selected_memory_index': box['memory_index'],
            'matching_support_view_ids': [s['view_id'] for s in matches],
            'reason': 'verified' if geo['feasible'] and eligible else
                      'current_mask_absent_or_inconsistent' if not matches else
                      'current_sample_outside_goal_range' if not eligible else 'camera_or_distance_not_ready'}


def plan_observed_goal(wm, position, box, supports) -> dict | None:
    """Prefer an actual supported camera; approach from it only if too far.

    Endpoints approach an actual accepted surface sample with full ground-segment checks,
    and camera-center feasibility. Never invent a viewpoint by maximizing a
    box halfspace clearance. Every arrival still requires fresh learned evidence.
    """
    current = np.asarray(position, dtype=float)
    center = np.asarray(box['position'], dtype=float)
    resolution = min(abs(float(wm.x_coords[1] - wm.x_coords[0])),
                     abs(float(wm.y_coords[1] - wm.y_coords[0])))
    candidates = []
    for support in supports:
        if not consistent_support(support, box):
            continue
        origin = np.asarray(support['position'], dtype=float)
        if abs(origin[2] - current[2]) > 1e-8 or not wm.is_walkable(*origin[:2]):
            continue
        surface = np.asarray(support['surface_sample']['world_xyz'], dtype=float)
        delta = surface[:2] - origin[:2]
        length = float(np.linalg.norm(delta))
        if length <= 1e-8:
            continue
        yaw = math.atan2(center[1] - origin[1], center[0] - origin[0])
        endpoint = None
        # Preserve exact acquired pose rather than snapping a witness across a wall.
        if geometry(origin, yaw, box, support['intrinsics'])['feasible'] and sample_range(origin, support) <= 2.0:
            endpoint = origin[:2].tolist()
        elif (geometry(origin, yaw, box, support['intrinsics'])['predicted_surface_distance_m'] > 2.0
              or sample_range(origin, support) > 2.0):
            # First reachable camera entering the fixed region, not a tuned standoff.
            seen = set()
            for amount in np.linspace(0., 1., max(2, math.ceil(2 * length / resolution) + 1)):
                point = origin[:2] + amount * delta
                if not strict_segment(wm, origin[:2], point):
                    break
                cell = wm.world_to_cell(*point)
                if cell in seen:
                    continue
                seen.add(cell)
                candidate = wm.cell_to_world(*cell)
                heading = math.atan2(center[1] - candidate[1], center[0] - candidate[0])
                if (strict_segment(wm, origin[:2], candidate)
                        and geometry([*candidate, current[2]], heading, box, support['intrinsics'])['feasible']
                        and sample_range([*candidate, current[2]], support) <= 2.0):
                    endpoint, yaw = list(candidate), heading
                    break
        if endpoint is None:
            continue
        path = strict_path(wm, current[:2], endpoint)
        # A* returns cell centers; an acquired camera may be anywhere inside
        # that cell. Validate the final connector without snapping the witness.
        if not path or not strict_segment(wm, path[-1], endpoint):
            continue
        route = (math.dist(current[:2], path[0])
                 + sum(math.dist(a, b) for a, b in zip(path, path[1:]))
                 + math.dist(path[-1], endpoint))
        plan = {'navigation_endpoint_xy': endpoint, 'target_yaw': yaw,
                'predicted_box': box, 'source_view': support, 'route_length_m': route,
                'source_to_endpoint_m': math.dist(origin[:2], endpoint),
                'arrival_tolerance_m': 1e-5, 'fresh_current_view_verified': False,
                'surface_sample_distance_at_endpoint_m': sample_range([*endpoint, current[2]], support),
                'source': 'actual_accepted_surface_sample_source_ray_goal_entry'}
        candidates.append((route, support['view_id'], plan))
    return min(candidates, key=lambda x: x[:2])[-1] if candidates else None


def approach_action(position, yaw, plan) -> dict:
    if plan is None:
        return {'action': 'NO_SUPPORTED_ROUTE'}
    if math.dist(position[:2], plan['navigation_endpoint_xy']) > plan['arrival_tolerance_m']:
        return {'action': 'MOVE'}
    delta = math.atan2(math.sin(plan['target_yaw'] - yaw), math.cos(plan['target_yaw'] - yaw))
    if abs(delta) > 1e-6:
        return {'action': 'ROTATE_TARGET_RECENTER',
                'target_yaw': yaw + max(-math.pi / 2, min(math.pi / 2, delta))}
    return {'action': 'REOBSERVE'}
