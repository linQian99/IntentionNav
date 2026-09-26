"""Audit acquired support using the producing policy's coordinate precision.

Versioned replacement for the fixed raster-bound check in the historical
supported-view audit. The policy, goal radius, depth tolerance, planes and
source identity are unchanged. Import the precision helper only after the
caller selects the pinned policy source; historical auditors stay immutable.
"""
from __future__ import annotations

import math
import numpy as np


def verify_support(support, requests, decisions, current_index, cache, describe_view, volume_fn):
    """Check actual consumed view identity, camera and four frustum witnesses.

    Membership in a learned mask is supplied by the frozen merge observer. This
    CPU audit neither reruns segmentation nor certifies the semantic category.
    """
    from mtu3d_projection_precision import raster_tolerance

    index, view_index = support['decision_index'], support['view_index']
    if type(index) is not int or type(view_index) is not int or not 0 <= index <= current_index:
        raise ValueError('Future or malformed source decision')
    request, decision = requests[index], decisions[index]
    views = request['context_observations'] + [request]
    if view_index not in decision['valid_view_indices']:
        raise ValueError('Object support uses an unconsumed view')
    identity = (index, view_index)
    if identity not in cache:
        view = views[view_index]
        with np.load(view['observation'], allow_pickle=False) as arrays:
            cache[identity] = describe_view(view | {'rgb': arrays['rgb'], 'depth': arrays['depth']}, index, view_index)
    if any(support.get(k) != v for k, v in cache[identity].items()):
        raise ValueError('Support camera/RGB-D identity differs from its acquired source')
    volume = support['view_volume']
    witnesses = np.asarray(volume['plane_witness_points_xyz'], dtype=float)
    if witnesses.shape != (4, 3) or not np.isfinite(witnesses).all():
        raise ValueError('Invalid surface witnesses')
    reconstructed = volume_fn(witnesses, support)
    for key in ['plane_normals', 'plane_lower_bounds', 'forward', 'source_yaw']:
        value = np.asarray(volume[key], dtype=float)
        if not np.isfinite(value).all() or not np.allclose(value, reconstructed[key], atol=1e-6, rtol=0):
            raise ValueError(f'Surface plane witness mismatch: {key}')
    minimum = volume['minimum_surface_forward']
    if (not math.isfinite(minimum) or minimum > reconstructed['minimum_surface_forward'] + 1e-6
            or volume['surface_points'] != support['sampled_mask_points']
            or type(support['sampled_mask_points']) is not int or support['sampled_mask_points'] <= 0):
        raise ValueError('Invalid learned support count or near plane')
    relative = witnesses - np.asarray(support['position'])
    forward = np.asarray(reconstructed['forward'])
    axial = relative @ forward
    intr = support['intrinsics']
    pixels = np.column_stack([intr['cx'] + intr['fx'] * (relative @ np.cross(forward, [0., 0., 1.])) / axial,
                              intr['cy'] - intr['fy'] * relative[:, 2] / axial])
    # Use the same coordinate-precision contract as the frozen observation
    # producer and representative-depth verifier. Source identity, planes,
    # positive depth, finite coordinates and unique-pixel bounds still apply.
    # All four witnesses and the representative originate in the same cloud.
    coordinate_dtype = support['surface_sample']['source_coordinate_dtype']
    tolerance = raster_tolerance(witnesses, support, coordinate_dtype)
    if (np.any(axial <= 0) or not np.isfinite(pixels).all()
            or np.any(pixels < -tolerance)
            or np.any(pixels[:, 0] > intr['width'] - 1 + tolerance[:, 0])
            or np.any(pixels[:, 1] > intr['height'] - 1 + tolerance[:, 1])):
        raise ValueError('Surface witnesses do not project to the actual source image')

