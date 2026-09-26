"""Retain a representative actual accepted point, with its source pixel/depth.

The point nearest the coordinatewise median is a sampled representative, not
an exact medoid and not a ground-truth object surface. Its semantic membership
is supplied by the unchanged learned segmentation/merge.
"""
from __future__ import annotations

import math
import numpy as np
from mtu3d_projection_precision import raster_tolerance


def representative_sample(points, mask, view):
    mask = np.asarray(mask, dtype=bool)
    selected_rows = np.flatnonzero(mask)
    world = np.asarray(points)[selected_rows, :3].astype(float, copy=True)
    if not len(world) or not np.isfinite(world).all():
        raise ValueError('Empty/nonfinite accepted points')
    world[:, 1] *= -1  # Stage1 [Habitat x,Habitat z,Habitat y] -> Isaac.
    median = np.median(world, axis=0)
    index = int(np.argmin(np.sum((world - median) ** 2, axis=1)))
    point = world[index]
    forward = np.asarray(view['look_dir'], dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    delta = point - np.asarray(view['position'], dtype=float)
    axial = float(delta @ forward)
    if axial <= 0:
        raise ValueError('Accepted representative is behind its camera')
    intr = view['intrinsics']
    pixel = [float(intr['cx'] + intr['fx'] * (delta @ right) / axial),
             float(intr['cy'] - intr['fy'] * (delta @ up) / axial)]
    return {'world_xyz': point.tolist(), 'point_cloud_row': int(selected_rows[index]),
            'accepted_mask_rank': index, 'pixel_xy': pixel, 'axial_depth_m': axial,
            'source_coordinate_dtype': np.asarray(points).dtype.name,
            'selection_rule': 'actual_accepted_sample_nearest_coordinatewise_median',
            'ground_truth_used': False}


def sample_range(position, support):
    sample = support.get('surface_sample')
    if sample is None:
        return None
    point = np.asarray(sample['world_xyz'], dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError('Invalid accepted sample')
    return math.dist(position[:2], point[:2])


def verify_sample_depth(support, depth):
    """Check that the recorded point reconstructs from its actual source pixel.

    This checks sensor provenance, not segmentation correctness. Metre tolerance
    covers float32 camera/world transformation only; goal radius stays exactly2m.
    """
    sample = support['surface_sample']
    if (sample['selection_rule'] != 'actual_accepted_sample_nearest_coordinatewise_median'
            or sample['ground_truth_used'] is not False
            or type(sample['point_cloud_row']) is not int or sample['point_cloud_row'] < 0
            or type(sample['accepted_mask_rank']) is not int
            or not 0 <= sample['accepted_mask_rank'] < support['sampled_mask_points']):
        raise ValueError('Invalid representative source metadata')
    intr = support['intrinsics']
    depth = np.asarray(depth)
    if depth.shape != (intr['height'], intr['width']):
        raise ValueError('Representative source depth/calibration mismatch')
    uv = np.asarray(sample['pixel_xy'], dtype=float)
    if uv.shape != (2,) or not np.isfinite(uv).all():
        raise ValueError('Invalid source pixel')
    pixel = np.rint(uv).astype(int)
    tolerance = raster_tolerance([sample['world_xyz']], support,
        sample.get('source_coordinate_dtype', 'float32'))[0]
    if (np.any(np.abs(uv - pixel) > tolerance) or not 0 <= pixel[0] < intr['width']
            or not 0 <= pixel[1] < intr['height']):
        raise ValueError('Representative does not correspond to a source raster pixel')
    value = float(depth[pixel[1], pixel[0]])
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Representative uses invalid source depth')
    forward = np.asarray(support['look_dir'], dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    point = (np.asarray(support['position'], dtype=float) + value * forward
             + value * (pixel[0] - intr['cx']) / intr['fx'] * right
             - value * (pixel[1] - intr['cy']) / intr['fy'] * up)
    xyz = np.asarray(sample['world_xyz'], dtype=float)
    axial = float(sample['axial_depth_m'])
    if (xyz.shape != (3,) or not np.isfinite(xyz).all() or not math.isfinite(axial)
            or not np.allclose(point, xyz, atol=1e-5, rtol=0)
            or abs(value - axial) > 1e-5):
        raise ValueError('Representative point does not match the actual pixel depth')
    return {'passed': True, 'pixel_xy': pixel.tolist(), 'depth_m': value,
            'maximum_world_error_m': float(np.max(np.abs(point - xyz)))}
