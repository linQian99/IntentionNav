"""Bound raster reprojection error caused by rounded world coordinates.

The learned point cloud stores world coordinates in float32. At centimetre
camera depth, micrometre world rounding can exceed a fixed .01-pixel check.
This bound changes only provenance validation, never points or navigation radii.
"""
from __future__ import annotations

import numpy as np


def pixel_roundoff_bound(world, view: dict, coordinate_dtype='float32') -> np.ndarray:
    """Propagate half-ULP coordinate intervals through the pinhole quotient.

The denominator must remain positive over its uncertainty interval. Bounds
at or above half a pixel do not establish a unique source raster sample.
"""
    world = np.asarray(world, dtype=np.float64).reshape(-1, 3)
    dtype = np.dtype(coordinate_dtype)
    if dtype not in (np.dtype('float32'), np.dtype('float64')) or not np.isfinite(world).all():
        raise ValueError('Require finite float32/float64 source coordinates')
    rounded = world.astype(dtype)
    if not np.array_equal(rounded.astype(np.float64), world):
        raise ValueError('World coordinates do not have the declared storage precision')
    previous = np.nextafter(rounded, np.full_like(rounded, -np.inf)).astype(np.float64)
    following = np.nextafter(rounded, np.full_like(rounded, np.inf)).astype(np.float64)
    # A half-ULP bounds round-to-nearest at exponent boundaries as well.
    uncertainty = .5*np.maximum(world-previous, following-world)
    position = np.asarray(view['position'], dtype=np.float64)
    forward = np.asarray(view['look_dir'], dtype=np.float64)
    forward = forward/np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right = right/np.linalg.norm(right)
    up = np.cross(right, forward)
    # Include float64 pose/quaternion and dot-product arithmetic, without a
    # distance-dependent empirical pixel threshold.
    uncertainty += 32*np.finfo(np.float64).eps*(np.abs(world)+np.abs(position)+1.)
    delta = world-position
    axial = delta@forward
    axial_error = uncertainty@np.abs(forward)
    if np.any(axial <= axial_error):
        raise ValueError('Coordinate precision cannot establish positive camera depth')
    intr = view['intrinsics']
    bounds = []
    for direction, focal in [(right, intr['fx']), (up, intr['fy'])]:
        lateral = delta@direction
        lateral_error = uncertainty@np.abs(direction)
        bounds.append(focal*(lateral_error+np.abs(lateral/axial)*axial_error)/(axial-axial_error))
    result = np.column_stack(bounds)
    if not np.isfinite(result).all() or np.any(result >= .5):
        raise ValueError('Coordinate precision cannot identify a unique source pixel')
    return result


def raster_tolerance(world, view: dict, coordinate_dtype='float32') -> np.ndarray:
    """Keep the original small tolerance, expanding only by a proved bound."""
    return np.maximum(.01, pixel_roundoff_bound(world, view, coordinate_dtype))
