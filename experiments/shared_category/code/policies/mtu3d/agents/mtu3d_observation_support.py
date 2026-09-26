"""Observe the real MTU merge to retain camera sources for learned objects.

This observer never alters predictions, masks, merging, RNG, or model outputs.
Membership comes from the accepted learned masks before voxel downsampling;
it is observation provenance, not proof that the queried category is correct.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import sys

import numpy as np
from mtu3d_projection_precision import raster_tolerance
from mtu3d_supported_view import view_volume
from mtu3d_surface_sample import representative_sample


def array_digest(array) -> str:
    array = np.asarray(array)
    header = json.dumps([array.dtype.str, list(array.shape)]).encode()
    return hashlib.sha256(header + array.tobytes(order='C')).hexdigest()


def describe_view(view: dict, decision_index: int, view_index: int) -> dict:
    """Identify only the actual supplied RGB-D and its camera calibration."""
    result = {key: deepcopy(view[key]) for key in ['position', 'look_dir', 'intrinsics']}
    result.update(rgb_sha256=array_digest(view['rgb']),
                  depth_sha256=array_digest(view['depth']))
    result['view_id'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    result.update(decision_index=int(decision_index), view_index=int(view_index))
    return result


def summarize_mask(points, mask, view: dict) -> dict:
    """Record the learned mask's actual sampled surface in its source image."""
    selected = np.asarray(points)[np.asarray(mask, dtype=bool), :3].copy()
    if not len(selected) or not np.isfinite(selected).all():
        raise ValueError('Empty or nonfinite accepted object support')
    # Stage1/memory coordinates are [Habitat x, Habitat z, Habitat y].
    selected[:, 1] *= -1
    forward = np.asarray(view['look_dir'], dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = selected - np.asarray(view['position'])
    axial = relative @ forward
    if np.any(axial <= 0):
        raise ValueError('Accepted source points are behind their source camera')
    intr = view['intrinsics']
    pixels = np.column_stack([intr['cx'] + intr['fx'] * (relative @ right) / axial,
                              intr['cy'] - intr['fy'] * (relative @ up) / axial])
    tolerance = raster_tolerance(selected, view, selected.dtype)
    if (np.any(pixels < -tolerance) or np.any(pixels[:, 0] > intr['width'] - 1 + tolerance[:, 0])
            or np.any(pixels[:, 1] > intr['height'] - 1 + tolerance[:, 1])):
        raise ValueError('Accepted mask does not project into its recorded source view')
    return {**deepcopy(view), 'sampled_mask_points': len(selected),
            'mask_sha256': array_digest(np.asarray(mask, dtype=bool)),
            'sampled_cloud_sha256': array_digest(points),
            'surface_median_xyz': np.median(selected, axis=0).tolist(),
            'view_volume': view_volume(selected, view),
            'surface_sample': representative_sample(points, mask, view),
            'pixel_bounds_xyxy': [*pixels.min(axis=0).tolist(), *pixels.max(axis=0).tolist()]}


class ObservationSupport:
    """Attach an observer to the exact existing merge and stage1 input path.

The original merge only appends point rows and reorders/removes object columns.
Old column masks in the retained point prefix therefore identify their exact
history. Ambiguous or empty old masks discard history rather than guessing.
New support uses accepted membership in the actual appended source rows.
"""

    def __init__(self, model):
        self.manager = model.representation_manager
        self.module = sys.modules[type(self.manager).__module__]
        self.original_merge = self.manager.merge
        self.supports = []
        self.pending_views = []
        self.cloud_views = {}
        self.last_audit = None
        self.manager.merge = self.merge
        model.pq3d_stage1.register_forward_pre_hook(self.capture_stage1)

    def reset(self) -> None:
        self.supports = []
        self.pending_views = []
        self.cloud_views = {}
        self.last_audit = None

    def begin(self, views: list[dict], decision_index: int, valid_indices: list[int]) -> None:
        self.pending_views = [describe_view(views[index], decision_index, index)
                              for index in valid_indices]
        self.cloud_views = {}

    def capture_stage1(self, module, inputs) -> None:
        clouds = inputs[0]['raw_coordinates']
        if len(clouds) != len(self.pending_views):
            raise ValueError('Stage1 source order differs from consumed camera views')
        # Preserve object identity; pred_dict_list can omit zero-query frames.
        self.cloud_views = {id(cloud): (cloud, view)
                            for cloud, view in zip(clouds, self.pending_views)}

    def merge(self, predictions):
        old_mask = self.manager.object_mask
        old_points = self.manager.point_cloud
        if len(self.supports) != old_mask.shape[1]:
            raise ValueError('Object provenance and memory columns are misaligned')
        blocks = []
        offset = len(old_points)
        for prediction in predictions:
            cloud = prediction['point_cloud']
            entry = self.cloud_views.get(id(cloud))
            if entry is None or entry[0] is not cloud:
                raise ValueError('Predicted mask has no exact consumed-frame identity')
            blocks.append((offset, offset + len(cloud), cloud, entry[1]))
            offset += len(cloud)
        original_downsample = self.module.voxel_downsample_point_cloud_and_mask
        observed = []

        def observe(points, masks, *args, **kwargs):
            if observed or len(points) != offset or len(masks) != offset:
                raise ValueError('Unexpected merge/downsample point accounting')
            if not np.array_equal(points[:len(old_points)], old_points):
                raise ValueError('Merge changed the historical point prefix')
            old_signatures = {}
            for column in range(old_mask.shape[1]):
                values = old_mask[:, column]
                if np.any(values):
                    old_signatures.setdefault(array_digest(values), []).append(column)
            retained = [[] for _ in range(masks.shape[1])]
            ambiguous = 0
            for column in range(masks.shape[1]):
                prefix = masks[:len(old_points), column]
                matches = old_signatures.get(array_digest(prefix), []) if np.any(prefix) else []
                if len(matches) == 1:
                    retained[column] = deepcopy(self.supports[matches[0]])
                elif len(matches) > 1:
                    ambiguous += 1
            for first, last, cloud, view in blocks:
                if not np.array_equal(points[first:last], cloud):
                    raise ValueError('Merge changed the appended source cloud')
                for column in np.flatnonzero(np.any(masks[first:last], axis=0)):
                    evidence = summarize_mask(cloud, masks[first:last, column], view)
                    retained[column] = [v for v in retained[column] if v['view_id'] != view['view_id']]
                    retained[column].append(evidence)
            result = original_downsample(points, masks, *args, **kwargs)
            if result[1].shape[1] != len(retained):
                raise ValueError('Voxel downsampling reordered object columns')
            observed.append(retained)
            self.last_audit = {'consumed_views': len(self.pending_views),
                'predicted_view_blocks': len(blocks), 'old_points': len(old_points),
                'appended_points': offset - len(old_points), 'objects': len(retained),
                'ambiguous_historical_masks_discarded': ambiguous,
                'objects_with_support': sum(bool(v) for v in retained),
                'source': 'actual_accepted_merge_mask_membership', 'ground_truth_used': False}
            return result

        self.module.voxel_downsample_point_cloud_and_mask = observe
        try:
            result = self.original_merge(predictions)
        finally:
            self.module.voxel_downsample_point_cloud_and_mask = original_downsample
        if len(observed) != 1:
            raise ValueError('Original merge did not execute its expected downsample')
        self.supports = observed[0]
        return result

    def for_object(self, index: int) -> list[dict]:
        return deepcopy(self.supports[index])
