"""Integrate each acquired RGB-D measurement once into persistent MTU memory.

Perception still receives the complete requested context. Identity uses actual
RGB, depth, camera pose and calibration, never a scene, category or target label.
This is an adaptation change, not an assertion about upstream model quality.
"""
from __future__ import annotations

from copy import deepcopy

from mtu3d_empty_observation import EmptyObjectMemory


class UniqueEvidence:
    """Filter repeated measurement predictions before the existing merge chain."""

    def __init__(self, manager, observer):
        self.observer = observer
        self.original_merge = manager.merge
        self.seen: set[str] = set()
        self.last_audit = None
        manager.merge = self.merge

    def reset(self) -> None:
        self.seen.clear()
        self.last_audit = None

    def merge(self, predictions):
        views = self.observer.pending_views
        identities = [view['view_id'] for view in views]
        if not identities or any(not isinstance(v, str) or not v for v in identities):
            raise ValueError('Unique evidence requires actual consumed-view identities')
        allowed = set(identities)
        new_ids = list(dict.fromkeys(v for v in identities if v not in self.seen))
        retained, integrated, skipped = [], [], []
        predicted_clouds = set()
        for prediction in predictions:
            cloud = prediction['point_cloud']
            entry = self.observer.cloud_views.get(id(cloud))
            if entry is None or entry[0] is not cloud or entry[1]['view_id'] not in allowed:
                raise ValueError('Prediction lacks an exact current stage1 source identity')
            if id(cloud) in predicted_clouds:
                raise ValueError('Upstream produced duplicate prediction blocks for one cloud')
            predicted_clouds.add(id(cloud))
            identity = entry[1]['view_id']
            if identity in self.seen or identity in integrated:
                skipped.append(identity)
            else:
                retained.append(prediction)
                integrated.append(identity)
        audit = dict(mode='unique_rgbd_evidence_v1', consumed_view_ids=identities,
            newly_consumed_view_ids=new_ids, integrated_prediction_view_ids=integrated,
            skipped_prediction_view_ids=skipped,
            new_views_without_predictions=[v for v in new_ids if v not in integrated],
            unique_views_before=len(self.seen), unique_views_after=len(self.seen | allowed),
            predictions_supplied=len(predictions), predictions_integrated=len(retained),
            ground_truth_used=False, perception_context_changed=False,
            identity_fields=['rgb_sha256', 'depth_sha256', 'position', 'look_dir', 'intrinsics'])
        try:
            result = self.original_merge(retained)
        except EmptyObjectMemory:
            # The existing guard raises after a successful empty merge. A
            # zero-query view is observed evidence too, not a free retry slot.
            self.seen.update(allowed)
            self.last_audit = deepcopy(audit)
            raise
        self.seen.update(allowed)
        self.last_audit = deepcopy(audit)
        return result
