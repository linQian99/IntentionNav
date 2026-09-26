"""Read-only snapshots of actual learned memory; overlap is not identity proof."""
from __future__ import annotations

import hashlib
import itertools
import math
from pathlib import Path

import numpy as np


def cosine(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    # Scalar fsum avoids BLAS-dependent summation in cross-environment replay.
    scale = math.sqrt(math.fsum(float(x) ** 2 for x in a)) * math.sqrt(math.fsum(float(x) ** 2 for x in b))
    return math.fsum(float(x) * float(y) for x, y in zip(a, b)) / scale if scale else None


def describe_pairs(arrays):
    """Describe shared actual memory rows and features without association gates."""
    ids = arrays['memory_indices'].tolist()
    masks = arrays['object_mask']
    if masks.ndim != 2 or masks.shape[1] != len(ids) or len(set(ids)) != len(ids):
        raise ValueError('Memory indices and mask columns disagree')
    if not np.isin(masks, [0, 1]).all():
        raise ValueError('Expected actual binary memory membership')
    masks = masks.astype(bool)
    rows = []
    for i, j in itertools.combinations(range(len(ids)), 2):
        left, right = masks[:, i], masks[:, j]
        nleft, nright = int(left.sum()), int(right.sum())
        shared = int(np.count_nonzero(left & right))
        union = nleft + nright - shared
        rows.append(dict(left_memory_index=ids[i], right_memory_index=ids[j],
            left_points=nleft, right_points=nright, shared_points=shared,
            mask_iou=shared / union if union else None,
            left_containment=shared / nleft if nleft else None,
            right_containment=shared / nright if nright else None,
            object_feature_cosine=cosine(arrays['object_feat'][i], arrays['object_feat'][j]),
            open_vocab_feature_cosine=cosine(arrays['open_vocab_feat'][i], arrays['open_vocab_feat'][j])))
    return rows


def snapshot_memory(manager, indices, output, episode_key, decision_index):
    """Archive the selected top5 columns after the original merge and stage2.

    These are the original voxel-subsampled memory rows, not a new segmentation
    or ground-truth identity. No state, prediction, threshold or RNG is changed.
    """
    if not isinstance(decision_index, int) or decision_index < 0:
        raise ValueError('Invalid decision index')
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or np.any(indices < 0):
        raise ValueError('Invalid top5 memory indices')
    arrays = {'memory_indices': indices.copy(), 'point_cloud': manager.point_cloud.copy(),
              'object_mask': manager.object_mask[:, indices].copy()}
    for name in ['object_box', 'object_feat', 'open_vocab_feat', 'object_score', 'object_count', 'object_class']:
        arrays[name] = getattr(manager, name)[indices].copy()
    pairs = describe_pairs(arrays)
    directory = Path(output) / hashlib.sha256(episode_key.encode()).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'decision_{decision_index:03d}.npz'
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    return dict(snapshot=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        episode_key=episode_key, decision_index=decision_index,
        memory_indices=indices.tolist(), memory_rows=len(arrays['point_cloud']), pairs=pairs,
        source='actual_postmerge_voxel_memory_columns', ground_truth_used=False,
        modifies_memory=False, semantic_identity_established=False)


def verify_snapshot(report):
    path = Path(report['snapshot'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != report['sha256']:
        raise ValueError('Memory snapshot checksum mismatch')
    with np.load(path, allow_pickle=False) as arrays:
        if (len(arrays['point_cloud']) != report['memory_rows']
                or arrays['memory_indices'].tolist() != report['memory_indices']
                or describe_pairs(arrays) != report['pairs']):
            raise ValueError('Memory snapshot metrics mismatch')
    if report['modifies_memory'] or report['ground_truth_used'] or report['semantic_identity_established']:
        raise ValueError('Unexpected diagnostic scope')
    return True
