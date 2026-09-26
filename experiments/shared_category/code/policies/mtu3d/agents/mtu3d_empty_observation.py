"""Handle absent object evidence without inventing an object or swallowing errors."""
from __future__ import annotations

import numpy as np


class EmptyObjectMemory(RuntimeError):
    """Perception and merging completed, but no real object is available."""


def guard_empty_object_memory(manager) -> None:
    """Stop before the upstream object-only argmax when merging yields no object."""
    original_merge = manager.merge

    def merge(predictions):
        result = original_merge(predictions)
        if len(manager.object_box) == 0:
            raise EmptyObjectMemory('No object survived perception and merging')
        return result

    manager.merge = merge


def exploration_without_object(position, frontiers) -> dict:
    """Use a supplied geometric frontier, or request a charged turn if none exists."""
    current = np.asarray(position, dtype=float)
    candidates = np.asarray(frontiers, dtype=float).reshape(-1, 3)
    if not np.isfinite(current).all() or not np.isfinite(candidates).all():
        raise ValueError('Nonfinite pose or frontier')
    distances = np.linalg.norm(candidates[:, :2] - current[:2], axis=1)
    valid = np.flatnonzero(distances >= .05)
    if len(valid):
        index = int(valid[np.argmin(distances[valid])])
        target = candidates[index]
        hint = 'MOVE'
    else:
        target = current
        hint = 'ROTATE_SCAN'
    return {'target_position': target.tolist(), 'is_object_decision': False,
            'action_hint': hint, 'decision_source': 'geometry_without_object_evidence'}


def close_simulator(env) -> None:
    """Consume Isaac's successful SystemExit so an earlier exception survives."""
    try:
        env.close()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise RuntimeError(f'Isaac shutdown exited with status {exc.code}') from exc
