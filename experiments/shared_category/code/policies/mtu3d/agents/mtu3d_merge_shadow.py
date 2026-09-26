"""Diagnostic exact shadow of the real CPU merge, sharing one GPU perception."""
from __future__ import annotations
from copy import deepcopy
import random

import numpy as np


def rng_snapshot() -> tuple:
    import torch
    return (random.getstate(), deepcopy(np.random.get_state()),
            torch.get_rng_state().clone(), [state.clone() for state in torch.cuda.get_rng_state_all()])


def same_rng(a: tuple, b: tuple) -> bool:
    import torch
    return (a[0] == b[0] and a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1])
            and a[1][2:] == b[1][2:] and torch.equal(a[2], b[2])
            and len(a[3]) == len(b[3]) and all(torch.equal(x, y) for x, y in zip(a[3], b[3])))


def install_merge_shadow(model, observer) -> None:
    """Compare all original manager state and RNG, without a second GPU forward."""
    manager = model.representation_manager
    actual_merge = manager.merge
    observer.shadow_audits = []

    def merge(predictions):
        shadow = object.__new__(type(manager))
        keys = []
        for key, value in vars(manager).items():
            if callable(value):
                continue
            if not isinstance(value, (np.ndarray, int, float, bool, str)):
                raise TypeError(f'Unsupported upstream manager state: {key}')
            setattr(shadow, key, deepcopy(value))
            keys.append(key)
        baseline_rng = rng_snapshot()
        shadow.merge(deepcopy(predictions))
        if not same_rng(baseline_rng, rng_snapshot()):
            raise ValueError('Original merge consumes RNG')
        result = actual_merge(predictions)
        if not same_rng(baseline_rng, rng_snapshot()):
            raise ValueError('Observer changed RNG')
        array_keys = []
        for key in keys:
            old, new = getattr(shadow, key), getattr(manager, key)
            if isinstance(old, np.ndarray):
                if (old.dtype != new.dtype or not np.isfinite(old).all()
                        or not np.array_equal(old, new)):
                    raise ValueError(f'Observer changed actual merge state: {key}')
                array_keys.append(key)
            elif old != new:
                raise ValueError(f'Observer changed merge configuration: {key}')
        audit = {'passed': True, 'array_keys_equal': array_keys, 'all_state_keys': keys,
                 'objects': len(manager.object_box), 'rng_unchanged': True,
                 'same_perception_inputs': True, 'extra_gpu_forward_calls': 0}
        observer.shadow_audits.append(audit)
        observer.last_shadow_audit = audit
        return result

    manager.merge = merge
