"""Stable per-episode and per-decision random streams for MTU3D inference."""
from __future__ import annotations

import hashlib
import json
import random

import numpy as np


def decision_seed(base_seed: int, episode_key: str, decision_index: int) -> int:
    """Derive a seed without Python hash randomization or run/arm identity."""
    if not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError('base_seed must be a nonnegative integer')
    if not isinstance(episode_key, str) or not episode_key:
        raise ValueError('A nonempty episode key is required')
    if not isinstance(decision_index, int) or decision_index < 0:
        raise ValueError('decision_index must be a nonnegative integer')
    payload=json.dumps(['mtu3d-decision-v1',base_seed,episode_key,decision_index],
                       separators=(',',':'),ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4],'big')


def seed_generators(seed: int) -> dict:
    """Reset standard generators; this alone does not promise deterministic CUDA."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return {'seed':seed,'python':True,'numpy':True,'torch_cpu_cuda':True,
            'custom_cuda_determinism_guaranteed':False}
