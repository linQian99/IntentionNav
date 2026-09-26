"""Archive already computed R055 observations without extra policy calls."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def verification_errors(value, path: str = 'verification') -> list[str]:
    """Find actual model exceptions, including errors nested by the ensemble."""
    if isinstance(value, dict):
        errors = [f"{path}.error: {value['error']}"] if value.get('error') else []
        return errors + [error for key, child in value.items() if key != 'error'
                         for error in verification_errors(child, f'{path}.{key}')]
    if isinstance(value, list):
        return [error for index, child in enumerate(value)
                for error in verification_errors(child, f'{path}[{index}]')]
    return []


class ObservationArchive:
    """Keep one lossless input and one computed-candidate trace per paid step.

    The observer never receives an environment, ground-truth objects, model or
    callback. It cannot acquire views or make policy decisions. Use only in
    bounded diagnostic runs; serialization failures invalidate their closure.
    """

    def __init__(self, directory: Path, step_cap: int) -> None:
        if not 1 <= step_cap <= 30:
            raise ValueError('Archive requires the declared 1–30 action cap')
        self.directory = Path(directory)
        self.step_cap = step_cap
        self.frames: list[dict] = []
        self.started_at = time.perf_counter()
        self.archive_seconds = 0.

    def capture(self, *, step: int, pose: dict, rgb: np.ndarray,
                depth: np.ndarray, detections: list, strict_detections: list,
                verifications: list, selected: tuple | None,
                evidence: dict | None, error: str | None) -> None:
        start = time.perf_counter()
        if step != len(self.frames) + 1 or step > self.step_cap:
            raise ValueError('Noncontiguous or over-budget observation')
        if rgb.dtype != np.uint8 or rgb.shape != (768, 768, 3):
            raise ValueError('Require actual 768-square RGB')
        if depth is None or depth.shape != (768, 768) or not np.issubdtype(depth.dtype, np.floating):
            raise ValueError('Require actual floating-point depth at matching resolution')
        if set(pose) != {'position', 'yaw'}:
            raise ValueError('Only the policy camera pose is accepted')
        errors = ([error] if error is not None else []) + verification_errors(verifications)
        # The unchanged verifier intentionally evaluates at most five boxes.
        if min(len(strict_detections), 5) != len(verifications) and not errors:
            raise ValueError('Computed verification count differs from the original five-box cap')
        strict_indices = {id(d): i for i, d in enumerate(strict_detections)}
        if len(strict_indices) != len(strict_detections):
            raise ValueError('Duplicate strict candidate identity')
        if any(id(d) not in {id(x) for x in detections} for d in strict_detections):
            raise ValueError('Strict candidate was not actually proposed')
        if selected is not None and not any(selected is d for d in detections):
            raise ValueError('Selected candidate was not actually proposed')
        candidates = []
        for index, d in enumerate(detections):
            strict_index = strict_indices.get(id(d))
            candidates.append({'index': index, 'label': d[0], 'score': float(d[1]),
                'bbox': [float(x) for x in d[2]], 'strict_label_match': strict_index is not None,
                'verification': copy.deepcopy(verifications[strict_index])
                    if strict_index is not None and strict_index < len(verifications) else None,
                'verification_status': ('label_filtered' if strict_index is None else
                    'computed' if strict_index < len(verifications) else
                    'not_selected_for_verification'),
                'selected': d is selected})
        data = {'schema': 'r055_acquired_observation_v1', 'step': step,
                'pre_action_pose': copy.deepcopy(pose), 'candidates': candidates,
                'admitted_evidence': copy.deepcopy({k: v for k, v in (evidence or {}).items()
                                                   if k != 'cluster'}),
                'perception_error': error, 'perception_errors': errors,
                'input': f'step_{step:02d}_policy_input.npz',
                'intrinsics': {'width': 768, 'height': 768, 'fx': 384., 'fy': 384.,
                               'cx': 384., 'cy': 384., 'depth_convention': 'axial'}}
        # Validate metadata before creating an immutable observation artifact.
        json.dumps(data, allow_nan=False)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / data['input']
        with path.open('xb') as stream:
            np.savez_compressed(stream, rgb=rgb, depth=depth)
        data['input_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        with (self.directory / f'step_{step:02d}_perception.json').open('x') as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
        self.frames.append(data)
        self.archive_seconds += time.perf_counter() - start
        if errors:
            raise RuntimeError(f'Perception error preserved; diagnostic run is invalid: {errors}')

    def finish(self, trajectory: list[dict]) -> dict:
        if any(frame['perception_errors'] for frame in self.frames):
            raise ValueError('Perception errors prevent successful closure')
        steps = [r['step'] for r in trajectory[1:]]
        if steps != [r['step'] for r in self.frames]:
            raise ValueError('Observation/action closure mismatch')
        if any(r.get('action') == 'STOP' for r in trajectory[:-1]):
            raise ValueError('Activity after STOP')
        for previous, frame in zip(trajectory, self.frames):
            if (frame['pre_action_pose']['position'] != previous['position'] or
                    frame['pre_action_pose']['yaw'] != previous['yaw']):
                raise ValueError('Acquired pose differs from pre-action trajectory')
        return {'schema': 'r055_acquired_observation_v1', 'observations': len(self.frames),
                'engine_to_policy_termination_seconds': time.perf_counter() - self.started_at,
                'archive_seconds': self.archive_seconds,
                'timing_scope': 'Engine entry through policy termination; excludes scene load and terminal evaluation render.',
                'input_bytes': sum((self.directory / r['input']).stat().st_size for r in self.frames),
                'extra_render_or_inference_calls': 0}
