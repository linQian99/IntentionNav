"""Check acquired R055 inputs against saved images, decisions and camera geometry.

This CPU audit runs after actual child success and before independent scoring.
It checks archive consistency; it does not infer visibility or model accuracy.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from shared_navigation_run_inputs import audit as audit_shared

SCHEMA = 'r055_acquired_observation_v1'
INTRINSICS = {'width': 768, 'height': 768, 'fx': 384., 'fy': 384.,
              'cx': 384., 'cy': 384., 'depth_convention': 'axial'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def has_error(value) -> bool:
    if isinstance(value, dict):
        return bool(value.get('error')) or any(has_error(v) for v in value.values())
    if isinstance(value, list):
        return any(has_error(v) for v in value)
    return False


def geometry_and_vocabulary(source: Path):
    """Load CPU vocabulary and the exact frozen projection function only."""
    path = source / 'agents/evidence_nav.py'
    spec = importlib.util.spec_from_file_location('archive_audit_vocabulary', path)
    vocabulary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vocabulary)
    text = (source / 'agents/agent_vlm_engine.py').read_text()
    node, = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)
             and n.name == '_backproject_pixel_to_world']
    # The pinned shared launcher clears VM_* and uses these source defaults.
    context = {'np': np, 'math': math,
               'os': SimpleNamespace(environ={'VM_BACKPROJECT_DEPTH_HALF': '3'}),
               'INAV_CAMERA_HFOV_DEG': 90.}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), context)
    return vocabulary, context['_backproject_pixel_to_world']


def audit_episode(path: Path, vocabulary, project) -> dict:
    record = json.loads(path.read_text())
    trajectory = record['trajectory']
    if (record['step_cap'] != 30 or not 2 <= len(trajectory) <= 31 or
            [r['step'] for r in trajectory] != list(range(len(trajectory))) or
            any(r.get('action') == 'STOP' for r in trajectory[:-1])):
        raise ValueError('Action budget or STOP closure differs')
    flags = record['ablation_flags']
    if (record['evaluation_protocol'].get('automatic_start_reorientation')
            or flags.get('inav_mask_refinement')):
        raise ValueError('Require the declared unmasked single-view policy')
    metadata = record['observation_archive']
    if (metadata['schema'] != SCHEMA or metadata['observations'] != len(trajectory) - 1 or
            metadata['extra_render_or_inference_calls'] != 0):
        raise ValueError('Observation/action count differs')
    elapsed = metadata['engine_to_policy_termination_seconds']
    overhead = metadata['archive_seconds']
    if not (math.isfinite(elapsed) and math.isfinite(overhead) and 0 <= overhead <= elapsed):
        raise ValueError('Invalid recorded timing boundaries')
    expected_steps = range(1, len(trajectory))
    for pattern, suffix in [('step_*_perception.json', 'perception.json'),
                            ('step_*_policy_input.npz', 'policy_input.npz')]:
        if {p.name for p in path.parent.glob(pattern)} != {f'step_{i:02d}_{suffix}' for i in expected_steps}:
            raise ValueError('Missing or extra acquired observation files')
    query = record['plan']['target_guess']
    phrases = vocabulary.target_detector_queries(query)
    if flags['inav_target_detector_queries'] != phrases:
        raise ValueError('Detector query differs from frozen vocabulary')
    other_candidates = {c.lower().strip().replace('_', ' ') for c in record['plan']['candidate_objects']
                        if c.lower().strip() != query.lower().strip()}
    hashes = {str(path.resolve()): digest(path)}
    totals = {'observations': 0, 'proposals': 0, 'computed_verifications': 0,
              'not_verified_due_to_original_cap': 0, 'admitted_projections_replayed': 0,
              'input_bytes': 0}
    for previous, step in zip(trajectory, trajectory[1:]):
        index = step['step']
        trace_path = path.parent / f'step_{index:02d}_perception.json'
        trace = json.loads(trace_path.read_text())
        expected_keys = {'schema', 'step', 'pre_action_pose', 'candidates', 'admitted_evidence',
                         'perception_error', 'perception_errors', 'input', 'intrinsics', 'input_sha256'}
        if (set(trace) != expected_keys or trace['schema'] != SCHEMA or trace['step'] != index or
                trace['input'] != f'step_{index:02d}_policy_input.npz' or trace['intrinsics'] != INTRINSICS):
            raise ValueError('Unexpected observation schema or camera input')
        if trace['perception_error'] or trace['perception_errors'] or has_error(trace['candidates']):
            raise ValueError('Perception errors cannot be scored')
        pose = trace['pre_action_pose']
        if (set(pose) != {'position', 'yaw'} or
                not all(math.isfinite(v) for v in [*pose['position'], pose['yaw']]) or
                math.dist(pose['position'], previous['position']) > 1e-5 or
                abs(math.remainder(pose['yaw'] - previous['yaw'], 2 * math.pi)) > 1e-6):
            raise ValueError('Observation is not at its recorded pre-action pose')
        data_path = path.parent / trace['input']
        if digest(data_path) != trace['input_sha256']:
            raise ValueError('Acquired input hash differs')
        with np.load(data_path, allow_pickle=False) as arrays:
            if set(arrays.files) != {'rgb', 'depth'}:
                raise ValueError('Only acquired RGB-D arrays are allowed')
            rgb, depth = arrays['rgb'], arrays['depth']
        if (rgb.dtype != np.uint8 or rgb.shape != (768, 768, 3) or depth.shape != (768, 768) or
                not np.issubdtype(depth.dtype, np.floating)):
            raise ValueError('Unexpected acquired array shape or dtype')
        # The original engine saved this JPEG before candidate processing.
        encoded = io.BytesIO()
        Image.fromarray(rgb).save(encoded, 'JPEG', quality=70, optimize=True)
        jpeg_path = path.parent / f'step_{index:02d}_rgb.jpg'
        if encoded.getvalue() != jpeg_path.read_bytes():
            raise ValueError('Archived RGB differs from the original policy JPEG')
        candidates = trace['candidates']
        raw = step.get('dino') or []
        if len(candidates) != len(raw):
            raise ValueError('Archived proposal count differs from actual trajectory')
        strict_count = 0
        selected = []
        for i, (candidate, proposal) in enumerate(zip(candidates, raw)):
            if (candidate['index'] != i or any(candidate[k] != proposal[k] for k in ['label', 'score', 'bbox']) or
                    proposal['detector_source'] != 'dino'):
                raise ValueError('Archived proposal differs from actual trajectory')
            label = candidate['label'].lower().strip().replace('_', ' ')
            strict = label not in other_candidates and vocabulary.target_query_match(label, phrases)
            if candidate['strict_label_match'] != strict:
                raise ValueError('Archived strict-candidate filtering differs')
            computed = strict and strict_count < 5
            expected_status = 'label_filtered' if not strict else 'computed' if computed else 'not_selected_for_verification'
            if candidate['verification_status'] != expected_status:
                raise ValueError('Original five-box verification budget differs')
            if computed != isinstance(candidate['verification'], dict):
                raise ValueError('Computed verification is missing or extra')
            strict_count += int(strict)
            totals['computed_verifications'] += int(computed)
            totals['not_verified_due_to_original_cap'] += int(strict and not computed)
            if candidate['selected']:
                if not computed:
                    raise ValueError('Unverified candidate selected by diagnostic policy')
                selected.append(candidate)
        if len(selected) > 1:
            raise ValueError('Multiple candidates selected')
        evidence = step.get('current_dino_evidence') or {}
        if trace['admitted_evidence'] != evidence:
            raise ValueError('Archived admission differs from actual trajectory')
        if evidence:
            if len(selected) != 1:
                raise ValueError('Admitted evidence has no selected proposal')
            box = selected[0]['bbox']
            y_fraction = .5  # Pinned launcher's unchanged VM_DINO_BBOX_Y_FRAC default.
            xy = project((box[0] + box[2]) / 2 / 768,
                         (box[1] * (1 - y_fraction) + box[3] * y_fraction) / 768, depth, pose)
            if xy is None or not all(math.isfinite(v) for v in [*xy, *evidence['xy']]) or math.dist(xy, evidence['xy']) > 1e-5:
                raise ValueError('Admitted projection differs from acquired depth and calibrated camera')
            verification = evidence['clip_verification']
            if any(verification.get(k) != v for k, v in selected[0]['verification'].items()):
                raise ValueError('Selected verification differs from admitted evidence')
            totals['admitted_projections_replayed'] += 1
        totals['observations'] += 1
        totals['proposals'] += len(candidates)
        totals['input_bytes'] += data_path.stat().st_size
        for p in [trace_path, data_path, jpeg_path]:
            hashes[str(p.resolve())] = digest(p)
    if metadata['input_bytes'] != totals['input_bytes']:
        raise ValueError('Recorded archive byte total differs')
    return {**totals, 'source_hashes': hashes}


def audit_run(run: Path, source: Path) -> dict:
    if not (run / '.RUN_SUCCESS').is_file():
        raise ValueError('Require actual successful run closure before auditing')
    result = audit_shared(run, run / 'shared_input_snapshot', 'r055', source)
    if (run / 'simulator_exit_code.txt').read_text().strip() != '0':
        raise ValueError('Actual simulator exit was not zero')
    for p in source.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc':
            snapshot = run / 'source_snapshot/eval' / p.relative_to(source)
            if not snapshot.is_file() or digest(snapshot) != digest(p):
                raise ValueError(f'Actual launcher source archive differs: {p}')
            result['source_hashes'][str(snapshot.resolve())] = digest(snapshot)
            result['source_hashes'][str(p.resolve())] = digest(p)
    output_manifest = run / 'output_manifest.sha256'
    listed = set()
    for line in output_manifest.read_text().splitlines():
        expected, filename = line.split('  ', 1)
        path = Path(filename)
        if not path.is_absolute() or path.resolve() in listed:
            raise ValueError('Output manifest contains relative or duplicate paths')
        listed.add(path.resolve())
        if digest(path) != expected:
            raise ValueError(f'Closed runtime output changed: {filename}')
    expected_files = {p.resolve() for folder in ['episodes', 'shared_input_snapshot']
                      for p in (run / folder).rglob('*') if p.is_file()}
    if listed != expected_files:
        raise ValueError('Output manifest does not cover exactly the closed runtime files')
    vocabulary, project = geometry_and_vocabulary(source)
    totals, hashes = {}, dict(result['source_hashes'])
    for p in sorted((run / 'episodes').rglob('record.json')):
        episode = audit_episode(p, vocabulary, project)
        hashes.update(episode.pop('source_hashes'))
        for key, value in episode.items():
            totals[key] = totals.get(key, 0) + value
    hashes[str(Path(__file__).resolve())] = digest(Path(__file__))
    for p in [output_manifest, run / 'simulator_exit_code.txt']:
        hashes[str(p.resolve())] = digest(p)
    return {'passed': True, 'records': result['records'], **totals,
            'shared_input_mode': result['input_mode'], 'source_hashes': hashes,
            'scope': 'Archive/query/action/geometry consistency; no visibility, model-accuracy or navigation-gain claim.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Archive audit must run with CUDA hidden')
    result = audit_run(args.run, args.source)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'source_hashes'}))


if __name__ == '__main__':
    main()
