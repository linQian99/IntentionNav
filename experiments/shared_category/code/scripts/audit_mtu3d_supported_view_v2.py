"""Replay paid supported-view navigation and verify its acquired camera sources."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np

from audit_mtu3d_acquired_history import audit_frontier_history_run
from audit_mtu3d_view_consumption import audit_consumption
from hash_navigation_inputs import digest_file
from run_visible_navigation_diagnostics import verify_sources


def verify_support(support, requests, decisions, current_index, cache, describe_view, volume_fn):
    """Check actual consumed view identity, camera and four frustum witnesses.

    Membership in a learned mask is supplied by the frozen merge observer. This
    CPU audit neither reruns segmentation nor certifies the semantic category.
    """
    index, view_index = support['decision_index'], support['view_index']
    if type(index) is not int or type(view_index) is not int or not 0 <= index <= current_index:
        raise ValueError('Future or malformed source decision')
    request, decision = requests[index], decisions[index]
    views = request['context_observations'] + [request]
    if view_index not in decision['valid_view_indices']:
        raise ValueError('Object support uses an unconsumed view')
    identity = (index, view_index)
    if identity not in cache:
        view = views[view_index]
        with np.load(view['observation'], allow_pickle=False) as arrays:
            cache[identity] = describe_view(view | {'rgb': arrays['rgb'], 'depth': arrays['depth']}, index, view_index)
    if any(support.get(k) != v for k, v in cache[identity].items()):
        raise ValueError('Support camera/RGB-D identity differs from its acquired source')
    volume = support['view_volume']
    witnesses = np.asarray(volume['plane_witness_points_xyz'], dtype=float)
    if witnesses.shape != (4, 3) or not np.isfinite(witnesses).all():
        raise ValueError('Invalid surface witnesses')
    reconstructed = volume_fn(witnesses, support)
    for key in ['plane_normals', 'plane_lower_bounds', 'forward', 'source_yaw']:
        value = np.asarray(volume[key], dtype=float)
        if not np.isfinite(value).all() or not np.allclose(value, reconstructed[key], atol=1e-6, rtol=0):
            raise ValueError(f'Surface plane witness mismatch: {key}')
    minimum = volume['minimum_surface_forward']
    if (not math.isfinite(minimum) or minimum > reconstructed['minimum_surface_forward'] + 1e-6
            or volume['surface_points'] != support['sampled_mask_points']
            or type(support['sampled_mask_points']) is not int or support['sampled_mask_points'] <= 0):
        raise ValueError('Invalid learned support count or near plane')
    relative = witnesses - np.asarray(support['position'])
    forward = np.asarray(reconstructed['forward'])
    axial = relative @ forward
    intr = support['intrinsics']
    pixels = np.column_stack([intr['cx'] + intr['fx'] * (relative @ np.cross(forward, [0., 0., 1.])) / axial,
                              intr['cy'] - intr['fy'] * relative[:, 2] / axial])
    if (np.any(axial <= 0) or not np.isfinite(pixels).all() or pixels.min() < -.01
            or np.any(pixels[:, 0] > intr['width'] - 1 + .01)
            or np.any(pixels[:, 1] > intr['height'] - 1 + .01)):
        raise ValueError('Surface witnesses do not project to the actual source image')


def verify_pose_transition(previous, step):
    """Reject free terminal turns and poses inconsistent with executed moves."""
    for pose in [previous, step]:
        if (len(pose['position']) != 3 or len(pose['look_dir']) != 3
                or not all(math.isfinite(v) for v in [*pose['position'], *pose['look_dir'], pose['yaw']])
                or math.dist(pose['look_dir'], [math.cos(pose['yaw']), math.sin(pose['yaw']), 0.]) > 1e-6):
            raise ValueError('Inconsistent or nonfinite camera orientation')
    turn = math.atan2(math.sin(step['yaw'] - previous['yaw']), math.cos(step['yaw'] - previous['yaw']))
    if step['action'] == 'STOP' and abs(turn) > 1e-6:
        raise ValueError('STOP cannot rotate the camera')
    if step['action'] == 'MOVE':
        dx = step['position'][0] - previous['position'][0]
        dy = step['position'][1] - previous['position'][1]
        if abs(dx) + abs(dy) <= 1e-6:
            raise ValueError('MOVE must have a nonzero displacement')
        expected = math.atan2(dy, dx)
        if abs(math.atan2(math.sin(step['yaw'] - expected), math.cos(step['yaw'] - expected))) > 1e-6:
            raise ValueError('MOVE camera must face the executed displacement')
    elif step['action'] in {'ROTATE_SCAN', 'ROTATE_TARGET'}:
        if abs(turn) > math.pi / 2 + 1e-6:
            raise ValueError('Paid camera turn exceeds90degrees')
    elif step['action'] != 'STOP':
        raise ValueError('Unexpected navigation action')


def audit_pose_record(path, source):
    sys.path[:0] = [str(source / 'agents'), str(source / 'simulator')]
    from mtu3d_observation_support import describe_view
    from mtu3d_supported_view import plan_supported_view, terminal_view_action, view_volume
    from mtu3d_strict_grid import strict_segment, strict_path_action
    from walkable_map import WalkableMap
    record = json.loads(path.read_text())
    if record['run_config']['object_goal_mode'] != 'observed_pose':
        raise ValueError('Unexpected object controller')
    wm = WalkableMap.load(record['scene_id'], metaroot=Path('/path/to/workspace/datasets/vlntube/TaTaMeta/metadata_train'))
    requests, decisions, cache = {}, {}, {}
    result = {'selection_id': record['selection_id'], 'commitments': 0, 'support_records': 0,
              'strict_moves': 0, 'object_moves': 0, 'paid_target_rotations': 0,
              'supported_stops': 0, 'unsupported_route_rotations': 0}
    committed = False
    plan = None
    for previous, step in zip(record['trajectory'], record['trajectory'][1:]):
        verify_pose_transition(previous, step)
        decision = step.get('policy_decision')
        position, yaw = previous['position'], previous['yaw']
        if decision:
            if committed:
                raise ValueError('Replanned while an object commitment remains active')
            index = decision['decision_index']
            requests[index] = json.loads((path.parent / f"request_{step['step']:02d}.json").read_text())['request']
            decisions[index] = decision
            if decision['is_object_decision']:
                supports = decision['stage2_trace']['selected_observation_support']
                for support in supports:
                    verify_support(support, requests, decisions, index, cache, describe_view, view_volume)
                    result['support_records'] += 1
                plan = plan_supported_view(wm, position, decision['target_position'], supports)
                if plan != decision['object_approach_plan']:
                    raise ValueError('Actual supported-view plan differs from deterministic replay')
                committed = True
                result['commitments'] += 1
        if step['action'] == 'MOVE':
            if not strict_segment(wm, position[:2], step['position'][:2]):
                raise ValueError('Movement touches a blocked full-segment grid cell')
            result['strict_moves'] += 1
        if committed:
            expected = terminal_view_action(position, yaw, plan)
            action = expected['action']
            if action == 'MOVE':
                endpoint = strict_path_action(wm, position[:2], plan['navigation_endpoint_xy'])
                action = 'MOVE' if endpoint is not None else 'NO_SUPPORTED_ROUTE'
                if endpoint is not None:
                    if math.dist(endpoint, step['position'][:2]) > 1e-6:
                        raise ValueError('Executed object movement differs from strict route')
                    result['object_moves'] += 1
            if action == 'NO_SUPPORTED_ROUTE':
                action = 'ROTATE_SCAN'
                turn = math.atan2(math.sin(step['yaw'] - yaw), math.cos(step['yaw'] - yaw))
                if abs(turn - math.pi / 2) > 1e-6:
                    raise ValueError('Wrong paid no-route turn')
                committed, plan = False, None
                result['unsupported_route_rotations'] += 1
            elif action == 'ROTATE_TARGET':
                delta = math.atan2(math.sin(step['yaw'] - expected['target_yaw']), math.cos(step['yaw'] - expected['target_yaw']))
                if abs(delta) > 1e-6 or step['object_approach_plan'] != plan:
                    raise ValueError('Wrong paid terminal orientation')
                result['paid_target_rotations'] += 1
            elif action == 'STOP':
                if (step.get('object_stop_criterion') != 'observed_surface_view_pose'
                        or step.get('object_approach_plan') != plan
                        or record['stop_reason'] != 'model_object_observed_view_reached'):
                    raise ValueError('Unjustified terminal pose STOP')
                result['supported_stops'] += 1
            if step['action'] != action:
                raise ValueError(f'Expected paid action {action}, recorded {step["action"]}')
        elif step['action'] in {'STOP', 'ROTATE_TARGET'}:
            raise ValueError('Terminal action without object evidence')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    verify_sources(manifest)
    actual = {'numpy_version': np.__version__, 'numpy_file': str(Path(np.__file__).resolve()),
              'opencv_version': cv2.__version__, 'opencv_file': str(Path(cv2.__file__).resolve())}
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or actual != manifest['audit_runtime']:
        raise ValueError('Require exact Isaac CPU libraries for recorded depth/frontier replay')
    job = next(j for j in manifest['jobs'] if j['id'] == args.job)
    run, source = Path(job['output']), Path(manifest['source'])
    summary = json.loads((run / 'summary.json').read_text())
    runtime = json.loads((run / 'run_manifest.json').read_text())
    if (not summary['complete'] or summary['failed'] or summary['expected'] != job['count']
            or len(summary['completed']) != job['count'] or not (run / '.RUN_SUCCESS').is_file()
            or runtime['object_goal_mode'] != 'observed_pose'):
        raise ValueError('Incomplete runtime closure')
    for name, expected in runtime['inputs'].items():
        if manifest['source_hashes'].get(name) != expected:
            raise ValueError(f'Unpinned or changed runtime input: {name}')
    required = ['agent_mtu3d.py', 'mtu3d_strict_grid.py', 'mtu3d_supported_view.py', 'mtu3d_observation_support.py']
    if any(str(source / 'agents' / name) not in runtime['inputs'] for name in required):
        raise ValueError('New controller source absent from actual run manifest')
    result = audit_frontier_history_run(run, source, job['exploration_mode'])
    if result['records'] != job['count']:
        raise ValueError('Incomplete six-case diagnostic')
    consumed = audit_consumption(run, result)
    result['source_hashes'].update(consumed['source_hashes'])
    result['view_consumption'] = {k: v for k, v in consumed.items() if k != 'source_hashes'}
    result['supported_view'] = [audit_pose_record(path, source) for path in sorted((run / 'episodes').rglob('record.json'))]
    result['audit_runtime'] = actual
    for path in [Path(__file__), run / 'summary.json', run / 'run_manifest.json', run / '.RUN_SUCCESS']:
        result['source_hashes'][str(path.resolve())] = digest_file(path)[0]
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_hashes'}))


if __name__ == '__main__':
    main()
