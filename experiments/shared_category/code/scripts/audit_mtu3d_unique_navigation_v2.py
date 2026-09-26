"""Audit accepted-surface approaches, real sample pixels/depth, and fresh STOP."""
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
from audit_mtu3d_unique_evidence import audit_record as audit_unique_record
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval/aggregate'))
from mtu3d_support_precision import verify_support
from audit_mtu3d_supported_view_v2 import verify_pose_transition
from audit_mtu3d_view_consumption import audit_consumption
from hash_navigation_inputs import digest_file
from run_visible_navigation_diagnostics import verify_sources


def audit_pose_record(path, source, wm=None):
    sys.path[:0] = [str(source / 'agents'), str(source / 'simulator')]
    from mtu3d_observation_support import describe_view
    from mtu3d_surface_sample import verify_sample_depth
    from mtu3d_supported_view import view_volume
    from mtu3d_fresh_goal import selected_box, fresh_stop_evidence, plan_observed_goal, approach_action
    from mtu3d_strict_grid import strict_segment, strict_path_action
    from walkable_map import WalkableMap
    record = json.loads(path.read_text())
    if record['run_config']['object_goal_mode'] != 'fresh_surface_pose':
        raise ValueError('Wrong fresh controller')
    if wm is None:
        wm = WalkableMap.load(record['scene_id'], metaroot=Path('/path/to/workspace/datasets/vlntube/TaTaMeta/metadata_train'))
    requests, decisions, cache = {}, {}, {}
    counts = dict(selection_id=record['selection_id'], commitments=0, fresh_checks=0,
                  current_evidence_stops=0, strict_moves=0, object_moves=0,
                  paid_target_turns=0, unsupported_or_unverified_scans=0, support_records=0)
    committed, plan = False, None
    for previous, step in zip(record['trajectory'], record['trajectory'][1:]):
        # Canonical paid turn has the same <=90deg pose semantics as old turn.
        normalized = {**step, 'action': 'ROTATE_TARGET' if step['action']=='ROTATE_TARGET_RECENTER' else step['action']}
        verify_pose_transition(previous, normalized)
        position, yaw = previous['position'], previous['yaw']
        reobserve = committed and approach_action(position, yaw, plan)['action'] == 'REOBSERVE'
        if reobserve:
            committed, plan = False, None
            if not step.get('policy_decision'):
                raise ValueError('Arrived view lacks the required new model call')
        decision = step.get('policy_decision')
        fresh = None
        if decision:
            if committed:
                raise ValueError('Model called while an unfinished object approach remains active')
            index = decision['decision_index']
            request = json.loads((path.parent / f"request_{step['step']:02d}.json").read_text())['request']
            requests[index], decisions[index] = request, decision
            if decision['is_object_decision']:
                supports = decision['stage2_trace']['selected_observation_support']
                for support in supports:
                    verify_support(support, requests, decisions, index, cache, describe_view, view_volume)
                    source_request = requests[support['decision_index']]
                    source_view = (source_request['context_observations'] + [source_request])[support['view_index']]
                    with np.load(source_view['observation'], allow_pickle=False) as source_arrays:
                        verify_sample_depth(support, source_arrays['depth'])
                    counts['support_records'] += 1
                with np.load(request['observation'], allow_pickle=False) as arrays:
                    fresh = fresh_stop_evidence(decision, request, arrays['rgb'], arrays['depth'])
                if fresh != decision['fresh_stop_evidence']:
                    raise ValueError('Current STOP evidence differs from actual RGB-D replay')
                plan = plan_observed_goal(wm, position, selected_box(decision), supports)
                if plan != decision['object_approach_plan']:
                    raise ValueError('Observed approach plan differs from deterministic replay')
                committed = True
                counts['commitments'] += 1
                counts['fresh_checks'] += 1
        if step['action'] == 'MOVE':
            if not strict_segment(wm, position[:2], step['position'][:2]):
                raise ValueError('Movement touches blocked full-segment cells')
            counts['strict_moves'] += 1
        if not committed:
            if step['action'] in {'STOP', 'ROTATE_TARGET_RECENTER'}:
                raise ValueError('Terminal action without a learned object commitment')
            continue
        if fresh and fresh['passed']:
            if (step['action'] != 'STOP' or record['stop_reason'] != 'fresh_current_object_verified'
                    or step.get('object_stop_criterion') != 'fresh_current_learned_mask_and_goal_region'
                    or step.get('object_approach_plan') != plan):
                raise ValueError('Fresh STOP record is inconsistent')
            counts['current_evidence_stops'] += 1
            continue
        expected = approach_action(position, yaw, plan)
        action = expected['action']
        if action == 'MOVE':
            endpoint = strict_path_action(wm, position[:2], plan['navigation_endpoint_xy'])
            if endpoint is None:
                action = 'NO_SUPPORTED_ROUTE'
            elif math.dist(endpoint, step['position'][:2]) > 1e-6:
                raise ValueError('Object MOVE differs from strict route')
            else:
                counts['object_moves'] += 1
        if action in {'NO_SUPPORTED_ROUTE', 'REOBSERVE'}:
            action = 'ROTATE_SCAN'
            turn = math.atan2(math.sin(step['yaw'] - yaw), math.cos(step['yaw'] - yaw))
            if abs(turn - math.pi / 2) > 1e-6:
                raise ValueError('Wrong unsupported/unverified paid scan')
            counts['unsupported_or_unverified_scans'] += 1
            committed, plan = False, None
        elif action == 'ROTATE_TARGET_RECENTER':
            turn = math.atan2(math.sin(step['yaw'] - expected['target_yaw']), math.cos(step['yaw'] - expected['target_yaw']))
            if abs(turn) > 1e-6 or step.get('object_approach_plan') != plan:
                raise ValueError('Wrong paid recenter orientation')
            counts['paid_target_turns'] += 1
        if step['action'] != action:
            raise ValueError(f'Expected {action}; got {step["action"]}; no fresh evidence means no STOP')
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    verify_sources(manifest)
    actual = dict(numpy_version=np.__version__, numpy_file=str(Path(np.__file__).resolve()),
                  opencv_version=cv2.__version__, opencv_file=str(Path(cv2.__file__).resolve()))
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or actual != manifest['audit_runtime']:
        raise ValueError('Require exact Isaac CPU replay libraries')
    job = next(j for j in manifest['jobs'] if j['id'] == args.job)
    run, source = Path(job['output']), Path(manifest['source'])
    summary = json.loads((run / 'summary.json').read_text())
    runtime = json.loads((run / 'run_manifest.json').read_text())
    if (not summary['complete'] or summary['failed'] or summary['expected'] != job['count']
            or len(summary['completed']) != job['count'] or not (run / '.RUN_SUCCESS').is_file()
            or runtime['object_goal_mode'] != 'fresh_surface_pose'):
        raise ValueError('Incomplete actual runtime')
    for name, expected in runtime['inputs'].items():
        if manifest['source_hashes'].get(name) != expected:
            raise ValueError(f'Unpinned runtime input: {name}')
    for name in ['agent_mtu3d.py', 'mtu3d_fresh_goal.py', 'mtu3d_observation_support.py', 'mtu3d_strict_grid.py', 'mtu3d_surface_sample.py', 'mtu3d_unique_evidence.py', 'mtu3d_memory_relations.py', 'mtu3d_adapter.py']:
        if str(source / 'agents' / name) not in runtime['inputs']:
            raise ValueError('Missing fresh controller source in actual runtime manifest')
    result = audit_frontier_history_run(run, source, job['exploration_mode'])
    if result['records'] != job['count']:
        raise ValueError('Incomplete diagnostic case coverage')
    consumption = audit_consumption(run, result)
    result['source_hashes'].update(consumption['source_hashes'])
    result['view_consumption'] = {k:v for k,v in consumption.items() if k != 'source_hashes'}
    result['fresh_goal'] = [audit_pose_record(p, source) for p in sorted((run / 'episodes').rglob('record.json'))]
    result['unique_evidence'] = [audit_unique_record(p, source) for p in sorted((run / 'episodes').rglob('record.json'))]
    if sum(r['model_calls'] for r in result['unique_evidence']) != result['model_calls']:
        raise ValueError('Incomplete integration accounting')
    result['audit_runtime'] = actual
    result['projection_validation'] = dict(version='coordinate_precision_v2',
        coordinate_bound_source=str(source/'agents/mtu3d_projection_precision.py'),
        policy_changed=False, goal_radius_changed=False)
    for path in [Path(__file__).resolve().parents[1]/'eval/aggregate/mtu3d_support_precision.py',
                 source/'agents/mtu3d_projection_precision.py']:
        result['source_hashes'][str(path.resolve())] = digest_file(path)[0]
    for path in [Path(__file__), run/'summary.json', run/'run_manifest.json', run/'.RUN_SUCCESS']:
        result['source_hashes'][str(path.resolve())] = digest_file(path)[0]
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k:v for k,v in result.items() if k != 'source_hashes'}))


if __name__ == '__main__':
    main()
