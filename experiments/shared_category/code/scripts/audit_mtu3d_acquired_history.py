"""Audit one paid initial scan and recent acquired-view reuse independently.

The ordinary frontier replay is preserved below with a schedule-specific
input gate. Existing endpoint and depth-consumption auditors remain separate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np

from audit_mtu3d_object_endpoint import audit_endpoints
from audit_mtu3d_view_consumption import audit_consumption
from hash_navigation_inputs import digest_file


def audit_history_inputs(run: Path) -> dict:
    result={'records':0,'model_calls':0,'four_view_calls':0,'scans_started':0,
            'scans_completed':0,'source_hashes':{},'passed':False}
    def pin(path):
        result['source_hashes'][str(path.resolve())]=digest_file(path)[0]
    for path in sorted((run/'episodes').rglob('record.json')):
        record=json.loads(path.read_text());steps=record['trajectory'];meta=record['model_meta']
        episode_key=f"{record['scene_id']}/{record['selection_id']}"
        if (meta.get('randomness_mode')!='per_decision_v1'
                or meta.get('episode_randomness_key')!=episode_key):
            raise ValueError('Missing isolated episode randomness')
        if (record['run_config']['scan_schedule']!='initial_then_history'
                or record['run_config']['commit_scan']!='multiview' or record['step_cap']!=30):
            raise ValueError('Unexpected acquired-history protocol')
        if [s['step'] for s in steps]!=list(range(len(steps))) or not 5<=len(steps)<=31:
            raise ValueError('Noncontiguous, incomplete or over-budget actions')
        if any(s['action']=='STOP' for s in steps[:-1]):raise ValueError('Post-STOP action')
        scan_steps=[s['step'] for s in steps if s.get('commit_scan_phase')]
        if scan_steps!=[1,2,3] or [steps[s]['commit_scan_phase'] for s in scan_steps]!=['begin','acquire','acquire']:
            raise ValueError('Not exactly one three-action initial scan')
        for index in scan_steps:
            previous,current=steps[index-1],steps[index]
            turn=math.atan2(math.sin(current['yaw']-previous['yaw']),math.cos(current['yaw']-previous['yaw']))
            if (current['action']!='ROTATE_SCAN' or current.get('policy_decision') is not None
                    or abs(turn-math.pi/2)>1e-6 or math.dist(previous['position'],current['position'])>1e-6):
                raise ValueError('Initial scan has an uncharged view, movement or wrong turn')
        confirmations=[s['step'] for s in steps if (s.get('policy_decision') or {}).get('commit_scan_confirmation')]
        if (confirmations!=[4] or meta['commit_scans_started']!=1 or meta['commit_scans_completed']!=1
                or meta['commit_scan_incomplete_at_budget']):
            raise ValueError('Wrong scan completion or an extra commitment scan')
        acquisitions=sorted(path.parent.glob('obs_*.npz'))
        if [int(p.stem.split('_')[1]) for p in acquisitions]!=list(range(1,len(steps))):
            raise ValueError('Extra, missing or unpaid observations')
        # Pin all acquired frames, including ones never selected into a call.
        for observation in acquisitions:pin(observation)
        request_paths=sorted(path.parent.glob('request_*.json'))
        decision_steps=[s['step'] for s in steps if s.get('policy_decision')]
        if ([int(p.stem.split('_')[1]) for p in request_paths]!=decision_steps
                or len(request_paths)!=meta['decision_calls'] or not request_paths):
            raise ValueError('Model request/decision closure mismatch')
        for index,request_path in enumerate(request_paths):
            saved=json.loads(request_path.read_text());request=saved['request']
            step=int(request_path.stem.split('_')[1]);decision=steps[step]['policy_decision']
            if (set(request)!={'observation','position','look_dir','intrinsics','frontiers','query','decision_index','context_observations'}
                    or request['decision_index']!=index or decision.get('decision_index')!=index
                    or request['query']!=record['prediction']['target']):
                raise ValueError('Unexpected query, privileged input or decision identity')
            payload=json.dumps(['mtu3d-decision-v1',record['run_config']['seed'],episode_key,index],
                               separators=(',',':'),ensure_ascii=True).encode()
            expected_seed=int.from_bytes(hashlib.sha256(payload).digest()[:4],'big')
            randomness=decision.get('randomness',{})
            if (randomness.get('mode')!='per_decision_v1' or randomness.get('episode_key')!=episode_key
                    or randomness.get('seed')!=expected_seed):
                raise ValueError('Wrong per-decision seed or episode randomness identity')
            expected_steps=list(range(step-3,step+1))
            contexts=request['context_observations'];views=[*contexts,request]
            if len(views)!=4 or decision.get('acquired_history_steps')!=expected_steps:
                raise ValueError('History is not the four most recent acquired views')
            hashes={request['observation']:saved['observation_sha256'],**saved['context_observation_sha256']}
            if len(hashes)!=4 or set(hashes)!={v['observation'] for v in views}:
                raise ValueError('Missing or duplicate view provenance')
            for source_step,view in zip(expected_steps,views):
                expected_path=(path.parent/f'obs_{source_step:02d}.npz').resolve()
                if Path(view['observation'])!=expected_path:
                    raise ValueError('History contains a foreign, stale, duplicated or future view')
                previous=steps[source_step-1]
                if (math.dist(view['position'],previous['position'])>1e-6
                        or math.dist(view['look_dir'],previous['look_dir'])>1e-6
                        or view['intrinsics']!={'width':768,'height':768,'fx':384.,'fy':384.,'cx':384.,'cy':384.}):
                    raise ValueError('Historical view pose or actual camera intrinsics mismatch')
                if result['source_hashes'][str(expected_path)]!=hashes[str(expected_path)]:
                    raise ValueError('An acquired view changed after submission')
            if decision.get('observations_supplied')!=4:
                raise ValueError('Actual worker did not receive four views')
            if decision.get('decision_source')=='mtu3d_learned' and not decision.get('stage2_trace'):
                raise ValueError('Missing actual learned decision evidence')
            decision_path=request_path.with_name(request_path.name.replace('request_','decision_',1))
            actual=json.loads(decision_path.read_text())
            if actual['request']!=request or actual['decision']!=decision:
                raise ValueError('Actual request archive, decision copy and record differ')
            pin(request_path);pin(decision_path)
            result['model_calls']+=1;result['four_view_calls']+=1
        pin(path);result['records']+=1;result['scans_started']+=1;result['scans_completed']+=1
    result['passed']=bool(result['records'])
    return result


def audit_frontier_history_run(run: Path, source: Path, mode: str) -> dict:
    result = audit_history_inputs(run)
    sys.path[:0] = [str(source / 'agents'), str(source / 'simulator')]
    from mtu3d_frontier_memory import FrontierMemory
    from walkable_map import WalkableMap
    result.update(frontier_selections=0, frontier_arrivals=0, frontier_route_failures=0,
                  continued_frontier_moves=0, boundary_requests_replayed=0)
    for path in sorted((run / 'episodes').rglob('record.json')):
        record = json.loads(path.read_text())
        if record['run_config']['exploration_mode'] != mode:
            raise ValueError('Wrong exploration mode')
        if record['run_config']['scan_schedule'] != 'initial_then_history':
            raise ValueError('Wrong scan schedule')
        meta = Path('/path/to/workspace/datasets/vlntube/TaTaMeta/metadata_train')
        wm = WalkableMap.load(record['scene_id'], metaroot=meta)
        memory = FrontierMemory(wm) if mode == 'boundary_commit' else None
        steps = record['trajectory']
        if len(list(path.parent.glob('obs_*.npz'))) != len(steps) - 1:
            raise ValueError('Observation acquisition count does not match paid actions')
        target = None
        counts = dict(frontier_selections=0, frontier_arrivals=0, frontier_route_failures=0)
        for previous, step in zip(steps, steps[1:]):
            if not wm.is_walkable(*step['position'][:2]):
                raise ValueError('Action reached an unwalkable cell')
            if step['action'] == 'MOVE':
                if math.dist(previous['position'], step['position']) > 1.7 + 1e-6:
                    raise ValueError('Move exceeds the shared 1.7m action limit')
                if not wm.line_of_sight(*previous['position'][:2], *step['position'][:2]):
                    raise ValueError('Move crossed a blocked grid segment')
            elif math.dist(previous['position'], step['position']) > 1e-6:
                raise ValueError('Nonmovement action changed position')
            if memory is None:
                continue
            observed = memory.observe(previous['position'], previous['look_dir'])
            if target is not None and math.dist(previous['position'][:2], target) <= memory.arrival_radius:
                target = None
                counts['frontier_arrivals'] += 1
            decision = step.get('policy_decision')
            if decision:
                if target is not None:
                    raise ValueError('Model replanned before reaching the committed frontier')
                request_path = path.parent / f"request_{step['step']:02d}.json"
                request = json.loads(request_path.read_text())['request']
                expected = memory.proposals(previous['position'])
                actual = request['frontiers']
                if len(actual) != len(expected) or any(math.dist(a, b) > 1e-6 for a, b in zip(actual, expected)):
                    raise ValueError('Frontiers do not replay from acquired view poses')
                state = decision['frontier_state']
                if any(state[k] != v for k, v in observed.items()) or state['proposal_count'] != len(expected):
                    raise ValueError('Frontier observation provenance does not reconcile')
                if not decision['is_object_decision'] and decision.get('action_hint') != 'ROTATE_SCAN':
                    target = decision['target_position'][:2]
                    if not any(math.dist(target, p[:2]) <= 1e-4 for p in expected):
                        raise ValueError('Unproposed exploration target')
                    memory.visit(target)
                    counts['frontier_selections'] += 1
                if state['visited_frontiers'] != len(memory.visited):
                    raise ValueError('Visited frontier count mismatch')
                result['boundary_requests_replayed'] += 1
                result['source_hashes'][str(request_path.resolve())] = hashlib.sha256(request_path.read_bytes()).hexdigest()
            elif target is not None and step['action'] == 'MOVE':
                result['continued_frontier_moves'] += 1
            if target is not None and step['action'] == 'ROTATE_SCAN':
                if step.get('commit_scan_phase'):
                    raise ValueError('A new scan interrupted committed exploration')
                target = None
                counts['frontier_route_failures'] += 1
        if record['model_meta']['frontier_acquired_views'] != (memory.observations if memory is not None else 0):
            raise ValueError('Acquired visibility count mismatch')
        for key, value in counts.items():
            if record['model_meta'][key] != value:
                raise ValueError(f'Frontier execution counter mismatch: {key}')
            result[key] += value
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--job',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();manifest=json.loads(args.manifest.read_text())
    actual={'numpy_version':np.__version__,'numpy_file':str(Path(np.__file__).resolve()),
            'opencv_version':cv2.__version__,'opencv_file':str(Path(cv2.__file__).resolve())}
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or actual!=manifest['audit_runtime']:
        raise ValueError('Require the exact Isaac CPU replay environment')
    job=next(j for j in manifest['jobs'] if j['id']==args.job)
    run=Path(job['output']);source=Path(manifest['source'])
    result=audit_frontier_history_run(run,source,job['exploration_mode'])
    endpoints=audit_endpoints(run,source)
    if result['records']!=job['count'] or endpoints['records']!=job['count']:
        raise ValueError('Incomplete acquired-history run')
    consumed=audit_consumption(run,result)
    result['view_consumption']={k:v for k,v in consumed.items() if k!='source_hashes'}
    result['source_hashes'].update(consumed['source_hashes'])
    result.update(object_endpoints=endpoints,audit_runtime=actual)
    result['source_hashes'][str(Path(__file__).resolve())]=digest_file(Path(__file__))[0]
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'}))


if __name__=='__main__':main()
