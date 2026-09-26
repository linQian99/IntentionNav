"""Reconcile charged scan actions, acquired views and real worker decisions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import math


def audit_run(run: Path, mode: str) -> dict:
    records = sorted((run / 'episodes').rglob('record.json'))
    result = {'records': len(records), 'scans_started': 0, 'scans_completed': 0,
              'model_calls': 0, 'four_view_calls': 0, 'empty_fallbacks': 0,
              'source_hashes': {}}
    for path in records:
        record = json.loads(path.read_text())
        steps = record['trajectory']
        if record['run_config']['commit_scan'] != mode or record['step_cap'] != 30:
            raise ValueError('Unexpected experiment mode or budget')
        if [s['step'] for s in steps] != list(range(len(steps))) or len(steps) > 31:
            raise ValueError('Noncontiguous or out-of-budget actions')
        if any(s['action'] == 'STOP' for s in steps[:-1]):
            raise ValueError('Post-STOP action')
        requests = sorted(path.parent.glob('request_*.json'))
        if len(requests) != record['model_meta']['decision_calls']:
            raise ValueError('Worker request count mismatch')
        begins = {s['step'] for s in steps if s.get('commit_scan_phase') == 'begin'}
        confirmations = {s['step'] for s in steps
                         if (s.get('policy_decision') or {}).get('commit_scan_confirmation')}
        for step in confirmations:
            if step - 3 not in begins:
                raise ValueError('Confirmation lacks a scan beginning')
            prior = steps[step-3:step]
            if any(s['action'] != 'ROTATE_SCAN' for s in prior):
                raise ValueError('Confirmation did not pay three rotation actions')
            for s in prior:
                previous = steps[s['step']-1]
                if math.dist(s['position'], previous['position']) > 1e-6:
                    raise ValueError('Scan rotation moved the agent')
                turn = math.atan2(math.sin(s['yaw']-previous['yaw']),
                                  math.cos(s['yaw']-previous['yaw']))
                if abs(turn-math.pi/2) > 1e-6:
                    raise ValueError('Scan rotation was not ninety degrees')
        for request_path in requests:
            row = json.loads(request_path.read_text()); request = row['request']
            index = int(request_path.stem.split('_')[-1])
            if set(request) - {'observation','position','look_dir','intrinsics',
                               'frontiers','query','decision_index','context_observations'}:
                raise ValueError('Unexpected policy input fields')
            expected_observation = (path.parent/f'obs_{index:02d}.npz').resolve()
            if Path(request['observation']) != expected_observation:
                raise ValueError('Current view is not from this action')
            if math.dist(request['position'], steps[index-1]['position']) > 1e-6:
                raise ValueError('Observation pose mismatch')
            if math.dist(request['look_dir'], steps[index-1]['look_dir']) > 1e-6:
                raise ValueError('Observation heading mismatch')
            contexts = request.get('context_observations', [])
            if index in confirmations:
                if len(contexts) != (3 if mode == 'multiview' else 0):
                    raise ValueError('Confirmation observation count mismatch')
            elif contexts:
                raise ValueError('Unexpected context outside confirmation')
            pins = {request['observation']: row['observation_sha256'],
                    **row.get('context_observation_sha256', {})}
            if set(pins) != {request['observation'], *(v['observation'] for v in contexts)}:
                raise ValueError('Missing or unexpected observation provenance')
            for offset, view in enumerate(contexts):
                observation_step = index-3+offset
                if Path(view['observation']) != (path.parent/f'obs_{observation_step:02d}.npz').resolve():
                    raise ValueError('Context includes a foreign, repeated or future view')
                previous = steps[observation_step-1]
                if math.dist(view['position'],previous['position']) > 1e-6 or \
                        math.dist(view['look_dir'],previous['look_dir']) > 1e-6:
                    raise ValueError('Context pose does not match its acquired view')
            for name, expected in pins.items():
                actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
                if actual != expected:
                    raise ValueError(f'Observation changed: {name}')
                result['source_hashes'][name] = actual
            decision = steps[index]['policy_decision']
            if decision['observations_supplied'] != 1+len(contexts):
                raise ValueError('Actual worker observation count mismatch')
            if decision.get('decision_source') == 'mtu3d_learned' and not decision.get('stage2_trace'):
                raise ValueError('Missing learned decision evidence')
            result['four_view_calls'] += int(decision['observations_supplied'] == 4)
            result['model_calls'] += 1
            result['empty_fallbacks'] += int(bool(decision.get('fallback_reason')))
        if len(begins) != record['model_meta']['commit_scans_started'] or \
                len(confirmations) != record['model_meta']['commit_scans_completed']:
            raise ValueError('Scan counters mismatch')
        result['scans_started'] += len(begins)
        result['scans_completed'] += len(confirmations)
        result['source_hashes'][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    result['passed'] = bool(records)
    return result
