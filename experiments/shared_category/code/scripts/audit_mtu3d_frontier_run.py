"""Reconcile observed-frontier requests and committed execution after a run."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

from check_mtu3d_commit_scan import audit_run


def audit_frontier_run(run: Path, source: Path, mode: str) -> dict:
    result = audit_run(run, 'multiview')
    sys.path[:0] = [str(source / 'agents'), str(source / 'simulator')]
    from mtu3d_frontier_memory import FrontierMemory
    from walkable_map import WalkableMap
    result.update(frontier_selections=0, frontier_arrivals=0, frontier_route_failures=0,
                  continued_frontier_moves=0, boundary_requests_replayed=0)
    for path in sorted((run / 'episodes').rglob('record.json')):
        record = json.loads(path.read_text())
        if record['run_config']['exploration_mode'] != mode:
            raise ValueError('Wrong exploration mode')
        if record['run_config']['scan_schedule'] != 'before_decision':
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
