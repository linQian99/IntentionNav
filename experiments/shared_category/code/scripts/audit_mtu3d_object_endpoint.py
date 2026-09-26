"""Audit endpoint termination independently of the new controller helper."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np

from audit_mtu3d_frontier_run import audit_frontier_run
from hash_navigation_inputs import digest_file


def audit_endpoints(run, source):
    sys.path.insert(0, str(source / 'simulator'))
    from walkable_map import WalkableMap
    counts = {'records': 0, 'object_plans': 0, 'projected_endpoint_stops': 0, 'center_radius_stops': 0}
    for path in sorted((run / 'episodes').glob('*/*/formal/record.json')):
        record = json.loads(path.read_text())
        if record['run_config'].get('object_goal_mode') != 'projected_endpoint':
            raise ValueError('Wrong realized endpoint controller')
        wm = WalkableMap.load(record['scene_id'], metaroot=Path('/path/to/workspace/datasets/vlntube/TaTaMeta/metadata_train'))
        center, plan = None, None
        for previous, step in zip(record['trajectory'], record['trajectory'][1:]):
            decision = step.get('policy_decision')
            if decision and decision['is_object_decision']:
                center = decision['target_position'][:2]
                route = wm.shortest_path_2d(tuple(previous['position'][:2]), tuple(center))
                plan = None if not route else {'predicted_center_xy': list(map(float, center)),
                    'navigation_endpoint_xy': list(map(float, route[-1])),
                    'center_is_walkable': bool(wm.is_walkable(*center)),
                    'projection_distance_m': math.dist(center, route[-1]),
                    'source': 'existing_shortest_path_2d_endpoint', 'arrival_tolerance_m': 1e-5}
                if decision.get('object_approach_plan') != plan:
                    raise ValueError('Plan does not match model prediction and actual map route')
                counts['object_plans'] += 1
            if step['action'] == 'STOP':
                if center is None or step.get('object_approach_plan') != plan:
                    raise ValueError('STOP lacks its model-selected object plan')
                distance = math.dist(previous['position'][:2], center)
                criterion = step.get('object_stop_criterion')
                if criterion == 'predicted_center_radius':
                    if distance > .8 or record['stop_reason'] != 'model_object_reached':
                        raise ValueError('Center STOP violates unchanged0.8m threshold')
                    counts['center_radius_stops'] += 1
                elif criterion == 'planned_projected_endpoint':
                    if (distance <= .8 or plan is None or plan['center_is_walkable']
                            or math.dist(previous['position'][:2], plan['navigation_endpoint_xy']) > 1e-5
                            or not wm.is_walkable(*plan['navigation_endpoint_xy'])
                            or record['stop_reason'] != 'model_object_projected_endpoint_reached'):
                        raise ValueError('STOP is not an actual arrival at a projected endpoint')
                    counts['projected_endpoint_stops'] += 1
                else:
                    raise ValueError('Unidentified STOP criterion')
            elif step['action'] == 'ROTATE_SCAN' and not step.get('commit_scan_phase'):
                center, plan = None, None
        counts['records'] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    actual = {'numpy_version': np.__version__, 'numpy_file': str(Path(np.__file__).resolve()),
              'opencv_version': cv2.__version__, 'opencv_file': str(Path(cv2.__file__).resolve())}
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or actual != manifest['audit_runtime']:
        raise ValueError('Require exact Isaac CPU replay environment with CUDA hidden')
    job = next(j for j in manifest['jobs'] if j['id'] == args.job)
    result = audit_frontier_run(Path(job['output']), Path(manifest['source']), job['exploration_mode'])
    endpoints = audit_endpoints(Path(job['output']), Path(manifest['source']))
    if result['records'] != job['count'] or endpoints['records'] != job['count']:
        raise ValueError('Incomplete endpoint run')
    result.update(object_endpoints=endpoints, audit_runtime=actual)
    result['source_hashes'][str(Path(__file__).resolve())] = digest_file(Path(__file__))[0]
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_hashes'}))


if __name__ == '__main__':
    main()
