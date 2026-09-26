"""Replay recorded RGB-D validity and reconcile the adapter's consumed views.

This establishes which views were accepted into the model call, not whether
the neural model attended to an object. The existing action audit runs first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hash_navigation_inputs import digest_file


def check_consumption(decision: dict, valid: list[int], supplied: int) -> None:
    if (type(decision.get('observations_supplied')) is not int
            or decision['observations_supplied'] != supplied
            or type(decision.get('observations_consumed')) is not int
            or decision['observations_consumed'] != len(valid)
            or decision.get('valid_view_indices') != valid
            or any(type(i) is not int for i in decision.get('valid_view_indices', []))):
        raise ValueError('Actual consumed-view identity/count differs from recorded depth validity')
    if not valid and (decision.get('fallback_reason') != 'insufficient_depth'
                      or decision.get('is_object_decision') is not False):
        raise ValueError('No valid RGB-D must not produce a learned object decision')
    if valid and decision.get('fallback_reason') == 'insufficient_depth':
        raise ValueError('Valid RGB-D incorrectly reported as insufficient depth')


def audit_consumption(run: Path, base: dict) -> dict:
    if not base.get('passed'):
        raise ValueError('Existing source/action audit must pass first')
    result = {'records': 0, 'model_calls': 0, 'supplied_views': 0, 'consumed_views': 0,
              'filtered_views': [], 'source_hashes': {}}
    def checked(path: Path) -> None:
        name = str(path.resolve())
        actual = digest_file(path)[0]
        if base['source_hashes'].get(name) != actual:
            raise ValueError(f'Evidence absent from or changed since action audit: {name}')
        result['source_hashes'][name] = actual
    for path in sorted((run / 'episodes').rglob('record.json')):
        checked(path)
        record = json.loads(path.read_text())
        for step in record['trajectory']:
            decision = step.get('policy_decision')
            if not decision:
                continue
            request_path = path.parent / f"request_{step['step']:02d}.json"
            checked(request_path)
            request = json.loads(request_path.read_text())['request']
            views = list(request.get('context_observations', [])) + [request]
            valid = []
            for index, view in enumerate(views):
                observation = Path(view['observation'])
                checked(observation)
                with np.load(observation, allow_pickle=False) as arrays:
                    rgb, depth = arrays['rgb'], arrays['depth']
                    intrinsic = view['intrinsics']
                    if (rgb.ndim != 3 or rgb.shape[2] != 3 or depth.shape != rgb.shape[:2]
                            or depth.shape != (intrinsic['height'], intrinsic['width'])
                            or intrinsic != request['intrinsics']):
                        raise ValueError('Mismatched RGB-D raster/intrinsics')
                    clean = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
                    positive = int(np.count_nonzero(clean))
                    if positive > 1000 and positive / depth.size >= .5:
                        valid.append(index)
            check_consumption(decision, valid, len(views))
            result['model_calls'] += 1
            result['supplied_views'] += len(views)
            result['consumed_views'] += len(valid)
            if len(valid) < len(views):
                result['filtered_views'].append({'selection_id': record['selection_id'],
                    'step': step['step'], 'valid_view_indices': valid})
        result['records'] += 1
    if result['records'] != base['records'] or result['model_calls'] != base['model_calls']:
        raise ValueError('Consumption audit has incomplete record/call closure')
    result['passed'] = bool(result['records'])
    result['source_hashes'][str(Path(__file__).resolve())] = digest_file(Path(__file__))[0]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--action-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit_consumption(args.run, json.loads(args.action_audit.read_text()))
    result['source_hashes'][str(args.action_audit.resolve())] = digest_file(args.action_audit)[0]
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'source_hashes'}))


if __name__ == '__main__':
    main()
