#!/usr/bin/env python3
"""Verify and recompute the complete hosted repeat release without model calls."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def recompute(root):
    checked = 0
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        expected, relative = line.split('  ', 1)
        path = (root / relative).resolve()
        assert path.is_relative_to(root), relative
        assert sha256(path) == expected, relative
        checked += 1
    manifest = json.loads((root / 'provenance/execution_manifest.json').read_text())
    state = json.loads((root / 'provenance/state.json').read_text())
    assert state['status'] == 'complete' and state['completed'] == 480
    assert manifest['planner_cache_policy'] == 'clear_before_and_after_every_episode'
    ids = manifest['selection_ids']
    jobs = manifest['schedule']
    styles = ['formal', 'natural', 'casual', 'emotional']
    repeats = ['r1', 'r2', 'r3']
    assert len(ids) == len(set(ids)) == 40
    assert len(jobs) == 480
    expected_keys = set(itertools.product(ids, styles, repeats))
    assert {(j['selection_id'], j['style'], j['repeat']) for j in jobs} == expected_keys
    episodes = {d['selection_id']: d for d in map(json.loads, (root / 'input/episodes.jsonl').read_text().splitlines())}
    items = {d['selection_id']: d for d in map(json.loads, (root / 'input/selected_500_intents.jsonl').read_text().splitlines())}
    outcomes = np.zeros((40, 4, 3), dtype=int)
    oracle = np.zeros_like(outcomes)
    for job in jobs:
        sid, style, repeat = (job[k] for k in ['selection_id', 'style', 'repeat'])
        directory = root / 'records' / repeat / job['scene_id'] / sid / style
        raw = (directory / 'record.json').read_bytes()
        record = json.loads(raw)
        receipt = json.loads((directory / 'repeat_receipt.json').read_text())
        assert receipt['record_sha256'] == hashlib.sha256(raw).hexdigest()
        assert all(receipt[k] == value for k, value in job.items())
        assert receipt['policy_revision'] == manifest['policy_revision']
        assert record['selection_id'] == sid and record['style'] == style
        assert record['scene_id'] == job['scene_id'] == episodes[sid]['scene_id']
        assert record['episode_meta'] == episodes[sid]
        assert record['intent'] == items[sid][style + '_en']
        assert record['model_meta'] == manifest['inference']
        assert record['step_cap'] == 30
        witness = receipt['planner_witness']
        assert witness['fresh'] is True and witness['cached'] is False
        assert witness['client_calls'] == 1 and not witness['error']
        assert state['started_at'] <= witness['started_at'] <= witness['finished_at'] <= state['finished_at']
        plan = record['plan']
        assert not plan.get('_cached') and not plan.get('error')
        assert plan['plan_model'] == manifest['inference']['model']
        assert plan['plan_usage'] == witness['usage'] and witness['usage'] is not None
        content = {k: v for k, v in plan.items() if k not in ['plan_model', 'plan_usage']}
        assert hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest() == witness['plan_sha256']
        distances = [math.dist(t['position'][:2], episodes[sid]['target_position'][:2])
                     for t in record['trajectory'] if t.get('position')]
        assert distances and all(math.isfinite(d) for d in distances)
        index = (ids.index(sid), styles.index(style), repeats.index(repeat))
        outcomes[index] = distances[-1] <= 2
        oracle[index] = min(distances) <= 2
    assert len(list((root / 'records').glob('*/*/*/*/record.json'))) == 480
    within = np.array([np.mean([outcomes[i, s, a] != outcomes[i, s, b]
                                for s in range(4) for a, b in itertools.combinations(range(3), 2)])
                       for i in range(40)])
    across = np.array([np.mean([outcomes[i, a, r] != outcomes[i, b, q]
                                for a, b in itertools.combinations(range(4), 2)
                                for r, q in itertools.product(range(3), repeat=2)])
                       for i in range(40)])
    gap = across - within
    rng = np.random.default_rng(20260924)
    resamples = rng.integers(0, 40, size=(50000, 40))
    scenes = np.array([episodes[sid]['scene_id'] for sid in ids])
    groups = [np.flatnonzero(scenes == scene) for scene in sorted(set(scenes))]
    scene_gap = []
    for _ in range(10000):
        sample = np.concatenate([groups[k] for k in rng.integers(0, len(groups), size=len(groups))])
        scene_gap.append(float(gap[sample].mean() * 100))

    def statistic(values):
        return {'mean_percent': float(values.mean() * 100),
                'task_ci95_percent': np.quantile(values[resamples].mean(axis=1) * 100, [.025, .975]).tolist()}

    result = {'records': 480, 'tasks': 40, 'scenes': len(groups),
              'fresh_initial_plans': 480, 'cached_initial_plans': 0,
              'within_expression_disagreement': statistic(within),
              'cross_expression_disagreement': statistic(across),
              'cross_minus_within_pp': statistic(gap),
              'scene_ci95_difference_pp': np.quantile(scene_gap, [.025, .975]).tolist(),
              'per_repeat': {repeat: {'SR_percent': float(outcomes[:, :, k].mean() * 100),
                                      'All_four_percent': float(np.all(outcomes[:, :, k], axis=1).mean() * 100)}
                             for k, repeat in enumerate(repeats)}}
    reported = json.loads((root / 'analysis/results.json').read_text())

    def compare(actual, expected):
        if isinstance(actual, dict):
            for key, value in actual.items():
                compare(value, expected[key])
        elif isinstance(actual, list):
            assert len(actual) == len(expected)
            for a, b in zip(actual, expected):
                compare(a, b)
        else:
            assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (actual, expected)

    compare(result, reported)
    result.update(verified_files=checked, SR_successes=int(outcomes.sum()),
                  OSR_successes=int(oracle.sum()), reported_statistics_match=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('release', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = recompute(args.release.resolve())
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')
