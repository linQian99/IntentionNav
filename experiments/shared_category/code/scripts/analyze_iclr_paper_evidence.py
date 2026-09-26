"""Build paper-facing evidence from closed, matched ICLR navigation arms on CPU."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from hash_navigation_inputs import digest_file
from probe_mtu3d_worker_reset import write
from score_iclr_core_comparison import (
    AMBIGUOUS, ARMS, inputs, interval, protocol_trajectory, score_record, summarize, verify,
)

CORE_SHA256 = 'ac71460fd6675861a36df1be4e34e2430b7f87cfe2d1e7ebcccc53c10b457507'
RADII = (1, 2, 3)
BUCKETS = ('designated_success', 'category_success_only', 'stop_outside_category',
           'reached_designated_no_stop', 'reached_category_only_no_stop', 'never_reached_category_no_stop')


def read(path):
    return json.loads(Path(path).read_text())


def center_metrics(record, episode):
    """Original paper metric: terminal/closest recorded XY pose to target center."""
    trajectory, invalid = protocol_trajectory(record)
    if invalid or not trajectory or len(trajectory) > 31 or record['step_cap'] != 30:
        raise ValueError('Invalid budgeted trajectory')
    target = episode['target_position']
    positions = [t['position'] for t in trajectory]
    if (len(target) != 3 or any(len(p) != 3 for p in positions)
            or not all(math.isfinite(v) for p in [target, *positions] for v in p)):
        raise ValueError('Nonfinite or incomplete center geometry')
    if math.dist(positions[0], episode['start_position']) > 1e-5:
        raise ValueError('Start position mismatch')
    distances = [math.dist(p[:2], target[:2]) for p in positions]
    return dict(center_final_m=distances[-1], center_min_m=min(distances),
                **{f'paper_{metric}_{radius}m': int(value <= radius)
                   for radius in RADII for metric, value in [('SR', distances[-1]), ('OSR', min(distances))]})


def failure_bucket(row):
    for name, condition in zip(BUCKETS, [row['Fixed_SR_hit'], row['Any_SR_hit'], row['stopped'],
                                        row['Fixed_OSR_hit'], row['Any_OSR_hit'], True]):
        if condition:
            return name
    raise AssertionError('Unreachable partition')


def estimate(values, rows):
    if len(values) != len(rows):
        raise ValueError('Statistic and cluster identities differ')
    if not rows:
        return dict(n=0, mean=None, item_ci95=None, scene_ci95=None, object_ci95=None)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite statistic')
    return dict(n=len(rows), mean=float(np.mean(values)),
                item_ci95=interval(values, [r['selection_id'] for r in rows]),
                scene_ci95=interval(values, [r['scene_id'] for r in rows]),
                object_ci95=interval(values, [r['scene_id'] + '/' + r['target_object_id'] for r in rows]))


def paired_values(arms, coefficients, metric, ids):
    """Align on task identity, never CSV order or a silently intersected subset."""
    if not ids or any(set(arms[arm]) != set(ids) for arm in coefficients):
        raise ValueError('Incomplete matched comparison')
    values = []
    for sid in ids:
        selected = [arms[arm][sid] for arm in coefficients]
        identity = lambda r: (r['scene_id'], r['target_object_id'], r['target_category'], r['frontend_IM_hit'])
        if len({identity(r) for r in selected}) != 1:
            raise ValueError('Paired task/target/frontend differs')
        values.append(sum(arms[arm][sid][metric] * weight for arm, weight in coefficients.items()))
    return values


def load_closed(manifest_path, stage):
    if digest_file(manifest_path)[0] != CORE_SHA256:
        raise ValueError('Unexpected frozen core manifest')
    m = read(manifest_path)
    verify(m['source_hashes'])
    state = read(Path(m['output']) / 'queue_state.json')
    if state['manifest_sha256'] != CORE_SHA256:
        raise ValueError('Queue manifest identity differs')
    if stage == 'full500':
        if (state['status'] != 'completed_comparison' or state['complete_new_records'] != 1828
                or set(state['checkpoints']) != {j['id'] for j in m['jobs']}):
            raise ValueError('Full comparison has not closed')
        comparison = read(Path(m['output']) / 'comparison.json')
        if not comparison['passed'] or comparison['full_records'] != 2000:
            raise ValueError('Final core comparison failed')
        verify(comparison['source_hashes'])
    jobs = [j for j in m['jobs'] if j['stage'] == stage]
    if any(j['id'] not in state['checkpoints'] for j in jobs):
        raise ValueError('Requested stage is incomplete')
    eps, goals = inputs(m)
    pins = {str(manifest_path.resolve()): CORE_SHA256}
    if stage == 'full500':
        p = Path(m['output']) / 'comparison.json'
        pins[str(p)] = digest_file(p)[0]
    frontend = {}
    for sid, ep in eps.items():
        packet_path = Path(m['shared_inputs']) / 'implicit_category/formal' / f'{sid}.json'
        packet = read(packet_path)
        raw_path = Path(m['output']).parent / 'intent_frontend/records/formal' / f'{sid}.json'
        expected = packet['provenance']['source_record_sha256']
        verify({str(raw_path): expected})
        raw = read(raw_path)
        if (raw['selection_id'] != sid or raw['scene_id'] != ep['scene_id']
                or raw['target_category'] != ep['target_category'] or raw['IM_hit'] not in (0, 1)
                or raw['prediction']['target'] != packet['query_category']):
            raise ValueError('Frontend evidence identity mismatch')
        frontend[sid] = dict(query=packet['query_category'], IM_hit=raw['IM_hit'])
        pins[str(raw_path)] = expected
    groups = {}

    def add(paths, arm, repeat, expected_hashes, saved=None):
        system, mode = arm.split('_', 1)
        verify({path: expected_hashes[path] for path in paths.values()})
        group = groups.setdefault(f'{arm}/{repeat}', {})
        for sid, path in paths.items():
            if sid in group:
                raise ValueError('Duplicate arm/task')
            rec = read(path)
            ep = eps[sid]
            if rec['selection_id'] != sid:
                raise ValueError('Record identity mismatch')
            query = ep['target_category'] if mode == 'explicit' else frontend[sid]['query']
            # Match the already-frozen shared-input auditor: MTU's worker
            # consumes display strings, while R055 retains category underscores.
            if system == 'mtu3d':
                query = query.replace('_', ' ')
            if rec['prediction']['target'] != query:
                raise ValueError('Actual query does not match assigned arm')
            row = {**score_record(rec, ep, goals[(ep['scene_id'], ep['target_category'])], system), 'record_path': path}
            if saved is not None and row != saved[sid]:
                raise ValueError('Independent strict score replay differs')
            row.update(center_metrics(rec, ep), frontend_IM_hit=frontend[sid]['IM_hit'])
            row['failure_bucket'] = failure_bucket(row)
            group[sid] = row
            pins[path] = expected_hashes[path]

    for job in jobs:
        pointer = state['checkpoints'][job['id']]
        verify(pointer['source_hashes'])
        cp = read(pointer['checkpoint'])
        verify({cp['scores']: cp['source_hashes'][cp['scores']]})
        report = read(cp['scores'])
        saved = {r['selection_id']: r for r in report['rows']}
        if (not report['passed'] or report['records'] != job['count']
                or len(report['rows']) != job['count'] or set(saved) != set(job['selection_ids'])):
            raise ValueError('Incomplete or duplicate score records')
        pins.update(pointer['source_hashes'])
        pins[cp['scores']] = cp['source_hashes'][cp['scores']]
        add({sid: r['record_path'] for sid, r in saved.items()}, job['system'] + '_' + job['mode'],
            job['repeat'], report['source_hashes'], saved)
    if stage == 'full500':
        ref = read(m['reference_reuse'])
        add(ref['records'], 'r055_explicit', 'a', ref['source_hashes'])
    expected_ids = (set(eps) if stage == 'full500' else set(jobs[0]['selection_ids']))
    repeats = ['a'] if stage == 'full500' else ['a', 'b']
    if (set(groups) != {a + '/' + r for a in ARMS for r in repeats}
            or any(set(group) != expected_ids for group in groups.values())):
        raise ValueError('Incomplete four-arm task coverage')
    return m, groups, pins


def analyze(groups, stage):
    tables, partitions, conditional, radius, comparisons = {}, {}, {}, {}, {}
    for key, group in groups.items():
        ordered = [group[s] for s in sorted(group)]
        tables[key] = {metric: estimate([r[metric] for r in ordered], ordered)
                       for metric in ['paper_SR_2m', 'paper_OSR_2m']}
        tables[key]['IM'] = (estimate([r['frontend_IM_hit'] for r in ordered], ordered)
                            if 'implicit_category' in key else None)
        tables[key]['GSR'] = None
        partitions[key] = {bucket: sum(r['failure_bucket'] == bucket for r in ordered) for bucket in BUCKETS}
        conditional[key] = {}
        for hit in (0, 1):
            selected = [r for r in ordered if r['frontend_IM_hit'] == hit]
            conditional[key][str(hit)] = dict(n=len(selected),
                paper_SR=estimate([r['paper_SR_2m'] for r in selected], selected),
                fixed_SR=estimate([r['Fixed_SR_hit'] for r in selected], selected),
                fixed_OSR=estimate([r['Fixed_OSR_hit'] for r in selected], selected),
                failures={b: sum(r['failure_bucket'] == b for r in selected) for b in BUCKETS})
        radius[key] = {str(r): dict(n=len(ordered), SR=sum(x[f'paper_SR_{r}m'] for x in ordered) / len(ordered),
                                   OSR=sum(x[f'paper_OSR_{r}m'] for x in ordered) / len(ordered)) for r in RADII}
    coefficients = {
        'MTU_minus_R055_correct_category': {'mtu3d_explicit': 1, 'r055_explicit': -1},
        'MTU_minus_R055_predicted_category': {'mtu3d_implicit_category': 1, 'r055_implicit_category': -1},
        'R055_correct_minus_predicted': {'r055_explicit': 1, 'r055_implicit_category': -1},
        'MTU_correct_minus_predicted': {'mtu3d_explicit': 1, 'mtu3d_implicit_category': -1},
        'input_effect_MTU_minus_R055': {'mtu3d_explicit': 1, 'mtu3d_implicit_category': -1,
                                      'r055_explicit': -1, 'r055_implicit_category': 1},
    }
    for repeat in (['a'] if stage == 'full500' else ['a', 'b']):
        arms = {arm: groups[arm + '/' + repeat] for arm in ARMS}
        ids = sorted(arms[ARMS[0]])
        ordered = [arms[ARMS[0]][sid] for sid in ids]
        for name, coef in coefficients.items():
            comparisons[name + '/' + repeat] = {}
            for metric in ['paper_SR_2m', 'paper_OSR_2m', 'Fixed_SR_hit', 'Any_SR_hit']:
                values = paired_values(arms, coef, metric, ids)
                comparisons[name + '/' + repeat][metric] = {
                    'all': estimate(values, ordered),
                    'excluding_ambiguous': estimate([v for v, r in zip(values, ordered) if r['selection_id'] not in AMBIGUOUS],
                                                   [r for r in ordered if r['selection_id'] not in AMBIGUOUS])}
    stability = {}
    if stage == 'repeat40':
        for arm in ARMS:
            left, right = groups[arm + '/a'], groups[arm + '/b']
            ids = sorted(left)
            stability[arm] = {}
            for metric in ['paper_SR_2m', 'Fixed_SR_hit', 'Any_SR_hit']:
                delta = paired_values({'a': left, 'b': right}, {'a': -1, 'b': 1}, metric, ids)
                stability[arm][metric] = dict(repeat_b_minus_a=estimate(delta, [left[s] for s in ids]),
                                              changed_outcomes=sum(v != 0 for v in delta))
    return dict(passed=True, stage=stage, records=sum(map(len, groups.values())),
        full_repeats=1, dev40_repeats=2, publication_ready=False, full_B0_certified=False,
        units='Means/intervals are fractions; paper tables multiply rates and rate differences by 100.',
        definitions=dict(SR='Final recorded XY position within 2m of designated target center, independent of STOP.',
                         OSR='Any recorded XY position within 2m of designated target center.',
                         GSR='Not computed: no equivalent visibility evidence certified for these arms.',
                         strict='Fixed/Any retain frozen actual-STOP and XY-AABB-surface definitions.',
                         partition='Six mutually exclusive buckets, denominator all tasks in each arm/repeat.',
                         conditional='Strata use the SAME cached frontend for both input conditions; descriptive, not causal mediation.',
                         comparison='Only matched new four-arm comparisons; no paired contrasts to historical hosted-VLM rows.'),
        bootstrap=dict(repetitions=10000, seed=20260919, task_primary=True, scene_and_object_sensitivity=True),
        main_table=tables, failure_partition=partitions, frontend_conditioned=conditional,
        radius_sensitivity=radius, paired_comparisons=comparisons, repeat_stability=stability,
        strict_summaries={k: summarize(v) for k, v in groups.items()})


def save_report(output, report, groups, pins):
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'evidence.json').exists():
        raise FileExistsError(output / 'evidence.json')
    report['source_hashes'] = pins
    report['created_at'] = datetime.now(timezone.utc).isoformat()
    rows_path = output / 'episodes.csv'
    with rows_path.open('w') as f:
        fields = ['arm_repeat', *sorted(next(iter(next(iter(groups.values())).values())))]
        writer = csv.DictWriter(f, fields); writer.writeheader()
        for key, group in sorted(groups.items()):
            writer.writerows(dict(arm_repeat=key, **group[s]) for s in sorted(group))
    lines = ['| System / input / repeat | N | IM (%) | OSR (%) | SR (%) | SR 95% CI |',
             '|---|---:|---:|---:|---:|---|']
    for key, row in report['main_table'].items():
        sr, osr = row['paper_SR_2m'], row['paper_OSR_2m']
        im = f"{100 * row['IM']['mean']:.2f}" if row['IM'] is not None else '—'
        ci = ', '.join(f'{100 * v:.2f}' for v in sr['item_ci95'])
        lines.append(f"| {key} | {sr['n']} | {im} | {100 * osr['mean']:.2f} | {100 * sr['mean']:.2f} | [{ci}] |")
    text = (f"# {report['stage']} navigation evidence\n\n" + '\n'.join(lines)
            + '\n\nSR uses final center distance; strict STOP statistics remain in evidence.json. '
              'Development repeats are separate; full500 is one run per arm. GSR is unavailable. '
              'This evidence does not certify full B0 or authorize publication automatically.\n')
    (output / 'main_table.md').write_text(text)
    report['artifacts'] = {str(p.resolve()): digest_file(p)[0] for p in [rows_path, output / 'main_table.md']}
    write(output / 'evidence.json', report)


def dependency_state(state):
    if state['status'] in {'failed_preserved', 'deadline_stopped', 'stopped_by_user'}:
        raise RuntimeError('Core stopped: ' + state['status'] + ' ' + str(state.get('error', '')))
    return state['status'] == 'completed_comparison'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--stage', choices=['repeat40', 'full500'], default='full500')
    p.add_argument('--output', type=Path)
    p.add_argument('--check', action='store_true')
    p.add_argument('--wait-core', action='store_true')
    p.add_argument('--analysis-pins', type=Path)
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Run this CPU analysis with CUDA_VISIBLE_DEVICES empty')
    if a.wait_core and (a.stage != 'full500' or a.output is None or a.analysis_pins is None or a.check):
        raise ValueError('Wait requires pinned full500 output and cannot be a check')
    if a.analysis_pins:
        verify(read(a.analysis_pins))
    if not a.check and a.output is None:
        raise ValueError('Output is required')
    status_path = None
    lock = None
    try:
        if a.wait_core:
            a.output.mkdir(parents=True, exist_ok=True)
            lock = (a.output / 'analysis.lock').open('a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            status_path = a.output / 'state.json'
            if (a.output / 'evidence.json').exists():
                report = read(a.output / 'evidence.json')
                if not report['passed'] or report['stage'] != 'full500' or report['records'] != 2000:
                    raise ValueError('Existing analysis is incomplete')
                verify(report['source_hashes']); verify(report['artifacts'])
                write(status_path, dict(status='completed', evidence=str(a.output / 'evidence.json')))
                return
            if status_path.exists() and read(status_path)['status'] == 'failed_preserved':
                raise RuntimeError('Previous analysis failure requires inspection')
            last_count = None
            while True:
                if digest_file(a.manifest)[0] != CORE_SHA256:
                    raise ValueError('Core manifest changed while waiting')
                m = read(a.manifest)
                state = read(Path(m['output']) / 'queue_state.json')
                if dependency_state(state):
                    break
                elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(state['started_at'])).total_seconds()
                if elapsed > (m['max_wall_hours'] + 8) * 3600:
                    raise RuntimeError('Core completion exceeded bounded waiting time')
                if state['complete_new_records'] != last_count:
                    last_count = state['complete_new_records']
                    write(status_path, dict(status='waiting_for_core', core_complete_new_records=last_count,
                                            gpu_used=False, updated_at=datetime.now(timezone.utc).isoformat()))
                time.sleep(60)
            verify(read(a.analysis_pins))
            write(status_path, dict(status='analyzing', gpu_used=False))
        m, groups, pins = load_closed(a.manifest.resolve(), a.stage)
        report = analyze(groups, a.stage)
        if a.analysis_pins:
            pins.update(read(a.analysis_pins))
            pins[str(a.analysis_pins.resolve())] = digest_file(a.analysis_pins)[0]
        pins[str(Path(__file__).resolve())] = digest_file(Path(__file__))[0]
        if not a.check:
            save_report(a.output, report, groups, pins)
        if status_path:
            write(status_path, dict(status='completed', evidence=str(a.output / 'evidence.json'),
                                    finished_at=datetime.now(timezone.utc).isoformat(), gpu_used=False))
        print(json.dumps(dict(passed=True, stage=a.stage, groups=len(groups), records=report['records'],
                              gpu_initialized=False, output_created=not a.check)), flush=True)
    except BaseException as error:
        if status_path:
            write(status_path, dict(status='failed_preserved', error=repr(error), gpu_used=False))
        raise
    finally:
        if lock:
            lock.close()


if __name__ == '__main__':
    main()
