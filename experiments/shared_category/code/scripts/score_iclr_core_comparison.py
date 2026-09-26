"""Score closed ICLR comparisons; keep designated and category goals separate."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

from hash_navigation_inputs import digest_file
from probe_mtu3d_worker_reset import write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval/aggregate'))
from compute_metrics import goal_region_metrics, protocol_trajectory

ARMS = ['r055_explicit', 'mtu3d_explicit', 'r055_implicit_category', 'mtu3d_implicit_category']
METRICS = ['SR_hit', 'OSR_hit', 'terminal_false_stop', 'reached_without_stop', 'reached_but_failed', 'SPL']
AMBIGUOUS = {'SEL_077', 'SEL_314'}


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def verify(pins):
    for name, expected in pins.items():
        if digest_file(Path(name))[0] != expected:
            raise ValueError(f'Changed evidence: {name}')


def score_record(record, episode, goal, system):
    trajectory, invalid = protocol_trajectory(record)
    if (invalid or not trajectory or len(trajectory) > 31 or record['step_cap'] != 30
            or record['scene_id'] != episode['scene_id']
            or math.dist(trajectory[0]['position'], episode['start_position']) > 1e-5):
        raise ValueError('Invalid trajectory, budget, scene or start position')
    fixed = [x for x in goal['exact_category_instances'] if x['object_id'] == episode['target_object_id']]
    if len(fixed) != 1:
        raise ValueError('Designated target missing or duplicated')
    stopped = any(str(t.get('action', '')).strip().upper() == 'STOP' for t in trajectory)
    result = dict(selection_id=episode['selection_id'], scene_id=episode['scene_id'],
        target_object_id=episode['target_object_id'], target_category=episode['target_category'],
        query=record['prediction']['target'], stopped=int(stopped), ambiguous=int(episode['selection_id'] in AMBIGUOUS),
        actions=len(trajectory) - 1, path_length_m=sum(math.dist(a['position'][:2], b['position'][:2])
            for a, b in zip(trajectory, trajectory[1:])))
    for prefix, instances, region_key in [('Fixed', fixed, 'fixed_instance'),
            ('Any', goal['exact_category_instances'], 'any_exact_category_instance')]:
        region = episode['goal_regions_v2'][region_key]
        if region['radius_m'] != 2.0 or region['distance_mode'] != 'xy_aabb_surface':
            raise ValueError('Unexpected scoring geometry')
        scored = goal_region_metrics(record, instances, 2.0, prefix, region['geodesic_to_goal_region'])
        if not scored or bool(scored[prefix + '_start_inside']) != bool(region['start_inside']):
            raise ValueError('Incomplete geometry or inconsistent start cohort')
        # SPL with zero reference distance is undefined here. Keep such tasks,
        # report their SR separately, and never turn undefined SPL into zero.
        if not region['start_inside'] and prefix + '_SPL' not in scored:
            raise ValueError('Primary task lacks positive goal-region distance')
        scored.setdefault(prefix + '_SPL', None)
        scored.update({prefix + '_reached_without_stop': int(scored[prefix + '_OSR_hit'] and not stopped),
                       prefix + '_reached_but_failed': int(scored[prefix + '_OSR_hit'] and not scored[prefix + '_SR_hit'])})
        result.update(scored)
    timer = (record['elapsed_seconds'] if system == 'mtu3d'
             else record['observation_archive']['engine_to_policy_termination_seconds'])
    if not math.isfinite(timer) or timer < 0:
        raise ValueError('Invalid native episode timer')
    result['native_episode_seconds'] = timer
    return result


def inputs(m):
    return ({r['selection_id']: r for r in rows(m['episodes'])},
            {(r['scene_id'], r['target_category']): r for r in rows(m['goals'])})


def score_paths(paths, m, system):
    episodes, goals = inputs(m)
    scored = {}
    for sid, name in sorted(paths.items()):
        record = json.loads(Path(name).read_text()); ep = episodes[sid]
        if record['selection_id'] != sid:
            raise ValueError('Record identity differs from selection')
        scored[sid] = {**score_record(record, ep, goals[(ep['scene_id'], ep['target_category'])], system),
                       'record_path': name}
    return scored


def summarize(scored):
    result = {}
    for prefix in ['Fixed', 'Any']:
        result[prefix] = {}
        for cohort in ['all', 'outside_start', 'inside_start', 'outside_start_unambiguous']:
            selected = [r for r in scored.values() if (cohort == 'all'
                or (cohort == 'inside_start' and r[prefix + '_start_inside'])
                or (cohort.startswith('outside_start') and not r[prefix + '_start_inside']
                    and (cohort != 'outside_start_unambiguous' or not r['ambiguous'])))]
            table = dict(tasks=len(selected), metrics={})
            for metric in METRICS:
                values = [r[prefix + '_' + metric] for r in selected if r[prefix + '_' + metric] is not None]
                table['metrics'][metric] = dict(n=len(values), mean=float(np.mean(values)) if values else None,
                    total=float(sum(values)) if values else None)
            for metric in ['actions', 'path_length_m', 'native_episode_seconds']:
                table['metrics'][metric] = dict(n=len(selected), mean=float(np.mean([r[metric] for r in selected])) if selected else None)
            result[prefix][cohort] = table
    return result


def score_job(m, manifest_path, job, run, output):
    if output.exists() or not (run / '.RUN_SUCCESS').is_file():
        raise ValueError('Use a fresh score path and a closed run')
    paths = {}
    for path in (run / 'episodes').rglob('record.json'):
        sid = json.loads(path.read_text())['selection_id']
        if sid in paths:
            raise ValueError('Duplicate record')
        paths[sid] = str(path)
    if set(paths) != set(job['selection_ids']) or len(paths) != job['count']:
        raise ValueError('Missing or unexpected episode')
    scored = score_paths(paths, m, job['system'])
    files = [manifest_path, Path(__file__), ROOT / 'eval/aggregate/compute_metrics.py',
             Path(m['episodes']), Path(m['goals']), run / '.RUN_SUCCESS', *map(Path, paths.values())]
    report = dict(passed=True, records=len(scored), rows=list(scored.values()), summary=summarize(scored),
        source_hashes={str(p.resolve()): digest_file(p)[0] for p in files})
    write(output, report)
    print(json.dumps(dict(passed=True, records=len(scored), output=str(output))), flush=True)


def interval(values, keys):
    """Paired cluster bootstrap, with tasks weighted equally within each draw."""
    values = np.asarray(values, dtype=float)
    groups = defaultdict(list)
    for i, key in enumerate(keys):
        groups[key].append(i)
    totals = np.array([values[ix].sum() for ix in groups.values()])
    counts = np.array([len(ix) for ix in groups.values()])
    rng = np.random.default_rng(20260919)
    samples = []
    for _ in range(10):
        draw = rng.integers(0, len(groups), size=(1000, len(groups)))
        samples.append(totals[draw].sum(axis=1) / counts[draw].sum(axis=1))
    return np.quantile(np.concatenate(samples), [.025, .975]).tolist()


def contrast(left, right):
    if set(left) != set(right) or not left:
        raise ValueError('Incomplete paired cohort')
    result = {}
    for prefix in ['Fixed', 'Any']:
        if any(left[s][prefix + '_start_inside'] != right[s][prefix + '_start_inside'] for s in left):
            raise ValueError('Paired start cohorts differ')
        result[prefix] = {}
        for cohort in ['outside_start', 'outside_start_unambiguous']:
            ids = [s for s in sorted(left) if not left[s][prefix + '_start_inside']
                   and (cohort != 'outside_start_unambiguous' or not left[s]['ambiguous'])]
            values = {}
            for metric in [prefix + '_' + name for name in METRICS] + ['actions', 'path_length_m']:
                a = np.array([left[s][metric] for s in ids], dtype=float)
                b = np.array([right[s][metric] for s in ids], dtype=float)
                if not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError('Nonfinite paired metric')
                scenes = [left[s]['scene_id'] for s in ids]
                physical = [left[s]['scene_id'] + '/' + left[s]['target_object_id'] for s in ids]
                values[metric] = dict(left_mean=float(a.mean()), right_mean=float(b.mean()),
                    right_minus_left=float((b-a).mean()), paired_item_ci95=interval(b-a, ids),
                    paired_scene_ci95=interval(b-a, scenes), paired_designated_object_ci95=interval(b-a, physical))
            result[prefix][cohort] = dict(tasks=len(ids), metrics=values)
    return result


def compare(m, manifest_path, output):
    if output.exists():
        raise FileExistsError(output)
    verify(m['source_hashes'])
    state_path = Path(m['output']) / 'queue_state.json'
    state = json.loads(state_path.read_text())
    if (state['manifest_sha256'] != digest_file(manifest_path)[0]
            or set(state['checkpoints']) != {j['id'] for j in m['jobs']}
            or state['complete_new_records'] != m['new_navigation_records']):
        raise ValueError('Incomplete comparison; do not produce a partial headline')
    arms = defaultdict(dict); pins = {str(manifest_path): digest_file(manifest_path)[0]}
    for job in m['jobs']:
        pointer = state['checkpoints'][job['id']]; verify(pointer['source_hashes'])
        cp = json.loads(Path(pointer['checkpoint']).read_text()); verify(cp['source_hashes'])
        pins.update(pointer['source_hashes'])
        report = json.loads(Path(cp['scores']).read_text())
        if not report['passed'] or report['records'] != job['count']:
            raise ValueError('Incomplete score report')
        saved = {r['selection_id']: r for r in report['rows']}
        rescored = score_paths({sid: r['record_path'] for sid, r in saved.items()}, m, job['system'])
        if saved != rescored:
            raise ValueError('Independent final score recomputation differs')
        key = (job['stage'], job['system'] + '_' + job['mode'], job['repeat'])
        if set(arms[key]) & set(rescored):
            raise ValueError('Duplicate arm/task')
        arms[key].update(rescored)
    reference = json.loads(Path(m['reference_reuse']).read_text()); verify(reference['source_hashes'])
    arms[('full500', 'r055_explicit', 'a')] = score_paths(reference['records'], m, 'r055')
    if any(len(arms[('full500', a, 'a')]) != 500 for a in ARMS):
        raise ValueError('Four full arms must each contain 500 tasks')
    if any(len(arms[('repeat40', a, r)]) != 40 for a in ARMS for r in ['a', 'b']):
        raise ValueError('Repeat stability requires all eight dev40 arms')
    pairs = [(ARMS[0], ARMS[1]), (ARMS[2], ARMS[3]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[3])]
    comparisons = {a + '_vs_' + b: contrast(arms[('full500', a, 'a')], arms[('full500', b, 'a')]) for a, b in pairs}
    repeats = {a: contrast(arms[('repeat40', a, 'a')], arms[('repeat40', a, 'b')]) for a in ARMS}
    report = dict(passed=True, full_records=2000, fresh_navigation=1828, reused_navigation=500,
        full_repeats=1, dev40_repeats=2, publication_ready=False, full_B0_certified=False,
        scope=m['scope'], bootstrap=dict(repetitions=10000, seed=20260919, paired=True),
        timer_scope='Native policy-episode timers exclude outer scene/model startup; report per-system scope, not matched wall time.',
        oracle_scope='OSR is only an offline geometric oracle-stop upper bound on these unchanged paths, not an executed oracle policy.',
        summaries={'/'.join(k): summarize(v) for k, v in arms.items()}, full_comparisons=comparisons,
        repeat_stability=repeats, source_hashes=pins)
    write(output, report)
    with output.with_suffix('.csv').open('x') as handle:
        fields = ['stage', 'arm', 'repeat', *sorted({key for scored in arms.values() for row in scored.values() for key in row})]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for (stage, arm, repeat), scored in sorted(arms.items()):
            for row in scored.values():
                writer.writerow(dict(stage=stage, arm=arm, repeat=repeat, **row))
    print(json.dumps(dict(passed=True, full_records=2000, publication_ready=False)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['job', 'compare'])
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--job')
    p.add_argument('--run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); path = a.manifest.resolve(); m = json.loads(path.read_text())
    if a.mode == 'job':
        score_job(m, path, next(j for j in m['jobs'] if j['id'] == a.job), a.run, a.output)
    else:
        compare(m, path, a.output)
