"""Run six isolated component episodes and audit their paid navigation actions."""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from hash_navigation_inputs import digest_file
from probe_mtu3d_worker_reset import write
from run_baseline_acceptance import terminate_job


def verify_sources(manifest):
    for name, expected in manifest['source_hashes'].items():
        if digest_file(Path(name))[0] != expected:
            raise ValueError(f'Frozen diagnostic input changed: {name}')


def summarize(manifest, root):
    source = Path(manifest['source'])
    sys.path.insert(0, str(source / 'aggregate'))
    from compute_metrics import goal_region_metrics, protocol_trajectory
    inputs = Path(manifest['inputs'])
    episodes = {e['selection_id']: e for e in map(json.loads, (inputs / 'episodes.jsonl').read_text().splitlines())}
    goals = {(g['scene_id'], g['target_category']): g for g in map(json.loads, (inputs / 'category_goal_sets.jsonl').read_text().splitlines())}
    run = Path(manifest['jobs'][0]['output'])
    paths = list((run / 'episodes').glob('*/*/formal/record.json'))
    records = {json.loads(p.read_text())['selection_id']: (p, json.loads(p.read_text())) for p in paths}
    if len(paths) != len(records) or set(records) != set(episodes):
        raise ValueError('Unexpected diagnostic record closure')
    rows = []
    for sid, episode in sorted(episodes.items()):
        path, record = records[sid]
        trajectory, dropped = protocol_trajectory(record)
        if dropped or len(trajectory) > 31 or math.dist(trajectory[0]['position'], episode['start_position']) > 1e-5:
            raise ValueError('Diagnostic start/action mismatch')
        goal = goals[(episode['scene_id'], episode['target_category'])]
        fixed = [x for x in goal['exact_category_instances'] if x['object_id'] == episode['target_object_id']]
        any_metric = goal_region_metrics(record, goal['exact_category_instances'], 2., 'Any',
            episode['goal_regions_v2']['any_exact_category_instance']['geodesic_to_goal_region'])
        fixed_metric = goal_region_metrics(record, fixed, 2., 'Fixed',
            episode['goal_regions_v2']['fixed_instance']['geodesic_to_goal_region'])
        if bool(any_metric['Any_start_inside']) != (episode['diagnostic_kind'] == 'near_stop'):
            raise ValueError('New start region was not scored consistently')
        decisions = [step['policy_decision'] for step in record['trajectory'] if step.get('policy_decision')]
        if not decisions:
            raise ValueError('No actual learned decision')
        first = decisions[0]
        if first['decision_index'] != 0 or first['randomness']['episode_key'] != f"{episode['scene_id']}/{sid}":
            raise ValueError('Wrong initial worker identity')
        rows.append({'selection_id': sid, 'source_selection_id': episode['source_selection_id'],
            'kind': episode['diagnostic_kind'], 'category': episode['target_category'],
            'actions': len(trajectory) - 1, 'stop_reason': record['stop_reason'],
            'first_model_choice_is_object': first['is_object_decision'],
            'first_head_trace': first.get('stage2_trace'),
            'model_calls': len(decisions), 'any_goal': any_metric, 'fixed_goal': fixed_metric,
            'record': str(path.resolve()), 'first_actual_rgb': str((path.parent / 'step_01.png').resolve()),
            'initial_actual_rgb_class_review': 'pending', 'benchmark_member': False})
    grouped = {}
    for kind in ['near_stop', 'visible_approach']:
        group = [r for r in rows if r['kind'] == kind]
        grouped[kind] = {'cases': len(group),
            'terminal_stop_in_any_category_region': sum(r['any_goal']['Any_SR_hit'] for r in group),
            'terminal_stop_in_selected_instance_region': sum(r['fixed_goal']['Fixed_SR_hit'] for r in group),
            'false_stops': sum(r['any_goal']['Any_terminal_false_stop'] for r in group)}
    evidence = paths + [inputs / 'suite.json', inputs / 'episodes.jsonl', inputs / 'category_goal_sets.jsonl']
    return {'complete': True, 'rows': rows, 'by_diagnostic_type': grouped,
        'human_signoff': 'pending', 'navigation_acceptance': False, 'benchmark_scores': False,
        'source_hashes': {str(p.resolve()): digest_file(p)[0] for p in evidence},
        'scope': 'Six artificial, geometry/visibility-selected diagnostic starts; four begin within2m. No pooled SR, no formal improvement claim, no generalization or repeatability claim. Review actual first RGB before judging visible-target capability.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    verify_sources(manifest)
    root = args.manifest.resolve().parent
    if (root / 'queue_state.json').exists():
        raise FileExistsError('Diagnostic queue already submitted')
    job, = manifest['jobs']
    if manifest['gpu'] != 0 or job['count'] != 6 or Path(job['output']).exists():
        raise ValueError('Require six fresh diagnostic episodes on GPU0 only')
    environment = dict(os.environ, **job['environment'], PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    subprocess.run(job['command'] + ['--check'], cwd=manifest['repo'], env=environment, check=True)
    if args.check:
        print(json.dumps({'passed': True, 'episodes': 6, 'source_pins': len(manifest['source_hashes']),
            'gpu_initialized': False, 'output_created': False, 'benchmark_scores': False,
            'manifest_sha256': digest_file(args.manifest)[0]}))
        return
    memory = [int(x) for x in subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,noheader,nounits'], text=True).splitlines()]
    if any(x >= 500 for x in memory):
        raise RuntimeError('Both GPUs must be idle before serial GPU0 diagnostics')
    state = {'status': 'starting', 'pid': os.getpid(), 'gpu': 0, 'maximum_episodes': 6,
        'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'manifest_sha256': digest_file(args.manifest)[0], 'human_signoff': 'pending'}
    write(root / 'queue_state.json', state)
    try:
        with (root / 'navigation.log').open('x') as log:
            process = subprocess.Popen(job['command'], cwd=manifest['repo'], env=environment,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='running', child_pid=process.pid)
            write(root / 'queue_state.json', state)
            try:
                code = process.wait(timeout=1800)
            except BaseException:
                terminate_job(process)
                raise
        state['navigation_exit_code'] = code
        if code or not (Path(job['output']) / '.RUN_SUCCESS').is_file():
            raise RuntimeError(f'Navigation failed, actual exit{code}; preserve unscored records')
        verify_sources(manifest)
        state.update(status='auditing')
        write(root / 'queue_state.json', state)
        # The unchanged auditor must use the exact original Isaac CPU libraries.
        subprocess.run(['bash', '-c', 'source "$1"; export CUDA_VISIBLE_DEVICES=""; exec "$2" "$3" --manifest "$4" --job diagnostic --output "$5"',
            'diagnostic-audit', manifest['isaac_setup'], manifest['python'], manifest['auditor'],
            str(args.manifest.resolve()), str(root / 'audit.json')], cwd=manifest['repo'],
            env=environment, check=True)
        result = summarize(manifest, root)
        write(root / 'analysis.json', result)
        state.update(status='completed', finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     complete_records=6, benchmark_scores=False, navigation_acceptance=False)
        state.pop('child_pid', None)
        write(root / 'queue_state.json', state)
        print(json.dumps(result['by_diagnostic_type']), flush=True)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        write(root / 'queue_state.json', state)
        raise


if __name__ == '__main__':
    main()
