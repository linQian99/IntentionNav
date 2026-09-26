"""Run frozen R055/MTU3D development arms serially on GPU0 and reconcile them."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
PYTHON = '/path/to/workspace/envs/goodnav/bin/python'


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def check_pins(manifest: dict) -> None:
    for name, expected in manifest['source_hashes'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen source/data changed: {name}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    check_pins(manifest)
    root = args.manifest.resolve().parent
    witness = json.loads(Path(manifest['gpu_witness']).read_text())
    if not witness['gpu_witness'] or not witness['model_decision']:
        raise ValueError('Actual model witness is required')
    smoke = Path(manifest['smoke_run'])
    if not (smoke / '.RUN_SUCCESS').is_file():
        raise ValueError('Complete navigation smoke is required')
    for filename in ['iss_env.py', 'walkable_map.py']:
        if (REPO / 'eval_paper_baseline_r055_frozen_20260909/simulator' / filename).read_bytes() != \
                (REPO / 'eval_mtu3d_frozen_20260909/simulator' / filename).read_bytes():
            raise ValueError(f'Simulator mismatch: {filename}')
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    for job in manifest['jobs']:
        if Path(job['output']).exists():
            raise FileExistsError(f'Use a new queue/output; no implicit resume: {job["output"]}')
    mtu = next(j for j in manifest['jobs'] if j['system'] == 'mtu3d')
    subprocess.run(mtu['command'] + ['--check'], cwd=REPO,
                   env={**environment, **mtu['environment']}, check=True)
    if args.check:
        print(json.dumps({'passed': True, 'jobs': len(manifest['jobs']),
                          'maximum_episodes': 40 * len(manifest['jobs']),
                          'primary_per_arm': 37, 'source_hashes': len(manifest['source_hashes']),
                          'gpu_initialized': False, 'output_created': False}))
        return
    state = {'status': 'waiting_for_gpu', 'pid': os.getpid(), 'gpu': 0,
             'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
             'completed': [], 'skipped': [], 'submitted_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write(root / 'queue_state.json', state)
    results = {}
    try:
        for job in manifest['jobs']:
            # A zero-success first MTU arm is a reason to inspect adaptation,
            # not to spend another arm pretending the comparator is accepted.
            if job['system'] == 'mtu3d' and job['repeat'] == 'b':
                first = results['mtu3d_a']
                if first['counts']['Any_SR_hit'] == 0:
                    state['skipped'].append({'job': job['id'], 'reason': 'first_arm_zero_success_requires_method_review'})
                    write(root / 'queue_state.json', state)
                    continue
            check_pins(manifest)
            deadline = time.monotonic() + 12 * 3600
            while True:
                memory = [int(s) for s in subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()]
                state.update(status='waiting_for_gpu', next_job=job['id'], gpu_memory_mib=memory)
                write(root / 'queue_state.json', state)
                if all(value < 500 for value in memory):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('GPU availability wait exceeded 12 hours')
                time.sleep(15)
            if os.statvfs(REPO).f_bavail * os.statvfs(REPO).f_frsize < 20 * 1024**3:
                raise RuntimeError('Less than 20 GiB free for development outputs')
            check_pins(manifest)
            with (root / f'{job["id"]}.log').open('x') as log:
                process = subprocess.Popen(job['command'], cwd=REPO,
                    env={**environment, **job['environment']}, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                state.update(status='running', active_job=job['id'], child_pid=process.pid)
                write(root / 'queue_state.json', state)
                try:
                    code = process.wait(timeout=6 * 3600)
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=20)
                    raise
            if code:
                raise RuntimeError(f'{job["id"]} exited {code}; all partial outputs retained')
            check_pins(manifest)
            output = root / f'analysis_{job["id"]}'
            command = [PYTHON, str(REPO / 'scripts/analyze_navigation_system.py'), '--run', job['output'],
                       '--goals', manifest['goals'], '--episodes', manifest['episodes'],
                       '--primary', manifest['primary'], '--selections', manifest['selections'],
                       '--output', str(output)]
            subprocess.run(command, cwd=REPO, env=environment, check=True)
            results[job['id']] = json.loads((output / 'summary.json').read_text())
            state['completed'].append({'job': job['id'], 'analysis': str(output)})
            write(root / 'queue_state.json', state)
        state['status'] = 'completed_requires_method_review' if state['skipped'] else 'completed_development'
        state['publication_ready'] = False
        state['human_signoff'] = 'pending'
        write(root / 'queue_state.json', state)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        write(root / 'queue_state.json', state)
        raise


if __name__ == '__main__':
    main()
