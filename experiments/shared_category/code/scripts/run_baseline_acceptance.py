"""Run frozen development arms; runtime failure blocks dependent arms only."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from run_two_navigation_dev40 import PYTHON, REPO, check_pins, write


def terminate_job(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=20)


def wait_for_gpu(root: Path, state: dict, job: dict) -> None:
    deadline = time.monotonic() + 12 * 3600
    while True:
        memory = [int(value) for value in subprocess.check_output([
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,noheader,nounits'], text=True).splitlines()]
        state.update(status='waiting_for_gpu', next_job=job['id'], gpu_memory_mib=memory)
        write(root / 'queue_state.json', state)
        if all(value < 500 for value in memory):
            return
        if time.monotonic() > deadline:
            raise TimeoutError('GPU availability wait exceeded 12 hours')
        time.sleep(15)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    # Reuse all existing CPU input, source, smoke and output-immutability gates.
    subprocess.run([PYTHON, str(REPO / 'scripts/run_two_navigation_dev40.py'),
                    '--manifest', str(args.manifest), '--check'], check=True, cwd=REPO)
    if args.check:
        return
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.resolve().parent
    state = {'status': 'waiting_for_gpu', 'pid': os.getpid(), 'gpu': 0,
             'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
             'completed': [], 'skipped': [], 'failed_jobs': [],
             'submitted_at': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write(root / 'queue_state.json', state)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    results = {}
    try:
        for job in manifest['jobs']:
            if job['repeat'] == 'b':
                first = results.get(job['system'] + '_a')
                reason = None
                if first is None:
                    reason = 'first_arm_runtime_failure_requires_repair'
                elif job['system'] == 'mtu3d' and first['counts']['Any_SR_hit'] == 0:
                    reason = 'first_arm_zero_success_requires_method_review'
                if reason:
                    state['skipped'].append({'job': job['id'], 'reason': reason})
                    write(root / 'queue_state.json', state)
                    continue
            check_pins(manifest)
            wait_for_gpu(root, state, job)
            disk = os.statvfs(REPO)
            if disk.f_bavail * disk.f_frsize < 20 * 1024**3:
                raise RuntimeError('Less than 20 GiB free')
            check_pins(manifest)
            with (root / f'{job["id"]}.log').open('x') as log:
                process = subprocess.Popen(job['command'], cwd=REPO,
                    env={**environment, **job['environment']}, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                state.update(status='running', active_job=job['id'], child_pid=process.pid)
                write(root / 'queue_state.json', state)
                timed_out = False
                try:
                    code = process.wait(timeout=6 * 3600)
                except subprocess.TimeoutExpired:
                    terminate_job(process)
                    code = process.returncode
                    timed_out = True
                except BaseException:
                    terminate_job(process)
                    raise
            check_pins(manifest)
            run = Path(job['output'])
            if timed_out or code != 0 or not (run / '.RUN_SUCCESS').is_file():
                detail = {'job': job['id'], 'returncode': code, 'timed_out': timed_out,
                          'has_complete_marker': (run / '.RUN_SUCCESS').is_file(),
                          'output': str(run), 'classification': 'runtime_failure_not_navigation_score'}
                if (run / 'summary.json').is_file():
                    detail['run_summary'] = json.loads((run / 'summary.json').read_text())
                state['failed_jobs'].append(detail)
                write(root / 'queue_state.json', state)
                continue
            # Source, protocol or record-integrity errors still stop the whole
            # queue. They are never silently classified as a method failure.
            output = root / f'analysis_{job["id"]}'
            subprocess.run([PYTHON, str(REPO / 'scripts/analyze_navigation_system.py'),
                '--run', str(run), '--goals', manifest['goals'], '--episodes', manifest['episodes'],
                '--primary', manifest['primary'], '--selections', manifest['selections'],
                '--output', str(output)], cwd=REPO, env=environment, check=True)
            results[job['id']] = json.loads((output / 'summary.json').read_text())
            state['completed'].append({'job': job['id'], 'analysis': str(output)})
            write(root / 'queue_state.json', state)
        state.update(status='completed_requires_method_review' if state['skipped'] or
                     state['failed_jobs'] else 'completed_development',
                     publication_ready=False, human_signoff='pending')
        state.pop('active_job', None)
        state.pop('child_pid', None)
        write(root / 'queue_state.json', state)
        subprocess.run([PYTHON, str(REPO / 'scripts/compare_navigation_systems.py'),
                        '--queue', str(root), '--output', str(root / 'comparison')],
                       cwd=REPO, env=environment, check=True)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        write(root / 'queue_state.json', state)
        raise


if __name__ == '__main__':
    main()
