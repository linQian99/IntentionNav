"""Run a bounded two-system, two-input integration smoke on GPU0 serially.

Wait for the existing dev40 queue to close successfully before claiming GPU0.
Only eight trajectories are authorized by this manifest, no promotion stage.
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

from hash_navigation_inputs import digest_file
from probe_mtu3d_worker_reset import write
from run_baseline_acceptance import terminate_job, wait_for_gpu
from shared_navigation_run_inputs import audit, snapshot


def verify(manifest: dict) -> None:
    for name, expected in manifest['source_hashes'].items():
        if digest_file(Path(name))[0] != expected:
            raise ValueError(f'Source/model/input changed: {name}')


def cpu(manifest: dict, command: list[str]) -> list[str]:
    return ['bash', '-c', 'source "$1" && shift && export CUDA_VISIBLE_DEVICES="" && exec "$@"',
            'shared-input-audit', manifest['isaac_setup'], manifest['python'], *command]


def dependency_complete(manifest: dict) -> bool:
    root = Path(manifest['dependency'])
    state = json.loads((root / 'queue_state.json').read_text())
    status = state['status']
    if status in {'failed', 'completed_requires_method_review'}:
        raise RuntimeError(f'Development dependency ended in {status}; review before shared smoke')
    if status != 'completed_development':
        pid = state['pid']
        if (not Path(f'/proc/{pid}/cmdline').exists()
                or str(root / 'manifest.json') not in Path(f'/proc/{pid}/cmdline').read_text()):
            raise RuntimeError('Development dependency is incomplete and its queue is absent')
        return False
    # The old queue writes its status just before final comparison. Wait for
    # the actual parent to exit and for the independent comparison artifact.
    pid = state['pid']
    cmdline = Path(f'/proc/{pid}/cmdline')
    if cmdline.exists() and str(root / 'manifest.json') in cmdline.read_text():
        return False
    if state['new_complete_records'] != 80 or state['failed_jobs'] or state['skipped']:
        raise ValueError('Dependency terminal counts are incomplete')
    comparison = root / 'comparison/comparison.json'
    if not comparison.is_file():
        raise ValueError('Development dependency has no final comparison')
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.parent
    if (root / 'queue_state.json').exists():
        raise FileExistsError('Already submitted; inspect the existing queue')
    jobs = manifest['jobs']
    selected = [s.strip() for s in Path(manifest['selections']).read_text().splitlines()
                if s.strip() and not s.startswith('#')]
    if (manifest['gpu'] != 0 or manifest['max_parallel'] != 1 or len(set(selected)) != 2
            or len(selected) != 2 or len(jobs) != 4
            or {(j['system'], j['mode']) for j in jobs} !=
                {(s, m) for s in ['r055', 'mtu3d'] for m in ['explicit', 'implicit_category']}
            or any(j['count'] != 2 or j['seed'] != 20260910 for j in jobs)):
        raise ValueError('Require the bounded8-episode shared-input smoke')
    verify(manifest)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    for job in jobs:
        if Path(job['output']).exists():
            raise FileExistsError(job['output'])
        result = subprocess.run(job['command'] + ['--check'], cwd=manifest['repo'],
            env={**environment, **job['environment'], 'CUDA_VISIBLE_DEVICES': ''},
            check=True, capture_output=True, text=True, timeout=300)
        check = json.loads(result.stdout.strip().splitlines()[-1])
        if (not check['passed'] or check.get('queries', check.get('episodes')) != 2
                or check['gpu_initialized'] or check['output_created']):
            raise ValueError(f'Unexpected actual CLI check: {check}')
        print(json.dumps({'job': job['id'], 'preflight': check}), flush=True)
    if args.check:
        print(json.dumps({'passed': True, 'jobs': 4, 'maximum_new_episodes': 8,
            'gpu_initialized': False, 'output_created': False,
            'source_pins': len(manifest['source_hashes']),
            'manifest_sha256': digest_file(args.manifest)[0]}))
        return
    state = {'status': 'waiting_for_development', 'pid': os.getpid(), 'gpu': 0,
             'maximum_new_episodes': 8, 'complete_records': 0, 'completed': [],
             'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
             'manifest_sha256': digest_file(args.manifest)[0],
             'human_signoff': 'pending', 'publication_ready': False}
    with (root / 'queue_state.json').open('x') as stream:
        json.dump(state, stream, indent=2)
    try:
        deadline = time.monotonic() + 12 * 3600
        while not dependency_complete(manifest):
            if time.monotonic() > deadline:
                raise TimeoutError('Development dependency did not finish within12h')
            time.sleep(15)
        for job in jobs:
            wait_for_gpu(root, state, job)
            disk = os.statvfs(root)
            if disk.f_bavail * disk.f_frsize < 20 * 1024**3:
                raise RuntimeError('Less than20GiB free')
            verify(manifest)
            run = Path(job['output'])
            archive = run / 'shared_input_snapshot' if job['system'] == 'r055' else root / f"inputs_{job['id']}"
            if job['system'] == 'mtu3d':
                snapshot(Path(manifest['shared_inputs']), Path(manifest['items']),
                         Path(manifest['selections']), job['mode'], archive)
            with (root / f"{job['id']}.log").open('x') as log:
                process = subprocess.Popen(job['command'], cwd=manifest['repo'],
                    env={**environment, **job['environment']}, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                state.update(status='running', active_job=job['id'], child_pid=process.pid)
                write(root / 'queue_state.json', state)
                try:
                    code = process.wait(timeout=3600)
                except BaseException:
                    terminate_job(process)
                    raise
            state.pop('child_pid', None)
            state['last_exit_code'] = code
            if code or not (run / '.RUN_SUCCESS').is_file():
                raise RuntimeError(f"{job['id']} incomplete/runtime exit={code}; unscored")
            verify(manifest)
            state.update(status='auditing')
            write(root / 'queue_state.json', state)
            result = audit(run, archive, job['system'], Path(job['source']))
            write(root / f"input_audit_{job['id']}.json", result)
            if job['system'] == 'mtu3d':
                action_audit = root / f"action_audit_{job['id']}.json"
                subprocess.run(cpu(manifest, [manifest['auditor'], '--manifest', str(args.manifest),
                    '--job', job['id'], '--output', str(action_audit)]), check=True,
                    cwd=manifest['repo'], env=environment, timeout=1800)
                subprocess.run(cpu(manifest, [manifest['consumption_auditor'], '--run', str(run),
                    '--action-audit', str(action_audit), '--output',
                    str(root / f"consumption_{job['id']}.json")]), check=True,
                    cwd=manifest['repo'], env=environment, timeout=1800)
            analysis = root / f"analysis_{job['id']}"
            subprocess.run(cpu(manifest, [manifest['scorer'], '--run', str(run),
                '--goals', manifest['goals'], '--episodes', manifest['episodes'],
                '--primary', manifest['primary'], '--selections', manifest['selections'],
                '--output', str(analysis)]), check=True, cwd=manifest['repo'],
                env=environment, timeout=1800)
            summary = json.loads((analysis / 'summary.json').read_text())
            if summary['complete_records'] != 2 or summary['primary_count'] != 2:
                raise ValueError('Unexpected scoring cohort')
            state['completed'].append({'job': job['id'], 'analysis': str(analysis),
                                       'exit_code': code})
            state['complete_records'] += 2
            write(root / 'queue_state.json', state)
        verify(manifest)
        state.update(status='completed_integration_smoke',
            finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            scope='Input/launch/closure/scoring integration only;8episodes provide no method ranking')
        state.pop('active_job', None)
        state.pop('next_job', None)
        write(root / 'queue_state.json', state)
        print(json.dumps(state), flush=True)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc),
                     failed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(root / 'queue_state.json', state)
        raise


if __name__ == '__main__':
    main()
