"""Run two finite basic-control repeats of the exact qualified MTU policy."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess

from hash_navigation_inputs import digest_file
from probe_mtu3d_worker_reset import write
from run_baseline_acceptance import terminate_job, wait_for_gpu
from run_visible_navigation_diagnostics import summarize, verify_sources


def execute(command, *, cwd, environment, log_path, state_path, state, timeout=1800):
    """Protect every operation after spawn, including state-file writes."""
    with Path(log_path).open('x') as log:
        child = subprocess.Popen(command, cwd=cwd, env=environment, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            state['child_pid'] = child.pid
            write(state_path, state)
            return child.wait(timeout=timeout)
        except BaseException:
            terminate_job(child)
            raise
        finally:
            state.pop('child_pid', None)


def cpu_command(manifest, arguments):
    return ['bash', '-c', 'source "$1" && shift && export CUDA_VISIBLE_DEVICES="" && exec "$@"',
            'final-controls-cpu', manifest['isaac_setup'], manifest['python'], *arguments]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--summarize-job', choices=['visible_a', 'visible_b'])
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.parent
    if args.summarize_job:
        job = next(j for j in manifest['jobs'] if j['id'] == args.summarize_job)
        result = summarize({**manifest, 'jobs': [job]}, root)
        with (root / f"analysis_{job['id']}.json").open('x') as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result['by_diagnostic_type']))
        return
    state_path = root / 'queue_state.json'
    if state_path.exists():
        raise FileExistsError('Already submitted; follow the existing queue')
    if (manifest['gpu'] != 0 or manifest['max_parallel'] != 1
            or [(j['id'], j['seed'], j['count']) for j in manifest['jobs']]
            != [('visible_a', 20260909, 6), ('visible_b', 20260910, 6)]):
        raise ValueError('Require exactly two declared six-case repeats onGPU0')
    verify_sources(manifest)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    for job in manifest['jobs']:
        if Path(job['output']).exists():
            raise FileExistsError(job['output'])
        for flag, expected in [('--seed', str(job['seed'])), ('--output', job['output']),
                               ('--scan-schedule', 'initial_then_history'),
                               ('--object-goal-mode', 'fresh_surface_pose'),
                               ('--commit-scan', 'multiview'), ('--exploration-mode', 'boundary_commit')]:
            if job['command'].count(flag) != 1 or job['command'][job['command'].index(flag)+1] != expected:
                raise ValueError(f'Unexpected {flag}')
        if job['environment']['INAV_MTU_AGENT_SOURCE'] != str(Path(manifest['source']) / 'agents'):
            raise ValueError('Wrong frozen source')
        check = subprocess.run(job['command'] + ['--check'], cwd=manifest['repo'],
            env={**environment, **job['environment'], 'CUDA_VISIBLE_DEVICES': ''},
            capture_output=True, text=True, check=True, timeout=300)
        result = json.loads(check.stdout.strip().splitlines()[-1])
        if not result['passed'] or result['episodes'] != 6 or result['gpu_initialized'] or result['output_created']:
            raise ValueError('Unexpected actual entrypoint preflight')
        print(json.dumps(result), flush=True)
    if args.check:
        print(json.dumps(dict(passed=True, jobs=2, maximum_episodes=12,
            source_pins=len(manifest['source_hashes']), gpu_initialized=False, output_created=False,
            manifest_sha256=digest_file(args.manifest)[0])))
        return
    state = dict(status='starting', pid=os.getpid(), gpu=0, maximum_episodes=12,
                 complete_records=0, completed=[], publication_ready=False,
                 started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 manifest_sha256=digest_file(args.manifest)[0])
    with state_path.open('x') as stream:
        json.dump(state, stream, indent=2)
    try:
        for job in manifest['jobs']:
            wait_for_gpu(root, state, job)
            disk = os.statvfs(root)
            if disk.f_bavail * disk.f_frsize < 20 * 1024**3:
                raise RuntimeError('Less than20GiB free')
            verify_sources(manifest)
            state.update(status='running', active_job=job['id'])
            code = execute(job['command'], cwd=manifest['repo'],
                environment={**environment, **job['environment']}, log_path=root/f"{job['id']}.log",
                state_path=state_path, state=state)
            state['last_exit_code'] = code
            if code or not (Path(job['output']) / '.RUN_SUCCESS').exists():
                raise RuntimeError(f'Navigation exit{code}; incomplete arm remains unscored')
            verify_sources(manifest)
            state['status'] = 'auditing'
            tasks = [(manifest['auditor'], ['--manifest', str(args.manifest), '--job', job['id'],
                       '--output', str(root / f"audit_{job['id']}.json")]),
                     (str(Path(__file__).resolve()), ['--manifest', str(args.manifest),
                       '--summarize-job', job['id']])]
            for index, (script, arguments) in enumerate(tasks):
                code = execute(cpu_command(manifest, [script, *arguments]), cwd=manifest['repo'],
                    environment=environment, log_path=root/f"{job['id']}_audit_{index}.log",
                    state_path=state_path, state=state)
                if code:
                    raise RuntimeError(f'Audit/scoring exited{code}')
            report = json.loads((root / f"analysis_{job['id']}.json").read_text())
            audit = json.loads((root / f"audit_{job['id']}.json").read_text())
            if not report['complete'] or len(report['rows']) != 6 or not audit['passed'] or audit['records'] != 6:
                raise ValueError('Incomplete six-case audit/scoring')
            state['completed'].append(dict(job=job['id'], exit_code=0,
                                          results=report['by_diagnostic_type']))
            state['complete_records'] += 6
            write(state_path, state)
        verify_sources(manifest)
        state.update(status='completed_controls', active_job=None,
                     finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     navigation_acceptance=False, actual_rgb_review='pending')
        state.pop('next_job', None)
        write(state_path, state)
        print(json.dumps(state), flush=True)
    except BaseException as exc:
        state.update(status='failed', error=repr(exc),
                     failed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(state_path, state)
        raise


if __name__ == '__main__':
    main()
