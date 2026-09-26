"""Run unchanged calibrated R055 twice on the declared MTU validation cohort."""
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
from run_mtu3d_history_visible import dependency_ready
from run_shared_navigation_smoke import cpu, verify


def ids(path):
    return [s.strip() for s in Path(path).read_text().splitlines()
            if s.strip() and not s.startswith('#')]


def clean_environment():
    """Prevent inherited experimental arms from changing the frozen launcher."""
    prefixes = ('INAV_', 'VM_', 'AUTO_REORIENT', 'FALLBACK_', 'MEMORY_HINT_',
                'ROBUSTNESS_', 'PP_API_')
    names = {'ANGULAR_SPREAD', 'REPLAN_ENABLE', 'STRICT_VLM_FAILURE', 'ISS_RENDER_TICKS'}
    result = {k: v for k, v in os.environ.items() if not k.startswith(prefixes) and k not in names}
    result.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4',
                  OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    return result


def validate_plan(manifest):
    selected = ids(manifest['selections'])
    jobs = manifest['jobs']
    partner = json.loads(Path(manifest['partner_manifest']).read_text())
    split = json.loads(Path(partner['split_manifest']).read_text())
    if (manifest['gpu'] != 0 or manifest['max_parallel'] != 1
            or len(selected) != 40 or len(set(selected)) != 40
            or not set(selected) <= set(ids(manifest['primary']))
            or {x['selection_id'] for x in split['selected']} != set(selected)
            or len({x['scene_id'] for x in split['selected']}) != 40
            or {x['scene_id'] for x in split['selected']} & set(split['development_scenes'])
            or split['uses_navigation_outcomes']
            or [j['id'] for j in jobs] != ['r055_a', 'r055_b']
            or [j['seed'] for j in jobs] != [20260909, 20260910]):
        raise ValueError('Require the original40scene cohort and two unchanged seeds')
    for key in ['items', 'episodes', 'goals', 'primary', 'selections']:
        if manifest[key] != partner[key]:
            raise ValueError(f'Two-system input/evaluation mismatch: {key}')
    for job in jobs:
        env = job['environment']
        if (job['system'] != 'r055' or job['count'] != 40 or job['mode'] != 'explicit'
                or job['source'] != manifest['source']
                or job['command'] != ['bash', manifest['launcher']]
                or env['INAV_AGENT_SOURCE_DIR'] != str(Path(manifest['source']) / 'agents')
                or env['INAV_PHYSICAL_GPU'] != '0' or env['INAV_EXPECTED_RECORDS'] != '40'
                or env['INAV_SHARED_QUERY_INPUTS'] != manifest['shared_inputs']
                or env['INAV_SHARED_INPUT_MODE'] != 'explicit'
                or env['INAV_SELECTION_FILE'] != manifest['selections']
                or env['INAV_EXECUTION_SEED'] != str(job['seed'])
                or env['INAV_RESUME_RUN'] != '0'
                or Path(job['output']).name != env['INAV_RUN_NAME']):
            raise ValueError('R055 invocation differs from frozen comparison contract')


def close_partner(manifest, output):
    """Record actual closed partner evidence only after its parent has exited."""
    partner = Path(manifest['partner_manifest']).parent
    state = json.loads((partner / 'queue_state.json').read_text())
    if (state['status'] != 'completed_validation' or state['new_complete_records'] != 80
            or state['failed_jobs'] or state['skipped']
            or {x['job'] for x in state['completed']} != {'mtu3d_a', 'mtu3d_b'}):
        raise ValueError('Partner did not complete its two audited arms')
    pins = {}
    for arm in state['completed']:
        analysis = Path(arm['analysis'])
        summary = json.loads((analysis / 'summary.json').read_text())
        audit = json.loads((partner / f"audit_{arm['job']}.json").read_text())
        if (arm['exit_code'] != 0 or summary['complete_records'] != 40
                or summary['primary_count'] != 40 or audit['records'] != 40 or not audit['passed']):
            raise ValueError('Partner score or independent audit is incomplete')
        for evidence in [summary, audit]:
            for name, expected in evidence['source_hashes'].items():
                if digest_file(Path(name))[0] != expected:
                    raise ValueError(f'Partner evidence changed: {name}')
                if name in pins and pins[name] != expected:
                    raise ValueError('Conflicting partner evidence')
                pins[name] = expected
        for path in [analysis / 'summary.json', analysis / 'per_item.csv',
                     partner / f"audit_{arm['job']}.json"]:
            pins[str(path.resolve())] = digest_file(path)[0]
    for path in [partner / 'queue_state.json', partner / 'manifest.json']:
        pins[str(path.resolve())] = digest_file(path)[0]
    write(output, dict(passed=True,complete_records=80,source_hashes=pins,
                      scope='Closed provisional partner evidence; not formal method acceptance'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.parent
    if (root / 'queue_state.json').exists():
        raise FileExistsError('Already submitted; inspect the existing process')
    validate_plan(manifest)
    environment = clean_environment()
    state = dict(status='startup_verification', pid=os.getpid(), gpu=0,
        maximum_new_episodes=80, complete_records=0, completed=[],
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        manifest_sha256=digest_file(args.manifest)[0], publication_ready=False,
        human_signoff='pending', navigation_acceptance=False)
    if not args.check:
        with (root / 'queue_state.json').open('x') as stream:
            json.dump(state, stream, indent=2)
    try:
        verify(manifest)
        for job in manifest['jobs']:
            if Path(job['output']).exists():
                raise FileExistsError(job['output'])
            result = subprocess.run(job['command'] + ['--check'], cwd=manifest['repo'],
                env={**environment, **job['environment'], 'CUDA_VISIBLE_DEVICES': ''},
                check=True, capture_output=True, text=True, timeout=300)
            check = json.loads(result.stdout.strip().splitlines()[-1])
            if (not check['passed'] or check['queries'] != 40
                    or check['gpu_initialized'] or check['output_created']):
                raise ValueError('Unexpected actual R055 entrypoint preflight')
            print(json.dumps({'job': job['id'], 'preflight': check}), flush=True)
        if args.check:
            print(json.dumps(dict(passed=True, jobs=2, maximum_new_episodes=80,
                source_pins=len(manifest['source_hashes']), gpu_initialized=False,
                output_created=False, manifest_sha256=digest_file(args.manifest)[0])))
            return
        state.update(status='waiting_for_partner', waiting_on=manifest['dependency']['root'])
        write(root / 'queue_state.json', state)
        deadline = time.monotonic() + 14 * 3600
        while not dependency_ready(manifest['dependency']):
            if time.monotonic() > deadline:
                raise TimeoutError('Partner wait exceeded14hours')
            time.sleep(15)
        state.update(status='auditing_partner')
        write(root / 'queue_state.json', state)
        close_partner(manifest, root / 'partner_closure.json')
        state.pop('waiting_on', None)
        for job in manifest['jobs']:
            wait_for_gpu(root, state, job)
            disk = os.statvfs(root)
            if disk.f_bavail * disk.f_frsize < 20 * 1024**3:
                raise RuntimeError('Less than20GiBfree')
            state.update(status='verifying_sources', active_job=job['id'])
            write(root / 'queue_state.json', state)
            verify(manifest)
            run = Path(job['output'])
            with (root / f"{job['id']}.log").open('x') as log:
                process = subprocess.Popen(job['command'], cwd=manifest['repo'],
                    env={**environment, **job['environment']}, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                state.update(status='running', child_pid=process.pid)
                write(root / 'queue_state.json', state)
                try:
                    code = process.wait(timeout=6 * 3600)
                except BaseException:
                    terminate_job(process)
                    raise
            state.pop('child_pid', None)
            state['last_exit_code'] = code
            if code or not (run / '.RUN_SUCCESS').is_file():
                raise RuntimeError(f"{job['id']} incomplete/runtime exit={code}; unscored")
            state.update(status='auditing')
            write(root / 'queue_state.json', state)
            verify(manifest)
            audit_path = root / f"audit_{job['id']}.json"
            subprocess.run(cpu(manifest, [manifest['auditor'], '--run', str(run),
                '--source', manifest['source'], '--output', str(audit_path)]),
                check=True, cwd=manifest['repo'], env=environment, timeout=1800)
            audit = json.loads(audit_path.read_text())
            if not audit['passed'] or audit['records'] != 40:
                raise ValueError('Incomplete R055 acquired-observation audit')
            analysis = root / f"analysis_{job['id']}"
            subprocess.run(cpu(manifest, [manifest['scorer'], '--run', str(run),
                '--goals', manifest['goals'], '--episodes', manifest['episodes'],
                '--primary', manifest['primary'], '--selections', manifest['selections'],
                '--output', str(analysis)]), check=True, cwd=manifest['repo'],
                env=environment, timeout=1800)
            summary = json.loads((analysis / 'summary.json').read_text())
            if summary['complete_records'] != 40 or summary['primary_count'] != 40:
                raise ValueError('Incomplete R055 scoring cohort')
            state['completed'].append(dict(job=job['id'], analysis=str(analysis),
                                           audit=str(audit_path), exit_code=code))
            state['complete_records'] += 40
            write(root / 'queue_state.json', state)
        verify(manifest)
        state.update(status='completed_r055_validation',
            finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        state.pop('active_job', None)
        state.pop('next_job', None)
        write(root / 'queue_state.json', state)
        print(json.dumps(state))
    except BaseException as exc:
        if not args.check:
            state.update(status='failed', error=repr(exc),
                         failed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write(root / 'queue_state.json', state)
        raise


if __name__ == '__main__':
    main()
