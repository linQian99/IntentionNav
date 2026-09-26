"""Run immutable ICLR comparisons serially on GPU0, retaining failed attempts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from hash_navigation_inputs import digest_file
from probe_mtu3d_worker_reset import write
from run_mtu3d_final_controls import cpu_command, execute
from run_r055_scene_validation import clean_environment
from shared_navigation_run_inputs import snapshot
from trim_isaac_derived_cache import trim

ROOT = Path(__file__).resolve().parents[1]
# Inherited exactly from the pinned motion-repair manifest. These small runtime
# selectors must be checked as well as the already-pinned weight files.
EXECUTION_BINDINGS = {
    '/path/to/isaacsim/setup_conda_env.sh': '9f2f6a6ae1c4a372a5e4cb2ca722954246d8f01214a30a0c3e49bc04d15c43d0',
    str(ROOT / 'results/category_instance_review_20260909/mtu3d_weights_path.txt'): 'bdc9d069c322ee4b55213395ae338187c9ad767fab079df8e63fa8c1040279e1',
    str(ROOT / 'eval/agents/navigation_motion.py'): '27d1a972dd3feaefe99de48ce0f575874a0e249b94d147adf7c28be184cc4e1d',
    str(ROOT / 'eval/agents/mtu3d_strict_grid.py'): 'a006edd4c6aeefa3e19c108652d86d78f1ee62fc17691a8f3a1613bc2e0fd3d8',
    str(ROOT / 'eval/simulator/walkable_map.py'): 'a99280043f9ac035b03cea7d14e99b77e6407b1ea824124f2b70bd132ca6418c',
}


def verify(pins):
    for name, expected in pins.items():
        if digest_file(Path(name))[0] != expected:
            raise ValueError(f'Frozen input changed: {name}')


def invocation(m, job, attempt):
    directory = Path(m['output']) / 'attempts' / job['id'] / f'attempt{attempt}'
    env = clean_environment()
    if job['system'] == 'r055':
        name = f"iclr_core_20260919_{job['id']}_attempt{attempt}"
        output = ROOT / 'results/navigation_active_perception_20260825' / name
        env.update(INAV_RUN_NAME=name, INAV_REVIEWED_DATA_ROOT=m['data_root'],
            INAV_SHARED_QUERY_INPUTS=m['shared_inputs'], INAV_SHARED_INPUT_MODE=job['mode'],
            INAV_AGENT_SOURCE_DIR=job['source'] + '/agents', INAV_INPUT_RESOLVER=job['input_resolver'],
            INAV_PHYSICAL_GPU='0', INAV_RESUME_RUN='0', INAV_EXECUTION_SEED=str(job['seed']),
            INAV_RENDER_ANTIALIASING_MODE='2', INAV_RENDER_TICKS='8', INAV_GEOMETRY_SAFE_FRONTIER='1',
            INAV_SELECTION_FILE=job['selections'], INAV_EXPECTED_RECORDS=str(job['count']))
        command = ['bash', job['launcher']]
    else:
        output = directory / 'navigation'
        env.update(INAV_MTU_AGENT_SOURCE=job['source'] + '/agents', INAV_MTU_SHADOW_SUPPORT='0',
                   INAV_MTU_RELATION_ARCHIVE='')
        command = ['bash', job['launcher'], '--items', m['items'], '--episodes', m['episodes'],
            '--selections', job['selections'], '--output', str(output), '--seed', str(job['seed']),
            '--commit-scan', 'multiview', '--scan-schedule', 'initial_then_history',
            '--exploration-mode', 'boundary_commit', '--object-goal-mode', 'fresh_surface_pose',
            '--input-mode', 'shared_' + job['mode'], '--shared-query-inputs', m['shared_inputs']]
    return directory, output, env, command


def validate(m):
    verify(EXECUTION_BINDINGS)
    if m['gpu'] != 0 or m['max_parallel'] != 1 or m['new_navigation_records'] != 1828:
        raise ValueError('Unexpected comparison scope or GPU')
    if sum(j['count'] for j in m['jobs']) != 1828 or len({j['id'] for j in m['jobs']}) != len(m['jobs']):
        raise ValueError('Duplicate or incomplete jobs')
    expected = set(Path(m['data_root'], 'selections500.txt').read_text().splitlines())
    for system, mode in [('mtu3d', 'explicit'), ('r055', 'implicit_category'), ('mtu3d', 'implicit_category')]:
        sids = [sid for j in m['jobs'] if j['stage'] == 'full500' and j['system'] == system and j['mode'] == mode
                for sid in j['selection_ids']]
        if len(sids) != 500 or set(sids) != expected:
            raise ValueError('Incomplete full comparison arm')
    for job in m['jobs']:
        sids = Path(job['selections']).read_text().splitlines()
        if sids != job['selection_ids'] or len(set(sids)) != job['count'] or not set(sids) <= expected:
            raise ValueError('Selection contract changed')
        if job['seed'] != {'a': 20260909, 'b': 20260910}[job['repeat']] or job['source'] != m['sources'][job['system']]:
            raise ValueError('Policy or repeat configuration changed')
    for system in ['r055', 'mtu3d']:
        for mode in ['explicit', 'implicit_category']:
            repeat = [j for j in m['jobs'] if j['stage'] == 'repeat40' and j['system'] == system and j['mode'] == mode]
            if (len(repeat) != 2 or {j['repeat'] for j in repeat} != {'a', 'b'}
                    or any(j['count'] != 40 for j in repeat) or repeat[0]['selection_ids'] != repeat[1]['selection_ids']):
                raise ValueError('Incomplete paired repeat plan')


def check(m):
    validate(m); verify(m['source_hashes'])
    # Full500 input preflights cover every task, while per-job selected IDs are
    # checked above. Repeating 540 identical imports adds no input coverage.
    jobs = []
    for system in ['r055', 'mtu3d']:
        for mode in ['explicit', 'implicit_category']:
            j = dict(next(j for j in m['jobs'] if j['system'] == system and j['mode'] == mode))
            j.update(selections=str(Path(m['data_root']) / 'selections500.txt'), count=500)
            jobs.append(j)
    for job in jobs:
        _, _, env, command = invocation(m, job, 0)
        env['CUDA_VISIBLE_DEVICES'] = ''
        result = subprocess.run(command + ['--check'], cwd=m['repo'], env=env, capture_output=True, text=True, timeout=600)
        if result.returncode:
            raise RuntimeError(result.stderr[-2500:] + result.stdout[-2500:])
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        if not payload['passed'] or payload.get('queries', payload.get('episodes')) != 500 or payload['gpu_initialized'] or payload['output_created']:
            raise ValueError('Actual full input preflight failed')
        print(json.dumps(dict(system=job['system'], mode=job['mode'], preflight=payload)), flush=True)
    print(json.dumps(dict(passed=True, checked_inputs=2000, planned_new_navigation=1828,
                         reused_reference=500, gpu_initialized=False, output_created=False)), flush=True)


def gpu_ready(m, state, path):
    while True:
        value = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=uuid,memory.used,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip().split(',')
        if value[0].strip() != m['gpu_uuid']:
            raise ValueError('Physical GPU0 identity changed')
        fs = os.statvfs(m['output'])
        if fs.f_bavail * fs.f_frsize < 20 * 1024 ** 3:
            raise RuntimeError('Disk below twenty GiB; preserve outputs')
        if int(value[1]) < 500 and int(value[2]) <= 10:
            return
        state['status'] = 'waiting_for_gpu0'; write(path, state)
        time.sleep(15)


def audit(m, job, directory, output, manifest_path, state, state_path):
    archive = output / 'shared_input_snapshot' if job['system'] == 'r055' else directory / 'shared_inputs'
    executed = dict(job, output=str(output))
    audit_manifest = directory / 'audit_manifest.json'
    write(audit_manifest, {**m, 'jobs': [executed]})
    commands = [
        [m['motion_auditor'], '--run', str(output), '--metaroot', m['metaroot'], '--expected-records', str(job['count']), '--output', str(directory / 'motion_audit.json')],
        [m['auditors'][job['system']], *(['--run', str(output), '--source', job['source']] if job['system'] == 'r055'
            else ['--manifest', str(audit_manifest), '--job', job['id']]), '--output', str(directory / 'policy_audit.json')],
        [str(ROOT / 'scripts/shared_navigation_run_inputs.py'), 'audit', '--run', str(output), '--archive', str(archive),
            '--system', job['system'], '--source', job['source'], '--output', str(directory / 'input_audit.json')],
        [str(ROOT / 'scripts/score_iclr_core_comparison.py'), 'job', '--manifest', str(manifest_path), '--job', job['id'],
            '--run', str(output), '--output', str(directory / 'scores.json')],
    ]
    hashes = {}
    for i, command in enumerate(commands):
        code = execute(cpu_command(m, command), cwd=m['repo'], environment=clean_environment(),
            log_path=directory / f'audit_{i}.log', state_path=state_path, state=state, timeout=3600)
        if code:
            raise RuntimeError(f"{job['id']} audit {i} exited {code}")
        path = Path(command[-1]); report = json.loads(path.read_text())
        if not report['passed'] or report['records'] != job['count']:
            raise ValueError('Incomplete per-job evidence')
        hashes.update(report['source_hashes']); hashes[str(path)] = digest_file(path)[0]
    return dict(job=job['id'], stage=job['stage'], system=job['system'], mode=job['mode'], repeat=job['repeat'],
        records=job['count'], output=str(output), scores=str(directory / 'scores.json'), source_hashes=hashes,
        finished_at=datetime.now(timezone.utc).isoformat())


def run(m, manifest_path, auto_resume):
    root = Path(m['output']); state_path = root / 'queue_state.json'
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    manifest_hash = digest_file(manifest_path)[0]
    with (root / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state['manifest_sha256'] != manifest_hash:
                raise ValueError('Submitted manifest changed')
            if state['status'] == 'completed_comparison':
                return
            if not auto_resume or state['boot_id'] == boot or state['status'] in {'failed_preserved', 'deadline_stopped', 'stopped_by_user'}:
                raise RuntimeError('Only abrupt previous-boot interruption may auto-resume')
            for pointer in state['checkpoints'].values():
                verify(pointer['source_hashes'])
                verify(json.loads(Path(pointer['checkpoint']).read_text())['source_hashes'])
            for attempt in state['attempts']:
                if attempt['status'] in {'running', 'auditing'}:
                    attempt['status'] = 'interrupted_previous_boot_preserved'
        else:
            state = dict(status='initializing', checkpoints={}, attempts=[], complete_new_records=0,
                manifest_sha256=manifest_hash, started_at=datetime.now(timezone.utc).isoformat(),
                gpu=0, publication_ready=False, reused_reference_records=500)
        state.update(pid=os.getpid(), boot_id=boot)
        write(state_path, state)
        signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM')))
        try:
            validate(m); verify(m['source_hashes'])
            heavy = json.loads(Path(m['heavy_input_hashes']).read_text())
            reference_pins = json.loads(Path(m['reference_reuse']).read_text())['source_hashes']
            verify(heavy); verify(reference_pins)
            protected = set(m['source_hashes']) | set(heavy) | set(reference_pins)
            for job in m['jobs']:
                if job['id'] in state['checkpoints']:
                    continue
                elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(state['started_at'])).total_seconds()
                if elapsed > m['max_wall_hours'] * 3600:
                    state.update(status='deadline_stopped', active_job=None); write(state_path, state); return
                if job['stage'] != 'smoke' and sum(v['records'] for v in state['checkpoints'].values() if v['stage'] == 'smoke') != 8:
                    raise ValueError('All four integration arms must close first')
                state.update(status='waiting_for_gpu0', active_job=job['id']); write(state_path, state)
                stat = os.statvfs(root)
                if stat.f_bavail * stat.f_frsize < 45 * 1024 ** 3:
                    state['cache_maintenance'] = trim(root / 'cache_maintenance' / f"{job['id']}_{time.time_ns()}", protected)
                    write(state_path, state)
                gpu_ready(m, state, state_path)
                attempt = 1 + sum(a['job'] == job['id'] for a in state['attempts'])
                if attempt > m['maximum_attempts']:
                    raise RuntimeError('Interrupted-attempt budget exhausted')
                directory, output, env, command = invocation(m, job, attempt)
                if output.exists() or directory.exists():
                    raise FileExistsError('Attempt output already exists')
                directory.mkdir(parents=True)
                if digest_file(manifest_path)[0] != manifest_hash:
                    raise ValueError('Submitted manifest changed')
                verify(m['source_hashes'])
                verify(EXECUTION_BINDINGS)
                if job['system'] == 'mtu3d':
                    snapshot(Path(m['shared_inputs']), Path(m['items']), Path(job['selections']), job['mode'], directory / 'shared_inputs')
                entry = dict(job=job['id'], attempt=attempt, output=str(output), status='running', started_at=datetime.now(timezone.utc).isoformat())
                state['attempts'].append(entry); state['status'] = 'running'; write(state_path, state)
                code = execute(command, cwd=m['repo'], environment=env, log_path=directory / 'navigation.log',
                    state_path=state_path, state=state, timeout=6 * 3600)
                entry['exit_code'] = code
                if code or not (output / '.RUN_SUCCESS').is_file():
                    raise RuntimeError(f"{job['id']} navigation exit {code}, missing closure or failed worker")
                entry['status'] = state['status'] = 'auditing'; write(state_path, state)
                if digest_file(manifest_path)[0] != manifest_hash:
                    raise ValueError('Submitted manifest changed')
                verify(m['source_hashes'])
                receipt = audit(m, job, directory, output, manifest_path, state, state_path)
                path = directory / 'checkpoint.json'; write(path, receipt)
                state['checkpoints'][job['id']] = dict(checkpoint=str(path), stage=job['stage'], records=job['count'],
                    source_hashes={str(path): digest_file(path)[0]})
                entry['status'] = 'completed'
                state['complete_new_records'] = sum(c['records'] for c in state['checkpoints'].values())
                state['status'] = 'checkpoint_complete'; write(state_path, state)
                print(json.dumps(dict(closed=job['id'], complete_new=state['complete_new_records'], planned_new=1828)), flush=True)
            command = [str(ROOT / 'scripts/score_iclr_core_comparison.py'), 'compare', '--manifest', str(manifest_path), '--output', str(root / 'comparison.json')]
            code = execute(cpu_command(m, command), cwd=m['repo'], environment=clean_environment(), log_path=root / 'comparison.log',
                state_path=state_path, state=state, timeout=7200)
            if code:
                raise RuntimeError('Final paired comparison failed')
            state.update(status='completed_comparison', active_job=None, finished_at=datetime.now(timezone.utc).isoformat())
            write(state_path, state)
        except BaseException as error:
            state.update(status='failed_preserved', error=repr(error), failed_at=datetime.now(timezone.utc).isoformat())
            write(state_path, state)
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--check', action='store_true')
    p.add_argument('--auto-resume', action='store_true')
    a = p.parse_args(); path = a.manifest.resolve(); m = json.loads(path.read_text())
    if a.check:
        check(m)
    else:
        run(m, path, a.auto_resume)
