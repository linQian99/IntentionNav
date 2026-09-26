"""Dispatch one six-case acquired-history test after both existing queues close.

No development/main expansion. A failed dependency, child, provenance or
scoring check stops this queue and leaves original artifacts available.
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
from run_baseline_acceptance import terminate_job,wait_for_gpu
from run_visible_navigation_diagnostics import verify_sources,summarize


def dependency_ready(dependency: dict) -> bool:
    root=Path(dependency['root']);path=root/'queue_state.json'
    state=json.loads(path.read_text());status=state['status']
    if status in {'failed','completed_requires_method_review'}:
        raise RuntimeError(f'Dependency {root.name} ended in {status}; review before new method test')
    proc=Path('/proc')/str(state['pid'])/'cmdline'
    try:command=proc.read_text()
    except FileNotFoundError:command=''
    live=str(root/'manifest.json') in command
    if status!=dependency['terminal_status']:
        if not live:raise RuntimeError(f'Incomplete dependency process is absent: {root}')
        return False
    if state.get(dependency['count_field'])!=dependency['count']:
        raise ValueError('Dependency terminal episode count mismatch')
    if live:return False
    for relative in dependency['required_outputs']:
        if not (root/relative).is_file():raise ValueError(f'Missing dependency closure output: {relative}')
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args();args.manifest=args.manifest.resolve()
    manifest=json.loads(args.manifest.read_text());root=args.manifest.parent
    if (root/'queue_state.json').exists():raise FileExistsError('Already submitted; inspect existing queue')
    job,=manifest['jobs'];run=Path(job['output'])
    if manifest['gpu']!=0 or manifest['max_parallel']!=1 or job['count']!=6 or run.exists():
        raise ValueError('Require six fresh component episodes on GPU0 serially')
    command=job['command']
    if command[command.index('--scan-schedule')+1]!='initial_then_history':
        raise ValueError('Require the declared acquired-history schedule')
    verify_sources(manifest)
    environment=dict(os.environ,**job['environment'],PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
    completed=subprocess.run(command+['--check'],cwd=manifest['repo'],
        env={**environment,'CUDA_VISIBLE_DEVICES':''},capture_output=True,text=True,check=True,timeout=300)
    check=json.loads(completed.stdout.strip().splitlines()[-1])
    if not check['passed'] or check['episodes']!=6 or check['gpu_initialized'] or check['output_created']:
        raise ValueError('Unexpected actual CLI preflight')
    print(json.dumps(check),flush=True)
    if args.check:
        print(json.dumps({'passed':True,'episodes':6,'source_pins':len(manifest['source_hashes']),
            'gpu_initialized':False,'output_created':False,'benchmark_scores':False,
            'manifest_sha256':digest_file(args.manifest)[0]}));return
    state={'status':'waiting_for_dependencies','pid':os.getpid(),'gpu':0,'maximum_episodes':6,
        'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'manifest_sha256':digest_file(args.manifest)[0],'human_signoff':'pending'}
    with (root/'queue_state.json').open('x') as stream:json.dump(state,stream,indent=2)
    try:
        deadline=time.monotonic()+12*3600
        for dependency in manifest['dependencies']:
            state['waiting_on']=dependency['root'];write(root/'queue_state.json',state)
            while not dependency_ready(dependency):
                if time.monotonic()>deadline:raise TimeoutError('Dependency wait exceeded12h')
                time.sleep(15)
        state.pop('waiting_on',None)
        wait_for_gpu(root,state,job)
        disk=os.statvfs(root)
        if disk.f_bavail*disk.f_frsize<20*1024**3:raise RuntimeError('Less than20GiB free')
        verify_sources(manifest)
        with (root/'navigation.log').open('x') as log:
            process=subprocess.Popen(command,cwd=manifest['repo'],env=environment,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
            state.update(status='running',child_pid=process.pid);write(root/'queue_state.json',state)
            try:code=process.wait(timeout=1800)
            except BaseException:
                terminate_job(process);raise
        state['navigation_exit_code']=code;state.pop('child_pid',None)
        if code or not (run/'.RUN_SUCCESS').is_file():
            raise RuntimeError(f'Actual navigation exit{code}, incomplete run stays unscored')
        verify_sources(manifest)
        state.update(status='auditing');write(root/'queue_state.json',state)
        subprocess.run(['bash','-c','source "$1" && shift && export CUDA_VISIBLE_DEVICES="" && exec "$@"',
            'history-audit',manifest['isaac_setup'],manifest['python'],manifest['auditor'],
            '--manifest',str(args.manifest),'--job','diagnostic','--output',str(root/'audit.json')],
            cwd=manifest['repo'],env=environment,check=True,timeout=1800)
        result=summarize(manifest,root)
        write(root/'analysis.json',result)
        state.update(status='completed',complete_records=6,
            finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            benchmark_scores=False,navigation_acceptance=False,
            actual_rgb_review='pending',next_stage='Manual result review; no automatic expansion')
        write(root/'queue_state.json',state)
        print(json.dumps(result['by_diagnostic_type']),flush=True)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),failed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(root/'queue_state.json',state);raise


if __name__=='__main__':main()
