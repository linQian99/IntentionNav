"""Replay saved histories through two independent production MTU3D workers."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path: Path, data: dict) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    pins = manifest['source_hashes']
    for name, expected in pins.items():
        if digest(Path(name)) != expected:
            raise ValueError(f'Pinned source/input changed: {name}')
    if str(Path(__file__).resolve()) not in pins:
        raise ValueError('Diagnostic script must be pinned')
    output = Path(manifest['output'])
    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    cases = {}
    for case in manifest['cases']:
        sequence = []
        for name in case['requests']:
            if name not in pins:
                raise ValueError('Unpinned request')
            request = json.loads(Path(name).read_text())['request']
            if set(request) - {'observation', 'position', 'look_dir', 'intrinsics',
                               'frontiers', 'query', 'decision_index', 'context_observations'}:
                raise ValueError('Unexpected policy input')
            observations = [request['observation']] + [
                view['observation'] for view in request.get('context_observations', [])]
            if any(name not in pins for name in observations):
                raise ValueError('Unpinned observation')
            sequence.append(request)
        if [r['decision_index'] for r in sequence] != list(range(len(sequence))):
            raise ValueError('A complete consecutive history prefix is required')
        cases[case['episode_key']] = sequence
    if len(cases) != len(manifest['cases']) or len(manifest['orders']) != 2:
        raise ValueError('Expected unique episode identities and two workers')
    if any(sorted(order) != sorted(cases) for order in manifest['orders']):
        raise ValueError('Each worker must replay each case exactly once')
    if args.check:
        print(json.dumps({'passed': True, 'cases': len(cases), 'workers': 2,
                          'decisions': 2 * sum(map(len, cases.values())),
                          'source_pins': len(pins), 'gpu_initialized': False,
                          'output_created': False, 'manifest_sha256': digest(args.manifest)}))
        return
    memory = [int(s) for s in subprocess.check_output([
        'nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        text=True).splitlines()]
    if any(value >= 500 for value in memory):
        raise RuntimeError('Both GPUs must be idle before GPU0 serial replay')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Bind physical GPU0 explicitly')
    sys.path.insert(0, manifest['agents_source'])
    from agent_mtu3d import Worker
    from mtu3d_randomness import decision_seed

    output.mkdir()
    started = time.monotonic()
    results = {}
    process_ids = []
    write(output / 'state.json', {'status': 'starting', 'pid': os.getpid()})
    try:
        for repeat, order in enumerate(manifest['orders']):
            worker_output = output / f'worker_{repeat}'
            worker_output.mkdir()
            worker = Worker(worker_output, manifest['base_seed'])
            process_ids.append(worker.process.pid)
            try:
                if worker.ready.get('randomness_mode') != 'per_decision_v1':
                    raise ValueError('Worker did not advertise the integrated RNG fix')
                write(worker_output / 'ready.json', worker.ready)
                for key in order:
                    ack = worker.request({'command': 'reset', 'episode_key': key})
                    if ack.get('episode_key') != key or ack.get('randomness_mode') != 'per_decision_v1':
                        raise ValueError('Wrong episode reset acknowledgement')
                    answers = []
                    for request in cases[key]:
                        answer = worker.request(request)
                        rng = answer['randomness']
                        if (rng['mode'] != 'per_decision_v1' or rng['episode_key'] != key or
                                rng['seed'] != decision_seed(manifest['base_seed'], key, request['decision_index'])):
                            raise ValueError('Wrong decision RNG identity')
                        answers.append(answer)
                    results[f'{repeat}/{key}'] = answers
                    write(output / 'decisions.json', results)
                    write(output / 'state.json', {'status': 'running', 'repeat': repeat,
                          'last_completed': key, 'worker_pid': worker.process.pid})
                    print(json.dumps({'repeat': repeat, 'episode': key, 'decisions': len(answers)}), flush=True)
            finally:
                worker.close()
            if worker.process.returncode != 0:
                raise RuntimeError(f'Worker failed during close: {worker.process.returncode}')
        comparisons = []
        for key in cases:
            a, b = results[f'0/{key}'], results[f'1/{key}']
            deltas = [math.dist(x['target_position'], y['target_position']) for x, y in zip(a, b)]
            if not all(math.isfinite(value) for value in deltas):
                raise ValueError('Nonfinite prediction')
            comparisons.append({'episode_key': key, 'n': len(a), 'max_target_delta_m': max(deltas),
                'decisions_over_1cm': sum(value > .01 for value in deltas),
                'object_choice_flips': sum(x['is_object_decision'] != y['is_object_decision'] for x, y in zip(a, b)),
                'source_flips': sum(x['decision_source'] != y['decision_source'] for x, y in zip(a, b))})
        passed = all(row['decisions_over_1cm'] == row['object_choice_flips'] == row['source_flips'] == 0
                     for row in comparisons)
        report = {'status': 'completed', 'passed': passed, 'comparisons': comparisons,
                  'worker_pids': process_ids, 'elapsed_seconds': time.monotonic() - started,
                  'manifest_sha256': digest(args.manifest),
                  'scope': 'Production worker IPC and reversed episode order, fixed observations; not navigation SR'}
        write(output / 'summary.json', report)
        write(output / 'state.json', {'status': 'completed', 'passed': passed})
        print(json.dumps(report), flush=True)
        if not passed:
            raise RuntimeError('Worker reproducibility regression failed')
    except BaseException as exc:
        write(output / 'state.json', {'status': 'failed', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
