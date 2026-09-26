"""Hash experiment inputs without retaining the entire dataset in page cache.

Output is compatible with sha256sum. Optional CUDA validation happens after
hashing and before launching Isaac; it does not alter the navigation policy.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def memory_snapshot() -> dict[str, int]:
    names = {'MemFree', 'MemAvailable', 'Cached', 'Buffers'}
    return {key.rstrip(':'): int(value) for key, value, *_ in
            (line.split() for line in Path('/proc/meminfo').read_text().splitlines())
            if key.rstrip(':') in names}


def digest_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
            total += len(block)
        # Advisory eviction of clean pages of this input only. File contents,
        # other processes and global VM settings remain untouched.
        os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return digest.hexdigest(), total


def cuda_witness() -> dict:
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('This recovery witness requires physical GPU0 only')
    used = int(subprocess.check_output(['nvidia-smi', '--id=0',
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    if used >= 500:
        raise RuntimeError(f'GPU0 became occupied while hashing ({used} MiB)')
    import torch
    value = torch.ones((1024, 1024), device='cuda:0')
    total = float(value.sum().item())
    torch.cuda.synchronize()
    if total != 1048576.0:
        raise RuntimeError('CUDA allocation/kernel witness returned wrong result')
    return {'passed': True, 'physical_gpu': 0, 'torch': torch.__version__,
            'sum': total, 'gpu_name': torch.cuda.get_device_name(0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file-list', type=Path, action='append', required=True)
    parser.add_argument('--telemetry', type=Path, required=True)
    parser.add_argument('--cuda-witness', action='store_true')
    args = parser.parse_args()
    if args.telemetry.exists():
        raise FileExistsError(args.telemetry)
    started = time.monotonic()
    record = {'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'memory_before_kib': memory_snapshot(), 'files': 0, 'bytes': 0,
              'cache_policy': 'per_input_POSIX_FADV_DONTNEED_after_hash', 'complete': False}
    try:
        for listing in args.file_list:
            for name in listing.read_text().splitlines():
                if not name:
                    continue
                digest, size = digest_file(Path(name))
                # GNU sha256sum prefixes a backslash when escaping filenames.
                escaped = name.replace('\\', '\\\\')
                prefix = '\\' if escaped != name else ''
                print(f'{prefix}{digest}  {escaped}', flush=True)
                record['files'] += 1
                record['bytes'] += size
        record['memory_after_hash_kib'] = memory_snapshot()
        if args.cuda_witness:
            record['cuda_witness'] = cuda_witness()
        record['complete'] = True
    except BaseException as exc:
        record['error'] = repr(exc)
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - started
        record['memory_after_kib'] = memory_snapshot()
        args.telemetry.write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    main()
