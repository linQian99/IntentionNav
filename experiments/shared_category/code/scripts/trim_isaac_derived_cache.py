"""Reclaim only an idle, unpinned Isaac derived database between scene jobs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess

CACHE = Path('/path/to/isaacsim/kit/cache/DerivedDataCache')
PATTERN = re.compile(r'(?:largecachedata_\d+|smallcachedata_\d+|cacheindex\.dat|cachelru\.dat)')


def trim(output: Path, protected_paths: set[str], *, threshold_gib: float = 45) -> dict:
    stat = os.statvfs(CACHE)
    free = stat.f_bavail*stat.f_frsize
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), cache=str(CACHE),
                  free_before=free, deleted_bytes=0, status='not_needed')
    if free >= threshold_gib*1024**3:
        return result
    # This database is shared across Isaac processes. Other GPU occupancy does
    # not block navigation admission, but does block shared-cache reclamation.
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                   '--format=csv,noheader,nounits'],text=True).strip()
    if apps:
        return dict(result,status='busy_preserved')
    candidates = [p for p in CACHE.iterdir() if PATTERN.fullmatch(p.name)]
    for p in candidates:
        if p.is_symlink() or not p.is_file() or str(p.resolve()) in protected_paths:
            raise ValueError(f'Unsafe derived cache candidate: {p}')
    # lsof reports all processes, including non-CUDA readers and writers.
    opened = subprocess.run(['lsof','+D',str(CACHE)],capture_output=True,text=True)
    if opened.returncode not in (0,1):
        raise RuntimeError('Cannot verify derived cache ownership')
    if opened.stdout.strip():
        return dict(result,status='open_files_preserved')
    files = [dict(path=str(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
             for p in sorted(candidates)]
    output.mkdir(parents=True,exist_ok=False)
    (output/'inventory.json').write_text(json.dumps(files,indent=2)+'\n')
    for row in files:
        p = Path(row['path'])
        st = p.stat()
        if st.st_size != row['size'] or st.st_mtime_ns != row['mtime_ns']:
            raise RuntimeError('Derived cache changed during idle check')
    for row in files:
        Path(row['path']).unlink()
        result['deleted_bytes'] += row['size']
    st = os.statvfs(CACHE)
    result.update(status='reclaimed_idle_derived_database',files=len(files),
                  free_after=st.f_bavail*st.f_frsize,protected_paths=len(protected_paths))
    (output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pins',type=Path,action='append',required=True)
    parser.add_argument('--threshold-gib',type=float,default=45)
    args = parser.parse_args()
    protected = set()
    for path in args.pins:
        doc = json.loads(path.read_text())
        protected.update(doc.get('source_hashes',doc))
    print(json.dumps(trim(args.output,protected,threshold_gib=args.threshold_gib)))
