"""Reconcile a complete navigation run with explicit, pinned goal-region data.

This writes a separate analysis and never edits episodes or historical scores.
It reports descriptive counts, not a significance or method-improvement claim.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'eval/aggregate'))
from compute_metrics import goal_region_metrics, protocol_trajectory


def rows(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def ids(path: Path) -> set[str]:
    return {s.strip() for s in path.read_text().splitlines()
            if s.strip() and not s.startswith('#')}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--goals', type=Path, required=True)
    parser.add_argument('--episodes', type=Path, required=True)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--selections', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh analysis directory')
    if not (args.run / '.RUN_SUCCESS').is_file():
        raise ValueError('The requested run has no complete-run marker')
    episodes = {e['selection_id']: e for e in rows(args.episodes)}
    goals = {(g['scene_id'], g['target_category']): g for g in rows(args.goals)}
    requested = ids(args.selections) if args.selections else set(episodes)
    primary = ids(args.primary) & requested
    source_paths = [args.goals, args.episodes, args.primary, Path(__file__),
                    REPO / 'eval/aggregate/compute_metrics.py']
    if args.selections:
        source_paths.append(args.selections)
    records = {}
    for path in sorted((args.run / 'episodes').rglob('record.json')):
        record = json.loads(path.read_text())
        sid = record['selection_id']
        if sid not in requested or sid in records:
            raise ValueError(f'Unexpected or duplicate record: {sid}')
        records[sid] = record
        source_paths.append(path)
    if set(records) != requested:
        raise ValueError(f'Record closure failed: missing {sorted(requested - set(records))}')
    scored = []
    for sid, record in sorted(records.items()):
        ep = episodes[sid]
        if record['scene_id'] != ep['scene_id']:
            raise ValueError(f'Scene mismatch: {sid}')
        trajectory, dropped = protocol_trajectory(record)
        if dropped or len(trajectory) > 31:
            raise ValueError(f'Out-of-budget events: {sid}')
        if trajectory[0]['position'] != ep['start_position']:
            # Float32 simulator pose serialization may introduce tiny noise.
            import math
            if math.dist(trajectory[0]['position'], ep['start_position']) > 1e-5:
                raise ValueError(f'Start mismatch: {sid}')
        goal = goals[(ep['scene_id'], ep['target_category'])]
        region = ep['goal_regions_v2']['any_exact_category_instance']
        metric = goal_region_metrics(record, goal['exact_category_instances'], 2.,
                                     'Any', region['geodesic_to_goal_region'])
        if not metric:
            raise ValueError(f'Missing geometry: {sid}')
        if bool(metric['Any_start_inside']) != bool(region['start_inside']):
            raise ValueError(f'Start classification mismatch: {sid}')
        if (sid in primary) == bool(region['start_inside']):
            raise ValueError(f'Primary cohort mismatch: {sid}')
        scored.append({'selection_id': sid, 'scene_id': ep['scene_id'],
                       'primary': int(sid in primary), **metric})
    primary_rows = [r for r in scored if r['primary']]
    count = len(primary_rows)
    totals = {key: sum(r[key] for r in primary_rows)
              for key in ['Any_SR_hit', 'Any_OSR_hit', 'Any_terminal_false_stop']}
    summary = {'complete_records': len(records), 'primary_count': count,
               'counts': totals, 'rates': {k: v / count if count else None
                                         for k, v in totals.items()},
               'scope': 'Descriptive rescore with supplied data; no human signoff or method-improvement claim',
               'source_hashes': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in source_paths}}
    args.output.mkdir(parents=True)
    fields = sorted({key for row in scored for key in row})
    with (args.output / 'per_item.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scored)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'source_hashes'}))


if __name__ == '__main__':
    main()
