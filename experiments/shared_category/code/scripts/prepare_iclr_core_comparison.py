"""Freeze the six-day ICLR comparison without changing either navigation policy."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil

from build_navigation_query_inputs import build_inputs
from hash_navigation_inputs import digest_file
from probe_mtu3d_worker_reset import write

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/dataset_ai_reviewed_20260918_v3'
PARENT = ROOT / 'results/navigation_motion_repair_recovery_20260914/manifest.json'
REFERENCE = ROOT / 'results/r055_reviewed_full500_20260916'


def prepare(root: Path, doc: Path):
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    parent = json.loads(PARENT.read_text())
    ref = json.loads((REFERENCE / 'manifest.json').read_text())
    items = [json.loads(s) for s in (DATA / 'selected_500_intents.jsonl').read_text().splitlines()]
    episodes = [json.loads(s) for s in (DATA / 'episodes_explicit_category.jsonl').read_text().splitlines()]
    assert len(items) == len(episodes) == 500
    source_pins, heavy = {}, {}

    def pin(path, expected=None, target=None):
        path = Path(path)
        value = digest_file(path)[0]
        if expected is not None and value != expected:
            raise ValueError(f'Changed frozen input: {path}')
        (source_pins if target is None else target)[str(path)] = value

    # Models and scene assets have their own immutable receipt, verified once per
    # process; the per-job auditor verifies only its code/data/input contract.
    old_assets = json.loads(Path(ref['runtime_input_hashes']).read_text())
    for n, (path, expected) in enumerate(old_assets.items()):
        if '/datasets/vlntube/' in path:
            pin(path, expected, heavy)
        if (n + 1) % 20000 == 0:
            print(f'Verified scene inputs {n + 1}/{len(old_assets)}', flush=True)
    for m in [parent, ref]:
        for path, expected in m['source_hashes'].items():
            if '/.cache/huggingface/' in path or '/work_dirs/external/MTU3D/' in path or path.endswith(('.pth', '.pt')):
                pin(path, expected, heavy)
    for path, expected in json.loads((DATA / 'output_sha256.json').read_text()).items():
        pin(path, expected)
    pin(DATA / 'output_sha256.json')
    shared = root / 'shared_inputs'
    build_inputs(DATA / 'selected_500_intents.jsonl', root.parent / 'intent_frontend', shared, 'formal')
    for path in shared.rglob('*.json'):
        pin(path)
    bundle = root / 'source_snapshot'
    bundle.mkdir()
    for name in ['launch_r055_reviewed_data.sh', 'hash_navigation_inputs.py', 'launch_mtu3d_navigation.sh']:
        shutil.copyfile(ROOT / 'scripts' / name, bundle / name)
        pin(bundle / name)
    for system, source in parent['sources'].items():
        for path in Path(source).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                expected = parent['source_hashes'].get(str(path))
                if expected is None:
                    raise ValueError(f'Unattested policy file: {path}')
                pin(path, expected)
    manifest = {k: parent[k] for k in ['repo', 'python', 'isaac_setup', 'audit_runtime', 'metaroot', 'gpu', 'gpu_uuid', 'sources', 'source', 'auditors', 'motion_auditor']}
    manifest.update(output=str(root), data_root=str(DATA), items=str(DATA / 'selected_500_intents.jsonl'),
        episodes=str(DATA / 'episodes_explicit_category.jsonl'), goals=str(DATA / 'category_goal_sets.jsonl'),
        primary=str(DATA / 'primary_selection_ids.txt'), shared_inputs=str(shared),
        max_parallel=1, full_records=2000, repeat_records=320, smoke_records=8,
        new_navigation_records=1828, reused_navigation_records=500,
        publication_ready=False, full_B0_certified=False, reviewer_type='ai_assistant',
        scope='Frozen two-stage category bottleneck systems; designated-instance and category surface scores separate. '
              'One full repeat plus two fresh dev40 repeats. Data ambiguity remains disclosed. Not a policy improvement.',
        invocation_doc=str(doc), max_wall_hours=96, maximum_attempts=2, jobs=[])
    prototype = next(j for j in ref['jobs'] if j['id'] == 'smoke')['environment']
    resolver = prototype['INAV_INPUT_RESOLVER']

    def job(stage, system, mode, repeat, sids, suffix):
        key = f'{stage}_{system}_{mode}_{repeat}_{suffix}'
        selection = root / 'selections' / f'{key}.txt'
        selection.parent.mkdir(exist_ok=True)
        selection.write_text('\n'.join(sids) + '\n')
        pin(selection)
        seed = {'a': 20260909, 'b': 20260910}[repeat]
        manifest['jobs'].append(dict(id=key, stage=stage, system=system, mode=mode, repeat=repeat,
            seed=seed, count=len(sids), selection_ids=sids, selections=str(selection),
            source=parent['sources'][system], exploration_mode='boundary_commit',
            launcher=str(bundle / ('launch_r055_reviewed_data.sh' if system == 'r055' else 'launch_mtu3d_navigation.sh')),
            input_resolver=resolver))

    arms = [('r055', 'explicit'), ('mtu3d', 'explicit'), ('r055', 'implicit_category'), ('mtu3d', 'implicit_category')]
    for system, mode in arms:
        job('smoke', system, mode, 'a', ['SEL_078', 'SEL_142'], 'integration')
    dev = [s.strip() for s in (ROOT / 'eval/splits/navigation_dev_40.txt').read_text().splitlines() if s.startswith('SEL_')]
    assert len(dev) == 40
    for repeat in ['a', 'b']:
        for system, mode in (arms if repeat == 'a' else list(reversed(arms))):
            job('repeat40', system, mode, repeat, dev, 'dev40')
    grouped = defaultdict(list)
    for e in episodes:
        grouped[e['scene_id']].append(e['selection_id'])
    new_arms = [('mtu3d', 'explicit'), ('r055', 'implicit_category'), ('mtu3d', 'implicit_category')]
    for index, (scene, sids) in enumerate(sorted(grouped.items())):
        order = new_arms[index % 3:] + new_arms[:index % 3]
        for system, mode in order:
            job('full500', system, mode, 'a', sorted(sids), scene)
    assert sum(j['count'] for j in manifest['jobs']) == 1828
    # Reuse requires actual closed episodes, unchanged policy/query/start inputs,
    # and verified original evidence, never simply the presence of old scores.
    state = json.loads((REFERENCE / 'queue_state.json').read_text())
    if state['status'] != 'completed_full500' or state['complete_records'] != 500:
        raise ValueError('Reference full500 is incomplete')
    rows = {}
    reference_pins = {}
    item_by_id = {i['selection_id']: i for i in items}
    ep_by_id = {e['selection_id']: e for e in episodes}
    old_episodes = {e['selection_id']: e for e in map(json.loads, Path(ref['episodes']).read_text().splitlines())}
    for sid, ep in ep_by_id.items():
        for field in ['scene_id', 'target_category', 'target_object_id', 'start_position', 'start_rotation_quat_wxyz']:
            if ep[field] != old_episodes[sid][field]:
                raise ValueError(f'Reference episode input changed: {sid}/{field}')
    for old_job in ref['jobs']:
        for field, expected in [('INAV_AGENT_SOURCE_DIR', parent['sources']['r055'] + '/agents'),
                ('INAV_EXECUTION_SEED', '20260909'), ('INAV_RENDER_ANTIALIASING_MODE', '2'),
                ('INAV_RENDER_TICKS', '8'), ('INAV_GEOMETRY_SAFE_FRONTIER', '1')]:
            if old_job['environment'][field] != expected:
                raise ValueError('Reference policy settings differ')
    for key, pointer in state['checkpoints'].items():
        if key == 'smoke':
            continue
        for name, digest in pointer['source_hashes'].items():
            pin(name, digest, reference_pins)
        cp = json.loads(Path(pointer['checkpoint']).read_text())
        for name, expected in cp['source_hashes'].items():
            pin(name, expected, reference_pins)
        for path in (Path(cp['output']) / 'episodes').rglob('record.json'):
            r = json.loads(path.read_text()); sid = r['selection_id']
            if sid in rows or r['scene_id'] != ep_by_id[sid]['scene_id']:
                raise ValueError('Duplicate or mismatched reference episode')
            if (r['prediction']['target'] != item_by_id[sid]['target_category'] or r['step_cap'] != 30
                    or r['navigation_input']['input_mode'] != 'explicit'):
                raise ValueError(f'Reference policy input differs: {sid}')
            rows[sid] = str(path)
    if set(rows) != set(item_by_id):
        raise ValueError('Incomplete reference episode identity closure')
    for path in [REFERENCE / 'manifest.json', REFERENCE / 'queue_state.json', REFERENCE / 'full500_summary.json']:
        pin(path, target=reference_pins)
    reference_receipt = root / 'reference_reuse.json'
    write(reference_receipt, dict(records=rows, source_hashes=reference_pins,
        policy_changed=False, actual_queries_match=True, old_data_root=ref['data_root'],
        new_data_root=str(DATA), data_revision_is_evaluation_only_for_these_500_queries=True,
        scope='Historical explicit R055 trajectories; recompute both fixed/any strict metrics. '
              'Do not claim contemporaneous full-repeat stability; fresh dev40 repeats are separate.'))
    manifest['reference_reuse'] = str(reference_receipt)
    heavy_path = root / 'heavy_input_hashes.json'; write(heavy_path, heavy)
    manifest['heavy_input_hashes'] = str(heavy_path)
    for path in [reference_receipt, heavy_path, doc, PARENT, Path(__file__),
                 ROOT / 'scripts/run_iclr_core_comparison.py', ROOT / 'scripts/score_iclr_core_comparison.py',
                 ROOT / 'scripts/build_navigation_query_inputs.py', ROOT / 'scripts/trim_isaac_derived_cache.py',
                 ROOT / 'scripts/shared_navigation_run_inputs.py', ROOT / 'scripts/run_r055_scene_validation.py',
                 ROOT / 'scripts/run_mtu3d_final_controls.py', ROOT / 'scripts/run_baseline_acceptance.py',
                 ROOT / 'scripts/run_visible_navigation_diagnostics.py', resolver,
                 ROOT / 'scripts/probe_mtu3d_worker_reset.py', ROOT / 'scripts/analyze_navigation_system.py',
                 ROOT / 'eval/agents/navigation_query.py', ROOT / 'eval/aggregate/compute_metrics.py',
                 ROOT / 'eval/aggregate/strict_goal_regions.py', ROOT / 'eval/aggregate/mtu3d_support_precision.py',
                 ROOT / 'experiments/run', ROOT / 'scripts/with_eval_sources.py', ROOT / 'experiments/sources.json',
                 ROOT / 'experiments/records.json', ROOT / 'scripts/hash_navigation_inputs.py',
                 *[ROOT / 'scripts' / name for name in ['audit_navigation_motion.py', 'audit_r055_observation_archive.py',
                    'audit_mtu3d_unique_navigation_v2.py', 'audit_mtu3d_acquired_history.py', 'audit_mtu3d_unique_evidence.py',
                    'audit_mtu3d_supported_view_v2.py', 'audit_mtu3d_view_consumption.py']]]:
        pin(path)
    # Pin the transitive project-local Python import closure used by audits.
    import ast
    pending = [Path(p) for p in source_pins if p.endswith('.py')]
    visited = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module
                       else [n.name for n in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                for directory in [ROOT / 'scripts', ROOT / 'eval/aggregate']:
                    dependency = directory / (module.replace('.', '/') + '.py')
                    if dependency.is_file() and str(dependency) not in source_pins:
                        pin(dependency); pending.append(dependency)
    manifest['source_hashes'] = source_pins
    write(root / 'manifest.json', manifest)
    print(json.dumps(dict(prepared=True, jobs=len(manifest['jobs']), new_navigation=1828,
        reused_navigation=500, source_pins=len(source_pins), heavy_pins=len(heavy))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--doc', type=Path, required=True)
    a = p.parse_args(); prepare(a.output.resolve(), a.doc.resolve())
