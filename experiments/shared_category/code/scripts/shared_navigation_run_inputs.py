"""Archive shared policy inputs and reconcile their use in completed episodes.

This is an input/record gate, not navigation scoring or proof of semantic
correctness. MTU requests are checked individually; R055's saved plan and
category-derived detector queries are reconciled with its frozen vocabulary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'eval/agents'))
from navigation_query import resolve_category


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_inputs(inputs: Path, items_file: Path, selections: Path, mode: str) -> tuple[dict, dict]:
    selected = [s.strip() for s in selections.read_text().splitlines()
                if s.strip() and not s.startswith('#')]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError('Require nonempty unique selection IDs')
    items = {r['selection_id']: r for r in
             (json.loads(s) for s in items_file.read_text().splitlines() if s.strip())}
    records = {}
    paths = [inputs / 'policy_manifest.json', items_file, selections,
             Path(__file__), REPO / 'eval/agents/navigation_query.py']
    for sid in selected:
        item = items[sid]
        category, provenance = resolve_category(inputs, item=item, style='formal', input_mode=mode)
        packet = Path(provenance['policy_record'])
        paths.append(packet)
        records[sid] = {'scene_id': item['scene_id'], 'query_category': category,
                        'target_category': item['target_category'],
                        'navigation_input': provenance}
    return records, {str(p.resolve()): digest(p) for p in paths}


def snapshot(inputs: Path, items: Path, selections: Path, mode: str, output: Path) -> dict:
    records, hashes = read_inputs(inputs, items, selections, mode)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(inputs / 'policy_manifest.json', output / 'policy_manifest.json')
    for sid, row in records.items():
        relative = f'{mode}/formal/{sid}.json'
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inputs / relative, target)
    result = {'input_mode': mode, 'records': records, 'source_hashes': hashes,
              'archive_hashes': {str(p.relative_to(output)): digest(p)
                                for p in sorted(output.rglob('*.json'))},
              'scope': 'Evaluation labels in this receipt are offline only; agents read policy packets.'}
    (output / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def audit(run: Path, archive: Path, system: str, source: Path) -> dict:
    receipt_path = archive / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    hashes = dict(receipt['source_hashes'])
    for name, expected in hashes.items():
        if digest(Path(name)) != expected:
            raise ValueError(f'Input changed during execution: {name}')
    for relative, expected in receipt['archive_hashes'].items():
        path = archive / relative
        if digest(path) != expected:
            raise ValueError(f'Archived input changed: {relative}')
        hashes[str(path.resolve())] = expected
    hashes[str(receipt_path.resolve())] = digest(receipt_path)
    expected_rows = receipt['records']
    paths = list((run / 'episodes').rglob('record.json'))
    found = {}
    for path in paths:
        record = json.loads(path.read_text())
        sid = record['selection_id']
        if sid not in expected_rows or sid in found:
            raise ValueError(f'Unexpected or duplicate completed record: {sid}')
        found[sid] = (path, record)
    if set(found) != set(expected_rows):
        raise ValueError('Shared-input record closure is incomplete')
    query_function = None
    if system == 'r055':
        # Import only CPU evidence helpers, no simulator/model initialization.
        import importlib.util
        path = source / 'agents/evidence_nav.py'
        spec = importlib.util.spec_from_file_location('shared_audit_vocabulary', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        query_function = module.target_detector_queries
        hashes[str(path.resolve())] = digest(path)
        vocab = source / 'vocab/category_synonyms.yaml'
        hashes[str(vocab.resolve())] = digest(vocab)
    request_count = 0
    for sid, (path, record) in found.items():
        expected = expected_rows[sid]
        category = expected['query_category']
        if (record['scene_id'] != expected['scene_id'] or record['style'] != 'formal'
                or record['target_category'] != expected['target_category']
                or record.get('navigation_input') != expected['navigation_input']
                or record['evaluation_protocol']['goal_input'] != 'shared_' + receipt['input_mode']
                or record['step_cap'] != 30):
            raise ValueError(f'Record input/label/protocol mismatch: {sid}')
        query = category if system == 'r055' else category.replace('_', ' ')
        if record['prediction']['target'] != query:
            raise ValueError(f'Realized prediction differs from shared packet: {sid}')
        if system == 'r055':
            plan = record['plan']
            flags = record['ablation_flags']
            if (plan['target_guess'] != category or plan['candidate_objects'] != [category]
                    or plan['plan_model'] != 'provided_shared_category'
                    or flags['inav_target_detector_queries'] != query_function(category)
                    or flags['inav_visual_target_descriptors']):
                raise ValueError(f'R055 plan/detector input differs from contract: {sid}')
        else:
            calls = 0
            for decision_path in sorted(path.parent.glob('decision_*.json')):
                decision = json.loads(decision_path.read_text())
                request = decision.get('request')
                if request is not None:
                    if request['query'] != query:
                        raise ValueError(f'MTU worker query differs from contract: {decision_path}')
                    request_path = decision_path.with_name(decision_path.name.replace('decision_', 'request_', 1))
                    submitted = json.loads(request_path.read_text())['request']
                    if submitted != request:
                        raise ValueError(f'Archived worker request differs from decision evidence: {request_path}')
                    hashes[str(request_path.resolve())] = digest(request_path)
                    calls += 1
                hashes[str(decision_path.resolve())] = digest(decision_path)
            if calls != record['model_meta']['decision_calls'] or calls == 0:
                raise ValueError(f'Missing model-request evidence: {sid}')
            if len(list(path.parent.glob('request_*.json'))) != calls:
                raise ValueError(f'Unexpected archived model-request evidence: {sid}')
            request_count += calls
        hashes[str(path.resolve())] = digest(path)
    if system == 'mtu3d':
        path = run / 'run_manifest.json'
        actual = json.loads(path.read_text())
        if (actual['input_mode'] != 'shared_' + receipt['input_mode']
                or actual['queries'] != {s: r['query_category'].replace('_', ' ')
                                        for s, r in expected_rows.items()}
                or actual['shared_query_provenance'] != {s: r['navigation_input']
                                                       for s, r in expected_rows.items()}):
            raise ValueError('MTU run manifest differs from shared packets')
        hashes[str(path.resolve())] = digest(path)
    return {'passed': True, 'system': system, 'input_mode': receipt['input_mode'],
            'records': len(found), 'checked_mtu_requests': request_count,
            'source_hashes': hashes, 'navigation_performance_claim': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['snapshot', 'audit'])
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--items', type=Path)
    parser.add_argument('--selections', type=Path)
    parser.add_argument('--mode', choices=['explicit', 'implicit_category'])
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--system', choices=['r055', 'mtu3d'])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'snapshot':
        if not all([args.inputs, args.items, args.selections, args.mode]):
            parser.error('Snapshot requires inputs, items, selections and mode')
        result = snapshot(args.inputs, args.items, args.selections, args.mode, args.archive)
        print(json.dumps({'passed': True, 'archived_records': len(result['records'])}))
    else:
        if not all([args.run, args.system, args.source, args.output]):
            parser.error('Audit requires run, system, source and output')
        result = audit(args.run, args.archive, args.system, args.source)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        print(json.dumps({k: v for k, v in result.items() if k != 'source_hashes'}))


if __name__ == '__main__':
    main()
