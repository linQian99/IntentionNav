"""Prepare shared explicit/category-prediction inputs; never launch navigation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'eval/agents'))
from navigation_query import PROTOCOL, sha256_bytes, validate_query


def build_inputs(items_path: Path, predictions: Path, output: Path, style: str) -> dict:
    if output.exists():
        raise FileExistsError('Use a fresh immutable query input directory')
    raw_items = items_path.read_bytes()
    items = [json.loads(s) for s in raw_items.decode().splitlines() if s.strip()]
    if not items or len({i['selection_id'] for i in items}) != len(items):
        raise ValueError('Empty or duplicate dataset')
    summary_path = predictions / 'summary.json'
    summary = json.loads(summary_path.read_text())
    if (summary['protocol_version'] != 'open_weight_intent_inference_v1'
            or summary['dataset_sha256'] != sha256_bytes(raw_items)
            or style not in summary['styles']):
        raise ValueError('Require the original frozen category-only cache for the exact dataset/style')
    allowed = {'protocol_version', 'selection_id', 'scene_id', 'style', 'intent_sha256',
               'model', 'prediction', 'source_record_sha256'}
    rows, pins = {}, {str(items_path.resolve()): sha256_bytes(raw_items),
                    str(summary_path.resolve()): sha256_bytes(summary_path.read_bytes())}
    for item in items:
        sid, scene = item['selection_id'], item['scene_id']
        policy = predictions / 'policy_records' / style / f'{sid}.json'
        raw = policy.read_bytes()
        cached = json.loads(raw)
        intent_hash = sha256_bytes(item[f'{style}_en'].encode())
        if (set(cached) != allowed or set(cached['model']) != {'id', 'revision'}
                or set(cached['prediction']) != {'target'}
                or cached['protocol_version'] != summary['protocol_version']
                or cached['selection_id'] != sid or cached['scene_id'] != scene
                or cached['style'] != style or cached['intent_sha256'] != intent_hash
                or cached['model'] != {'id': summary['model_id'], 'revision': summary['model_revision']}):
            raise ValueError(f'Mismatched, contextual or label-bearing policy input: {sid}')
        pins[str(policy.resolve())] = sha256_bytes(raw)
        for mode in ['explicit', 'implicit_category']:
            category = item['target_category'] if mode == 'explicit' else cached['prediction']['target']
            provenance = ({'origin': 'provided_category', 'dataset_sha256': sha256_bytes(raw_items)}
                if mode == 'explicit' else {'origin': 'frozen_intent_prediction',
                    'intent_sha256': intent_hash, 'model_id': summary['model_id'],
                    'model_revision': summary['model_revision'],
                    'policy_record_sha256': sha256_bytes(raw),
                    'source_record_sha256': cached['source_record_sha256']})
            row = {'protocol_version': PROTOCOL, 'selection_id': sid, 'scene_id': scene,
                   'style': style, 'input_mode': mode, 'query_category': category,
                   'provenance': provenance}
            validate_query(row, selection_id=sid, scene_id=scene, style=style, input_mode=mode)
            rows[f'{mode}/{style}/{sid}.json'] = json.dumps(row, indent=2) + '\n'
    # All records are checked before writing any query output.
    output.mkdir(parents=True)
    index = {'protocol_version': PROTOCOL, 'records': {}}
    for relative, text in sorted(rows.items()):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        index['records'][relative] = sha256_bytes(text.encode())
    (output / 'policy_manifest.json').write_text(json.dumps(index, indent=2) + '\n')
    for path in [Path(__file__), REPO / 'eval/agents/navigation_query.py']:
        pins[str(path.resolve())] = sha256_bytes(path.read_bytes())
    report = {'items': len(items), 'policy_records': len(rows), 'style': style,
        'intent_model': {'id': summary['model_id'], 'revision': summary['model_revision']},
        'source_hashes': pins, 'policy_index_sha256': sha256_bytes((output / 'policy_manifest.json').read_bytes()),
        'source_selection': 'Original v1 category-only cache; no selection based on navigation outcomes',
        'scope': 'Provisional inputs only, no navigation scores or human/data/system acceptance. '
                 'One predicted category only; no predicted room/support, true goal instance/position or intent text sent to navigation.'}
    (output / 'build_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--items', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--style', choices=['formal', 'natural', 'casual', 'emotional'], default='formal')
    args = parser.parse_args()
    result = build_inputs(args.items, args.predictions, args.output, args.style)
    print(json.dumps({k: v for k, v in result.items() if k != 'source_hashes'}))


if __name__ == '__main__':
    main()
