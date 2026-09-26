"""Shared, frozen category-only input contract for complete navigation systems.

Policy input and evaluation labels are distinct. Both systems consume the
same category record; each retains its existing detector/query normalization.
This module performs no inference and imports no simulator or ML runtime.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

PROTOCOL = 'shared_navigation_category_v1'
MODES = {'explicit', 'implicit_category'}
STYLES = {'formal', 'natural', 'casual', 'emotional'}
ROW_KEYS = {'protocol_version', 'selection_id', 'scene_id', 'style', 'input_mode',
            'query_category', 'provenance'}
PROVENANCE_KEYS = {
    'explicit': {'origin', 'dataset_sha256'},
    'implicit_category': {'origin', 'intent_sha256', 'model_id', 'model_revision',
                          'policy_record_sha256', 'source_record_sha256'},
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def validate_query(row: dict, *, selection_id: str, scene_id: str,
                   style: str, input_mode: str) -> None:
    if not re.fullmatch(r'SEL_\d+', selection_id) or not re.fullmatch(r'kujiale_\d+(?:_fix)?', scene_id):
        raise ValueError('Invalid query identity')
    if style not in STYLES or input_mode not in MODES:
        raise ValueError('Unsupported input mode or style')
    if not isinstance(row, dict) or set(row) != ROW_KEYS:
        raise ValueError('Unexpected or missing query fields')
    expected = {'protocol_version': PROTOCOL, 'selection_id': selection_id,
                'scene_id': scene_id, 'style': style, 'input_mode': input_mode}
    if any(row[k] != v for k, v in expected.items()):
        raise ValueError('Query identity/protocol mismatch')
    category = row['query_category']
    if not isinstance(category, str) or not category.strip() or category != category.strip().lower():
        raise ValueError('Missing or nonnormalized category query')
    provenance = row['provenance']
    if not isinstance(provenance, dict) or set(provenance) != PROVENANCE_KEYS[input_mode]:
        raise ValueError('Unexpected or incomplete query provenance')
    if provenance['origin'] != ('provided_category' if input_mode == 'explicit' else 'frozen_intent_prediction'):
        raise ValueError('Wrong query origin')
    for key, value in provenance.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Empty query provenance')
        if key.endswith('_sha256') and not re.fullmatch(r'[0-9a-f]{64}', value):
            raise ValueError('Malformed query provenance hash')


def load_navigation_query(directory: Path, *, selection_id: str, scene_id: str,
                          style: str, input_mode: str) -> tuple[str, dict]:
    """Read only the hash-pinned policy packet, never ground-truth fallback."""
    if input_mode not in MODES or style not in STYLES or not re.fullmatch(r'SEL_\d+', selection_id):
        raise ValueError('Invalid query path or input mode')
    index_path = directory / 'policy_manifest.json'
    index = json.loads(index_path.read_text())
    if set(index) != {'protocol_version', 'records'} or index['protocol_version'] != PROTOCOL:
        raise ValueError('Unexpected policy index fields or protocol')
    relative = f'{input_mode}/{style}/{selection_id}.json'
    raw = (directory / relative).read_bytes()
    digest = sha256_bytes(raw)
    if index['records'].get(relative) != digest:
        raise ValueError('Unpinned or changed category input')
    row = json.loads(raw)
    validate_query(row, selection_id=selection_id, scene_id=scene_id,
                   style=style, input_mode=input_mode)
    return row['query_category'], {'protocol_version': PROTOCOL, 'input_mode': input_mode,
        'style': style, 'policy_record': str((directory / relative).resolve()),
        'policy_record_sha256': digest, 'policy_index_sha256': sha256_bytes(index_path.read_bytes()),
        'provenance': row['provenance'], 'scope': 'category_only_bottleneck'}


def resolve_category(directory: Path | None, *, item: dict, style: str,
                     input_mode: str) -> tuple[str, dict | None]:
    """Preserve explicit baseline input; shared mode never falls back to labels."""
    if directory is None:
        if input_mode != 'explicit':
            raise ValueError('Implicit category mode requires frozen shared query inputs')
        category = item.get('target_category')
        if not isinstance(category, str) or not category.strip():
            raise ValueError('Explicit category is required')
        return category, None
    category, provenance = load_navigation_query(directory, selection_id=item['selection_id'],
        scene_id=item['scene_id'], style=style, input_mode=input_mode)
    if input_mode == 'explicit' and category != item['target_category']:
        raise ValueError('Shared explicit query differs from the provided dataset category')
    return category, provenance
