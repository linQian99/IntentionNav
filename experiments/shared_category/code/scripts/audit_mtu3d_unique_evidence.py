"""Check online integration accounting against actual acquired RGB-D identities."""
from pathlib import Path
import json
import sys

import numpy as np


def verify_integration(audit, identities, seen):
    if (not identities or not audit or audit.get('mode') != 'unique_rgbd_evidence_v1'
            or audit.get('ground_truth_used') is not False
            or audit.get('perception_context_changed') is not False):
        raise ValueError('Missing unique-evidence integration contract')
    new = list(dict.fromkeys(v for v in identities if v not in seen))
    integrated = audit['integrated_prediction_view_ids']
    skipped = audit['skipped_prediction_view_ids']
    if (audit['consumed_view_ids'] != identities
            or audit['newly_consumed_view_ids'] != new
            or audit['unique_views_before'] != len(seen)
            or audit['unique_views_after'] != len(seen | set(identities))
            or len(integrated) != len(set(integrated))
            or not set(integrated) <= set(new)
            or any(v not in identities or (v not in seen and v not in integrated) for v in skipped)
            or audit['new_views_without_predictions'] != [v for v in new if v not in integrated]
            or audit['predictions_integrated'] != len(integrated)
            or audit['predictions_supplied'] != len(integrated) + len(skipped)
            or audit['predictions_supplied'] > len(identities)):
        raise ValueError('Memory integration differs from actual consumed evidence')
    return seen | set(identities)


def audit_record(path, source):
    sys.path.insert(0, str(source / 'agents'))
    from mtu3d_observation_support import describe_view
    record = json.loads(path.read_text())
    seen, cache = set(), {}
    counts = dict(selection_id=record['selection_id'], model_calls=0,
                  unique_views=0, integrated_prediction_blocks=0, reused_blocks_skipped=0)
    for step in record['trajectory']:
        answer = step.get('policy_decision')
        if not answer:
            continue
        request = json.loads((path.parent / f"request_{step['step']:02d}.json").read_text())['request']
        identities = []
        for vi, view in enumerate(request['context_observations'] + [request]):
            if vi not in answer['valid_view_indices']:
                continue
            key = json.dumps({k: view[k] for k in ['observation', 'position', 'look_dir', 'intrinsics']}, sort_keys=True)
            if key not in cache:
                with np.load(view['observation'], allow_pickle=False) as arrays:
                    cache[key] = describe_view({**view, 'rgb': arrays['rgb'], 'depth': arrays['depth']},
                                              answer['decision_index'], vi)['view_id']
            identities.append(cache[key])
        trace = answer.get('stage2_trace') or {}
        audit = trace.get('memory_integration_audit') or answer.get('memory_integration_audit')
        if not identities:
            if answer.get('fallback_reason') != 'insufficient_depth' or audit:
                raise ValueError('No valid views must not update persistent evidence')
        else:
            seen = verify_integration(audit, identities, seen)
            if trace and trace['observation_support_audit']['predicted_view_blocks'] != audit['predictions_integrated']:
                raise ValueError('Actual observer did not consume the filtered blocks')
            counts['integrated_prediction_blocks'] += audit['predictions_integrated']
            counts['reused_blocks_skipped'] += len(audit['skipped_prediction_view_ids'])
        counts['model_calls'] += 1
    counts['unique_views'] = len(seen)
    return counts
