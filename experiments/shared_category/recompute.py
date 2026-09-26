#!/usr/bin/env python3
"""Recompute the released center, STOP/surface, GSR and input-accuracy labels.

Usage: python recompute.py /path/to/extracted/shared_category_20260924
Dependencies: numpy, PyYAML. No simulator, model weights or API calls.
"""
from pathlib import Path
import argparse, ast, collections, csv, hashlib, importlib.util, json, math, os, sys
sys.dont_write_bytecode = True


def main():
    ap=argparse.ArgumentParser();ap.add_argument('release',type=Path);args=ap.parse_args();root=args.release.resolve()
    for line in (root/'SHA256SUMS').read_text().splitlines():
        sha,name=line.split('  ',1)
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha:raise ValueError('Checksum mismatch: '+name)
    sys.path[:0]=[str(root/'code/eval/aggregate'),str(root/'code/eval/agents')]
    from compute_metrics import protocol_trajectory,goal_region_metrics
    # Execute the exact archived score_record function, without its machine-specific CLI imports.
    source=(root/'code/scripts/score_iclr_core_comparison.py').read_text()
    fn=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='score_record')
    scope=dict(protocol_trajectory=protocol_trajectory,goal_region_metrics=goal_region_metrics,math=math,AMBIGUOUS={'SEL_077','SEL_314'})
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'archived-score-record','exec'),scope)
    score_record=scope['score_record']
    data=root/'data/reviewed-v3'
    episodes={x['selection_id']:x for x in map(json.loads,(data/'episodes_explicit_category.jsonl').read_text().splitlines())}
    goals={(x['scene_id'],x['target_category']):x for x in map(json.loads,(data/'category_goal_sets.jsonl').read_text().splitlines())}
    spec=importlib.util.spec_from_file_location('legacy_visibility',root/'analysis/legacy_gsr/legacy_visibility.py');vis=importlib.util.module_from_spec(spec);spec.loader.exec_module(vis)
    vis.DATASET_ROOT=data;vis.DATASET_JSONL=data/'selected_500_intents.jsonl';os.environ['INTENTEQA_SCENE_SUMMARY']=str(root/'geometry')
    gsr={(r['arm_repeat'],r['selection_id']):r for r in csv.DictReader((root/'analysis/legacy_gsr/episodes.csv').open())}
    stats=collections.defaultdict(lambda:collections.defaultdict(float));n=0
    for cohort in ['full500','dev40']:
        rows=list(csv.DictReader((root/f'analysis/{cohort}/episodes.csv').open()))
        assert len(rows)==(2000 if cohort=='full500' else 320)
        for row in rows:
            record=json.loads((root/row['record_path']).read_text());sid=row['selection_id'];ep=episodes[sid]
            scored=score_record(record,ep,goals[(ep['scene_id'],ep['target_category'])],row['arm_repeat'].split('_')[0])
            for key,val in scored.items():
                if key=='record_path':continue
                expected=row[key]
                if isinstance(val,(int,float,bool)):
                    assert math.isclose(float(val),float(expected),abs_tol=1e-7), (cohort,sid,key,val,expected)
                elif val is None:assert expected==''
                else:assert str(val)==expected,(key,val,expected)
            traj,invalid=protocol_trajectory(record);assert invalid==0
            dist=[math.dist(t['position'][:2],ep['target_position'][:2]) for t in traj]
            for radius in [1,2,3]:
                assert int(dist[-1]<=radius)==int(row[f'paper_SR_{radius}m'])
                assert int(min(dist)<=radius)==int(row[f'paper_OSR_{radius}m'])
            key=cohort+'/'+row['arm_repeat'];s=stats[key];s['n']+=1;s['SR']+=dist[-1]<=2;s['OSR']+=min(dist)<=2
            frontend=json.loads((root/f'frontend/records/formal/{sid}.json').read_text())
            # Cached category labels were normalized by the original frontend scorer.
            assert int(frontend['IM_hit'])==int(row['frontend_IM_hit'])
            s['cached_IM']+=int(row['frontend_IM_hit'])
            if cohort=='full500':
                v=vis.trajectory_visibility(record);assert v
                hit=int(dist[-1]<=1 and v['G_seen']);assert hit==int(gsr[(row['arm_repeat'],sid)]['paper_GSR_1m'])
                s['GSR']+=hit
            n+=1
    result={'verified_records':n,'all_expected_labels_match':True,'summary':{k:{metric:round(value/v['n']*100,5) if metric!='n' else int(value) for metric,value in v.items()} for k,v in sorted(stats.items())}}
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
