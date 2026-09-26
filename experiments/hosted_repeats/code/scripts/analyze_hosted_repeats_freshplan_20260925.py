#!/usr/bin/env python3
"""Analyze the prespecified 40 x 4 x 3 full-pipeline control after completion and fresh-planner verification."""
from pathlib import Path
from itertools import combinations, product
import argparse, collections, csv, hashlib, json, math, time
import numpy as np
from run_hosted_repeat40_freshplan_20260925 import verify_record_plan


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--wait',action='store_true');args=ap.parse_args();root=args.run.resolve()
    while True:
        state=json.loads((root/'state.json').read_text())
        if state['status']=='complete':break
        if state['status']=='failed':raise RuntimeError('Run failed; preserve partial data: '+state.get('error',''))
        if not args.wait:raise RuntimeError('480 fresh runs are not complete yet')
        time.sleep(30)
    manifest=json.loads((root/'manifest.json').read_text());styles=['formal','natural','casual','emotional'];repeats=['r1','r2','r3'];ids=manifest['selection_ids']
    for relative_path, expected_hash in manifest['sha256'].items():
        assert hashlib.sha256((root.parents[1]/relative_path).read_bytes()).hexdigest()==expected_hash, relative_path
    inputs={d['selection_id']:d for d in map(json.loads,(root/'input/episodes.jsonl').read_text().splitlines())}
    items={d['selection_id']:d for d in map(json.loads,(root/'input/selected_500_intents.jsonl').read_text().splitlines())}
    assert manifest['planner_cache_policy']=='clear_before_and_after_every_episode'
    assert len(ids)==40 and len(set(ids))==40 and len(manifest['schedule'])==480
    assert len({(j['selection_id'],j['style'],j['repeat']) for j in manifest['schedule']})==480
    outcomes=np.zeros((40,4,3),dtype=int);rows=[];usage=collections.Counter();pins={};timestamps=[]
    for job in manifest['schedule']:
        sid=job['selection_id'];style=job['style'];rep=job['repeat']
        p=root/'records'/rep/job['scene_id']/sid/style/'record.json';raw=p.read_bytes();r=json.loads(raw)
        receipt=json.loads((p.parent/'repeat_receipt.json').read_text());sha=hashlib.sha256(raw).hexdigest()
        assert receipt['record_sha256']==sha and r['selection_id']==sid and r['style']==style
        assert all(receipt[k] == job[k] for k in job)
        assert receipt['policy_revision']==manifest['policy_revision']
        witness=receipt['planner_witness']
        assert witness['fresh'] is True and witness['cached'] is False and witness['client_calls']==1
        assert not witness['error'] and witness['usage'] is not None
        assert state['started_at'] <= witness['started_at'] <= witness['finished_at'] <= state['finished_at']
        assert 0 <= witness['finished_at']-witness['started_at'] <= receipt['elapsed_s']+1
        verify_record_plan(r['plan'],witness,manifest['inference']['model'])
        assert r['episode_meta']==inputs[sid] and r['intent']==items[sid][style+'_en']
        assert r['model_meta']['model']==manifest['inference']['model'] and r['model_meta']['temperature']==0
        assert r['trajectory'] and r['step_cap']==30
        # Preserve the reference study's endpoint metric, including its recorded terminal pose.
        distances=[math.dist(t['position'][:2],inputs[sid]['target_position'][:2]) for t in r['trajectory'] if t.get('position')]
        assert distances and all(math.isfinite(d) for d in distances)
        sr=int(distances[-1]<=2);osr=int(min(distances)<=2)
        outcomes[ids.index(sid),styles.index(style),repeats.index(rep)]=sr
        rows.append({**job,'SR':sr,'OSR':osr,'final_distance_m':distances[-1],'record_path':str(p.relative_to(root)),'sha256':sha})
        pins[str(p.relative_to(root))]=sha;usage.update(r['usage_total']);timestamps.append(r['timestamp'])
    assert len(rows)==480 and len(pins)==480 and len(list((root/'records').glob('*/*/*/*/record.json')))==480
    within=np.mean(np.stack([outcomes[:,s,a]!=outcomes[:,s,b] for s in range(4) for a,b in combinations(range(3),2)]),axis=0)
    between=np.mean(np.stack([outcomes[:,a,r]!=outcomes[:,b,q] for a,b in combinations(range(4),2) for r,q in product(range(3),repeat=2)]),axis=0)
    delta=between-within;rng=np.random.default_rng(20260924);boot=rng.integers(0,40,size=(50000,40))
    scenes=np.array([inputs[sid]['scene_id'] for sid in ids]);groups=[np.where(scenes==s)[0] for s in sorted(set(scenes))]
    scene_delta=[]
    for _ in range(10000):
        idx=np.concatenate([groups[k] for k in rng.integers(0,len(groups),size=len(groups))]);scene_delta.append(float(delta[idx].mean()*100))
    def stat(a):return {'mean_percent':float(a.mean()*100),'task_ci95_percent':np.quantile(a[boot].mean(axis=1)*100,[.025,.975]).tolist()}
    result={'complete':True,'records':480,'fresh_initial_plans':480,'cached_initial_plans':0,'tasks':40,'scenes':len(groups),'model':manifest['inference'],'execution_period':[min(timestamps),max(timestamps)],'selection_seed':manifest['selection_seed'],'within_expression_disagreement':stat(within),'cross_expression_disagreement':stat(between),'cross_minus_within_pp':stat(delta),'scene_ci95_difference_pp':np.quantile(scene_delta,[.025,.975]).tolist(),'per_repeat':{rep:{'SR_percent':float(outcomes[:,:,i].mean()*100),'All_four_percent':float(np.all(outcomes[:,:,i],axis=1).mean()*100)} for i,rep in enumerate(repeats)},'usage':dict(usage),'record_hashes':pins,'interpretation':'Fresh initial planning and navigation per episode; within-expression and cross-expression observed outcome variability; the difference is not a causal attributable fraction. Original historical runs are excluded.'}
    out=root/'analysis';out.mkdir(exist_ok=False)
    with (out/'episodes.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    w=result['within_expression_disagreement']['mean_percent'];b=result['cross_expression_disagreement']['mean_percent'];ci=result['cross_minus_within_pp']['task_ci95_percent']
    (out/'RESULTS.md').write_text(f'# Hosted same-expression repeat control\n\nCompleted all 480 fresh trajectories with 480 verified new initial planner calls and zero cached plans: 40 original tasks, four expressions, three repetitions.\n\nWithin-expression outcome disagreement: {w:.2f}%. Cross-expression disagreement: {b:.2f}%. Paired difference: {b-w:+.2f} percentage points, 95% task interval [{ci[0]:.2f}, {ci[1]:.2f}].\n\nThis quantifies observed variability during the recorded execution period; it does not estimate a causal fraction of language-induced failures. Per-repeat SR, All-four, scene-cluster sensitivity and every episode appear in the accompanying JSON/CSV.\n')
    print(json.dumps({k:v for k,v in result.items() if k!='record_hashes'},indent=2),flush=True)

if __name__=='__main__':main()
