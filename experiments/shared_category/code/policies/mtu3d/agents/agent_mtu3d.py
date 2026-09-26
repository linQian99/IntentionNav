"""MTU3D perception/memory/decision policy adapted to the shared Isaac action API.

Use a separate worker environment for MTU3D's compiled dependencies. Input
permissions, 30-action cap, evaluator and geometry checks remain explicit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time

import numpy as np

REPO=Path(__file__).resolve().parents[2]
EVAL_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EVAL_ROOT))
from navigation_query import resolve_category
from common import save_atomic, now_iso
from simulator.walkable_map import WalkableMap
from mtu3d_empty_observation import close_simulator
from mtu3d_commit_scan import CommitScan
from mtu3d_acquired_history import AcquiredHistory
from mtu3d_object_endpoint import plan_object_endpoint, reached_projected_endpoint
from mtu3d_supported_view import plan_supported_view, terminal_view_action
from mtu3d_strict_grid import strict_path_action
from mtu3d_fresh_goal import selected_box, fresh_stop_evidence, plan_observed_goal, approach_action


class Worker:
    def __init__(self, output: Path, seed: int):
        self.log=(output/'worker.log').open('a')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',
                 OPENBLAS_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                 WANDB_MODE='disabled')
        for key in ['PYTHONPATH','LD_LIBRARY_PATH','PYTHONHOME']:
            env.pop(key,None)
        weights=(REPO/'results/category_instance_review_20260909/mtu3d_weights_path.txt').read_text().strip()
        self.process=subprocess.Popen(['/path/to/workspace/envs/mtu3d_inav/bin/python',
            str(Path(__file__).with_name('mtu3d_adapter.py')),'--source',str(REPO/'work_dirs/external/MTU3D'),
            '--weights',weights,'--seed',str(seed)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
            text=True,bufsize=1,env=env)
        try:
            self.ready=self.receive(300)
            if not self.ready.get('ready'):raise RuntimeError(self.ready)
        except BaseException:
            self.close()
            raise

    def receive(self, timeout=180):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout,selectors.EVENT_READ)
            if not selector.select(timeout):raise TimeoutError('MTU3D worker did not respond')
            line=self.process.stdout.readline()
        if not line:raise RuntimeError(f'MTU3D worker exited: {self.process.poll()}')
        response=json.loads(line)
        if response.get('ok') is False:raise RuntimeError(response)
        return response

    def request(self, payload):
        self.process.stdin.write(json.dumps(payload)+'\n');self.process.stdin.flush()
        return self.receive()

    def close(self):
        try:self.process.stdin.close()
        except BrokenPipeError:pass
        try:self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill();self.process.wait(timeout=15)
        self.log.close()


def path_action(wm, current, target, maximum=1.7):
    """Execute only complete collision-free segments on the shared grid."""
    return strict_path_action(wm,current,target,maximum)


def run_episode(env,worker,wm,item,episode,args,query,output,input_provenance=None):
    started=time.monotonic()
    output.mkdir(parents=True,exist_ok=False)
    episode_key=f"{item['scene_id']}/{item['selection_id']}"
    reset=worker.request({'command':'reset','episode_key':episode_key})
    if worker.ready.get('randomness_mode')=='per_decision_v1' and \
            (reset.get('episode_key')!=episode_key or reset.get('randomness_mode')!='per_decision_v1'):
        raise RuntimeError('Worker did not acknowledge isolated episode randomness')
    env.place_agent(episode['start_position'],episode['start_rotation_quat_wxyz'])
    wm.reset_explored()
    pose=env.get_pose();wm.mark_explored(*pose['position'][:2],radius_m=.8)
    trajectory=[{'step':0,'action':'START',**pose}]
    selected_object=None;stop_reason='step_cap';calls=0
    object_endpoint=None
    object_goal_mode=getattr(args,'object_goal_mode','center_only')
    if object_goal_mode not in {'center_only','projected_endpoint','observed_pose','fresh_surface_pose'}:
        raise ValueError('Unknown object goal mode')
    scan_schedule=getattr(args,'scan_schedule','at_commitment')
    exploration_mode=getattr(args,'exploration_mode','local_step')
    if exploration_mode not in {'local_step','boundary_commit'}:
        raise ValueError('Unknown exploration interface')
    frontier_memory=None
    if exploration_mode=='boundary_commit':
        if scan_schedule not in {'before_decision','initial_then_history'}:
            raise ValueError('Boundary exploration requires the paid pre-decision scan schedule')
        from mtu3d_frontier_memory import FrontierMemory
        frontier_memory=FrontierMemory(wm)
    exploration_target=None
    frontier_arrivals=0;frontier_selections=0;frontier_route_failures=0
    if scan_schedule not in {'at_commitment','before_decision','initial_then_history'}:
        raise ValueError('Unknown observation schedule')
    if scan_schedule=='before_decision' and args.commit_scan=='off':
        raise ValueError('Before-decision scans require current or multiview input')
    if scan_schedule=='initial_then_history' and args.commit_scan!='multiview':
        raise ValueError('Acquired history requires multiview input')
    history=AcquiredHistory() if scan_schedule=='initial_then_history' else None
    scan=CommitScan(args.commit_scan)
    scans_started=0;scans_completed=0
    for step in range(1,args.step_cap+1):
        pose=env.get_pose();current=pose['position'][:2]
        rgb=env.render_rgb();depth=env.render_depth()
        if depth is None:raise RuntimeError('Missing image-plane depth')
        observation=output/f'obs_{step:02d}.npz'
        np.savez_compressed(observation,rgb=rgb,depth=depth)
        env.save_frame(rgb,output/f'step_{step:02d}.png')
        frontier_observation=None
        if frontier_memory is not None:
            frontier_observation=frontier_memory.observe(pose['position'],pose['look_dir'])
            if exploration_target is not None and math.dist(current,exploration_target)<=frontier_memory.arrival_radius:
                exploration_target=None
                frontier_arrivals+=1
        view={'observation':str(observation.resolve()),'position':pose['position'],
              'look_dir':pose['look_dir'],'intrinsics':{'width':768,'height':768,
              'fx':384.,'fy':384.,'cx':384.,'cy':384.}}
        if history is not None:
            history.observe(view,step)
        # A reached/turned-to waypoint is a request for new learned evidence,
        # never a terminal decision carried over from the approach commitment.
        if (object_goal_mode=='fresh_surface_pose' and selected_object is not None
                and approach_action(pose['position'],pose['yaw'],object_endpoint)['action']=='REOBSERVE'):
            selected_object=None
            object_endpoint=None
        request=None;decision=None
        confirming=False
        scan_due=(scan_schedule=='before_decision' or
                  (scan_schedule=='initial_then_history' and scans_started==0))
        beginning=scan_due and selected_object is None and exploration_target is None and not scan.active
        if beginning:
            scan.begin(view);scans_started+=1
        if scan.active:
            if not beginning:
                scan.observe(view)
            if not scan.ready:
                phase='begin' if beginning else 'acquire'
                env.look_at_yaw(pose['yaw']+math.pi/2)
                trajectory.append({'step':step,'action':'ROTATE_SCAN',**env.get_pose(),
                    'policy_decision':None,'commit_scan_phase':phase,
                    'scan_views_acquired':len(scan.views)})
                save_atomic({'request':None,'decision':None,'action':'ROTATE_SCAN',
                    'commit_scan_phase':phase,'view':view,
                    'observation_sha256':hashlib.sha256(observation.read_bytes()).hexdigest()},
                    output/f'decision_{step:02d}.json')
                continue
            confirming=True
        if selected_object is None and exploration_target is None:
            proposals=(frontier_memory.proposals(current) if frontier_memory is not None else
                env.sample_frontier_waypoints(wm,K=8,max_step_m=1.7,
                force_angular_spread_rad=2*math.pi,seed=args.seed+step))
            request={**view,
                'frontiers':[[x,y,0.] for x,y,_ in proposals],
                'query':query,'decision_index':calls}
            if confirming:
                request['context_observations']=scan.context()
            elif history is not None:
                request['context_observations']=history.context()
            save_atomic({'request': request,
                'observation_sha256': hashlib.sha256(observation.read_bytes()).hexdigest(),
                'context_observation_sha256': {
                    v['observation']:hashlib.sha256(Path(v['observation']).read_bytes()).hexdigest()
                    for v in request.get('context_observations', [])}},
                output/f'request_{step:02d}.json')
            decision=worker.request(request);calls+=1
            if history is not None:
                decision['acquired_history_steps']=history.steps
            if confirming:
                # Keep legacy scan-completion fields for the shared auditor;
                # schedule and reconfirmation distinguish sensing from a veto.
                decision['commit_scan_confirmation']=True
                decision['commit_scan_mode']=args.commit_scan
                decision['scan_schedule']=scan_schedule
                decision['is_object_reconfirmation']=scan_schedule=='at_commitment'
                scans_completed+=1
                scan.clear()
            target=decision['target_position'][:2]
            if decision['is_object_decision']:
                if args.commit_scan!='off' and not confirming and scan_schedule=='at_commitment':
                    scan.begin(view);scans_started+=1
                    env.look_at_yaw(pose['yaw']+math.pi/2)
                    trajectory.append({'step':step,'action':'ROTATE_SCAN',**env.get_pose(),
                        'policy_decision':decision,'commit_scan_phase':'begin',
                        'scan_views_acquired':1})
                    save_atomic({'request':request,'decision':decision,'action':'ROTATE_SCAN',
                        'commit_scan_phase':'begin',
                        'observation_sha256':hashlib.sha256(observation.read_bytes()).hexdigest()},
                        output/f'decision_{step:02d}.json')
                    continue
                selected_object=target
                if object_goal_mode=='projected_endpoint':
                    object_endpoint=plan_object_endpoint(wm,current,target)
                    decision['object_approach_plan']=object_endpoint
                elif object_goal_mode=='fresh_surface_pose':
                    decision['fresh_stop_evidence']=fresh_stop_evidence(decision,view,rgb,depth)
                    object_endpoint=plan_observed_goal(wm,pose['position'],selected_box(decision),
                        decision['stage2_trace']['selected_observation_support'])
                    decision['object_approach_plan']=object_endpoint
                elif object_goal_mode=='observed_pose':
                    support=decision['stage2_trace']['selected_observation_support']
                    object_endpoint=plan_supported_view(wm,pose['position'],decision['target_position'],support)
                    decision['object_approach_plan']=object_endpoint
            elif frontier_memory is not None:
                if decision.get('action_hint')!='ROTATE_SCAN':
                    if not any(math.dist(target,p[:2])<=1e-4 for p in proposals):
                        raise ValueError('Worker exploration target is not a supplied frontier')
                    exploration_target=target
                    frontier_memory.visit(target)
                    frontier_selections+=1
            if frontier_memory is not None:
                decision['frontier_state']={**frontier_observation,'proposal_count':len(proposals),
                    'selected_exploration_target':exploration_target,
                    'visited_frontiers':len(frontier_memory.visited)}
        else:target=selected_object if selected_object is not None else exploration_target
        distance=math.dist(current,target)
        view_action=None
        if selected_object is not None and object_goal_mode=='fresh_surface_pose':
            if decision is not None and decision['fresh_stop_evidence']['passed']:
                trajectory.append({'step':step,'action':'STOP',**pose,'policy_decision':decision,
                    'object_stop_criterion':'fresh_current_learned_mask_and_goal_region',
                    'object_approach_plan':object_endpoint})
                stop_reason='fresh_current_object_verified'
                save_atomic({'request':request,'decision':decision,'action':'STOP'},output/f'decision_{step:02d}.json')
                break
            view_action=approach_action(pose['position'],pose['yaw'],object_endpoint)
            if view_action['action']=='ROTATE_TARGET_RECENTER':
                env.look_at_yaw(view_action['target_yaw'])
                trajectory.append({'step':step,'action':'ROTATE_TARGET_RECENTER',**env.get_pose(),
                    'policy_decision':decision,'object_approach_plan':object_endpoint})
                save_atomic({'request':request,'decision':decision,'action':'ROTATE_TARGET_RECENTER',
                    'observation_sha256':hashlib.sha256(observation.read_bytes()).hexdigest()},
                    output/f'decision_{step:02d}.json')
                continue
            if view_action['action']=='REOBSERVE':
                # The new call at this pose did not verify a current target.
                # Consume a scan and resume inference; never synthesize STOP.
                object_endpoint=None
                view_action={'action':'NO_SUPPORTED_ROUTE'}
        if selected_object is not None and object_goal_mode=='observed_pose':
            view_action=terminal_view_action(pose['position'],pose['yaw'],object_endpoint)
            if view_action['action']=='ROTATE_TARGET':
                env.look_at_yaw(view_action['target_yaw'])
                trajectory.append({'step':step,'action':'ROTATE_TARGET',**env.get_pose(),
                    'policy_decision':decision,'object_approach_plan':object_endpoint})
                save_atomic({'request':request,'decision':decision,'action':'ROTATE_TARGET',
                    'observation_sha256':hashlib.sha256(observation.read_bytes()).hexdigest()},
                    output/f'decision_{step:02d}.json')
                continue
        # Upstream commits to its selected object and terminates after reaching
        # its predicted location. Keep this decision distinct from DINO gates.
        endpoint_arrived=(selected_object is not None and object_goal_mode=='projected_endpoint'
            and reached_projected_endpoint(wm,current,object_endpoint))
        stop_ready=(view_action['action']=='STOP' if view_action is not None else distance<=.8 or endpoint_arrived)
        if selected_object is not None and stop_ready:
            criterion=('observed_surface_view_pose' if view_action is not None else
                       'predicted_center_radius' if distance<=.8 else 'planned_projected_endpoint')
            trajectory.append({'step':step,'action':'STOP',**pose,'policy_decision':decision,
                'object_stop_criterion':criterion,'object_approach_plan':object_endpoint})
            stop_reason=('model_object_observed_view_reached' if view_action is not None else
                         'model_object_reached' if distance<=.8 else 'model_object_projected_endpoint_reached')
            save_atomic({'request':request,'decision':decision,'action':'STOP'},output/f'decision_{step:02d}.json')
            break
        if view_action is not None:
            next_xy=(path_action(wm,current,object_endpoint['navigation_endpoint_xy'])
                     if object_endpoint is not None else None)
        else:
            next_xy=path_action(wm,current,target)
        if next_xy is None:
            # A no-route state consumes one explicit rotation. It does not
            # acquire a free panorama or teleport to an unwalkable prediction.
            selected_object=None
            object_endpoint=None
            if exploration_target is not None:
                frontier_route_failures+=1
            exploration_target=None
            env.look_at_yaw(pose['yaw']+math.pi/2)
            action='ROTATE_SCAN'
        else:
            env.teleport_to(next_xy,face_direction_xy=target if selected_object is not None and view_action is None else None)
            action='MOVE'
        new_pose=env.get_pose();wm.mark_explored(*new_pose['position'][:2],radius_m=.8)
        if not wm.is_walkable(*new_pose['position'][:2]):raise RuntimeError('Action left the walkable map')
        trajectory.append({'step':step,'action':action,**new_pose,'policy_decision':decision})
        if frontier_memory is not None:
            trajectory[-1]['frontier_observation']=frontier_observation
            trajectory[-1]['committed_exploration_target']=exploration_target
        save_atomic({'request':request,'decision':decision,'action':action,
            'observation_sha256':hashlib.sha256(observation.read_bytes()).hexdigest()},output/f'decision_{step:02d}.json')
    final=env.render_rgb();env.save_frame(final,output/'final.png')
    record={'selection_id':item['selection_id'],'scene_id':item['scene_id'],
        'tier':'active','model':'mtu3d_ovon_adapted','style':args.style,
        'target_category':item['target_category'],'prediction':{'target':query},
        'intent':item[f'{args.style}_en'],'trajectory':trajectory,'step_cap':args.step_cap,
        'stop_reason':stop_reason,'final_frame':str(output/'final.png'),'episode_meta':episode,
        'evaluation_protocol':{'version':'strict_stop_2026_08','goal_input':args.input_mode,
            'sensor_protocol':'forward_view_only_no_free_scans','stop_is_terminal':True,
            'commit_scan_mode':args.commit_scan,'scan_turns_charged':True,
            'scan_schedule':scan_schedule,
            'exploration_mode':exploration_mode,
            'post_budget_actions':False,'post_budget_policy_observations':False,
            'privileged_target_position':False,'max_step_m':1.7},
        'model_meta':{'api_calls':0,'policy':'MTU3D trained perception, representation merging and object/frontier decision',
            'decision_calls':calls,'worker_checkpoints':worker.ready['checkpoints'],
            'randomness_mode':worker.ready.get('randomness_mode','legacy_process'),
            'episode_randomness_key':episode_key,
            'scan_schedule':scan_schedule,
            'empty_evidence_fallbacks': sum(
                bool((entry.get('policy_decision') or {}).get('fallback_reason'))
                for entry in trajectory),
            'commit_scans_started':scans_started,'commit_scans_completed':scans_completed,
            'commit_scan_incomplete_at_budget':scan.active,
            'frontier_selections':frontier_selections,'frontier_arrivals':frontier_arrivals,
            'frontier_route_failures':frontier_route_failures,
            'frontier_acquired_views':frontier_memory.observations if frontier_memory is not None else 0,
            'adaptations':['actual Isaac intrinsics and proper world rotation',
                'current or four acquired views; initial_then_history scans once then reuses recent movement views; other schedules retain paid scans',
                'exploration_mode specifies local action proposals or observed boundaries with persistent execution',
                'shared permitted nonsemantic map and bounded collision-checked actions',
                'fresh_surface_pose returns to observed cameras or source-ray range entry and requires a current learned mask plus camera/range geometry before STOP; original learned gates unchanged, no GT access']},
        'run_config':{'seed':args.seed,'commit_scan':args.commit_scan,'scan_schedule':scan_schedule,
            'exploration_mode':exploration_mode,'object_goal_mode':object_goal_mode},'elapsed_seconds':time.monotonic()-started,
        'timestamp':now_iso()}
    # Evaluation-only identity information is queried after the policy ends.
    record['evaluation_diagnostics']={'simulator_instance_visibility':
        env.target_instance_visibility(episode['target_object_id'])}
    if input_provenance is not None:
        record['navigation_input']=input_provenance
    save_atomic(record,output/'record.json')
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--items',type=Path,required=True);p.add_argument('--episodes',type=Path,required=True)
    p.add_argument('--selections',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input-mode',choices=['explicit_target','frozen_predicted_category','shared_explicit','shared_implicit_category'],default='explicit_target')
    p.add_argument('--intent-records',type=Path);p.add_argument('--style',default='formal')
    p.add_argument('--shared-query-inputs',type=Path)
    p.add_argument('--step-cap',type=int,default=30);p.add_argument('--seed',type=int,default=20260909)
    p.add_argument('--commit-scan',choices=['off','current','multiview'],default='off')
    p.add_argument('--scan-schedule',choices=['at_commitment','before_decision','initial_then_history'],default='at_commitment')
    p.add_argument('--exploration-mode',choices=['local_step','boundary_commit'],default='local_step')
    p.add_argument('--object-goal-mode',choices=['center_only','projected_endpoint','observed_pose','fresh_surface_pose'],default='center_only')
    p.add_argument('--check',action='store_true',help='Validate inputs without importing Isaac or starting a worker')
    args=p.parse_args()
    if args.input_mode.startswith("shared_") != (args.shared_query_inputs is not None):
        raise ValueError("Shared mode and --shared-query-inputs must be used together")
    if args.shared_query_inputs is not None and args.intent_records is not None:
        raise ValueError("Use one frozen input source")
    if args.scan_schedule=='before_decision' and args.commit_scan=='off':
        raise ValueError('Before-decision scans require current or multiview input')
    if args.scan_schedule=='initial_then_history' and args.commit_scan!='multiview':
        raise ValueError('Acquired history requires multiview input')
    if args.exploration_mode=='boundary_commit' and args.scan_schedule not in {'before_decision','initial_then_history'}:
        raise ValueError('Boundary exploration requires paid pre-decision sensing')
    import re
    selection_rows=[s.strip() for s in args.selections.read_text().splitlines() if s.strip() and not s.startswith('#')]
    if not selection_rows or any(not re.fullmatch(r'SEL_\d+',s) for s in selection_rows):
        raise ValueError('Empty or malformed selection list')
    selections=set(selection_rows)
    if len(selections)!=len(selection_rows):raise ValueError('Duplicate selections')
    items=[json.loads(s) for s in args.items.read_text().splitlines() if s.strip()]
    items=sorted([x for x in items if x['selection_id'] in selections],key=lambda x:(x['scene_id'],x['selection_id']))
    if len(items)!=len(selections) or {x['selection_id'] for x in items}!=selections:
        raise ValueError('Selection mismatch or duplicate items')
    episode_rows=[json.loads(s) for s in args.episodes.read_text().splitlines() if s.strip()]
    episodes={x['selection_id']:x for x in episode_rows}
    if len(episodes)!=len(episode_rows):raise ValueError('Duplicate episodes')
    if args.step_cap!=30:raise ValueError('This frozen development protocol requires 30 actions')
    queries={};input_hashes={};query_provenance={}
    for path in [args.items,args.episodes,args.selections,Path(__file__),
                 Path(__file__).with_name('mtu3d_adapter.py'),Path(__file__).with_name('mtu3d_preprocessing.py'),
                 Path(__file__).with_name('mtu3d_empty_observation.py'),Path(__file__).with_name('mtu3d_commit_scan.py'),
                 Path(__file__).with_name('mtu3d_randomness.py'),
                 Path(__file__).with_name('mtu3d_acquired_history.py'),
                 Path(__file__).with_name('mtu3d_observation_support.py'),
                 Path(__file__).with_name('mtu3d_supported_view.py'),
                 Path(__file__).with_name('mtu3d_merge_shadow.py'),
                 Path(__file__).with_name('mtu3d_strict_grid.py'),
                 Path(__file__).with_name('mtu3d_fresh_goal.py'),
                 Path(__file__).with_name('mtu3d_surface_sample.py'),
                 Path(__file__).with_name('mtu3d_unique_evidence.py'),
                 Path(__file__).with_name('mtu3d_memory_relations.py'),
                 Path(__file__).with_name('mtu3d_frontier_memory.py'),Path(__file__).with_name('mtu3d_object_endpoint.py'),
                 Path(__file__).with_name('mtu3d_frontier_grid.py'),
                 Path(__file__).with_name('MTU3D_FRONTIER_SOURCE_LICENSE.txt')]:
        input_hashes[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    for item in items:
        sid=item['selection_id'];ep=episodes[sid]
        if ep['scene_id']!=item['scene_id'] or ep['target_category']!=item['target_category']:
            raise ValueError(f'Item/episode mismatch: {sid}')
        if args.input_mode.startswith('shared_'):
            category,provenance=resolve_category(args.shared_query_inputs,item=item,
                style=args.style,input_mode=args.input_mode.removeprefix('shared_'))
            queries[sid]=category.replace('_',' ')
            query_provenance[sid]=provenance
            input_hashes[provenance['policy_record']]=provenance['policy_record_sha256']
            input_hashes[str((args.shared_query_inputs/'policy_manifest.json').resolve())]=provenance['policy_index_sha256']
        elif args.input_mode=='explicit_target':
            queries[sid]=item['target_category'].replace('_',' ')
        else:
            if args.intent_records is None:raise ValueError('Frozen predictions required')
            # Use the same checked and sanitized front-end contract as R055.
            from agent_vlm_engine import load_open_weight_policy_plan
            plan,provenance=load_open_weight_policy_plan(args.intent_records,item,args.style,item[f'{args.style}_en'])
            queries[sid]=plan['target_guess']
            input_hashes[provenance['policy_record']]=provenance['policy_record_sha256']
        scene_path=Path('/path/to/workspace/datasets/vlntube/TataServices')/ep['scene_id']/'start_result_navigation.usd'
        if not scene_path.is_file():raise FileNotFoundError(scene_path)
        wm=WalkableMap.load(ep['scene_id'],metaroot=Path('/path/to/workspace/datasets/vlntube/TaTaMeta/metadata_train'))
        if wm is None or not wm.is_walkable(*ep['start_position'][:2]):
            raise ValueError(f'Invalid start or missing map: {sid}')
        if args.exploration_mode=='boundary_commit':
            from mtu3d_frontier_memory import FrontierMemory
            FrontierMemory(wm)
    if args.check:
        print(json.dumps({'passed':True,'episodes':len(items),'input_mode':args.input_mode,
            'step_cap':args.step_cap,'commit_scan':args.commit_scan,'scan_schedule':args.scan_schedule,
            'exploration_mode':args.exploration_mode,'object_goal_mode':args.object_goal_mode,
            'gpu_initialized':False,'output_created':False}))
        return
    if args.output.exists():raise FileExistsError('Use a fresh immutable output directory')
    args.output.mkdir(parents=True)
    save_atomic({'inputs':input_hashes,'queries':queries,'seed':args.seed,
        'input_mode':args.input_mode,'step_cap':args.step_cap,'commit_scan':args.commit_scan,
        'scan_schedule':args.scan_schedule,
        'exploration_mode':args.exploration_mode,'object_goal_mode':args.object_goal_mode,
        'shared_query_provenance':query_provenance,
        'status':'development_only'},args.output/'run_manifest.json')
    sys.argv=sys.argv[:1]
    env=None;worker=None;completed=[];failed=[]
    try:
        from simulator.iss_env import IsaacSimEnv
        env=IsaacSimEnv(headless=True)
        worker=Worker(args.output,args.seed);scene=None
        for item in items:
            sid=item['selection_id'];episode=episodes[sid]
            query=queries[sid]
            if scene!=episode['scene_id']:
                scene=episode['scene_id']
                env.load_scene(scene,str(Path('/path/to/workspace/datasets/vlntube/TataServices')/scene/'start_result_navigation.usd'))
                wm=WalkableMap.load(scene,metaroot=Path('/path/to/workspace/datasets/vlntube/TaTaMeta/metadata_train'))
                if wm is None:raise FileNotFoundError(f'No map: {scene}')
            try:
                run_episode(env,worker,wm,item,episode,args,query,args.output/'episodes'/scene/sid/args.style,
                    input_provenance=query_provenance.get(sid))
                completed.append(sid);print('EPISODE_COMPLETE',sid,flush=True)
            except Exception as exc:
                failed.append({'selection_id':sid,'error':repr(exc)})
                raise
    except BaseException as exc:
        if not failed:failed.append({'error':repr(exc),'stage':'initialization_or_scene_load'})
        raise
    finally:
        try:
            if worker is not None:worker.close()
        except Exception as exc:
            failed.append({'stage':'worker_cleanup','error':repr(exc)})
        success=len(completed)==len(items) and not failed
        save_atomic({'complete':success,'completed':completed,'failed':failed,'expected':len(items)},args.output/'summary.json')
        if success:(args.output/'.RUN_SUCCESS').write_text('all requested episode records persisted\n')
        # Isaac SimulationApp.close may exit Python directly. Persist the
        # already-complete records before invoking it, then let the supervisor
        # verify process termination and release of GPU0.
        if env is not None:
            try:close_simulator(env)
            except Exception as exc:
                failed.append({'stage':'simulator_cleanup','error':repr(exc)})
                (args.output/'.RUN_SUCCESS').unlink(missing_ok=True)
                save_atomic({'complete':False,'completed':completed,'failed':failed,'expected':len(items)},args.output/'summary.json')
                raise
        if not success:
            raise RuntimeError(f'Navigation run incomplete: {failed}')


if __name__=='__main__':main()
