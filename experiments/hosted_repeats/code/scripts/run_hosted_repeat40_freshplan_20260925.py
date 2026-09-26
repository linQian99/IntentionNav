#!/usr/bin/env python3
"""Frozen hosted policy, original inputs, full-pipeline repetitions with fresh planner calls."""
from pathlib import Path
import argparse, hashlib, json, os, random, shutil, subprocess, sys, time
from contextlib import contextmanager
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/hosted_repeat40_freshplan_20260925'
SOURCE = ROOT / 'experiments/sources/history/eval/hosted_repeat_c3c9ee08_20260924/eval'

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def verify_record_plan(plan, witness, model):
    """The frozen engine adds model/usage metadata when serializing its plan."""
    if plan.get('_cached') or plan.get('error'):
        raise RuntimeError('Record contains a cached or failed plan')
    if plan['plan_model'] != model or plan['plan_usage'] != witness['usage']:
        raise RuntimeError('Record planner metadata differs from witnessed call')
    content = {k: v for k, v in plan.items() if k not in ('plan_model', 'plan_usage')}
    if json_digest(content) != witness['plan_sha256']:
        raise RuntimeError('Record plan content differs from witnessed call')

@contextmanager
def fresh_episode_plan(agent, planner):
    """Reset episode state and witness the frozen planner's real client call."""
    native_plan = agent.make_episode_plan
    if native_plan is not planner.make_episode_plan:
        raise RuntimeError('Unexpected planner binding')
    planner._plan_cache.clear()
    agent._scene_plan_cache.clear()
    witnesses = []

    def watched_plan(*args, **kwargs):
        if witnesses:
            raise RuntimeError('More than one initial plan requested')
        native_call = planner.call_vlm
        calls = []
        def watched_call(*call_args, **call_kwargs):
            started = time.time()
            result = native_call(*call_args, **call_kwargs)
            calls.append({'started_at': started, 'finished_at': time.time(),
                          'usage': result[2], 'error': bool(result[1])})
            return result
        planner.call_vlm = watched_call
        try:
            plan, usage = native_plan(*args, **kwargs)
        finally:
            planner.call_vlm = native_call
        if (plan.get('_cached') or plan.get('error') or len(calls) != 1
                or calls[0]['error'] or usage is None):
            raise RuntimeError('Fresh initial planner call not witnessed')
        witnesses.append({'fresh': True, 'cached': False, 'client_calls': 1,
                          'plan_sha256': json_digest(plan), **calls[0]})
        return plan, usage

    agent.make_episode_plan = watched_plan
    try:
        yield witnesses
        if len(witnesses) != 1:
            raise RuntimeError('Episode did not call the initial planner exactly once')
    finally:
        agent.make_episode_plan = native_plan
        planner._plan_cache.clear()
        agent._scene_plan_cache.clear()

def save(p, value):
    tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)); tmp.replace(p)

def check():
    m = json.loads((OUT / 'manifest.json').read_text())
    for p, h in m['sha256'].items():
        if digest(ROOT / p) != h: raise RuntimeError('Changed frozen input: ' + p)
    assert len(m['selection_ids']) == 40 and len(m['schedule']) == 480
    assert len({(x['selection_id'], x['style'], x['repeat']) for x in m['schedule']}) == 480
    assert shutil.disk_usage(OUT).free > 20 * 1024**3
    print('PREFLIGHT_OK', len(m['sha256']), 'pins; 40 tasks, 480 unique episodes', flush=True)
    return m

def worker(m, scene):
    # Clear inherited policy overrides; original snapshot defaults define this run.
    for key in list(os.environ):
        if key.startswith(('VM_', 'AUTO_REORIENT', 'ANGULAR_SPREAD', 'FALLBACK_RETRY')):
            del os.environ[key]
    os.environ.update(CUDA_VISIBLE_DEVICES='0', DINO_DEVICE='cuda:0', STRICT_VLM_FAILURE='1',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PP_API_RETRY_LOG='1')
    sys.path[:0] = [str(SOURCE / 'agents'), str(SOURCE / 'simulator'), str(SOURCE)]
    import common
    common.REPO = ROOT
    common.DATASET_ROOT = OUT / 'input'
    common.DATASET_JSONL = common.DATASET_ROOT / 'selected_500_intents.jsonl'
    import agent_vlm_engine as agent
    import agent_vlm as planner
    agent.REPO = ROOT
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(20260924)
    x = torch.randn((16, 16), device='cuda:0')
    assert bool(torch.isfinite(x @ x).all())
    print('CUDA_WITNESS', torch.cuda.get_device_name(0), flush=True)
    # Resource-only wrapper: preserve the frozen rendering settings, force GPU0.
    import isaacsim
    native = isaacsim.SimulationApp
    def gpu0_app(config, *args, **kwargs):
        return native({**config, 'active_gpu': 0, 'physics_gpu': 0, 'multi_gpu': False}, *args, **kwargs)
    isaacsim.SimulationApp = gpu0_app
    from simulator.iss_env import IsaacSimEnv
    from walkable_map import WalkableMap
    items = {d['selection_id']: d for d in map(json.loads, (OUT / 'input/selected_500_intents.jsonl').read_text().splitlines())}
    episodes = {d['selection_id']: d for d in map(json.loads, (OUT / 'input/episodes.jsonl').read_text().splitlines())}
    jobs = [j for j in m['schedule'] if j['scene_id'] == scene]
    sys.argv = sys.argv[:1]
    env = IsaacSimEnv(headless=True)
    try:
        env.load_scene(scene, str(common.USD_ROOT / scene / 'start_result_navigation.usd'))
        for j in jobs:
            p = OUT / 'records' / j['repeat'] / scene / j['selection_id'] / j['style'] / 'record.json'
            if p.exists(): raise RuntimeError('Output exists; no automatic overwrite/retry: ' + str(p))
            if shutil.disk_usage(OUT).free < 20 * 1024**3: raise RuntimeError('20GiB disk floor')
            seed = j['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            wm = WalkableMap.load(scene)
            if wm is None: raise RuntimeError('Missing walkable map')
            start = time.monotonic()
            with fresh_episode_plan(agent, planner) as plan_witness:
                agent.run_episode_engine(env, wm, items[j['selection_id']], episodes[j['selection_id']],
                                         j['style'], 'gemini_3_1_flash', 30, p)
            r = json.loads(p.read_text())
            assert r['selection_id'] == j['selection_id'] and r['style'] == j['style']
            assert r['episode_meta'] == episodes[j['selection_id']]
            verify_record_plan(r['plan'], plan_witness[0], m['inference']['model'])
            save(p.parent / 'repeat_receipt.json', {**j, 'elapsed_s': time.monotonic()-start,
                 'record_sha256': digest(p), 'policy_revision': m['policy_revision'],
                 'planner_witness': plan_witness[0]})
            print('EPISODE_DONE', j['selection_id'], j['style'], j['repeat'], flush=True)
    finally:
        env.close()

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--check', action='store_true'); ap.add_argument('--worker-scene')
    args = ap.parse_args(); m = check()
    if args.check: return
    if args.worker_scene: return worker(m, args.worker_scene)
    if (OUT / 'state.json').exists():
        raise RuntimeError('Run already submitted; preserve state and all existing outputs')
    usage = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],text=True)
    if int(usage.strip()) >= 500: raise RuntimeError('GPU0 is occupied')
    state = {'status': 'running', 'pid': os.getpid(), 'started_at': time.time(), 'expected':480, 'completed_scenes': []}
    save(OUT / 'state.json', state)
    try:
        for scene in sorted({j['scene_id'] for j in m['schedule']}):
            state['active_scene'] = scene; save(OUT / 'state.json', state)
            env = os.environ.copy(); env['OMNI_USER_PATH'] = str(OUT / 'omni_user' / scene)
            with (OUT / 'logs' / (scene + '.log')).open('w') as log:
                proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker-scene', scene],
                                      stdout=log, stderr=subprocess.STDOUT, env=env, timeout=3*3600)
            if proc.returncode: raise RuntimeError(f'{scene}: worker exit {proc.returncode}; see preserved log')
            state['completed_scenes'].append(scene)
            state['completed'] = len(list((OUT/'records').glob('*/*/*/*/record.json')))
            save(OUT / 'state.json', state)
        assert state['completed'] == 480
        state.update(status='complete', finished_at=time.time())
    except BaseException as exc:
        state.update(status='failed', error=str(exc), finished_at=time.time()); raise
    finally: save(OUT / 'state.json', state)

if __name__ == '__main__': main()
