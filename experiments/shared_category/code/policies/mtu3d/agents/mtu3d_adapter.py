"""Isolated MTU3D worker and exact Isaac camera-coordinate adapter.

The worker sees only RGB, image-plane depth, camera pose, allowed frontier
coordinates and the supplied query. Instance labels and evaluation goals are
never sent to it. This is an adaptation to this benchmark's observations and
actions, not reproduction of upstream Habitat headline metrics.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from mtu3d_preprocessing import configure_full_frame_features
from mtu3d_observation_support import ObservationSupport
from mtu3d_merge_shadow import install_merge_shadow, rng_snapshot, same_rng
from mtu3d_memory_relations import snapshot_memory
from mtu3d_unique_evidence import UniqueEvidence
from mtu3d_empty_observation import (
    EmptyObjectMemory, exploration_without_object, guard_empty_object_memory,
)
from mtu3d_randomness import decision_seed, seed_generators

# Proper rotation from Isaac Z-up to Habitat Y-up. det=+1, not reflection.
ISAAC_TO_HABITAT = np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])


def camera_to_habitat(position, look_dir) -> tuple[np.ndarray, np.ndarray]:
    forward = np.asarray(look_dir,dtype=np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward,[0.,0.,1.])
    if np.linalg.norm(right)<1e-9:
        raise ValueError('Vertical camera direction is outside the navigation protocol')
    right /= np.linalg.norm(right)
    rotation = np.column_stack([right,np.cross(right,forward),-forward])
    return ISAAC_TO_HABITAT @ np.asarray(position), ISAAC_TO_HABITAT @ rotation


def unproject_normalized(u, v, depth_mm, pose, *, width, height, fx, fy, cx, cy):
    """Convert upstream normalized raster samples with the actual pinhole intrinsics."""
    col=(np.asarray(u)+1)*(width-1)/2
    row=(1-np.asarray(v))*(height-1)/2
    depth=np.asarray(depth_mm)/1000.
    local=np.column_stack([(col-cx)/fx*depth, (cy-row)/fy*depth, -depth,
                           np.ones_like(depth)])
    return (np.asarray(pose) @ local.T).T[:,:3]


class MTU3DAdapter:
    def __init__(self, source: Path, weights: Path, *, base_seed: int | None = None):
        self.base_seed=base_seed
        self.episode_key=None
        self.source=source.resolve()
        # The upstream loader uses relative config and FastSAM paths.
        os.chdir(self.source)
        # MTU3D's common/ is a namespace package. The benchmark's common.py
        # would otherwise shadow it even when MTU3D appears earlier in sys.path.
        own_dir=Path(__file__).resolve().parent
        sys.path[:]=[p for p in sys.path if Path(p or '.').resolve()!=own_dir]
        sys.path[:0]=[str(self.source),str(self.source/'hm3d-online'),str(self.source/'hm3d-online/FastSAM')]
        import torch
        import data_utils as upstream
        self.upstream=upstream
        # Resolve model IDs to recorded immutable snapshots without changing
        # shared Hugging Face refs or relying on whatever "main" names today.
        from transformers import CLIPTextModelWithProjection
        pinned={
            'facebook/dinov2-large':'47b73eefe95e8d44ec3623f8890bd894b6ea2d6c',
            'openai/clip-vit-large-patch14':'32bd64288804d66eefd0ccbe215aa642df71cc41',
        }
        from unittest.mock import patch
        def loader(original):
            def load(name,*args,**kwargs):
                if name in pinned: kwargs.update(revision=pinned[name],local_files_only=True)
                return original(name,*args,**kwargs)
            return load
        classes=[upstream.AutoImageProcessor,upstream.AutoModel,upstream.AutoTokenizer,
                 CLIPTextModelWithProjection]
        with contextlib.ExitStack() as stack:
            for cls in classes:
                stack.enter_context(patch.object(cls,'from_pretrained',side_effect=loader(cls.from_pretrained)))
            self.model=upstream.PQ3DModel(str(weights/'stage1-pretrain-all'),
                                       str(weights/'stage2-fine-tune-ovon'))
        configure_full_frame_features(self.model.image_backbone[0])
        self.checkpoints={}
        for name, module in [('stage1-pretrain-all',self.model.pq3d_stage1),
                             ('stage2-fine-tune-ovon',self.model.pq3d_stage2)]:
            payload=torch.load(weights/name/'pytorch_model.bin',map_location='cpu',weights_only=True)
            disabled=[]
            if name=='stage2-fine-tune-ovon':
                # The released OVON checkpoint has no IMAGE-prompt projection.
                # Its RGB-D mv_encoder and text encoder ARE fully checkpointed.
                # Remove only that unused branch and reject image prompts,
                # rather than permitting arbitrary missing inference weights.
                missing=set(module.state_dict())-set(payload)
                expected={f'image_encoder.input_feat_proj.{layer}.{kind}'
                          for layer in [0,1] for kind in ['weight','bias']}
                if missing!=expected or set(payload)-set(module.state_dict()):
                    raise RuntimeError(f'Unexpected OVON checkpoint mismatch: {sorted(missing)}')
                class RejectImagePrompt(torch.nn.Module):
                    def forward(self,*args,**kwargs):
                        raise RuntimeError('Image-goal prompts are unsupported by the released OVON weights')
                module.image_encoder=RejectImagePrompt()
                from data.datasets.constant import PromptType
                def require_text_prompt(model,inputs):
                    if not bool((inputs[0]['prompt_type']==PromptType.TXT).all()):
                        raise RuntimeError('Only checkpointed text-goal inference is permitted')
                module.register_forward_pre_hook(require_text_prompt)
                disabled=sorted(missing)
            module.load_state_dict(payload,strict=True)
            self.checkpoints[name]={'keys':len(payload),'strict_load':True,
                'disabled_uncheckpointed_image_prompt_keys':disabled}
        self.model.pq3d_stage1.eval();self.model.pq3d_stage2.eval()
        self.observation_support = ObservationSupport(self.model)
        guard_empty_object_memory(self.model.representation_manager)
        if os.environ.get('INAV_MTU_SHADOW_SUPPORT')=='1':
            install_merge_shadow(self.model,self.observation_support)
        self.unique_evidence = UniqueEvidence(self.model.representation_manager, self.observation_support)
        self.stage2_trace = None
        self.model.pq3d_stage2.register_forward_hook(self._record_stage2)

    def _record_stage2(self, module, inputs, output):
        """Log learned evidence without replacing or modifying model outputs."""
        logits = output['og3d_logits'].detach().cpu()[0]
        real = output['real_obj_pad_masks'].bool().detach().cpu()[0]
        boxes = output['query_locs'].detach().cpu()[0]
        probability = output['decision_logits'][0].softmax(dim=-1).detach().cpu()
        object_logits = logits[real]
        indices = object_logits.argsort(descending=True)[:5].tolist()
        candidates = []
        for index in indices:
            box = boxes[real][index].numpy()
            # Upstream stage2 uses [Habitat x, Habitat z, Habitat y].
            world = ISAAC_TO_HABITAT.T @ box[:3][[0, 2, 1]]
            candidates.append({'memory_index': index, 'logit': float(object_logits[index]),
                               'position': world.tolist(), 'box_size': box[3:].tolist(),
                               'observation_support': self.observation_support.for_object(index)})
        relation = None
        archive = os.environ.get('INAV_MTU_RELATION_ARCHIVE')
        if archive:
            relation_rng = rng_snapshot()
            decision_index = self.observation_support.pending_views[0]['decision_index']
            relation = snapshot_memory(self.model.representation_manager, indices,
                archive, self.episode_key, decision_index)
            relation['rng_unchanged'] = same_rng(relation_rng, rng_snapshot())
            if not relation['rng_unchanged']:
                raise ValueError('Memory diagnostic changed RNG')
        self.stage2_trace = {'memory_integration_audit': self.unique_evidence.last_audit,
                            'memory_relation_diagnostic': relation,
                            'goto_frontier_probability': float(probability[0]),
                            'object_count': int(real.sum()),
                            'frontier_count': int((~real).sum()),
                            'top_objects': candidates,
                            'min_decision_num': self.model.min_decision_num,
                            'selected_memory_index': int(object_logits.argmax()),
                            'selected_observation_support': self.observation_support.for_object(int(object_logits.argmax())),
                            'observation_support_audit': self.observation_support.last_audit,
                            'merge_shadow_audit': getattr(self.observation_support,'last_shadow_audit',None)}

    def reset(self, episode_key: str | None = None):
        if self.base_seed is not None:
            # Validate before mutating an existing episode's memory.
            decision_seed(self.base_seed,episode_key,0)
        self.episode_key=episode_key
        self.model.reset()
        self.observation_support.reset()
        self.unique_evidence.reset()
        self.stage2_trace = None

    def decide(self, rgb, depth, *, position, look_dir, intrinsics, frontiers, query,
               decision_index, context_views=None):
        import quaternion
        import torch
        rng_meta={'mode':'legacy_process'}
        if self.base_seed is not None:
            # Index-based streams prevent an extra scan or preceding episode
            # from perturbing the next task's point/attention sampling.
            seed=decision_seed(self.base_seed,self.episode_key,decision_index)
            rng_meta={'mode':'per_decision_v1','episode_key':self.episode_key,
                      **seed_generators(seed)}
        views = list(context_views or []) + [dict(rgb=rgb, depth=depth,
                    position=position, look_dir=look_dir, intrinsics=intrinsics)]
        if len(views) not in {1, 4}:
            raise ValueError('Expected one current view or four charged scan views')
        colors=[];depths=[];states=[];valid_indices=[]
        self.stage2_trace = None
        for index, view in enumerate(views):
            color=view['rgb'];raw=view['depth']
            if color.ndim!=3 or color.shape[2]!=3 or raw.shape!=color.shape[:2]:
                raise ValueError('Expected aligned RGB/depth with matching raster dimensions')
            h,w=raw.shape
            if view['intrinsics']!=intrinsics or intrinsics['width']!=w or intrinsics['height']!=h:
                raise ValueError('All scan views must share the recorded raster and intrinsics')
            clean=np.where(np.isfinite(raw)&(raw>0),raw,0).astype(np.float32)
            valid_depth=int(np.count_nonzero(clean))
            if valid_depth<=1000 or valid_depth/clean.size<.5:
                continue
            pos,rot=camera_to_habitat(view['position'],view['look_dir'])
            sensor=SimpleNamespace(position=pos,rotation=quaternion.from_rotation_matrix(rot))
            colors.append(color);depths.append(clean)
            states.append(SimpleNamespace(position=pos,sensor_states={'color_sensor':sensor}))
            valid_indices.append(index)
        observation_meta={'observations_supplied':len(views),'observations_consumed':len(colors),
                          'valid_view_indices':valid_indices,'decision_index':decision_index,
                          'randomness':rng_meta}
        if not colors:
            result = exploration_without_object(position, [])
            return {**result, 'fallback_reason': 'insufficient_depth',
                    **observation_meta,
                    'object_count': len(self.model.representation_manager.object_box)}
        # Upstream hardcodes 42deg/640x360; use this actual camera instead.
        def unproject(u,v,depth_mm,intr,pose):
            return unproject_normalized(u,v,depth_mm,pose,**intrinsics)
        self.upstream.convert_from_uvd=unproject
        habitat_frontiers=[(ISAAC_TO_HABITAT@np.asarray(x,dtype=np.float32)).astype(np.float32)
                           for x in frontiers]
        self.observation_support.begin(views, decision_index, valid_indices)
        try:
            with torch.inference_mode():
                target,is_object=self.model.decision(colors,depths,states,habitat_frontiers,
                                                    query,decision_index)
        except EmptyObjectMemory:
            result = exploration_without_object(position, frontiers)
            return {**result, 'fallback_reason': 'empty_object_memory',
                    'memory_integration_audit': self.unique_evidence.last_audit,
                    **observation_meta,
                    'object_count': 0,
                    'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2}
        target=ISAAC_TO_HABITAT.T@np.asarray(target,dtype=float)
        if target.shape!=(3,) or not np.isfinite(target).all():
            raise ValueError('Model returned an invalid waypoint')
        return {'target_position':target.tolist(),'is_object_decision':bool(is_object),
                'decision_source': 'mtu3d_learned',
                **observation_meta,'stage2_trace':self.stage2_trace,
                'object_count':len(self.model.representation_manager.object_box),
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20260909)
    args=p.parse_args()
    import torch
    seed_generators(args.seed)
    with contextlib.redirect_stdout(sys.stderr):
        model=MTU3DAdapter(args.source,args.weights,base_seed=args.seed)
    print(json.dumps({'ready':True,'checkpoints':model.checkpoints,
                      'randomness_mode':'per_decision_v1'}),flush=True)
    for line in sys.stdin:
        try:
            request=json.loads(line)
            if request.get('command')=='reset':
                with contextlib.redirect_stdout(sys.stderr): model.reset(request.get('episode_key'))
                result={'reset':True,'episode_key':model.episode_key,
                        'randomness_mode':'per_decision_v1'}
            else:
                context_views=[]
                for view in request.pop('context_observations', []):
                    if set(view)!={'observation','position','look_dir','intrinsics'}:
                        raise ValueError('Unexpected context observation fields')
                    with np.load(view['observation'],allow_pickle=False) as arrays:
                        if set(arrays.files)!={'rgb','depth'}:
                            raise ValueError('Policy context must contain only RGB and depth')
                        context_views.append({k:v for k,v in view.items() if k!='observation'} |
                                             {'rgb':arrays['rgb'],'depth':arrays['depth']})
                with np.load(request.pop('observation'),allow_pickle=False) as arrays:
                    if set(arrays.files)!={'rgb','depth'}: raise ValueError('Policy input must contain only RGB and depth')
                    with contextlib.redirect_stdout(sys.stderr):
                        result=model.decide(arrays['rgb'],arrays['depth'],context_views=context_views,**request)
            print(json.dumps({'ok':True,**result}),flush=True)
        except Exception as exc:
            print(json.dumps({'ok':False,'error':repr(exc)}),flush=True)
            raise


if __name__=='__main__':main()
