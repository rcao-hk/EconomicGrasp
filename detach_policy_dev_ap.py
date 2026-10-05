"""Fixed-manifest P4 development AP; production model, decoder and evaluator."""
import argparse
import gc
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch
import checkerboard_probe as q
from detach_policy_gate import set_cfg, resolved

ARMS = {'A': 'A_all_open_r1', 'B': 'B_detach_E_r1', 'C': 'C_detach_all_r1'}
MODEL_KEYS = ('img', 'K', 'camera_pose_vec', 'camera_gravity_vec')


def checkpoint_model(prod, out, arm):
    """Reconstruct canonical frozen tensors plus the complete rank-0 model delta."""
    manifest = json.loads((out/'canonical_manifest.json').read_text())
    assert q.sha(out/'canonical_step0.pt') == manifest['sha256']
    entries = [json.loads(s) for s in (out/ARMS[arm]/'checkpoint_manifest.jsonl').read_text().splitlines()]
    entry = next(e for e in reversed(entries) if e['position']['step'] == 17070)
    assert q.sha(entry['path']) == entry['sha256']
    initial = torch.load(out/'canonical_step0.pt', map_location='cpu', weights_only=False)
    saved = torch.load(entry['path'], map_location='cpu', weights_only=False)
    assert saved['canonical_sha256'] == manifest['sha256'] and saved['arm'] == arm
    assert saved['position'] == dict(step=17070, epoch=6, batch=0)
    state = initial['model_state_dict']
    assert set(state)-set(saved['model_delta']) == set(initial['frozen_keys'])
    assert not set(saved['model_delta'])-set(state)
    state.update(saved['model_delta'])
    c = prod.cfgs
    model = prod.economicgrasp_dpt(
        min_depth=c.min_depth, max_depth=c.max_depth, bin_num=c.bin_num,
        is_training=False, use_obs_depth=False, use_depth_comp=False,
        geometry_depth_source='pred', seed_selection_mode='point_fps',
        pose_depth_mode=c.pose_depth_mode, use_cdf=True,
        detach_depth=bool(c.detach_depth), detach_depth_gse=c.detach_depth_gse,
        detach_depth_seed_xyz=c.detach_depth_seed_xyz, detach_depth_support=c.detach_depth_support,
        vis_dir=None).cuda().eval()
    model.load_state_dict(state, strict=True)
    assert resolved(model) == saved['resolved']
    assert not model.is_training and not model.use_obs_depth and model.geometry_depth_source == 'pred'
    digest = q.tree_hash(model.state_dict())
    assert digest == q.tree_hash(state)
    return model, dict(checkpoint=entry, model_hash=digest, canonical_sha256=manifest['sha256'],
                       resolved=resolved(model))


def run(a):
    out=a.output
    dest=out/('dev_ap_smoke' if a.smoke else 'dev_ap')/ARMS[a.arm]
    dest.mkdir(parents=True, exist_ok=False)
    try:
        prod=q.imports(out); c=prod.cfgs; set_cfg(c,a.arm)
        # Freeze a common inference contract before inspecting AP.
        c.use_top4_view_infer=False; c.use_cdf=True; c.use_obs_depth=False
        c.vis_dir=None; c.batch_size=1; c.num_workers=0; c.seed=0
        q.seed(0)
        from dataset.graspnet_dataset import GraspNetMultiDataset, collate_fn
        from inference_cva import _move_fixed_inputs
        from models.economicgrasp_bip3d import pred_decode_center_view_angle
        from utils.collision_detector import ModelFreeCollisionDetectorTorch
        from graspnetAPI import GraspGroup, GraspNetEval
        dev=json.loads((out/'development_manifest.json').read_text())
        assert dev['AP_scenes']==[100,110,119,129] and dev['AP_frames']==list(range(256))
        frames=[0] if a.smoke else dev['AP_frames']
        indices=[(s-100)*256+f for s in dev['AP_scenes'] for f in frames]
        ds=GraspNetMultiDataset(c.dataset_root,split='test_seen',camera=c.camera,num_points=c.num_point,
            remove_outlier=True,augment=False,load_label=False,use_gt_depth=False,
            use_fuse_depth=c.use_fuse_depth,graspness_mode=c.graspness_mode,
            min_depth=c.min_depth,max_depth=c.max_depth,bin_num=c.bin_num)
        model,load_audit=checkpoint_model(prod,out,a.arm)
        contract=dict(arm=a.arm,seed=0,global_step=17070,split='test_seen',scenes=dev['AP_scenes'],frames=frames,
            development_manifest_sha256=q.sha(out/'development_manifest.json'),load=load_audit,
            config=vars(c),batch_size=1,workers=0,per_frame_seed='dataset index; common across arms',
            model_input_keys=MODEL_KEYS,geometry_depth_source='pred',use_top4_view_infer=False,
            collision=dict(threshold=.01,voxel_size=.01,approach_dist=.05,cloud='observed sensor cloud; repository postfilter'),
            preprocessing='unchanged dataset RGB crop based on sensor workspace mask and segmentation; not a sensor-free pipeline',
            evaluator=dict(path=inspect.getfile(GraspNetEval),sha256=q.sha(inspect.getfile(GraspNetEval)),
                           TOP_K=50,max_width=.1,frictions=[.2,.4,.6,.8,1.,1.2]),
            code_git=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            source_hashes={n:q.sha(n) for n in ['detach_policy_dev_ap.py','inference_cva.py',
                'models/economicgrasp_bip3d.py','dataset/graspnet_dataset.py','utils/collision_detector.py']})
        q.dump(dest/'contract.json',contract)
        record=[];start=time.time()
        with (dest/'inference_frames.jsonl').open('w',buffering=1) as log:
            for index in indices:
                q.seed(index)
                sample=ds[index]
                batch=_move_fixed_inputs(collate_fn([{k:sample[k] for k in MODEL_KEYS}]),torch.device('cuda:0'))
                input_hash=q.tree_hash(batch)
                # No GT/sensor depth, label, point cloud or segmentation enters the model.
                q.seed(index)
                with torch.inference_mode():
                    end=model(batch);pred=pred_decode_center_view_angle(end,use_cdf=True)[0]
                assert torch.isfinite(pred).all() and pred.shape[1]==17
                if a.smoke:
                    q.seed(index)
                    with torch.inference_mode():
                        full=model(_move_fixed_inputs(collate_fn([sample]),torch.device('cuda:0')))
                        ref=pred_decode_center_view_angle(full,use_cdf=True)[0]
                    assert torch.equal(pred,ref), 'Restricted and repository full-input forwards differ'
                    del full,ref
                gg=GraspGroup(pred.cpu().numpy());raw=len(gg)
                if raw:
                    cloud,_=ds.get_data(index,return_raw_cloud=True)
                    detector=ModelFreeCollisionDetectorTorch(cloud.reshape(-1,3),voxel_size=.01)
                    mask=detector.detect(gg,approach_dist=.05,collision_thresh=.01)
                    gg=gg[~mask.cpu().numpy()]
                    del cloud,detector,mask
                sid=index//256+100;frame=index%256
                path=dest/f'scene_{sid:04d}'/c.camera/f'{frame:04d}.npy'
                path.parent.mkdir(parents=True,exist_ok=True);gg.save_npy(str(path))
                row=dict(group=a.arm,seed=0,global_step=17070,split='test_seen',scene=sid,frame=frame,
                    input_sha256=input_hash,decoded=raw,retained=len(gg),empty=len(gg)==0,
                    prediction_sha256=q.sha(path),seconds=time.time()-start)
                record.append(row);log.write(json.dumps(row)+'\n')
                if len(record)%32==0 or a.smoke:
                    print(json.dumps(dict(stage='inference',done=len(record),total=len(indices),
                        seconds_per_frame=(time.time()-start)/len(record))),flush=True)
                del batch,end,pred,gg,sample
        q.dump(dest/'inference_status.json',dict(status='complete',frames=len(record),seconds=time.time()-start,
            coverage_nonempty=sum(r['retained']>0 for r in record)/len(record),
            mean_decoded=float(np.mean([r['decoded'] for r in record])),
            mean_retained=float(np.mean([r['retained'] for r in record]))))
        del model,ds;gc.collect();torch.cuda.empty_cache()
        ge=GraspNetEval(root=c.dataset_root,camera=c.camera,split='test_seen')
        results=[];summaries=[]
        for scene in dev['AP_scenes']:
            q.seed(0)
            acc=np.asarray(ge.eval_scene(scene,str(dest),anno_sample_ratio=1/256 if a.smoke else 1.,
                                         TOP_K=50,max_width=.1))
            assert acc.shape==(len(frames),50,6) and np.isfinite(acc).all()
            assert ((acc>=0)&(acc<=1)).all()
            np.save(dest/f'ap_scene_{scene:04d}.npy',acc)
            r=dict(group=a.arm,seed=0,global_step=17070,split='test_seen_dev',scene=scene,
                   frames=len(frames),AP_percent=float(acc.mean()*100),
                   AP04_percent=float(acc[:,:,1].mean()*100),AP08_percent=float(acc[:,:,3].mean()*100))
            summaries.append(r);results.append(acc);q.dump(dest/'scene_summary.json',summaries)
            print('\n'+json.dumps(dict(stage='evaluation',**r)),flush=True)
        res=np.stack(results);np.save(dest/'ap_development.npy',res)
        q.dump(dest/'status.json',dict(status='complete',smoke=a.smoke,arm=a.arm,frames=len(indices),
            AP_percent=float(res.mean()*100),seconds=time.time()-start))
    except Exception:
        q.dump(dest/'status.json',dict(status='failed',traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--smoke',action='store_true')
    run(parser.parse_args())

