#!/usr/bin/env python3
"""Server-only flag-off parity: native main versus wrapper, same weights/input.

Uses a fresh cold model at reduced seed count for a correctness check, not an
AP measurement. Does not alter source code, train parameters, or write a cache.
"""
import argparse,copy,sys
import torch
from moge_rayrope.config import ModelConfig,LossConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',default='/data/robotarm/dataset/graspnet')
    p.add_argument('--label-folder',default='economic_grasp_label_300views_extend_angle_cdf_depth')
    p.add_argument('--output',default='baseline_parity.json')
    p.add_argument('--seeds',type=int,default=32)
    a=p.parse_args(); sys.argv=[sys.argv[0]]
    if not torch.cuda.is_available(): raise RuntimeError('This parity test requires main CUDA extensions and GraspNet')
    from moge_rayrope.runtime import verify_main_contract,configure_main,make_dataset,move_batch,seed_all,atomic_json
    from moge_rayrope.model import EconomicGraspMoGeRayRoPE,filter_empty_objects
    from moge_rayrope.objective import objective
    mc=ModelConfig(seeds=a.seeds,use_moge=False,use_rayrope=False)
    verify_main_contract(); configure_main(mc,a.dataset_root,a.label_folder,True); seed_all(0)
    from dataset.graspnet_dataset import collate_fn
    from models.loss_economicgrasp_depth_kview_transformer import get_loss
    from models.economicgrasp_bip3d import pred_decode_center_view_angle
    wrapped=EconomicGraspMoGeRayRoPE(mc).cuda().eval()
    native=copy.deepcopy(wrapped.base).eval()
    for module in native.modules():
        if hasattr(module,'is_training'): module.is_training=False
    _,data,_,_=make_dataset(a.dataset_root,'test_seen',.1,True,mc,a.label_folder,True,max_frames=2)
    differences=[]
    with torch.no_grad():
        for i in range(len(data)):
            raw=move_batch(collate_fn([data[i]]),'cuda',True)
            plain=filter_empty_objects(raw); plain['cva_force_process_grasp_labels']=True; plain['cva_compute_diagnostics']=False
            seed_all(71+i); ref=native(dict(plain)); loss_ref,_=get_loss(ref,use_cdf=True)
            seed_all(71+i); out=wrapped(dict(raw)); loss_new,_,_=objective(out,mc,LossConfig())
            one={}
            for name in ('depth_map_pred','xyz_graspable','grasp_top_view_xyz',
                         'grasp_width_pred_angle_depth','grasp_cdf_pred_angle_depth'):
                torch.testing.assert_close(out[name],ref[name],atol=2e-6,rtol=2e-6)
                one[name]=float((out[name]-ref[name]).abs().max())
            torch.testing.assert_close(loss_new,loss_ref,atol=2e-6,rtol=2e-6)
            da=pred_decode_center_view_angle(out,use_cdf=True); db=pred_decode_center_view_angle(ref,use_cdf=True)
            for aa,bb in zip(da,db): torch.testing.assert_close(aa,bb,atol=2e-6,rtol=2e-6)
            one['loss_abs_diff']=float((loss_new-loss_ref).abs()); differences.append(one)
    atomic_json(a.output,{'baseline_parity':'passed','samples':len(data),'seeds':a.seeds,
                          'max_abs_differences':differences,'not_an_AP_result':True})
    print('MR_BASELINE_PARITY_PASSED',flush=True)

if __name__=='__main__': main()
