"""DCR math, gradient isolation, dispatch and mocked CPU integration tests.

Mocks explicitly replace the absent CUDA Stage-1/CAD environment. These tests
are NOT evidence of real GraspNet throughput, AP or GPU correctness.
"""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import numpy as np
import pytest
import torch
from torch import nn
from test_e1e2_cva import mock_reference, mock_rotations, native_actions
from e1e2_common import (VERSION, array_sha, batch_fingerprint, digest, expand_centers,
                        file_sha, relative_cdf_loss, save_npz, save_torch)
from dcr_cva_common import (RankResidualHead, anchor_logit, anchored_score,
                           make_outputs, pairwise_ranking_loss, selected_rank_loss)
from models.economicgrasp_cva_centers import CenterHypothesisCVA
from models.economicgrasp_cva_dcr import DecoupledCenterRankingCVA


def install_rotations(monkeypatch):
    stub=types.ModuleType('utils.label_generation')
    stub.batch_viewpoint_params_to_matrix=mock_rotations
    monkeypatch.setitem(sys.modules,'utils.label_generation',stub)


def batch_for(sid=0,aid=0):
    rng=torch.Generator().manual_seed(sid*1000+aid)
    return {'img':torch.randn(1,3,28,28,generator=rng),'K':torch.eye(3)[None],
            'camera_pose_vec':torch.ones(1,3)}


def bundle_for():
    actions,valid=expand_centers(native_actions(),[-20,0,20])
    actions[:,:,0]=torch.tensor([.15,.35,.6,.85])[None]
    return {'actions':actions,'valid':valid,'token_ids':torch.tensor([0,7,30,140]),
            'view_xyz':torch.tensor([[1.,0,0]]).expand(4,3),
            'angle_ids':torch.tensor([0,1,2,0]),'depth_ids':torch.tensor([0,1,0,1])}


def test_optional_latent_return_preserves_original_logits(monkeypatch):
    install_rotations(monkeypatch)
    m=CenterHypothesisCVA(mock_reference(),[-20,0,20]).eval()
    old,_,_=m(batch_for(),bundle_for(),query_chunk=2)
    new,_,_,latent=m(batch_for(),bundle_for(),query_chunk=2,return_features=True)
    assert torch.equal(old,new)
    assert latent.shape==(3,4,8) and latent.requires_grad


def test_zero_residual_is_bitwise_native_score_and_bound_is_respected():
    s=torch.tensor([0.,1e-7,.1,.3,.7,1.])
    assert torch.equal(anchored_score(s,torch.zeros_like(s)),s)
    assert torch.equal(anchored_score(s,torch.randn_like(s),0),s)
    r=torch.tensor([-.5,-.5,-.5,.5,.5,.5])
    out=anchored_score(s,r)
    assert torch.isfinite(out).all() and ((out>=0)&(out<=1)).all()
    torch.testing.assert_close(out[2:5],(anchor_logit(s)+r).sigmoid()[2:5])


def test_rank_features_and_cdf_are_detached_native_is_always_unchanged():
    h=torch.randn(3,4,8,requires_grad=True)
    l=torch.randn(3,4,6,requires_grad=True)
    s=torch.tensor([.1,.2,.4,.6],requires_grad=True)
    head=RankResidualHead(8)
    with torch.no_grad(): head.net[-1].weight.fill_(.03)
    residual=head(h,l,s,torch.tensor([-20.,0,20]),1)
    assert torch.equal(residual[1],torch.zeros(4))
    assert residual.abs().max()<=.5
    residual.sum().backward()
    assert h.grad is None and l.grad is None and s.grad is None
    assert sum(float(p.grad.abs().sum()) for p in head.parameters() if p.grad is not None)>0


def test_rank_loss_uses_cross_query_order_not_center_order():
    y=torch.tensor([0.,.5,1.])
    good,_=pairwise_ranking_loss(torch.tensor([-2.,0,2.]),y)
    bad,n=pairwise_ranking_loss(torch.tensor([2.,0,-2.]),y)
    assert n==3 and good<bad
    z=torch.ones(3,requires_grad=True)
    loss,n=pairwise_ranking_loss(z,torch.ones(3)); loss.backward()
    assert n==0 and loss==0 and torch.equal(z.grad,torch.zeros(3))
    z=torch.ones(1,requires_grad=True)
    loss,n=pairwise_ranking_loss(z,torch.ones(1)); loss.backward(); assert n==0


def test_same_selected_action_in_every_score_control():
    b=bundle_for()
    l=torch.full((3,4,6),-2.)
    l[0,0]=3.; l[2,1]=4.; l[1,2]=2.; l[0,3]=3.
    b['valid'][0,3]=False
    r=torch.tensor([[.4,.1,0.,.2],[0.,0.,0.,0.],[-.4,.2,.3,.1]])
    outputs,sel=make_outputs(l,r,b,1)
    for name in ('local','anchored'):
        assert torch.equal(outputs[name][:,1:],outputs['stage1'][:,1:])
    assert torch.equal(outputs['stage1'][:,0],b['actions'][1,:,0])
    assert sel[3]==1 and sel[2]==1
    assert outputs['anchored'][2,0]==outputs['native'][2,0]


def test_rank_backward_cannot_change_corrector_training_step(monkeypatch):
    install_rotations(monkeypatch); torch.manual_seed(11)
    m=DecoupledCenterRankingCVA(mock_reference(),[-20,0,20])
    other=copy.deepcopy(m)
    b=bundle_for(); batch=batch_for()
    y=torch.tensor([0.,1.,.5,1/6])[None,:,None].expand(3,4,6)
    for model,train_rank in ((m,True),(other,False)):
        l,r,_,_=model(batch,b,query_chunk=2)
        selected=torch.tensor([0,2,0,2])
        if train_rank:
            loss,_=selected_rank_loss(r,b['actions'][1,:,0],selected,y)
            loss.backward()
            assert all(p.grad is None for p in model.corrector.parameters())
        local,_=relative_cdf_loss(l,y,b['valid'],1,'E2')
        local.backward()
        cp=[p for p in model.corrector.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(cp,5.)
        torch.optim.SGD(cp,lr=.01).step()
        if train_rank: torch.optim.SGD(model.ranker.parameters(),lr=.01).step()
        assert all(p.grad is None for p in model.reference.parameters())
    for k,v in m.corrector.state_dict().items():
        torch.testing.assert_close(v,other.corrector.state_dict()[k],rtol=0,atol=0)


def test_model_state_roundtrip_and_warm_start(monkeypatch):
    install_rotations(monkeypatch)
    base=CenterHypothesisCVA(mock_reference(),[-20,0,20]).eval()
    dcr=DecoupledCenterRankingCVA(copy.deepcopy(base.reference),[-20,0,20]).eval()
    dcr.warm_start_corrector(base.learned_state())
    old=base(batch_for(),bundle_for())[0]; new,r,_,_=dcr(batch_for(),bundle_for())
    assert torch.equal(old,new) and torch.count_nonzero(r)==0
    clone=DecoupledCenterRankingCVA(copy.deepcopy(dcr.reference),[-20,0,20]).eval()
    clone.load_learned_state(dcr.learned_state())
    assert torch.equal(new,clone(batch_for(),bundle_for())[0])


def test_cli_help_does_not_import_legacy_model_parser():
    root=Path(__file__).resolve().parents[1]
    for name in ('train_dcr_cva.py','inference_dcr_cva.py','eval_dcr_cva.py','summarize_dcr_cva.py'):
        p=subprocess.run([sys.executable,str(root/name),'--help'],capture_output=True,text=True)
        assert p.returncode==0,(name,p.stderr)
        assert 'usage:' in p.stdout


def test_official_split_concurrency_matches_gpu_slots(tmp_path):
    root=Path(__file__).resolve().parents[1]
    events=tmp_path/'events.jsonl'; mock=tmp_path/'python_mock'
    mock.write_text('''#!/usr/bin/env python3
import json,os,sys,time
with open(os.environ['EVENTS'],'a') as f:
 f.write(json.dumps({'phase':'start','args':sys.argv[1:],'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')})+'\\n')
time.sleep(.3)
with open(os.environ['EVENTS'],'a') as f:
 f.write(json.dumps({'phase':'end','args':sys.argv[1:]})+'\\n')
'''); mock.chmod(0o755)
    env=dict(os.environ,PYTHON_BIN=str(mock),EVENTS=str(events),GPUS='0,3,7',PHASES='eval',
             LOSS_MODES='cdf',WORK_ROOT=str(tmp_path/'run'),SPLITS='test_seen,test_similar,test_novel')
    p=subprocess.run(['bash',str(root/'scripts/run_dcr_cva.sh')],env=env,capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr
    rows=[json.loads(x) for x in events.read_text().splitlines()]
    assert [r['phase'] for r in rows[:3]]==['start']*3
    starts=[r for r in rows if r['phase']=='start']
    assert {r['gpu'] for r in starts}=={'0','3','7'}
    assert {r['args'][r['args'].index('--split')+1] for r in starts}=={'test_seen','test_similar','test_novel'}


def test_mocked_cpu_train_resume_and_infer_pipeline(monkeypatch,tmp_path):
    """Real trainer/optimizer/checkpoint/cache I/O using an explicitly mocked backbone."""
    install_rotations(monkeypatch)
    import train_dcr_cva as train
    import inference_dcr_cva as infer
    import models.economicgrasp_cva_centers as centers
    cpfile=tmp_path/'stage1.pt'; cpfile.write_bytes(b'mock checkpoint, not a neural model')
    cache=tmp_path/'cache'; cache.mkdir()
    proto={'version':VERSION,'reference_sha256':file_sha(cpfile),'sample_interval':.1,
           'camera':'realsense','offsets_mm':[-20,0,20],'max_frames_per_split':1}
    (cache/'protocol.json').write_text(json.dumps(proto))
    sig=digest(proto); bundle=bundle_for()
    target=torch.tensor([0.,1.,.5,1/6])[None,:,None].expand(3,4,6).numpy()
    for split,sid in [('train',0),('test_seen',100)]:
        fr={k:np.stack([v.numpy()]*2) for k,v in bundle.items()}
        save_npz(cache/split/f'scene_{sid:04d}'/'ann_0000.npz',**fr,
                 cases=np.array(['nominal','bias:10']),case_seeds=np.array([1,2]),
                 target=np.stack([target,target]),nominal_depth=np.full((1,28,28),.5,np.float32),
                 scene_id=np.array(sid),anno_id=np.array(0),signature=np.array(sig),
                 action_sha=np.array(array_sha(fr['actions'][...,1:16])),
                 input_sha=np.array(batch_fingerprint(batch_for(sid,0))))
    frozen_ref=mock_reference()
    monkeypatch.setattr(centers,'load_reference',lambda *a:copy.deepcopy(frozen_ref))
    for m in (train,infer):
        monkeypatch.setattr(m,'make_dataset',lambda *a:(None,None))
        monkeypatch.setattr(m,'get_batch',lambda ds,lk,sid,aid,dev:batch_for(sid,aid))
    run=tmp_path/'train'
    argv=['train_dcr_cva.py','--dataset-root','mock','--stage1-checkpoint',str(cpfile),
          '--cache-root',str(cache),'--output-root',str(run),'--epochs','1',
          '--device','cpu','--grad-accum','1','--query-chunk','2','--resume']
    monkeypatch.setattr(sys,'argv',argv.copy()); train.main()
    assert (run/'checkpoint_best.pt').is_file()
    audit=json.loads((run/'gradient_check.json').read_text())
    assert audit['ranking_to_corrector']==0 and audit['reference']==0
    monkeypatch.setattr(sys,'argv',argv.copy()); train.main()  # resume completed epoch
    assert len(json.loads((run/'metrics.json').read_text()))==1
    def mock_candidates(reference,batch,pack,case,seed,offsets,query_limit):
        b=bundle_for(); b['native']=b['actions'][1].clone(); b['query_ids']=torch.arange(4)
        return b,pack[0]
    monkeypatch.setattr(centers,'reference_candidates',mock_candidates)
    output=tmp_path/'infer'
    argv=['inference_dcr_cva.py','--dataset-root','mock','--stage1-checkpoint',str(cpfile),
          '--checkpoint',str(run/'checkpoint_best.pt'),'--output-root',str(output),
          '--split','test_seen','--device','cpu','--max-frames','1','--resume']
    monkeypatch.setattr(sys,'argv',argv.copy()); infer.main()
    for case in ('nominal','bias_m20','bias_20'):
        loc=np.load(output/'dump'/'local'/case/'scene_0100'/'realsense'/'0000.npy')
        fix=np.load(output/'dump'/'stage1'/case/'scene_0100'/'realsense'/'0000.npy')
        rank=np.load(output/'dump'/'anchored'/case/'scene_0100'/'realsense'/'0000.npy')
        np.testing.assert_array_equal(loc[:,1:],fix[:,1:]); np.testing.assert_array_equal(rank[:,1:],fix[:,1:])
    monkeypatch.setattr(sys,'argv',argv.copy()); infer.main()
    assert json.loads((output/'infer_test_seen_shard0.json').read_text())['skipped']==1
