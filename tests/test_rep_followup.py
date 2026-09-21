"""CPU regression tests. Real Stage-1/CAD integration is enforced at runtime.

The fake Stage-1 below tests intervention routing and method restoration; it
is NOT a claim to have run the user's real checkpoint or GraspNet evaluator.
"""
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from rep_b0_controls import RepB0Control
from rep_b_model import RepBModel
from rep_fullpath_runtime import (FullpathReplay,expand_actions,parse_offsets,replace_method,routed_depths)
from rep_followup_common import decision_metrics,aggregate_decisions,write_csv
from evaluate_rep_fullpath import label_actions,physical_key
from infer_rep_fullpath import query_indices,score_queries


def data():
    a=torch.zeros(3,4,17); a[...,0]=.5; a[...,1]=.06; a[...,2]=.02; a[...,3]=.03
    a[...,4:13]=torch.eye(3).flatten(); a[:,:,15]=torch.tensor([.48,.5,.52])[:,None]
    a[:,:,13]=torch.linspace(-.01,.01,4)[None]
    return dict(actions=a,valid=torch.ones(3,4,dtype=torch.bool),zero_index=1,
                offsets_mm=torch.tensor([-20.,0,20.]),token_ids=torch.tensor([490,491,492,493]),
                image_feature=torch.randn(8,8,8),depth=torch.ones(1,32,32)*.5,
                K=torch.tensor([[60.,0,15.5],[0,60.,15.5],[0,0,1.]]),
                objectness=torch.ones(2,32,32),graspness=torch.ones(1,32,32))


def model(control):
    torch.manual_seed(17)
    return RepB0Control(control=control,channels=8,dim=32,heads=4,layers=1,dropout=0.).eval()


def test_full_reproduces_actual_b0_exactly():
    torch.manual_seed(17)
    original=RepBModel(channels=8,dim=32,heads=4,layers=1,dropout=0.,variant='B0').eval()
    full=model('full'); d=data()
    assert original.state_dict().keys()==full.state_dict().keys()
    assert all(torch.equal(v,full.state_dict()[k]) for k,v in original.state_dict().items())
    with torch.no_grad(): assert torch.equal(original(d),full(d))


@pytest.mark.parametrize('control',['full','no_image','independent_rgb'])
def test_parameter_counts_initialization_and_gradients(control):
    base=model('full'); m=model(control).train(); d=data()
    assert sum(p.numel() for p in m.parameters())==sum(p.numel() for p in base.parameters())
    assert all(torch.equal(v,base.state_dict()[k]) for k,v in m.state_dict().items())
    out=m(d); nn.functional.binary_cross_entropy_with_logits(out,torch.rand_like(out)).backward()
    for name,part in m.named_children():
        vals=[p.grad for p in part.parameters() if p.grad is not None]
        assert vals and all(torch.isfinite(x).all() for x in vals)
        assert sum(float(x.abs().sum()) for x in vals)>0,name


def test_no_image_ignores_content_but_not_action():
    d=data(); m=model('no_image')
    with torch.no_grad():
        a=m(d); b=m(dict(d,image_feature=torch.randn_like(d['image_feature'])*100))
        assert torch.equal(a,b)
        changed=copy.deepcopy(d); changed['actions'][0,:,1]*=1.5
        assert not torch.allclose(a,m(changed))


def test_independent_has_no_cross_candidate_influence():
    d=data(); x=copy.deepcopy(d); x['actions'][2,:,1]*=2
    x['offsets_mm'][2]=80
    with torch.no_grad():
        m=model('independent_rgb'); a=m(d); b=m(x)
        assert torch.allclose(a[:2],b[:2],atol=1e-7,rtol=0)
        m=model('full'); a=m(d); b=m(x)
        assert not torch.allclose(a[:2],b[:2])


@pytest.mark.parametrize('control',['full','no_image','independent_rgb'])
def test_reader_depth_invariant_chunking_and_invalid(control):
    d=data(); d['valid'][0,0]=False; d['actions'][0,0]=float('nan')
    m=model(control)
    with torch.no_grad():
        out=m(d); other=m(d,depth=d['depth']+.04)
        assert torch.isfinite(out).all() and torch.equal(out,other)
        chunk=score_queries(m,d,2)
        assert np.allclose(chunk,out.sigmoid().numpy(),atol=1e-6)


class FakeDepth(nn.Module):
    def __init__(self): super().__init__(); self.calls=0
    def forward(self,img):
        self.calls+=1
        d=img[:,0:1]*0+.5
        return d,None,None,None,[img],{}


class FakeProposal(nn.Module):
    def __init__(self): super().__init__(); self.calls=0
    def forward(self,feats):
        self.calls+=1
        img=feats[0]; logits=torch.zeros(img.shape[0],3,*img.shape[-2:])
        logits[:,1:]=1
        return img,logits


class FakeStage(nn.Module):
    def __init__(self):
        super().__init__(); self.depth_net=FakeDepth(); self.proposal_head=FakeProposal()
        self.is_training=False; self.geometry_depth_source='pred'; self.use_obs_depth=False
        self.seed_selection_mode='image_fps'; self.min_depth=.2; self.max_depth=1.
        self.masks=[]
    def _select_graspable_seed_queries(self,feat_grid,depth_map,camera_K,graspable_mask,valid_tok,grasp_score,end_points):
        ids=torch.arange(4)[None]
        z=depth_map[:,0].flatten(1).gather(1,ids).clamp(self.min_depth,self.max_depth)
        xyz=torch.zeros(1,4,3); xyz[...,2]=z
        self.masks.append(valid_tok.clone())
        return feat_grid,xyz,ids,None,None,4.
    def forward(self,batch):
        depth,*tail=self.depth_net(batch['img'])
        feat,logits=self.proposal_head(tail[-2]); h,w=depth.shape[-2:]
        batch['depth_map_used_for_geometry']=depth
        batch['objectness_score']=logits[:,:2].reshape(1,2,-1)
        valid=(depth>.2).reshape(1,-1)&(depth<1).reshape(1,-1)
        _,xyz,ids,*_=self._select_graspable_seed_queries(feat_grid=feat,depth_map=depth,
            camera_K=batch['K'],graspable_mask=valid,valid_tok=valid,grasp_score=logits[:,2].reshape(1,-1),end_points=batch)
        batch['xyz']=xyz; batch['token_ids']=ids
        # Stand-in for depth-conditioned ViewNet/head outputs.
        batch['head_output']=depth.mean()+xyz.mean()
        return batch


def test_replay_routes_both_paths_and_bypasses_backbone_only():
    m=FakeStage().eval(); r=FullpathReplay(m,.1)
    batch=dict(img=torch.rand(1,3,8,8),K=torch.eye(3)[None])
    native=r.capture(batch); d0=r.nominal_depth
    check=r.run(batch,d0,d0)
    assert torch.equal(native['xyz'],check['xyz'])
    assert m.depth_net.calls==1 and m.proposal_head.calls==1
    for mode in ('reader_only','anchor_only','joint'):
        a,b=routed_depths(d0,d0+.04,mode); out=r.run(batch,a,b)
        assert torch.allclose(out['xyz'][...,2],a[0,0,0,0].expand(1,4))
        assert torch.equal(out['depth_map_used_for_geometry'],b)
        assert 'forward' not in m.depth_net.__dict__ and '_select_graspable_seed_queries' not in m.__dict__
    assert m.depth_net.calls==1
    m(batch); assert m.depth_net.calls==2


def test_methods_restored_on_exception():
    m=FakeDepth()
    with pytest.raises(RuntimeError):
        with replace_method(m,'forward',lambda *a:None): raise RuntimeError('injected')
    assert 'forward' not in m.__dict__


def test_anchor_validity_not_reader_validity():
    m=FakeStage().eval(); r=FullpathReplay(m,.1)
    b=dict(img=torch.zeros(1,3,8,8),K=torch.eye(3)[None]); r.capture(b)
    r.run(b,r.nominal_depth,torch.ones_like(r.nominal_depth)*1.5)
    assert m.masks[-1].all()
    r.run(b,torch.ones_like(r.nominal_depth)*1.5,r.nominal_depth)
    assert not m.masks[-1].any()


def test_new_actions_and_physical_fingerprint():
    g=data()['actions'][1].numpy(); shifted=g.copy(); shifted[:,13:16]*=1.04
    a,v,z=expand_actions(shifted,parse_offsets('-20,0,20'))
    assert np.array_equal(a[z],shifted) and not np.array_equal(a[z],g)
    assert np.array_equal(a[0,:,1:13],g[:,1:13])
    assert physical_key(a[z,0])!=physical_key(g[0])
    other=g[0].copy(); other[0]=.99; other[16]=12
    assert physical_key(g[0])==physical_key(other)


def test_exact_labels_recomputed_and_deduplicated_not_old_cache():
    class Evaluator:
        def __init__(self): self.count=0
        def evaluate(self,sid,aid,g):
            self.count+=len(g)
            # Label changes with actual center, not the old cache's labels.
            f=np.where(g[:,15]>.505,.2,-1).astype(np.float32)
            return SimpleNamespace(friction=f,collision_or_empty=f<0,pure_collision=f<0,empty=f<0)
    a=data()['actions'].numpy(); v=np.ones(a.shape[:2],bool); e=Evaluator(); memo={}
    f,*_=label_actions(e,100,0,a,v,memo,chunk=3); first=e.count
    assert np.all(f[2]>.0) and np.all(f[1]<0)
    b=a.copy(); b[:,:,0]=.1
    label_actions(e,100,0,b,v,memo,chunk=3); assert e.count==first
    b[:,:,15]+=.01
    f2,*_=label_actions(e,100,0,b,v,memo,chunk=3)
    assert e.count>first and np.all(f2[1]>0)


def test_decision_metrics_optimal_ties_and_native_fallback():
    fr=np.array([[.2,-1],[.4,.2],[.2,.4]],np.float32)
    from rep_a_common import cdf_targets
    d=dict(valid=np.ones((3,2),bool),friction=fr,utility=cdf_targets(fr).mean(-1),zero_index=1,offsets_mm=np.array([-20,0,20]))
    prob=cdf_targets(fr)*.98+.01
    r,sel=decision_metrics(prob,d,0)
    assert r['top1_optimal_count']==2 and r['native_best_count']==2
    assert r['forced_top1_regret_sum']==0
    out=aggregate_decisions([r]); assert out['top1_optimal_recall']==1
    _,sel=decision_metrics(prob,d,1); assert np.array_equal(sel,np.array([1,1]))


def test_heterogeneous_csv_and_queries(tmp_path):
    p=tmp_path/'table.csv'; write_csv(p,[dict(a=1),dict(a=2,b=3)])
    assert p.read_text().splitlines()==['a,b','1,','2,3']
    a=np.repeat(data()['actions'][1].numpy(),16,axis=0)
    for n in (0,16,64):
        ids=query_indices(a,n); assert len(np.unique(ids))==len(ids)


@pytest.mark.parametrize('label_scope',['selected','all'])
def test_fullpath_eval_resume_and_summary_without_old_labels(tmp_path,monkeypatch,label_scope):
    import json,sys,types
    import evaluate_rep_fullpath as ev
    import summarize_rep_followup as su
    from rep_a_common import choose,digest,save_npz,save_json
    root=tmp_path/'run';root.mkdir()
    protocol={'camera':'realsense'};save_json(root/'protocol.json',protocol)
    a=data()['actions'].numpy();v=np.ones(a.shape[:2],bool)
    prob=np.ones((*v.shape,6),np.float32)*.2;prob[2]=.8
    sel=choose(prob.mean(-1),v,1,0)
    payload=dict(signature=np.array(digest(protocol)),scene_id=np.array(100),anno_id=np.array(0),
        mode=np.array('joint'),case=np.array('nominal'),actions=a,valid=v,zero_index=np.array(1),
        offsets_mm=np.array([-20,0,20]),selected=np.stack([np.ones(4,int),sel]),
        output_methods=np.array(['stage1_native','B0']),output_policies=np.array(['native','fixed_0']),
        output_margins=np.array([0.,0.]),rank_scores=np.stack([np.arange(4),np.ones(4)]),
        scorer_names=np.array(['B0']),probabilities=prob[None],depth_rms_mm=np.array(0.),
        anchor_depth_rms_mm=np.array(0.),reader_depth_rms_mm=np.array(0.))
    path=root/'inference'/'test_seen'/'scene_0100'/'ann_0000_joint_nominal.npz'
    save_npz(path,payload)
    calls=[]
    class FakeEval:
        def __init__(self,*a,**kw):self.scene_cache={}
        def evaluate(self,sid,aid,g):
            self.scene_cache[sid]=True;calls.append(len(g))
            f=np.where(g[:,15]>.51,.2,-1).astype(np.float32)
            return SimpleNamespace(friction=f,collision_or_empty=f<0,pure_collision=f<0,empty=f<0)
    fake=types.ModuleType('exact_action_graspnet_evaluator');fake.ExactGraspNetActionEvaluator=FakeEval
    monkeypatch.setitem(sys.modules,'exact_action_graspnet_evaluator',fake)
    monkeypatch.setattr(ev,'source_digest',lambda *_:'test-only-mocked-evaluator')
    argv=['evaluate_rep_fullpath.py','--dataset-root','not-used-by-fixture','--work-root',str(root),
          '--split','test_seen','--resume','--min-host-free-gib','0','--label-scope',label_scope]
    monkeypatch.setattr(sys,'argv',argv);ev.main();assert sum(calls)>0
    count=sum(calls)
    monkeypatch.setattr(sys,'argv',argv);ev.main();assert sum(calls)==count
    monkeypatch.setattr(sys,'argv',['summarize_rep_followup.py','--kind','fullpath','--root',str(root),'--splits','test_seen'])
    su.main()
    result=json.loads((root/'comparison.json').read_text())
    learned=next(r for r in result if r['method']=='B0')
    assert learned['success08']==1 and learned['native_success08']==0
    assert learned['utility_drop_from_nominal']==0
    if label_scope=='selected': assert learned['oracle_utility'] is None


def test_official_ap_rejects_query_subset_before_import(tmp_path,monkeypatch):
    import sys
    import eval_rep_fullpath_official as ev
    from rep_a_common import save_json
    save_json(tmp_path/'protocol.json',{'query_limit':64})
    monkeypatch.setattr(sys,'argv',['x','--dataset-root','unused','--work-root',str(tmp_path),'--split','test_seen'])
    with pytest.raises(RuntimeError,match='QUERY_LIMIT=0'):ev.main()
