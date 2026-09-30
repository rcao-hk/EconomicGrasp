"""CPU contracts. Full dataset/CUDA integration must run in the robot environment."""
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mgf_p0_core import (Metrics,assert_frozen,cdf_targets,comparison_metrics,exact_targets,
                         fingerprint,freeze,perturb_actions,score_loss)
from mgf_p0_online import make_control
from metric_grasp_field_core import MetricFieldConfig,TaskFeatureAdapter,GraspFieldReadout


def actions(n):
    a=torch.zeros(1,n,17); a[...,0]=1.; a[...,1:4]=torch.tensor([.06,.02,.02])
    a[...,4:13]=torch.eye(3).flatten(); a[...,15]=.5; a[...,16]=-1
    return a


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared=nn.Conv1d(8,8,1)
        self.cdf_head=nn.Conv1d(8,12,1)
        self.width_head=nn.Conv1d(8,2,1)
    def forward(self,x,e):
        x=self.shared(x)
        q,a=e['kview_angle_query_base_q'],e['kview_angle_query_num_angle']
        raw=self.cdf_head(x).reshape(x.shape[0],2,6,q,a)
        y=torch.cat((raw[:,:,:1],raw[:,:,:1]+F.softplus(raw[:,:,1:]-4).cumsum(2)),2)
        e['grasp_cdf_pred_angle_depth']=y.permute(0,2,3,4,1).contiguous()
        e['grasp_width_pred_angle_depth']=self.width_head(x).reshape(x.shape[0],2,q,a)
        return e


def setup():
    torch.manual_seed(4)
    cfg=MetricFieldConfig(bins=8,hidden=8,action_chunk=12,checkpoint_chunks=False)
    source=nn.Module(); source.model=nn.Module(); source.config=cfg
    source.model.task_adapter=TaskFeatureAdapter(8,8,8,8)
    source.model.readout=GraspFieldReadout(cfg)
    source.model.base=nn.Module(); source.model.base.kview_grasp_module=nn.Module()
    source.model.base.kview_grasp_module.decoder=Decoder()
    freeze(source.model)
    grouped=torch.randn(1,8,6)
    ep=source.model.base.kview_grasp_module.decoder(grouped,dict(kview_angle_query_base_q=2,kview_angle_query_num_angle=3))
    c=dict(ep=ep,grouped=grouped,proposal=torch.randn(1,8,16,16),relative=torch.randn(1,8,16,16),
           metric=torch.randn(1,8,16,16),prob=torch.ones(1,8,16,16)/8,
           K=torch.tensor([[[80.,0,31.5],[0,80,31.5],[0,0,1.]]]),hw=(64,64),
           actions=actions(12),shape=(1,2,3,2))
    return source,c


@pytest.mark.parametrize('variant',['base','feature_only','full','cva'])
def test_initial_control_matches_same_base(variant):
    source,c=setup(); m=make_control(source,variant).eval()
    torch.testing.assert_close(m(c),c['ep']['grasp_cdf_pred_angle_depth'],atol=2e-6,rtol=2e-6)


def test_feature_and_full_have_identical_weights_and_parameter_count():
    source,c=setup(); a=make_control(source,'feature_only'); b=make_control(source,'full')
    assert fingerprint(a)==fingerprint(b)
    assert sum(x.numel() for x in a.parameters())==sum(x.numel() for x in b.parameters())


@pytest.mark.parametrize('variant',['feature_only','full','cva'])
def test_optimizer_cannot_change_source_or_executed_width(variant):
    source,c=setup(); m=make_control(source,variant); m.train()
    h=fingerprint(source.model); widths=c['ep']['grasp_width_pred_angle_depth'].clone()
    b=torch.tensor([[[[0,1],[2,0],[0,3]],[[0,1],[0,0],[4,0]]]])
    valid=torch.ones_like(b,dtype=torch.bool)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-3)
    for _ in range(2):
        opt.zero_grad(); logits=m(c); loss,_=score_loss(logits,b,valid,distributed=False)
        loss.backward(); opt.step(); assert_frozen(source.model)
    assert fingerprint(source.model)==h
    torch.testing.assert_close(widths,c['ep']['grasp_width_pred_angle_depth'],atol=0,rtol=0)
    assert all(p.grad is None for p in source.parameters())


def test_feature_only_ignores_profile_but_full_can_use_it():
    source,c=setup(); full=make_control(source,'full'); feat=make_control(source,'feature_only')
    torch.manual_seed(11)
    with torch.no_grad(): full.reader.head[-1].weight.normal_(0,.2)
    feat.load_state_dict(full.state_dict())
    c2=dict(c); c2['prob']=torch.zeros_like(c['prob']); c2['prob'][:,-1]=1
    torch.testing.assert_close(feat(c),feat(c2),atol=0,rtol=0)
    assert (full(c)-full(c2)).abs().max()>1e-5


def test_geometry_inputs_do_not_receive_task_gradients():
    source,c=setup(); m=make_control(source,'full')
    for k in ('relative','metric','prob','K'): c[k]=c[k].detach().requires_grad_()
    with torch.no_grad(): m.reader.head[-1].weight.normal_(0,.1)
    m(c).sum().backward()
    assert all(c[k].grad is None for k in ('relative','metric','prob','K'))


def test_empty_labels_keep_zero_graph_and_finite_gradients():
    logits=torch.randn(1,6,2,3,2,requires_grad=True)
    bins=torch.zeros(1,2,3,2,dtype=torch.long); valid=torch.zeros_like(bins,dtype=torch.bool)
    loss,_=score_loss(logits,bins,valid,distributed=False)
    assert float(loss.detach())==0; loss.backward()
    torch.testing.assert_close(logits.grad,torch.zeros_like(logits))


def test_constant_queries_have_no_ranking_gradient():
    logits=torch.randn(1,6,2,3,2,requires_grad=True)
    bins=torch.zeros(1,2,3,2,dtype=torch.long); valid=torch.ones_like(bins,dtype=torch.bool)
    _,s=score_loss(logits,bins,valid,distributed=False)
    assert float(s['ranking'])==0


def test_pooled_metrics_ignore_batch_partition_and_flag_one_class():
    bins=torch.tensor([[[[0,1],[0,3],[2,0]]],[[[6,0],[0,0],[1,4]]]])
    valid=torch.ones_like(bins,dtype=torch.bool)
    logits=torch.randn(2,6,1,3,2)
    a=Metrics(); b=Metrics(); a.update(logits,bins,valid)
    for i in range(2): b.update(logits[i:i+1],bins[i:i+1],valid[i:i+1])
    np.testing.assert_allclose(a.values,b.values,atol=1e-5,rtol=1e-5)
    c=Metrics(); c.update(logits,torch.zeros_like(bins),valid)
    assert c.report()['auroc64'] is None
    assert not c.report()['ranking_metric_defined']


def test_ray_shift_preserves_projection_and_other_pose_parameters():
    a=actions(3); a[...,13]=.07; a[...,14]=-.03
    b,m=perturb_actions(a,'ray_z',.02)
    torch.testing.assert_close(a[...,13:15]/a[...,15,None],b[...,13:15]/b[...,15,None])
    torch.testing.assert_close(b[...,:13],a[...,:13]); assert m.all()


def test_roll_is_about_local_approach_axis():
    a=actions(2); b,_=perturb_actions(a,'roll',math.pi/12)
    r=b[...,4:13].reshape(1,2,3,3)
    torch.testing.assert_close(r[..., :,0],torch.tensor([1.,0,0]).expand(1,2,3))
    torch.testing.assert_close(r@r.transpose(-1,-2),torch.eye(3).expand(1,2,3,3),atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(b[...,13:16],a[...,13:16])


def test_width_invalid_not_clamped_or_fake_negative():
    a=actions(1); b,valid=perturb_actions(a,'width',.08)
    assert not valid.any(); assert float(b[0,0,1])==pytest.approx(.14)


def test_exact_labels_preserve_order_and_mask_collisions():
    r=SimpleNamespace(friction=np.array([.4,-1,.2]),collision_or_empty=np.array([False,False,True]))
    y=exact_targets(r,3)
    np.testing.assert_array_equal(y,[[0,1,1,1,1,1],[0,0,0,0,0,0],[0,0,0,0,0,0]])


def test_audit_selection_uses_exact_candidate_targets():
    y=np.array([[0.,1.,.5]])
    r=comparison_metrics(np.array([[.1,.8,.2]]),y,np.ones_like(y,bool))
    assert r['selection_regret']==0; assert r['top1_best_hit']==1


@pytest.mark.parametrize('script',['mgf_p0_train.py','mgf_p0_infer.py','mgf_p0_audit.py','mgf_p0_compare.py'])
def test_cli_help_without_cuda_imports(script):
    p=subprocess.run([sys.executable,str(ROOT/script),'--help'],capture_output=True,text=True)
    assert p.returncode==0,p.stderr


def test_bash_syntax():
    for p in (ROOT/'scripts').glob('run_mgf_p0_*.sh'):
        subprocess.run(['bash','-n',str(p)],check=True)


def _ddp_loss_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    torch.distributed.init_process_group('gloo', init_method='file://'+rendezvous, rank=rank, world_size=2)
    # Rank 0 has no valid candidates; rank 1 carries all supervision.
    p=torch.tensor(0.,requires_grad=True)
    logits=p.expand(1,6,1,1,2)
    bins=torch.tensor([[[[2,0]]]])
    mask=torch.full_like(bins,rank==1,dtype=torch.bool)
    loss,_=score_loss(logits,bins,mask)
    loss.backward()
    g=p.grad.clone(); torch.distributed.all_reduce(g); g/=2
    if rank==0: Path(output).write_text(str(float(g)))
    torch.distributed.destroy_process_group()


def test_ddp_masked_mean_matches_unsharded_loss(tmp_path):
    if not torch.distributed.is_available(): pytest.skip('Distributed unavailable')
    import torch.multiprocessing as mp
    out=tmp_path/'gradient.txt'
    mp.spawn(_ddp_loss_worker,args=(str(tmp_path/'rdzv'),str(out)),nprocs=2,join=True)
    p=torch.tensor(0.,requires_grad=True)
    loss,_=score_loss(p.expand(1,6,1,1,2),torch.tensor([[[[2,0]]]]),
                      torch.ones(1,1,1,2,dtype=torch.bool),distributed=False)
    loss.backward()
    assert float(out.read_text())==pytest.approx(float(p.grad),abs=1e-7)
