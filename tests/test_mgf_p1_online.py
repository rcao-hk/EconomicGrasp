"""CPU contracts for P1 online controls. Full CUDA integration runs on robot host."""
from __future__ import annotations
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from mgf_p0_core import freeze
from mgf_p1_online import (
    P11_VARIANTS,P13_VARIANTS,make_control,prepare_context,SupportScorer,
)
from mgf_p1_label_audit import _aggregate
from metric_grasp_field_core import (
    MetricFieldConfig,MetricRayEvidenceHead,TaskFeatureAdapter,GraspFieldReadout,
    monotone_grasp_logits,
)


def _actions(q=2,a=3,d=2):
    n=q*a*d
    x=torch.zeros(1,n,17)
    x[...,0]=1.; x[...,1:4]=torch.tensor([.06,.02,.02])
    x[...,4:13]=torch.eye(3).reshape(9)
    x[...,15]=.5; x[...,16]=-1
    return x


def _source_and_context():
    torch.manual_seed(7)
    cfg=MetricFieldConfig(bins=8,hidden=8,field_stride=4,action_chunk=4,checkpoint_chunks=False)
    source=nn.Module(); source.config=cfg; source.model=nn.Module()
    source.model.task_adapter=TaskFeatureAdapter(8,8,8,8)
    source.model.readout=GraspFieldReadout(cfg)
    source.model.ray_head=MetricRayEvidenceHead(8,8,cfg)
    # Make the mock ray profile actually depend on relative inputs.
    with torch.no_grad():
        source.model.ray_head.net[-1].weight.normal_(0,.05)
        source.model.ray_head.net[-1].bias.normal_(0,.02)
    freeze(source.model)
    q,a,d=2,3,2
    raw=torch.randn(1,q,a,d,6)
    base=monotone_grasp_logits(raw).movedim(-1,1).contiguous()
    ctx=dict(
        ep={'grasp_cdf_pred_angle_depth':base},
        proposal=torch.randn(1,8,16,16),
        relative=torch.randn(1,8,16,16),
        relative_inverse=torch.randn(1,1,64,64).abs(),
        metric=torch.randn(1,8,16,16),
        depth=torch.full((1,1,64,64),.5),
        prob=torch.softmax(torch.randn(1,8,16,16),1),
        K=torch.tensor([[[80.,0.,31.5],[0.,80.,31.5],[0.,0.,1.]]]),
        hw=(64,64),actions=_actions(q,a,d),shape=(1,q,a,d),
    )
    return source,ctx


@pytest.mark.parametrize('variant',P11_VARIANTS)
def test_p11_zero_init_reconstructs_base(variant):
    source,ctx=_source_and_context()
    prepared=prepare_context(source,ctx,'p1_1',variant)
    model=make_control(source,'p1_1',variant).eval()
    torch.testing.assert_close(
        model(prepared),ctx['ep']['grasp_cdf_pred_angle_depth'],
        atol=3e-6,rtol=3e-6)


@pytest.mark.parametrize('variant',P13_VARIANTS)
def test_p13_zero_init_reconstructs_base(variant):
    source,ctx=_source_and_context()
    model=make_control(source,'p1_3',variant).eval()
    torch.testing.assert_close(
        model(ctx),ctx['ep']['grasp_cdf_pred_angle_depth'],
        atol=3e-6,rtol=3e-6)


def test_profile_modes_share_same_profile_mean_input():
    source,ctx=_source_and_context()
    prepared=[prepare_context(source,ctx,'p1_1',v) for v in
              ('profile_hard','profile_fixed','profile_learned')]
    for item in prepared[1:]:
        torch.testing.assert_close(item['p1_prob'],prepared[0]['p1_prob'],atol=0,rtol=0)
        torch.testing.assert_close(item['p1_relative'],prepared[0]['p1_relative'],atol=0,rtol=0)
    assert [x['p1_evidence_mode'] for x in prepared]==['hard','fixed','learned']


def test_relative_input_ablation_contracts():
    source,ctx=_source_and_context()
    metric=prepare_context(source,ctx,'p1_1','relative_metric_only')
    scalar=prepare_context(source,ctx,'p1_1','relative_scalar')
    feat=prepare_context(source,ctx,'p1_1','relative_feature')
    assert torch.count_nonzero(metric['p1_relative'])==0
    assert torch.count_nonzero(scalar['p1_relative'])==0
    assert torch.count_nonzero(feat['p1_relative'])>0
    # Mock ray head was made input-sensitive: scalar/feature interventions should
    # not all collapse to byte-identical learned profiles.
    assert not torch.equal(metric['p1_prob'],scalar['p1_prob'])
    assert not torch.equal(metric['p1_prob'],feat['p1_prob'])


def test_evidence_gate_is_bounded_and_drops_for_invalid_support():
    source,ctx=_source_and_context()
    scorer=SupportScorer(source,'learned').eval()
    feature=torch.randn(1,8,16,16)
    _,gate=scorer(feature,ctx['prob'],ctx['actions'],ctx['K'],ctx['hw'])
    assert bool(((gate>=0)&(gate<=1)).all())
    # Move actions outside supported camera-Z; every projected support is invalid.
    bad=ctx['actions'].clone(); bad[...,15]=1.5
    _,g2=scorer(feature,ctx['prob'],bad,ctx['K'],ctx['hw'])
    torch.testing.assert_close(g2,torch.zeros_like(g2),atol=0,rtol=0)


def test_shared_shift_preserves_base_threshold_increments():
    source,ctx=_source_and_context()
    model=make_control(source,'p1_3','shared_shift').eval()
    with torch.no_grad():
        model.out.bias.fill_(.25)
    final=model(ctx).movedim(1,-1)
    base=ctx['ep']['grasp_cdf_pred_angle_depth'].movedim(1,-1)
    delta=final-base
    torch.testing.assert_close(delta,delta[...,:1].expand_as(delta),atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(
        final[...,1:]-final[...,:-1],
        base[...,1:]-base[...,:-1],
        atol=1e-6,rtol=1e-6)


def test_p13_trainable_parameterizations_are_distinct():
    source,_=_source_and_context()
    counts={v:sum(p.numel() for p in make_control(source,'p1_3',v).parameters() if p.requires_grad)
            for v in P13_VARIANTS}
    assert counts['base_conditioned']>counts['residual6']
    assert counts['shared_shift']<counts['residual6']


def test_label_audit_aggregation_strata():
    rows=[
        dict(threshold_agreement=1.,transfer_utility=1.,exact_utility=1.,
             transfer_any_success=True,exact_any_success=True,
             center_bin='0-1',width_bin='0-2',collision_group='clear'),
        dict(threshold_agreement=.5,transfer_utility=1.,exact_utility=0.,
             transfer_any_success=True,exact_any_success=False,
             center_bin='4-5',width_bin='10-20',collision_group='pure_collision'),
    ]
    out=_aggregate(rows)
    assert out['overall']['n']==2
    assert out['overall']['threshold_agreement']==pytest.approx(.75)
    assert out['overall']['false_positive_fraction']==pytest.approx(.5)
    assert out['center/4-5']['n']==1


@pytest.mark.parametrize('script',[
    'mgf_p1_train.py','mgf_p1_infer.py','mgf_p1_label_audit.py','mgf_p1_compare.py'])
def test_cli_help_without_cuda(script):
    p=subprocess.run([sys.executable,str(ROOT/script),'--help'],capture_output=True,text=True)
    assert p.returncode==0,p.stderr


def test_p1_bash_syntax():
    for path in (ROOT/'scripts').glob('run_mgf_p1_*.sh'):
        subprocess.run(['bash','-n',str(path)],check=True)
