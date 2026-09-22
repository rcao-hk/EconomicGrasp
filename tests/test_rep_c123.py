import numpy as np
import torch

from rep_a_common import cdf_targets
from rep_c1_oracle_acceptance import oracle_accept
from rep_c2_model import (
    EVIDENCE, FULL_DIM, SCORE_DIM, LOCAL_DIM, RELIABILITY_DIM,
    RepC2Verifier, apply_evidence, build_pair_evidence,
)
from infer_rep_c3 import residual_stress


def _friction_for_utility(u):
    # Utilities represent mean success over thresholds [.2,.4,.6,.8,1.,1.2].
    table = {
        0.0: -1.0, 1/6: 1.2, 2/6: 1.0, 3/6: .8,
        4/6: .6, 5/6: .4, 1.0: .2,
    }
    return table[round(float(u), 7)]


def test_c1_oracle_rejects_harm_and_accepts_benefit():
    k,q=3,4
    actions=np.zeros((k,q,17),np.float32)
    actions[...,15]=.5
    valid=np.ones((k,q),bool)
    # zero=1; proposal uses k=2 for all queries.
    utility=np.array([
        [.0,.0,.0,.0],
        [2/6,3/6,4/6,5/6],
        [3/6,2/6,4/6,1.0],
    ],np.float32)
    friction=np.vectorize(_friction_for_utility)(utility).astype(np.float32)
    probs=np.zeros((1,k,q,6),np.float32)
    probs[0,1]=.2; probs[0,2]=.8
    payload={
        "actions":actions,"valid":valid,"zero_index":np.array(1),
        "output_methods":np.array(["stage1_native","A1"]),
        "output_policies":np.array(["native","fixed_0"]),
        "selected":np.array([[1,1,1,1],[2,2,2,2]],np.int64),
        "original_native_score":np.array([.9,.8,.7,.6],np.float32),
        "scorer_names":np.array(["A1"]),"probabilities":probs,
        "rank_scores":np.array([[.9,.8,.7,.6],[.8,.8,.8,.8]],np.float32),
    }
    labels={"friction":friction,"evaluated_mask":valid}
    out=oracle_accept(payload,labels,policy="fixed_0")
    # q0 improves, q1 harms, q2 ties, q3 improves.
    np.testing.assert_array_equal(out["accepted"],[True,False,False,True])
    np.testing.assert_array_equal(out["selected"],[2,1,1,2])
    np.testing.assert_allclose(out["stage1_score"],payload["original_native_score"])
    # Rejected queries use A1 zero score, accepted queries selected score.
    np.testing.assert_allclose(out["a1_score"],[.8,.2,.2,.8],atol=1e-6)


def _c2_data():
    k,q=3,5
    actions=torch.zeros(k,q,17)
    actions[...,1]=.06; actions[...,2]=.02; actions[...,3]=.03
    actions[...,4]=1; actions[...,8]=1; actions[...,12]=1
    actions[...,15]=.5
    # Same ray, three z offsets.
    actions[0,...,15]=.48; actions[1,...,15]=.5; actions[2,...,15]=.52
    valid=torch.ones(k,q,dtype=torch.bool)
    return {
        "actions":actions,"valid":valid,"zero_index":1,
        "offsets_mm":torch.tensor([-20.,0.,20.]),
        "K":torch.tensor([[400.,0.,224.],[0.,400.,224.],[0.,0.,1.]]),
        "native_score":torch.linspace(.2,.8,q),
    }


def test_c2_evidence_has_paired_fixed_capacity_blocks():
    data=_c2_data(); q=5
    depth=torch.full((1,448,448),.5)
    prob=torch.rand(3,q,6)
    proposal=torch.tensor([0,2,0,2,2])
    x=build_pair_evidence(data,depth,prob,proposal)
    assert x.shape==(q,FULL_DIM)
    assert FULL_DIM==SCORE_DIM+LOCAL_DIM+RELIABILITY_DIM
    assert torch.isfinite(x).all()
    counts=[]
    for kind in EVIDENCE:
        xm=apply_evidence(x,kind)
        assert xm.shape==x.shape
        m=RepC2Verifier()
        y=m(xm)
        assert y.shape==(q,)
        counts.append(sum(p.numel() for p in m.parameters()))
    assert len(set(counts))==1
    # Score evidence cannot see local/reliability blocks.
    xs=apply_evidence(x,"score")
    assert torch.count_nonzero(xs[:,SCORE_DIM:])==0
    xl=apply_evidence(x,"local")
    assert torch.count_nonzero(xl[:,SCORE_DIM+LOCAL_DIM:])==0


def test_c3_residual_stress_full_and_local():
    gt=torch.ones(1,1,4,4)
    pred=gt.clone()
    pred[...,0:2,:]+=.02
    pred[...,2:,:]-=.01
    full,meta=residual_stress(pred,gt,"residual_full:1.0")
    np.testing.assert_allclose(
        (full-gt).numpy(), (2*(pred-gt)).numpy(), atol=1e-7
    )
    assert meta["stress_kind"]=="residual_full"
    local,lmeta=residual_stress(pred,gt,"residual_local:1.0")
    # Local stress removes the residual median before amplification.
    resid=(pred-gt)
    med=resid[torch.isfinite(resid)].median()
    expected=pred+(resid-med)
    np.testing.assert_allclose(local.numpy(),expected.numpy(),atol=1e-7)
    assert abs(lmeta["removed_center_mm"]-float(med)*1000)<1e-5
