import numpy as np
import torch

from rep_c2v2_common import (
    BENEFICIAL, EQUIVALENT, HARMFUL, class_from_delta,
    compact_from_source, full_query_gate_metrics, select_move_subset,
)
from rep_c2v2_model import RepC2V2Verifier


def _payload_and_labels():
    k,q=3,4
    actions=np.zeros((k,q,17),np.float32)
    actions[...,1]=.06; actions[...,2]=.02; actions[...,3]=.03
    actions[...,4]=1; actions[...,8]=1; actions[...,12]=1
    actions[...,15]=.5
    actions[0,...,15]=.48; actions[2,...,15]=.52
    valid=np.ones((k,q),bool)
    # zero=1; A1 moves q0,q1,q3; q2 stays native.
    selected=np.array([[1,1,1,1],[2,0,1,2]],np.int64)
    probs=np.zeros((1,k,q,6),np.float32)
    probs[0,1]=.4
    probs[0,0]=.6
    probs[0,2]=.7

    # Utilities: native [2,3,4,5]/6; proposed q0 better, q1 worse, q3 tie.
    # friction maps to CDF utility count over [.2,.4,.6,.8,1.,1.2].
    def fr_for_count(n):
        return {0:-1.,1:1.2,2:1.0,3:.8,4:.6,5:.4,6:.2}[n]
    native_counts=[2,3,4,5]
    prop_counts=[3,2,4,5]
    fr=np.full((k,q),-1.,np.float32)
    fr[1]=np.array([fr_for_count(x) for x in native_counts],np.float32)
    fr[2,0]=fr_for_count(prop_counts[0])
    fr[0,1]=fr_for_count(prop_counts[1])
    fr[2,3]=fr_for_count(prop_counts[3])
    # Fill unused valid cells with native-equivalent labels.
    for kk in range(k):
        for qq in range(q):
            if fr[kk,qq] < 0:
                fr[kk,qq]=fr[1,qq]

    payload={
        "actions":actions,"valid":valid,"zero_index":np.array(1),
        "query_ids":np.arange(q,dtype=np.int64),
        "offsets_mm":np.array([-20.,0.,20.],np.float32),
        "output_methods":np.array(["stage1_native","A1"]),
        "output_policies":np.array(["native","fixed_0"]),
        "selected":selected,
        "scorer_names":np.array(["A1"]),
        "probabilities":probs,
        "original_native_score":np.array([.9,.8,.7,.6],np.float32),
        "rank_scores":np.array([[.9,.8,.7,.6],[.7,.6,.4,.7]],np.float32),
    }
    labels={"evaluated_mask":valid,"friction":fr}
    return payload,labels


def test_compact_source_keeps_only_a1_moves_and_three_way_targets():
    payload,labels=_payload_and_labels()
    ex=compact_from_source(payload,labels)
    np.testing.assert_array_equal(ex["query_pos"],[0,1,3])
    assert ex["actions"].shape==(2,3,17)
    assert ex["probabilities"].shape==(2,3,6)
    np.testing.assert_array_equal(
        ex["target_class"],
        [BENEFICIAL,HARMFUL,EQUIVALENT],
    )
    np.testing.assert_allclose(ex["offsets_mm"],[20,-20,20])


def test_class_from_delta_treats_ties_explicitly():
    got=class_from_delta(np.array([-.2,-1e-9,0,1e-9,.2],np.float32),eps=1e-7)
    np.testing.assert_array_equal(
        got,[HARMFUL,EQUIVALENT,EQUIVALENT,EQUIVALENT,BENEFICIAL]
    )


def test_move_subset_mixes_high_margin_and_tail():
    margin=np.array([.9,.8,.7,.6,.5,.4,.3,.2,.1])
    ids=select_move_subset(margin,4)
    assert len(ids)==4 and len(np.unique(ids))==4
    assert 0 in ids and 1 in ids
    assert np.any(ids>=2)


def test_c2v2_variants_have_same_parameters_and_valid_outputs():
    torch.manual_seed(1)
    c,q,h,w=16,5,64,64
    feature=torch.randn(c,16,16)
    K=torch.tensor([[60.,0.,32.],[0.,60.,32.],[0.,0.,1.]])
    actions=torch.zeros(2,q,17)
    actions[...,1]=.06; actions[...,2]=.02; actions[...,3]=.03
    actions[...,4]=1; actions[...,8]=1; actions[...,12]=1
    actions[0,...,15]=.48; actions[1,...,15]=.52
    probs=torch.rand(2,q,6)
    offsets=torch.tensor([20.,-20.,10.,-10.,20.])
    native_score=torch.rand(q)

    models={
        v:RepC2V2Verifier(c,dim=32,heads=4,dropout=0.,variant=v)
        for v in ("score","rgb","rgb_only")
    }
    counts=[sum(p.numel() for p in m.parameters()) for m in models.values()]
    assert len(set(counts))==1
    for m in models.values():
        out=m(feature,K,actions,probs,offsets,native_score,(h,w))
        assert out["class_logits"].shape==(q,3)
        assert out["delta"].shape==(q,)
        assert torch.isfinite(out["class_logits"]).all()
        assert torch.isfinite(out["delta"]).all()



def test_full_query_metrics_use_all_queries_and_define_verifier_increment():
    # Four moved proposals embedded in ten total Stage-1 queries.
    delta=np.array([.2,-.1,0.,.3],np.float32)
    p=np.array([.9,.8,.7,.6],np.float32)

    # Accept all: verifier is exactly A1 fixed-0, so increment/recovery are zero.
    all_on=full_query_gate_metrics(p,delta,0.,total_queries=10)
    assert np.isclose(all_on.a1_fixed0_gain,.04)
    assert np.isclose(all_on.verified_gain,.04)
    assert np.isclose(all_on.oracle_accept_gain,.05)
    assert np.isclose(all_on.verifier_increment,0.)
    assert np.isclose(all_on.oracle_gap,.01)
    assert np.isclose(all_on.oracle_gap_recovery,0.)

    # A perfect accept/reject policy accepts only positive-delta proposals.
    perfect=full_query_gate_metrics(
        np.array([.9,.1,.1,.9]),delta,.5,total_queries=10
    )
    assert np.isclose(perfect.verified_gain,.05)
    assert np.isclose(perfect.verifier_increment,.01)
    assert np.isclose(perfect.oracle_gap_recovery,1.)
    assert np.isclose(perfect.beneficial_retention,1.)
    assert np.isclose(perfect.harmful_rejection,1.)


def test_full_query_metrics_reject_all_is_not_confused_with_zero_proposal_denominator():
    delta=np.array([.2,-.4],np.float32)
    p=np.zeros(2,np.float32)
    m=full_query_gate_metrics(p,delta,1.,total_queries=20)
    assert np.isclose(m.verified_gain,0.)
    assert np.isclose(m.a1_fixed0_gain,-.01)
    assert np.isclose(m.oracle_accept_gain,.01)
    assert np.isclose(m.verifier_increment,.01)
    assert np.isclose(m.oracle_gap,.02)
    assert np.isclose(m.oracle_gap_recovery,.5)
    assert np.isclose(m.accept_rate,0.)
    assert np.isclose(m.proposal_rate,.1)
