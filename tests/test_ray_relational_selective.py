import numpy as np
import torch

from utils.ray_relational_selective import (
    CrossCenterRelationalSelective,
    compose_relational_tokens,
    relational_exact_action_losses,
    relational_token_dim,
    select_relational_correction,
)


def test_relational_token_contract():
    K,N,C=7,5,8
    selected=torch.randn(K,N,C)
    mean=torch.randn(K,N,C)
    raw=torch.randn(K,N)
    offsets=torch.tensor([-40,-20,-10,0,10,20,40],dtype=torch.float32)
    tokens=compose_relational_tokens(selected,mean,raw,offsets,3)
    assert tokens.shape==(N,K,relational_token_dim(C))
    native_flag=tokens[:,:, -1]
    assert torch.all(native_flag[:,3]==1)
    assert torch.all(native_flag[:,:3]==0)
    assert torch.all(native_flag[:,4:]==0)


def test_relational_model_and_losses():
    K,N,C=7,11,8
    selected=torch.randn(K,N,C)
    mean=torch.randn(K,N,C)
    raw=torch.randn(K,N)
    offsets=torch.tensor([-40,-20,-10,0,10,20,40],dtype=torch.float32)
    tokens=compose_relational_tokens(selected,mean,raw,offsets,3)
    valid=torch.ones(K,N,dtype=torch.bool)
    valid[0,:2]=False
    model=CrossCenterRelationalSelective(tokens.shape[-1],d_model=32,nhead=4,num_layers=1,ff_dim=64,dropout=0.0)
    out=model(tokens,valid.T,3)
    assert out["gate_logit"].shape==(N,)
    assert out["selector_logits"].shape==(N,K)
    assert out["delta_pred"].shape==(N,K)

    utility=torch.rand(K,N)
    utility[4,:5]=utility[3,:5]+0.5
    losses,targets=relational_exact_action_losses(out,utility,valid,3)
    total=losses["gate"]+losses["selector"]+losses["delta"]
    assert torch.isfinite(total)
    assert targets["move_target"].shape==(N,)


def test_gate_controls_native_fallback():
    N,K,z=4,7,3
    gate=torch.tensor([-10.0,10.0,0.0,2.0])
    logits=torch.zeros(N,K)
    logits[:,5]=2.0
    valid=torch.ones(N,K,dtype=torch.bool)
    selected,best,prob=select_relational_correction(gate,logits,valid,z,0.6)
    assert best.tolist()==[5,5,5,5]
    assert selected.tolist()==[z,5,z,5]
    assert np.all((prob>=0)&(prob<=1))


def test_selector_loss_ignores_tie_only_queries():
    K,N,C=7,6,4
    selected=torch.randn(K,N,C)
    mean=torch.randn(K,N,C)
    raw=torch.randn(K,N)
    offsets=torch.tensor([-40,-20,-10,0,10,20,40],dtype=torch.float32)
    tokens=compose_relational_tokens(selected,mean,raw,offsets,3)
    valid=torch.ones(K,N,dtype=torch.bool)
    model=CrossCenterRelationalSelective(tokens.shape[-1],d_model=16,nhead=4,num_layers=1,ff_dim=32,dropout=0.0)
    out=model(tokens,valid.T,3)
    utility=torch.full((K,N),0.5)
    losses,targets=relational_exact_action_losses(out,utility,valid,3)
    assert not bool(targets["move_target"].any())
    assert float(losses["selector"].detach())==0.0
