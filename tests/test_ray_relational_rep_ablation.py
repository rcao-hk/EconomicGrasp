import torch

from utils.ray_relational_rep_ablation import (
    REPRESENTATION_MODES,
    compose_relational_ablation_tokens,
    relational_ablation_token_dim,
)
from utils.ray_relational_selective import compose_relational_tokens


def _inputs(K=7, N=5, C=8):
    selected=torch.randn(K,N,C)
    mean=torch.randn(K,N,C)
    raw=torch.randn(K,N)
    offsets=torch.tensor([-40,-20,-10,0,10,20,40],dtype=torch.float32)
    valid=torch.ones(K,N,dtype=torch.bool)
    valid[0,0]=False
    return selected,mean,raw,offsets,valid,3


def test_g0_exactly_matches_existing_composer():
    selected,mean,raw,offsets,valid,z=_inputs()
    old=compose_relational_tokens(selected,mean,raw,offsets,z)
    new=compose_relational_ablation_tokens(
        selected,mean,raw,offsets,z,"G0_current_full",valid_kn=valid
    )
    assert old.shape==new.shape
    assert torch.equal(old,new)


def test_all_representation_dimensions():
    selected,mean,raw,offsets,valid,z=_inputs(C=8)
    for mode in REPRESENTATION_MODES:
        tokens=compose_relational_ablation_tokens(
            selected,mean,raw,offsets,z,mode,valid_kn=valid
        )
        assert tokens.shape[:2]==(selected.shape[1],selected.shape[0])
        assert tokens.shape[-1]==relational_ablation_token_dim(8,mode)
        assert torch.isfinite(tokens).all()


def test_no_abs_raw_invariant_to_global_raw_shift():
    selected,mean,raw,offsets,valid,z=_inputs()
    a=compose_relational_ablation_tokens(
        selected,mean,raw,offsets,z,"G1_no_abs_raw",valid_kn=valid
    )
    b=compose_relational_ablation_tokens(
        selected,mean,raw+7.3,offsets,z,"G1_no_abs_raw",valid_kn=valid
    )
    assert torch.allclose(a,b,atol=1e-6,rtol=0)


def test_residual_modes_invariant_to_global_feature_translation():
    selected,mean,raw,offsets,valid,z=_inputs()
    shift_sel=torch.randn(1,1,selected.shape[-1])
    shift_mean=torch.randn(1,1,mean.shape[-1])
    for mode in ("G2_residual_only","G3_residual_profile","G4_mean_profile"):
        a=compose_relational_ablation_tokens(
            selected,mean,raw,offsets,z,mode,valid_kn=valid
        )
        b=compose_relational_ablation_tokens(
            selected+shift_sel,mean+shift_mean,raw+3.0,offsets,z,mode,valid_kn=valid
        )
        assert torch.allclose(a,b,atol=2e-6,rtol=0)


def test_profile_modes_ignore_invalid_center_values():
    selected,mean,raw,offsets,valid,z=_inputs()
    valid[0,0]=False
    selected2=selected.clone(); mean2=mean.clone(); raw2=raw.clone()
    selected2[0,0]=1e6
    mean2[0,0]=-1e6
    raw2[0,0]=1e6

    for mode in ("G3_residual_profile","G4_mean_profile"):
        a=compose_relational_ablation_tokens(
            selected,mean,raw,offsets,z,mode,valid_kn=valid
        )
        b=compose_relational_ablation_tokens(
            selected2,mean2,raw2,offsets,z,mode,valid_kn=valid
        )
        # Invalid center's own token may change, but all valid tokens for that ray
        # must keep the same broadcast profile statistics.
        valid_nk=valid.T
        assert torch.allclose(a[valid_nk],b[valid_nk],atol=2e-6,rtol=0)
