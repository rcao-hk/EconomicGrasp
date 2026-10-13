"""Unit tests for the exact evidence switches used in frozen GVAR P0-A."""
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests'))
from test_gvar import inputs
from models.gripper_volume_reader import COUNTERFACTUAL_SOURCES
from analyze_gvar_p0a import EXPECTED_SOURCES


def test_reader_and_analysis_mode_contract_agree():
    assert {key:list(sources) for key,sources in COUNTERFACTUAL_SOURCES.items()}==EXPECTED_SOURCES


def test_geometry_and_unchanged_sources_are_byte_identical():
    reader,queries,context=inputs('volume_rel')
    reader.eval()
    centers=context['centers'].detach().clone()
    centers[...,0]=-.08  # ensure fixed/dynamic projection masks differ near boundary
    context['centers']=centers
    def capture(mode):
        record={'geom':[],'meta':[],'tokens':[],'padding':[]}
        def geom(module,args):record['geom'].append(args[0].detach().clone())
        def meta(module,args):record['meta'].append(args[0].detach().clone())
        def attn(module,args,kwargs):
            record['tokens'].append(args[1].detach().clone())
            record['padding'].append(kwargs['key_padding_mask'].detach().clone())
        hooks=[reader.geometry_projection.register_forward_pre_hook(geom),
               reader.metadata_projection.register_forward_pre_hook(meta),
               reader.attn.register_forward_pre_hook(attn,with_kwargs=True)]
        try:
            reader.set_counterfactual(mode)
            with torch.no_grad():result=reader(queries,context)
        finally:
            for hook in hooks:hook.remove()
        return result,{key:torch.cat(value,dim=0) for key,value in record.items()}
    observed={mode:capture(mode) for mode in COUNTERFACTUAL_SOURCES}
    ref=observed['r0'][1]
    for mode,(prediction,trace) in observed.items():
        assert prediction.shape==queries.shape and bool(torch.isfinite(prediction).all()),mode
        assert trace['geom'].shape[1:]==(36,6)
        torch.testing.assert_close(trace['geom'],ref['geom'],atol=0,rtol=0)
    # R1 only switches RGB samples; R2 only switches metadata (xyz+valid);
    # R3 only switches key-padding masks; geometry stays dynamic in all.
    for mode in ('r1','r3'):
        torch.testing.assert_close(observed[mode][1]['meta'],ref['meta'],atol=0,rtol=0)
    torch.testing.assert_close(observed['r2'][1]['meta'],observed['r4'][1]['meta'],atol=0,rtol=0)
    assert not torch.equal(observed['r2'][1]['meta'],ref['meta'])
    for mode in ('r1','r2'):
        assert torch.equal(observed[mode][1]['padding'],ref['padding'])
    assert torch.equal(observed['r3'][1]['padding'],observed['r4'][1]['padding'])
    assert not torch.equal(observed['r3'][1]['padding'],ref['padding'])
    # R3 cannot change token contents where both masks permit the same probes.
    shared=(~ref['padding']) & (~observed['r3'][1]['padding'])
    assert bool(shared.any())
    torch.testing.assert_close(ref['tokens'][shared],observed['r3'][1]['tokens'][shared],
                               atol=0,rtol=0)
    # Nontrivial RGB/metadata interventions really reach attention.
    assert not torch.equal(observed['r1'][1]['tokens'],ref['tokens'])
    assert not torch.equal(observed['r2'][1]['tokens'],ref['tokens'])
