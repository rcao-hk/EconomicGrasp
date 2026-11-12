from __future__ import annotations
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from interactive_grasp.model import ModelConfig,InteractiveGraspRefiner,PerActionRead
from interactive_grasp.objectives import sparse_quality_loss,pair_changes
from interactive_grasp.contracts import make_plan,validate_plan,physical_ids,validate_frame
from interactive_grasp.capture import snapshot,FrozenScalarSelector,capture_hooks


def inputs(B=2,N=7,F=12,E=8):
    gen=torch.Generator().manual_seed(71)
    f=torch.randn(B,N,F,generator=gen);g=torch.zeros(B,N,17)
    g[...,0]=.5;g[...,1]=.06;g[...,2]=.02;g[...,3]=.02
    g[...,4:13]=torch.eye(3).reshape(9);g[...,13:16]=torch.randn(B,N,3,generator=gen)*.05
    g[...,15]+=.5;g[...,16]=-1
    mem=torch.randn(B,N,36,E,generator=gen);valid=torch.ones(B,N,36,dtype=torch.bool)
    valid[:,0]=False;mem[:,0]=0
    return dict(features=f,grasps=g,base_logit=torch.randn(B,N,generator=gen),
                memory=mem,memory_valid=valid,action_valid=torch.ones(B,N,dtype=torch.bool))


def learned_model(v,seed=5):
    m=InteractiveGraspRefiner(ModelConfig(12,8,v,hidden_dim=16,heads=4,layers=2,seed=seed)).eval()
    with torch.no_grad():
        gen=torch.Generator().manual_seed(19)
        m.head.weight.copy_(torch.randn(m.head.weight.shape,generator=gen)*.1)
    return m


class ArchitectureTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_epoch0_identity_all_variants(self):
        x=inputs()
        for v in 'ABCD':
            m=InteractiveGraspRefiner(ModelConfig(12,8,v,hidden_dim=16,heads=4)).eval()
            self.assertTrue(torch.equal(m(**x),x['base_logit']))
            self.assertFalse(m.describe()['graph_or_nms_mask'])

    def test_permutation_equivariance_after_nonzero_update(self):
        x=inputs();order=torch.tensor([6,2,0,3,4,1,5]);inv=torch.argsort(order)
        for v in 'ABCD':
            m=learned_model(v)
            a=m(**x);b=m(**{k:z[:,order] for k,z in x.items()})[:,inv]
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)

    def test_cross_candidate_effect_only_interaction_variants(self):
        x=inputs();changed={k:v.clone() for k,v in x.items()}
        changed['features'][:,1:]+=torch.linspace(-3,3,12)
        for v in 'ABCD':
            m=learned_model(v);delta=(m(**x)[:,0]-m(**changed)[:,0]).abs().max().item()
            if v in 'AB':self.assertEqual(delta,0.)
            else:self.assertGreater(delta,1e-6)

    def test_memory_effect_only_read_variants(self):
        x=inputs();changed={k:v.clone() for k,v in x.items()}
        changed['memory'][:,1:]+=torch.linspace(-2,3,8)
        for v in 'ABCD':
            m=learned_model(v);delta=(m(**x)-m(**changed)).abs().max().item()
            if v in 'AC':self.assertEqual(delta,0.)
            else:self.assertGreater(delta,1e-6)

    def test_all_invalid_read_is_exact_zero_including_bias(self):
        m=PerActionRead(16,8,4,3)
        with torch.no_grad():m.out.bias.fill_(7.)
        q=torch.randn(2,5,16);mem=torch.full((2,5,36,8),float('nan'))
        y=m(q,mem,torch.zeros(2,5,36,dtype=torch.bool),torch.ones(2,5,dtype=torch.bool))
        self.assertTrue(torch.equal(y,torch.zeros_like(y)))

    def test_padding_has_no_effect_and_nan_padding_is_safe(self):
        x=inputs();x['action_valid'][:,-2:]=False
        altered={k:v.clone() for k,v in x.items()}
        altered['features'][:,-2:]=float('nan');altered['grasps'][:,-2:]=float('nan')
        altered['memory'][:,-2:]=float('nan')
        for v in 'ABCD':
            m=learned_model(v)
            a=m(**x);b=m(**altered)
            self.assertTrue(torch.isfinite(b).all())
            torch.testing.assert_close(a[:,:-2],b[:,:-2],atol=1e-6,rtol=1e-6)

    def test_no_hidden_geometry_regression_or_mutation(self):
        x=inputs();copy={k:v.clone() for k,v in x.items()};m=learned_model('D')
        m(**x)
        for k in x:torch.testing.assert_close(x[k],copy[k])

    def test_capacity_and_shared_init(self):
        models=[InteractiveGraspRefiner(ModelConfig(256,64,v)) for v in 'ABCD']
        counts=[m.describe()['parameters'] for m in models]
        self.assertLess((max(counts)-min(counts))/min(counts),.01)
        for m in models[1:]:
            for k,v in m.feature.state_dict().items():self.assertTrue(torch.equal(v,models[0].feature.state_dict()[k]))
            for block_index in range(2):
                for k,v in m.blocks[block_index].ffn.state_dict().items():
                    self.assertTrue(torch.equal(v,models[0].blocks[block_index].ffn.state_dict()[k]))

    def test_training_does_not_allow_bypass_intervention(self):
        m=learned_model('D').train()
        with self.assertRaises(ValueError):m(**inputs(),bypass_interaction_query=True)

    def test_chunking_agrees(self):
        x=inputs();a=learned_model('D');b=learned_model('D')
        for block in b.blocks:block.reader.chunk=1
        torch.testing.assert_close(a(**x),b(**x),atol=1e-6,rtol=1e-6)


class LossContractTests(unittest.TestCase):
    def test_sparse_labels_never_become_negative(self):
        z=torch.tensor([[1.,-1.,2.,4.]],requires_grad=True)
        y=torch.tensor([[6,0,-1,-1]]);mask=y>=0;valid=torch.ones_like(mask)
        loss,r=sparse_quality_loss(z,y,mask,valid)
        loss.backward()
        self.assertTrue(torch.equal(z.grad[0,2:],torch.zeros(2)))
        self.assertEqual(r['labelled'],2);self.assertEqual(r['used_pairs'],1)
        changed=y.clone();changed[0,2:]=6
        other,_=sparse_quality_loss(z,changed,mask,valid)
        self.assertAlmostEqual(other.item(),loss.item())

    def test_ties_keep_mse_but_no_pair(self):
        z=torch.tensor([[1.,-1.,2.]],requires_grad=True);y=torch.zeros(1,3,dtype=torch.long)
        mask=torch.ones_like(y,dtype=torch.bool)
        loss,r=sparse_quality_loss(z,y,mask,mask)
        self.assertEqual(r['pair'],0.);self.assertEqual(r['used_pairs'],0)
        self.assertGreater(r['mse'],0);loss.backward()

    def test_pair_scope_is_within_image(self):
        z=torch.zeros(2,3);y=torch.tensor([[0,0,0],[6,6,6]]);mask=torch.ones_like(y,dtype=torch.bool)
        _,r=sparse_quality_loss(z,y,mask,mask)
        self.assertEqual(r['eligible_pairs'],0)

    def test_invalid_labels_and_zero_supervision_fail(self):
        z=torch.zeros(1,2);y=torch.tensor([[7,-1]]);mask=y>=0
        with self.assertRaises(ValueError):sparse_quality_loss(z,y,mask,torch.ones_like(mask))
        with self.assertRaises(ValueError):sparse_quality_loss(z,y,torch.zeros_like(mask),torch.ones_like(mask))

    def test_pair_sign_and_repairs(self):
        y=torch.tensor([[0,6]]);mask=torch.ones_like(y,dtype=torch.bool)
        _,a=sparse_quality_loss(torch.tensor([[-2.,2.]]),y,mask,mask)
        _,b=sparse_quality_loss(torch.tensor([[2.,-2.]]),y,mask,mask)
        self.assertLess(a['pair'],b['pair'])
        m=pair_changes(np.array([.9,.2]),np.array([.1,.8]),np.array([0,6]),np.ones(2,bool))
        self.assertEqual(m['repair'],1);self.assertEqual(m['damage'],0)

    def test_plan_has_stage_scene_holdout(self):
        p=make_plan();validate_plan(p)
        self.assertEqual(len(p['train']),2560);self.assertEqual(len(p['validation']),160)
        self.assertEqual(len(p['test']),23040)
        p['validation'][0]['scene']=0
        with self.assertRaises(ValueError):validate_plan(p)

    def test_source_selector_state_size(self):
        self.assertEqual(sum(p.numel() for p in FrozenScalarSelector().parameters()),4673)


# Minimal faithful native interfaces: full Q*A*D, reader chunking, MHA K/V and
# production flat width/score branches. Synthetic test, NOT CUDA/CAD acceptance.
class FakeReader(nn.Module):
    def __init__(self):
        super().__init__();self.attn=nn.MultiheadAttention(8,2,batch_first=True)
        self.proj=nn.Linear(8,8);self.last_debug={};self.counterfactual_mode='r0'
    def forward(self,queries,ctx):
        flat=queries.reshape(-1,8);chunks=[]
        for start in range(0,len(flat),17):
            mem=ctx['tokens'][start:start+17];valid=ctx['valid'][start:start+17]
            mem=torch.where(valid[...,None],mem,0.);pad=~valid.clone();pad[~valid.any(-1),0]=False
            out,_=self.attn(flat[start:start+17,None],mem,mem,key_padding_mask=pad)
            out=self.proj(out[:,0]);out=torch.where(valid.any(-1)[:,None],out,0.)
            chunks.append(out)
        self.last_debug={'called':1}
        return queries+torch.cat(chunks).reshape_as(queries)


def rotation(view,theta):
    n=len(theta);r=torch.eye(3)[None].repeat(n,1,1)
    c,s=theta.cos(),theta.sin();r[:,0,0]=c;r[:,0,1]=-s;r[:,1,0]=s;r[:,1,1]=c
    return r


class FakeBase(nn.Module):
    def __init__(self,N=4,raise_forward=False):
        super().__init__();self.N=N;self.raise_forward=raise_forward
        self.kview_grasp_module=nn.Module();d=nn.Module();self.kview_grasp_module.decoder=d
        d.reader=FakeReader();d.width_proj=nn.Conv1d(8,64,1);d.score_dropout=nn.Identity()
        q=torch.randn(1,N,12,4,8)
        self.register_buffer('queries',q)
        mem=torch.randn(N*12*4,36,8);valid=torch.ones(N*12*4,36,dtype=torch.bool)
        valid[4*4+2]=False
        self.register_buffer('tokens',mem);self.register_buffer('valid',valid)
    def forward(self,batch):
        N=self.N;centers=torch.tensor([0.,0.,.5]).expand(1,N,3).clone()
        centers[0,:,0]=torch.arange(N)*.01
        centersqa=centers[:,:,None].expand(1,N,12,3).reshape(1,N*12,3)
        theta=(torch.arange(12).float()*(np.pi/12))[None].expand(N,12).reshape(-1)
        r=rotation(None,theta).reshape(1,N*12,3,3)
        ctx=dict(tokens=self.tokens,valid=self.valid,centers=centersqa,rotations=r)
        action=self.kview_grasp_module.decoder.reader(self.queries,ctx)
        flat=action.reshape(1,N*12*4,8).transpose(1,2)
        features=self.kview_grasp_module.decoder.width_proj(flat)
        self.kview_grasp_module.decoder.score_dropout(features.transpose(1,2).reshape(-1,64))
        if self.raise_forward:raise RuntimeError('synthetic forward failure')
        cdf=torch.zeros(1,6,N,12,4);cdf[:,:,:,4,2]=2.
        depth=torch.full((1,1,2,2),.5)
        return dict(grasp_cdf_pred_angle_depth=cdf,grasp_width_pred_angle_depth=torch.ones(1,4,N,12),
                    xyz_graspable=centers,grasp_top_view_xyz=torch.tensor([0.,0.,-1.]).expand(1,N,3),
                    depth_net_pred=depth,depth_map_used_for_geometry=depth)


def decode(ep,use_cdf=True):
    n=ep['xyz_graspable'].shape[1];g=torch.zeros(n,17)
    g[:,0]=torch.sigmoid(torch.tensor(2.));g[:,1]=.1;g[:,2]=.02;g[:,3]=.03
    g[:,4:13]=rotation(None,torch.full((n,),4.)*(np.pi/12)).reshape(n,9)
    g[:,13:16]=ep['xyz_graspable'][0];g[:,16]=-1
    return [g]


class CaptureTests(unittest.TestCase):
    def test_native_full_chunk_capture_and_cleanup(self):
        model=FakeBase().eval().requires_grad_(False)
        reader=model.kview_grasp_module.decoder.reader
        valid=lambda ctx,ids,Q,A,D:ctx['valid'][ids]
        a,parity=snapshot(model,{},decode,rotation,valid,expected_queries=4)
        self.assertEqual(a['features'].shape,(4,8));self.assertEqual(a['memory'].shape,(4,36,8))
        self.assertFalse(a['memory_valid'][0].any());self.assertEqual(a['memory'][0].sum(),0.)
        self.assertEqual(parity['native_reader_replay_max_abs'],0.)
        self.assertLessEqual(parity['native_decode_max_abs'],1e-6)
        for m in model.modules():
            self.assertFalse(m._forward_pre_hooks);self.assertFalse(m._forward_hooks)
        self.assertEqual(reader.last_debug,{'called':1})
        original=physical_ids(a['grasps']);a['grasps'][:,0]=.3
        np.testing.assert_array_equal(original,physical_ids(a['grasps']))

    def test_exception_removes_hooks(self):
        model=FakeBase(raise_forward=True).eval().requires_grad_(False)
        with self.assertRaises(RuntimeError):
            snapshot(model,{},decode,rotation,lambda *args:None,expected_queries=4)
        for m in model.modules():
            self.assertFalse(m._forward_pre_hooks);self.assertFalse(m._forward_hooks)

    def test_refuse_unfrozen_model(self):
        with self.assertRaises(ValueError):snapshot(FakeBase(),{},decode,rotation,lambda *args:None,expected_queries=4)

    def test_bad_true_mask_fails(self):
        model=FakeBase().eval().requires_grad_(False)
        with self.assertRaises(RuntimeError):
            snapshot(model,{},decode,rotation,lambda ctx,ids,*_:torch.ones(len(ids),36,dtype=torch.bool),expected_queries=4)
        for m in model.modules():self.assertFalse(m._forward_pre_hooks)


if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
