"""CPU tests including synthetic train/cache/infer-contract CLI regressions."""
from __future__ import annotations
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from rare_grasp.core import *
from rare_grasp.pipeline import atomic_npz, load_index
from rare_grasp.runtime import check_mix10, Runtime

ROOT=Path(__file__).resolve().parents[1]


def synthetic(families=3,seed=0):
    rng=np.random.default_rng(seed);n=families*4
    g=np.zeros((n,17),dtype=np.float32)
    g[:,0]=.5;g[:,1]=.06;g[:,2]=.02;g[:,3]=np.tile([.01,.02,.03,.04],families)
    g[:,4:13]=np.eye(3).reshape(1,9);g[:,13]=np.repeat(np.arange(families)*.01,4);g[:,15]=.5;g[:,16]=-1
    count=np.tile(np.array([0,2,6,3],dtype=np.int64),families)
    bits=(np.arange(6)[None]>=6-count[:,None]).astype(np.uint8)
    return dict(features=rng.normal(size=(n,64)).astype(np.float32),base_logit=np.zeros(n,np.float32),
                count=count,bits=bits,family=np.repeat(np.arange(families),4),
                depth=np.tile(np.arange(4),families),grasps=g,physical_id=physical_ids(g))


def create_index(root,part,seed=0):
    entries=[];config={'checkpoint_sha256':'synthetic','backbone':'point_gvar'}
    for i in range(2):
        row=dict(domain=('original','gntrans')[i],scene=0,anno=0 if part=='train' else 5,
                 partition=part,split='train')
        path=root/f'{part}_{i}.npz';atomic_npz(path,**synthetic(seed=seed+i))
        entries.append(dict(row=row,file=str(path),sha256=sha256(path)))
    contract=dict(partition=part,config=config,config_digest=digest(config),families=3,
                  protocol_sha256='test-only',seed=0)
    obj=dict(schema=SCHEMA,contract=contract,entries=entries,complete=False,
             actual_frames=2,expected_frames=5200 if part=='train' else 40,missing=['smoke'])
    obj['index_sha256']=digest(obj);dest=root/f'{part}_index.json';write_json(dest,obj)
    return dest


class LossTests(unittest.TestCase):
    def test_original_formula_and_sign(self):
        good=torch.tensor([-1.,1.],requires_grad=True);y=torch.tensor([0,6]);f=torch.tensor([0,1])
        p,l,meta=ranking_loss(good,y,f)
        self.assertAlmostEqual(l.item(),torch.nn.functional.softplus(torch.tensor(-2.)).item(),places=7)
        _,bad,_=ranking_loss(-good,y,f);self.assertLess(l.item(),bad.item())
        self.assertEqual(meta['used_pairs'],1)
        (10*p+.5*l).backward();self.assertTrue(torch.isfinite(good.grad).all())

    def test_ties_no_margin(self):
        z=torch.tensor([2.,-3.,.5],requires_grad=True)
        _,pair,report=ranking_loss(z,torch.tensor([0,0,0]),torch.arange(3))
        self.assertEqual(pair.item(),0);self.assertEqual(report['used_pairs'],0)
        pair.backward();self.assertEqual(float(z.grad.abs().sum()),0)

    def test_quality_pairs_bidirectional_depth(self):
        family=torch.zeros(4,dtype=torch.long)
        for q in (torch.tensor([6,4,2,0]),torch.tensor([0,2,4,6])):
            good=q.float()-3
            self.assertLess(ranking_loss(good,q,family)[1].item(),ranking_loss(-good,q,family)[1].item())

    def test_scope_is_not_cross_image(self):
        y=torch.tensor([0,0,6,6]);fam=torch.tensor([0,0,1,1]);z=torch.zeros(4)
        _,a,m=ranking_loss(z,y,fam,scope='family');self.assertEqual(m['eligible_pairs'],0)
        _,b,m=ranking_loss(z,y,fam,scope='image');self.assertEqual(m['eligible_pairs'],4)
        self.assertGreater(b.item(),a.item())

    def test_pair_sampling_repeatable(self):
        z=torch.linspace(-2,2,28);y=torch.arange(28)%7;f=torch.arange(28)//4
        a=ranking_loss(z,y,f,seed=5,max_pairs=9);b=ranking_loss(z,y,f,seed=5,max_pairs=9)
        self.assertEqual(a[1].item(),b[1].item());self.assertEqual(a[2]['used_pairs'],9)

    def test_quality_count_validation(self):
        for y in (torch.tensor([0.,1.]),torch.tensor([-1,6]),torch.tensor([0,7])):
            with self.assertRaises(ValueError):ranking_loss(torch.zeros(2),y,torch.arange(2))

    def test_extreme_logit_stability(self):
        z=torch.tensor([1000.,-1000.],requires_grad=True)
        p,l,_=ranking_loss(z,torch.tensor([0,6]),torch.zeros(2,dtype=torch.long))
        (p+l).backward();self.assertTrue(torch.isfinite(z.grad).all())


class ContractTests(unittest.TestCase):
    def test_manifest_counts_and_leakage(self):
        p=make_protocol();verify_protocol(p)
        self.assertEqual(len(p['train']),5200);self.assertEqual(len(p['validation']),40)
        self.assertEqual(len(p['test']),4680)
        self.assertFalse({frame_key(x) for x in p['train']}&{frame_key(x) for x in p['validation']})
        for domain in ('original','gntrans'):
            self.assertEqual(sum(x['domain']==domain for x in p['train']),2600)
            for split in ('test_seen','test_similar','test_novel'):
                self.assertEqual(sum(x['domain']==domain and x['split']==split for x in p['test']),780)

    def test_protocol_tampering_fails(self):
        p=make_protocol();p['train'][0]['anno']=1
        with self.assertRaises(ValueError):verify_protocol(p)

    def test_cache_physical_identity(self):
        x=synthetic();validate_cache(x)
        x['grasps'][0,1]=.08
        with self.assertRaises(ValueError):validate_cache(x)

    def test_cache_no_missing_depths(self):
        x=synthetic();x['depth'][0]=2
        with self.assertRaises(ValueError):validate_cache(x)

    def test_cache_label_consistency(self):
        x=synthetic();x['bits'][0,0]=1
        with self.assertRaises(ValueError):validate_cache(x)

    def test_friction_float32_boundary(self):
        f=np.array([.2,.4,.6,1.2,-1],dtype=np.float32)
        b=success_bits(f,np.array([False,False,False,True,False]))
        self.assertEqual(b.sum(1).tolist(),[6,5,4,0,0])

    def test_new_head_epoch0_equals_base(self):
        model=ResidualRanker();x=torch.randn(3,4,64);z=torch.randn(3,4)
        d=torch.arange(4).expand(3,4)
        self.assertTrue(torch.equal(model(x,z,d),z))
        model(x,z,d).mean().backward();self.assertIsNotNone(model.net[-1].weight.grad)

    def test_decode_policies_freeze_physical_width(self):
        bank=synthetic(2)['grasps'].reshape(2,4,17)
        bank[...,0]=np.array([.2,.8,.3,.1]);bank[...,1]=np.array([.04,.05,.06,.08])
        scores=np.array([[.1,.2,.9,.4],[.9,.3,.1,.5]])
        old=select_grasps(bank,scores,'native');both=select_grasps(bank,scores,'both')
        depth=select_grasps(bank,scores,'depth_only');rank=select_grasps(bank,scores,'rank_only')
        np.testing.assert_array_equal(depth[:,1:],both[:,1:])
        np.testing.assert_array_equal(rank[:,1:],old[:,1:])
        np.testing.assert_allclose(depth[:,0],old[:,0])
        np.testing.assert_allclose(rank[:,0],scores.max(1))
        np.testing.assert_allclose(both[:,1],[.06,.04])

    def test_selected_gain_equals_regret_reduction(self):
        q=synthetic(2)['count'].reshape(2,4)
        a=candidate_metrics(np.tile([0,3,2,1],(2,1)),q)
        b=candidate_metrics(np.tile([0,1,3,2],(2,1)),q)
        self.assertEqual(a['oracle_q'],b['oracle_q'])
        self.assertAlmostEqual(b['selected_q']-a['selected_q'],a['regret']-b['regret'])

    def test_mixed50_weights_rejected(self):
        good=dict(epoch=20,use_cdf=True,gntrans_mix_protocol=dict(original_train_count=2600,
             gntrans_train_count=2600,mixed_train_count=5200,train_fraction_each_domain=.1))
        check_mix10(good)
        bad=copy.deepcopy(good);bad['gntrans_mix_protocol']['original_train_count']=12800
        with self.assertRaises(ValueError):check_mix10(bad)
        good['epoch']=16
        with self.assertRaises(ValueError):check_mix10(good)


class RuntimeCaptureTests(unittest.TestCase):
    @staticmethod
    def rotation(view,angle):
        R=torch.eye(3)[None].repeat(len(angle),1,1)
        R[:,0,0]=angle.cos();R[:,1,1]=angle.cos()
        R[:,0,1]=-angle.sin();R[:,1,0]=angle.sin()
        return R

    def test_point_and_gvar_native_full_feature_capture(self):
        for backbone in ('point','point_gvar'):
            class FakeModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.kview_grasp_module=nn.Module()
                    self.kview_grasp_module.decoder=nn.Module()
                    self.kview_grasp_module.decoder.score_dropout=nn.Dropout(0)
                def forward(self,batch):
                    shape=(1,64,1024*12) if backbone=='point' else (1024*12*4,64)
                    self.kview_grasp_module.decoder.score_dropout(torch.zeros(shape))
                    cdf=torch.zeros(1,6,1024,12,4)
                    cdf[:,:,:,:,2]=1;cdf[:,:,:,4,2]=2
                    depth=torch.ones(1,1,2,2)*.5
                    return dict(grasp_cdf_pred_angle_depth=cdf,
                        grasp_width_pred_angle_depth=torch.ones(1,4,1024,12),
                        xyz_graspable=torch.tensor([0.,0.,.5]).expand(1,1024,3),
                        grasp_top_view_xyz=torch.tensor([0.,0.,-1.]).expand(1,1024,3),
                        depth_net_pred=depth,depth_map_used_for_geometry=depth)
            rt=Runtime.__new__(Runtime);rt.model=FakeModel().eval()
            rt.config={'backbone':backbone};rt.cfg=SimpleNamespace(grasp_max_width=.1)
            rt.sample=lambda row,seed: ({},0);rt.rotation=self.rotation
            def decode(ep,use_cdf=True):
                g=torch.zeros(1024,17);g[:,0]=torch.sigmoid(torch.tensor(2.))
                g[:,1]=.1;g[:,2]=.02;g[:,3]=.03;g[:,15]=.5;g[:,16]=-1
                angles=torch.full((1024,),4.)*(np.pi/12)
                g[:,4:13]=self.rotation(None,angles).reshape(1024,9)
                return [g]
            rt.decode=decode
            snap=rt.snapshot({},0)
            self.assertEqual(snap['features'].shape,(1024,4,64))
            self.assertEqual(snap['grasps'].shape,(1024,4,17))
            self.assertLessEqual(snap['native_parity_max_abs'],1e-6)
            self.assertEqual(len(rt.model.kview_grasp_module.decoder.score_dropout._forward_hooks),0)


class CLITests(unittest.TestCase):
    def run_cli(self,*args):
        return subprocess.run([sys.executable,'-m','rare_grasp.pipeline',*map(str,args)],
                              cwd=ROOT,capture_output=True,text=True,timeout=40)

    def test_manifest_and_help(self):
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'manifest.json';p=self.run_cli('manifest','--output',f)
            self.assertEqual(p.returncode,0,p.stderr);verify_protocol(json.loads(f.read_text()))
        for cmd in ('mine','fit','infer','eval','bind','index','compare','diagnose-cache'):
            p=self.run_cli(cmd,'--help');self.assertEqual(p.returncode,0,p.stderr)

    def test_cache_index_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            f=create_index(Path(td),'train')
            with self.assertRaises(ValueError):load_index(f,'test',True)
            with self.assertRaises(ValueError):load_index(f,'train',False)
            load_index(f,'train',True)
            obj=json.loads(f.read_text());obj['entries'][0]['sha256']='wrong';write_json(f,obj)
            with self.assertRaises(ValueError):load_index(f,'train',True)

    def test_synthetic_train_both_objectives_and_diagnose(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);tr=create_index(root,'train');val=create_index(root,'validation')
            initial=[]
            for objective in ('point','rare'):
                output=root/objective
                p=self.run_cli('fit','--train-index',tr,'--val-index',val,'--output',output,
                       '--objective',objective,'--epochs','2','--batch-frames','2',
                       '--device','cpu','--smoke','--seed','3')
                self.assertEqual(p.returncode,0,p.stderr)
                ck=torch.load(output/'last.pt',weights_only=False)
                self.assertEqual(ck['epoch'],2);self.assertEqual(ck['updates'],2)
                history=json.loads((output/'history.json').read_text());initial.append(history[0])
                p=self.run_cli('diagnose-cache','--index',val,'--ranker',output/'last.pt',
                              '--output',output/'diagnostics.json','--partition','validation','--allow-partial')
                self.assertEqual(p.returncode,0,p.stderr)
                report=json.loads((output/'diagnostics.json').read_text())
                self.assertEqual(len(report['records']),2)
                # Existing outputs must not be silently overwritten.
                p=self.run_cli('fit','--train-index',tr,'--val-index',val,'--output',output,
                              '--objective',objective,'--epochs','1','--device','cpu','--smoke')
                self.assertNotEqual(p.returncode,0)
            self.assertEqual(initial[0],initial[1])

    def test_official_eval_refuses_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            write_json(Path(td)/'inference_manifest.json',dict(smoke_only=True,frames=1,status='complete'))
            p=self.run_cli('eval','--inference',td,'--dataset-root','/missing')
            self.assertNotEqual(p.returncode,0);self.assertIn('complete 30 scenes',p.stderr)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
