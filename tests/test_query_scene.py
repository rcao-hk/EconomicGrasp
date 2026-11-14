"""CPU tests; real historical checkpoints, CUDA and CAD require server acceptance."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from query_scene.core import *
from query_scene.model import Config,ContextScorer,intervene
from query_scene.pipeline import index,fit,load_model,load_models,metrics,Reuse
from query_scene.runtime import SceneRuntime
from query_scene.diagnose import diagnose,scene_delta,donor_entry
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def fixture(n=8,grid=2,f=16,s=12,seed=0):
    rng=np.random.default_rng(seed);g=np.zeros((n,17),np.float32)
    g[:,0]=np.linspace(.2,.8,n);g[:,1]=.06;g[:,2]=.02;g[:,3]=.02
    g[:,4:13]=np.eye(3).reshape(1,9);g[:,13]=np.arange(n)*.01;g[:,15]=.5;g[:,16]=-1
    z=np.log(g[:,0]/(1-g[:,0]));sg=np.zeros((grid*grid,8),np.float32);sg[:,-1]=1
    sg[:,:2]=rng.uniform(-1,1,(grid*grid,2));sg[:,6]=.5
    a=dict(grasps=g,features=rng.normal(size=(n,f)).astype(np.float32),base_logit=z,
           scene_features=rng.normal(size=(grid*grid,s)).astype(np.float32),scene_geometry=sg,
           scene_valid=np.ones(grid*grid,bool),query_id=np.arange(n),physical_id=physical_ids(g),
           label_mask=np.ones(n,bool),quality_count=np.arange(n)%7,sensor_keep=np.ones(n,bool))
    return a


def active_model(variant='scene'):
    m=ContextScorer(Config(16,12,variant,hidden=16,heads=4,layers=2))
    with torch.no_grad():m.score.weight.normal_();m.score.bias.fill_(.01)
    return m.eval()


def make_indices(root):
    plan=make_plan();source={'base_sha256':'base','selector_sha256':'mse','config_sha256':'cfg'}
    for part,scenes in [('train',(0,1)),('validation',(80,81))]:
        entries=[]
        c=dict(schema=SCHEMA,source=source,provider='synthetic',code=code_hash(),plan=plan['fingerprint'],partition=part,
               grid_size=2,queries=8,seed=0,labels=8,dtype='float32',reuse={})
        for domain in ('original','gntrans'):
            for scene in scenes:
                row=dict(domain=domain,scene=scene,anno=8 if part=='validation' else 0,partition=part,split='train')
                file=root/(frame_key(row)+'.npz');a=fixture(seed=scene);atomic_npz(file,a)
                entries.append(dict(row=row,path=str(file),sha256=sha256(file)))
        atomic_json(root/(part+'_index.json'),signed(dict(schema=SCHEMA,partition=part,complete=False,entries=entries,
                    contract=c,feature_dim=16,scene_dim=12)))
    return root/'train_index.json',root/'validation_index.json'


class SamplingTests(unittest.TestCase):
    def test_camera_geometry_and_no_candidate_dependency(self):
        fmap=torch.arange(16.).reshape(1,1,4,4);depth=torch.ones(1,1,4,4)*2
        K=torch.tensor([[[2.,0,1.5],[0,2.,1.5],[0,0,1.]]])
        a=sample_scene(fmap,depth,K,(4,4),2)
        np.testing.assert_allclose(a['scene_features'].flatten(),[2.5,4.5,10.5,12.5])
        np.testing.assert_allclose(a['scene_geometry'][:,4:7],[[-1,-1,2],[1,-1,2],[-1,1,2],[1,1,2]])
        self.assertEqual(a['scene_features'].shape,(4,1))
    def test_invalid_depth_does_not_remove_rgb(self):
        a=sample_scene(torch.ones(1,3,4,4),torch.full((1,1,4,4),float('nan')),torch.eye(3)[None],(4,4),2)
        self.assertTrue(a['scene_valid'].all());self.assertTrue(np.isfinite(a['scene_geometry']).all())
        self.assertEqual(float(a['scene_geometry'][:,4:].sum()),0.)
        self.assertEqual(float(a['scene_features'].sum()),12.)
    def test_invalid_intrinsics_rejected(self):
        k=torch.eye(3)[None];k[:,0,0]=0
        with self.assertRaises(ValueError):sample_scene(torch.ones(1,2,4,4),torch.ones(1,1,4,4),k,(4,4),2)
    def test_hook_cleanup_even_on_failure(self):
        class Reader(nn.Module):
            def forward(self,q,context):return q
        reader=Reader();model=SimpleNamespace(kview_grasp_module=SimpleNamespace(decoder=SimpleNamespace(reader=reader)))
        class Provider:
            provenance={'base_sha256':'b','selector_sha256':'s','config_sha256':'c'}
            def __init__(self):self.model=model
            def snapshot(self,row,seed):
                reader(torch.zeros(1),dict(pregeom=torch.ones(1,4,4,4),depth=torch.ones(1,1,4,4),K=torch.eye(3)[None],image_hw=(4,4)))
                raise RuntimeError('intentional')
            def label(self,*x):pass
            def sensor_keep(self,*x):pass
        rt=SceneRuntime(Provider(),2)
        with self.assertRaisesRegex(RuntimeError,'intentional'):rt.snapshot({},0)
        self.assertEqual(len(reader._forward_pre_hooks),0)


class ModelTests(unittest.TestCase):
    def test_zero_initialization_and_fixed_physics(self):
        a=fixture();x,_,_=tensors(a,'cpu');before=a['grasps'].copy()
        for v in VARIANTS:
            m=ContextScorer(Config(16,12,v,hidden=16,heads=4))
            self.assertTrue(torch.equal(m(**x),x['base_logit']))
        np.testing.assert_array_equal(before,a['grasps'])
    def test_parameter_budget_active_not_dummy(self):
        for f,s in ((128,128),(256,128),(128,256),(256,256)):
            counts=[ContextScorer(Config(f,s,v)).describe()['parameters'] for v in VARIANTS]
            self.assertLess((max(counts)-min(counts))/min(counts),.003)
    def test_all_query_permutation_equivariant(self):
        x,_,_=tensors(fixture(),'cpu');order=torch.tensor([5,1,3,7,0,4,2,6]);inv=order.argsort()
        for v in VARIANTS:
            m=active_model(v);y=m(**x);other=dict(x)
            for k in ('features','grasps','base_logit','action_valid'):other[k]=x[k][:,order]
            torch.testing.assert_close(y,m(**other)[:,inv],rtol=1e-5,atol=1e-5)
    def test_joint_scene_permutation_invariant_not_mismatch(self):
        x,_,_=tensors(fixture(),'cpu');m=active_model()
        torch.testing.assert_close(m(**x),m(**intervene(x,'scene_joint_permute',5)),atol=1e-5,rtol=1e-5)
    def test_scene_spatial_mismatch_and_source_isolation(self):
        x,_,_=tensors(fixture(),'cpu');changed=intervene(x,'scene_appearance_shuffle',4)
        self.assertTrue(torch.equal(changed['scene_geometry'],x['scene_geometry']))
        for v in ('local','candidate'):
            m=active_model(v);self.assertTrue(torch.equal(m(**x),m(**changed)))
        m=active_model();self.assertGreater(float((m(**x)-m(**changed)).abs().max().detach()),1e-7)
    def test_duplicate_density_not_new_scene_evidence(self):
        x,_,_=tensors(fixture(),'cpu');dup=intervene(x,'duplicate_candidates')
        for v in ('local','scene'):
            m=active_model(v);torch.testing.assert_close(m(**x),m(**dup)[:,:8],atol=1e-5,rtol=1e-5)
    def test_scene_changes_reach_only_scene_model(self):
        x,_,_=tensors(fixture(),'cpu');d,_,_=tensors(fixture(seed=5),'cpu');other=intervene(x,'scene_swap',donor=d)
        for v in ('local','candidate'):
            m=active_model(v);self.assertTrue(torch.equal(m(**x),m(**other)))
        m=active_model();self.assertGreater(float((m(**x)-m(**other)).abs().max().detach()),1e-6)
    def test_all_invalid_scene_safe(self):
        x,_,_=tensors(fixture(),'cpu');x['scene_valid'].fill_(False);x['scene_features'].fill_(float('nan'))
        self.assertTrue(torch.isfinite(active_model()(**x)).all())
    def test_gradients_reach_context_attention(self):
        x,y,mask=tensors(fixture(),'cpu')
        for v in VARIANTS:
            m=active_model(v).train();loss=(m(**x).sigmoid()-y/6).square().mean();loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            if v!='local':self.assertGreater(float(m.context[1].weight.grad.abs().sum()),0)


class ContractTests(unittest.TestCase):
    def test_missing_labels_are_not_zeros(self):
        a=fixture();a['label_mask'][0]=False;a['quality_count'][0]=-1;validate(a,8,2)
        a['quality_count'][0]=0
        with self.assertRaises(ValueError):validate(a,8,2)
    def test_physical_id_and_scene_count(self):
        a=fixture();validate(a,8,2);a['grasps'][0,1]=.08
        with self.assertRaises(ValueError):validate(a,8,2)
    def test_empty_sensor_pool_and_sparse_coverage(self):
        a=fixture();a['sensor_keep'][:]=False;r=metrics(a,a['grasps'][:,0]);self.assertIsNone(r['pre_nms_top1_q'])
        a=fixture();a['label_mask'][7]=False;a['quality_count'][7]=-1
        self.assertIsNone(metrics(a,a['grasps'][:,0])['pre_nms_top1_q'])
    def test_pair_bootstrap_uses_exact_frame_ids(self):
        rows=[]
        for v,delta in [('local',0),('scene',.1)]:
            for d in ('original','gntrans'):
                for s in (80,81):rows.append(dict(variant=v,condition='aligned',domain=d,scene=s,anno=8,mse=.3+delta))
        r=scene_delta(rows,('scene','aligned'),('local','aligned'),'mse',20)
        self.assertEqual(r['scenes'],2);self.assertAlmostEqual(r['mean'],.1)
        rows[-1]['anno']=72
        with self.assertRaises(ValueError):scene_delta(rows,('scene','aligned'),('local','aligned'),'mse',20)
    def test_donor_not_same_scene_or_other_domain(self):
        entries=[{'row':dict(domain=d,scene=s,anno=8)} for d in ('original','gntrans') for s in (80,81)]
        self.assertEqual(donor_entry(entries,entries[0]['row'])['row'],entries[1]['row'])
    def test_reuse_identity_and_hash(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);tr,va=make_indices(root);reuse=Reuse([tr,va],{'base_sha256':'base','selector_sha256':'mse'})
            idx=load_signed(tr);arr=read_frame(idx['entries'][0],idx['contract'])
            np.testing.assert_array_equal(reuse.lookup(idx['entries'][0]['row'],arr['physical_id']),arr['quality_count'])
            with self.assertRaises(ValueError):Reuse([tr],{'base_sha256':'wrong','selector_sha256':'mse'})


class IntegrationTests(unittest.TestCase):
    def test_train_diagnose_and_resume_all_variants(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);tr,va=make_indices(root);weights=[]
            for v in VARIANTS:
                args=SimpleNamespace(train_index=str(tr),val_index=str(va),output=str(root/v),variant=v,epochs=2,accumulate=2,
                    hidden=16,heads=4,layers=2,seed=0,residual_scale=.2,lr=1e-4,weight_decay=1e-4,max_pairs=16,
                    cpu_threads=1,device='cpu',smoke=True,resume=False,train_scene_shuffle=False)
                fit(args);model,p=load_model(root/v/'last.pt','cpu',True)
                ck=torch.load(root/v/'last.pt',weights_only=False);self.assertEqual(ck['updates'],4)
                args.resume=True;fit(args)
                self.assertEqual(torch.load(root/v/'last.pt',weights_only=False)['updates'],4)
                weights.append(f'{v}={root/v}/last.pt')
                with self.assertRaises(ValueError):load_model(root/v/'last.pt','cpu',False)
            out=root/'diagnostic.json'
            diagnose(SimpleNamespace(index=str(va),weight=weights,device='cpu',smoke=True,output=str(out),limit=0,seed=0,
                                     bootstrap=20,permutation_tol=1e-5))
            r=load_signed(out);self.assertEqual(len(r['checks']),12)
            self.assertIn("('scene', 'aligned')-('local', 'aligned')",r['contrasts'])
    def test_scene_shuffle_training_is_optional_control(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);tr,va=make_indices(root)
            a=SimpleNamespace(train_index=str(tr),val_index=str(va),output=str(root/'control'),variant='scene',epochs=1,accumulate=2,
                hidden=16,heads=4,layers=1,seed=0,residual_scale=.2,lr=1e-4,weight_decay=1e-4,max_pairs=16,
                cpu_threads=1,device='cpu',smoke=True,resume=False,train_scene_shuffle=True)
            fit(a);self.assertTrue((root/'control'/'last.pt').is_file())
    def test_cli_and_shell(self):
        for cmd in ('prepare','bind','export','index','train','diagnose','infer','eval','summarize'):
            p=subprocess.run([sys.executable,'-m','query_scene.cli',cmd,'--help'],cwd=ROOT,capture_output=True,text=True,timeout=15)
            self.assertEqual(p.returncode,0,p.stderr)
        p=subprocess.run(['bash','-n','scripts/query_scene/run.sh'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)



class ExtraIntegrationTests(unittest.TestCase):
    def test_export_index_and_inference_using_mock_provider(self):
        from query_scene.pipeline import export,infer
        from interactive_grasp.infer import emit_frame
        class Provider:
            provenance={'base_sha256':'b','selector_sha256':'s','config_sha256':'cfg'}
            def snapshot(self,row,seed):return fixture(1024,32,seed=row['scene']),{'native_decode_max_abs':0.,'native_reader_replay_max_abs':0.}
            def label(self,row,g):return np.full(len(g),3,dtype=np.int64)
            def sensor_keep(self,row,g):return np.arange(len(g))%2==0
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);plan=make_plan();atomic_json(root/'plan.json',plan)
            a=SimpleNamespace(plan=str(root/'plan.json'),shard=0,shards=1,labels=32,partition='train',domain='original',limit=1,
                              output=str(root/'cache'),provider='synthetic',grid_size=32,seed=0,reuse_index=[],resume=False,reserve_gib=0)
            with patch('query_scene.pipeline.create',return_value=Provider()):export(a)
            idxargs=SimpleNamespace(plan=a.plan,partition='train',cache=a.output,output=str(root/'idx.json'),smoke=True)
            index(idxargs);idx=read_index(idxargs.output,'train',True)
            self.assertEqual(len(idx['entries']),1)
            arr=read_frame(idx['entries'][0],idx['contract']);self.assertEqual(int(arr['label_mask'].sum()),32)
            self.assertTrue(np.all(arr['quality_count'][~arr['label_mask']]==-1))
            a.resume=True
            with patch('query_scene.pipeline.create',return_value=Provider()):export(a)
            models={v:active_model(v) for v in VARIANTS}
            proto={v:{'plan':plan['fingerprint'],'grid_size':32} for v in VARIANTS}
            arg=SimpleNamespace(plan=a.plan,domain='original',split='test_seen',limit=1,reference_root='',
                output=str(root/'infer'),weight=[],device='cpu',grid_size=32,seed=0)
            with patch('query_scene.pipeline.create',return_value=Provider()),patch('query_scene.pipeline.load_models',return_value=(models,proto,{})):
                infer(arg)
            m=load_signed(root/'infer'/'inference_manifest.json');self.assertEqual(len(m['records']),4)
            baseline=np.load(root/'infer/native/scene_0100/realsense/0000.npy')
            for v in VARIANTS:
                other=np.load(root/f'infer/{v}/scene_0100/realsense/0000.npy')
                np.testing.assert_array_equal(baseline[:,1:],other[:,1:])
    def test_precision_restores_on_exception(self):
        from query_scene.model import head_precision
        old=torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32=True
        try:
            with self.assertRaises(RuntimeError):
                with head_precision():
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32);raise RuntimeError('expected')
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        finally:torch.backends.cuda.matmul.allow_tf32=old

if __name__=='__main__':unittest.main()
