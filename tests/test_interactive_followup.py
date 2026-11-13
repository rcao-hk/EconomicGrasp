"""Synthetic CPU checks; no claim of real CUDA/CAD/AP compatibility."""
from __future__ import annotations
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from interactive_grasp.contracts import signed,atomic_json,atomic_npz,sha256,physical_ids,parameter_source_hash
from interactive_grasp.model import ModelConfig,InteractiveGraspRefiner
from interactive_followup.core import parse_alphas,mix_logits,quality_pairs,selection_stats,choose_alpha,paired_scene_bootstrap
from interactive_followup.sweep import val_ap_proxy,analyse
from interactive_followup.audit import audit


class PureTests(unittest.TestCase):
    def test_alpha_zero_exact_and_monotone_contract(self):
        alphas=parse_alphas('0,0.05,0.2,1')
        np.testing.assert_array_equal(mix_logits(np.array([0.,2.]),np.array([4.,-3.]),0,np.array([.5,.880797])),[.5,.880797])
        self.assertAlmostEqual(float(mix_logits(np.array([0]),np.array([4]),1)[0]),float(1/(1+np.exp(-4))))
        for bad in ('0.1,1','0,1,0.2','0,0,1','0,-1,1','0,1.01','0,nan,1'):
            with self.assertRaises(ValueError):parse_alphas(bad)

    def test_pairs_exclude_ties_and_include_repair_damage(self):
        q=np.array([6,0,4,4]);b=np.array([.1,.9,.4,.5]);s=np.array([.9,.1,.4,.6])
        p=quality_pairs(b,s,q,np.ones(4,bool))
        self.assertEqual(p['pairs'],5)
        self.assertGreaterEqual(p['repair'],1)
        self.assertEqual(p['damage'],0)
        self.assertIsNone(quality_pairs(b,s,q,np.array([0,0,1,1],bool))['base_agreement'])

    def test_selection_labels_are_not_imputed(self):
        q=np.array([-1,-1,6,-1]);lab=np.array([False,False,True,False]);keep=np.array([1,1,1,1],bool)
        a=selection_stats(np.array([.9,.8,.2,.1]),np.array([.1,.1,.9,.1]),q,lab,keep)
        self.assertTrue(a['top1_changed']);self.assertIsNone(a['top1_original_q']);self.assertEqual(a['top1_new_q'],1)

    def test_alpha_selection_nonimprovement_falls_back_to_zero(self):
        rows=[]
        for sid in (80,81):
            for d in ('original','gntrans'):
                for alpha,q in [(0.,.50),(.2,.49),(1.,.48)]:
                    rows.append(dict(scene=sid,domain=d,alpha=alpha,partition='validation',proxy_ap=q))
        c=choose_alpha(rows,alphas=[0.,.2,1.]);self.assertEqual(c['alpha'],0.)
        for r in rows:
            if r['alpha']==.2:r['proxy_ap']=.60
        c=choose_alpha(rows,alphas=[0.,.2,1.]);self.assertEqual(c['alpha'],.2)

    def test_scene_bootstrap_keeps_rgb_domains_paired(self):
        rows=[]
        for sid in (80,81,82):
            for d in ('original','gntrans'):
                rows.extend([dict(scene=sid,domain=d,alpha=1.,variant='D',proxy_ap=60.),
                             dict(scene=sid,domain=d,alpha=1.,variant='C',proxy_ap=40.)])
        p=paired_scene_bootstrap(rows,'D','C',1.,'proxy_ap',n_boot=100,seed=3)
        self.assertEqual(p['n_scenes'],3);self.assertAlmostEqual(p['mean'],20.)
        self.assertEqual(p['ci95'],[20.,20.])

    def test_proxy_ap_uses_six_friction_labels(self):
        module=types.ModuleType('graspnetAPI')
        class GraspGroup:
            def __init__(self,rows):self.rows=rows
            def nms(self,*_):return SimpleNamespace(grasp_group_array=self.rows[np.argsort(self.rows[:,0])[::-1]])
        module.GraspGroup=GraspGroup
        old=sys.modules.get('graspnetAPI');sys.modules['graspnetAPI']=module
        try:
            g=np.zeros((2,17),np.float32);g[:,4:13]=np.eye(3).reshape(9);g[:,13]=[0,.12];g[:,1]=.05
            labels=dict(assigned_obj=np.array([0,0]),friction=np.array([.2,1.2]),collision_or_empty=np.array([0,0]))
            p=val_ap_proxy(g,np.array([.9,.8]),np.array([1,1],bool),labels)
            self.assertGreater(p,0);self.assertLess(p,100)
            self.assertEqual(p,val_ap_proxy(g,np.array([.9,.8]),np.array([1,1],bool),labels))
        finally:
            if old is None:del sys.modules['graspnetAPI']
            else:sys.modules['graspnetAPI']=old


class IntegrationTests(unittest.TestCase):
    def _frame(self):
        n=1024;g=np.zeros((n,17),np.float32);g[:,0]=.5;g[:,1]=.05;g[:,2]=.02;g[:,3]=.01
        g[:,4:13]=np.eye(3,dtype=np.float32).reshape(-1);g[:,13]=np.arange(n)*.001;g[:,15]=.5;g[:,16]=-1
        lab=np.zeros(n,bool);lab[np.linspace(0,n-1,16,dtype=int)]=True
        q=np.full(n,-1,np.int64);q[lab]=np.arange(lab.sum())%7
        return dict(grasps=g,physical_id=physical_ids(g),query_id=np.arange(n),
           features=np.random.default_rng(1).normal(size=(n,8)).astype(np.float32),
           base_logit=np.zeros(n,np.float32),memory=np.zeros((n,36,8),np.float32),
           memory_valid=np.zeros((n,36),bool),angle_id=np.zeros(n,np.int64),depth_id=np.zeros(n,np.int64),
           label_mask=lab,quality_count=q,sensor_keep=np.ones(n,bool))

    def test_sweep_smoke_and_audit(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);frame=self._frame();path=root/'frame.npz';atomic_npz(path,frame)
            source=dict(base_sha256='synthetic_base',selector_sha256='synthetic_selector')
            idx=signed(dict(schema='interactive-frozen-grasp-stage1-v1',partition='validation',complete=False,
                            contract=dict(expected_queries=1024,source=source),entries=[dict(row=dict(scene=80,anno=8,domain='original',split='train',partition='validation'),path=str(path),sha256=sha256(path))]))
            indexfile=root/'index.json';atomic_json(indexfile,idx)
            weights=[]
            for variant in 'ABCD':
                cfg=ModelConfig(feature_dim=8,memory_dim=8,variant=variant,hidden_dim=16,layers=1,heads=4,read_chunk=256)
                model=InteractiveGraspRefiner(cfg)
                with torch.no_grad():model.head.bias.fill_(.05)
                protocol=dict(model=dict(cfg.__dict__),hyper=dict(epochs=1,smoke=True),source=source,
                              memory_dtype='float32',source_code_sha256=parameter_source_hash(),model_description=model.describe())
                ck=dict(schema='interactive-frozen-grasp-stage1-v1',epoch=1,protocol=protocol,model=model.state_dict())
                f=root/f'{variant}.pt';torch.save(ck,f);weights.append(f'{variant}={f}')
            a=SimpleNamespace(index=str(indexfile),weight=weights,output=str(root/'sweep'),full_labels_root=None,
                 alphas='0,0.1,1',select_alpha=False,device='cpu',cpu_threads=1,allow_code_drift=False,
                 allow_partial=True,limit=1,bootstrap=10,seed=0)
            analyse(a)
            result=json.loads((root/'sweep'/'summary.json').read_text())
            self.assertEqual(result['num_frames'],1)
            self.assertEqual(result['aggregate']['D']['0.0']['score_residual_abs_mean'],0.0)
            self.assertEqual(result['choices'],{})

    def test_audit_exposes_missing_files_without_claiming_success(self):
        with tempfile.TemporaryDirectory() as td:
            result=audit(td)
            self.assertEqual(result['status'],'needs_investigation')
            self.assertEqual(len(result['missing']),20)

if __name__=='__main__':unittest.main()