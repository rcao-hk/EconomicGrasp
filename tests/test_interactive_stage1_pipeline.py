from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from interactive_grasp.contracts import (SCHEMA,signed,atomic_json,atomic_npz,sha256,
                    physical_ids,validate_frame,load_signed,digest)
from interactive_grasp.data import load_index,load_frame,ExactLabelReuse
from interactive_grasp.train import fit,load_refiner,atomic_torch
from interactive_grasp.infer import load_models,score_frame,emit_frame,evaluate
from interactive_grasp.analyze import paired_bootstrap,diagnose

ROOT=Path(__file__).resolve().parents[1]
SOURCE={'base_sha256':'base','selector_sha256':'mse','config_sha256':'cfg'}


def synthetic(N=8,seed=0):
    rng=np.random.default_rng(seed)
    g=np.zeros((N,17),dtype=np.float32)
    g[:,0]=.5;g[:,1]=.06;g[:,2]=.02;g[:,3]=(np.arange(N)%4+1)*.01
    g[:,4:13]=np.eye(3).reshape(9);g[:,13]=np.arange(N)*.01;g[:,15]=.5;g[:,16]=-1
    memory=rng.normal(size=(N,36,8)).astype(np.float32);valid=np.ones((N,36),bool)
    valid[0]=False;memory[0]=0
    label=np.ones(N,bool);label[-2:]=False;quality=np.arange(N)%7;quality[-2:]=-1
    return dict(grasps=g,features=rng.normal(size=(N,12)).astype(np.float32),
        base_logit=np.zeros(N,np.float32),physical_id=physical_ids(g),query_id=np.arange(N),
        angle_id=np.arange(N)%12,depth_id=np.arange(N)%4,memory=memory,memory_valid=valid,
        label_mask=label,quality_count=quality.astype(np.int64),sensor_keep=np.arange(N)%3!=0)


def make_index(root,part):
    entries=[]
    for i in range(4 if part=='train' else 2):
        row=dict(domain=('original','gntrans')[i%2],scene=(i//2 if part=='train' else 80),
                 anno=0,partition=part,split='train')
        a=synthetic(seed=i);validate_frame(a,8)
        f=root/f'{part}_{i}.npz';atomic_npz(f,a)
        entries.append(dict(row=row,path=str(f),sha256=sha256(f)))
    contract=dict(source=SOURCE,provider='synthetic',seed=0,memory_dtype='float32',
                  expected_queries=8,code_sha256='fixture',plan_sha256='fixture',partition=part)
    obj=signed(dict(schema=SCHEMA,partition=part,complete=False,entries=entries,
                    contract=contract,feature_dim=12,memory_dim=8,expected_frames=len(entries)))
    f=root/f'{part}_index.json';atomic_json(f,obj);return f


def args(root,variant='D',epochs=2):
    return SimpleNamespace(train_index=str(root/'train_index.json'),val_index=str(root/'validation_index.json'),
        smoke=True,epochs=epochs,accumulate=2,max_pairs=20,lr=1e-3,weight_decay=1e-4,
        output=str(root/variant),device='cpu',seed=7,cpu_threads=1,variant=variant,
        hidden=16,heads=4,layers=2,read_chunk=3,resume=False)


class DataTests(unittest.TestCase):
    def test_physical_contract_and_unknown_labels(self):
        a=synthetic();validate_frame(a,8)
        a['quality_count'][-1]=0
        with self.assertRaises(ValueError):validate_frame(a,8)
        a=synthetic();a['grasps'][0,1]=.08
        with self.assertRaises(ValueError):validate_frame(a,8)
        a=synthetic();a['memory'][0,0,0]=1.
        with self.assertRaises(ValueError):validate_frame(a,8)

    def test_complete_context_is_required(self):
        a=synthetic()
        with self.assertRaises(ValueError):validate_frame(a,1024)

    def test_cache_partition_hash_gates(self):
        with tempfile.TemporaryDirectory() as td:
            f=make_index(Path(td),'train')
            load_index(f,'train',True)
            with self.assertRaises(ValueError):load_index(f,'test',True)
            with self.assertRaises(ValueError):load_index(f,'train',False)
            data=json.loads(f.read_text());data['entries'][0]['sha256']='bad';atomic_json(f,data)
            with self.assertRaises(ValueError):load_index(f,'train',True)

    def test_legacy_label_reuse_matches_frame_and_action(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);a=synthetic();file=root/'legacy.npz'
            cnt=np.arange(8)%7;bits=np.arange(6)[None]>=6-cnt[:,None]
            atomic_npz(file,dict(grasps=a['grasps'],physical_id=a['physical_id'],count=cnt,bits=bits))
            obj=dict(contract=dict(config=dict(checkpoint_sha256='base'),fc_mode='official'),
                     entries=[dict(row=dict(domain='original',scene=0,anno=0),file=str(file),sha256=sha256(file))])
            obj['index_sha256']=digest(obj);idx=root/'legacy.json';atomic_json(idx,obj)
            reuse=ExactLabelReuse([str(idx)],'base')
            y=reuse.lookup(dict(domain='original',scene=0,anno=0),a['physical_id'])
            np.testing.assert_array_equal(y,cnt)
            missing=reuse.lookup(dict(domain='gntrans',scene=0,anno=0),a['physical_id'])
            self.assertTrue(np.all(missing==-1))
            changed=a['grasps'].copy();changed[:,1]=.09
            self.assertTrue(np.all(reuse.lookup(dict(domain='original',scene=0,anno=0),physical_ids(changed))==-1))
            with self.assertRaises(ValueError):ExactLabelReuse([str(idx)],'different_base')


class TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_four_variants_train_score_and_freeze_actions(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);make_index(root,'train');make_index(root,'validation');initial=[]
            paths=[]
            for v in 'ABCD':
                a=args(root,v);fit(a)
                ck=torch.load(root/v/'last.pt',weights_only=False)
                self.assertEqual(ck['updates'],4);self.assertEqual(ck['epoch'],2)
                initial.append(ck['history'][0]['validation'])
                paths.append(f'{v}={root/v/"last.pt"}')
                with self.assertRaises(ValueError):load_refiner(root/v/'last.pt','cpu')
                with self.assertRaises(FileExistsError):fit(a)
            self.assertTrue(all(x==initial[0] for x in initial))
            models,protocols,_=load_models(paths,'cpu',SOURCE,True)
            a=synthetic();scores=score_frame(models,protocols,a,'cpu')
            row=dict(domain='original',scene=100,anno=0,partition='test',split='test_seen')
            recs=emit_frame(root/'dumps',row,a,scores)
            self.assertEqual(len(recs),5)
            expected=a['grasps'][a['sensor_keep']]
            for rec in recs:
                actual=np.load(root/'dumps'/rec['path'])
                np.testing.assert_array_equal(actual[:,1:],expected[:,1:])
                self.assertEqual(sha256(root/'dumps'/rec['path']),rec['sha256'])
            d=SimpleNamespace(index=str(root/'validation_index.json'),partition='validation',allow_partial=True,
                weight=str(root/'D'/'last.pt'),device='cpu',limit=1,seed=0,duplicates=3,output=str(root/'diagnose.json'))
            diagnose(d);report=load_signed(d.output)
            self.assertIn('pre_interaction_read_query_pairs',report['records'][0])

    def test_epoch_boundary_resume_is_exact(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);make_index(root,'train');make_index(root,'validation')
            whole=args(root);whole.output=str(root/'whole');fit(whole)
            split=args(root);split.output=str(root/'resume')
            def crash(path,obj):
                atomic_torch(path,obj)
                if obj['epoch']==1:raise InterruptedError('synthetic interruption after atomic checkpoint')
            with patch('interactive_grasp.train.atomic_torch',side_effect=crash):
                with self.assertRaises(InterruptedError):fit(split)
            split.resume=True;fit(split)
            a=torch.load(root/'whole'/'last.pt',weights_only=False);b=torch.load(root/'resume'/'last.pt',weights_only=False)
            for k,v in a['model'].items():self.assertTrue(torch.equal(v,b['model'][k]),k)
            self.assertEqual(a['history'],b['history'])

    def test_stage_scene_overlap_rejected_across_domains(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);make_index(root,'train');v=make_index(root,'validation')
            val=load_signed(v);val.pop('fingerprint');val['entries'][0]['row']['scene']=0
            atomic_json(v,signed(val))
            with self.assertRaises(ValueError):fit(args(root))

    def test_eval_refuses_partial_context_or_subset(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);atomic_json(root/'inference_manifest.json',signed(dict(partial_smoke=True,frames=1)))
            with self.assertRaises(ValueError):evaluate(SimpleNamespace(inference=str(root)))

    def test_split_stratified_paired_bootstrap(self):
        b=paired_bootstrap([np.array([1.,3.]),np.array([2.,2.]),np.array([0.,6.])],100,3)
        self.assertAlmostEqual(b['mean_pp'],7/3)
        self.assertEqual(len(b['ci95']),2)

    def test_all_cli_help_and_shell_syntax(self):
        for cmd in ('plan','bind','export','index','train','infer','eval','diagnose','summarize'):
            p=subprocess.run([sys.executable,'-m','interactive_grasp.cli',cmd,'--help'],
                             cwd=ROOT,capture_output=True,text=True,timeout=15)
            self.assertEqual(p.returncode,0,p.stderr)
        p=subprocess.run(['bash','-n','scripts/interactive_stage1/run.sh'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)


if __name__=='__main__':
    torch.set_num_threads(1);unittest.main()
