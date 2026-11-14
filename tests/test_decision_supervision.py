"""CPU synthetic integration tests. No real CAD/CUDA or experiment AP claims."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from diagnostics.decision_supervision.core import (
    BASE_SHA, SCHEMA, QualityProbe, physical_ids, validate, subset, native_rows,
    predict, stratify, pair_batch, loss, metrics, decode, digest, sha, write_json, write_npz)
from diagnostics.decision_supervision.data import (
    source_index, union_frame, read_pack, prepare, load_plan, frame_name)
from diagnostics.decision_supervision.runtime import compiled_relations, filtered_indices, infer, evaluate
from diagnostics.decision_supervision.train import fit, cross, load_head
from diagnostics.decision_supervision.report import effects, paired_ci

ROOT = Path(__file__).resolve().parents[1]


def bank(n=96):
    rng = np.random.default_rng(4)
    f = np.repeat(np.arange(n), 4); d = np.tile(np.arange(4), n)
    g = np.zeros((n*4,17), dtype=np.float64)
    g[:,1:3] = [.06,.02]; g[:,3] = (d+1)*.01
    g[:,4:13] = np.eye(3).reshape(9); g[:,13] = f*.001; g[:,15] = .5
    g[:,16] = -1
    logits = (np.repeat(np.linspace(.2,2,n),4) + np.tile([0,.1,.3,-.1],n)).astype(np.float32)
    g[:,0] = torch.as_tensor(logits).sigmoid().numpy()
    counts = np.repeat(np.tile([0,2,6],n//3+1)[:n],4)
    counts[d==0] = np.minimum(6, counts[d==0]+1)
    bits = np.array([np.arange(6)>=6-c for c in counts],dtype=np.uint8)
    x = dict(grasps=g, physical_id=physical_ids(g), features=rng.standard_normal((n*4,64)).astype(np.float32),
             base_logit=logits, family=f, depth=d, count=counts.astype(np.int64),bits=bits,
             edges=np.empty((0,2),np.int64))
    return x


def fixture(root):
    x = bank(); entries=[]; traced=[]
    contract = dict(seed=20261113)
    for scene in (0,80,90):
        for domain in ('original','gntrans'):
            row = dict(domain=domain,scene=scene,anno=0,split='train',partition='train')
            for strategy,families in [('uniform',np.arange(64)),('decision',np.arange(32,96))]:
                arr=subset(x,families)
                p=root/f'{domain}_{scene}_{strategy}.npz';write_npz(p,arr)
                entries.append(dict(row=row,strategy=strategy,file=str(p),sha256=sha(p)))
    idx=dict(schema='quality-decision-p0p2-v1',kind='decision-cache-index',entries=entries,
             config_digest='config',base_checkpoint_sha256=BASE_SHA,contract=contract)
    idx['index_sha256']=digest(idx);write_json(root/'old.json',idx)
    acceptance=root/'acceptance.json';write_json(acceptance,{'synthetic':True})
    _,pairs=source_index(root/'old.json',formal=False)
    trace_contract=dict(source_index_sha256=idx['index_sha256'],acceptance_sha256=sha(acceptance),synthetic=True)
    for _,es in pairs:
        union=union_frame(es);nr=native_rows(union)
        union['edges']=np.stack([nr[:-1],nr[1:]],axis=1)
        union['native_sensor_valid']=np.ones(96,dtype=bool)
        p=root/(frame_name(es['uniform']['row'])+'.traced.npz');write_npz(p,union)
        traced.append(dict(row=es['uniform']['row'],file=str(p),sha256=sha(p),contract_sha256=digest(trace_contract),
                           source_sha256={s:es[s]['sha256'] for s in es}))
    tm=dict(contract=trace_contract,entries=traced);tm['manifest_sha256']=digest(tm)
    write_json(root/'trace.json',tm)
    plan=prepare(root/'old.json',[root/'trace.json'],root/'plan.json',acceptance,formal=False)
    return idx,plan


class CoreTests(unittest.TestCase):
    def test_capacity_initial_native_and_state_layout(self):
        m=QualityProbe();x=bank()
        self.assertEqual(sum(p.numel() for p in m.parameters()),5633)
        torch.testing.assert_close(predict(m,x,'cpu'),torch.as_tensor(x['base_logit']),rtol=0,atol=0)
        self.assertEqual(set(m.state_dict()),{'feature_norm.weight','feature_norm.bias','net.0.weight','net.0.bias','net.2.weight','net.2.bias'})

    def test_compatible_forward_math(self):
        m=QualityProbe();torch.nn.init.normal_(m.net[-1].weight)
        x=bank();feat=torch.tensor(x['features']);base=torch.tensor(x['base_logit'])
        g=torch.tensor(x['grasps'],dtype=torch.float32);d=torch.tensor(x['depth'])
        geom=g[:,1:16].clone();geom[:,0]/=.1;geom[:,1]/=.02;geom[:,2]/=.04
        ref=base+m.net(torch.cat([m.feature_norm(feat),geom,torch.nn.functional.one_hot(d,4),base[:,None]],-1)).squeeze(-1)
        torch.testing.assert_close(predict(m,x,'cpu'),ref,rtol=0,atol=0)

    def test_validate_labels_and_identity(self):
        x=bank();validate(x)
        for name in ('bits','count','physical_id'):
            bad=copy.deepcopy(x)
            if name=='physical_id':bad[name][0]='0'*64
            elif name=='bits':bad[name][0,0]=1-bad[name][0,0]
            else:bad[name][0]=6
            with self.subTest(name=name),self.assertRaises(ValueError):validate(bad)

    def test_subset_edge_remap_and_native_depth(self):
        x=bank();nr=native_rows(x);x['edges']=np.array([[nr[0],nr[2]],[nr[1],nr[2]]])
        y=subset(x,[0,2]);self.assertEqual(y['edges'].tolist(),[[2,6]])
        self.assertEqual(native_rows(y).tolist(),[2,6])

    def test_no_duplicate_or_quality_tie_pairs(self):
        count=np.repeat([0,1,6,3],4);e=np.array([[0,4],[0,8],[4,12]])
        for mode in ('random','targeted'):
            pairs,w,a=pair_batch(count,e,mode,5,budget=24)
            self.assertEqual(len(np.unique(pairs,axis=0)),len(pairs))
            self.assertTrue(np.all(count[pairs[:,0]]!=count[pairs[:,1]]))
            self.assertAlmostEqual(float(w.sum()),1)
            self.assertEqual(len(pairs),24)
        self.assertGreaterEqual(a['conflict_loss_mass'],.5)
        self.assertEqual(a['targeted_group_pairs'],3)

    def test_no_edges_exact_random_fallback(self):
        count=np.arange(12)%7;e=np.empty((0,2),int)
        a,w,_=pair_batch(count,e,'random',5)
        b,v,m=pair_batch(count,e,'targeted',5)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(w,v)
        self.assertTrue(m['fallback_to_random'])

    def test_quality_ties_have_no_pair_loss(self):
        x=bank();x['count'][:]=0;x['edges']=np.array([[2,6]])
        z=torch.zeros(len(x['count']),requires_grad=True)
        l,a=loss(z,x,'targeted',3);l.backward()
        self.assertEqual(a['sampled_pairs'],0);self.assertEqual(a['pair_loss'],0)
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_targeted_order_direction_is_true_quality(self):
        count=np.array([0,6]);e=np.array([[0,1]])
        pairs,weights,_=pair_batch(count,e,'targeted',0)
        self.assertEqual(pairs.tolist(),[[0,1]])
        z=torch.tensor([1.,0.],requires_grad=True)
        q=torch.tensor(count)/6
        pairloss=torch.nn.functional.softplus(-torch.sign(q[0]-q[1])*(z[0]-z[1]))
        pairloss.backward();self.assertGreater(float(z.grad[0]),0);self.assertLess(float(z.grad[1]),0)

    def test_stratified_budget_anchors_and_audited_shortage(self):
        x=bank();x.update(uniform_families=np.arange(64),decision_families=np.arange(32,96),
                          native_sensor_valid=np.ones(96,bool))
        picked,a=stratify(x,20261114)
        self.assertEqual(len(np.unique(picked)),64)
        self.assertGreaterEqual(len(np.intersect1d(picked,np.arange(32,64))),32)
        self.assertEqual(a['hard_zero_selected']+a['middle_selected']+a['high_selected']+a['fallback_families'],32)
        y=copy.deepcopy(x);y['native_sensor_valid'][:]=False
        p,b=stratify(y,0);self.assertEqual(b['fallback_families'],32)
        self.assertEqual(len(p),64)

    def test_decode_fixed_actual_depth_and_no_geometry_mutation(self):
        b=bank(96)['grasps'].reshape(96,4,17);scores=np.zeros((96,4));scores[:,0]=.9;scores[:,2]=.1
        before=b.copy()
        exact=decode(b,scores,'rank_executed');maximum=decode(b,scores,'rank_max')
        np.testing.assert_array_equal(exact[:,1:16],b[:,2,1:16])
        np.testing.assert_array_equal(exact[:,1:16],maximum[:,1:16])
        self.assertTrue(np.all(exact[:,0]==.1));self.assertTrue(np.all(maximum[:,0]==.9))
        np.testing.assert_array_equal(before,b)


class TraceTests(unittest.TestCase):
    @staticmethod
    def nms(g):
        order=np.argsort(-g[:,0],kind='stable');keep=[]
        for i in order:
            if all(np.linalg.norm(g[i,13:16]-g[j,13:16])>.03 for j in keep):keep.append(i)
        return g[keep]

    def test_full_pool_context_not_subset_nms(self):
        g=bank(3)['grasps'][[0,4,8]].copy();g[:,13]=[0,.01,.02];g[:,0]=[.5,.9,.6]
        e,a=compiled_relations((g,np.arange(3)),[0,2],self.nms)
        self.assertEqual(e,[]);self.assertEqual(a['cached_top50_survivors'],0)
        e,a=compiled_relations((g,np.arange(3)),[0,1,2],self.nms)
        self.assertEqual(e,[(0,1),(1,2)])

    def test_duplicate_geometry_mapping(self):
        g=bank(3)['grasps'][[0,0,4]].copy()
        self.assertEqual(filtered_indices(g,g).tolist(),[0,1,2])
        bad=g.copy();bad[0,0]=9
        with self.assertRaises(ValueError):filtered_indices(g,bad)

    def test_compiler_identity_loss_rejected(self):
        g=bank(3)['grasps'][[0,4,8]]
        def wrong(z):
            out=z[:1].copy();out[0,13]+=.5;return out
        with self.assertRaises(RuntimeError):compiled_relations((g,np.arange(3)),[0,1,2],wrong)


class IntegrationTests(unittest.TestCase):
    def test_source_union_integrity_and_forbid_test_scene(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);idx,p=fixture(root)
            _,frames=source_index(root/'old.json',formal=False)
            u=union_frame(frames[0][1]);self.assertEqual(len(u['count']),384)
            with self.assertRaises(ValueError):source_index(root/'old.json',formal=True)
            bad=copy.deepcopy(idx);bad.pop('index_sha256');bad['entries'][0]['row']['scene']=160
            bad['index_sha256']=digest(bad);write_json(root/'bad.json',bad)
            with self.assertRaises(ValueError):source_index(root/'bad.json',formal=False)

    def test_shared_physical_cache_precision_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);idx,_=fixture(root)
            _,frames=source_index(root/'old.json',formal=False);es=frames[0][1]
            e=copy.deepcopy(es);p=Path(e['decision']['file'])
            with np.load(p) as f:x={k:f[k] for k in f.files}
            # Legacy ID has float32 geometry but union must still reject different
            # underlying float64 executed actions even when ID stays unchanged.
            x['grasps'][0,13]+=1e-12;write_npz(p,x);e['decision']['sha256']=sha(p)
            with self.assertRaises(ValueError):union_frame(e)

    def test_prepare_never_stratifies_val_confirmation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,p=fixture(root)
            for e in p['entries']:
                if e['row']['scene']>=80:self.assertIsNone(e['stratified_families'])
                else:self.assertEqual(len(e['stratified_families']),64)
            with self.assertRaises(ValueError):load_plan(root/'plan.json',require_formal=True)

    def test_four_arm_training_and_common_pool_cross_cli(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);_,p=fixture(root);specs={};inits=[]
            for arm in 'ABCD':
                out=root/arm
                fit(SimpleNamespace(plan=str(root/'plan.json'),arm=arm,output=str(out),device='cpu',
                    seed=20261113,epochs=1,batch_frames=2,smoke=True))
                proto=json.loads((out/'protocol.json').read_text());inits.append(proto['initial_state_sha256'])
                self.assertEqual(proto['train_frames'],2)
                complete=json.loads((out/'complete.json').read_text());self.assertEqual(complete['steps'],1)
                specs[arm]=dict(file=str(out/'last.pt'),sha256=sha(out/'last.pt'))
            self.assertEqual(len(set(inits)),1)
            write_json(root/'heads.json',specs)
            cross(SimpleNamespace(heads=str(root/'heads.json'),plan=str(root/'plan.json'),source_index=None,
                                  output=str(root/'cross'),device='cpu',smoke=True))
            summary=json.loads((root/'cross/summary.json').read_text())
            self.assertEqual(set(summary['metrics']),{'uniform','decision','union'})
            self.assertEqual(len(summary['metrics']['union']),5)
            self.assertTrue(summary['reused_audit_scenes'])
            self.assertIn('不是', (root/'cross/REPORT_CN.md').read_text())
            bad=copy.deepcopy(specs['A']);bad['sha256']='0'*64
            with self.assertRaises(ValueError):load_head(bad,'cpu')

    def test_cross_legacy_checkpoint_all_pools(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);idx,_=fixture(root);heads={}
            for name in ('uniform_mse','uniform_rare','decision_mse','decision_rare'):
                cfg=dict(base_checkpoint_sha256=BASE_SHA,base_config_digest='config',hidden=64,
                         mode='score_feature_geometry',test_labels_used=False,
                         cache_contract={'index_sha256':idx['index_sha256']})
                path=root/(name+'.pt');torch.save(dict(schema='quality-decision-p0p2-v1',config=cfg,
                                                      model=QualityProbe().state_dict()),path)
                heads[name]=dict(file=str(path),sha256=sha(path))
            write_json(root/'heads.json',heads)
            cross(SimpleNamespace(heads=str(root/'heads.json'),plan=None,source_index=str(root/'old.json'),
                                  output=str(root/'cross'),device='cpu',smoke=True))
            m=json.loads((root/'cross/summary.json').read_text())['metrics']
            for pool in m:
                self.assertAlmostEqual(m[pool]['native']['mse'],m[pool]['uniform_mse']['mse'])

    def test_native_relation_endpoints_must_match_executed_depth(self):
        x=bank();x['edges']=np.array([[0,4]])
        with self.assertRaises(ValueError):validate(x)

    def test_inference_smoke_freezes_geometry_and_cannot_run_official_eval(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);x=bank(1024);g=x['grasps'].reshape(1024,4,17)
            rt=SimpleNamespace(config={'checkpoint_sha256':BASE_SHA},config_digest='config')
            rt.snapshot=lambda row,seed:dict(grasps=g,features=x['features'].reshape(1024,4,64),
                                           base_logit=x['base_logit'].reshape(1024,4))
            rt.sensor_collision_filter=lambda row,actions:actions[:16].copy()
            cfg=dict(mode='score_feature_geometry',hidden=64,base_checkpoint_sha256=BASE_SHA,
                     base_config_digest='config',test_labels_used=False,arm='A',seed=20261113,
                     plan_sha256='plan',smoke=False)
            path=root/'head.pt';torch.save(dict(schema=SCHEMA,config=cfg,model=QualityProbe().state_dict(),epoch=10),path)
            args=SimpleNamespace(weights=str(path),weights_sha256=sha(path),device='cpu',
                                 policies='rank_executed,both',output=str(root/'infer'),
                                 split='test_seen',interval=10,limit=1,snapshot_seed=20261113)
            with patch('diagnostics.decision_supervision.runtime.make_runtime',return_value=(rt,{'mock':True})):
                infer(args)
            m=json.loads((root/'infer/inference.json').read_text());self.assertFalse(m['complete'])
            self.assertEqual(m['records'][0]['frozen_bank_sha256'],m['records'][1]['frozen_bank_sha256'])
            out=np.load(root/'infer/rank_executed/scene_0100/realsense/0000.npy')
            np.testing.assert_array_equal(out[:,1:16],g[:16,2,1:16])
            with self.assertRaises(ValueError):evaluate(SimpleNamespace(inference=str(root/'infer'),dataset_root='none',workers=1))

    def test_cli_help(self):
        for module in ('data','runtime','train','report'):
            r=subprocess.run([sys.executable,'-m','diagnostics.decision_supervision.'+module,'--help'],
                             cwd=ROOT,capture_output=True,text=True,timeout=15)
            self.assertEqual(r.returncode,0,r.stderr)


class StatisticsTests(unittest.TestCase):
    def test_factorial_interaction_and_split_stratified_ci(self):
        a={k:np.ones((3,30))*v for k,v in dict(A=1,B=2,C=4,D=8).items()}
        e=effects(a);self.assertTrue(np.all(e['interaction_D-C-B+A']==3))
        ci=paired_ci(e['D-A_combined'],0,100);self.assertEqual(ci['mean_pp'],7)
        self.assertEqual(ci['ci95_pp'],[7,7])

if __name__=='__main__':unittest.main()
