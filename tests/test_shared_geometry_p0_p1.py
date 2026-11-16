"""Synthetic offline P0/P1 tests; no access to production GPU/CAD/test data."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
import numpy as np
import torch
from shared_geometry.data import atomic_json,atomic_npz,signed,sha256,frame_key,digest,tensor_scene
from shared_geometry.geometry import VisibleSurfaceTeacher
from shared_geometry.runtime import TargetLoader
from shared_geometry.model import ModelConfig,PredictedGeometryControl,quality_loss
from shared_geometry.p0_relative_decision import metrics_for_family,summarize,scene_bootstrap
from shared_geometry.p1_clean_visible import (make_cache_frame,score_cached,build,fit,compare,
    read_cache_index,load_cached,collate_cache,validated_target)
from tests.test_shared_geometry import fixture


def synthetic_sources(root):
    """Two independent RGB domains in two disjoint train/validation scenes."""
    root=Path(root)
    config=dict(source='rendered_visible',same_camera_as_native=True,
        calibration_description='Synthetic same-camera rendered-visible plane',
        depth_template=str(root/'vis_{partition}_{scene}_{domain}.npy'),
        intrinsics_template=str(root/'K_{partition}_{scene}_{domain}.npy'),
        depth_scale_to_m=1.,max_depth_m=2.)
    cfg_path=root/'teacher.json';cfg_path.write_text(json.dumps(config))
    loader=TargetLoader(cfg_path)
    out=[]
    for part,scene in (('train',0),('validation',80)):
        contract=dict(source={'base_sha256':'b','selector_sha256':'s','config_sha256':'cfg'},
            plan='synthetic',offsets_m=[0.,-.005,.005,-.01,.01,-.02,.02],
            tau=.03,band=.002,target_config=loader.config_sha256,partition=part)
        entries=[]
        for i,domain in enumerate(('original','gntrans')):
            row=dict(partition=part,scene=scene,anno=0,domain=domain,split='train')
            a=fixture(n=2,seed=i+scene+10)
            visible=a['pred_depth'].copy();visible.fill(.46+i*.025)
            dp=root/f'vis_{part}_{scene}_{domain}.npy';kp=root/f'K_{part}_{scene}_{domain}.npy'
            np.save(dp,visible);np.save(kp,a['K'])
            p=root/(frame_key(row)+'.npz');atomic_npz(p,a)
            _,provenance=loader.load(row,a['K'],a['pred_depth'].shape)
            atomic_json(p.with_suffix('.json'),signed(dict(row=row,sha256=sha256(p),
                contract=contract,target=provenance)))
            entries.append(dict(row=row,path=str(p.resolve()),sha256=sha256(p)))
        name=root/(part+'_source.json')
        atomic_json(name,signed(dict(schema='shared-visible-geometry-v1',complete=False,
                     partition=part,contract=contract,entries=entries)))
        out.append(name)
    return out[0],out[1],cfg_path


class P0Decisions(unittest.TestCase):
    def test_pairwise_accuracy_does_not_imply_safe_argmax(self):
        q=np.asarray([5,5,5,5,5,4,6])
        z=np.asarray([4,0,0,0,0,5,-1],dtype=float)
        offsets=[0,-.005,.005,-.01,.01,-.02,.02]
        r=metrics_for_family(z,q,offsets)
        self.assertTrue(r['harm']);self.assertTrue(r['opportunity']);self.assertTrue(r['move'])
        self.assertEqual(r['choice'],5)
        self.assertAlmostEqual(r['gain'],-1/6)
        self.assertEqual(r['nativepair_n'],2)
        self.assertTrue(any(x['wrong_improvement'] for x in r['alternatives']))
        self.assertTrue(r['allpair_correct'] is not None)

    def test_ties_and_native_wins_exact_ties(self):
        r=metrics_for_family([1,1,0,0,0,0,0],[3]*7,[0,-.005,.005,-.01,.01,-.02,.02])
        self.assertEqual(r['choice'],0);self.assertEqual(r['nativepair_n'],0)
        self.assertIsNone(r['nativepair_correct']);self.assertIsNone(r['allpair_correct'])

    def test_bootstrap_pairs_by_scene_not_action(self):
        a=[];b=[]
        for scene in (80,81):
            for domain in ('original','gntrans'):
                a += [dict(scene=scene,domain=domain,gain=.2)]*3
                b += [dict(scene=scene,domain=domain,gain=.1)]*3
        got=scene_bootstrap(a,b,n_boot=20)
        self.assertEqual(got['n_scenes'],2)
        self.assertAlmostEqual(got['mean'],.1)
        with self.assertRaises(ValueError):scene_bootstrap(a,b[:-3],n_boot=20)

    def test_p0_real_checkpoint_smoke_does_not_touch_test(self):
        from tests.test_shared_geometry import make_indices
        from shared_geometry.experiment import fit as original_fit
        from shared_geometry.p0_relative_decision import run as run_p0
        from shared_geometry.data import load_signed
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);tr,va=make_indices(root)
            args=SimpleNamespace(mode='local',epochs=1,batch_size=2,accumulate=1,
                val_batch_size=2,train_index=str(tr),val_index=str(va),
                output=str(root/'local'),smoke=True,resume=False,hidden=16,seed=0,
                device='cpu',cpu_threads=1,lr=1e-3,weight_decay=1e-4,geometry_weight=1.)
            original_fit(args)
            out=root/'p0.json'
            run_p0(SimpleNamespace(index=str(va),weight=['local='+str(root/'local'/'best.pt')],
                batch_size=2,bootstrap=16,seed=0,device='cpu',visibility=True,
                limit=0,smoke=True,output=str(out)))
            r=load_signed(out)
            self.assertEqual(r['partition'],'validation')
            self.assertIn('by_selected_unobserved_ratio',r['summaries']['local'])
            self.assertEqual(len(r['per_family']['local']),4)
            self.assertTrue(r['metrics_not_official_ap'])

    def test_p0_summary_has_native_quality_buckets(self):
        r=metrics_for_family([2,0,0,0,0,0,3],[3,1,1,1,1,1,4],
                             [0,-.005,.005,-.01,.01,-.02,.02])
        r.update(scene=80,domain='original',native_count=3,no_rescue=False)
        for a in r['alternatives']:a['true_benefit']=a['delta_q']>0
        s=summarize([r]);self.assertIn('5.0',s['by_offset_mm'])
        self.assertEqual(s['by_native_quality_count']['3']['n'],1)


class P1MatchedGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_analytic_cache_matches_original_predicted_reader(self):
        f=fixture(n=2);clean=f['pred_depth'].copy();clean.fill(.465)
        a=make_cache_frame(f,clean,.03,.002)
        self.assertEqual(a['predicted_physics'].shape,(2,7,75,6))
        self.assertEqual(a['clean_visible_physics'].shape,(2,7,75,6))
        self.assertGreater(np.abs(a['predicted_physics']-a['clean_visible_physics']).max(),.01)
        torch.manual_seed(4);m=PredictedGeometryControl(ModelConfig(8,16,.03,128,2)).eval()
        with torch.no_grad():
            original=m(tensor_scene(f,'cpu'),torch.from_numpy(f['bank']))
            ref=score_cached(m,collate_cache([a]),'predicted','cpu')[0]
        torch.testing.assert_close(original,ref,atol=1e-6,rtol=1e-6)

    def test_clean_reader_does_not_require_model_input_to_contain_teacher(self):
        f=fixture(n=2);clean=f['pred_depth'].copy();clean.fill(.46)
        cached=make_cache_frame(f,clean,.03,.002)
        self.assertNotIn('visible_depth',cached)
        self.assertNotIn('target_xyz',cached)
        self.assertEqual(set(k for k in cached if k.endswith('physics')),
                         {'predicted_physics','clean_visible_physics'})

    def test_batch4_is_same_as_independent_forward_and_gradient(self):
        f=fixture(n=2);items=[]
        for d in (.46,.47,.48,.49):
            depth=f['pred_depth'].copy();depth.fill(d)
            items.append(make_cache_frame(f,depth,.03,.002))
        torch.manual_seed(33);m=PredictedGeometryControl(ModelConfig(8,16,.03,128,2))
        batch=collate_cache(items)
        batched=score_cached(m,batch,'clean_visible','cpu')
        solo=torch.stack([score_cached(m,collate_cache([x]),'clean_visible','cpu')[0] for x in items])
        torch.testing.assert_close(batched,solo,atol=1e-6,rtol=0)
        counts=torch.as_tensor(batch['quality_count']).long()
        a,_=quality_loss(batched,counts)
        grads=torch.autograd.grad(a,tuple(m.parameters()),retain_graph=True)
        loss=sum(quality_loss(solo[i],counts[i])[0] for i in range(4))/4
        grad2=torch.autograd.grad(loss,tuple(m.parameters()))
        for g,h in zip(grads,grad2):torch.testing.assert_close(g,h,atol=1e-6,rtol=1e-5)

    def test_cache_source_hashes_are_strict_and_test_forbidden(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);tr,va,cfg=synthetic_sources(root)
            raw=__import__('shared_geometry.data',fromlist=['read_index']).read_index(tr,'train',True)
            frame=__import__('shared_geometry.data',fromlist=['read_frame']).read_frame(raw['entries'][0])
            depth,prov=validated_target(raw,raw['entries'][0],frame,TargetLoader(cfg))
            self.assertEqual(prov['source'],'rendered_visible')
            np.save(Path(prov['depth_path']),np.full_like(depth,.9))
            with self.assertRaises(ValueError):validated_target(raw,raw['entries'][0],frame,TargetLoader(cfg))
            bad=dict(raw,partition='test')
            with self.assertRaises(ValueError):validated_target(bad,raw['entries'][0],frame,TargetLoader(cfg))

    def test_smoke_build_train_compare_and_reuse(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);tr,va,cfg=synthetic_sources(root)
            manifests=[]
            for part,source in (('train',tr),('validation',va)):
                args=SimpleNamespace(index=str(source),targets=str(cfg),output=str(root/'cache'/part),
                                     resume=False,smoke=True,limit=0)
                build(args)
                args.resume=True;build(args)
                idx=read_cache_index(root/'cache'/part/'index.json',part,True)
                self.assertEqual(len(idx['entries']),2)
                manifests.append(str(root/'cache'/part/'index.json'))
                self.assertEqual(len(load_cached(idx['entries'][0])['quality_count']),2)
            results={}
            for arm in ('predicted','clean_visible'):
                args=SimpleNamespace(arm=arm,train_index=manifests[0],val_index=manifests[1],
                    output=str(root/arm),device='cpu',batch_size=2,accumulate=1,val_batch_size=2,
                    epochs=2,hidden=16,seed=7,smoke=True,cpu_threads=1,resume=False,
                    lr=1e-3,weight_decay=1e-4)
                fit(args);args.resume=True;fit(args)
                ck=torch.load(root/arm/'last.pt',weights_only=False)
                self.assertEqual(ck['updates'],2)
                results[arm]=str(root/arm/'best.pt')
            from shared_geometry.experiment import fit as original_fit
            original_fit(SimpleNamespace(mode='h2',epochs=1,batch_size=2,accumulate=1,
                val_batch_size=2,train_index=str(tr),val_index=str(va),
                output=str(root/'h2'),smoke=True,resume=False,hidden=16,seed=7,
                device='cpu',cpu_threads=1,lr=1e-3,weight_decay=1e-4,geometry_weight=1.))
            cp=SimpleNamespace(index=manifests[1],predicted_weight=results['predicted'],
                clean_weight=results['clean_visible'],device='cpu',batch_size=2,
                bootstrap=30,seed=0,smoke=True,output=str(root/'compare.json'),
                h2_weight=str(root/'h2'/'best.pt'),original_val_index=str(va))
            compare(cp)
            r=__import__('shared_geometry.data',fromlist=['load_signed']).load_signed(cp.output)
            self.assertIn('paired_clean_minus_predicted',r)
            self.assertIn('learned_h2',r['summary'])
            self.assertIn('clean_visible_minus_learned_h2',r['descriptive_learned_h2_contrasts'])
            self.assertTrue(r['privileged_clean_visible'])
            self.assertTrue(r['official_ap_not_run'])


if __name__=='__main__':unittest.main()