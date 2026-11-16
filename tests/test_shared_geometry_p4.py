"""CPU P4 contracts, geometry/score sensitivity, real batching and full smoke."""
from __future__ import annotations
import copy
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as Args
import unittest
import numpy as np
import torch
from torch import nn
from shared_geometry.p4_common import (formal_rows, score_physics, freeze_reader,
    finite_correlation, load_signed, atomic_json, signed, sha256)
from shared_geometry.p4_score_diagnostics import matched_masks, margin_attributions, run as diagnose
from shared_geometry.p4_depth_data import (build as build_depth, read_depth_index,
    DepthDataset, collate_depth)
from shared_geometry.p4_depth_baseline import (PlainDepthCorrector, DepthConfig,
    depth_loss, fit, load_corrector, evaluate_actions, predict_depth, support_errors)
from shared_geometry.p1_clean_visible import (build as build_p1, read_cache_index,
    load_cached, collate_cache, score_cached)
from shared_geometry.p2_geometry_diagnostics import fit_shared, compare_shared
from tests.test_shared_geometry_p0_p1 import synthetic_sources

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def setup_sources(root):
    tr,va,cfg=synthetic_sources(root)
    for part,source in (('train',tr),('validation',va)):
        build_p1(Args(index=str(source),targets=str(cfg),output=str(root/'p1'/part),
                      smoke=True,resume=False,limit=0))
        build_depth(Args(index=str(source),targets=str(cfg),output=str(root/'depth'/part),
                        smoke=True,resume=False,limit=0))
    fit_shared(Args(train_index=str(root/'p1/train/index.json'),val_index=str(root/'p1/validation/index.json'),
        output=str(root/'reader'),epochs=1,hidden=8,device='cpu',batch_size=2,val_batch_size=2,
        accumulate=1,lr=1e-3,weight_decay=1e-4,cpu_threads=1,seed=7,resume=False,smoke=True))
    compare_shared(Args(index=str(root/'p1/validation/index.json'),weight=str(root/'reader/best.pt'),
        device='cpu',batch_size=2,bootstrap=20,seed=0,smoke=True,output=str(root/'p2.json')))
    return tr,va,cfg


def train_args(root,name='corrector',**kw):
    a=dict(train_index=str(root/'depth/train/index.json'),val_index=str(root/'depth/validation/index.json'),
        output=str(root/name),device='cpu',epochs=2,batch_size=2,accumulate=1,val_batch_size=2,
        hidden=8,max_correction_m=.05,gradient_weight=.5,lr=1e-3,weight_decay=1e-4,
        workers=0,cpu_threads=1,seed=8,resume=False,smoke=True,stop_after_epoch=0)
    a.update(kw);return Args(**a)


class P40Units(unittest.TestCase):
    def test_masks_equal_height_count_and_xz_matching(self):
        masks=matched_masks(15,4,3)
        x,y,z=np.indices((5,5,3));x=x.ravel();z=z.ravel()
        for name,a in masks.items():
            self.assertTrue(np.all(a.sum(1)==15))
            for h in range(3):self.assertTrue(np.all(a[:,z==h].sum(1)==5))
        for name in ('palm_proxy','lateral_finger_proxy','opening_core_proxy'):
            for src,tgt in zip(masks[name],masks[name+'_xz_matched']):
                for i in range(5):
                    for h in range(3):self.assertEqual(int((src&(x==i)&(z==h)).sum()),int((tgt&(x==i)&(z==h)).sum()))
        np.testing.assert_array_equal(masks['palm_proxy'],masks['palm_proxy_xz_matched'])

    def test_masks_deterministic_invalid_count_rejected(self):
        a,b=matched_masks(9,3,5),matched_masks(9,3,5)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        for count in (0,2,16,24):
            with self.assertRaises(ValueError):matched_masks(count,2,0)

    def test_linear_attribution_exact_with_native_subtraction(self):
        class LinearReader(nn.Module):
            def __init__(self):
                super().__init__();self.point_reader=nn.Linear(9,2,bias=False);self.quality=nn.Linear(7,1,bias=False)
                with torch.no_grad():
                    self.point_reader.weight.zero_();self.point_reader.weight[0,0]=1;self.point_reader.weight[1,1]=2
                    self.quality.weight.zero_();self.quality.weight[0,:2]=torch.tensor([2.,3.])
        m=LinearReader().eval()
        for p in m.parameters():p.requires_grad_(False)
        torch.manual_seed(2);p=torch.rand(2,3,7,75,6);c=torch.rand_like(p)
        local=torch.zeros(2,3,7,75,3);dims=torch.ones(2,3,7,3)
        a=margin_attributions(m,p,c,local,dims,4)
        np.testing.assert_allclose(a['pred_endpoint'].sum((-1,-2)),a['actual_margin_change'],atol=1e-6)
        self.assertLess(float(np.abs(a['completeness_residual']).max()),1e-6)
        self.assertTrue(all(x.grad is None for x in m.parameters()))

    def test_ig_optional_and_correlation_constant(self):
        from shared_geometry.model import PredictedGeometryControl,ModelConfig
        m=PredictedGeometryControl(ModelConfig(1,8));p=torch.zeros(1,1,7,75,6);p[...,2]=1
        c=p.clone();c[...,0]=.2
        r=margin_attributions(m,p,c,torch.zeros(1,1,7,75,3),torch.ones(1,1,7,3),0)
        self.assertNotIn('integrated',r)
        self.assertIsNone(finite_correlation([1,1,1],[1,2,3])['spearman'])

    def test_smoke_sampling_includes_both_rgb_domains(self):
        from shared_geometry.p4_common import select_entries
        entries=[dict(row=dict(domain=d,scene=0,anno=i)) for d in ('original','gntrans') for i in range(4)]
        sample=select_entries(entries,5)
        self.assertEqual([e['row']['domain'] for e in sample[:2]],['original','gntrans'])
        self.assertEqual(len(sample),5)
        self.assertEqual(select_entries(entries,0),entries)

    def test_formal_diagnostics_reject_partial_and_test(self):
        idx=dict(partition='validation',entries=[dict(row=dict(partition='validation',scene=80,anno=8,domain='original'))])
        with self.assertRaises(ValueError):formal_rows(idx,'validation',False)
        formal_rows(idx,'validation',True)
        idx['entries'][0]['row']['scene']=130
        with self.assertRaises(ValueError):formal_rows(idx,'validation',True)


class DepthUnits(unittest.TestCase):
    def inputs(self,b=4):
        torch.manual_seed(4)
        return dict(feature_map=torch.randn(b,8,8,8),pred_depth=torch.ones(b,32,32)*.5)

    def test_initial_identity_and_missing_rays_preserved(self):
        x=self.inputs();x['pred_depth'][:,0,0]=0;m=PlainDepthCorrector(DepthConfig(8,8))
        torch.testing.assert_close(m(x),x['pred_depth'],rtol=0,atol=0)
        with torch.no_grad():m.refine[-1].bias.fill_(100)
        y=m(x);self.assertTrue(torch.all(y[:,0,0]==0));self.assertLessEqual(float((y-x['pred_depth']).abs().max().detach()),.0500001)

    def test_forward_no_target_or_action_shortcut(self):
        m=PlainDepthCorrector(DepthConfig(8,8));x=self.inputs(2)
        y=m(x);polluted=dict(x,visible_depth=torch.ones_like(y)*9,quality_count=torch.zeros(7),bank=torch.randn(8,17))
        torch.testing.assert_close(y,m(polluted))

    def test_true_batch_gradient_and_scene_isolation(self):
        x=self.inputs();m=PlainDepthCorrector(DepthConfig(8,8))
        with torch.no_grad():m.refine[-1].weight.normal_(0,.02)
        batch=m(x);solo=torch.cat([m({k:v[i:i+1] for k,v in x.items()}) for i in range(4)])
        torch.testing.assert_close(batch,solo,atol=1e-6,rtol=1e-6)
        target=torch.ones_like(batch)*.48
        loss,_=depth_loss(batch,target,x['pred_depth'])
        g=torch.autograd.grad(loss,tuple(m.parameters()))
        losses=[]
        for i in range(4):
            y=m({k:v[i:i+1] for k,v in x.items()});l,_=depth_loss(y,target[i:i+1],x['pred_depth'][i:i+1]);losses.append(l)
        g2=torch.autograd.grad(sum(losses)/4,tuple(m.parameters()))
        for a,b in zip(g,g2):torch.testing.assert_close(a,b,atol=1e-6,rtol=2e-5)
        xx={k:v.clone() for k,v in x.items()};xx['feature_map'][1:]*=100
        torch.testing.assert_close(m(xx)[0],batch[0])

    def test_metric_pixel_loss_and_invalid_mask(self):
        d=torch.ones(2,4,4)*.5;t=d-.01;t[:,0,0]=0
        loss,stats=depth_loss(d,t,d,gradient_weight=0)
        self.assertAlmostEqual(float(loss),1.,places=5)
        self.assertAlmostEqual(stats[0]['mae_mm'],10.,places=4)
        with self.assertRaises(ValueError):depth_loss(d,torch.zeros_like(d),d)

    def test_geometry_metrics_same_mask_and_unknown_not_zero(self):
        pred=np.zeros((2,7,75,6),np.float32);pred[...,2]=1
        clean=pred.copy();corr=pred.copy();corr[...,0]=.1
        r=support_errors(pred,corr,clean,.03)
        self.assertAlmostEqual(r['corrected_udf_mae_mm'],3.,places=5)
        # Common action-independent error cancels in the relative metric.
        self.assertAlmostEqual(r['corrected_udf_delta_mae_mm'],0.)
        pred[...,2:]=0;pred[...,5]=1
        self.assertIsNone(support_errors(pred,corr,clean,.03)['corrected_udf_mae_mm'])


class P4EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.tr,cls.va,cls.cfg=setup_sources(cls.root)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_target_index_reuse_no_feature_duplication(self):
        idx=read_depth_index(self.root/'depth/validation/index.json','validation',True)
        sample=DepthDataset(idx)[0];self.assertNotIn('visible_depth',sample['inputs'])
        with np.load(idx['entries'][0]['target_path']) as z:self.assertEqual(z.files,['visible_depth'])
        build_depth(Args(index=str(self.va),targets=str(self.cfg),output=str(self.root/'depth/validation'),smoke=True,resume=True,limit=0))
        with self.assertRaises(ValueError):read_depth_index(self.root/'depth/validation/index.json','validation',False)

    def test_target_corruption_rejected(self):
        root=self.root;src=load_signed(root/'depth/validation/index.json')
        modified=copy.deepcopy(src);modified.pop('fingerprint');modified['entries'][0]['target_sha256']='bad'
        path=root/'bad_index.json';atomic_json(path,signed(modified))
        with self.assertRaises(ValueError):read_depth_index(path,'validation',True)

    def test_p40_frozen_pipeline_and_output_contract(self):
        root=self.root
        a=Args(index=str(root/'p1/validation/index.json'),weight=str(root/'reader/best.pt'),
            expected_p2=str(root/'p2.json'),output=str(root/'p40'),device='cpu',batch_size=2,
            points=15,repeats=2,ig_steps=2,bootstrap=20,seed=0,limit=0,smoke=True)
        before=sha256(a.weight);r=diagnose(a)
        self.assertEqual(r['n_families'],4);self.assertEqual(r['n_pairs'],24)
        self.assertTrue(r['published_p2_gain_parity_checked']);self.assertEqual(before,sha256(a.weight))
        self.assertTrue(r['mask_design']['palm_matched_is_self'])
        with gzip.open(root/'p40/pairs.jsonl.gz','rt') as f:
            pair=json.loads(next(f));self.assertIn('integrated',pair['attribution'])
        with self.assertRaises(FileExistsError):diagnose(a)

    def test_training_resume_exact_and_refuses_protocol_change(self):
        root=self.root
        a=train_args(root,'uninterrupted');fit(a)
        b=train_args(root,'resumed',stop_after_epoch=1);fit(b);b.stop_after_epoch=0;b.resume=True;fit(b)
        x=torch.load(root/'uninterrupted/last.pt',weights_only=False)
        y=torch.load(root/'resumed/last.pt',weights_only=False)
        self.assertEqual(x['updates'],2);self.assertEqual(y['updates'],2)
        for name in x['model']:torch.testing.assert_close(x['model'][name],y['model'][name],rtol=0,atol=0)
        b.gradient_weight=.25
        with self.assertRaises(ValueError):fit(b)

    def test_train_evaluate_predict_no_reader_update(self):
        root=self.root;a=train_args(root,'eval_corrector',epochs=1);fit(a)
        before=sha256(root/'reader/best.pt')
        e=Args(depth_index=str(root/'depth/validation/index.json'),p1_index=str(root/'p1/validation/index.json'),
            original_index=str(self.va),weight=str(root/'eval_corrector/best.pt'),shared_weight=str(root/'reader/best.pt'),
            expected_p2=str(root/'p2.json'),output=str(root/'depth_eval'),device='cpu',batch_size=2,
            bootstrap=20,seed=0,limit=0,smoke=True)
        r=evaluate_actions(e)
        self.assertEqual(set(r['scores']),{'predicted','corrected','clean_visible'})
        self.assertTrue(r['published_p2_gain_parity_checked']);self.assertFalse(r['reader_updated'])
        self.assertLessEqual(r['analytical_predicted_parity_max_abs'],1e-6)
        self.assertEqual(before,sha256(e.shared_weight))
        source=read_depth_index(e.depth_index,'validation',True)['entries'][0]['source_path']
        out=root/'rgb_only_depth.npz'
        predict_depth(Args(weight=e.weight,inputs=source,output=str(out),device='cpu',smoke=True))
        self.assertTrue(load_signed(out.with_suffix('.json'))['no_teacher_access'])
        with self.assertRaises(FileExistsError):evaluate_actions(e)

    def test_real_batch_and_accumulation_weight_parity(self):
        root=self.root
        fit(train_args(root,'physical_b4',batch_size=4,accumulate=1))
        fit(train_args(root,'micro_b1',batch_size=1,accumulate=4))
        x=torch.load(root/'physical_b4/last.pt',weights_only=False)
        y=torch.load(root/'micro_b1/last.pt',weights_only=False)
        self.assertEqual(x['updates'],y['updates'])
        for key in x['model']:
            torch.testing.assert_close(x['model'][key],y['model'][key],atol=3e-6,rtol=2e-5)

    def test_epoch0_depth_selection_can_keep_baseline(self):
        root=self.root
        fit(train_args(root,'no_change',epochs=1,lr=1e-20))
        selection=load_signed(root/'no_change/selection.json')
        self.assertEqual(selection['selected_epoch'],0)
        self.assertFalse(selection['depth_admitted'])
        self.assertTrue(selection['complete'])

    def test_partial_smoke_does_not_claim_full_parity(self):
        root=self.root
        a=Args(index=str(root/'p1/validation/index.json'),weight=str(root/'reader/best.pt'),
            expected_p2=str(root/'p2.json'),output=str(root/'p40_partial'),device='cpu',batch_size=1,
            points=3,repeats=1,ig_steps=0,bootstrap=10,seed=2,limit=1,smoke=True)
        r=diagnose(a);self.assertFalse(r['published_p2_gain_parity_checked']);self.assertEqual(r['n_frames'],1)


class CLITests(unittest.TestCase):
    def test_help_and_shell_syntax(self):
        for args in ([sys.executable,'-m','shared_geometry.p4_score_diagnostics','--help'],
                     *[[sys.executable,'-m','shared_geometry.p4_depth_baseline',stage,'--help'] for stage in ('build','train','evaluate','predict')]):
            result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
        script=ROOT/'scripts/shared_geometry/run_p4.sh'
        if script.exists():
            result=subprocess.run(['bash','-n',str(script)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
