"""CPU synthetic P2 diagnostics and common-reader controls. No test-set access."""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
import torch
from shared_geometry.data import load_signed
from shared_geometry.p0_relative_decision import metrics_for_family
from shared_geometry.p1_clean_visible import build, fit as fit_p1, read_cache_index, load_cached, collate_cache
from shared_geometry.p2_geometry_diagnostics import (
    _family_records, _pair_stats, _source_contrast, _describe, diagnostic,
    fit_shared, compare_shared, load_shared, _evaluate_models, _check_indices)
from tests.test_shared_geometry_p0_p1 import synthetic_sources


class P2NativeRelative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_native_relative_family_definition(self):
        q=np.asarray([6,5,4,6,3,2,1],dtype=np.int64)
        z=np.asarray([1.,2.,0.,-.5,0.,0.,0.],dtype=np.float32)
        from shared_geometry.geometry import DEFAULT_OFFSETS_M
        phys=np.zeros((1,1,7,75,6),np.float32)
        phys[...,0]=.5;phys[...,2]=1
        arr=dict(quality_count=q.reshape(1,1,7),family_id=np.asarray([[7]]),
                 predicted_physics=phys,clean_visible_physics=phys)
        row=[dict(scene=80,domain='original',anno=8)]
        r=_family_records(z.reshape(1,1,7),arr,row,'predicted',DEFAULT_OFFSETS_M)[0]
        expected=metrics_for_family(z,q,DEFAULT_OFFSETS_M)
        self.assertEqual(r['choice'],expected['choice'])
        self.assertAlmostEqual(r['gain'],expected['gain'])
        self.assertTrue(r['harm'])
        self.assertEqual(r['native_count'],6)
        self.assertEqual(r['family'],7)
        self.assertIn('native_full_count6',_pair_stats([r]))

    def test_paired_record_identity_prevents_mismatch(self):
        from shared_geometry.geometry import DEFAULT_OFFSETS_M
        z=np.asarray([1.,0,0,0,0,0,0])
        q=np.asarray([3,4,4,4,4,4,4])
        a=metrics_for_family(z,q,DEFAULT_OFFSETS_M)
        a.update(scene=80,domain='original',anno=1,family=2,native_count=3,no_rescue=True)
        b=dict(a,family=3)
        with self.assertRaises(ValueError):_source_contrast([a],[b],20,0)
        compare=_source_contrast([a],[dict(a)],20,0)
        self.assertAlmostEqual(compare['gain']['mean'],0)

    def test_missing_or_mixed_partition_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);tr,va,cfg=synthetic_sources(root)
            with self.assertRaises(Exception):_check_indices(root/'absent.json',va,False)


class P2EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_matched_diagnostic_and_shared_training_are_isolated(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);tr,va,cfg=synthetic_sources(root)
            indices={}
            for part,src in [('train',tr),('validation',va)]:
                build(SimpleNamespace(index=str(src),targets=str(cfg),
                     output=str(root/'cache'/part),resume=False,smoke=True,limit=0))
                indices[part]=str(root/'cache'/part/'index.json')
            trained={}
            for arm in ('predicted','clean_visible'):
                args=SimpleNamespace(arm=arm,train_index=indices['train'],val_index=indices['validation'],
                    output=str(root/'p1'/arm),device='cpu',batch_size=2,accumulate=1,val_batch_size=2,
                    epochs=1,hidden=16,seed=7,smoke=True,cpu_threads=1,resume=False,
                    lr=1e-3,weight_decay=1e-4)
                fit_p1(args);trained[arm]=str(Path(args.output)/'best.pt')
            diag=diagnostic(SimpleNamespace(index=indices['validation'],predicted_weight=trained['predicted'],
                clean_weight=trained['clean_visible'],output=str(root/'diagnostic.json'),device='cpu',
                batch_size=2,bootstrap=20,seed=0,smoke=True,limit=0))
            self.assertEqual(diag['study'],'matched_p1_native_relative')
            self.assertEqual(len(diag['per_family']['predicted']),4)
            self.assertEqual(len(diag['per_family']['clean_visible']),4)
            self.assertIn('native_full_count6',diag['native_relative_subgroups']['predicted'])
            self.assertEqual(set(diag['models']),{'predicted','clean_visible'})
            args=SimpleNamespace(train_index=indices['train'],val_index=indices['validation'],
                 output=str(root/'shared'),epochs=2,hidden=16,device='cpu',batch_size=2,
                 val_batch_size=2,accumulate=1,lr=1e-3,weight_decay=1e-4,cpu_threads=1,
                 seed=7,resume=False,smoke=True)
            fit_shared(args)
            ck=torch.load(Path(args.output)/'last.pt',map_location='cpu',weights_only=False)
            self.assertEqual(ck['updates'],2)
            self.assertEqual(len(ck['history']),2)
            self.assertEqual(ck['protocol']['study'],'shared_reader')
            args.resume=True;fit_shared(args)
            comparison=compare_shared(SimpleNamespace(index=indices['validation'],
                 weight=str(Path(args.output)/'best.pt'),device='cpu',batch_size=2,
                 bootstrap=20,seed=0,smoke=True,output=str(root/'shared_compare.json')))
            self.assertEqual(comparison['study'],'shared_reader_fixed_weights')
            self.assertIn('gain',comparison['clean_minus_predicted'])
            self.assertEqual(len(comparison['per_family']['predicted']),4)
            self.assertTrue(comparison['privileged_clean_visible'])
            self.assertTrue(comparison['official_ap_not_run'])
            with self.assertRaises(FileExistsError):compare_shared(SimpleNamespace(index=indices['validation'],
                 weight=str(Path(args.output)/'best.pt'),device='cpu',batch_size=2,
                 bootstrap=20,seed=0,smoke=True,output=str(root/'shared_compare.json')))

if __name__=='__main__': unittest.main()