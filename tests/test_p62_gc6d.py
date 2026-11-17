"""CPU synthetic tests. No pretrained models, CAD labels or GPU results are claimed."""
from __future__ import annotations
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as A
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from shared_geometry import p62_gc6d as gen
from shared_geometry.p62_gc6d_contract import (OFFSETS, CAMERAS, FIELDS, image_id,
    physical_ids, choose_families, quality_counts, validate_frame, validate_bank,
    atomic_json, atomic_npz, signed, load_signed, sha256, check_lineage, cad_paths)
from shared_geometry.p62_gc6d_runtime import (read_rgb_frame, no_tf32, FrozenCorrector,
                                             make_physics, ExactGC6D, native_dexnet_object,
                                             _same_serialized_state)
from shared_geometry.geometry import action_bank, support_points


def native(n=1024):
    g = np.zeros((n, 17), np.float32)
    g[:, 0] = np.linspace(1, .1, n)
    g[:, 1:4] = [.05, .02, .02]
    g[:, 4:13] = np.eye(3, dtype=np.float32).reshape(1, 9)
    g[:, 13] = np.arange(n) * 1e-5
    g[:, 15] = .5; g[:, 16] = -1
    return g


def snap():
    return dict(grasps=native(), feature_map=np.ones((2, 2, 2), np.float32),
                pred_depth=np.full((8, 8), .5, np.float32),
                K=np.array([[20, 0, 4], [0, 20, 4], [0, 0, 1]], np.float32))


class ContractTests(unittest.TestCase):
    def test_camera_annotation_is_not_image_id(self):
        for camera, offset in CAMERAS.items():
            self.assertEqual(image_id(0, camera), offset)
            self.assertEqual(image_id(12, camera), 48+offset)
        for bad in (-1, 13, 2.5):
            with self.assertRaises(ValueError): image_id(bad, 'realsense-d435')

    def test_bank_uses_euclidean_camera_ray_not_insertion(self):
        g = native(3); g[:, 13] = .2
        b = action_bank(g); validate_bank(b)
        self.assertTrue(np.array_equal(b[:, 0], g))
        np.testing.assert_allclose(np.linalg.norm(b[:, 1, 13:16]-g[:, 13:16], axis=1), .005, atol=1e-7)
        self.assertTrue(np.array_equal(b[..., 1:13], np.repeat(g[:, None, 1:13], 7, axis=1)))

    def test_offset_reorder_fails(self):
        b = action_bank(native(2)); b[:, [1, 2]] = b[:, [2, 1]]
        with self.assertRaises(ValueError): validate_bank(b)

    def test_physical_hash_ignores_score_not_geometry(self):
        g = native(2); old = physical_ids(g)
        g[:, 0] = 0; np.testing.assert_array_equal(old, physical_ids(g))
        g[:, 13] += .001; self.assertFalse(np.array_equal(old, physical_ids(g)))

    def test_exact_count_rounding_and_failed_actions(self):
        f = np.array([-1, .2, .4, .6, .8, 1., 1.2], np.float32)
        np.testing.assert_array_equal(quality_counts(f, np.zeros(7, bool)), [0, 6, 5, 4, 3, 2, 1])
        np.testing.assert_array_equal(quality_counts(f, np.ones(7, bool)), np.zeros(7, int))
        with self.assertRaises(ValueError): quality_counts(np.array([.3]), np.zeros(1, bool))

    def test_sampling_is_repeatable_and_not_q_based(self):
        x = native()[:, 0]
        a = choose_families(x, 32, 0)
        np.testing.assert_array_equal(a, choose_families(x, 32, 0))
        self.assertEqual(len(set(a)), 32); self.assertTrue((a < 128).sum() >= 16)

    def test_seven_field_output_and_pose_link(self):
        s = snap(); b = action_bank(s['grasps'][:2]); d = s['pred_depth']+.001
        a = make_physics(s, d, b, .03, .002)
        a.update(physical_id=physical_ids(b.reshape(-1, 17)).reshape(2, 7),
                 family_id=np.arange(2), quality_count=np.zeros((2, 7), np.int64))
        validate_frame(a, b)
        self.assertEqual(set(a), FIELDS)
        a['quality_count'] = a['quality_count'].astype(np.float32)
        with self.assertRaises(ValueError): validate_frame(a, b)

    def test_clean_geometry_and_malformed_state_rejected(self):
        s = snap(); b = action_bank(s['grasps'][:2])
        a = make_physics(s, s['pred_depth'], b, .03, .002)
        a.update(physical_id=physical_ids(b.reshape(-1, 17)).reshape(2, 7),
                 family_id=np.arange(2), quality_count=np.zeros((2, 7), np.int64))
        bad = dict(a, clean_visible_physics=a['predicted_physics'])
        with self.assertRaises(ValueError): validate_frame(bad, b)
        a['predicted_physics'][0, 0, 0, 2:] = 0
        with self.assertRaises(ValueError): validate_frame(a, b)

    def test_trusted_dex_cache_preference_is_read_only(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self.assertEqual([p.suffix for p in cad_paths(root, 3)], ['.ply', '.obj', '.sdf'])
            (root/'dex_models').mkdir(); (root/'dex_models/obj_000003.pkl').write_bytes(b'test')
            self.assertEqual([p.suffix for p in cad_paths(root, 3)], ['.ply', '.pkl'])


class RGBAndPrecisionTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_rgb_loader_never_requires_depth_or_gt(self):
        with tempfile.TemporaryDirectory() as t:
            folder = Path(t)/'scenes/000321'; (folder/'rgb').mkdir(parents=True)
            Image.fromarray(np.full((10, 20, 3), 128, np.uint8)).save(folder/'rgb/000002.png')
            info = {'2': dict(cam_K=[20,0,10,0,20,5,0,0,1], cam_R_w2c=np.eye(3).ravel().tolist())}
            (folder/'scene_camera.json').write_text(json.dumps(info))
            x, audit = read_rgb_frame(t, dict(scene=321,anno=0), 'realsense-d435', [2,1,18,9])
            self.assertEqual(x['img'].shape, (1,3,448,448))
            self.assertFalse(audit['sensor_depth_used']); self.assertFalse(audit['gt_mask_used'])
            self.assertAlmostEqual(x['K'][0,0,2].item(), 224)
            self.assertEqual(set(x), {'img','K','camera_pose_vec','camera_gravity_vec','camera_tilt_deg'})

    def test_tf32_restored_even_on_failure(self):
        old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        try:
            with self.assertRaises(RuntimeError):
                with no_tf32():
                    self.assertFalse(torch.backends.cudnn.allow_tf32)
                    raise RuntimeError('synthetic')
            self.assertTrue(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old

    def test_corrector_reproduces_feature_storage_quantization(self):
        class M(torch.nn.Module):
            def forward(self, x):
                self.last = x['feature_map'].clone()
                self.asserted_no_tf32 = not torch.backends.cudnn.allow_tf32
                return x['pred_depth'] + x['feature_map'].mean((1,2,3))[:,None,None]*.001
        c = FrozenCorrector.__new__(FrozenCorrector); c.model = M(); c.device = 'cpu'
        s = snap(); s['feature_map'] += .0003
        out = c.predict([s,s,s], 'float16')
        np.testing.assert_array_equal(c.model.last[0].numpy(), s['feature_map'].astype(np.float16).astype(np.float32))
        self.assertTrue(c.model.asserted_no_tf32); self.assertEqual(out.shape, (3,8,8))


class DexNetNamespaceTests(unittest.TestCase):
    def test_trusted_namespace_conversion_preserves_arrays_and_cache(self):
        import sys
        from types import ModuleType
        old = 'graspnetAPI.utils.dexnet.bridge_fixture'
        new = 'graspclutter6dAPI.utils.dexnet.bridge_fixture'
        modules = {}
        for path in (old, new):
            for i in range(1, len(path.split('.'))+1):
                name = '.'.join(path.split('.')[:i])
                modules.setdefault(name, ModuleType(name))
        A = type('GraspableObject3D', (), {'__module__': old})
        B = type('GraspableObject3D', (), {'__module__': new})
        modules[old].GraspableObject3D = A
        modules[new].GraspableObject3D = B
        with patch.dict(sys.modules, modules):
            source = A()
            source.data_ = np.arange(24, dtype=np.float32).reshape(2,3,4)
            source.cached_samples = [np.eye(3), {'resolution': .001, 'origin': np.ones(3)}]
            converted = native_dexnet_object(source, B)
            self.assertIsInstance(converted, B)
            np.testing.assert_array_equal(source.data_, converted.data_)
            _same_serialized_state(source, converted)
            converted.data_[0,0,0] = 100
            with self.assertRaises(ValueError): _same_serialized_state(source, converted)

    def test_unrecognized_namespace_fails_closed(self):
        with self.assertRaises(TypeError): native_dexnet_object(A(), dict)


class FakeEU:
    GraspableObject3D = object
    @staticmethod
    def transform_points(points, pose): return points @ pose[:3,:3].T+pose[:3,3]
    @staticmethod
    def compute_closest_points(a,b): return ((a[:,None]-b[None])**2).sum(-1).argmin(-1)
    @staticmethod
    def collision_detection(gs, models, dex, poses, scene, **kw):
        g=gs[0]; col=g[:,14]<0; emp=g[:,14]<-.5
        return [col], [emp], [[row.copy() for row in g]]
    @staticmethod
    def get_grasp_score(g, obj, friction, configs): return .2 if g[13]>0 else .4


class ExactKernelTests(unittest.TestCase):
    def engine(self):
        e=ExactGC6D.__new__(ExactGC6D); e.eu=FakeEU(); e.chunk=1
        e.fc_list=np.array([1.2,1,.8,.6,.4,.2]); e.fc_configs={}
        return e

    def test_interleaved_object_assignments_keep_all_input_order(self):
        g=native(4); g[:,13]=[.8,-.8,.9,-.9]; g[:,14]=[0,0,-.1,0]
        models=[np.array([[-1.,0,.5]]),np.array([[1.,0,.5]])]
        r=self.engine().evaluate_arrays(g,models,[None,None],[np.eye(4),np.eye(4)])
        np.testing.assert_array_equal(r['assigned_obj'],[1,0,1,0])
        np.testing.assert_array_equal(r['quality_count'],[6,5,0,5])
        self.assertEqual(len(r['friction']),4)

    def test_chunk_size_has_no_label_effect(self):
        g=native(5); g[:,13]=[-.8,.8,-.7,.7,.9]
        models=[np.array([[-1.,0,.5]]),np.array([[1.,0,.5]])]
        e=self.engine(); a=e.evaluate_arrays(g,models,[None,None],[np.eye(4)]*2)
        e.chunk=20; b=e.evaluate_arrays(g,models,[None,None],[np.eye(4)]*2)
        for k in a: np.testing.assert_array_equal(a[k],b[k])


class FakeSource:
    provenance={'kind':'synthetic'}
    def __init__(self): self.calls=0
    def snapshot(self,row,seed):
        self.calls+=1
        return snap(), dict(replay={'native_decode_max_abs':0.,'native_reader_replay_max_abs':0.})

class FakeCorrector:
    def __init__(self): self.batches=[]
    def predict(self,snapshots,dtype):
        self.batches.append(len(snapshots)); return np.stack([s['pred_depth']+.001 for s in snapshots])

class FakeEvaluator:
    provenance={'kind':'synthetic'}
    def evaluate(self,row,g):
        f=np.resize(np.array([-1,.2,.4,.6,.8,1,1.2],np.float32),len(g)); c=np.zeros(len(g),bool)
        return dict(friction=f,collision_or_empty=c,empty=c.copy(),assigned_obj=np.zeros(len(g),np.int64)), \
               dict(reference_parity=dict(n_checked=3,n_positive=2,mismatches=0))


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        ro=self.root/'readonly'; ro.mkdir(); data=self.root/'dataset'; data.mkdir()
        def file(name):
            p=ro/name;p.write_text('synthetic immutable '+name);return gen.artifact(p)
        self.assets={k:file(k) for k in ('base','selector','depth_corrector','predicted_reader','corrected_reader')}
        binding=dict(assets=self.assets,selections=[file('selection')],config=file('config'),
            p1_index=file('p1'),depth_index=file('depth'),feature_reference=file('feature'),
            feature_dtype='float16',channels=2,tau=.03,band=.002,source={'synthetic':True})
        lineage=dict(schema='p62-gc6d-lineage-v1',attested_complete=True,reviewer='test',reviewed_at='test',
            holdout_used_for_tuning=False,training_scene_keys=[['graspnet',0]],
            development_scene_keys=[['graspnet',80]],official_test_scene_keys=[['graspnet',100]],
            components={k:dict(sha256=v['sha256'],gc6d_training_scenes=[],gc6d_development_scenes=[])
                        for k,v in self.assets.items()})
        rows=[dict(scene=s,anno=a,domain='original',partition='independent_holdout',split='external',
                   img_id=image_id(a,'realsense-d435'),input_files=[file('input_%d_%d'%(s,a))])
              for s in (301,302) for a in (0,1,2)]
        self.plan=signed(dict(schema='p62-gc6d-plan-v1',binding=binding,lineage=lineage,dataset_root=str(data),
            camera='realsense-d435',rows=rows,seed=0,families=4,offsets_m=list(OFFSETS),batch_size=4,
            crop_box=None,code_hashes={},formal_not_smoke=True))
        self.path=self.root/'plan.json';atomic_json(self.path,self.plan)
        self.args=A(plan=str(self.path),output=str(self.root/'export'),smoke=True,limit=0,
                    parity_report=None,device='cpu',chunk=8,verify_per_frame=3,resume=False)
    def tearDown(self): self.tmp.cleanup()
    def run_export(self):
        source=FakeSource();corrector=FakeCorrector()
        r=gen.build(self.args,source,corrector,FakeEvaluator(),make_physics)
        return r,source,corrector
    def test_full_synthetic_export_b4_tail_and_geometry_audit(self):
        m,src,c=self.run_export()
        self.assertEqual(c.batches,[4,2]);self.assertEqual(m['n_actions'],168)
        self.assertEqual(src.calls,6);self.assertFalse(m['complete'])
        r=gen.audit_manifest(Path(self.args.output)/'smoke_index.json',True,True)
        self.assertEqual(r['n_frames'],6);self.assertEqual(r['n_actions'],168)
    def test_smoke_is_not_admitted_as_holdout(self):
        self.run_export()
        with self.assertRaises(ValueError):gen.audit_manifest(Path(self.args.output)/'smoke_index.json')
    def test_injected_backend_cannot_claim_formal(self):
        self.args.smoke=False
        with self.assertRaises(ValueError):self.run_export()
    def test_resume_does_not_rescore_existing_frames(self):
        m,_,_=self.run_export(); self.args.resume=True
        r,src,c=self.run_export()
        self.assertEqual(src.calls,0);self.assertEqual(c.batches,[]);self.assertEqual(m,r)
    def test_repeated_without_resume_fails(self):
        self.run_export()
        with self.assertRaises(FileExistsError):self.run_export()
    def test_mutated_frame_and_source_rejected(self):
        m,_,_=self.run_export(); Path(m['entries'][0]['path']).write_bytes(b'corrupt')
        with self.assertRaises(ValueError):gen.audit_manifest(Path(self.args.output)/'smoke_index.json',allow_smoke=True)
    def test_holdout_exposure_rejected(self):
        lin=copy.deepcopy(self.plan['lineage']);lin['components']['base']['gc6d_training_scenes']=[301]
        with self.assertRaises(ValueError):check_lineage(lin,self.assets,[301,302])
    def test_plan_requires_all_component_lineage(self):
        lin=copy.deepcopy(self.plan['lineage']);lin['components'].pop('selector')
        with self.assertRaises(ValueError):check_lineage(lin,self.assets,[301])
    def test_nonvacuous_parity_required(self):
        with self.assertRaises(ValueError):gen.check_parity(None,self.plan)
        p=self.root/'parity.json'
        atomic_json(p,signed(dict(schema='p62-graspnet-parity-v1',mismatches=0,n_positive=0,
                                 plan_fingerprint=self.plan['fingerprint'],n_actions=224)))
        with self.assertRaises(ValueError):gen.check_parity(p,self.plan)
    def test_changed_runtime_cannot_resume(self):
        self.run_export();self.args.resume=True
        ev=FakeEvaluator();ev.provenance={'kind':'different'}
        with self.assertRaises(ValueError):gen.build(self.args,FakeSource(),FakeCorrector(),ev,make_physics)
    def test_label_trace_cannot_be_relabelled(self):
        m,_,_=self.run_export(); e=m['entries'][0]
        with np.load(e['raw']['path'],allow_pickle=False) as z:raw=dict(z)
        raw['friction'][:]=-1;atomic_npz(e['raw']['path'],raw);e['raw']=gen.artifact(e['raw']['path'])
        with self.assertRaises(ValueError):gen.verify_entry(e,self.plan)


if __name__=='__main__':unittest.main()
