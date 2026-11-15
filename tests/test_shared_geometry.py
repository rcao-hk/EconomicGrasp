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

from shared_geometry.geometry import (VisibleSurfaceTeacher, align_depth, action_bank,
                                       support_points, relative_coordinates)
from shared_geometry.model import (ModelConfig, SharedField, FieldGraspModel, LocalControl,
                                  PredictedGeometryControl, quality_loss, geometry_loss)
from shared_geometry.data import (validate_cache, tensor_scene, signed, atomic_json, atomic_npz,
                                 physical_ids, sha256, frame_key, digest, read_index, make_plan)
from shared_geometry.experiment import fit, load_model, diagnose, family_metrics
from shared_geometry.inference import instantiate, infer
from shared_geometry.runtime import FieldRuntime, TargetLoader

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def fixture(n=2, channels=8, seed=0):
    rng = np.random.default_rng(seed)
    K = np.array([[100, 0, 15.5], [0, 100, 15.5], [0, 0, 1]], np.float32)
    depth = np.ones((32, 32), np.float32)*.5
    g = np.zeros((n, 17), np.float32)
    g[:, 0] = .8; g[:, 1:4] = [.06, .02, .02]
    g[:, 4:13] = np.eye(3).flatten(); g[:, 15] = .5; g[:, 16] = -1
    g[:, 13] = np.linspace(-.02, .02, n)
    bank = action_bank(g); support, _ = support_points(torch.from_numpy(bank))
    teacher = VisibleSurfaceTeacher(depth, K)
    x, labels = teacher.sample(96, seed, support.numpy())
    out = dict(feature_map=rng.normal(size=(channels, 8, 8)).astype(np.float32), pred_depth=depth, K=K,
               bank=bank, physical_id=physical_ids(bank.reshape(-1,17)).reshape(n,7),
               quality_count=np.tile([3,4,2,5,1,6,0], (n,1)).astype(np.int64),
               family_id=np.arange(n), local_features=rng.normal(size=(n,12)).astype(np.float32),
               target_xyz=x, **{'target_'+k:v for k,v in labels.items()})
    validate_cache(out)
    return out


def make_indices(root):
    paths = []
    for part, scene in (('train', 0), ('validation', 80)):
        entries = []
        for i, domain in enumerate(('original', 'gntrans')):
            row = dict(partition=part, scene=scene, anno=0, domain=domain, split='train')
            p = root/(frame_key(row)+'.npz'); atomic_npz(p, fixture(seed=i))
            entries.append(dict(row=row, path=str(p), sha256=sha256(p)))
        contract = dict(source={'base_sha256':'b','selector_sha256':'s','config_sha256':'cfg'},
                        plan='synthetic', offsets_m=[0.,-.005,.005,-.01,.01,-.02,.02],tau=.03,band=.002,
                        target_config='test', partition=part)
        idx = signed(dict(schema='shared-visible-geometry-v1',complete=False,partition=part,entries=entries,contract=contract))
        path = root/(part+'.json'); atomic_json(path, idx); paths.append(path)
    return paths


class GeometryTests(unittest.TestCase):
    def test_plane_projective_states_not_occupancy(self):
        K=np.array([[100,0,15],[0,100,15],[0,0,1]],np.float32)
        t=VisibleSurfaceTeacher(np.ones((32,32),np.float32)*.5,K)
        r=t.label(np.array([[0,0,.48],[0,0,.5],[0,0,.52],[10,0,.5]],np.float32))
        self.assertEqual(r['state'].tolist(),[0,1,2,3])
        self.assertAlmostEqual(r['udf'][0],2/3,places=5)
        self.assertAlmostEqual(r['projective'][2],-2/3,places=5)
        self.assertFalse(r['projective_mask'][3])

    def test_missing_depth_unknown_not_free(self):
        a=fixture(); d=a['pred_depth'].copy(); d[15:17,15:17]=0
        r=VisibleSurfaceTeacher(d,a['K']).label(np.array([[0,0,.3]],np.float32))
        self.assertEqual(r['state'][0],3);self.assertFalse(r['projective_mask'][0])

    def test_invalid_depth_throws(self):
        with self.assertRaises(ValueError):VisibleSurfaceTeacher(np.zeros((4,4)),np.eye(3))
        with self.assertRaises(ValueError):VisibleSurfaceTeacher(np.full((4,4),np.nan),np.eye(3))

    def test_intrinsics_alignment_crop(self):
        raw=np.arange(64,dtype=np.float32).reshape(8,8)/100+1
        K=np.array([[10,0,4],[0,10,4],[0,0,1]],np.float32)
        crop=K.copy();crop[0,2]-=2;crop[1,2]-=1
        got=align_depth(raw,K,crop,(4,4))
        np.testing.assert_array_equal(got,raw[1:5,2:6])

    def test_native_bank_and_orthogonal_parameters(self):
        a=fixture();g=a['bank'][:,0];b=action_bank(g)
        np.testing.assert_array_equal(b[:,0],g)
        np.testing.assert_array_equal(b[:,:,:13],np.repeat(g[:,None,:13],7,1))
        dist=np.linalg.norm(b[:,5,13:16]-g[:,13:16],axis=1)
        np.testing.assert_allclose(dist,.02,atol=1e-7)

    def test_support_uses_width_and_camera_transform(self):
        g=torch.from_numpy(fixture()['bank'][:,0])
        world,local=support_points(g)
        torch.testing.assert_close(relative_coordinates(world,g),local,atol=1e-7,rtol=0)
        g2=g.clone();g2[:,1]=.08;_,l2=support_points(g2)
        self.assertTrue(torch.all(l2[...,1].amax(-1)>local[...,1].amax(-1)))

    def test_no_empty_or_changed_bank_silently(self):
        a=fixture();a['bank'][0,2,1]+=.001
        with self.assertRaises(ValueError):validate_cache(a)

    def test_visible_teacher_uses_no_action_identity(self):
        a=fixture();t=VisibleSurfaceTeacher(a['pred_depth'],a['K'])
        x=a['target_xyz']; left=t.label(x)['udf'];right=t.label(x[::-1])['udf'][::-1]
        np.testing.assert_array_equal(left,right)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2);self.a=fixture();self.s=tensor_scene(self.a,'cpu')
        self.cfg=ModelConfig(8,hidden=16,point_chunk=256,family_chunk=2)
        self.m=FieldGraspModel(self.cfg)

    def test_field_shared_and_query_order_independent(self):
        enc=self.m.field.encode(self.s);x=torch.from_numpy(self.a['target_xyz'])
        r=self.m.field.query(enc,self.s,x);r2=self.m.field.query(enc,self.s,x.flip(0)).flip(0)
        torch.testing.assert_close(r,r2,atol=1e-6,rtol=0)
        same=self.m.field.query(enc,self.s,x[:1].expand(12,-1))
        torch.testing.assert_close(same,same[:1].expand_as(same))

    def test_targets_are_not_network_inputs(self):
        a=copy.deepcopy(self.a);a['target_udf'][:]=999;a['quality_count'][:]=0
        torch.testing.assert_close(self.m(self.s,torch.from_numpy(a['bank'])),
                                   self.m(tensor_scene(a,'cpu'),torch.from_numpy(a['bank'])))
        self.assertEqual(set(tensor_scene(a,'cpu')),{'feature_map','pred_depth','K'})

    def test_score_and_candidate_count_cannot_change_field(self):
        g=torch.from_numpy(self.a['bank']);z=self.m(self.s,g)
        other=g.clone();other[...,0]=.01
        torch.testing.assert_close(z,self.m(self.s,other))
        torch.testing.assert_close(z,self.m(self.s,torch.cat((g,g),0))[:len(g)],atol=1e-6,rtol=0)

    def test_gradients_reach_geometry_reader(self):
        bank=torch.from_numpy(self.a['bank']);z=self.m(self.s,bank)
        loss,_=quality_loss(z,torch.from_numpy(self.a['quality_count']));loss.backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in self.m.field.parameters() if p.grad is not None),0)

    def test_stale_coordinates_are_truthful_not_identical(self):
        bank=torch.from_numpy(self.a['bank'])
        fresh=self.m(self.s,bank,'fresh');stale=self.m(self.s,bank,'stale')
        torch.testing.assert_close(fresh[:,0],stale[:,0],atol=1e-6,rtol=0)
        self.assertFalse(torch.allclose(fresh[:,1:],stale[:,1:],atol=1e-7,rtol=0))

    def test_h1_h2_init_identical(self):
        torch.manual_seed(9);a=FieldGraspModel(self.cfg)
        torch.manual_seed(9);b=FieldGraspModel(self.cfg)
        for k in a.state_dict():self.assertTrue(torch.equal(a.state_dict()[k],b.state_dict()[k]))

    def test_geometry_loss_valid_mask(self):
        x=torch.from_numpy(self.a['target_xyz']);r=self.m.field.query(self.m.field.encode(self.s),self.s,x)
        t={k:torch.as_tensor(self.a['target_'+k]) for k in ('udf','projective','state','udf_mask','projective_mask')}
        loss,stats=geometry_loss(r,t);self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertIn('udf_l1',stats)

    def test_quality_ties_no_direction_guard(self):
        z=torch.randn(2,7,requires_grad=True);q=torch.zeros((2,7),dtype=torch.long)
        loss,s=quality_loss(z,q);self.assertEqual(s['pair'].item(),0)
        loss.backward();self.assertTrue(torch.isfinite(z.grad).all())

    def test_local_control_zero_and_bound(self):
        m=LocalControl(12,16);g=torch.from_numpy(self.a['bank']);f=torch.from_numpy(self.a['local_features'])
        z=m(f,g);old=torch.logit(g[:,:,0])
        torch.testing.assert_close(z,old)
        with torch.no_grad():m.net[-1].bias.fill_(999)
        self.assertLessEqual((m(f,g)-old).abs().max(),.200001)

    def test_analytic_control_no_learned_field(self):
        m=PredictedGeometryControl(self.cfg)
        self.assertFalse(any(k.startswith('field.') for k in m.state_dict()))
        out=m(self.s,torch.from_numpy(self.a['bank']));self.assertEqual(out.shape,(2,7))


class ContractAndPipelineTests(unittest.TestCase):
    def test_instantiation_exact_native_and_actual_move(self):
        a=fixture();bank=a['bank'];s=np.zeros((2,7));s[:,2]=1
        native,_=instantiate(bank,s,s,'native');move,choice=instantiate(bank,s,s,'move_keep_priority')
        np.testing.assert_array_equal(native,bank[:,0]);np.testing.assert_array_equal(move[:,1:13],native[:,1:13])
        self.assertTrue(np.all(choice==2));self.assertFalse(np.array_equal(move[:,13:16],native[:,13:16]))
        np.testing.assert_array_equal(move[:,0],native[:,0])

    def test_family_metric_regret_identity(self):
        q=fixture()['quality_count'];s=-np.arange(7)[None].repeat(2,axis=0)
        v=family_metrics(s,q)
        self.assertAlmostEqual(v['selected_q']+v['regret'],v['oracle_q'])
        self.assertEqual(v['move'],0)

    def test_teacher_file_alignment_and_explicit_config(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td);a=fixture();np.save(td/'d.npy',a['pred_depth']);np.save(td/'k.npy',a['K'])
            cfg=dict(source='rendered_visible',same_camera_as_native=True,calibration_description='synthetic known camera',
                     depth_template=str(td/'d.npy'),intrinsics_template=str(td/'k.npy'),depth_scale_to_m=1.)
            (td/'cfg.json').write_text(json.dumps(cfg));t=TargetLoader(td/'cfg.json')
            d,p=t.load({},a['K'],a['pred_depth'].shape);np.testing.assert_allclose(d,a['pred_depth'])
            self.assertEqual(p['source'],'rendered_visible')

    def test_formal_cache_rejects_smoke_and_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            tr,va=make_indices(Path(td))
            with self.assertRaises(ValueError):read_index(tr,'train',False)
            x=read_index(tr,'train',True);Path(x['entries'][0]['path']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):read_index(tr,'train',True)

    def test_all_modes_train_diagnose_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);tr,va=make_indices(root)
            for mode in ('local','predicted','h1','h2','h2_stale'):
                args=SimpleNamespace(mode=mode,epochs=1,accumulate=2,train_index=str(tr),val_index=str(va),
                    output=str(root/mode),smoke=True,resume=False,hidden=16,seed=0,device='cpu',cpu_threads=1,
                    lr=.001,weight_decay=.0001,geometry_weight=1.)
                fit(args);m,p=load_model(root/mode/'best.pt','cpu',True)
                self.assertEqual(p['mode'],mode)
                with self.assertRaises(ValueError):load_model(root/mode/'best.pt','cpu',False)
                diagnose(SimpleNamespace(index=str(va),weight=str(root/mode/'best.pt'),device='cpu',smoke=True,output=str(root/(mode+'.json'))))
                args.resume=True;fit(args)
                self.assertEqual(torch.load(root/mode/'last.pt',weights_only=False)['updates'],1)

    def test_infer_recomputes_sensor_masks_and_never_labels(self):
        a=fixture(n=1024);native=a['bank'][:,0].copy()
        source={'base_sha256':'b','selector_sha256':'s','config_sha256':'cfg'}
        class Provider:
            provenance=source
            calls=[]
            def snapshot(self,row,seed):
                return dict(grasps=native,features=a['local_features'],feature_map=a['feature_map'],pred_depth=a['pred_depth'],K=a['K']),{}
            def sensor_keep(self,row,g):
                self.calls.append(g.copy());return np.ones(len(g),bool)
            def label(self,*args):raise AssertionError('Test inference must not request GT labels')
        class Dummy(torch.nn.Module):pass
        def scores(*args,**kw):
            s=torch.zeros((1024,7));s[:,2]=2;return s,None
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);plan=make_plan();atomic_json(root/'plan.json',plan);(root/'w.pt').write_bytes(b'w')
            proto=dict(mode='h2',source=source,plan=plan['fingerprint'],offsets_m=[0.,-.005,.005,-.01,.01,-.02,.02])
            args=SimpleNamespace(plan=str(root/'plan.json'),weight=str(root/'w.pt'),device='cpu',limit=1,
                reference_root='',torch_threads=1,output=str(root/'infer'),split='test_seen',seed=0)
            provider=Provider()
            with patch('shared_geometry.inference.create_runtime',return_value=provider),patch('shared_geometry.inference.load_model',return_value=(Dummy(),proto)),patch('shared_geometry.inference.predict',side_effect=scores):
                infer(args)
            self.assertEqual(len(provider.calls),4) # native plus three changed policies
            self.assertFalse(np.array_equal(provider.calls[0][:,13:16],provider.calls[1][:,13:16]))

    def test_export_index_resume_mock_provider(self):
        from shared_geometry.data import export, index
        a=fixture(n=1024)
        class Provider:
            provenance={'base_sha256':'b','selector_sha256':'s','config_sha256':'cfg'}
            def snapshot(self,row,seed):
                return dict(grasps=a['bank'][:,0],features=a['local_features'],feature_map=a['feature_map'],
                            pred_depth=a['pred_depth'],K=a['K']),{'native_decode_max_abs':0.,'native_reader_replay_max_abs':0.}
            def label(self,row,g):return np.full(len(g),3,np.int64)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);atomic_json(root/'plan.json',make_plan())
            np.save(root/'d.npy',a['pred_depth']);np.save(root/'k.npy',a['K'])
            target=dict(source='rendered_visible',same_camera_as_native=True,calibration_description='test aligned',
                        depth_template=str(root/'d.npy'),intrinsics_template=str(root/'k.npy'),depth_scale_to_m=1.)
            (root/'target.json').write_text(json.dumps(target))
            args=SimpleNamespace(plan=str(root/'plan.json'),partition='train',shards=1,shard=0,domain='original',
                                 limit=1,output=str(root/'cache'),targets=str(root/'target.json'),seed=0,
                                 families=2,field_points=96,tau=.03,band=.002,resume=False)
            with patch('shared_geometry.data.create_runtime',return_value=Provider()):export(args)
            args.resume=True
            with patch('shared_geometry.data.create_runtime',return_value=Provider()):export(args)
            ix=SimpleNamespace(plan=args.plan,cache=args.output,partition='train',output=str(root/'idx.json'),smoke=True)
            index(ix);loaded=read_index(ix.output,'train',True)
            self.assertEqual(len(loaded['entries']),1)
            args.tau=.04
            with patch('shared_geometry.data.create_runtime',return_value=Provider()):
                with self.assertRaises(ValueError):export(args)

    def test_reader_hook_cleanup_after_exception(self):
        class Reader(torch.nn.Module):
            def forward(self,*args,**kwargs):return args[0]
        reader=Reader()
        base=SimpleNamespace(model=SimpleNamespace(kview_grasp_module=SimpleNamespace(decoder=SimpleNamespace(reader=reader))),
                             snapshot=lambda *a: (_ for _ in ()).throw(RuntimeError('expected')),
                             label=lambda *a:None,sensor_keep=lambda *a:None,provenance={})
        runtime=FieldRuntime(base)
        with self.assertRaises(RuntimeError):runtime.snapshot({},0)
        self.assertFalse(reader._forward_pre_hooks)

    def test_cli_and_shell(self):
        for command in ('prepare','bind','export','index','train','diagnose','infer','eval','summarize','paired'):
            p=subprocess.run([sys.executable,'-m','shared_geometry.cli',command,'--help'],cwd=ROOT,capture_output=True,text=True,timeout=20)
            self.assertEqual(p.returncode,0,p.stderr)
        p=subprocess.run(['bash','-n','scripts/shared_geometry/run.sh'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)


if __name__=='__main__':unittest.main()
