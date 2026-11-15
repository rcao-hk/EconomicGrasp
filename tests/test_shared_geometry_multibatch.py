"""Real multi-image forward/backward, scene isolation and end-to-end regression.

All tests are CPU-only and use synthetic aligned camera-frame inputs. These do
not substitute for the 117 GPU/CAD integration gate documented in the runbook.
"""
from __future__ import annotations
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from tests.test_shared_geometry import fixture, make_indices
from shared_geometry.data import (collate_inputs, collate_frames, signed, atomic_json,
                                  atomic_npz, sha256, frame_key, make_plan, tensor_scene)
from shared_geometry.model import (ModelConfig, FieldGraspModel, LocalControl,
                                   PredictedGeometryControl, geometry_loss, quality_loss)
from shared_geometry.experiment import predict, fit, load_model, evaluate_index
from shared_geometry.inference import infer

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def frames(n=4, families=3):
    result = [fixture(n=families, seed=i+37) for i in range(n)]
    # Make geometries, intrinsics and image evidences genuinely distinct.
    for i, a in enumerate(result):
        a['pred_depth'] = a['pred_depth'] * (1+.04*i)
        a['K'] = a['K'].copy()
        a['K'][0, 0] += i*5
        a['quality_count'] = np.roll(a['quality_count'], i, axis=1)
        a['target_udf_mask'] = a['target_udf_mask'].copy()
        a['target_udf_mask'][i::5] = False
        a['target_projective_mask'] = a['target_projective_mask'].copy()
        a['target_projective_mask'][i::7] = False
    return result


def model_for(mode, cfg=None):
    cfg = cfg or ModelConfig(8, 16, .03, 41, 2)
    return (LocalControl(12, cfg.hidden) if mode == 'local' else
            PredictedGeometryControl(cfg) if mode == 'predicted' else FieldGraspModel(cfg))


def full_loss(model, mode, batch, device='cpu'):
    z, field = predict(model, mode, batch, device)
    loss, _ = quality_loss(z, torch.as_tensor(batch['quality_count']).long())
    if mode in ('h2', 'h2_stale'):
        enc, scene = field
        xyz = torch.as_tensor(batch['target_xyz'], dtype=torch.float32)
        raw = model.field.query(enc, scene, xyz)
        target = {k: torch.as_tensor(batch['target_'+k])
                  for k in ('udf', 'projective', 'state', 'udf_mask', 'projective_mask')}
        geo, _ = geometry_loss(raw, target)
        loss = loss + geo
    return loss


class ForwardBatchParity(unittest.TestCase):
    def test_collation_labels_are_excluded_from_model_inputs(self):
        batch = collate_frames(frames(4))
        self.assertEqual(batch['feature_map'].shape, (4,8,8,8))
        self.assertEqual(batch['bank'].shape, (4,3,7,17))
        self.assertEqual(batch['target_xyz'].shape, (4,96,3))
        self.assertEqual(batch['quality_count'].shape, (4,3,7))
        inputs = collate_inputs(frames(4))
        self.assertNotIn('quality_count', inputs)
        self.assertNotIn('target_xyz', inputs)
        self.assertNotIn('target_state', inputs)
        self.assertEqual(set(inputs), {'feature_map','pred_depth','K','bank','local_features'})

    def test_b1_b2_b4_parity_all_five_modes(self):
        items = frames(4)
        for mode in ('local','predicted','h1','h2','h2_stale'):
            with self.subTest(mode=mode):
                torch.manual_seed(24)
                model = model_for(mode).eval()
                expected = torch.stack([predict(model,mode,x,'cpu')[0] for x in items])
                for bs in (1,2,4):
                    chunks = []
                    for start in range(0,4,bs):
                        inp = collate_inputs(items[start:start+bs]); got,_=predict(model,mode,inp,'cpu')
                        self.assertEqual(got.shape,(min(bs,4-start),3,7))
                        chunks.append(got)
                    torch.testing.assert_close(torch.cat(chunks), expected, rtol=1e-5, atol=3e-6)

    def test_scene_perturbation_cannot_change_other_scene(self):
        batch = collate_inputs(frames(2)); model = model_for('h2')
        z0,_=predict(model,'h2',batch,'cpu')
        bad=copy.deepcopy(batch)
        bad['feature_map'][1].fill(100)
        bad['pred_depth'][1].fill(.9)
        bad['K'][1,0,0]*=2
        z1,_=predict(model,'h2',bad,'cpu')
        torch.testing.assert_close(z0[0],z1[0],atol=1e-7,rtol=0)
        self.assertGreater((z1[1]-z0[1]).abs().max().item(),1e-6)

    def test_field_same_x_independent_from_candidate_identity_and_other_images(self):
        items=frames(4); batch=collate_frames(items)
        m=model_for('h2'); scene=tensor_scene(batch,'cpu'); encoded=m.field.encode(scene)
        xyz=torch.as_tensor(batch['target_xyz'])
        raw=m.field.query(encoded,scene,xyz)
        self.assertEqual(raw.shape,(4,96,6))
        for b in range(4):
            single_scene=tensor_scene(items[b],'cpu')
            single=m.field.query(m.field.encode(single_scene),single_scene,xyz[b])
            torch.testing.assert_close(raw[b],single,atol=1e-6,rtol=1e-5)

    def test_chunking_and_stale_access_are_batch_safe(self):
        batch=collate_inputs(frames(4))
        torch.manual_seed(42)
        small=model_for('h2',ModelConfig(8,16,.03,11,1))
        large=model_for('h2',ModelConfig(8,16,.03,100000,999))
        large.load_state_dict(small.state_dict())
        for mode in ('fresh','stale'):
            p=small(tensor_scene(batch,'cpu'),torch.as_tensor(batch['bank']),mode)
            q=large(tensor_scene(batch,'cpu'),torch.as_tensor(batch['bank']),mode)
            torch.testing.assert_close(p,q,atol=1e-6,rtol=1e-5)

    def test_ragged_or_malformed_inputs_fail(self):
        items=frames(2)
        broken=copy.deepcopy(items);broken[1]['bank']=broken[1]['bank'][:2]
        with self.assertRaisesRegex(ValueError,'Ragged bank'):collate_inputs(broken)
        broken=copy.deepcopy(items);broken[1]['target_xyz']=broken[1]['target_xyz'][:30]
        with self.assertRaisesRegex(ValueError,'Ragged target_xyz'):collate_frames(broken)
        with self.assertRaisesRegex(ValueError,'Empty multi-image'):collate_inputs([])
        bank=collate_inputs(items)
        model=model_for('h2')
        with self.assertRaises(ValueError):model(tensor_scene(bank,'cpu'),torch.as_tensor(bank['bank'][0]))
        scene=tensor_scene(bank,'cpu')
        enc=model.field.encode(scene)
        with self.assertRaises(ValueError):model.field.query(enc,scene,torch.randn(3,2,3))
        with self.assertRaises(ValueError):model.field.query(enc,scene,torch.randn(2,3))

    def test_appearance_interventions_match_individual_calls(self):
        items=frames(4); inp=collate_inputs(items)
        for intervention in ('appearance_mean','appearance_shuffle','stale'):
            m=model_for('h2'); bat,_=predict(m,'h2',inp,'cpu',intervention)
            singled=torch.stack([predict(m,'h2',x,'cpu',intervention)[0] for x in items])
            torch.testing.assert_close(bat,singled,atol=1e-6,rtol=1e-5)


class BackpropBatchParity(unittest.TestCase):
    def test_gradient_parity_all_modes_b2_b4(self):
        items=frames(4)
        for mode in ('local','predicted','h1','h2','h2_stale'):
            torch.manual_seed(133)
            reference=model_for(mode)
            baseline=copy.deepcopy(reference)
            seq_grad=[]
            for sample in items:
                loss=full_loss(reference,mode,sample)/len(items)
                loss.backward()
            seq_grad={k:p.grad.detach().clone() for k,p in reference.named_parameters() if p.grad is not None}
            for bs in (2,4):
                with self.subTest(mode=mode,bs=bs):
                    m=copy.deepcopy(baseline)
                    for start in range(0,len(items),bs):
                        subset=items[start:start+bs]
                        loss=full_loss(m,mode,collate_frames(subset))*(len(subset)/len(items))
                        loss.backward()
                    for k,p in m.named_parameters():
                        if k in seq_grad:
                            torch.testing.assert_close(p.grad,seq_grad[k],atol=2e-5,rtol=2e-4)

    def test_geometry_and_quality_losses_mean_per_image(self):
        items=frames(4); m=model_for('h2')
        batch=collate_frames(items);z,field=predict(m,'h2',batch,'cpu')
        enc,scene=field
        raw=m.field.query(enc,scene,torch.as_tensor(batch['target_xyz']))
        t={k:torch.as_tensor(batch['target_'+k]) for k in ('udf','projective','state','udf_mask','projective_mask')}
        group_geo,_=geometry_loss(raw,t)
        geo=[];qs=[]
        for i,x in enumerate(items):
            small={k:v[i] for k,v in t.items()}
            lo,_=geometry_loss(raw[i],small)
            geo.append(lo)
            qq,_=quality_loss(z[i],torch.as_tensor(x['quality_count']).long())
            qs.append(qq)
        torch.testing.assert_close(group_geo,torch.stack(geo).mean(),atol=1e-7,rtol=1e-7)
        total_q,_=quality_loss(z,torch.as_tensor(batch['quality_count']).long())
        torch.testing.assert_close(total_q,torch.stack(qs).mean(),atol=1e-7,rtol=1e-7)


def make_multibatch_indices(root, ntrain=8, nval=4):
    paths=[]
    for part,n,scenes in (('train',ntrain,range(0,ntrain)),('validation',nval,range(80,80+nval))):
        entries=[]
        for i,scene in enumerate(scenes):
            row=dict(partition=part,scene=scene,anno=i,domain='original' if i%2==0 else 'gntrans',split='train')
            p=root/(frame_key(row)+'.npz');atomic_npz(p,fixture(n=2,seed=i+1))
            entries.append(dict(row=row,path=str(p),sha256=sha256(p)))
        contract=dict(source={'base_sha256':'b','selector_sha256':'s','config_sha256':'cfg'},
                      plan='synthetic-multibatch',offsets_m=[0.,-.005,.005,-.01,.01,-.02,.02],
                      tau=.03,band=.002,target_config='test',partition=part)
        path=root/(part+'.json')
        atomic_json(path,signed(dict(schema='shared-visible-geometry-v1',complete=False,
                                     partition=part,entries=entries,contract=contract)))
        paths.append(path)
    return paths


class TrainAndInferMultiBatch(unittest.TestCase):
    def test_train_true_B4_and_resume_protocol(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);tr,va=make_multibatch_indices(root)
            for mode in ('local','predicted','h1','h2','h2_stale'):
                with self.subTest(mode=mode):
                    args=SimpleNamespace(mode=mode,epochs=1,batch_size=4,accumulate=1,val_batch_size=2,
                        train_index=str(tr),val_index=str(va),output=str(root/mode),smoke=True,resume=False,
                        hidden=12,seed=7,device='cpu',cpu_threads=1,lr=1e-3,weight_decay=1e-4,geometry_weight=1.)
                    fit(args)
                    ck=torch.load(root/mode/'last.pt',weights_only=False)
                    self.assertEqual(ck['updates'],2)
                    self.assertEqual(ck['protocol']['batch_size'],4)
                    self.assertEqual(ck['protocol']['effective_batch_size'],4)
                    self.assertEqual(ck['protocol']['val_batch_size'],2)
                    args.resume=True;fit(args)
                    self.assertEqual(torch.load(root/mode/'last.pt',weights_only=False)['updates'],2)
                    args.batch_size=2;args.accumulate=2
                    with self.assertRaisesRegex(ValueError,'Resume protocol'):fit(args)

    def test_real_training_B4_vs_B1accumulate4_close(self):
        # Same source/labels/initialization/frame order. Numerical summation
        # differs, so compare trainable weights within float tolerance.
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);tr,va=make_multibatch_indices(root)
            def run(bs,acc,name):
                args=SimpleNamespace(mode='h2',epochs=1,batch_size=bs,accumulate=acc,val_batch_size=2,
                    train_index=str(tr),val_index=str(va),output=str(root/name),smoke=True,resume=False,
                    hidden=12,seed=9,device='cpu',cpu_threads=1,lr=1e-4,weight_decay=1e-4,geometry_weight=1.)
                fit(args)
                return torch.load(root/name/'last.pt',weights_only=False)
            serial=run(1,4,'serial');batched=run(4,1,'batched')
            self.assertEqual(serial['updates'],batched['updates'])
            for k in serial['model']:
                torch.testing.assert_close(serial['model'][k],batched['model'][k],atol=3e-5,rtol=3e-4)

    def test_validation_batching_is_metric_equivalent(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);tr,va=make_multibatch_indices(root)
            from shared_geometry.data import read_index
            idx=read_index(va,'validation',True)
            for mode in ('local','predicted','h1','h2','h2_stale'):
                m=model_for(mode)
                first=evaluate_index(m,mode,idx,'cpu',batch_size=1)
                third=evaluate_index(m,mode,idx,'cpu',batch_size=3)
                self.assertEqual(len(first['rows']),len(third['rows']))
                for metric in first['summary']:
                    a,b=first['summary'][metric],third['summary'][metric]
                    if a is None:self.assertIsNone(b)
                    else:self.assertAlmostEqual(a,b,places=5,msg=metric)

    def test_head_inference_batched_and_partial_batch_with_isolated_masks(self):
        from shared_geometry.inference import predict_head_batch
        from shared_geometry.geometry import action_bank
        class Provider:
            provenance={'base_sha256':'b','selector_sha256':'s','config_sha256':'cfg'}
            def snapshot(self,row,seed):
                x=fixture(n=1024,seed=row['anno']+10)
                return dict(grasps=x['bank'][:,0],features=x['local_features'],
                    feature_map=x['feature_map'],pred_depth=x['pred_depth'],K=x['K']),{}
            def sensor_keep(self,row,g):
                # Per-image collision-like mask; depends on modified center.
                return (g[:,13] + .1*row['anno'] > -.03)
            def label(self,*args):raise AssertionError('Never load GT in inference')
        class Stub(torch.nn.Module):pass
        sizes=[]
        def synthetic_predict(model,mode,frame,device,intervention='aligned'):
            bank=torch.as_tensor(frame['bank'])
            sizes.append(int(bank.shape[0]) if bank.ndim==4 else 1)
            # Keep new scores per-action and per-scene, not per-batch index.
            centers=bank[...,13]; depth=bank[...,15]
            return centers*3+depth+bank[...,0],None
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);plan=make_plan();atomic_json(root/'plan.json',plan)
            (root/'weights.pt').write_bytes(b'synthetic test')
            proto=dict(mode='h2',source=Provider.provenance,plan=plan['fingerprint'],
                       offsets_m=[0.,-.005,.005,-.01,.01,-.02,.02])
            outputs=[]
            for bs in (1,2,4):
                args=SimpleNamespace(plan=str(root/'plan.json'),weight=str(root/'weights.pt'),
                    device='cpu',limit=3,head_batch_size=bs,reference_root='',torch_threads=1,
                    output=str(root/f'infer_{bs}'),split='test_seen',seed=0)
                sizes.clear()
                with patch('shared_geometry.inference.create_runtime',return_value=Provider()), \
                     patch('shared_geometry.inference.load_model',return_value=(Stub(),proto)), \
                     patch('shared_geometry.inference.predict',side_effect=synthetic_predict):
                    infer(args)
                if bs==2:self.assertIn(2,sizes)
                if bs==4:self.assertIn(3,sizes)
                outputs.append(args.output)
            for policy in ('native','field_score_native','move_keep_priority','move_rescore','stale_move_keep_priority'):
                for sid in range(3):
                    paths=[Path(o)/policy/f'scene_0100/realsense/{sid:04d}.npy' for o in outputs]
                    for path in paths:self.assertTrue(path.exists(),path)
                    baseline=np.load(paths[0]);
                    for path in paths[1:]:np.testing.assert_array_equal(np.load(path),baseline)

    def test_cli_batch_arguments_and_shell_syntax(self):
        for cmd in ('train','diagnose','infer'):
            res=subprocess.run([sys.executable,'-m','shared_geometry.cli',cmd,'--help'],
                               cwd=ROOT,text=True,capture_output=True,timeout=20)
            self.assertEqual(res.returncode,0,res.stderr)
        self.assertIn('--batch-size',subprocess.run([sys.executable,'-m','shared_geometry.cli','train','--help'],
            cwd=ROOT,text=True,capture_output=True).stdout)
        self.assertIn('--head-batch-size',subprocess.run([sys.executable,'-m','shared_geometry.cli','infer','--help'],
            cwd=ROOT,text=True,capture_output=True).stdout)
        res=subprocess.run(['bash','-n','scripts/shared_geometry/run.sh'],cwd=ROOT,
                           text=True,capture_output=True)
        self.assertEqual(res.returncode,0,res.stderr)


if __name__ == '__main__':unittest.main()