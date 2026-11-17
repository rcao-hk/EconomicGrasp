"""CPU contracts and end-to-end toy smoke; not production checkpoint/CUDA validation."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from camera_geometry.geometry import (crop_matrix,transform_uv,rays,crop_rgb,pixel_grid,
    plane_from_table,height_from_z,z_from_height,losses,balanced_mask_mean,sample)
from camera_geometry.data import (atomic_json,prepare,load_manifest,load_input,load_reference,
    inputs_for_views,common_queries,validate_entries,import_gc6d)
from camera_geometry.model import MetricModel,ARMS
from camera_geometry.render import camera_pose,rasterize,render
from camera_geometry.evaluate import distance_response,grouped_mean_ci
from camera_geometry.run import DEFAULT,train,predict,evaluate,summarize,preflight,setup,load_run


def fixture(root,steps=2):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    y,x=np.mgrid[:32,:40]
    rgb=np.stack((x*5+20,y*6+20,x+y+80),-1).astype(np.uint8)
    Image.fromarray(rgb).save(root/'rgb.png'); np.save(root/'depth.npy',(.4+x*.001).astype(np.float32))
    entries=[]
    for i,p in enumerate(('train','dev','holdout')):
        entries.append(dict(id='frame'+str(i),group='scene'+str(i),object_ids=['object'+str(i)],partition=p,
            domain='toy',previously_used=False,rgb_path=str(root/'rgb.png'),depth_path=str(root/'depth.npy'),
            depth_scale_m=1.,K=[[45.,0,19.5],[0,46.,15.5],[0,0,1]],
            T_table_from_cam=camera_pose(.65,0).tolist(),plane_source='renderer',plane_provenance='synthetic unit test',
            reference_source='synthetic z in metres',crop_source='predeclared fixed boxes',
            crop_boxes={'full':[0,0,40,32],'roi':[3,2,37,30],'workspace':[5,4,35,28]}))
    atomic_json(root/'records.json',{'entries':entries})
    prepare(root/'records.json',root/'manifest.json')
    cfg={**DEFAULT,'backend':'tiny','manifest':str(root/'manifest.json'),'image_hw':[16,16],
         'train_rays':16,'eval_rays':24,'batch_size':2,'steps':steps,'save_every':1,'bootstrap':50}
    atomic_json(root/'config.json',cfg)
    return cfg


class GeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.K=torch.tensor([[600.,0,317.4],[0,550.,237.1],[0,0,1.]],dtype=torch.float64)
        self.uv=torch.tensor([[50.,30.],[200.,150.],[500.,400.]],dtype=torch.float64)

    def test_anisotropic_crop_ray_identity(self):
        A=crop_matrix([12,17,590,420],(448,448),dtype=torch.float64)
        self.assertTrue(torch.allclose(rays(self.uv,self.K),rays(transform_uv(self.uv,A),A@self.K),atol=1e-12))

    def test_projection_commutes(self):
        A=crop_matrix([10,20,400,300],(80,100),dtype=torch.float64)
        X=rays(self.uv,self.K)*torch.tensor([[.4],[.8],[1.]],dtype=torch.float64)
        pix=X@(A@self.K).T; expected=pix[:,:2]/pix[:,2:]
        self.assertTrue(torch.allclose(expected,transform_uv(self.uv,A),atol=1e-10))

    def test_half_pixel_not_naive_scale(self):
        A=crop_matrix([0,0,640,480],(240,320))
        self.assertAlmostEqual(float(A[0,2]),-.25)

    def test_identity_crop_rgb(self):
        rgb=torch.rand(3,10,12); K=torch.eye(3)
        out,k,a,valid=crop_rgb(rgb,K,[0,0,12,10],(10,12))
        recovered=out*torch.tensor([.229,.224,.225])[:,None,None]+torch.tensor([.485,.456,.406])[:,None,None]
        self.assertTrue(torch.allclose(rgb,recovered,atol=1e-6)); self.assertTrue(valid.all())

    def test_padding_is_not_valid(self):
        out,_,_,valid=crop_rgb(torch.rand(3,8,8),torch.eye(3),[-4,-4,12,12],(16,16))
        self.assertFalse(bool(valid.all())); self.assertTrue(torch.all(out[:,~valid]==0))

    def test_plane_rotation_and_translation(self):
        T=torch.tensor(camera_pose(.8,35),dtype=torch.float64); p=plane_from_table(T)
        X=torch.tensor([[.1,.1,.5]],dtype=torch.float64)
        table=X@T[:3,:3].T+T[:3,3]
        self.assertTrue(torch.allclose(X@p[:3]+p[3],table[:,2]))

    def test_height_depth_roundtrip(self):
        p=plane_from_table(torch.tensor(camera_pose(.8,35),dtype=torch.float64))[None]
        uv=self.uv[None]; K=self.K[None]; z=torch.tensor([[.3,.5,.7]],dtype=torch.float64)
        h=height_from_z(z,uv,K,p); rebuilt,valid=z_from_height(h,uv,K,p)
        self.assertTrue(valid.all()); self.assertTrue(torch.allclose(z,rebuilt,atol=1e-12))

    def test_reject_reflection_or_wrong_plane(self):
        T=torch.eye(4); T[2,3]=.5; T[0,0]=-1
        with self.assertRaises(ValueError): plane_from_table(T)
        with self.assertRaises(ValueError): plane_from_table(torch.eye(4))

    def test_grazing_mask_finite(self):
        z,ok=z_from_height(torch.ones(1,1)*.1,torch.zeros(1,1,2),torch.eye(3)[None],torch.tensor([[1.,0,0,.5]]))
        self.assertFalse(ok.any()); self.assertTrue(torch.isfinite(z).all())

    def test_negative_depth_not_silently_dropped(self):
        z,ok=z_from_height(torch.tensor([[.8]]),torch.zeros(1,1,2),torch.eye(3)[None],torch.tensor([[0.,0,-1.,.5]]))
        self.assertTrue(ok.all()); self.assertLess(float(z),0)
        loss,_=losses(z,z,torch.tensor([[.4]]),torch.tensor([[[0.,0,1.]]]),ok,1.)
        self.assertGreater(float(loss),70.)

    def test_consistency_and_collapse(self):
        z=torch.ones(2,4)*.4; r=torch.ones(2,4,3); m=torch.ones_like(z,dtype=torch.bool)
        loss,_=losses(z,z,z,r,m,1.); self.assertEqual(float(loss),0.)
        loss,t=losses(z*.1,z*.1,z,r,m,1.)
        self.assertGreater(float(loss),0); self.assertEqual(float(t['xyz_consistency']),0)

    def test_equal_image_weighting(self):
        x=torch.tensor([[2.,0.],[6.,6.]]); m=torch.tensor([[True,False],[True,True]])
        self.assertEqual(float(balanced_mask_mean(x,m)),4.)
        with self.assertRaises(ValueError): balanced_mask_mean(x,torch.zeros_like(m))

    def test_crop_same_ray_linear_depth(self):
        image=torch.arange(10).float()[None,None,None].expand(1,1,8,10)
        q=torch.tensor([[[2.4,3.],[7.1,4.]]])
        self.assertTrue(torch.allclose(sample(image,q)[0,:,0],q[0,:,0],atol=1e-5))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name); self.cfg=fixture(self.root)
        self.m=load_manifest(self.root/'manifest.json')
    def tearDown(self): self.tmp.cleanup()

    def test_reference_metre_and_raw_shapes(self):
        z,m=load_reference(self.m['entries'][0]); self.assertEqual(z.shape,(32,40)); self.assertTrue(m.all())

    def test_split_group_leakage(self):
        es=copy.deepcopy(self.m['entries']); es[1]['group']=es[0]['group']
        with self.assertRaises(ValueError): validate_entries(es)

    def test_holdout_history_and_object_overlap(self):
        es=copy.deepcopy(self.m['entries']); es[2]['previously_used']=True
        with self.assertRaises(ValueError): validate_entries(es)
        es=copy.deepcopy(self.m['entries']); es[2]['object_ids']=es[0]['object_ids']
        with self.assertRaises(ValueError): validate_entries(es)

    def test_official_test_not_train(self):
        es=copy.deepcopy(self.m['entries']); es[0]['official_test']=True
        with self.assertRaises(ValueError): validate_entries(es)

    def test_target_fitted_plane_refused(self):
        es=copy.deepcopy(self.m['entries']); es[0]['plane_source']='fit_from_target_depth'
        with self.assertRaises(ValueError): validate_entries(es)

    def test_file_tampering(self):
        np.save(self.root/'depth.npy',np.ones((32,40),dtype=np.float32))
        with self.assertRaises(ValueError): load_manifest(self.root/'manifest.json')

    def test_same_raw_rays_two_views(self):
        e=self.m['entries'][0]; vs=inputs_for_views(e,['full','roi'],[16,16])
        uv,_,_=common_queries(e,vs,[16,16],16,np.random.default_rng(4))
        rr=[rays(transform_uv(uv,v['A']),v['K']) for v in vs]
        self.assertTrue(torch.allclose(rr[0],rr[1],atol=1e-6))

    def test_insufficient_overlap_stops(self):
        e=self.m['entries'][0]; vs=inputs_for_views(e,['full','roi'],[16,16])
        with self.assertRaises(ValueError): common_queries(e,vs,[16,16],100000,np.random.default_rng(0))


    def test_gc6d_native_import_does_not_invent_table(self):
        import shutil
        ds=self.root/'gc6d'; scene=ds/'scenes/000001'
        for name in ('rgb','depth','label'): (scene/name).mkdir(parents=True)
        shutil.copyfile(self.root/'rgb.png',scene/'rgb/000002.png')
        Image.fromarray(np.full((32,40),400,dtype=np.uint16)).save(scene/'depth/000002.png')
        Image.fromarray(np.ones((32,40),dtype=np.uint8)).save(scene/'label/000002.png')
        T=camera_pose(.65,0)
        atomic_json(scene/'scene_camera.json',{'2':dict(cam_K=[45,0,19.5,0,46,15.5,0,0,1],
            cam_R_w2c=T[:3,:3].reshape(-1).tolist(),cam_t_w2c=(T[:3,3]*1000).tolist(),depth_scale=1.)})
        atomic_json(scene/'scene_gt.json',{'2':[{'obj_id':5}]})
        atomic_json(ds/'split_info/grasp_test_scene_ids.json',[])
        cfg=dict(dataset_root=str(ds),camera='realsense-d435',camera_transform_semantics='c2w',
            world_up=[0,0,1],orientation_provenance='fixture world up verified',reference_source='sensor',
            reference_provenance='test sensor',fixed_roi_fraction=[.1,.1,.9,.9],
            scenes=[dict(scene=1,partition='train',previously_used=True)])
        atomic_json(self.root/'import.json',cfg)
        m=import_gc6d(self.root/'import.json',self.root/'native')
        self.assertNotIn('T_table_from_cam',m['entries'][0])
        z,_=load_reference(m['entries'][0]); self.assertAlmostEqual(float(z.mean()),.4,places=6)
        cfg.update(T_table_from_world=np.eye(4).tolist(),plane_provenance='explicit calibrated fixture table')
        atomic_json(self.root/'import2.json',cfg)
        m=import_gc6d(self.root/'import2.json',self.root/'native_calibrated')
        self.assertIn('T_table_from_cam',m['entries'][0])


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.c={**DEFAULT,'backend':'tiny','image_hw':[16,16]}
        self.x=dict(rgb=torch.rand(2,3,16,16),K=torch.tensor([[[30.,0,7.5],[0,30.,7.5],[0,0,1.]]]).repeat(2,1,1),
                    pose=torch.tensor([[0.,0,1.]]).repeat(2,1),gravity=torch.tensor([[0.,0,-1.]]).repeat(2,1),
                    plane=torch.tensor([[0.,0,-1.,.7]]).repeat(2,1))

    def test_camera_identity_initialization(self):
        setup(0); a=MetricModel(self.c,'C0'); setup(0); b=MetricModel(self.c,'C1')
        self.assertTrue(torch.equal(a(**self.x)['z'],b(**self.x)['z']))

    def test_c0_does_not_read_K_for_z(self):
        m=MetricModel(self.c,'C0'); a=m(**self.x)['z']
        other={**self.x,'K':self.x['K'].clone()}; other['K'][:,0,0]*=2
        self.assertTrue(torch.equal(a,m(**other)['z']))

    def test_learned_ray_module_can_read_K(self):
        m=MetricModel(self.c,'C1')
        with torch.no_grad(): m.ray_adapter.weight.fill_(.1)
        a=m(**self.x)['z']; other={**self.x,'K':self.x['K'].clone()}; other['K'][:,0,0]*=2
        self.assertGreater(float((a-m(**other)['z']).abs().max()),1e-7)

    def test_table_equal_capacity_and_shared_weights(self):
        models=[]
        for arm in ('T0','T1','T2'): setup(5); models.append(MetricModel(self.c,arm))
        self.assertEqual(len({sum(p.numel() for p in m.parameters()) for m in models}),1)
        for k,v in models[0].state_dict().items():
            self.assertTrue(torch.equal(v,models[1].state_dict()[k])); self.assertTrue(torch.equal(v,models[2].state_dict()[k]))

    def test_table_camera_translation_decoded_analytically(self):
        m=MetricModel(self.c,'T2'); a=m(**self.x)['z']
        other={**self.x,'plane':self.x['plane'].clone()}; other['plane'][:,3]+=.2
        self.assertTrue(torch.allclose(m(**other)['z']-a,torch.full_like(a,.2),atol=1e-6))

    def test_table0_does_not_consume_distance(self):
        m=MetricModel(self.c,'T0'); a=m(**self.x)['z']
        other={**self.x,'plane':self.x['plane'].clone()}; other['plane'][:,3]+=.2
        self.assertTrue(torch.equal(a,m(**other)['z']))

    def test_table_requires_calibration_all_arms(self):
        for arm in ('T0','T1','T2'):
            with self.assertRaises(ValueError): MetricModel(self.c,arm)(**{k:v for k,v in self.x.items() if k!='plane'})

    def test_frozen_weights_and_gradients(self):
        m=MetricModel(self.c,'C3'); before=m.frozen_fingerprint(); m.train()
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001)
        m(**self.x)['z'].mean().backward(); opt.step()
        self.assertEqual(before,m.frozen_fingerprint())
        self.assertTrue(all(p.grad is None for p in m.backbone.parameters()))


class MetricsRendererTests(unittest.TestCase):
    def test_metric_table_distance(self):
        v=np.array([[-.05,-.05,.05],[.05,-.05,.05],[0,.05,.05]])
        f=np.array([[0,1,2]]); K=np.array([[45.,0,15.5],[0,45.,15.5],[0,0,1.]])
        _,a,mask=rasterize(v,f,K,camera_pose(.5,0),(32,32))
        _,b,_=rasterize(v,f,K,camera_pose(.8,0),(32,32))
        self.assertAlmostEqual(float(a[0,0]),.5,places=6)
        self.assertAlmostEqual(float(b[0,0]-a[0,0]),.3,places=6)
        self.assertTrue(mask.any()); self.assertAlmostEqual(float(a[mask>0].mean()),.45,places=5)

    def test_track_response_one_and_no_fake_fallback(self):
        ts=[]
        for z in (.4,.6,.8):
            ts.append(dict(group='obj',domain='render',condition='fixed',crop='roi',track='p',
                           xyz_table=[0,0,0],target_z_m=z,pred_z_m=z+.01))
        r=distance_response(ts); self.assertAlmostEqual(r['by_crop']['roi']['mean'],1.)
        self.assertTrue(distance_response([])['status'].startswith('NOT_AVAILABLE'))

    def test_track_identity_mismatch(self):
        t=dict(group='obj',domain='r',condition='fixed',crop='roi',track='p',xyz_table=[0,0,0],target_z_m=.4,pred_z_m=.4)
        with self.assertRaises(ValueError): distance_response([t,{**t,'xyz_table':[1,0,0]}])

    def test_group_not_pixel_weighting(self):
        rs=[dict(group='a',domain='x',v=0.)]*100+[dict(group='b',domain='x',v=10.)]
        self.assertEqual(grouped_mean_ci(rs,'v',50)['mean'],5.)

    def test_renderer_produces_signed_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); np.savez(p/'mesh.npz',vertices=np.array([[-.08,-.08,0],[.08,-.08,0],[0,.08,.08]]),faces=np.array([[0,1,2]]))
            spec=dict(raw_hw=[64,64],focal_px=90,train_distances_m=[.35],eval_distances_m=[.45],tilts_deg=[0],
                focal_modes=['fixed'],min_foreground_pixels=30,
                objects=[dict(id='new',mesh=str(p/'mesh.npz'),scale_to_m=1.,partition='train',previously_used=False)])
            atomic_json(p/'spec.json',spec); m=render(p/'spec.json',p/'renders')
            self.assertEqual(len(m['entries']),1); load_manifest(p/'renders/manifest.json')


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name); self.c=fixture(self.root)
    def tearDown(self): self.tmp.cleanup()

    def test_preflight(self):
        preflight(self.root/'config.json',self.root/'preflight.json','cpu')
        self.assertEqual(json.loads((self.root/'preflight.json').read_text())['C0_C1_max_error_m'],0.)

    def test_all_seven_train_and_evaluate(self):
        self.c['steps']=1; atomic_json(self.root/'config.json',self.c)
        outputs=[]
        for arm in ARMS:
            train(self.root/'config.json',arm,self.root/arm,'cpu',smoke=True)
            outputs.append(self.root/arm/'final_dev.json')
        summarize(outputs,self.root/'summary.json')
        s=json.loads((self.root/'summary.json').read_text())
        self.assertIn('camera_x_consistency',s['comparisons']); self.assertIn('T2-T1_parameterization',s['comparisons'])

    def test_exact_cpu_resume(self):
        a=train(self.root/'config.json','C3',self.root/'continuous','cpu',smoke=True)
        train(self.root/'config.json','C3',self.root/'resumed','cpu',smoke=True,stop_after=1)
        b=train(self.root/'config.json','C3',self.root/'resumed','cpu',resume=True,smoke=True)
        ca=torch.load(a,weights_only=False); cb=torch.load(b,weights_only=False)
        for k,v in ca['model'].items(): self.assertTrue(torch.equal(v,cb['model'][k]),k)
        self.assertEqual(ca['history'],cb['history'])

    def test_resume_budget_change_rejected(self):
        train(self.root/'config.json','C0',self.root/'run','cpu',smoke=True,stop_after=1)
        self.c['steps']=3; atomic_json(self.root/'config.json',self.c)
        with self.assertRaises(ValueError): train(self.root/'config.json','C0',self.root/'run','cpu',resume=True,smoke=True)

    def test_no_overwrite(self):
        train(self.root/'config.json','C0',self.root/'run','cpu',smoke=True)
        with self.assertRaises(FileExistsError): train(self.root/'config.json','C0',self.root/'run','cpu',smoke=True)

    def test_inference_never_reads_teacher(self):
        ck=train(self.root/'config.json','T2',self.root/'run','cpu',smoke=True)
        (self.root/'depth.npy').unlink()
        with patch('camera_geometry.data.load_reference',side_effect=AssertionError('teacher read')):
            predict(ck,self.root/'manifest.json','holdout',self.root/'predictions','cpu')
        self.assertFalse(json.loads((self.root/'predictions/index.json').read_text())['teacher_files_read'])

    def test_toy_not_formal(self):
        with self.assertRaises(ValueError): train(self.root/'config.json','C0',self.root/'run','cpu')

    def test_partial_checkpoint_not_holdout(self):
        ck=train(self.root/'config.json','C0',self.root/'run','cpu',smoke=True,stop_after=1)
        with self.assertRaises(ValueError): evaluate(ck,self.root/'manifest.json','holdout',self.root/'eval.json','cpu')

if __name__=='__main__': unittest.main()
