import numpy as np
import pytest
from diagnostics.rgb_only_validation_geometry import (
    align_global_median_ratio,align_support_plane,assert_fixed_action_bank_identity,
    make_asymmetric_crop_homography,make_camera_roll_homography,make_zoom_homography,
    normalize_role_mask,plane_depth_map,project_points,transform_intrinsics,unproject_pixels,
)

def test_crop_zoom_intrinsics_preserve_camera_points():
    h,w=448,448
    K=np.array([[420.,0.,223.5],[0.,418.,221.5],[0.,0.,1.]])
    uv=np.array([[80.,90.],[220.,210.],[360.,350.]])
    z=np.array([.6,.8,1.])
    xyz=unproject_pixels(uv,z,K)
    for H in (make_zoom_homography(h,w,1.2),make_asymmetric_crop_homography(h,w,(20,30,410,400))):
      uvh=np.concatenate((uv,np.ones((len(uv),1))),axis=1)@H.T
      uv2=uvh[:,:2]/uvh[:,2:]
      assert np.allclose(xyz,unproject_pixels(uv2,z,transform_intrinsics(K,H)),atol=1e-8)

def test_roll_homography_and_camera_frame_transform_are_consistent():
    K=np.array([[430.,0.,223.5],[0.,420.,221.5],[0.,0.,1.]])
    xyz=np.array([[.1,-.05,.7],[-.2,.1,.9]])
    H,R=make_camera_roll_homography(K,30.)
    uv=project_points(xyz,K)
    uvh=np.concatenate((uv,np.ones((len(uv),1))),axis=1)@H.T
    uv2=uvh[:,:2]/uvh[:,2:]
    assert np.allclose(project_points(xyz@R,K),uv2,atol=1e-8)

def test_plane_intersection_and_global_scale():
    K=np.array([[400.,0.,20.],[0.,400.,20.],[0.,0.,1.]])
    plane=np.array([0.,0.,1.,-.8])
    z,valid=plane_depth_map(K,40,40,plane)
    assert valid.all() and np.allclose(z,.8)
    pred=np.full((40,40),.7,np.float32); gt=np.full((40,40),.8,np.float32)
    aligned,alpha=align_global_median_ratio(pred,gt,np.ones_like(pred,bool))
    assert alpha==pytest.approx(.8/.7) and np.allclose(aligned,gt)
    aligned2,alpha2=align_support_plane(pred,z,np.ones_like(pred,bool))
    assert alpha2==pytest.approx(.8/.7) and np.allclose(aligned2,gt)

def test_role_masks_and_fixed_bank_contract():
    assert normalize_role_mask("contact_closing").tolist()==[True,True,True,False,False,False]
    ref=np.zeros((4,17),np.float32); cand=ref.copy(); cand[:,0]=np.arange(4)
    assert_fixed_action_bank_identity(ref,cand)
    cand[0,13]=.01
    with pytest.raises(AssertionError): assert_fixed_action_bank_identity(ref,cand)


def test_rayrope_position_type_validation():
    from moge_rayrope.config import ModelConfig
    for mode in ("ray","pixel2d","point3d"):
        ModelConfig(position_type=mode)
    with pytest.raises(ValueError):
        ModelConfig(position_type="camera_xyz")
