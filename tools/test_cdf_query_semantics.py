"""CPU enumeration reference for CDF scene->object query label lookup."""
import json
import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.argv=[sys.argv[0]]
import torch
import utils.label_generation as l


def reference(top, mapping, scene, perm, tensors):
    outputs=[torch.zeros((len(scene),x.shape[2],x.shape[3]),dtype=x.dtype) for x in tensors]
    found=torch.zeros(len(scene),dtype=torch.bool)
    for query,s in enumerate(scene.tolist()):
        canonical=int(mapping[s]);slots=[k for k,v in enumerate(top[query].tolist()) if v>=0 and v==canonical]
        if not slots:continue
        assert len(slots)==1
        found[query]=True
        for aa in range(perm.shape[1]):
            for dd in range(tensors[0].shape[-1]):
                for out,source in zip(outputs,tensors):out[query,aa,dd]=source[query,slots[0],int(perm[s,aa]),dd]
    return (*outputs,found)


def run():
    torch.manual_seed(0);n,k,v,a,d=7,3,5,4,2
    top=torch.tensor([[0,1,2],[0,2,4],[-1,1,3],[-1,-1,4],[1,2,3],[0,2,4],[0,1,4]])
    scene=torch.tensor([0,1,2,3,4,0,4]);perm=torch.stack([torch.roll(torch.arange(a),i) for i in range(v)])
    cdf=torch.arange(n*k*a*d).reshape(n,k,a,d)%6;width=torch.arange(n*k*a*d).reshape(n,k,a,d).float()/10000;valid=cdf>1
    for mapping in [torch.arange(v),torch.tensor([0,0,1,4,3]),torch.tensor([4,3,2,1,0])]:
        got=l._select_cdf_query_labels(top,mapping,scene,perm,cdf,width,valid)
        expected=reference(top,mapping,scene,perm,(cdf,width,valid))
        assert all(torch.equal(x,y) for x,y in zip(got,expected))
    duplicate=top.clone();duplicate[0,1]=duplicate[0,0]
    try:l._select_cdf_query_labels(duplicate,torch.arange(v),scene,perm,cdf,width,valid)
    except ValueError as e:assert 'Duplicate' in str(e)
    else:raise AssertionError('Ambiguous duplicate slot accepted')
    # Legacy alignment is equivalent for the one-to-one view mapping.
    aligned=l._align_topk_angle_depth_labels(cdf,top,perm)
    for i,s in enumerate(scene.tolist()):
        slots=[j for j,x in enumerate(top[i].tolist()) if x==s]
        if slots:assert torch.equal(aligned[i,slots[0]],reference(top,torch.arange(v),scene,perm,(cdf,width,valid))[0][i])
    # Enumerate keypoint distances, including a nontrivial 3-D object rotation.
    views,rot=l._build_view_angle_rot_grid(17,12,torch.device('cpu'),torch.float64)
    rz=torch.tensor([[math.cos(.37),-math.sin(.37),0],[math.sin(.37),math.cos(.37),0],[0,0,1]],dtype=torch.float64)
    rx=torch.tensor([[1,0,0],[0,math.cos(.41),-math.sin(.41)],[0,math.sin(.41),math.cos(.41)]],dtype=torch.float64)
    for rotation in [torch.eye(3,dtype=torch.float64),rz@rx]:
        moved=views@rotation.T
        mapping=((views[:,None]-moved[None]).square().sum(-1)).argmin(-1)
        trans=(rotation@rot).index_select(0,mapping)
        got=l._build_angle_alignment_perm(rot,trans,stable_ties=True)
        count=17*12;centers=torch.zeros(count,3,dtype=torch.float64);sizes=torch.full((count,),.02,dtype=torch.float64)
        p,_=l._batch_get_key_points(centers,rot.reshape(-1,3,3),sizes,sizes)
        t,ts=l._batch_get_key_points(centers,trans.reshape(-1,3,3),sizes,sizes)
        p=p.reshape(17,12,-1);t=t.reshape(17,12,-1);ts=ts.reshape(17,12,-1)
        for sv in range(17):
            for sa in range(12):
                candidates=[(sum(float(z)**2 for z in (p[sv,sa]-src[sv,oa]).tolist()),sym,oa) for sym,src in enumerate((t,ts)) for oa in range(12)]
                # Floating sums follow the scalar reference; away from exact ties
                # candidate distances agree to roundoff. Explicit tie uses same tuple order.
                best=min(candidates);chosen=int(got[sv,sa])
                choice=min(x[0] for x in candidates if x[2]==chosen)
                assert abs(choice-best[0])<1e-15
        tied=rot[:,0:1].expand(-1,12,-1,-1).clone()
        assert torch.equal(l._build_angle_alignment_perm(tied,tied,stable_ties=True),torch.zeros(17,12,dtype=torch.long))
    print(json.dumps(dict(status='pass',cases=['one-to-one legacy equivalence','many scene views to one object view','missing slot','padding','first/last angle','duplicate rejection','identity rotation','nontrivial 3D rotation','exact angle tie lowest index'],reference='CPU scalar enumeration, not production vectorized gathers')))


if __name__=='__main__':run()
