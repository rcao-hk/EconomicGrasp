"""Pure deterministic score-sweep and quality-comparison helpers."""
from __future__ import annotations
import numpy as np

DEFAULT_ALPHAS=(0.,0.05,0.10,0.20,0.40,0.70,1.)


def parse_alphas(raw):
    a=np.asarray([float(x) for x in raw.split(',')],dtype=float)
    if not len(a) or not np.isfinite(a).all() or (a<0).any() or (a>1).any() or len(np.unique(a))!=len(a) or a[0]!=0 or a[-1]!=1 or np.any(np.diff(a)<=0):
        raise ValueError('Alphas must be strictly increasing, unique, include 0 and 1, and lie in [0,1]')
    return a


def mix_logits(base_z, trained_z, alpha, exact_base_probability=None):
    z0=np.asarray(base_z,dtype=float);z=np.asarray(trained_z,dtype=float)
    if z0.shape!=z.shape or not np.isfinite(z0).all() or not np.isfinite(z).all():
        raise ValueError('Invalid frozen/new logits')
    if not 0<=alpha<=1:raise ValueError('Invalid alpha')
    if alpha==0 and exact_base_probability is not None:
        p=np.asarray(exact_base_probability,dtype=float)
        if p.shape!=z0.shape or not np.isfinite(p).all() or not np.all((p>=0)&(p<=1)):
            raise ValueError('Invalid exact base probabilities')
        return p.copy()
    v=z0+alpha*(z-z0)
    return np.where(v>=0,1/(1+np.exp(-np.minimum(v,700))),np.exp(np.maximum(v,-700))/(1+np.exp(np.maximum(v,-700))))


def quality_pairs(base_scores,new_scores,quality,mask):
    """Compare strictly-quality-distinct *labelled* pairs; ties neither repair nor damage."""
    b=np.asarray(base_scores)[mask]; s=np.asarray(new_scores)[mask];q=np.asarray(quality)[mask]
    i,j=np.triu_indices(len(q),1);different=q[i]!=q[j];i,j=i[different],j[different]
    if not len(i):return dict(pairs=0,base_agreement=None,new_agreement=None,repair=0,damage=0,improve_tie=0,harm_tie=0)
    orient=np.sign(q[i]-q[j]);old=np.sign(b[i]-b[j])*orient;new=np.sign(s[i]-s[j])*orient
    return dict(pairs=int(len(i)),base_agreement=float(np.mean((old>0)+.5*(old==0))),
                new_agreement=float(np.mean((new>0)+.5*(new==0))),
                repair=int(np.sum((old<0)&(new>0))),damage=int(np.sum((old>0)&(new<0))),
                improve_tie=int(np.sum((old<=0)&(new>0))),harm_tie=int(np.sum((old>=0)&(new<0))))


def selection_stats(base,new,quality,label_mask,keep,ks=(1,5,10,50)):
    """Full-pool ranking/membership measured separately from sparse Q supervision."""
    b=np.asarray(base).reshape(-1);s=np.asarray(new).reshape(-1);q=np.asarray(quality).reshape(-1)
    lab=np.asarray(label_mask,dtype=bool).reshape(-1);good=np.asarray(keep,dtype=bool).reshape(-1)
    if not(b.shape==s.shape==q.shape==lab.shape==good.shape) or not good.any():
        raise ValueError('No candidates after sensor prefilter or shape mismatch')
    ids=np.flatnonzero(good)
    old=ids[np.argsort(-b[ids],kind='stable')];now=ids[np.argsort(-s[ids],kind='stable')]
    out=dict(top1_changed=bool(old[0]!=now[0]),top1_original_labelled=bool(lab[old[0]]),
             top1_new_labelled=bool(lab[now[0]]),top1_original_q=float(q[old[0]]/6) if lab[old[0]] else None,
             top1_new_q=float(q[now[0]]/6) if lab[now[0]] else None)
    for k in ks:
        oi=set(old[:k]);ni=set(now[:k]);out[f'top{k}_overlap']=float(len(oi&ni)/max(1,min(k,len(ids))))
        out[f'top{k}_labelled_coverage']=float(np.mean(lab[now[:k]]))
    return out


def weighted_summary(rows,metric):
    """Mean within (domain,scene), equal domain weight; avoids IID frame claims."""
    vals=[]
    for d in sorted(set(r['domain'] for r in rows)):
        scene=[]
        for sid in sorted(set(r['scene'] for r in rows if r['domain']==d)):
            x=[r[metric] for r in rows if r['domain']==d and r['scene']==sid and r[metric] is not None]
            if x:scene.append(np.mean(x))
        if scene:vals.append(np.mean(scene))
    return float(np.mean(vals)) if vals else None


def choose_alpha(rows, metric='proxy_ap', alphas=DEFAULT_ALPHAS):
    """Only use independently held-out *stage-1 validation* full-label proxy.

    Choose alpha=0 when no positive mean gain over baseline. Never use test.
    """
    by_alpha={a:[r for r in rows if np.isclose(r['alpha'],a) and r['partition']=='validation'] for a in alphas}
    if any(not rs for rs in by_alpha.values()):raise ValueError('Missing validation alpha rows')
    means={str(a):weighted_summary(rs,metric) for a,rs in by_alpha.items()}
    if any(v is None for v in means.values()):raise ValueError('Insufficient full-quality labels for alpha selection')
    base=means[str(alphas[0])]
    best=max(alphas,key=lambda a:(means[str(a)],-a))
    if means[str(best)]<=base+1e-12:best=0.
    return dict(alpha=float(best),means=means,selection_metric=metric,source='validation_only',
                rule='max mean heldout proxy; tie smallest alpha; fallback alpha=0')


def paired_scene_bootstrap(rows,variant_a,variant_b,alpha,metric,n_boot=2000,seed=0):
    """Bootstrap 20 scene IDs; Original/GNTrans paired within each scene."""
    if n_boot<1:raise ValueError('Invalid bootstrap repeats')
    scenes=sorted(set(r['scene'] for r in rows))
    values=[]
    for sid in scenes:
        per_domain=[]
        for domain in ('original','gntrans'):
            left=[r[metric] for r in rows if r['scene']==sid and r['domain']==domain and r['variant']==variant_a and r['alpha']==alpha and r[metric] is not None]
            right=[r[metric] for r in rows if r['scene']==sid and r['domain']==domain and r['variant']==variant_b and r['alpha']==alpha and r[metric] is not None]
            if len(left)!=len(right) or not left:raise ValueError('Missing paired validation scenes/frames or quality labels')
            per_domain.append(float(np.mean(left)-np.mean(right)))
        values.append(float(np.mean(per_domain)))
    if not values:return dict(mean=None,ci95=None,n_scenes=0)
    x=np.asarray(values,dtype=float)
    gen=np.random.default_rng(seed)
    draws=x[gen.integers(0,len(x),size=(n_boot,len(x)))].mean(1)
    return dict(mean=float(x.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),n_scenes=len(x),
                unit='paired scene bootstrap; two RGB domains paired within scene; fixed weights')