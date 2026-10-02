"""Label correctness and fixed-state audits on the production model/data."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import time
import torch
import checkerboard_probe as q


def labels(ep):
    return {k:v.detach().cpu().clone() for k,v in ep.items() if k.startswith('batch_') and torch.is_tensor(v)}


def changes(a,b):
    return {k:dict(changed=int((a[k]!=v).sum()),total=v.numel(),fraction=float((a[k]!=v).float().mean())) for k,v in b.items()}


def label_audit(a):
    prod=q.imports(a.output);q.seed();m=q.make_model(prod);ds=q.bases(prod)
    import utils.label_generation as fixed
    spec=importlib.util.spec_from_file_location('legacy_cdf_label_generation',a.legacy_source/'utils/label_generation.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    # The CPU reference uses scalar query/angle/depth enumeration.
    spec=importlib.util.spec_from_file_location('cdf_semantic_reference',Path(__file__).parent/'tools/test_cdf_query_semantics.py')
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    manifest=json.loads((a.previous/'audit_batch_manifest.json').read_text())
    batches={sp:q.batch_for(prod,ds[sp],v['frames']) for sp,v in manifest.items()}
    cps=json.loads((a.previous/'selected_checkpoints.json').read_text());records=[]
    for cp,name in [(cps[0],'early'),(cps[-1],'late')]:
        s=torch.load(cp['path'],map_location='cpu',weights_only=False);m.load_state_dict(s.get('model_state_dict',s),strict=True);del s
        state=q.tree_hash(m.state_dict())
        for split,cpu in batches.items():
            tick=time.time();inputhash=q.tree_hash(cpu);reference_calls=[]
            original_selector=fixed._select_cdf_query_labels
            def checked(*args):
                result=original_selector(*args)
                c=[x.detach().cpu() for x in args]
                want=ref.reference(c[0],c[1],c[2],c[3],c[4:])
                assert all(torch.equal(x.detach().cpu(),y) for x,y in zip(result,want))
                reference_calls.append(dict(queries=len(c[2]),valid=int(want[-1].sum())))
                return result
            with q.Preserve(m),torch.no_grad():
                m.train();q.seed(1901);b=prod.move_batch_to_device(copy.deepcopy(cpu),torch.device('cuda'),True,False)
                b.update(cva_compute_diagnostics=True,geometry_compute_diagnostics=False,cva_export_angle_feature=False)
                fixed._select_cdf_query_labels=checked
                try:ep=m(b)
                finally:fixed._select_cdf_query_labels=original_selector
                baseline=None;oldfirst=None;oldhashes=[];newhashes=[];oldvariation={};losses={}
                for version,fn in [('fixed',fixed.process_grasp_labels_cdf_width),('legacy',legacy.process_grasp_labels_cdf_width)]:
                    first=None
                    for repeat in range(20):
                        _,out=fn(dict(ep));now=labels(out);h=q.tree_hash(now)
                        (newhashes if version=='fixed' else oldhashes).append(h)
                        if first is None:
                            first=now
                            lv,out=prod.get_loss_economicgrasp(out,use_cdf=True)
                            losses[version]=dict(total=float(lv),coverage=q.loss_coverage(out),raw={k:float(v) for k,v in out.items() if k.startswith('B:') and torch.is_tensor(v) and v.numel()==1})
                        elif version=='fixed':assert all(torch.equal(first[k],v) for k,v in now.items()),'Fixed labels unstable'
                        else:
                            for k,v in changes(first,now).items():oldvariation[k]=max(oldvariation.get(k,0),v['changed'])
                    if version=='fixed':baseline=first
                    else:oldfirst=first
                delta=changes(oldfirst,baseline);query=torch.zeros_like(baseline['batch_valid_mask'],dtype=torch.bool)
                for k in ('batch_grasp_cdf_bins_angle_depth','batch_grasp_cdf_valid_mask','batch_grasp_width_angle_depth','batch_grasp_width_valid_mask_angle_depth'):
                    query|=(baseline[k]!=oldfirst[k]).flatten(2).any(2)
                records.append(dict(state=name,checkpoint=cp,split=split,frames=manifest[split]['frames'],input_hash=inputhash,
                    fixed_repeats=20,fixed_unique_hashes=len(set(newhashes)),legacy_repeats=20,legacy_unique_hashes=len(set(oldhashes)),legacy_max_changed_elements=oldvariation,
                    changed_query_count=int(query.sum()),query_count=query.numel(),changed_query_fraction=float(query.float().mean()),
                    differences_vs_first_legacy=delta,losses=losses,enumerated_object_calls=reference_calls,mode='train local batch2; fixed model output for every label replay'))
                del ep,b,out,lv
            assert q.tree_hash(cpu)==inputhash and q.tree_hash(m.state_dict())==state
            q.dump(a.output/'label_fix_audit.json',dict(status='running',matcher_sha256=q.sha(fixed.__file__),records=records))
            print(json.dumps(dict(state=name,split=split,seconds=time.time()-tick,fixed_unique=len(set(newhashes)),legacy_unique=len(set(oldhashes)),changed_query_fraction=records[-1]['changed_query_fraction'])),flush=True)
    q.dump(a.output/'label_fix_audit.json',dict(status='pass',matcher_sha256=q.sha(fixed.__file__),records=records,
        semantic_reference='CPU scalar query/angle/depth enumeration on actual selected object rows',tie_rule='normal orientation then lowest angle index at exactly equal keypoint squared distance',
        supervision_version='scene-query-forward-view-v1',training_allowed_after_P1_only=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--legacy-source',type=Path,required=True)
    label_audit(p.parse_args())
