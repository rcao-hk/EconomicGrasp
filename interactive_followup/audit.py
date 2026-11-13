"""Audit original A/B/C/D training curves and exact provenance; never retrain."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from interactive_grasp.contracts import load_signed, sha256, parameter_source_hash, atomic_json, signed


def audit(run_root, acceptance=None):
    root=Path(run_root)
    output={'status':'checked','current_interactive_code_hash':parameter_source_hash(),
            'run_root':str(root.resolve()),'variants':{},'missing':[],'protocol_issues':[],
            'interpretation':'Original Stage1 model records only; no new training or test selection.'}
    compare={}
    for v in 'ABCD':
        sub=root/'models'/v; item={'variant':v}
        for name in ('protocol.json','history.json','validation_last.json','complete.json','last.pt'):
            f=sub/name
            if not f.exists():output['missing'].append(str(f));continue
            item[name.replace('.json','').replace('.pt','')+'_sha256']=sha256(f)
        if not (sub/'protocol.json').exists() or not (sub/'history.json').exists():
            output['variants'][v]=item;continue
        p=load_signed(sub/'protocol.json')
        h=json.loads((sub/'history.json').read_text())
        if p['model']['variant']!=v:output['protocol_issues'].append(f'{v}: model variant mismatch')
        if p.get('source_code_sha256')!=parameter_source_hash():
            item['source_code_hash_status']='DIFFERS_FROM_CHECKOUT'
            output['protocol_issues'].append(f'{v}: source code hash differs; read actual server diff before interpreting')
        else:item['source_code_hash_status']='MATCH'
        if not h or h[0]['epoch']!=0 or len(h)<2 or any(not np.isfinite(float(x.get('validation',{}).get('mse', x.get('train_mse',0)))) for x in h[1:]):
            output['protocol_issues'].append(f'{v}: invalid/incomplete history')
        latest=h[-1]; item['epochs_recorded']=len(h)-1;item['last_epoch']=latest['epoch'];
        item['history_summary']=[{'epoch':r['epoch'],
                                  'updates':r.get('updates'),
                                  'train_mse':r.get('train_mse'),
                                  'train_pair':r.get('train_pair'),
                                  'validation':r.get('validation')} for r in h]
        if latest['epoch']!=p['hyper']['epochs']:
            output['protocol_issues'].append(f'{v}: incomplete planned epochs')
        if (sub/'validation_last.json').exists():
            val=json.loads((sub/'validation_last.json').read_text())
            item['last_validation_summary']=val.get('summary')
            for k in ('mse','pair','labelled_subset_top1_q'):
                x=val.get('summary',{}).get(k)
                y=latest.get('validation',{}).get(k)
                if x is not None and y is not None and abs(float(x)-float(y))>1e-8:
                    output['protocol_issues'].append(f'{v}: final validation mismatch for {k}')
        norm=dict(p);norm.pop('fingerprint',None);norm.pop('model_description',None)
        norm['model']=dict(norm['model']);norm['model'].pop('variant',None)
        compare[v]=norm
        output['variants'][v]=item
    if len(compare)==4 and any(p!=compare['A'] for p in compare.values()):
        output['protocol_issues'].append('A/B/C/D shared source and training budget differ')
    if acceptance:
        a=json.loads(Path(acceptance).read_text())
        output['acceptance_sha256']=sha256(acceptance)
        if a.get('source_code_sha256') and a['source_code_sha256']!=parameter_source_hash():
            output['protocol_issues'].append('Acceptance source hash differs from checkout')
        for v in 'ABCD':
            ck=a.get('weights',{}).get(v)
            actual=output['variants'].get(v,{}).get('last_sha256')
            if ck and actual and ck!=actual:
                output['protocol_issues'].append(f'{v}: acceptance checkpoint hash mismatch')
    if output['missing'] or output['protocol_issues']:output['status']='needs_investigation'
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',required=True);p.add_argument('--acceptance')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    path=Path(a.output)
    if path.exists():raise FileExistsError(path)
    obj=audit(a.run_root,a.acceptance)
    atomic_json(path,signed(obj))
    print(json.dumps({'status':obj['status'],'missing':len(obj['missing']),
                      'issues':obj['protocol_issues']},indent=2,ensure_ascii=False))

if __name__=='__main__':main()