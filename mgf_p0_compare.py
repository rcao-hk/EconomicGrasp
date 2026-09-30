#!/usr/bin/env python3
"""Validate P0-1 matching and summarize official NPY results when available."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--variants',default='feature_only,full,cva')
    a=p.parse_args(); root=Path(a.root)
    reference=None; source_state=None; result={}
    for variant in a.variants.split(','):
        directory=root/variant/'train'
        protocol=json.loads((directory/'protocol.json').read_text())
        normalized=dict(protocol); normalized.pop('variant')
        if reference is not None and normalized!=reference:
            raise RuntimeError(f'{variant} training protocol differs beyond variant')
        reference=normalized
        history=json.loads((directory/'metrics.json').read_text())
        if [x['epoch'] for x in history]!=list(range(protocol['epochs'])):
            raise RuntimeError(f'{variant} fine-tuning incomplete')
        contract=json.loads((directory/'gradient_contract.json').read_text())
        if source_state is not None and contract['frozen_state_sha256']!=source_state:
            raise RuntimeError('Variants did not use byte-identical frozen source')
        source_state=contract['frozen_state_sha256']
        if not all(x['frozen_state_unchanged'] for x in history): raise RuntimeError('Frozen source changed')
        result[variant]=dict(trainable_parameters=contract['trainable_parameters'],
                             last_validation=history[-1]['validation'],official={})
    result['base']={'official':{}}
    for variant,data in result.items():
        for mode in ('off','on'):
            scores={}
            for split in ('test_seen','test_similar','test_novel'):
                directory=root/variant/f'test_collision_{mode}'/'official'/split
                if (directory/'summary.json').is_file():
                    s=json.loads((directory/'summary.json').read_text())
                    accuracy=np.load(directory/'accuracy.npy',allow_pickle=False)
                    if not np.isfinite(accuracy).all() or accuracy.ndim!=4 or accuracy.shape[-2:]!=(50,6):
                        raise RuntimeError('Malformed official accuracy NPY')
                    mean=float(accuracy.mean())
                    if abs(mean-float(s['mean_accuracy']))>1e-7: raise RuntimeError('Summary/NPY mismatch')
                    scores[split]=mean
            if scores: data['official'][mode]=scores
    out=root/'comparison'; out.mkdir(exist_ok=True)
    (out/'comparison.json').write_text(json.dumps(dict(protocol=reference,results=result),indent=2)+'\n')
    lines=['# Frozen-source P0-1 comparison','',
           'Additional online fine-tuning; fixed source parameters, buffers and candidate generator.',
           'Feature-only/full have identical trainable shapes. CVA is a private scorer control, not parameter-count matched.','',
           '| Variant | Collision | Seen | Similar | Novel | Mean |','|---|---|---:|---:|---:|---:|']
    for name,data in result.items():
        for mode,s in data['official'].items():
            vals=[s.get(k) for k in ('test_seen','test_similar','test_novel')]
            mean=float(np.mean(vals)) if all(v is not None for v in vals) else None
            fmt=lambda v: 'pending' if v is None else f'{100*v:.3f}'
            lines.append('| '+ ' | '.join([name,mode]+[fmt(v) for v in vals]+[fmt(mean)])+' |')
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__': main()
