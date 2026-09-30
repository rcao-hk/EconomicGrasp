#!/usr/bin/env python3
"""Validate P1 protocols and summarize official accuracy arrays."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--family',choices=['p1_1','p1_3'],required=True)
    p.add_argument('--variants',required=True)
    return p


def main():
    a=parser().parse_args(); root=Path(a.root)
    variants=[x for x in a.variants.split(',') if x]
    if len(set(variants))!=len(variants): raise ValueError('Duplicate variants')
    reference=None; source_sha=None; result={}
    for variant in variants:
        train=root/variant/'train'
        protocol=json.loads((train/'protocol.json').read_text())
        if protocol['family']!=a.family or protocol['variant']!=variant:
            raise RuntimeError(f'Wrong family/variant in {train}')
        normalized=dict(protocol); normalized.pop('variant')
        if reference is not None and normalized!=reference:
            raise RuntimeError(f'{variant}: protocol differs beyond variant')
        reference=normalized
        hist=json.loads((train/'metrics.json').read_text())
        if [x['epoch'] for x in hist]!=list(range(protocol['epochs'])):
            raise RuntimeError(f'{variant}: fine-tuning incomplete')
        contract=json.loads((train/'gradient_contract.json').read_text())
        if source_sha is not None and contract['frozen_state_sha256']!=source_sha:
            raise RuntimeError('P1 variants did not use byte-identical frozen source')
        source_sha=contract['frozen_state_sha256']
        if not all(x['frozen_state_unchanged'] for x in hist):
            raise RuntimeError(f'{variant}: frozen source changed')
        result[variant]=dict(
            trainable_parameters=contract['trainable_parameters'],
            validation=hist[-1]['validation'],
            official={}
        )
        for mode in ('off','on'):
            scores={}
            for split in ('test_seen','test_similar','test_novel'):
                d=root/variant/f'test_collision_{mode}'/'official'/split
                if not (d/'summary.json').is_file(): continue
                s=json.loads((d/'summary.json').read_text())
                arr=np.load(d/'accuracy.npy',allow_pickle=False)
                if arr.ndim!=4 or arr.shape[-2:]!=(50,6) or not np.isfinite(arr).all():
                    raise RuntimeError(f'Malformed official accuracy {d}')
                mean=float(arr.mean())
                if abs(mean-float(s['mean_accuracy']))>1e-7:
                    raise RuntimeError(f'Summary/NPY mismatch {d}')
                scores[split]=mean
            if scores: result[variant]['official'][mode]=scores
    out=root/'comparison'; out.mkdir(exist_ok=True)
    payload=dict(family=a.family,protocol=reference,results=result)
    (out/'comparison.json').write_text(json.dumps(payload,indent=2)+'\n')
    lines=[f'# {a.family} frozen-source comparison','',
           '| Variant | Collision | Seen | Similar | Novel | Mean |',
           '|---|---|---:|---:|---:|---:|']
    for variant,data in result.items():
        for mode,scores in data['official'].items():
            vals=[scores.get(k) for k in ('test_seen','test_similar','test_novel')]
            mean=float(np.mean(vals)) if all(v is not None for v in vals) else None
            fmt=lambda x:'pending' if x is None else f'{100*x:.3f}'
            lines.append('| '+' | '.join([variant,mode]+[fmt(x) for x in vals]+[fmt(mean)])+' |')
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__': main()
