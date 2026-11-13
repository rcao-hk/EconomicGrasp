"""Check all three official test splits used identical validation-locked alphas."""
from __future__ import annotations
import argparse
from pathlib import Path
from interactive_grasp.contracts import load_signed,sha256
from interactive_grasp.analyze import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split-dir',action='append',required=True)
    p.add_argument('--locked',required=True);p.add_argument('--output',required=True);p.add_argument('--markdown')
    p.add_argument('--bootstrap',type=int,default=5000);p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    lock=load_signed(a.locked)
    if lock.get('partition')!='validation' or lock.get('status')!='complete' or not lock.get('full_exact_labels'):
        raise ValueError('Cannot aggregate test results without a completed validation-only alpha lock')
    for item in a.split_dir:
        split,dir=item.split('=',1)
        m=load_signed(Path(dir)/'inference_manifest.json')
        if m.get('locked_validation_fingerprint')!=lock['fingerprint'] or m.get('locked_validation_sha256')!=sha256(a.locked):
            raise ValueError(f'{split}: validation-locked alpha changed between runs')
        if m.get('test_labels_used') is not False or m.get('partial_smoke'):
            raise ValueError('Test inference must be complete and use no test quality labels')
        if any(float(m['locked_alphas'][v])!=float(lock['choices'][v]['alpha']) for v in 'ABCD'):
            raise ValueError(f'{split}: different alpha from validation lock')
    summarize(a)

if __name__=='__main__':main()