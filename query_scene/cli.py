"""Explicit-stage CLI: no automatic long-running work."""
import argparse
from pathlib import Path
from .core import make_plan,atomic_json


def main():
    from . import pipeline
    from .diagnose import diagnose,benchmark
    from .report import summarize
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--output',required=True)
    def prepare(a):
        if Path(a.output).exists():raise FileExistsError(a.output)
        atomic_json(a.output,make_plan())
    q.set_defaults(func=prepare)
    q=sub.add_parser('bind')
    for k in ('template','base-checkpoint','selector-checkpoint','output'):q.add_argument('--'+k,required=True)
    def bind(a):
        from interactive_grasp.native import bind as f
        f(a.template,a.base_checkpoint,a.selector_checkpoint,a.output)
    q.set_defaults(func=bind)
    def runtime_args(q):
        for k in ('config','plan','dataset-root','output'):q.add_argument('--'+k,required=True)
        q.add_argument('--gntrans-root',default='');q.add_argument('--device',default='cuda:0')
        q.add_argument('--provider',default='interactive_grasp.native:NativeRuntime')
        q.add_argument('--grid-size',type=int,default=32);q.add_argument('--seed',type=int,default=0)
        q.add_argument('--limit',type=int,default=0)
    q=sub.add_parser('export');runtime_args(q);q.set_defaults(func=pipeline.export)
    q.add_argument('--partition',choices=('train','validation'),required=True);q.add_argument('--domain',choices=('original','gntrans'))
    q.add_argument('--labels',type=int,required=True);q.add_argument('--reuse-index',action='append',default=[])
    q.add_argument('--shards',type=int,default=1);q.add_argument('--shard',type=int,default=0)
    q.add_argument('--reserve-gib',type=int,default=4);q.add_argument('--resume',action='store_true')
    q=sub.add_parser('index');q.set_defaults(func=pipeline.index)
    for k in ('plan','cache','output'):q.add_argument('--'+k,required=True)
    q.add_argument('--partition',choices=('train','validation'),required=True);q.add_argument('--smoke',action='store_true')
    q=sub.add_parser('train');q.set_defaults(func=pipeline.fit)
    for k in ('train-index','val-index','output'):q.add_argument('--'+k,required=True)
    q.add_argument('--variant',choices=('local','candidate','scene'),required=True)
    q.add_argument('--epochs',type=int,default=20);q.add_argument('--accumulate',type=int,default=4)
    q.add_argument('--hidden',type=int,default=128);q.add_argument('--heads',type=int,default=4);q.add_argument('--layers',type=int,default=2)
    q.add_argument('--lr',type=float,default=1e-4);q.add_argument('--weight-decay',type=float,default=1e-4)
    q.add_argument('--max-pairs',type=int,default=2048);q.add_argument('--seed',type=int,default=0)
    q.add_argument('--residual-scale',type=float,default=.2)
    q.add_argument('--train-scene-shuffle',action='store_true');q.add_argument('--cpu-threads',type=int,default=1)
    q.add_argument('--device',default='cuda:0');q.add_argument('--smoke',action='store_true');q.add_argument('--resume',action='store_true')
    q=sub.add_parser('diagnose');q.set_defaults(func=diagnose)
    q.add_argument('--index',required=True);q.add_argument('--weight',action='append',required=True);q.add_argument('--output',required=True)
    q.add_argument('--device',default='cuda:0');q.add_argument('--limit',type=int,default=0);q.add_argument('--smoke',action='store_true')
    q.add_argument('--bootstrap',type=int,default=5000);q.add_argument('--seed',type=int,default=0)
    q.add_argument('--permutation-tol',type=float,default=1e-5)
    q=sub.add_parser('benchmark');q.set_defaults(func=benchmark)
    q.add_argument('--index',required=True);q.add_argument('--weight',action='append',required=True);q.add_argument('--output',required=True)
    q.add_argument('--device',default='cuda:0');q.add_argument('--smoke',action='store_true')
    q.add_argument('--warmup',type=int,default=5);q.add_argument('--repeats',type=int,default=50)
    q=sub.add_parser('infer');runtime_args(q);q.set_defaults(func=pipeline.infer)
    q.add_argument('--weight',action='append',required=True);q.add_argument('--domain',choices=('original','gntrans'),default='original')
    q.add_argument('--split',choices=('test_seen','test_similar','test_novel'),required=True);q.add_argument('--reference-root',default='')
    q=sub.add_parser('eval');q.set_defaults(func=pipeline.official_eval)
    q.add_argument('--inference',required=True);q.add_argument('--dataset-root',required=True);q.add_argument('--workers',type=int,default=16)
    q=sub.add_parser('summarize');q.set_defaults(func=summarize)
    q.add_argument('--split-dir',action='append',required=True);q.add_argument('--output',required=True);q.add_argument('--markdown')
    q.add_argument('--bootstrap',type=int,default=5000);q.add_argument('--seed',type=int,default=0)
    a=p.parse_args();a.func(a)

if __name__=='__main__':main()
