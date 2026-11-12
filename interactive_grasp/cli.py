"""Explicit stage entrypoints; help never initializes a production model."""
from __future__ import annotations
import argparse
from pathlib import Path
from .contracts import make_plan,atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest='command',required=True)
    plan=sp.add_parser('plan');plan.add_argument('--output',required=True)
    def make(a):
        if Path(a.output).exists():raise FileExistsError(a.output)
        atomic_json(a.output,make_plan())
    plan.set_defaults(func=make)
    bind=sp.add_parser('bind')
    for k in ('template','base-checkpoint','selector-checkpoint','output'):bind.add_argument('--'+k,required=True)
    def do_bind(a):
        from .native import bind
        bind(a.template,a.base_checkpoint,a.selector_checkpoint,a.output)
    bind.set_defaults(func=do_bind)
    def runtime_args(parser):
        for k in ('config','plan','dataset-root','output'):parser.add_argument('--'+k,required=True)
        parser.add_argument('--gntrans-root',default='')
        parser.add_argument('--device',default='cuda:0')
        parser.add_argument('--provider',default='interactive_grasp.native:NativeRuntime',
                            help='Explicit strict historical adapter factory, if required; interface in workplan')
        parser.add_argument('--seed',type=int,default=0);parser.add_argument('--limit',type=int,default=0)
    export=sp.add_parser('export');runtime_args(export)
    export.add_argument('--partition',choices=('train','validation','test'),required=True)
    export.add_argument('--domain',choices=('original','gntrans'))
    export.add_argument('--labels',type=int,default=32)
    export.add_argument('--reuse-index',action='append',default=[])
    export.add_argument('--memory-dtype',choices=('float32','float16'),default='float32')
    export.add_argument('--reserve-gib',type=float,default=4.0)
    export.add_argument('--shard',type=int,default=0);export.add_argument('--shards',type=int,default=1)
    export.add_argument('--resume',action='store_true')
    def do_export(a):
        from .export import export
        export(a)
    export.set_defaults(func=do_export)
    index=sp.add_parser('index')
    for k in ('plan','cache','output'):index.add_argument('--'+k,required=True)
    index.add_argument('--partition',choices=('train','validation','test'),required=True)
    index.add_argument('--allow-partial',action='store_true')
    def do_index(a):
        from .export import index
        index(a)
    index.set_defaults(func=do_index)
    train=sp.add_parser('train')
    for k in ('train-index','val-index','output'):train.add_argument('--'+k,required=True)
    train.add_argument('--variant',choices=('A','B','C','D'),required=True)
    train.add_argument('--hidden',type=int,default=128);train.add_argument('--heads',type=int,default=4)
    train.add_argument('--layers',type=int,default=2);train.add_argument('--read-chunk',type=int,default=256)
    train.add_argument('--epochs',type=int,default=20);train.add_argument('--accumulate',type=int,default=4)
    train.add_argument('--max-pairs',type=int,default=2048);train.add_argument('--lr',type=float,default=1e-4)
    train.add_argument('--weight-decay',type=float,default=1e-4);train.add_argument('--seed',type=int,default=0)
    train.add_argument('--device',default='cuda:0');train.add_argument('--cpu-threads',type=int,default=1)
    train.add_argument('--resume',action='store_true');train.add_argument('--smoke',action='store_true')
    def do_train(a):
        from .train import fit
        fit(a)
    train.set_defaults(func=do_train)
    infer=sp.add_parser('infer');runtime_args(infer)
    infer.add_argument('--weight',action='append',required=True)
    infer.add_argument('--domain',choices=('original','gntrans'),default='original')
    infer.add_argument('--split',choices=('test_seen','test_similar','test_novel'),required=True)
    infer.add_argument('--reference-root',default='')
    def do_infer(a):
        from .infer import infer
        infer(a)
    infer.set_defaults(func=do_infer)
    evaluate=sp.add_parser('eval')
    evaluate.add_argument('--inference',required=True);evaluate.add_argument('--dataset-root',required=True)
    evaluate.add_argument('--workers',type=int,default=16)
    def do_eval(a):
        from .infer import evaluate
        evaluate(a)
    evaluate.set_defaults(func=do_eval)
    diagnose=sp.add_parser('diagnose')
    for k in ('index','weight','output'):diagnose.add_argument('--'+k,required=True)
    diagnose.add_argument('--partition',choices=('validation','test'),default='validation')
    diagnose.add_argument('--device',default='cuda:0');diagnose.add_argument('--seed',type=int,default=0)
    diagnose.add_argument('--duplicates',type=int,default=64);diagnose.add_argument('--limit',type=int,default=0)
    diagnose.add_argument('--allow-partial',action='store_true')
    def do_diagnose(a):
        from .analyze import diagnose
        diagnose(a)
    diagnose.set_defaults(func=do_diagnose)
    summary=sp.add_parser('summarize');summary.add_argument('--split-dir',action='append',required=True)
    summary.add_argument('--output',required=True);summary.add_argument('--markdown')
    summary.add_argument('--bootstrap',type=int,default=5000);summary.add_argument('--seed',type=int,default=0)
    def do_summary(a):
        from .analyze import summarize
        summarize(a)
    summary.set_defaults(func=do_summary)
    a=p.parse_args();a.func(a)


if __name__=='__main__':main()
