"""Shared visible geometry: prepare/bind/export/index/train/diagnose/infer/eval/report."""
from __future__ import annotations
import argparse
from . import data, experiment, inference, report, paired


def main():
    p = argparse.ArgumentParser(description=__doc__); sp = p.add_subparsers(dest='cmd', required=True)
    x = sp.add_parser('prepare'); x.add_argument('--output', required=True)
    def prepare(a):
        from pathlib import Path
        if Path(a.output).exists(): raise FileExistsError(a.output)
        data.atomic_json(a.output, data.make_plan())
    x.set_defaults(func=prepare)
    x = sp.add_parser('bind')
    for k in ('template', 'base-checkpoint', 'selector-checkpoint', 'output'): x.add_argument('--'+k, required=True)
    def bind(a):
        from interactive_grasp.native import bind
        bind(a.template, a.base_checkpoint, a.selector_checkpoint, a.output)
    x.set_defaults(func=bind)
    def runtime(x):
        for k in ('config', 'dataset-root', 'output', 'plan'): x.add_argument('--'+k, required=True)
        x.add_argument('--gntrans-root', default='')
        x.add_argument('--provider', default='interactive_grasp.native:NativeRuntime')
        x.add_argument('--device', default='cuda:0'); x.add_argument('--seed', type=int, default=0)
        x.add_argument('--limit', type=int, default=0)
    x = sp.add_parser('export'); runtime(x); x.set_defaults(func=data.export)
    x.add_argument('--targets', required=True); x.add_argument('--partition', choices=('train', 'validation'), required=True)
    x.add_argument('--domain', choices=('original', 'gntrans', 'both'), default='both')
    x.add_argument('--families', type=int, default=16); x.add_argument('--field-points', type=int, default=4096)
    x.add_argument('--tau', type=float, default=.03); x.add_argument('--band', type=float, default=.002)
    x.add_argument('--shard', type=int, default=0); x.add_argument('--shards', type=int, default=1)
    x.add_argument('--resume', action='store_true')
    x = sp.add_parser('index'); x.set_defaults(func=data.index)
    for k in ('plan', 'cache', 'output'): x.add_argument('--'+k, required=True)
    x.add_argument('--partition', choices=('train', 'validation'), required=True); x.add_argument('--smoke', action='store_true')
    x = sp.add_parser('train'); x.set_defaults(func=experiment.fit)
    for k in ('train-index', 'val-index', 'output'): x.add_argument('--'+k, required=True)
    x.add_argument('--mode', choices=experiment.MODES, required=True)
    x.add_argument('--hidden', type=int, default=64); x.add_argument('--epochs', type=int, default=20)
    x.add_argument('--accumulate', type=int, default=4); x.add_argument('--geometry-weight', type=float, default=1.)
    x.add_argument('--lr', type=float, default=1e-4); x.add_argument('--weight-decay', type=float, default=1e-4)
    x.add_argument('--seed', type=int, default=0); x.add_argument('--device', default='cuda:0')
    x.add_argument('--cpu-threads', type=int, default=2); x.add_argument('--smoke', action='store_true')
    x.add_argument('--resume', action='store_true')
    x = sp.add_parser('diagnose'); x.set_defaults(func=experiment.diagnose)
    for k in ('index', 'weight', 'output'): x.add_argument('--'+k, required=True)
    x.add_argument('--device', default='cuda:0'); x.add_argument('--smoke', action='store_true')
    x = sp.add_parser('paired'); x.set_defaults(func=paired.diagnose_paired)
    for k in ('index','weight','output'): x.add_argument('--'+k,required=True)
    x.add_argument('--device',default='cuda:0');x.add_argument('--smoke',action='store_true')
    x = sp.add_parser('infer'); runtime(x); x.set_defaults(func=inference.infer)
    x.add_argument('--weight', required=True); x.add_argument('--reference-root', default='')
    x.add_argument('--split', required=True, choices=('test_seen', 'test_similar', 'test_novel'))
    x.add_argument('--torch-threads', type=int, default=2)
    x = sp.add_parser('eval'); x.set_defaults(func=inference.evaluate)
    x.add_argument('--inference', required=True); x.add_argument('--dataset-root', required=True)
    x.add_argument('--workers', type=int, default=8)
    x = sp.add_parser('summarize'); x.set_defaults(func=report.summarize)
    x.add_argument('--split-dir', action='append', required=True); x.add_argument('--output', required=True)
    x.add_argument('--bootstrap', type=int, default=5000); x.add_argument('--seed', type=int, default=0)
    a = p.parse_args(); a.func(a)


if __name__ == '__main__': main()
