"""P0: attach GVAR action-reader features to existing exact-action RARE cache.

No CAD relabeling, no model training, no change to production forward. The
GVAR reader is captured at its native full Q*A*D shape and only the labelled
32-family ×4-depth actions are exported. Same frame/seed/physical ID required.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rare_grasp.pipeline import load_index, arrays, atomic_npz
from rare_grasp.core import digest, physical_ids, sha256, write_json


SCHEMA = "action-decision-reader-features-v1"


def gather_reader(reader, snapshot, sample):
    if len(reader) != 2 or any(t is None for t in reader):
        raise RuntimeError("Exactly one reader before/after capture is required")
    before, after = reader
    if before.ndim != 5 or before.shape != after.shape:
        raise RuntimeError("Reader must expose [B,Q,A,D,H] identical-shaped updates")
    if before.shape[:4] != (1,1024,12,4):
        raise RuntimeError(f"Unsupported native reader lattice: {before.shape}")
    ids = np.asarray(sample["family"], dtype=np.int64)
    d = np.asarray(sample["depth"], dtype=np.int64)
    if ids.min() < 0 or ids.max() >= 1024 or d.min() < 0 or d.max() >= 4:
        raise ValueError("Cached sample exceeds native query/depth lattice")
    angle=np.asarray(snapshot["angles"],dtype=np.int64)[ids]
    t=torch.as_tensor(ids,device=after.device)
    a=torch.as_tensor(angle,device=after.device)
    di=torch.as_tensor(d,device=after.device)
    readout=after[0,t,a,di].float().cpu().numpy()
    delta=(after[0,t,a,di]-before[0,t,a,di]).float().cpu().numpy()
    if not np.isfinite(readout).all() or not np.isfinite(delta).all():
        raise FloatingPointError("Nonfinite GVAR reader feature")
    if not np.array_equal(angle,np.asarray(sample["native_angle"])):
        raise RuntimeError("Selected angle differs from exact-action cache")
    native=np.asarray(snapshot["grasps"],dtype=np.float32)[ids,d]
    actual=np.asarray(sample["grasps"],dtype=np.float32)
    if not np.array_equal(physical_ids(native),physical_ids(actual)):
        raise RuntimeError("Executed physical grasps changed between reader capture and labels")
    if np.max(np.abs(native[:,:16]-actual[:,:16])) > 1e-6:
        raise RuntimeError("Native decoded 17D action mismatch")
    baseline=np.asarray(snapshot["base_logit"],dtype=np.float32)[ids,d]
    if np.max(np.abs(baseline-sample["base_logit"])) > 1e-5:
        raise RuntimeError("Original CDF logit replay mismatch")
    feat=np.asarray(snapshot["features"],dtype=np.float32)[ids,d]
    if np.max(np.abs(feat-sample["features"])) > 1e-5:
        raise RuntimeError("Native score-branch feature replay mismatch")
    return dict(reader=readout,reader_delta=delta,query=ids,depth=d,angle=angle)


def capture_one(runtime, row: dict, seed: int, sample: dict):
    module=getattr(runtime.model.kview_grasp_module.decoder,"reader",None)
    if module is None or module.training:
        raise RuntimeError("P0 requires a frozen Point+GVAR volume reader")
    cache={}
    def pre(_m,args):
        if "before" in cache or not args: raise RuntimeError("Reader invoked more than once")
        cache["before"]=args[0].detach()
    def post(_m,args,out):
        if "after" in cache: raise RuntimeError("Reader invoked more than once")
        cache["after"]=out.detach()
    h1=module.register_forward_pre_hook(pre)
    h2=module.register_forward_hook(post)
    try:
        snap=runtime.snapshot(row,seed)
        if set(cache)!={"before","after"}:
            raise RuntimeError("GVAR reader was not captured")
        return gather_reader((cache["before"],cache["after"]),snap,sample)
    finally:
        h2.remove();h1.remove()
        cache.clear()


def _sidecar(path, source):
    meta=path.with_suffix(".json")
    if path.exists() or meta.exists():
        if not (path.exists() and meta.exists()):
            raise RuntimeError(f"Orphaned feature cache: {path}")
        rec=json.loads(meta.read_text(encoding="utf-8"))
        if rec["contract"]!=source or rec["contract_sha256"]!=digest(source):
            raise RuntimeError(f"Feature cache contract changed: {path}")
        if sha256(path)!=rec["npz_sha256"]:
            raise RuntimeError(f"Reader feature cache modified: {path}")
        return rec
    return None


def run_capture(args):
    from rare_grasp.runtime import Runtime, read_bound_config
    idx=load_index(args.index,"train",allow_partial=args.smoke)
    base_config=read_bound_config(args.config)
    if digest(base_config)!=idx["contract"]["config_digest"]:
        raise ValueError("Base checkpoint does not match cached exact-action labels")
    if base_config["backbone"]!="point_gvar":
        raise ValueError("Requires GVAR reader, not Point-only scorer")
    seed=int(idx["contract"]["seed"])
    rows=idx["entries"]
    if args.scene_ids:
        if not args.smoke:raise ValueError("--scene-ids only permitted for a smoke/partial capture")
        scenes={int(x) for x in args.scene_ids.split(",") if x}
        if not scenes or any(x<0 or x>=100 for x in scenes):
            raise ValueError("Invalid train-domain smoke scene IDs")
        rows=[e for e in rows if int(e["row"]["scene"]) in scenes]
    if args.max_per_scene:
        if not args.smoke:raise ValueError("--max-per-scene only permitted for smoke")
        counts={};limited=[]
        for e in rows:
            key=(e["row"]["domain"],e["row"]["scene"])
            if counts.get(key,0)>=args.max_per_scene:continue
            counts[key]=counts.get(key,0)+1;limited.append(e)
        rows=limited
    if not 0<=args.shard_index<args.num_shards:
        raise ValueError("Invalid shard identity")
    rows=[e for i,e in enumerate(rows) if i%args.num_shards==args.shard_index]
    if args.limit:
        if not args.smoke:raise ValueError("--limit is only permitted in --smoke")
        rows=rows[:args.limit]
    if not rows:raise ValueError("No frames assigned to this shard")
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    runtime=Runtime(args.config,args.dataset_root,args.gntrans_root,args.device)
    source=dict(schema=SCHEMA,rare_train_index_sha256=idx["index_sha256"],
                config_digest=runtime.config_digest,base_sha256=base_config["checkpoint_sha256"],
                native_seed=seed,reader_variant=str(
                    runtime.model.kview_grasp_module.decoder.reader.cfg.variant),
                feature_definition="reader output and residual after action-aligned evidence before adapter.finish",
                dtype=args.dtype,action_identity="float32 executed grasp[1:16]")
    done=0
    for e in rows:
        row=e["row"]
        key=f'{row["domain"]}_scene{row["scene"]:04d}_frame{row["anno"]:04d}'
        out=root/"frames"/(key+".npz")
        contract={**source,"frame":row,"source_label_sha256":e["sha256"]}
        if _sidecar(out,contract) is not None:
            done+=1;continue
        arr=arrays(e)
        result=capture_one(runtime,row,seed,arr)
        out.parent.mkdir(parents=True,exist_ok=True)
        dtype=np.float16 if args.dtype=="float16" else np.float32
        atomic_npz(out,reader=result["reader"].astype(dtype),
                   reader_delta=result["reader_delta"].astype(dtype),
                   family=result["query"],depth=result["depth"],
                   angle=result["angle"],physical_id=arr["physical_id"])
        write_json(out.with_suffix(".json"),dict(contract=contract,
                   contract_sha256=digest(contract),
                   npz_sha256=sha256(out),output_shape=list(result["reader"].shape)))
        done+=1
        if done%50==0 or done==len(rows):
            print(f"[P0 cache] shard {args.shard_index}: {done}/{len(rows)}",flush=True)
    print(json.dumps({"completed":done,"shard_index":args.shard_index},indent=2))


def run_index(args):
    idx=load_index(args.index,"train",allow_partial=args.smoke)
    root=Path(args.cache)
    entries=[]
    base_contract=None
    for e in idx["entries"]:
        r=e["row"];key=f'{r["domain"]}_scene{r["scene"]:04d}_frame{r["anno"]:04d}'
        file=root/"frames"/(key+".npz")
        meta=file.with_suffix(".json")
        if not file.is_file() or not meta.is_file():
            if args.smoke:continue
            raise RuntimeError(f"Missing reader feature: {file}")
        rec=json.loads(meta.read_text(encoding="utf-8"))
        c=rec["contract"]
        if (c["frame"]!=r or c["source_label_sha256"]!=e["sha256"] or
            c["rare_train_index_sha256"]!=idx["index_sha256"] or
            rec["contract_sha256"]!=digest(c) or sha256(file)!=rec["npz_sha256"]):
            raise RuntimeError(f"Feature provenance mismatch: {file}")
        keyless={k:v for k,v in c.items() if k not in ("frame","source_label_sha256")}
        if base_contract is None:base_contract=keyless
        elif base_contract!=keyless:raise ValueError("Feature sources/dtypes mixed")
        with np.load(file,allow_pickle=False) as z:
            source=arrays(e)
            if z["reader"].shape!=(len(source["count"]),256) or z["reader_delta"].shape!=z["reader"].shape:
                raise RuntimeError("Unexpected GVAR reader hidden dimension")
            for k in ("family","depth","physical_id"):
                expected="family" if k=="family" else k
                if not np.array_equal(z[k],source[expected]):
                    raise RuntimeError(f"Reader cache action identity mismatch: {k}")
        entries.append(dict(row=r,source=e,reader_file=str(file.resolve()),
                            reader_sha256=rec["npz_sha256"]))
    if not entries:raise RuntimeError("Empty feature index")
    result=dict(schema=SCHEMA,complete=len(entries)==len(idx["entries"]),
                source_train_index_sha256=idx["index_sha256"],
                base_contract=base_contract,entries=entries,
                source_entries=len(idx["entries"]),reader_entries=len(entries))
    result["index_sha256"]=digest(result)
    write_json(args.output,result)
    print(json.dumps(dict(complete=result["complete"],entries=len(entries))))


def load_feature_index(path: str, allow_partial=False):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    saved=data.pop("index_sha256")
    if digest(data)!=saved or data["schema"]!=SCHEMA:
        raise RuntimeError("Reader feature index signature/schema mismatch")
    data["index_sha256"]=saved
    if not data["complete"] and not allow_partial:
        raise ValueError("Incomplete P0 cache forbidden in formal experiments")
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="cmd",required=True)
    c=sub.add_parser("capture")
    c.add_argument("--config",required=True)
    c.add_argument("--index",required=True)
    c.add_argument("--dataset-root",required=True)
    c.add_argument("--gntrans-root",required=True)
    c.add_argument("--output",required=True)
    c.add_argument("--device",default="cuda:0")
    c.add_argument("--num-shards",type=int,default=1)
    c.add_argument("--shard-index",type=int,default=0)
    c.add_argument("--dtype",choices=("float16","float32"),default="float16")
    c.add_argument("--limit",type=int,default=0)
    c.add_argument("--scene-ids",default="")
    c.add_argument("--max-per-scene",type=int,default=0)
    c.add_argument("--smoke",action="store_true")
    c.set_defaults(func=run_capture)
    i=sub.add_parser("index")
    i.add_argument("--index",required=True)
    i.add_argument("--cache",required=True)
    i.add_argument("--output",required=True)
    i.add_argument("--smoke",action="store_true")
    i.set_defaults(func=run_index)
    a=p.parse_args();a.func(a)

if __name__=="__main__":
    main()
