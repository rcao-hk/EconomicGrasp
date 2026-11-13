"""Merge independently mined P1 shards; verify contracts, hashes and paired budgets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rare_grasp.core import validate_cache
from .core import SCHEMA,digest,sha


def load_part(path):
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    signature=data.pop("index_sha256",None)
    if signature!=digest(data) or data.get("schema")!=SCHEMA or data.get("kind")!="decision-cache-index":
        raise ValueError(f"Invalid P1 index signature/schema: {path}")
    data["index_sha256"]=signature
    if data.get("expected_entries")!=len(data.get("entries",[])):
        raise ValueError(f"P1 shard incomplete: {path}")
    return data


def merge(paths,output,min_scenes=100):
    if not paths:raise ValueError("No shard indexes")
    shards=[load_part(path) for path in paths]
    reference=shards[0]
    for shard in shards[1:]:
        for k in ("schema","kind","config_digest","base_checkpoint_sha256","contract"):
            if shard[k]!=reference[k]:
                raise ValueError(f"Mixed P1 cache contracts: {k}")
    entries=[]
    keys=set()
    for shard in shards:
        for e in shard["entries"]:
            row=e["row"]
            key=(row["domain"],row["scene"],row["anno"],e["strategy"])
            if key in keys:raise ValueError(f"Duplicate P1 strategy/frame across shards: {key}")
            keys.add(key)
            if e["strategy"] not in ("uniform","decision"):
                raise ValueError("Unexpected sampling strategy")
            if sha(e["file"])!=e["sha256"]:
                raise ValueError(f"Missing/corrupt P1 cache: {e['file']}")
            with np.load(e["file"],allow_pickle=False) as z:
                a={k:z[k] for k in z.files}
            validate_cache(a)
            if len(a["family"])!=64*4:
                raise ValueError("P1 must have exactly 64 4-depth action families")
            entries.append(e)
    byframe={}
    for e in entries:
        r=e["row"];k=(r["domain"],r["scene"],r["anno"])
        byframe.setdefault(k,set()).add(e["strategy"])
    if any(v!={"uniform","decision"} for v in byframe.values()):
        raise ValueError("Missing a matched uniform/decision frame arm")
    scenes={r[1] for r in byframe}
    if len(scenes)<min_scenes:
        raise ValueError(f"Insufficient scene coverage: {len(scenes)} < {min_scenes}")
    entries.sort(key=lambda e:(e["row"]["domain"],e["row"]["scene"],
                               e["row"]["anno"],e["strategy"]))
    data=dict(schema=SCHEMA,kind="decision-cache-index",
              config_digest=reference["config_digest"],
              base_checkpoint_sha256=reference["base_checkpoint_sha256"],
              contract=reference["contract"],entries=entries,
              expected_entries=len(entries),complete=True,
              source_shard_sha256={str(p):sha(p) for p in paths},
              matched_frame_arms=len(byframe),
              unique_scenes=len(scenes))
    data["index_sha256"]=digest(data)
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,indent=2,ensure_ascii=False))
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard-index",required=True,action="append")
    p.add_argument("--output",required=True)
    p.add_argument("--min-scenes",type=int,default=100)
    a=p.parse_args()
    data=merge(a.shard_index,a.output,a.min_scenes)
    print(f"Merged {len(data['entries'])} validated P1 arms across {data['unique_scenes']} scenes")


if __name__=="__main__":
    main()
