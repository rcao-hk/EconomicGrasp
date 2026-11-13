"""Index P1b competition-labelled actions + paired GVAR reader features.

Run only after all shards finished; verify every source and all physical IDs.
For smoke, --smoke permits partial subsets (never formal training).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rare_grasp.pipeline import index_cache,load_index,arrays
from rare_grasp.core import write_json,sha256,digest
from diagnostics.action_decision.feature_cache import (
    SCHEMA as FEATURE_SCHEMA,run_index as index_reader
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol",required=True,help="Exact MIX50 paired sampling JSON")
    p.add_argument("--root",required=True,help="P1b root containing labels/ and features/")
    p.add_argument("--output",required=True,help="Directory for new source and feature indexes")
    p.add_argument("--source-index",required=True,help="Original uniform32 labels index")
    p.add_argument("--smoke",action="store_true")
    a=p.parse_args()
    root=Path(a.root);out=Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("P1b indexes must be produced in a new empty directory")
    out.mkdir(parents=True,exist_ok=True)

    label_index_path=out/"train_index.json"
    index_cache(SimpleNamespace(protocol=a.protocol,cache=str(root/"labels"),
                                output=str(label_index_path),partition="train",
                                allow_partial=a.smoke))
    combined=load_index(label_index_path,"train",allow_partial=a.smoke)
    original=load_index(a.source_index,"train",allow_partial=False)
    c=combined["contract"]
    orig=original["contract"]
    if (c["config_digest"]!=orig["config_digest"] or
        c["protocol_sha256"]!=orig["protocol_sha256"] or
        c["seed"]!=orig["seed"] or c["families"]!=64 or
        c["sampling"]!="uniform32_plus_native_prefilter_NMS_competitors32"):
        raise RuntimeError("P1b cache protocol mismatch to uniform source")
    source_by_frame={(e["row"]["domain"],e["row"]["scene"],e["row"]["anno"]):e
                     for e in original["entries"]}
    for e in combined["entries"]:
        row=e["row"]
        key=f'{row["domain"]}_scene{row["scene"]:04d}_frame{row["anno"]:04d}'
        feat=root/"features"/"frames"/(key+".npz")
        if not feat.exists():raise FileNotFoundError(feat)
        old=source_by_frame[(row["domain"],row["scene"],row["anno"])]
        source_meta=(root/"labels"/(key+".json"))
        if not source_meta.exists():raise FileNotFoundError(source_meta)
        import json
        provenance=json.loads(source_meta.read_text(encoding="utf-8"))
        if provenance["source_label_sha256"]!=old["sha256"] or provenance["source_index_sha256"]!=original["index_sha256"]:
            raise RuntimeError(f"P1b source reuse identity mismatch: {key}")
        with np.load(feat,allow_pickle=False) as z:
            labels=arrays(e)
            for name in ("physical_id","family","depth"):
                if not np.array_equal(z[name],labels[name]):
                    raise RuntimeError(f"P1b reader action identity mismatch: {key}/{name}")
            if z["reader"].shape!=(len(labels["count"]),256):
                raise RuntimeError("Unexpected reader hidden feature size")
        file_contract=dict(
            schema=FEATURE_SCHEMA,
            rare_train_index_sha256=combined["index_sha256"],
            config_digest=c["config_digest"],
            base_sha256=c["config"]["checkpoint_sha256"],
            native_seed=c["seed"],
            reader_variant="volume_rel",
            feature_definition="reader output and residual after action-aligned evidence before adapter.finish",
            dtype="float16",
            action_identity="float32 executed grasp[1:16]",
            frame=row,source_label_sha256=e["sha256"],
        )
        sidecar=dict(contract=file_contract,
                     contract_sha256=digest(file_contract),
                     npz_sha256=sha256(feat),
                     output_shape=[len(labels["count"]),256])
        write_json(feat.with_suffix(".json"),sidecar)
    index_reader(SimpleNamespace(index=str(label_index_path),
                                 cache=str(root/"features"),
                                 output=str(out/"feature_index.json"),
                                 smoke=a.smoke))
    print(f"P1b indices saved under {out}")


if __name__=="__main__":
    main()
